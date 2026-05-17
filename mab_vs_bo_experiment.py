"""
MAB vs BO 定量对比实验脚本

比较 5 种策略在相同大脑模型上的性能:
  1. UCB-BO (BayesianOptimizer, kappa=0.5, window=10)
  2. epsilon-Greedy (epsilon=0.1, K=10 arms)
  3. UCB1-Bandit (K=10 arms)
  4. Thompson Sampling (K=10 arms, Gaussian model)
  5. Linear UCB (Contextual Bandit, K=10 arms)

每个策略运行 300 步 (C=400, lambda=0.05, sigma=0.02, Optimal从40→80),
3 个随机种子平均。
"""

import sys
import os
import numpy as np
import json

# 确保导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain_model import SimulatedBrain


# ============================================================================
#  公共工具函数
# ============================================================================

def run_one_seed(controller, brain_factory, n_steps=300):
    """运行单个种子的一次仿真，返回历史记录。"""
    brain = brain_factory()
    history = []
    D_current = 50.0  # 起始难度

    for t in range(n_steps):
        attention = brain.attention(D_current, t)
        optimal = brain.optimal_difficulty(t)

        history.append({
            "t": t,
            "D": D_current,
            "attention": attention,
            "optimal": optimal,
        })

        # 新难度 = 控制器选择动作
        D_current = controller.select_action(attention, D_current, t)

    return history


def compute_metrics(history):
    """从历史记录中提取四项性能指标。"""
    n = len(history)
    # 稳态误差: 最后50步的平均 |target(1.0) - attention| / target
    tail = history[-50:]
    steady_error = np.mean([abs(1.0 - h["attention"]) for h in tail])

    # 累积遗憾: sum(1 - attention_i)
    cumulative_regret = np.sum([1.0 - h["attention"] for h in history])

    # 探索比例: |D_i - D_{i-1}| > 5 的步数占比
    D_vals = np.array([h["D"] for h in history])
    jumps = np.abs(np.diff(D_vals))
    exploration_ratio = np.mean(jumps > 5.0)

    # 收敛时间: 首次进入 target+-5% (即 attention >= 0.95) 且之后不再跌出
    target_band = 0.05
    in_band = np.array([abs(1.0 - h["attention"]) <= target_band for h in history])
    convergence_time = n  # 默认未收敛
    for i in range(len(in_band)):
        if in_band[i]:
            # 检查从 i 往后的所有步是否都在 band 内
            remaining = in_band[i:]
            if np.sum(remaining) >= 0.8 * len(remaining):  # 80% 以上在 band 内即视为稳定
                convergence_time = i
                break

    return {
        "steady_error": steady_error,
        "cumulative_regret": cumulative_regret,
        "exploration_ratio": exploration_ratio,
        "convergence_time": convergence_time,
    }


# ============================================================================
#  策略 1: UCB-BO (复用现有 BayesianOptimizer)
# ============================================================================

class BO_UCB_Controller:
    """包装 BayesianOptimizer 为统一接口。"""
    def __init__(self, kappa=0.5, window_size=10, n_candidates=21, rbf_lengthscale=15.0):
        from controllers import BayesianOptimizer
        self.bo = BayesianOptimizer(
            kappa=kappa,
            window_size=window_size,
            n_candidates=n_candidates,
            rbf_lengthscale=rbf_lengthscale,
        )

    def select_action(self, attention, current_D, t):
        return self.bo.compute(setpoint=1.0, feedback=attention, dt=1.0, current_D=current_D)

    def reset(self):
        self.bo.reset()


# ============================================================================
#  策略 2: epsilon-Greedy (K=10 臂)
# ============================================================================

class EpsilonGreedyController:
    def __init__(self, epsilon=0.1, n_arms=10):
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.arms = np.linspace(5.0, 95.0, n_arms)  # {5, 15, 25, ..., 95}
        self.counts = np.zeros(n_arms)
        self.means = np.zeros(n_arms)
        self._last_arm_idx = n_arms // 2  # 从中间臂开始

    def select_action(self, attention, current_D, t):
        # 更新上一个选择的臂的统计
        self.counts[self._last_arm_idx] += 1
        n = self.counts[self._last_arm_idx]
        self.means[self._last_arm_idx] += (attention - self.means[self._last_arm_idx]) / n

        # epsilon-Greedy 选择
        if np.random.random() < self.epsilon:
            # 随机探索
            arm_idx = np.random.randint(0, self.n_arms)
        else:
            # 利用：选择均值最高的臂
            # 对未探索的臂给予乐观初始值
            best = -np.inf
            arm_idx = self.n_arms // 2
            for i in range(self.n_arms):
                if self.counts[i] == 0:
                    val = 1.0  # 乐观初始值
                else:
                    val = self.means[i]
                if val > best:
                    best = val
                    arm_idx = i

        self._last_arm_idx = arm_idx
        return self.arms[arm_idx]

    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.means = np.zeros(self.n_arms)
        self._last_arm_idx = self.n_arms // 2


# ============================================================================
#  策略 3: UCB1-Bandit (K=10 臂)
# ============================================================================

class UCB1Controller:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.arms = np.linspace(5.0, 95.0, n_arms)
        self.counts = np.zeros(n_arms)
        self.means = np.zeros(n_arms)
        self.total_pulls = 0
        self._last_arm_idx = n_arms // 2

    def select_action(self, attention, current_D, t):
        # 更新上一个臂的统计
        self.counts[self._last_arm_idx] += 1
        self.total_pulls += 1
        n_a = self.counts[self._last_arm_idx]
        self.means[self._last_arm_idx] += (attention - self.means[self._last_arm_idx]) / n_a

        # UCB1: 先每臂至少拉一次
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            arm_idx = untried[0]
        else:
            # UCB1 score: mean + sqrt(2 * log(N) / n_a)
            scores = self.means + np.sqrt(2.0 * np.log(self.total_pulls) / self.counts)
            arm_idx = int(np.argmax(scores))

        self._last_arm_idx = arm_idx
        return self.arms[arm_idx]

    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.means = np.zeros(self.n_arms)
        self.total_pulls = 0
        self._last_arm_idx = self.n_arms // 2


# ============================================================================
#  策略 4: Thompson Sampling (Gaussian, K=10 臂)
# ============================================================================

class ThompsonSamplingController:
    """Gaussian Thompson Sampling.

    先验: N(0, 1) per arm
    似然: N(theta_a, sigma^2)
    后验: 每臂维护均值和精度 (precision = 1/variance)
    """
    def __init__(self, n_arms=10, prior_var=1.0, noise_var=0.01):
        self.n_arms = n_arms
        self.arms = np.linspace(5.0, 95.0, n_arms)
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.prior_precision = 1.0 / prior_var
        self.means = np.zeros(n_arms)
        self.precisions = np.full(n_arms, self.prior_precision)
        self._last_arm_idx = n_arms // 2

    def select_action(self, attention, current_D, t):
        idx = self._last_arm_idx
        # Bayesian update for the last arm
        old_prec = self.precisions[idx]
        new_prec = old_prec + 1.0 / self.noise_var
        self.means[idx] = (old_prec * self.means[idx] + attention / self.noise_var) / new_prec
        self.precisions[idx] = new_prec

        # Thompson Sampling: 从每个臂的后验采样
        samples = np.random.normal(
            self.means,
            np.sqrt(1.0 / np.maximum(self.precisions, 1e-8))
        )
        arm_idx = int(np.argmax(samples))
        self._last_arm_idx = arm_idx
        return self.arms[arm_idx]

    def reset(self):
        self.means = np.zeros(self.n_arms)
        self.precisions = np.full(self.n_arms, self.prior_precision)
        self._last_arm_idx = self.n_arms // 2


# ============================================================================
#  策略 5: Linear UCB (Contextual Bandit, K=10 臂)
# ============================================================================

class LinearUCBController:
    """LinUCB with disjoint linear models.

    上下文 x_t = [1, t/300, D_current/100]^T
    每个臂维护独立的线性模型参数。
    """
    def __init__(self, n_arms=10, alpha=0.5, context_dim=3):
        self.n_arms = n_arms
        self.alpha = alpha
        self.context_dim = context_dim
        self.arms = np.linspace(5.0, 95.0, n_arms)

        # 每个臂 A_a (d×d) 和 b_a (d×1)
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self._last_arm_idx = n_arms // 2
        self._last_context = None

    def select_action(self, attention, current_D, t):
        # 更新上一个臂的模型
        if self._last_context is not None:
            idx = self._last_arm_idx
            x = self._last_context
            self.A[idx] += np.outer(x, x)
            self.b[idx] += attention * x

        # 构建当前上下文
        x_t = np.array([1.0, t / 300.0, current_D / 100.0])
        self._last_context = x_t

        # LinUCB 选择: argmax_a ( x_t^T theta_a + alpha * sqrt(x_t^T A_a^{-1} x_t) )
        best_score = -np.inf
        best_arm = self.n_arms // 2
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            # sqrt term uses norm bound from LinUCB paper
            cb = self.alpha * np.sqrt(x_t @ A_inv @ x_t)
            score = float(x_t @ theta + cb)
            if score > best_score:
                best_score = score
                best_arm = i

        self._last_arm_idx = best_arm
        return self.arms[best_arm]

    def reset(self):
        self.A = [np.eye(self.context_dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
        self._last_arm_idx = self.n_arms // 2
        self._last_context = None


# ============================================================================
#  主实验
# ============================================================================

if __name__ == "__main__":
    # 固定参数
    N_STEPS = 300
    N_SEEDS = 3
    BRAIN_PARAMS = {
        "optimal_0": 40.0,
        "delta_max": 40.0,
        "lambda_": 0.05,
        "C": 400.0,
        "noise_std": 0.02,
    }

    # 策略定义
    strategies = {
        "BO-UCB": lambda: BO_UCB_Controller(kappa=0.5, window_size=10),
        "epsilon-Greedy": lambda: EpsilonGreedyController(epsilon=0.1, n_arms=10),
        "UCB1-Bandit": lambda: UCB1Controller(n_arms=10),
        "Thompson-Sampling": lambda: ThompsonSamplingController(n_arms=10),
        "Linear-UCB": lambda: LinearUCBController(n_arms=10, alpha=0.5),
    }

    results = {}

    for name, factory in strategies.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        all_metrics = []
        for seed in range(N_SEEDS):
            print(f"  Seed {seed}...", end=" ", flush=True)

            def brain_factory(s=seed):
                return SimulatedBrain(
                    optimal_0=BRAIN_PARAMS["optimal_0"],
                    delta_max=BRAIN_PARAMS["delta_max"],
                    lambda_=BRAIN_PARAMS["lambda_"],
                    C=BRAIN_PARAMS["C"],
                    noise_std=BRAIN_PARAMS["noise_std"],
                    seed=s,
                )

            controller = factory()
            history = run_one_seed(controller, brain_factory, N_STEPS)
            metrics = compute_metrics(history)
            all_metrics.append(metrics)

            print(f"steady_err={metrics['steady_error']:.4f}, "
                  f"regret={metrics['cumulative_regret']:.2f}, "
                  f"explore={metrics['exploration_ratio']:.3f}, "
                  f"conv={metrics['convergence_time']}")

        # 汇总三个种子
        avg = {}
        std = {}
        for key in all_metrics[0].keys():
            vals = [m[key] for m in all_metrics]
            avg[key] = np.mean(vals)
            std[key] = np.std(vals)

        results[name] = {"avg": avg, "std": std, "raw": all_metrics}

        print(f"  -> Avg: steady_err={avg['steady_error']:.4f}, "
              f"regret={avg['cumulative_regret']:.2f}, "
              f"explore={avg['exploration_ratio']:.3f}, "
              f"conv={avg['convergence_time']:.0f}")

    # 输出汇总表格
    print("\n\n" + "=" * 90)
    print("  综合性能对比")
    print("=" * 90)
    header = f"{'Strategy':<20s} {'SteadyErr':>10s} {'CumRegret':>10s} {'Explore%':>10s} {'ConvTime':>9s}"
    print(header)
    print("-" * 90)
    for name in strategies:
        r = results[name]["avg"]
        s = results[name]["std"]
        print(f"{name:<20s} {r['steady_error']:>8.4f}+-{s['steady_error']:.4f}"
              f" {r['cumulative_regret']:>8.2f}+-{s['cumulative_regret']:.2f}"
              f" {r['exploration_ratio']*100:>8.1f}+-{s['exploration_ratio']*100:.1f}%"
              f" {r['convergence_time']:>8.0f}+-{s['convergence_time']:.0f}")

    # 保存结果到 JSON 供文档引用
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.json")

    # 转换 numpy 类型为 Python 原生类型
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
