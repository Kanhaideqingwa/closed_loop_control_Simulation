"""
脑模型模块 —— SimulatedBrain 类

模拟人脑在游戏任务中的"难度-专注度"倒U型响应。
核心假设：存在一个最优难度 Optimal(t)，当实际难度 D 等于该值时专注度最高；
偏离越多，专注度越低（高斯型衰减曲线）。
"""

import numpy as np


class SimulatedBrain:
    """模拟大脑：根据任务难度 D 和时刻 t 输出专注度 Attention ∈ [0, 1]。

    核心公式:
        Attention(D, t) = exp( -(D - Optimal(t))^2 / C ) + Noise
        Optimal(t) = Optimal_0 + Delta_max * (1 - exp(-lambda * t))

    其中:
        - Optimal_0:  初始最优难度（t=0 时）
        - Delta_max:   最优难度最大增长幅度
        - lambda_:     最优难度向稳态收敛的速率
        - C:           倒U型曲线的宽度参数（C 越大曲线越平缓）
        - noise_std:   观测噪声标准差
    """

    def __init__(
        self,
        optimal_0: float = 40.0,
        delta_max: float = 40.0,
        lambda_: float = 0.05,
        C: float = 400.0,
        noise_std: float = 0.02,
        seed: int | None = None,
    ):
        """
        参数:
            optimal_0:  t=0 时的最优难度，默认 40
            delta_max:  最优难度的渐进增长上限，默认 40（最终最优难度 ≈ 80）
            lambda_:    指数衰减速率，越大则 Optimal(t) 越快逼近稳态
            C:          高斯宽度参数，越大则专注度对难度偏差越不敏感
            noise_std:  高斯观测噪声的标准差
            seed:       随机数种子，保证可复现性
        """
        self.optimal_0 = optimal_0
        self.delta_max = delta_max
        self.lambda_ = lambda_
        self.C = C
        self.noise_std = noise_std

        # 使用 numpy 新式随机数生成器，保证可复现
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def optimal_difficulty(self, t: int) -> float:
        """计算第 t 步的最优难度 Optimal(t)。

        最优难度从 Optimal_0 出发，以指数方式逐渐逼近 Optimal_0 + Delta_max。
        这模拟了玩家随着时间推移技能提升、能承受更高难度的过程。

        参数:
            t: 时间步（整数，t >= 0）

        返回:
            当前时刻的最优难度值
        """
        return self.optimal_0 + self.delta_max * (1.0 - np.exp(-self.lambda_ * t))

    def attention(self, D: float, t: int) -> float:
        """计算给定难度 D 和时间 t 下的专注度 Attention(D, t)。

        使用高斯型函数模拟倒U型曲线:
            - 当 D 接近 Optimal(t) 时，注意力接近 1.0
            - 当 D 偏离 Optimal(t) 时，注意力衰减
            - 叠加高斯噪声模拟实际测量中的随机波动

        参数:
            D: 当前任务难度
            t: 时间步

        返回:
            专注度值，严格裁剪至 [0, 1]
        """
        opt = self.optimal_difficulty(t)

        # 高斯型倒U曲线：偏离最优难度越多，专注度越低
        attention = np.exp(-((D - opt) ** 2) / self.C)

        # 叠加观测噪声
        noise = self._rng.normal(0.0, self.noise_std)
        attention += noise

        # 裁剪到合理范围 [0, 1]
        return float(np.clip(attention, 0.0, 1.0))

    def reset(self):
        """重置大脑状态。

        重新初始化随机数生成器，使仿真可从相同起点复现。
        若初始提供了 seed，则恢复该 seed；否则使用全新随机状态。
        """
        self._rng = np.random.default_rng(self._seed)

    def set_params(self, **kwargs):
        """动态修改模型参数。

        允许在仿真运行过程中调整大脑模型参数，用于参数敏感性分析。

        可修改参数:
            optimal_0, delta_max, lambda_, C, noise_std, seed

        示例:
            brain.set_params(C=600.0, noise_std=0.01)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"SimulatedBrain 没有参数 '{key}'。"
                    f"可用参数: optimal_0, delta_max, lambda_, C, noise_std, seed"
                )
        # 若修改了 seed，同步更新随机数生成器
        if "seed" in kwargs:
            self._seed = kwargs["seed"]
            self._rng = np.random.default_rng(self._seed)
