"""
BCI 闭环控制仿真平台 —— Streamlit 交互式控制面板
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import deque

# ── 页面配置 ─────────────────────────────────────────────
st.set_page_config(
    page_title="BCI 闭环控制仿真平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 18px 20px;
        color: #fff;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        margin-bottom: 8px;
    }
    .metric-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4); }
    .metric-card.orange { background: linear-gradient(135deg, #f12711 0%, #f5af19 100%); box-shadow: 0 4px 15px rgba(241, 39, 17, 0.4); }
    .metric-card.purple { background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%); box-shadow: 0 4px 15px rgba(142, 45, 226, 0.4); }
    .metric-label { font-size: 0.85rem; opacity: 0.9; margin-bottom: 4px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .control-section { padding: 8px 0; }
    hr { margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  MOCK 后端实现（当真实模块不可用时回退）
# ══════════════════════════════════════════════════════════

class MockBrain:
    def __init__(self, optimal_0=40, delta_max=40, lambda_=0.05, C=400, noise_std=0.02):
        self.optimal_0 = optimal_0
        self.delta_max = delta_max
        self.lambda_ = lambda_
        self.C = C
        self.noise_std = noise_std
        self.t = 0

    def optimal_difficulty(self, t):
        return self.optimal_0 + self.delta_max * (1 - np.exp(-self.lambda_ * t))

    def attention(self, D, t):
        opt = self.optimal_difficulty(t)
        att = np.exp(-((D - opt) ** 2) / self.C)
        noise = np.random.normal(0, self.noise_std)
        return np.clip(att + noise, 0.0, 1.0)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self):
        self.t = 0


class MockPIDController:
    def __init__(self):
        self.Kp, self.Ki, self.Kd = 2.0, 0.1, 0.5
        self.deadband = 0.01
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint, feedback, dt=1.0):
        error = setpoint - feedback
        if abs(error) < self.deadband:
            error = 0.0
        self._integral += error * dt
        self._integral = np.clip(self._integral, -50, 50)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        return np.clip(output, 0, 100)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class MockDiscretePIDController:
    def __init__(self):
        self.Kp, self.Ki, self.Kd = 2.0, 0.1, 0.5
        self.deadband = 0.01
        self.levels = 10
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint, feedback, dt=1.0):
        error = setpoint - feedback
        if abs(error) < self.deadband:
            error = 0.0
        self._integral += error * dt
        self._integral = np.clip(self._integral, -50, 50)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        raw = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        raw = np.clip(raw, 0, 100)
        step_size = 100.0 / (self.levels - 1) if self.levels > 1 else 1.0
        return np.round(raw / step_size) * step_size

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class MockFuzzyController:
    """双输入单输出模糊控制器 (Mock)。

    API 与真实 FuzzyController 一致: e_range, de_range, output_centers。
    内部使用简化的 Mamdani 推理 + 重心法去模糊化。
    """

    LABELS = ["NB", "NS", "ZO", "PS", "PB"]

    DEFAULT_RULES = [
        ["NB", "NB", "NS", "NS", "ZO"],
        ["NB", "NS", "NS", "ZO", "PS"],
        ["NS", "NS", "ZO", "PS", "PS"],
        ["NS", "ZO", "PS", "PS", "PB"],
        ["ZO", "PS", "PS", "PB", "PB"],
    ]

    def __init__(self):
        self.e_range = (-1.0, 1.0)
        self.de_range = (-0.5, 0.5)
        self.rules = self.DEFAULT_RULES
        self.output_centers = {"NB": 0.0, "NS": 25.0, "ZO": 50.0, "PS": 75.0, "PB": 100.0}
        self._prev_error = 0.0
        self._build_memberships()

    def _build_memberships(self):
        lo, hi = self.e_range
        self._e_centers = np.linspace(lo, hi, 5)
        self._e_hw = (hi - lo) / 4
        lo, hi = self.de_range
        self._de_centers = np.linspace(lo, hi, 5)
        self._de_hw = (hi - lo) / 4

    def _triangle_mu(self, x, c, hw):
        a, b, c_ = c - hw, c, c + hw
        if x <= a or x >= c_:
            return 0.0
        if x < b:
            return (x - a) / (b - a)
        return (c_ - x) / (c_ - b)

    def _fuzzify(self, x, centers, hw):
        return {label: self._triangle_mu(x, centers[i], hw) for i, label in enumerate(self.LABELS)}

    def compute(self, setpoint, feedback, dt=1.0):
        e = setpoint - feedback
        de = (e - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = e

        e_fuzzy = self._fuzzify(e, self._e_centers, self._e_hw)
        de_fuzzy = self._fuzzify(de, self._de_centers, self._de_hw)

        activations = {label: 0.0 for label in self.LABELS}
        for i, e_label in enumerate(self.LABELS):
            mu_e = e_fuzzy[e_label]
            if mu_e == 0.0:
                continue
            for j, de_label in enumerate(self.LABELS):
                mu_de = de_fuzzy[de_label]
                if mu_de == 0.0:
                    continue
                activation = min(mu_e, mu_de)
                out_label = self.rules[i][j]
                if activation > activations[out_label]:
                    activations[out_label] = activation

        num, den = 0.0, 0.0
        for label, activation in activations.items():
            num += activation * self.output_centers[label]
            den += activation
        u = num / den if den > 1e-9 else 50.0
        return float(np.clip(u, 0.0, 100.0))

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if "e_range" in kwargs or "de_range" in kwargs:
            self._build_memberships()

    def reset(self):
        self._prev_error = 0.0


class MockBayesianOptimizer:
    """基于 UCB 的贝叶斯优化器 (Mock)。

    使用 RBF 核近似 GP 后验的均值和方差，
    通过滑动窗口内的历史样本进行推断。
    API 与真实 BayesianOptimizer 一致。
    """

    def __init__(self):
        self.kappa = 2.0
        self.window_size = 20
        self.n_candidates = 21
        self.rbf_lengthscale = 15.0
        self._candidates = np.linspace(0.0, 100.0, self.n_candidates)
        self._history: list[tuple[float, float]] = []
        self._current_D = 50.0

    def _rbf_kernel(self, X1, X2):
        X1 = np.atleast_2d(X1).T if X1.ndim == 1 else X1
        X2 = np.atleast_2d(X2).T if X2.ndim == 1 else X2
        sq_dists = (X1[:, None] - X2[None, :]) ** 2
        return np.exp(-sq_dists / (2.0 * self.rbf_lengthscale ** 2))

    def compute(self, setpoint, feedback, dt=1.0):
        self._history.append((self._current_D, feedback))
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]

        if len(self._history) < 2:
            self._current_D = 50.0
            return self._current_D

        hist_D = np.array([h[0] for h in self._history])
        hist_y = np.array([h[1] for h in self._history])

        K = self._rbf_kernel(hist_D, hist_D)
        noise_var = 1e-3
        K_inv = np.linalg.inv(K + noise_var * np.eye(len(hist_D)))

        best_score = -np.inf
        best_D = self._current_D
        for D_cand in self._candidates:
            k_star = self._rbf_kernel(np.array([D_cand]), hist_D).ravel()
            k_starstar = self._rbf_kernel(np.array([D_cand]), np.array([D_cand])).ravel()[0]
            mu = k_star @ K_inv @ hist_y
            var = k_starstar - k_star @ K_inv @ k_star
            sigma = np.sqrt(max(var, 1e-8))
            score = mu + self.kappa * sigma
            if score > best_score:
                best_score = score
                best_D = D_cand

        self._current_D = float(best_D)
        return self._current_D

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if "n_candidates" in kwargs:
            self._candidates = np.linspace(0.0, 100.0, self.n_candidates)

    def reset(self):
        self._history.clear()
        self._current_D = 50.0


class MockSimulationEngine:
    def __init__(self, brain, controller, target_attention=0.95):
        self.brain = brain
        self.controller = controller
        self.target = target_attention
        self.history = []

    def step(self):
        t = len(self.history)
        if t == 0:
            D = 50.0
        else:
            prev_state = self.history[-1]
            D = self.controller.compute(self.target, prev_state['attention'], dt=1.0)

        attention = self.brain.attention(D, t)
        optimal = self.brain.optimal_difficulty(t)
        error = self.target - attention

        state = {
            't': t,
            'D': D,
            'attention': attention,
            'optimal': optimal,
            'error': error,
            'target': self.target,
        }
        self.history.append(state)
        return state

    def run(self, n):
        for _ in range(n):
            self.step()
        return self.history

    def reset(self):
        self.history.clear()
        self.brain.reset()
        self.controller.reset()

    def set_controller(self, ctrl):
        self.controller = ctrl

    def get_history_df(self):
        return pd.DataFrame(self.history)


# ══════════════════════════════════════════════════════════
#  尝试导入真实后端，失败则回退到 Mock
# ══════════════════════════════════════════════════════════

_use_mock = False
try:
    from brain_model import SimulatedBrain
    from controllers import PIDController, DiscretePIDController, FuzzyController, BayesianOptimizer
    from simulation_engine import SimulationEngine

    BrainCls = SimulatedBrain
    PIDCls = PIDController
    DPIDCls = DiscretePIDController
    FuzzyCls = FuzzyController
    BayesCls = BayesianOptimizer
    EngineCls = SimulationEngine
except ImportError:
    BrainCls = MockBrain
    PIDCls = MockPIDController
    DPIDCls = MockDiscretePIDController
    FuzzyCls = MockFuzzyController
    BayesCls = MockBayesianOptimizer
    EngineCls = MockSimulationEngine
    _use_mock = True


# ══════════════════════════════════════════════════════════
#  指标计算
# ══════════════════════════════════════════════════════════

def compute_metrics(history):
    if len(history) < 5:
        return {"steady_error": 0, "overshoot": 0, "chatter": 0, "settling_time": 0}

    df = pd.DataFrame(history)
    target = df['target'].iloc[0]
    attention = df['attention']

    # 稳态误差：最后 min(20, len) 步的平均绝对误差
    tail_n = min(20, len(df) // 4)
    steady_error = df['error'].tail(tail_n).abs().mean()

    # 超调量
    max_att = attention.max()
    overshoot = max(0, (max_att - target) / target * 100) if target > 0 else 0

    # 控制抖振：相邻步 D 差值的标准差
    chatter = df['D'].diff().dropna().std()

    # 响应时间：专注度首次进入 target±5% 且后续不离开
    band = target * 0.05
    in_band = (attention - target).abs() <= band
    settling_time = len(df)
    for i in range(len(in_band)):
        if in_band.iloc[i]:
            window = in_band.iloc[i:]
            if window.all():
                settling_time = i
                break

    return {
        "steady_error": round(steady_error, 4),
        "overshoot": round(overshoot, 2),
        "chatter": round(chatter, 3),
        "settling_time": settling_time,
    }


# ══════════════════════════════════════════════════════════
#  Session State 初始化
# ══════════════════════════════════════════════════════════

defaults = {
    "controller_type": "PID 控制器",
    "Kp": 2.0, "Ki": 0.1, "Kd": 0.5, "deadband": 0.01,
    "quant_levels": 10,
    "Ke": 0.1, "Kec": 0.1, "Ku": 10.0,
    "kappa": 2.0, "window_size": 20,
    "optimal_0": 40, "delta_max": 40, "lambda_": 0.05, "C": 400, "noise_std": 0.02,
    "target_attention": 0.95,
    "sim_steps": 200,
    "history": [],
    "ran": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════
#  标题
# ══════════════════════════════════════════════════════════

st.title("🧠 BCI 动态闭环控制系统仿真平台")
st.caption("基于 Yerkes-Dodson 定律的人脑注意力模型与四种闭环控制算法对比")

if _use_mock:
    st.info("后端模块未就绪，当前使用 Mock 引擎独立演示。Agent 1 模块就绪后将自动切换。", icon="ℹ️")


# ══════════════════════════════════════════════════════════
#  Sidebar - 左侧控制面板
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 控制面板")

    # ── 控制器选择 ──
    ctrl_type = st.selectbox(
        "控制器类型",
        ["PID 控制器", "离散 PID 控制器", "模糊控制器", "贝叶斯优化控制器"],
    )
    st.session_state["controller_type"] = ctrl_type

    st.markdown("---")

    # ── PID 参数 ──
    with st.expander("📐 PID 控制器参数", expanded=(ctrl_type.startswith("PID"))):
        st.session_state["Kp"] = st.slider("Kp (比例增益)", 0.0, 10.0, st.session_state["Kp"], 0.1)
        st.session_state["Ki"] = st.slider("Ki (积分增益)", 0.0, 5.0, st.session_state["Ki"], 0.01)
        st.session_state["Kd"] = st.slider("Kd (微分增益)", 0.0, 5.0, st.session_state["Kd"], 0.01)
        st.session_state["deadband"] = st.slider("Deadband (死区)", 0.0, 5.0, st.session_state["deadband"], 0.1)

    # ── 离散 PID 额外参数 ──
    with st.expander("📏 离散 PID 额外参数", expanded=(ctrl_type == "离散 PID 控制器")):
        st.session_state["quant_levels"] = st.slider("量化档位", 2, 20, st.session_state["quant_levels"], 1)

    # ── 模糊控制器参数 ──
    with st.expander("🌀 模糊控制器参数", expanded=(ctrl_type == "模糊控制器")):
        st.session_state["Ke"] = st.slider("误差增益 Ke", 0.01, 1.0, st.session_state["Ke"], 0.01)
        st.session_state["Kec"] = st.slider("误差变化率增益 Kec", 0.01, 1.0, st.session_state["Kec"], 0.01)
        st.session_state["Ku"] = st.slider("输出增益 Ku", 1.0, 50.0, st.session_state["Ku"], 1.0)

    # ── 贝叶斯优化参数 ──
    with st.expander("🎲 贝叶斯优化参数", expanded=(ctrl_type == "贝叶斯优化控制器")):
        st.session_state["kappa"] = st.slider("Kappa (探索系数)", 0.1, 10.0, st.session_state["kappa"], 0.1)
        st.session_state["window_size"] = st.slider("Window Size", 5, 100, st.session_state["window_size"], 1)

    st.markdown("---")

    # ── 大脑模型参数 ──
    with st.expander("🧠 大脑模型参数", expanded=True):
        st.session_state["optimal_0"] = st.slider("Optimal_0 (初始最佳难度)", 10, 80, st.session_state["optimal_0"], 1)
        st.session_state["delta_max"] = st.slider("Δmax (最大提升幅度)", 0, 60, st.session_state["delta_max"], 1)
        st.session_state["lambda_"] = st.slider("λ (学习率)", 0.0, 0.2, st.session_state["lambda_"], 0.001)
        st.session_state["C"] = st.slider("C (曲线宽度)", 100, 1000, st.session_state["C"], 10)
        st.session_state["noise_std"] = st.slider("Noise σ (噪声标准差)", 0.0, 0.2, st.session_state["noise_std"], 0.001)

    st.markdown("---")

    # ── 目标与仿真控制 ──
    with st.expander("🎯 目标与仿真", expanded=True):
        st.session_state["target_attention"] = st.slider("目标专注度", 0.5, 1.0, st.session_state["target_attention"], 0.01)
        st.session_state["sim_steps"] = st.slider("仿真步数", 50, 500, st.session_state["sim_steps"], 10)

    st.markdown("---")

    col_run, col_reset = st.columns(2)
    with col_run:
        run_clicked = st.button("▶ 运行仿真", type="primary", use_container_width=True)
    with col_reset:
        reset_clicked = st.button("🔄 重置", use_container_width=True)


# ══════════════════════════════════════════════════════════
#  构建控制器 & 执行仿真
# ══════════════════════════════════════════════════════════

ctrl_map = {
    "PID 控制器": PIDCls,
    "离散 PID 控制器": DPIDCls,
    "模糊控制器": FuzzyCls,
    "贝叶斯优化控制器": BayesCls,
}

if reset_clicked:
    st.session_state["history"] = []
    st.session_state["ran"] = False
    st.rerun()

if run_clicked:
    with st.spinner("仿真运行中..."):
        brain = BrainCls(
            optimal_0=st.session_state["optimal_0"],
            delta_max=st.session_state["delta_max"],
            lambda_=st.session_state["lambda_"],
            C=st.session_state["C"],
            noise_std=st.session_state["noise_std"],
        )

        ctrl_cls = ctrl_map[ctrl_type]
        ctrl = ctrl_cls()

        # 设置控制器参数
        if ctrl_type in ("PID 控制器", "离散 PID 控制器"):
            ctrl.set_params(
                Kp=st.session_state["Kp"],
                Ki=st.session_state["Ki"],
                Kd=st.session_state["Kd"],
                deadband=st.session_state["deadband"],
            )
        if ctrl_type == "离散 PID 控制器":
            ctrl.levels = st.session_state["quant_levels"]
        if ctrl_type == "模糊控制器":
            Ke = st.session_state["Ke"]
            Kec = st.session_state["Kec"]
            Ku = st.session_state["Ku"]
            e_range = (-1.0 / Ke, 1.0 / Ke)
            de_range = (-0.5 / Kec, 0.5 / Kec)
            output_centers = {
                "NB": max(0.0, 50.0 - 5.0 * Ku),
                "NS": max(0.0, 50.0 - 2.5 * Ku),
                "ZO": 50.0,
                "PS": min(100.0, 50.0 + 2.5 * Ku),
                "PB": min(100.0, 50.0 + 5.0 * Ku),
            }
            ctrl.set_params(e_range=e_range, de_range=de_range, output_centers=output_centers)
        if ctrl_type == "贝叶斯优化控制器":
            ctrl.set_params(kappa=st.session_state["kappa"], window_size=st.session_state["window_size"])

        engine = EngineCls(brain, ctrl, target_attention=st.session_state["target_attention"])
        engine.run(st.session_state["sim_steps"])
        st.session_state["history"] = engine.history
        st.session_state["ran"] = True


# ══════════════════════════════════════════════════════════
#  主区域渲染
# ══════════════════════════════════════════════════════════

if not st.session_state["ran"]:
    st.info("请在左侧设置参数后点击 **「▶ 运行仿真」** 开始。", icon="👈")
    st.stop()

history = st.session_state["history"]
df = pd.DataFrame(history)
metrics = compute_metrics(history)

# ── 评价指标卡片 ──
st.subheader("📊 评价指标")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""<div class="metric-card green">
        <div class="metric-label">稳态误差</div>
        <div class="metric-value">{metrics['steady_error']:.4f}</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""<div class="metric-card orange">
        <div class="metric-label">超调量 (%)</div>
        <div class="metric-value">{metrics['overshoot']:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""<div class="metric-card purple">
        <div class="metric-label">控制抖振</div>
        <div class="metric-value">{metrics['chatter']:.3f}</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">响应时间 (步)</div>
        <div class="metric-value">{metrics['settling_time']}</div>
    </div>""", unsafe_allow_html=True)

# ── 图1: 时域响应图 (双Y轴) ──
st.markdown("---")
st.subheader(f"📈 时域响应曲线 —— {ctrl_type}")

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

fig1.add_trace(
    go.Scatter(
        x=df['t'], y=df['target'],
        name="目标专注度",
        mode="lines",
        line=dict(dash="dash", color="#1f77b4", width=2),
    ),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(
        x=df['t'], y=df['attention'],
        name="实际专注度",
        mode="lines",
        line=dict(color="#17becf", width=2),
        fill='tozeroy',
        fillcolor='rgba(23, 190, 207, 0.1)',
    ),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(
        x=df['t'], y=df['D'],
        name="控制量 D",
        mode="lines",
        line=dict(color="#d62728", width=1.8),
    ),
    secondary_y=True,
)
fig1.add_trace(
    go.Scatter(
        x=df['t'], y=df['optimal'],
        name="最佳难度 Optimal(t)",
        mode="lines",
        line=dict(color="#9467bd", width=1.5, dash="dot"),
    ),
    secondary_y=True,
)

fig1.update_xaxes(title_text="时间步 t", showgrid=True, gridcolor='rgba(128,128,128,0.15)')
fig1.update_yaxes(title_text="专注度", range=[0, 1.05], secondary_y=False,
                   tickformat=".2f", gridcolor='rgba(128,128,128,0.15)')
fig1.update_yaxes(title_text="难度 / 控制量 D", range=[-2, 105], secondary_y=True,
                   gridcolor='rgba(128,128,128,0.08)')

fig1.update_layout(
    template="plotly_white",
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)
st.plotly_chart(fig1, use_container_width=True)

# ── 图2: 倒U曲线动态演化 ──
st.subheader("🔬 倒U型曲线动态演化 —— Yerkes-Dodson Law")

fig2 = go.Figure()

D_grid = np.linspace(0, 100, 200)
total_steps = len(df)
sample_count = min(12, total_steps)
sampled_indices = np.linspace(0, total_steps - 1, sample_count, dtype=int)

# 颜色渐变：从浅蓝 → 深红，表示时间推移
for idx_pos, idx in enumerate(sampled_indices):
    t_val = int(df['t'].iloc[idx])
    opt = df['optimal'].iloc[idx]
    att_curve = np.exp(-((D_grid - opt) ** 2) / st.session_state["C"])

    # 颜色插值
    frac = idx_pos / max(sample_count - 1, 1)
    r = int(255 * frac)
    g = int(100 + 100 * (1 - frac))
    b = int(255 * (1 - frac))
    color = f"rgb({r},{g},{b})"

    fig2.add_trace(go.Scatter(
        x=D_grid, y=att_curve,
        mode="lines",
        name=f"t={t_val}",
        line=dict(color=color, width=1.5),
        opacity=0.7,
        showlegend=False,
    ))
    # 峰值标注
    fig2.add_trace(go.Scatter(
        x=[opt], y=[1.0],
        mode="markers",
        marker=dict(color=color, size=6, symbol="x"),
        name=f"峰值 t={t_val}",
        showlegend=False,
    ))

# 实际工作点轨迹
fig2.add_trace(go.Scatter(
    x=df['D'], y=df['attention'],
    mode="markers+lines",
    name="实际轨迹",
    line=dict(color="#00ff88", width=2),
    marker=dict(
        size=4,
        color=df['t'],
        colorscale="Viridis",
        colorbar=dict(title="时间步", thickness=12, len=0.5),
        showscale=True,
    ),
))

fig2.update_xaxes(title_text="难度 D", range=[0, 100], gridcolor='rgba(128,128,128,0.15)')
fig2.update_yaxes(title_text="专注度", range=[0, 1.05], tickformat=".2f", gridcolor='rgba(128,128,128,0.15)')
fig2.update_layout(
    template="plotly_white",
    height=450,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="closest",
)
st.plotly_chart(fig2, use_container_width=True)

# ── 实时数据表格 ──
st.markdown("---")
st.subheader("📋 仿真数据表 (最近 20 步)")

display_df = df.tail(20).copy()
display_df = display_df.rename(columns={
    't': '时间步',
    'D': '控制量 D',
    'attention': '专注度',
    'optimal': '最佳难度',
    'error': '误差',
    'target': '目标',
})
display_df = display_df[['时间步', '控制量 D', '专注度', '最佳难度', '误差']]

st.dataframe(
    display_df.style.format({
        '控制量 D': '{:.2f}',
        '专注度': '{:.4f}',
        '最佳难度': '{:.2f}',
        '误差': '{:.4f}',
    }),
    use_container_width=True,
    height=340,
)
