"""
控制器模块 —— 四种闭环控制算法

包含:
    1. PIDController        —— 连续型 PID，带抗积分饱和与死区
    2. DiscretePIDController —— 离散阶梯 PID，输出量化为 N 个档位
    3. FuzzyController      —— 双输入模糊控制器（误差 + 误差变化率）
    4. BayesianOptimizer    —— 基于 UCB 的贝叶斯优化器

所有控制器继承或遵循统一接口:
    - compute(setpoint, feedback, dt) -> float  # 返回控制量 D ∈ [0, 100]
    - reset()                                   # 重置内部状态
    - set_params(**kwargs)                      # 动态修改参数
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist


# ============================================================================
#                              基础控制器接口
# ============================================================================

class BaseController(ABC):
    """所有控制器的抽象基类。

    定义统一接口，所有具体控制器必须实现 compute、reset、set_params。
    """

    @abstractmethod
    def compute(
        self,
        setpoint: float,
        feedback: float,
        dt: float = 1.0,
        current_D: float | None = None,
    ) -> float:
        """根据设定值和反馈值计算控制量。

        参数:
            setpoint:  目标值（期望专注度）
            feedback:  当前反馈值（实际专注度）
            dt:        采样时间间隔
            current_D: 当前正在施加的难度值（产生此反馈的 D），
                       用于自适应控制器估计 dA/dD 方向

        返回:
            控制量 D ∈ [0, 100]
        """
        ...

    @abstractmethod
    def reset(self):
        """重置控制器内部状态（积分、历史等）。"""
        ...

    @abstractmethod
    def set_params(self, **kwargs):
        """动态修改控制器参数。"""
        ...


# ============================================================================
#                         1. PIDController —— 连续型 PID
# ============================================================================

class PIDController(BaseController):
    """增量式（速度型）PID 控制器，适用于输出范围远大于误差范围的场景。

    控制律（增量形式）:
        delta_u = Kp*(e - e_prev) + Ki*e*dt + Kd*(e - 2*e_prev + e_prev2)/dt
        u = clip(u_prev + delta_u, output_limit)

    增强特性:
        - 增量式结构: 输出通过累加增量自然覆盖整个 [0, 100] 范围，
          避免位置式 PID 因误差小(≈0.1)而输出微小值(≈0.2)的问题
        - 自适应方向估计 (Adaptive): 针对注意力-难度倒U型曲线的非单调特性，
          通过实时估计 d(attention)/dD 的符号判断当前处于曲线左侧(D<optimal)
          还是右侧(D>optimal)，自动反转控制方向，并在方向切换时重置内态
        - 抗积分饱和 (Clamping): 当输出达到饱和边界时，冻结累加器，
          防止增量持续累加导致的"windup"现象
        - 死区 (Deadband): 当 |e| < deadband 时视为 0，避免微小误差
          引起不必要的控制动作
        - 输出限幅: 严格限制 u ∈ output_limit
    """

    def __init__(
        self,
        Kp: float = 3.0,
        Ki: float = 3.0,
        Kd: float = 3.0,
        deadband: float = 0.01,
        output_limit: tuple[float, float] = (0.0, 100.0),
        adaptive: bool = False,
        init_u: float = 50.0,
    ):
        """
        参数:
            Kp:           比例增益
            Ki:           积分增益
            Kd:           微分增益
            deadband:     死区阈值，|误差| < 此值时视为 0
            output_limit: 输出限幅 (min, max)
            adaptive:     是否启用自适应方向估计（针对倒U型曲线的非单调响应）
            init_u:       初始输出值（默认 50，范围中点）
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.deadband = deadband
        self.output_limit = output_limit
        self.adaptive = adaptive
        self.init_u = init_u

        self.reset()

    def reset(self):
        """重置 PID 内部状态。"""
        self._u = self.init_u                  # 当前累加输出
        self._prev_error = 0.0                 # e(t-1)
        self._prev_prev_error = 0.0            # e(t-2)，用于增量式微分项
        # 自适应方向估计状态
        self._direction = 1.0                  # 1.0=正常方向, -1.0=反转方向
        self._last_D: float | None = None      # 上一步产生 feedback 的 D（来自 current_D 参数）
        self._last_feedback: float | None = None  # 上一步的 feedback
        self._dA_dD_smooth: float = 0.0        # 平滑后的 dA/dD 估计
        self._stuck_counter: int = 0           # 方向卡死检测计数器

    def set_params(self, **kwargs):
        """动态修改 PID 参数。

        示例:
            pid.set_params(Kp=3.0, deadband=1.0)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"PIDController 没有参数 '{key}'。"
                    f"可用参数: Kp, Ki, Kd, deadband, output_limit, adaptive, init_u"
                )

    def compute(
        self,
        setpoint: float,
        feedback: float,
        dt: float = 1.0,
        current_D: float | None = None,
    ) -> float:
        """增量式 PID 控制量计算。

        步骤:
            1. 计算误差 e = setpoint - feedback
            2. 自适应方向估计: 若启用 adaptive，用 current_D 计算 dA/dD
               （dD = current_D - last_D, dA = feedback - last_feedback），
               判断曲线侧并在方向切换时重置内态
            3. 死区判断: |e| < deadband → e = 0
            4. 增量计算: delta = Kp*(e-e_prev) + Ki*e*dt + Kd*(e-2*e_prev+e_prev2)/dt
            5. 累加: u = clip(u_prev + delta, limit)，带抗饱和

        参数:
            setpoint:  目标专注度
            feedback:  实际专注度
            dt:        采样间隔
            current_D: 产生当前 feedback 的难度值（由仿真引擎传入）

        返回:
            难度控制量 D ∈ [0, 100]
        """
        lo, hi = self.output_limit

        # 1. 计算原始误差
        error = setpoint - feedback

        # 2. 自适应方向估计: 针对倒U型非单调曲线
        #    使用 current_D（引擎传入的、产生本次 feedback 的 D 值）
        #    与 _last_D（上一步的 D）之间的变化来估计 dA/dD 符号
        if self.adaptive:
            raw_error = error  # 保存原始误差用于卡死检测

            if (
                current_D is not None
                and self._last_D is not None
                and self._last_feedback is not None
            ):
                dD = current_D - self._last_D
                dA = feedback - self._last_feedback

                # 仅当 |dD| >= 0.5 时更新方向估计，避免小步进噪声
                if abs(dD) >= 0.5:
                    sensitivity = dA / dD
                    # 指数平滑 τ=0.55 适度响应
                    self._dA_dD_smooth = (
                        0.55 * self._dA_dD_smooth + 0.45 * sensitivity
                    )
                    prev_direction = self._direction
                    if self._dA_dD_smooth < -0.05:
                        self._direction = -1.0
                    elif self._dA_dD_smooth > 0.05:
                        self._direction = 1.0

                    if self._direction != prev_direction:
                        self._prev_error = 0.0
                        self._prev_prev_error = 0.0
                        self._stuck_counter = 0

            # 卡死检测: 若误差持续大且方向未变化，可能卡在错误方向
            if abs(raw_error) > 0.15:
                self._stuck_counter += 1
            else:
                self._stuck_counter = max(0, self._stuck_counter - 1)

            # 连续 40 步大误差 → 强制重置方向到正常模式
            if self._stuck_counter >= 40:
                self._direction = 1.0
                self._dA_dD_smooth = 0.0
                self._prev_error = 0.0
                self._prev_prev_error = 0.0
                self._stuck_counter = 0

            # 应用方向到误差
            error = self._direction * error

        # 3. 死区处理：微小误差直接置零
        if abs(error) < self.deadband:
            error = 0.0

        # 4. 增量式 PID 计算
        #    delta_P = Kp * (e - e_prev)
        #    delta_I = Ki * e * dt
        #    delta_D = Kd * (e - 2*e_prev + e_prev2) / dt  (二阶差分)
        delta_P = self.Kp * (error - self._prev_error)
        delta_I = self.Ki * error * dt
        delta_D = (
            self.Kd * (error - 2 * self._prev_error + self._prev_prev_error) / dt
            if dt > 0
            else 0.0
        )

        delta_u = delta_P + delta_I + delta_D

        # 5. 累加输出，带抗积分饱和
        candidate_u = self._u + delta_u

        if lo <= candidate_u <= hi:
            self._u = candidate_u
        # 否则冻结累加器，防止 windup

        # 更新误差历史（供下一轮增量计算）
        self._prev_prev_error = self._prev_error
        self._prev_error = error

        # 6. 记录用于下一轮自适应估计的 D 和 feedback
        #    _last_D = 产生本次 feedback 的 D → current_D（由引擎传入）
        self._last_feedback = feedback
        self._last_D = current_D if current_D is not None else self._u

        return float(np.clip(self._u, lo, hi))


# ============================================================================
#                    2. DiscretePIDController —— 离散阶梯 PID
# ============================================================================

class DiscretePIDController(PIDController):
    """离散阶梯 PID 控制器。

    继承 PIDController 的连续控制逻辑，但将输出量化为 N 个离散档位。
    适用于难度档位固定、不允许连续调节的场景（如游戏难度只有几档可选）。

    量化方式:
        step_size = 100 / levels
        quantized_output = round(u / step_size) * step_size
    """

    def __init__(
        self,
        levels: int = 8,
        Kp: float = 3.0,
        Ki: float = 1.0,
        Kd: float = 2.0,
        deadband: float = 0.01,
        output_limit: tuple[float, float] = (0.0, 100.0),
        adaptive: bool = False,
    ):
        """
        参数:
            levels:       离散档位数（如 10 表示 0, 10, 20, ..., 100）
            Kp, Ki, Kd:   同 PIDController
            deadband:     死区阈值
            output_limit: 输出限幅
            adaptive:     是否启用自适应方向估计（离散PID默认关闭，
                          量化会干扰 dA/dD 敏感度估计）
        """
        super().__init__(
            Kp=Kp, Ki=Ki, Kd=Kd,
            deadband=deadband, output_limit=output_limit,
            adaptive=adaptive,
        )
        self.levels = levels
        self._step_size = 100.0 / (levels - 1) if levels > 1 else 1.0

    def set_params(self, **kwargs):
        """动态修改参数，同步更新 step_size。"""
        super().set_params(**kwargs)
        if "levels" in kwargs:
            self._step_size = 100.0 / (self.levels - 1) if self.levels > 1 else 1.0

    def compute(
        self,
        setpoint: float,
        feedback: float,
        dt: float = 1.0,
        current_D: float | None = None,
    ) -> float:
        """离散 PID 控制量计算。

        先调用父类的连续 PID 计算，再将结果量化为最近的离散档位。

        参数:
            setpoint:  目标专注度
            feedback:  实际专注度
            dt:        采样间隔
            current_D: 产生当前 feedback 的难度值

        返回:
            量化后的难度控制量 D ∈ [0, 100]
        """
        # 先计算连续 PID 输出
        u_continuous = super().compute(setpoint, feedback, dt, current_D)

        # 量化为最近的离散档位
        u_quantized = round(u_continuous / self._step_size) * self._step_size

        # 再次确保在限幅范围内
        lo, hi = self.output_limit
        u_quantized = np.clip(u_quantized, lo, hi)
        return float(u_quantized)


# ============================================================================
#                    3. FuzzyController —— 模糊控制器
# ============================================================================

class FuzzyController(BaseController):
    """双输入单输出模糊控制器。

    输入变量:
        - e:   误差 = 设定值 - 反馈值
        - de:  误差变化率 = e(t) - e(t-1)

    模糊化:
        5 个模糊集 {NB(负大), NS(负小), ZO(零), PS(正小), PB(正大)}
        采用三角形隶属度函数 (Triangular Membership Function)

    模糊规则:
        5x5 规则矩阵，体现"误差大则大幅调整，误差小则微调"的直觉

    去模糊化:
        重心法 (Centroid)，将模糊输出映射到连续控制量 [0, 100]
    """

    # 模糊语言变量标签
    LABELS = ["NB", "NS", "ZO", "PS", "PB"]

    # 默认模糊规则表 (error × delta_error)
    # 输出含义: NB=大幅降低难度, NS=小幅降低, ZO=维持, PS=小幅提升, PB=大幅提升
    # 规则直觉: 误差越正(专注度不够)越要提高难度, 变化率越大越要预防性调整
    DEFAULT_RULES = [
        # de: NB      NS      ZO      PS      PB       <- delta_error
        ["NB", "NB", "NS", "NS", "ZO"],  # e = NB (专注度过高 → 降低难度)
        ["NB", "NS", "NS", "ZO", "PS"],  # e = NS
        ["NS", "NS", "ZO", "PS", "PS"],  # e = ZO
        ["NS", "ZO", "PS", "PS", "PB"],  # e = PS
        ["ZO", "PS", "PS", "PB", "PB"],  # e = PB (专注度不够 → 提高难度)
    ]

    # 输出隶属度函数中心点 (对应 NB, NS, ZO, PS, PB)
    DEFAULT_OUTPUT_CENTERS = {
        "NB": 0.0,
        "NS": 25.0,
        "ZO": 50.0,
        "PS": 75.0,
        "PB": 100.0,
    }

    def __init__(
        self,
        e_range: tuple[float, float] = (-10.0, 10.0),
        de_range: tuple[float, float] = (-2.5, 2.5),
        rules: list[list[str]] | None = None,
        output_centers: dict[str, float] | None = None,
        dither_amplitude: float = 10.0,
        patience: int = 20,
        incremental: bool = True,
    ):
        """
        参数:
            e_range:          误差的论域范围 (min, max)
            de_range:         误差变化率的论域范围 (min, max)
            rules:            5x5 模糊规则矩阵
            output_centers:   输出模糊集中心值字典
            dither_amplitude: 探索噪声幅度 (0=关闭)，用于跳出局部最优
            patience:         容忍步数，连续大误差超过此步数后触发探索 (0=关闭)
            incremental:      增量模式，True=输出ΔD并累加，False=直接输出绝对D
        """
        self.e_range = e_range
        self.de_range = de_range
        self.rules = rules if rules is not None else self.DEFAULT_RULES
        self.output_centers = (
            output_centers if output_centers is not None
            else self.DEFAULT_OUTPUT_CENTERS
        )
        self.dither_amplitude = dither_amplitude
        self.patience = patience
        self.incremental = incremental

        # 预计算输入隶属度函数参数（三角形的三个顶点 a, b, c）
        self._e_mfs = self._build_memberships(e_range)
        self._de_mfs = self._build_memberships(de_range)

        self._prev_error = 0.0
        self._stuck_counter = 0
        self._rng = np.random.default_rng()
        # 自适应方向估计状态
        self._direction = 1.0
        self._last_D: float | None = None
        self._last_feedback: float | None = None
        self._dA_dD_smooth = 0.0
        # 增量模式累加器
        self._u = 50.0

    def reset(self):
        """重置前一时刻误差记录。"""
        self._prev_error = 0.0
        self._stuck_counter = 0
        self._direction = 1.0
        self._last_D = None
        self._last_feedback = None
        self._dA_dD_smooth = 0.0
        self._u = 50.0

    def set_params(self, **kwargs):
        """动态修改模糊控制器参数。

        示例:
            fuzzy.set_params(e_range=(-0.8, 0.8))

        注意: 修改 e_range / de_range 后会自动重建隶属度函数。
        """
        for key, value in kwargs.items():
            if key in ("e_range", "de_range", "rules", "output_centers",
                        "dither_amplitude", "patience", "incremental"):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"FuzzyController 没有参数 '{key}'。"
                    f"可用参数: e_range, de_range, rules, output_centers, dither_amplitude, patience, incremental"
                )
        # 若论域变化则重建隶属度函数
        if "e_range" in kwargs:
            self._e_mfs = self._build_memberships(self.e_range)
        if "de_range" in kwargs:
            self._de_mfs = self._build_memberships(self.de_range)

    def _build_memberships(self, rng: tuple[float, float]) -> dict[str, np.ndarray]:
        """构建 5 个三角形隶属度函数的参数矩阵。

        在论域 [min, max] 上均匀分布 5 个三角形:
            NB  → 偏向最小值
            NS  → 偏向中下
            ZO  → 中心
            PS  → 偏向中上
            PB  → 偏向最大值

        每个三角形由三个顶点 (a, b, c) 定义，其中 b 是中心，a,c 是支撑边界。

        返回:
            {"NB": [a,b,c], "NS": [a,b,c], ...}
        """
        lo, hi = rng
        # 5 个三角形的中心点均匀分布
        centers = np.linspace(lo, hi, 5)
        half_width = (hi - lo) / 4  # 半宽

        mfs = {}
        for i, label in enumerate(self.LABELS):
            a = centers[i] - half_width
            b = centers[i]
            c = centers[i] + half_width
            mfs[label] = np.array([a, b, c])
        return mfs

    @staticmethod
    def _triangle_membership(x: float, params: np.ndarray) -> float:
        """三角形隶属度函数。

        μ(x) = max( min((x-a)/(b-a), (c-x)/(c-b)), 0 )

        参数:
            x:      输入值
            params: 三角形参数 [a, b, c]

        返回:
            隶属度 ∈ [0, 1]
        """
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)

    def _fuzzify(self, x: float, mfs: dict[str, np.ndarray]) -> dict[str, float]:
        """将精确输入值模糊化为各模糊集的隶属度。

        参数:
            x:   精确输入值
            mfs: 隶属度函数参数字典

        返回:
            {"NB": 0.0, "NS": 0.2, "ZO": 0.8, "PS": 0.0, "PB": 0.0}
        """
        return {
            label: self._triangle_membership(x, params)
            for label, params in mfs.items()
        }

    def compute(
        self,
        setpoint: float,
        feedback: float,
        dt: float = 1.0,
        current_D: float | None = None,
    ) -> float:
        """模糊控制器计算。

        流程:
            1. 计算误差 e 和误差变化率 de
            2. 模糊化: 将 e 和 de 映射为各模糊集的隶属度
            3. 规则推理: 对每条规则取 min(μ_e, μ_de) 作为激活度
            4. 规则聚合: 对同一输出模糊集取 max（Mamdani 推理）
            5. 去模糊化: 重心法将聚合结果转换为精确输出
            6. 输出限幅至 [0, 100]

        参数:
            setpoint: 目标专注度
            feedback: 实际专注度
            dt:       采样间隔

        返回:
            难度控制量 D ∈ [0, 100]
        """
        # 1. 计算误差及变化率
        e_raw = setpoint - feedback
        de_raw = (e_raw - self._prev_error) / dt if dt > 0 else 0.0

        # 自适应方向估计：感知当前处于曲线左侧还是右侧
        if hasattr(self, '_last_D') and self._last_D is not None and current_D is not None:
            dD = current_D - self._last_D
            dA = feedback - (self._last_feedback if hasattr(self, '_last_feedback') and self._last_feedback is not None else feedback)
            if abs(dD) > 0.5:
                sensitivity = dA / dD
                self._dA_dD_smooth = 0.6 * getattr(self, '_dA_dD_smooth', 0.0) + 0.4 * sensitivity
                if self._dA_dD_smooth < -0.05:
                    self._direction = -1.0
                elif self._dA_dD_smooth > 0.05:
                    self._direction = 1.0

        self._last_D = current_D if current_D is not None else self._last_D
        self._last_feedback = feedback
        self._prev_error = e_raw

        # 应用方向到误差
        direction = getattr(self, '_direction', 1.0)
        e = direction * e_raw
        de = direction * de_raw

        # 2. 模糊化
        e_fuzzy = self._fuzzify(e, self._e_mfs)
        de_fuzzy = self._fuzzify(de, self._de_mfs)

        # 3. 规则推理 + 聚合 (Mamdani max-min 组合)
        # rule_activations[output_label] = 该输出模糊集的最高激活度
        rule_activations: dict[str, float] = {label: 0.0 for label in self.LABELS}

        for i, e_label in enumerate(self.LABELS):
            mu_e = e_fuzzy[e_label]
            if mu_e == 0.0:
                continue  # 跳过零激活度以加速
            for j, de_label in enumerate(self.LABELS):
                mu_de = de_fuzzy[de_label]
                if mu_de == 0.0:
                    continue
                # 规则前提激活度 = min(μ_e, μ_de)
                activation = min(mu_e, mu_de)
                output_label = self.rules[i][j]
                # 同一输出模糊集取 max 聚合
                if activation > rule_activations[output_label]:
                    rule_activations[output_label] = activation

        # 4. 去模糊化 —— 重心法 (Centroid)
        numerator = 0.0
        denominator = 0.0
        for label, activation in rule_activations.items():
            center = self.output_centers[label]
            numerator += activation * center
            denominator += activation

        if denominator > 0:
            fuzzy_out = numerator / denominator
        else:
            # 无规则激活时保持中间值
            fuzzy_out = 50.0

        # 5. 探索抖动: 连续大误差时注入随机噪声跳出局部最优
        if self.dither_amplitude > 0 and self.patience > 0:
            if abs(e_raw) > 0.03:
                self._stuck_counter += 1
            else:
                self._stuck_counter = max(0, self._stuck_counter - 1)

            if self._stuck_counter >= self.patience:
                fuzzy_out += self._rng.uniform(-self.dither_amplitude, self.dither_amplitude)
                self._stuck_counter = 0  # 重置防止持续抖动

        # 6. 增量模式: 模糊输出为ΔD，累加到当前D
        if self.incremental:
            # 将 [0,100] 映射为 [-50, 50] 的增量
            delta = (fuzzy_out - 50.0) * 2.0  # 映射到 [-100, 100]
            # 缩放增量（误差小时增量也小）
            if abs(e_raw) < 0.05:
                delta *= 0.3
            candidate = self._u + delta
            self._u = np.clip(candidate, 0.0, 100.0)
            return float(self._u)

        # 7. 限幅（绝对模式）
        return float(np.clip(fuzzy_out, 0.0, 100.0))


# ============================================================================
#                 4. BayesianOptimizer —— 贝叶斯优化 (UCB)
# ============================================================================

class BayesianOptimizer(BaseController):
    """基于 Upper Confidence Bound (UCB) 的贝叶斯优化器。

    核心思想:
        将难度选择建模为黑箱优化问题 —— 寻找使专注度最大化的难度 D。
        使用 UCB 采集函数平衡"利用"(选择当前已知最佳难度)与"探索"(尝试不确定性高的难度)。

    采集函数:
        score(D) = μ(D) + κ * σ(D)

    其中:
        - μ(D): 使用 RBF 核的加权平均预测
        - σ(D): 使用 RBF 核的后验标准差
        - κ:    探索/利用权衡系数（越大越倾向探索）

    简化 GP 实现:
        不使用完整高斯过程，而是用滑动窗口内的历史样本通过 RBF 核
        直接计算均值和方差，避免矩阵求逆的开销。
    """

    def __init__(
        self,
        kappa: float = 0.5,
        window_size: int = 10,
        n_candidates: int = 21,
        rbf_lengthscale: float = 15.0,
    ):
        """
        参数:
            kappa:           UCB 探索系数，越大越倾向探索未知区域
            window_size:     滑动窗口大小，只使用最近 N 个历史样本
            n_candidates:    候选难度点的数量（均匀分布在 [0, 100]）
            rbf_lengthscale: RBF 核的长度尺度参数
        """
        self.kappa = kappa
        self.window_size = window_size
        self.n_candidates = n_candidates
        self.rbf_lengthscale = rbf_lengthscale

        # 候选难度点集合
        self._candidates = np.linspace(0.0, 100.0, n_candidates)

        # 历史记录: [(D, attention), ...]
        self._history: list[tuple[float, float]] = []

        # 当前建议的难度
        self._current_D = 50.0

    def reset(self):
        """清空历史样本和当前建议。"""
        self._history.clear()
        self._current_D = 50.0

    def set_params(self, **kwargs):
        """动态修改贝叶斯优化器参数。

        示例:
            bayes.set_params(kappa=3.0, window_size=30)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"BayesianOptimizer 没有参数 '{key}'。"
                    f"可用参数: kappa, window_size, n_candidates, rbf_lengthscale"
                )
        # 若候选数变化需重建候选点
        if "n_candidates" in kwargs:
            self._candidates = np.linspace(0.0, 100.0, self.n_candidates)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (高斯) 核函数。

        K(x, y) = exp( -||x - y||^2 / (2 * l^2) )

        参数:
            X1: 形状 (n,) 或 (n, 1) 的第一组点
            X2: 形状 (m,) 或 (m, 1) 的第二组点

        返回:
            核矩阵，形状 (n, m)
        """
        X1 = np.atleast_2d(X1).T if X1.ndim == 1 else X1
        X2 = np.atleast_2d(X2).T if X2.ndim == 1 else X2
        sq_dists = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-sq_dists / (2.0 * self.rbf_lengthscale ** 2))

    def compute(
        self,
        setpoint: float,
        feedback: float,
        dt: float = 1.0,
        current_D: float | None = None,
    ) -> float:
        """贝叶斯优化控制量计算。

        流程:
            1. 将当前反馈加入历史窗口（作为 (D, attention) 样本对）
            2. 若历史不足，返回默认难度 50
            3. 使用 RBF 核算 μ(D) 和 σ(D)
            4. 计算 UCB 分数: score = μ + κ * σ
            5. 选择分数最高的候选难度

        参数:
            setpoint: 目标专注度
            feedback:  实际专注度
            dt:        采样间隔

        返回:
            难度控制量 D ∈ [0, 100]
        """
        # 1. 记录历史样本
        self._history.append((self._current_D, feedback))

        # 滑动窗口限制
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]

        # 2. 历史不足时返回默认值
        if len(self._history) < 2:
            # 首次运行，选择一个中等难度作为起点
            self._current_D = 50.0
            return self._current_D

        # 3. 提取历史数据
        hist_D = np.array([h[0] for h in self._history])
        hist_y = np.array([h[1] for h in self._history])

        # 4. 计算 RBF 核矩阵
        K = self._rbf_kernel(hist_D, hist_D)  # (n, n)
        noise_var = 1e-3  # 小量噪声保证数值稳定
        K_inv = np.linalg.inv(K + noise_var * np.eye(len(hist_D)))

        # 5. 对每个候选难度计算 μ 和 σ
        best_score = -np.inf
        best_D = self._current_D

        for D_cand in self._candidates:
            # k_* = K(D_cand, hist_D)  形状 (n,)
            k_star = self._rbf_kernel(
                np.array([D_cand]), hist_D
            ).ravel()

            # k_** = K(D_cand, D_cand)  标量
            k_starstar = self._rbf_kernel(
                np.array([D_cand]), np.array([D_cand])
            ).ravel()[0]

            # 均值: μ(D) = k_*^T * K^{-1} * y
            mu = k_star @ K_inv @ hist_y

            # 方差: σ^2(D) = k_** - k_*^T * K^{-1} * k_*
            var = k_starstar - k_star @ K_inv @ k_star
            sigma = np.sqrt(max(var, 1e-8))

            # UCB 分数
            score = mu + self.kappa * sigma

            if score > best_score:
                best_score = score
                best_D = D_cand

        self._current_D = float(best_D)
        return self._current_D
