"""
仿真引擎模块 —— SimulationEngine 类

将大脑模型 (SimulatedBrain) 与控制器绑定，驱动闭环仿真运行。
每一步:
    1. 从大脑获取当前专注度 feedback
    2. 控制器根据 setpoint 和 feedback 计算新难度 D
    3. 记录当前步的状态到历史
    4. 时间步进
"""

from typing import Any

import pandas as pd

from brain_model import SimulatedBrain
from controllers import BaseController


class SimulationEngine:
    """闭环控制仿真引擎。

    将受控对象 (SimulatedBrain) 与控制器串联，形成完整的
    "感知 → 决策 → 执行 → 再感知" 闭环。

    使用方式:
        brain = SimulatedBrain()
        controller = PIDController()
        engine = SimulationEngine(brain, controller)
        history = engine.run(200)
        df = engine.get_history_df()
    """

    def __init__(
        self,
        brain: SimulatedBrain,
        controller: BaseController,
        target_attention: float = 0.95,
    ):
        """
        参数:
            brain:            受控对象 —— 模拟大脑
            controller:       控制器 —— 决定每次施加的难度
            target_attention: 目标专注度设定值
        """
        self.brain = brain
        self.controller = controller
        self.target = target_attention

        # 历史记录列表，每项为包含完整状态信息的字典
        self.history: list[dict[str, Any]] = []
        self.t = 0

        # 上一步的难度（用于引擎内部追踪）
        self._current_D = 50.0

    def step(self) -> dict[str, Any]:
        """执行单步仿真。

        流程:
            1. 从大脑获取当前专注度作为反馈
            2. 控制器根据目标值和反馈计算控制量（新难度）
            3. 将本步状态追加到历史记录
            4. 时间自增

        返回:
            当前步的状态字典，包含:
                - t:         时间步
                - D:         当前施加的难度
                - attention: 当前专注度
                - optimal:   当前最优难度
                - error:     误差 (= target - attention)
                - target:    目标专注度
        """
        # 1. 获取当前大脑状态
        attention = self.brain.attention(self._current_D, self.t)
        optimal = self.brain.optimal_difficulty(self.t)

        # 2. 控制器计算控制量
        D_new = self.controller.compute(
            setpoint=self.target,
            feedback=attention,
            dt=1.0,
            current_D=self._current_D,
        )

        # 3. 构建状态记录
        state = {
            "t": self.t,
            "D": self._current_D,
            "attention": attention,
            "optimal": optimal,
            "error": self.target - attention,
            "target": self.target,
        }
        self.history.append(state)

        # 4. 更新状态：新难度在下一步生效
        self._current_D = D_new
        self.t += 1

        return state

    def run(self, n_steps: int) -> list[dict[str, Any]]:
        """运行 n 步仿真。

        参数:
            n_steps: 仿真总步数

        返回:
            完整历史记录列表（与 self.history 相同引用）
        """
        for _ in range(n_steps):
            self.step()
        return self.history

    def reset(self):
        """重置引擎状态。

        同时重置大脑和控制器，清空历史记录，时间归零。
        适用于需要重复仿真的场景。
        """
        self.brain.reset()
        self.controller.reset()
        self.history.clear()
        self.t = 0
        self._current_D = 50.0

    def get_history_df(self) -> pd.DataFrame:
        """将历史记录转换为 pandas DataFrame。

        便于后续数据分析和可视化。

        返回:
            包含完整控制历史的 DataFrame，列:
                t, D, attention, optimal, error, target
        """
        return pd.DataFrame(self.history)

    def set_controller(self, controller: BaseController):
        """热切换控制器。

        在不重置大脑状态的前提下更换控制策略，
        允许在同一仿真运行中途切换控制器进行对比实验。

        注意: 新控制器会使用其初始状态，不继承旧控制器历史。

        参数:
            controller: 新的控制器实例
        """
        self.controller = controller
