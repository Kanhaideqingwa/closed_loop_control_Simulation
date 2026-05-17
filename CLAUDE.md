# CLAUDE.md

## 项目概述
BCI 动态闭环控制系统仿真平台 —— 基于 Yerkes-Dodson 定律（倒U型曲线）建立人脑注意力数学模型，实验四种闭环控制算法追踪移动最优难度的性能。

## 运行方式
```bash
cd D:\Claude_Code_Projects\closed_loop_control_Simulation
streamlit run app.py
```
浏览器打开 http://localhost:8501

## 架构

| 文件 | 说明 |
|------|------|
| `brain_model.py` | `SimulatedBrain` —— 受控对象。 Attention(D,t)=exp(-(D-Optimal(t))^2/C)+Noise，Optimal(t) 随时间指数右移模拟学习效应 |
| `controllers.py` | 四种控制器：`PIDController`(增量式+自适应方向估计)、`DiscretePIDController`(量化阶梯输出)、`FuzzyController`(Mamdani推理)、`BayesianOptimizer`(UCB/GP) |
| `simulation_engine.py` | `SimulationEngine` —— 闭环仿真引擎，连接大脑与控制器，记录完整历史 |
| `app.py` | Streamlit 前端 —— 参数面板、Plotly 双Y轴时域图、倒U曲线动态演化图、四维指标卡片、数据表 |
| `requirements.txt` | 依赖：numpy, pandas, scipy, streamlit, plotly |

## 控制算法关键设计决策

- **PID 默认非自适应** (`adaptive=False`)：增量式 PID，Kp=5/Ki=2/Kd=3/deadband=0.01。非自适应模式在倒U型曲线上表现优异(err~0.016)
- **自适应方向估计** (`adaptive=True` 可选)：通过 dA/dD 平滑估计判断曲线侧，含卡死检测安全网。阈值 ±0.05，|dD|>=0.5 才更新
- **离散 PID** (`adaptive=False` 默认)：量化公式 `100/(levels-1)`，levels=10 默认
- **贝叶斯优化**：RBF 核简化 GP + UCB 采集函数，窗口10，kappa=0.5。采样效率最高，3.86%稳态误差
- **模糊控制**：5×5 Mamdani 规则表，三角形隶属度，重心法去模糊化

## 控制器统一接口
```python
ctrl.compute(setpoint, feedback, dt=1.0, current_D=None) -> float  # D ∈ [0,100]
ctrl.reset()
ctrl.set_params(**kwargs)
```

## Git 仓库
- Remote: `git@github.com:Kanhaideqingwa/closed_loop_control_Simulation.git`
- Branch: `main`

## Agent 团队开发记录 (2026-05-15)
项目由 3 个 Agent 协作完成：backend-engineer(算法)、frontend-engineer(前端)、test-engineer(测试)。经过 3 轮迭代修复 10+ Bug，28 项边界测试通过，10/10 验收达标。
