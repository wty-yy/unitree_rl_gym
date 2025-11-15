# 20251115
## v0.1.3
1. 修改`terrain_level`计算方法, 取当前全部环境等级取平均
2. 使用`heading_command`, 能够更稳定的提升环境等级, 并可以采样到更多的角速度指令
3. 修复`torch.jit`导出问题
4. 分别记录每个地形的奖励
5. 加载训练模型时, 支持环境奖励系数课程加载
# 20251114
## v0.1.2
1. 解决CTS`rollout_storage_cts.py`中学生教授数据采样混乱的问题
# 20251113
## v0.1.1
1. 上下楼梯环境训练7h13min完成, 纯PPO无法学到上楼梯动作, 下楼梯基本能完成
2. 测试10k的wave, slope, rough_slope训练, 修复地形提升问题
3. play.py中加入onnx模型导出功能
4. 机身高度稍微有点低, 提升3cm, `base_height_target: 0.35 -> 0.38`
5. 加入CTS算法替代PPO:
    - 新文件: `on_policy_runner_cts.py, actor_critic_cts.py, cts.py, rollout_storage_cts.py`
    - 新配置: `LeggedRobotCfgCTS`
    - 新任务: `go2_cts`
    - 新导出: 修改torch.script和onnx导出代码, onnx模型的输入是按照IsaacLab的按照item堆叠的结果, 部署C++代码的帧堆叠[obsevation_manager.h](https://github.com/unitreerobotics/unitree_rl_lab/blob/61bfba15d35f1a93e3bacab85fe06b31643c83b7/deploy/include/isaaclab/manager/observation_manager.h#L63)
## v0.1
1. 添加地形选择`wave, slope, rough_slope, stairs down, stairs up, obstacles, stepping_stones, gap, flat`
2. 添加高度特征
3. 修改base_height计算方法, 通过高度特征平均值计算
4. 支持课程奖励系数`rewards.curriculum_rewards`
