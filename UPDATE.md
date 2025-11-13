# 20251113
## v0.1.1
1. 上下楼梯环境训练7h13min完成, 纯PPO无法学到上楼梯动作, 下楼梯基本能完成
2. 测试10k的wave, slope, rough_slope训练, 修复地形提升问题
3. play.py中加入onnx模型导出功能
TODO: 
- 加入CTS算法替代PPO
## v0.1
1. 添加地形选择`wave, slope, rough_slope, stairs down, stairs up, obstacles, stepping_stones, gap, flat`
2. 添加高度特征
3. 修改base_height计算方法, 通过高度特征平均值计算
4. 支持课程奖励系数`rewards.curriculum_rewards`
