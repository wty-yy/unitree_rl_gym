"""输入rsl_rl的日志打印奖励的柱状图"""

log_text = """
      Mean episode rew_action_rate: -0.0289
Mean episode rew_action_smoothness: -0.0399
       Mean episode rew_ang_vel_xy: -0.0347
        Mean episode rew_collision: -0.0002
Mean episode rew_correct_base_height: -0.0017
          Mean episode rew_dof_acc: -0.0327
   Mean episode rew_dof_pos_limits: -0.0039
        Mean episode rew_dof_power: -0.0586
  Mean episode rew_feet_regulation: -0.0183
   Mean episode rew_hip_to_default: -0.0324
"""

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5, 6))
for line in log_text.splitlines():
    if 'Mean episode rew_' in line:
        parts = line.split('Mean episode rew_')[1].split(':')
        reward_name = parts[0].strip()
        reward_value = float(parts[1].strip())
        plt.bar(reward_name, reward_value)
# 斜者显示x轴标签
plt.xticks(rotation=45, ha='right')
# 增大x轴标签距离
plt.subplots_adjust(bottom=0.25)
plt.grid(axis='x')
plt.tight_layout()
plt.show()
