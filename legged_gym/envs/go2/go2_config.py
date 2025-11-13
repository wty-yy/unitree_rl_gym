from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    class env(LeggedRobotCfg.env):
        num_observations = 45
        # obs(45) + base_lin_vel(3) + height_measurements(187)
        num_privileged_obs = 45 + 3 + 187  # 235
        # num_privileged_obs = 48  # without height measurements

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class terrain(LeggedRobotCfg.terrain):
        max_init_terrain_level = 5
        # wave, slope, rough_slope, stairs down, stairs up, obstacles, stepping_stones, gap, flat]
        # terrain_proportions = [0.2, 0.1, 0.1, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]
        terrain_proportions = [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 30. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
        
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35
        only_positive_rewards = False
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
            {'reward_name': 'dof_power', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
        ]
        class scales:
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.2
            # lin_vel_z = -10.0
            # base_height = -50.0
            # action_rate = -0.005
            # similar_to_default = -0.1
            # dof_power = -1e-3  # 能够明显抑制跳跃
            # dof_acc = -3e-7

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -1e-3  # 能够明显抑制跳跃
            # torques = -1e-4  # 无用会走着走着倒了
            correct_base_height = -10.0
            action_rate = -0.01
            action_smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -2.0
            feet_regulation = -0.05
            hip_to_default = -0.1
            # similar_to_default = -0.05

            # flat好奖励
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # dof_acc = -2.5e-7
            # dof_power = -1e-3  # 能够明显抑制跳跃
            # correct_base_height = -10.0
            # action_rate = -0.01
            # action_smoothness = -0.01

    class noise( LeggedRobotCfg.noise ):
        add_noise = True

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'
        max_iterations = 30000
        save_interval = 500
