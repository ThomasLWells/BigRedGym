from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg

# Nose-down handstand: the front feet support the robot and the trunk stands
# vertical, so gravity expressed in the body frame points along +x.
#
# The posture below is solved, not guessed: legs as straight as the joints
# allow, with the whole-body centre of mass over the front-foot contact line.
# Measured from the URDF, the result is a column 1.163 m tall -- front feet on
# the ground, rear feet at 1.185 m, and all four feet within 2.3 mm of each
# other in x. Residual CoM offset 0.13 mm (0.02 Nm), against 29.4 Nm for a
# naive bent-leg posture.
#
# Note the knee can never straighten: the calf range is [-2.723, -0.838], so
# it carries at least 48 degrees of bend no matter what. A perfectly straight
# leg is kinematically unreachable; -0.85 is the straightest usable value,
# keeping 0.012 rad off the limit. The thigh is also offset 0.0955 m laterally
# from the hip, so a leg axis exactly parallel to the trunk is impossible
# either -- the rear pair sits 13.8 degrees off, which is its minimum.
#
# The equilibrium is balance-on-a-line, not a stable basin: the support
# polygon is the segment between the two front feet, so the pose is an
# inverted pendulum in pitch. Holding it is an active task -- a joint-space PD
# at this posture produces zero torque and the robot falls in under a second,
# at any stiffness (tested 30 to 250; above 60 the calf saturates its 45.4 Nm
# limit and higher gains are worse).
HANDSTAND_HEIGHT = 0.6035
TARGET_PROJECTED_GRAVITY = [1.0, 0.0, 0.0]

# Front legs reach "down" and rear legs "up" along the trunk axis. The calf is
# at its straightest for both pairs; the thigh angles put the front feet under
# the CoM and the rear legs as close to the trunk axis as they can get.
FRONT_THIGH = -1.140
FRONT_CALF = -0.85
REAR_THIGH = 1.996
REAR_CALF = -0.85

# The calf sits 0.012 rad off its limit, so reset jitter has to be one-sided:
# sampling in the +ve direction would push it past the joint stop.
CALF_JITTER = 0.03


class Go2StandCfg(Go2Cfg):
    class env(Go2Cfg.env):
        # A hold, not a stride. Raised from 6 s because the first stage-1 run
        # saturated at 5.991 s of a 6 s cap, leaving no headroom to tell "just
        # barely holds" from "rock solid". Note the cap alone does not make the
        # policy value longer holds -- see algorithm.discount_horizon.
        episode_length_s = 10

    class init_state(Go2Cfg.init_state):
        # Policy targets are offsets from this posture, so action = 0 holds the
        # handstand. It also re-points the inherited dof_near_home reward at
        # this pose instead of a normal stance.
        default_joint_angles = {
            "hip_joint": 0.0,
            "FL_thigh_joint": FRONT_THIGH,
            "FR_thigh_joint": FRONT_THIGH,
            "FL_calf_joint": FRONT_CALF,
            "FR_calf_joint": FRONT_CALF,
            "RL_thigh_joint": REAR_THIGH,
            "RR_thigh_joint": REAR_THIGH,
            "RL_calf_joint": REAR_CALF,
            "RR_calf_joint": REAR_CALF,
        }

        reset_mode = "reset_to_range"
        pos = [0.0, 0.0, HANDSTAND_HEIGHT]
        # x,y,z,w. Pitch +90 deg about y -- the handstand attitude, which gives
        # projected_gravity = (1, 0, 0). Needed because play.py overrides
        # reset_mode to "reset_to_basic", and that path reads pos/rot rather
        # than the ranges below; the inherited rot is level, so without this a
        # trained policy would be evaluated from a flat drop.
        rot = [0.0, 0.7071068, 0.0, 0.7071068]

        # Stage 1 of the reverse-reachability curriculum: initialise inside the
        # goal set, so the policy only has to learn to hold the pose. Later
        # stages widen pitch and the joint ranges back toward a normal stance,
        # one layer of the backward reachable set per widening. reset_to_range
        # reads these bounds live at every reset (LeggedRobot.reset_to_range),
        # so a curriculum only has to write wider values into the tensors that
        # initialize_ranges_for_initial_conditions() builds from them.
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            # One-sided: the pose is 0.5 mm off the kinematic ceiling, so
            # sampling above it would spawn the robot with its feet in the air.
            [HANDSTAND_HEIGHT - 0.010, HANDSTAND_HEIGHT],  # z
            [0.0, 0.0],  # roll
            [1.55, 1.59],  # pitch [rad] -- pi/2 = 1.5708 is the vertical trunk
            [0.0, 0.0],  # yaw
        ]
        root_vel_range = [
            [-0.05, 0.05],  # x
            [-0.05, 0.05],  # y
            [-0.05, 0.05],  # z
            [-0.1, 0.1],  # roll
            [-0.1, 0.1],  # pitch
            [-0.1, 0.1],  # yaw
        ]
        dof_pos_range = {
            "hip_joint": [-0.05, 0.05],
            "FL_thigh_joint": [FRONT_THIGH - 0.05, FRONT_THIGH + 0.05],
            "FR_thigh_joint": [FRONT_THIGH - 0.05, FRONT_THIGH + 0.05],
            # Calf jitter is one-sided (more bend only) to stay off the stop.
            "FL_calf_joint": [FRONT_CALF - CALF_JITTER, FRONT_CALF],
            "FR_calf_joint": [FRONT_CALF - CALF_JITTER, FRONT_CALF],
            "RL_thigh_joint": [REAR_THIGH - 0.05, REAR_THIGH + 0.05],
            "RR_thigh_joint": [REAR_THIGH - 0.05, REAR_THIGH + 0.05],
            "RL_calf_joint": [REAR_CALF - CALF_JITTER, REAR_CALF],
            "RR_calf_joint": [REAR_CALF - CALF_JITTER, REAR_CALF],
        }
        dof_vel_range = {
            "hip": [-0.1, 0.1],
            "thigh": [-0.1, 0.1],
            "calf": [-0.1, 0.1],
        }

    class control(Go2Cfg.control):
        # Two legs carry the whole 16.3 kg (160 N) inverted, where four shared
        # it before, so the stance gains are raised from Go2's 20/0.5. Expect
        # to tune these: the calf torque limit is 45.4 Nm.
        stiffness = {"hip": 30.0, "thigh": 30.0, "calf": 30.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            # A handstand lives or dies on the front feet not slipping, so the
            # friction floor is raised from Go2's 0.5 for stage 1. Widen it
            # again once the pose holds.
            contact_friction_range = [0.8, 1.0]
            link_mass_scale_range = [0.95, 1.05]

    class termination:
        # Radians of attitude error past which the fall is unrecoverable. The
        # zero-action trace crosses 0.3 rad while still roughly upright and
        # reaches 1.414 rad lying flat, so 0.6 ends the episode without cutting
        # off recovery attempts. Without this the only termination is the trunk
        # hitting the ground, by which point ~80% of a 6 s episode has been
        # spent collecting samples from a robot that is already down.
        max_attitude_error = 0.6

    class reward_settings(Go2Cfg.reward_settings):
        base_height_target = HANDSTAND_HEIGHT
        # A miss of this size costs one full error unit. This is a tolerance,
        # not an observation normalizer -- do not collapse it into
        # scaling.base_height.
        base_height_tracking_scale = 0.04
        # Radians of attitude error, roughly: for small angles the norm of the
        # projected-gravity difference is the angle itself.
        orientation_tolerance = 0.2
        target_projected_gravity = TARGET_PROJECTED_GRAVITY

    class scaling(Go2Cfg.scaling):
        # Normalize the observed height by the goal, so the critic sees ~1.0
        # when the pose is held.
        base_height = HANDSTAND_HEIGHT
        # Residual radians per policy unit. Go2's default is
        # [0.5 * x for x in dof_pos] = 4 * [0.524, 1.265, 0.471], so one unit
        # of output swings a thigh over a radian -- with init_noise_std = 1.0
        # the untrained policy's exploration noise alone destroys the pose on
        # the first step. Balance needs finer authority than locomotion.
        dof_pos_target = 4 * [0.1, 0.2, 0.2]


class Go2StandRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        activation = "elu"
        # No velocity command in this task, so "commands" is dropped:
        # projected_gravity carries the attitude the policy has to achieve.
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]
        normalize_obs = False
        actions = ["dof_pos_target"]
        add_noise = False
        disable_actions = False

    class critic(Go2RunnerCfg.critic):
        hidden_dims = [128, 64]
        activation = "elu"
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]
        normalize_obs = False

        class reward:
            class weights:
                # The pose itself.
                orientation = 3.0
                min_base_height = 2.0
                front_feet_support = 2.0
                # Keep the support feet planted, so balance comes from
                # articulating the front legs rather than repositioning the
                # feet. Drift is the primary signal (metres from where the
                # foot was planted): at this weight 2 cm costs 0.1 and 10 cm
                # costs 0.5, against a total reward around 11.
                front_feet_drift = 5.0
                # Speed only damps oscillation about the anchor. Low weight
                # because it cannot tell sliding from the foot rolling in
                # place, and at 5.0 it plateaued at -0.5 by charging the
                # policy for the ankle motion the task needs.
                front_feet_slip = 1.0
                rear_feet_air = 1.0
                # default_dof_pos is the handstand posture, so this anchors the
                # support configuration rather than a normal stance.
                dof_near_home = 0.5
                # Regularizers.
                ang_vel_xy = 0.05
                torques = 5.0e-6
                action_rate = 0.1
                action_rate2 = 0.05
                # Meaningless here: there is no velocity command to track, and
                # vertical motion is how the robot gets into the pose.
                tracking_lin_vel = 0.0
                tracking_ang_vel = 0.0
                lin_vel_z = 0.0
                dof_vel = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0

            class termination_weight:
                termination = 0.01

    class algorithm(Go2RunnerCfg.algorithm):
        # Derive gamma from a time horizon rather than inheriting 0.99, which
        # at 100 Hz gives the critic only 1.0 s of lookahead -- shorter than
        # the episode, so the policy could not value a longer hold however
        # high the cap was set. set_discount_rates() converts this using
        # ctrl_dt (task_registry.set_discount_rates).
        discount_horizon = 3.0

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2stand"
        max_iterations = 1000
