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
# The knee can never straighten: the calf range is [-2.723, -0.838], so it
# carries at least 48 degrees of bend no matter what, and -0.85 is the
# straightest usable value, 0.012 rad off the limit. The thigh is also offset
# 0.0955 m laterally from the hip, so a leg axis exactly parallel to the trunk
# is impossible -- the rear pair sits 13.8 degrees off, its minimum.
#
# The equilibrium is balance-on-a-line, not a stable basin: the support polygon
# is the segment between the two front feet, so the pose is an inverted
# pendulum in pitch. Holding it is an active task -- a joint-space PD at this
# posture produces zero torque and the robot falls in under a second, at any
# stiffness (tested 30 to 250; above 60 the calf saturates its 45.4 Nm limit
# and higher gains are worse).
HANDSTAND_HEIGHT = 0.6035
TARGET_PROJECTED_GRAVITY = [1.0, 0.0, 0.0]

# Height of the whole-body centre of mass above the front-foot contact point:
# the pendulum length the reduced-order reward is built on.
#
# MEASURED with scripts/check_stand_init.py -- 0.5806 m at the nominal pose,
# 0.5793 +/- 0.001 over the reset distribution. Re-measure rather than re-derive
# if the posture or the URDF changes; a hand computation from link inertials
# lands 18 mm low, because the rotor and calf-shell bodies carry 1.07 kg that
# sits high in the raised rear legs.
#
# This is the *origin-proxy* centre of mass (see Go2Stand's docstring), which
# reads 8.9 mm above the true one. It belongs with the rewards that use the
# same proxy and must not be reused elsewhere.
HANDSTAND_COM_HEIGHT = 0.5806

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
        # A hold, not a stride, and long enough to tell "just barely holds"
        # from "rock solid". The cap alone does not make the policy value a
        # longer hold -- see algorithm.discount_horizon.
        episode_length_s = 10

    class init_state(Go2Cfg.init_state):
        # Policy targets are offsets from this posture, so action = 0 holds the
        # handstand.
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

        # Stage 0 of the reverse-reachability curriculum: initialise inside the
        # goal set, so the policy only has to learn to hold the pose. Later
        # stages widen pitch and pitch rate together along the paper's
        # closed-form balanceable region, one layer of the backward reachable
        # set per widening.
        #
        # The hook for that is NOT this config. initialize_ranges_for_initial_
        # conditions() runs exactly once, from _init_buffers, so editing these
        # lists mid-training is inert. reset_to_range reads the *tensors* it
        # built -- env.root_pos_range, env.root_vel_range, env.dof_pos_range --
        # live at every reset, and those are what a curriculum must mutate.
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
        # Two legs carry the whole 16.3 kg (160 N) inverted, so the stance
        # gains are raised above Go2's 20/0.5. Expect to tune these: the calf
        # torque limit is 45.4 Nm.
        stiffness = {"hip": 30.0, "thigh": 30.0, "calf": 30.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            # A handstand lives or dies on the front feet not slipping, so the
            # friction floor is high for stage 0. Widen it once the pose holds.
            contact_friction_range = [0.8, 1.0]
            link_mass_scale_range = [0.95, 1.05]

    class termination:
        # Radians of attitude error past which the fall is unrecoverable. The
        # zero-action trace crosses 0.3 rad while still roughly upright and
        # reaches 1.414 rad lying flat, so 0.6 ends the episode without cutting
        # off recovery attempts. A sample-efficiency gate: no reward term
        # mentions attitude, so without it most of the rollout would be spent
        # collecting samples from a robot that is already down.
        max_attitude_error = 0.6

    class reward_settings(Go2Cfg.reward_settings):
        # Read by scripts/check_stand_init.py and by the termination check; no
        # reward uses base_height_target, the height target being on the centre
        # of mass instead.
        base_height_target = HANDSTAND_HEIGHT
        target_projected_gravity = TARGET_PROJECTED_GRAVITY

        # Load-bearing even though _reward_com_height is zero-weighted: it is
        # the pendulum length in the angular-momentum normalizer.
        com_height_target = HANDSTAND_COM_HEIGHT
        # Loose, because varying centre-of-mass height is the control authority
        # the reduced-order model is about. 3 cm scores 0.73, 6 cm scores 0.27,
        # folding the legs to 0.30 m bottoms the term out. Not looser than
        # this: the available height range is only ~0.28 m.
        com_height_tolerance = 0.12
        # Metres of horizontal centre-of-mass offset per error unit, per axis
        # of the support frame: [perpendicular to the contact line, along it].
        #
        # Anisotropic because the support is a 0.284 m segment, not a disc.
        # Offset along x crosses the contact line, where there is no support
        # polygon at all; along y it stays inside the 0.142 m half-width.
        # Charging them equally would price a harmless 3 cm roll offset the
        # same as a 3 cm lean onto the knife edge.
        #
        # x is commensurate with angular_momentum_tolerance rather than chosen
        # independently -- see _reward_balance_offset for the capture-point
        # arithmetic that lands on 0.094 m, of which 0.06 is two-thirds. y puts
        # the polygon edge at e = 0.95.
        support_offset_tolerance = [0.06, 0.15]
        # Metres a support foot may slide from where it was planted, per error
        # unit. Against the logarithmic tail (_shaped_log_tail, not _shaped):
        # 2 cm scores 0.46, 10 cm scores -0.22, and 50 cm scores -0.48 while
        # still pushing back at -0.36 per metre.
        foot_drift_tolerance = 0.05
        # Metres per second of HORIZONTAL centre-of-mass speed per error unit.
        # Vertical speed is uncosted -- raising and lowering the centre of mass
        # is the control authority, and at reset the vertical component
        # (0.066 m/s) is already larger than the horizontal one.
        #
        # Measured over 256 envs: horizontal speed is 0.037 m/s at reset
        # (p95 0.060), 0.088 while the pose settles under zero actions, and
        # 0.66-0.89 at the termination boundary. At 0.25 the reset draw -- an
        # initial condition the policy does not control -- scores 0.885, the
        # settling transient 0.54, a 0.3 m/s translation -0.24, and a fall
        # bottoms out. Resolution is concentrated at the low end, since that is
        # where this term says something balance_momentum does not; in a real
        # fall the two saturate together.
        com_vel_tolerance = 0.25
        # In units of m*h*sqrt(g*h) = 22.6 kg m^2/s, the momentum of the whole
        # robot tipping at the pendulum's natural rate. Per axis of the support
        # frame, and note this is the OPPOSITE assignment to
        # support_offset_tolerance: [about x, about y]. Momentum about y is
        # pitch about the contact line, the underactuated mode, hence tight;
        # about x is roll, which the foot separation arrests, hence loose. A
        # displacement along x is dangerous; a rotation about x is not.
        #
        # Anchored on measurements: whole-body inertia about the contact line
        # is 6.48 kg m^2 (not m*h^2 -- the rear legs are 4.7 kg out at 0.93 m
        # and dominate the second moment), the reset distribution sits at
        # 0.0165 +/- 0.009, and a free fall at the 0.6 rad termination boundary
        # reaches 2.26 rad/s = 0.649 normalized. At 0.15 the reset state scores
        # 0.93, so an untrained policy starts near the ceiling with something
        # to protect; 0.3 rad/s of tipping scores 0.15; the boundary clamps at
        # -0.6. It is also about 1.3x the rear-leg flywheel's authority
        # (swinging the pair at 10 rad/s is dL ~ 2.6, i.e. 0.115 normalized),
        # so one tolerance is roughly one swing's worth of cancellable
        # momentum -- the right scale for a reward meant to let the policy find
        # the flywheel.
        #
        # All of these are against _sqrdexp = exp(-4 e^2): tracking_sigma is
        # 0.25, so the reward reaches 0.37 at e = 0.5, not e = 1. Halving a
        # tolerance here is a much bigger change than it looks.
        angular_momentum_tolerance = [0.25, 0.15]
        # Yaw has its own tolerance and its own reward term -- see
        # Go2Stand._reward_balance_yaw for why it must not join the norm above.
        #
        # 0.0075 keeps the term OFF its clamp in the regime the policy actually
        # occupies, which is what makes its weight a working knob. Measured
        # |L_z|/N: 0.0006 at reset (e = 0.08) and 0.0129 under exploration-scale
        # random actions (e = 1.72, short of the e = 3 clamp). A deliberate
        # 0.3 rad/s pivot costs e = 0.24, 1 rad/s costs e = 0.80. Tighten
        # toward 0.004 if pivoting survives, but re-measure the exploration
        # figure first: below ~0.0043 random actions alone saturate it.
        yaw_momentum_tolerance = 0.0075

    class scaling(Go2Cfg.scaling):
        # Normalize the observed height by the goal, so the critic sees ~1.0
        # when the pose is held.
        base_height = HANDSTAND_HEIGHT
        com_height = HANDSTAND_COM_HEIGHT
        com_offset = 0.1
        com_vel = 0.5
        # support_momentum is deliberately absent: it is already dimensionless
        # and O(0.5) at the edge of recoverability. get_state only divides when
        # the key exists.

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
        # Proprioception only, deliberately: the reduced-order state is
        # critic-side. With a planted front foot the robot is kinematically a
        # fixed-base arm, so trunk attitude, trunk rate and the joint state
        # already determine the centre-of-mass state up to foot slip -- the
        # actor is not missing information in principle, it has to learn the
        # forward kinematics implicitly.
        #
        # No velocity command in this task, so "commands" is dropped.
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
            # The reduced-order state the rewards are built from.
            "com_offset",
            "com_height",
            "com_vel",
            "support_momentum",
        ]
        normalize_obs = False

        class reward:
            class weights:
                # Every term is a bare scalar with ceiling 1.0, so the logged
                # per-second average of each approaches its own weight at the
                # goal and a glance says how close the policy is. Goal total
                # 16.5.
                #
                # The first three are pure functions of the reduced-order
                # state and name no joint, foot or posture, so the policy
                # chooses how it balances.
                balance_momentum = 3.0
                balance_offset = 2.0
                com_vel_xy = 1.0
                # The pivot tracks the live feet, so the balance terms do not
                # notice a foot sliding; these charge for foot motion directly
                # instead of corrupting the balance signal with it. Support
                # also keeps the model's precondition true -- in free flight
                # the reduced-order model does not apply at all.
                front_feet_drift = 2.0
                front_feet_support = 1.5
                # Separate from balance_momentum so it cannot swamp the pitch
                # gradient -- see Go2Stand._reward_balance_yaw.
                balance_yaw = 1.0
                # Not a shaping term -- a constant that keeps the per-step
                # total strictly positive. Without it, riding a fall down
                # costs more per step than terminating does (-1e-4), so the
                # only gradient available to an untrained policy points at
                # falling faster.
                #
                # Every _shaped term floors at -0.6, and front_feet_drift on
                # the logarithmic tail floors at -0.743, so the constraint is
                #
                #     balance_alive >= 0.6 * (balance_momentum + balance_offset
                #                             + com_vel_xy + balance_yaw)
                #                      + 0.743 * front_feet_drift
                #
                # Here 0.6 * 7 + 0.743 * 2 = 5.686, against 6.0 -> worst case
                # +0.31. (front_feet_support is a smoothstep in [0, 1] and
                # cannot go negative, so it never enters.)
                #
                # RAISE THIS IF YOU RAISE ANY OF THOSE WEIGHTS. A test asserts
                # the inequality; a GPU run would not.
                balance_alive = 6.0

            class termination_weight:
                # The terminal reward is -weight * dt, so this is tiny by
                # construction, and deliberately so: the real disincentive to
                # falling is the forgone future return. Normal weights are
                # masked off on a terminating step, so dying gives up the full
                # per-step total over the discount horizon.
                termination = 0.01

    class algorithm(Go2RunnerCfg.algorithm):
        # Derive gamma from a time horizon rather than inheriting 0.99, which
        # at 100 Hz gives the critic only 1.0 s of lookahead -- shorter than
        # the episode, so the policy could not value a longer hold however
        # high the cap was set. set_discount_rates() converts this using
        # ctrl_dt (task_registry.set_discount_rates).
        discount_horizon = 3.0
        # Locomotion's value, and not to be lowered: at 0.001 exploration
        # collapses (action_std 1.00 -> 0.11) and the policy locks into a rigid
        # sub-second hold before it can discover active balance. A rising
        # action_std is exploration staying healthy here, not a fault.
        entropy_coef = 0.01

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2stand"
        max_iterations = 1000
