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

# Reduced-order pendulum constants, for the curriculum (go2stand_curriculum.py).
#
# HANDSTAND_PENDULUM_LENGTH is NOT HANDSTAND_COM_HEIGHT and the two must not be
# swapped. The reward measures the centre of mass above the foot body ORIGIN,
# because that is where the support anchor sits. The pendulum pivots on the
# contact PLANE, one foot radius lower, so it is 22 mm longer. Using 0.5806
# here understates the goal energy by 3.5 J.
#
# HANDSTAND_INERTIA is measured kinematically, not fitted from a fall: impose a
# unit rigid rotation about the contact line with the joints locked and read
# 2*KE off the mass matrix. Fitting a zero-action fall instead gives nonsense,
# because a position PD at this posture produces zero torque, so the legs sag
# 6.5 cm before the robot tips -- that trajectory is variable-length, not a
# rigid rotation, and I_eff = 2*KE/theta_dot^2 blows up while the motion is
# radial. Curriculum states ARE rigid rotations, so this is the right number
# for them.
#
# Note m*r^2 = 5.91 against I = 6.48: the point-mass form is 9% low, the rear
# legs being 4.7 kg out at 0.93 m.
HANDSTAND_MASS = 16.2764
HANDSTAND_PENDULUM_LENGTH = 0.60262
HANDSTAND_INERTIA = 6.4796
# Front-foot contact point, 2.3 mm behind the base origin in the solved pose.
HANDSTAND_PIVOT_X = -0.00228

# Curriculum schedule, generated rather than listed -- see
# HandstandPendulum.schedule. Starts where the policy already is: the old reset
# box (pitch +/-0.02 rad, rate +/-0.1 rad/s) is 0.019 J of potential plus
# 0.032 J of kinetic, so stage 0 at 0.05 J is the task already solved.
#
# Geometric growth keeps the early steps sub-degree, but unchecked it turns
# into 6 degree jumps at the top, so the lean step is capped. Together: 19
# stages, biggest energy jump 2.3 J and biggest lean jump 2.5 degrees, against
# 8.0 J and 8.9 degrees for plain doubling.
CURRICULUM_START_ENERGY = 0.05
CURRICULUM_RATIO = 1.5
CURRICULUM_MAX_LEAN_STEP = 0.0436  # 2.5 degrees

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

# Foot collision sphere, from the URDF: <sphere radius="0.022"/> at
# <origin xyz="-0.002 0 0"/> relative to the foot link origin. Needed to find
# the contact point, which on flat ground sits directly below the sphere
# centre -- see Go2Stand._update_foot_slip.
FOOT_RADIUS = 0.022
FOOT_SPHERE_OFFSET = [-0.002, 0.0, 0.0]
# Metres per second past which a slip observation is clipped. Observations
# are not normalized in this task, so an unbounded spike reaches the network
# raw, and a crashing robot produces several m/s.
#
# Measured over a trained rollout, load-gated (see Go2Stand._update_foot_slip):
# 0.038 m/s mean, 0.085 at p90, 0.647 at p99, with 0.59% of samples reaching
# this clip. Those remaining clips are genuine -- a foot carrying weight and
# sliding at a metre per second. Before the gate the p99 sat exactly ON the
# clip, because an airborne foot was being reported as catastrophic slip.
FOOT_SLIP_MAX = 1.0


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

        reset_mode = "reset_to_curriculum"
        pos = [0.0, 0.0, HANDSTAND_HEIGHT]
        # x,y,z,w. Pitch +90 deg about y -- the handstand attitude, which gives
        # projected_gravity = (1, 0, 0). Needed because play.py overrides
        # reset_mode to "reset_to_basic", and that path reads pos/rot rather
        # than the ranges below; the inherited rot is level, so without this a
        # trained policy would be evaluated from a flat drop.
        rot = [0.0, 0.7071068, 0.0, 0.7071068]

        # SUPERSEDED for the root by reset_to_curriculum, which samples the
        # lean and lean rate together from an energy manifold -- a correlated
        # distribution, with the base position a function of the lean, that
        # independent per-axis boxes cannot express. Kept because play.py
        # overrides reset_mode to "reset_to_basic", because
        # initialize_ranges_for_initial_conditions reads all four lists, and
        # because this box is the measurement stage 0 is sized against:
        # pitch +/-0.02 rad is 0.019 J and +/-0.1 rad/s is 0.032 J, so
        # CURRICULUM_START_ENERGY = 0.05 J is this same set.
        #
        # The dof lists below are NOT superseded -- reset_to_curriculum draws
        # the joint jitter from exactly these, the whole-body lean being a
        # rigid rotation that leaves the joints where they were.
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

    class asset(Go2Cfg.asset):
        # Foot geometry, used to locate the contact point rather than by the
        # physics -- the backend takes its collision shapes from the URDF.
        foot_radius = FOOT_RADIUS
        foot_sphere_offset = FOOT_SPHERE_OFFSET
        foot_slip_max = FOOT_SLIP_MAX

    class control(Go2Cfg.control):
        # Two legs carry the whole 16.3 kg (160 N) inverted, so the stance
        # gains are raised above Go2's 20/0.5. Expect to tune these: the calf
        # torque limit is 45.4 Nm.
        stiffness = {"hip": 30.0, "thigh": 30.0, "calf": 30.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}

        # Radians the PD target may sit from the default pose, clipped in
        # Go2Stand._pre_decimation_step. The actor is an UNSQUASHED Gaussian
        # and nothing downstream bounds it, so without this the action is
        # limited only by what the network happens to output.
        #
        # It stopped being hypothetical. Measured on two trained policies:
        # the good one (Nov25 iteration 1450) commands |target| of 0.39 rad at
        # p99 and 0.96 at its worst, while the policy that followed commanded
        # 121 rad at p99 and 132 at its worst -- twenty-one full revolutions
        # of a joint that physically travels about two. That drove joints to
        # 173 rad/s against a URDF rating of 15.7 for the calf, with torque
        # pinned at the clip 89.5% of the time.
        #
        # 1.5 is 1.6x the good policy's observed worst case, so it is not a
        # constraint on anything that works, and 80x below what the blown-up
        # policy was asking for. It also BOUNDS the three actuation penalties
        # below, which an unclipped action does not -- see the weights block.
        max_dof_pos_target = 1.5

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
        # off recovery attempts. A sample-efficiency gate, deliberately far
        # looser than _reward_balance_attitude, which is what holds the pose.
        max_attitude_error = 0.6

        # Metres a support foot may travel from where the handstand SETTLED
        # before the episode ends. Walking is not a handstand, and pricing it
        # was not enough: the policy that trained without this held 1.2 deg of
        # lean and perfect balance while covering 57 cm from a motionless
        # start, because surviving paid far more than the drift term cost.
        #
        # Safe only because the anchor latches -- see Go2Stand._settle_drift.
        # Until the robot is upright the anchor follows its feet, so drift
        # reads ~0 and this cannot fire; stepping to recover from a 29 deg
        # spawn is never fatal. The two are a pair, and this threshold alone
        # would starve exactly the recovery the curriculum exists to teach.
        #
        # Sized on the MAX of a good policy, not its median. Replaying the two
        # measured policies over 153k env-steps each: the good one (iteration
        # 1450) runs 4.7 cm median, 8.7 cm p90 -- but 18.8 cm at its worst.
        # The walker runs 57 cm median, 85 cm max. A 0.20 m limit would have
        # left the good policy only 6% of headroom and clipped its tail
        # excursions for nothing; 0.30 gives it 1.6x its observed worst case
        # while still ending the walker's episodes in well under a second.
        max_foot_drift = 0.30

    class curriculum:
        # Stage energies, from HandstandPendulum.schedule -- see the
        # CURRICULUM_* constants above for why these three values.
        start_energy = CURRICULUM_START_ENERGY
        ratio = CURRICULUM_RATIO
        max_lean_step = CURRICULUM_MAX_LEAN_STEP

        # Advancement, by periodic REVIEW rather than a running average. A
        # review fires once the window holds both enough episodes and enough
        # elapsed time, scores them, and clears. Success is
        # timed_out & ~terminated: the flags are not mutually exclusive, so
        # gating on the timeout alone scores falls that happen to coincide
        # with the horizon as successes.
        #
        # A window closes when this fraction of the envs have each reported
        # ONE trial at the current stage. Counting envs rather than episodes is
        # not a detail -- it is what makes the rate an unbiased estimate.
        #
        # Episode length correlates with outcome by construction: a fall
        # resolves in well under a second, a hold takes the full 10 s. So a
        # window of "the first N episodes to finish" over-samples failures by
        # roughly the ratio of those durations, and a failing env can report
        # twenty times while a succeeding one reports once. Measured on a
        # policy with a true 90% success rate, such a window reads 0.68 in the
        # steady state and 0.40 on the first review after a stage change --
        # which would demote a perfectly competent policy. One trial per env
        # removes the bias entirely.
        #
        # 1.0, not "most of them". Closing early truncates the sample at
        # whichever envs report first, and those are the FAST ones -- failures
        # -- so any fraction below 1.0 reintroduces the same bias in miniature:
        # at 0.9 a true 30% policy reads 0.20. There are no stragglers to guard
        # against, because timed_out fires at max_episode_length and so every
        # env is guaranteed to report within two horizons.
        review_fraction = 1.0
        # Control steps a window must also span, which is what actually paces
        # the curriculum: episodes alone can arrive in a burst from envs whose
        # episode phases have synchronised, and 150 episodes from one cohort
        # are 150 samples of a single policy snapshot. common_step_counter is
        # the only clock the env has.
        #
        # CONVERSION, which moves if rollout_size or the actor frequency moves:
        # steps per iteration = num_steps_per_env * n_policy_steps = 8 * 1 = 8,
        # so 80 steps is 10 iterations.
        #
        # This floor is a guard, not the pacer, and it is worth knowing what
        # the pacer is. A window needs EVERY env to finish an episode begun
        # after the window opened; an env that reset just before it must first
        # run out the episode already in flight, so the bound is two horizons
        # -- 2000 control steps, 250 iterations -- and waiting for all of them
        # means always paying it. Measured at 250 iterations per promotion.
        # That latency is a property of episode_length_s, not a tunable, and
        # it is what runner.max_iterations is sized against.
        min_review_steps = 80
        # Promote on sustained success, demote on sustained failure. The gap
        # between the two IS the anti-oscillation mechanism: a policy holding
        # 0.6 sits in the dead band and the stage does not move.
        promote_rate = 0.8
        demote_rate = 0.4
        # Surviving the horizon is NOT success on its own -- that is the
        # criterion that promoted a leaning policy through fifteen stages. The
        # episode must also have HELD the handstand: mean attitude error over
        # the scored span, in chord units, at most this. 0.15 is 8.6 degrees
        # of time-averaged tilt; the policy measured at a steady 15 degree
        # lean reads 0.26 and fails.
        #
        # Averaged over the TAIL of the episode, not all of it. The robot
        # spawns at the stage's lean cap -- 21 degrees at stage 13, 34 at
        # stage 18 -- so a whole-episode mean charges every stage for its own
        # spawn transient, and charges the hard stages most. The tail is
        # stage-independent and asks the right question: a recovery that takes
        # five seconds is not a handstand.
        hold_attitude_error = 0.15
        hold_fraction = 0.5

        # There is deliberately no "N consecutive reviews" rule. A window is
        # one unbiased sample of the whole population, so at 4096 envs its
        # standard error is sqrt(0.25/4096) = 0.008 and the dead band above is
        # 25 standard errors wide on each side -- noise cannot walk the stage
        # across it. Requiring two reviews would double the one cost that
        # actually binds (a review is ~190 iterations) and buy nothing; one
        # window of 2N is a better estimator than two windows of N that must
        # both pass.
        # Radians of lean past the stage's spawn cap that the attitude
        # termination must allow. Without it the last stage spawns 0.017 rad
        # inside the threshold and dies on its first outward step.
        termination_margin = 0.25

    class reward_settings(Go2Cfg.reward_settings):
        # Read by scripts/check_stand_init.py and by the termination check; no
        # reward uses base_height_target, the height target being on the centre
        # of mass instead.
        base_height_target = HANDSTAND_HEIGHT
        target_projected_gravity = TARGET_PROJECTED_GRAVITY
        # Chord on the unit sphere per error unit of attitude: projected
        # gravity is a unit vector, so a lean of phi costs 2 sin(phi/2). This
        # is 2 sin(2.5 deg), i.e. ONE ERROR UNIT AT 5 DEGREES OF LEAN.
        #
        # Deliberately tight, and sharper than it looks: _sqrdexp is
        # exp(-4 e^2), so the term is nearly dead by one tolerance -- 2 deg
        # scores 0.46 and 5 deg is already slightly negative. That is what
        # buys a crisp handstand rather than a merely survivable one, and it
        # matches the convention elsewhere here (foot drift scores 0.46 at
        # 2 cm against a 5 cm tolerance). For a 5 deg lean to be roughly
        # neutral instead, this wants to be about 0.21.
        #
        # The tolerance governs the FINISH, not the escape from a lean: the
        # gain from 15 deg to vertical is 1.28 per unit weight here and 1.13
        # at 0.21. The weight is the knob for the escape.
        attitude_tolerance = 0.0872
        # Chord at which the handstand counts as ACHIEVED, latching the foot
        # drift anchor. Below this the robot is still righting itself and its
        # anchor follows its feet, so stepping to get upright costs nothing;
        # at this point the anchor freezes and every centimetre after is
        # charged. 0.15 is 8.6 deg -- the same value as
        # curriculum.hold_attitude_error, because it means the same thing.
        drift_settle_attitude = 0.15

        # Load-bearing even though _reward_com_height is zero-weighted: it is
        # the pendulum length in the angular-momentum normalizer.
        com_height_target = HANDSTAND_COM_HEIGHT

        # Reduced-order pendulum constants, read by HandstandPendulum.
        # from_config. No reward uses them.
        handstand_mass = HANDSTAND_MASS
        handstand_pendulum_length = HANDSTAND_PENDULUM_LENGTH
        handstand_inertia = HANDSTAND_INERTIA
        handstand_pivot_x = HANDSTAND_PIVOT_X
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
        # unit. Against the logarithmic tail (_shaped_log_tail, the task's one
        # shaping function): 2 cm scores 0.46, 10 cm scores -0.22, and 50 cm
        # scores -0.48 while
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
        # Size this against the rate the policy actually turns at, NOT against
        # exploration noise. An earlier 0.0075 was chosen to stay off the clamp
        # under untrained random actions, which put its knee at a yaw rate of
        # 1.24 rad/s while a trained policy wobbles at about 0.15 -- eight
        # times too loose to charge for the behaviour it exists to stop.
        #
        # The drift lesson does not transfer. Drift accumulates monotonically,
        # so a clamp there is permanent and the term dies. Yaw momentum is a
        # rate and recovers, so saturating early in training is transient and
        # survivable.
        #
        # At 0.002, against the measured |L_z|/N of a trained policy: median
        # 0.00026 scores 0.93, p90 0.00082 scores 0.51, p99 0.00153 scores
        # 0.09. That charges the tail without flattening the middle. For
        # reference, rate-matching the pitch tolerance would give 0.0031.
        yaw_momentum_tolerance = 0.002

    class scaling(Go2Cfg.scaling):
        # Normalize the observed height by the goal, so the critic sees ~1.0
        # when the pose is held.
        base_height = HANDSTAND_HEIGHT
        com_height = HANDSTAND_COM_HEIGHT
        com_offset = 0.1
        com_vel = 0.5
        # Metres per second per observation unit. Load-gated slip over a
        # trained rollout: median 0.008, mean 0.038, p90 0.085, p99 0.647.
        # Mostly near zero with a heavy tail, which is what a "something is
        # wrong" detector should look like. At 0.1 a planted foot reads 0.08,
        # a real slip event clears 1.0, and FOOT_SLIP_MAX bounds it at 10 --
        # the same order as dof_vel's worst case.
        foot_slip = 0.1
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
            # The one non-proprioceptive entry, and deliberate. Without it the
            # actor cannot tell a slippery floor from a grippy one, so across
            # the 0.8-1.0 friction range it can only learn a single policy
            # that averages over the draw -- domain randomization measures as
            # 81% of the episode-to-episode spread in foot drift. Slip is what
            # makes low friction something the policy can respond to rather
            # than merely suffer.
            "foot_slip",
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
            "foot_slip",
        ]
        normalize_obs = False

        class reward:
            class weights:
                # Every term is a bare scalar with ceiling 1.0, so the logged
                # per-second average of each approaches its own weight at the
                # goal and a glance says how close the policy is. Goal total
                # 22.5.
                #
                # The first three are pure functions of the reduced-order
                # state and name no joint, foot or posture, so the policy
                # chooses how it balances.
                balance_momentum = 3.0
                balance_offset = 2.0
                com_vel_xy = 1.0
                # NOT a reduced-order term despite the prefix -- it names the
                # trunk. It exists because the reduced-order state cannot see
                # a lean: the rear legs are exactly the redundancy that puts
                # the centre of mass over the contact line at ANY trunk angle,
                # so a motionless 15 degree lean scored the full goal total
                # and a policy trained for 5000 iterations learned precisely
                # that. Weighted level with balance_momentum because being
                # upright is what makes the pose a handstand rather than
                # merely a balance.
                balance_attitude = 3.0
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

                # Actuation regularisation, the three weights go2trot uses --
                # the only task here that produces physically plausible motion
                # -- at exactly its values. go2stand previously weighted NONE
                # of them, and the result was a policy running bang-bang:
                # torque at the clip 89.5% of the time, the PD target slewing
                # 1.7 rad per 10 ms control step, and joints reaching 173
                # rad/s against a URDF rating of 15.7 for the calf, 11x over.
                #
                # These are penalties: each returns a negative mean square, so
                # at the goal they contribute 0 and the goal total is
                # unchanged. They are also the only terms here NOT bounded by
                # their own shape -- control.max_dof_pos_target is what bounds
                # them, see below.
                action_rate = 0.1
                action_rate2 = 0.05
                torques = 5.0e-6
                # Joint speed past the motor's rating, overridden in
                # Go2Stand._reward_dof_vel_limits because the inherited
                # version saturates 1 rad/s past the limit and would be a
                # constant across the whole range that matters.
                #
                # 3.0 from the measured separation between the two policies:
                # the good one (Nov25 iteration 1450) reads -0.010 on average
                # and is charged at all on only 2.9% of steps, the blown-up
                # one reads -0.347 and is charged on 68.6%. So this costs a
                # working policy about 0.03 of a 22.5 total and a thrashing
                # one about 1.04 -- real without being punitive.
                #
                # It is the cheap stand-in for a torque-speed curve. The
                # motor model here delivers full torque at any speed, which
                # the real one cannot, and at 3820 rad/s^2 the calf covers
                # its entire 15.7 rad/s rating in 4.1 ms -- under half a
                # control step. So nothing in the torque path bounds joint
                # speed and this is a preference competing with survival
                # rather than a limit. If joints still run past rating, the
                # fix is the torque-speed curve in _compute_torques.
                dof_vel_limits = 3.0
                # Not a shaping term -- a constant that keeps the per-step
                # total strictly positive. Without it, riding a fall down
                # costs more per step than terminating does (-1e-4), so the
                # only gradient available to an untrained policy points at
                # falling faster.
                #
                # Every shaped term shares the one logarithmic tail, so each
                # floors at -0.743 -- EXCEPT balance_attitude, whose argument
                # is a chord on the unit sphere and so cannot exceed 2.0. Its
                # error tops out at 2.0 / 0.0872 = 22.9 and it bottoms at
                # -0.635, not -0.743. The constraint is therefore
                #
                #     balance_alive >= 0.743 * (balance_momentum
                #                               + balance_offset + com_vel_xy
                #                               + balance_yaw
                #                               + front_feet_drift)
                #                      + 0.635 * balance_attitude
                #
                # Here 0.743 * 9 + 0.635 * 3 = 8.59, against 9.0 -> worst case
                # +0.41. (front_feet_support is a smoothstep in [0, 1] and
                # cannot go negative, so it never enters.)
                #
                # Raised from 7.0 with balance_attitude. Note the ratio to the
                # shaped sum is preserved -- 7.0/17.5 and 9.0/22.5 are both
                # 40% -- so the relative value of surviving versus balancing
                # well has not moved.
                #
                # The three actuation penalties are deliberately NOT in that
                # inequality, and this is the one place the invariant is
                # knowingly relaxed. They are bounded -- the action clip caps
                # them at roughly -2.9 combined -- but covering that worst
                # case would mean an alive bonus near 12.0, pushing the
                # survive-at-all-costs pressure up by a further seven points
                # of share, which is the pressure that produced a walking
                # handstand in the first place.
                #
                # The distinction that makes this safe: the invariant exists
                # because a FALLING robot cannot stop its momentum and offset
                # terms blowing up, so without a floor it would learn that
                # dying fast is cheaper than living. Actuation cost is not
                # like that -- it is entirely the policy's own choice, zero at
                # the start of training when actions are near zero, and
                # reachable only by actively chattering at full amplitude,
                # which these very terms punish. The corner exists but nothing
                # pushes the policy into it.
                #
                # RAISE THIS IF YOU RAISE ANY OF THE SHAPED WEIGHTS. A test
                # asserts the inequality; a GPU run would not.
                balance_alive = 9.0

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
        # Sized against the curriculum, not picked round. An honest review
        # costs TWO episode horizons -- see curriculum.min_review_steps -- so a
        # stage costs about 250 iterations, measured. Reaching the frontier
        # found at stage 12-13 therefore costs roughly 3250, on top of the
        # ~210 that learning the handstand at all took last time. 5000 leaves
        # about 1500 iterations to consolidate once it settles.
        #
        # This is the dominant cost of the curriculum and it is set by
        # episode_length_s, not by the gate: halving the episode to 5 s would
        # halve it. Cut this budget only with that in mind.
        max_iterations = 5000
