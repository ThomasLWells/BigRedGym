"""Reduced-order state and rewards for the handstand, without a simulator.

Built with ``Go2Stand.__new__`` and hand-assigned tensors, following
test_go2trot.py: only the attributes each method actually touches are set, so
a test failure points at the maths rather than at the physics backend.
"""

import math

from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2stand import Go2Stand
from gym.envs.go2.go2stand_config import Go2StandCfg, Go2StandRunnerCfg

# Handstand-ish geometry, rounded: the measured pose has the centre of mass
# 0.5806 m above the support anchor and the front feet 0.284 m apart.
HEIGHT = Go2StandCfg.reward_settings.com_height_target
HALF_SPAN = 0.142


def _task(num_envs=1, masses=(1.0,), positions=None, velocities=None):
    """A Go2Stand carrying just enough state to compute the reduced order."""
    task = Go2Stand.__new__(Go2Stand)
    task.cfg = Go2StandCfg()
    task.num_envs = num_envs
    task.device = "cpu"
    task._backend = SimpleNamespace(
        link_mass=torch.tensor([list(masses)] * num_envs, dtype=torch.float)
    )
    bodies = len(masses)
    task._rigid_body_pos = (
        torch.zeros(num_envs, bodies, 3) if positions is None else positions.clone()
    )
    task._rigid_body_lin_vel = (
        torch.zeros(num_envs, bodies, 3) if velocities is None else velocities.clone()
    )
    task._support_anchor = torch.zeros(num_envs, 3)
    # Identity frame unless a test captures a real one.
    task._support_rot = torch.eye(2).expand(num_envs, 2, 2).clone()
    return task


def _lumped(offset, velocity, mass=16.2764, height=HEIGHT):
    """One point mass at (offset, 0, height) moving at ``velocity``."""
    positions = torch.tensor([[[offset, 0.0, height]]])
    velocities = torch.tensor([[list(velocity)]])
    return _task(masses=(mass,), positions=positions, velocities=velocities)


def _tolerances(task):
    """The per-axis tolerance tensors _init_buffers would have built."""
    task._offset_tolerance = torch.tensor(
        Go2StandCfg.reward_settings.support_offset_tolerance
    )
    task._momentum_tolerance = torch.tensor(
        Go2StandCfg.reward_settings.angular_momentum_tolerance
    )
    return task


def test_static_robot_over_the_anchor_has_no_offset_and_no_momentum():
    task = _lumped(0.0, (0.0, 0.0, 0.0))
    offset, height, velocity, momentum = task._compute_reduced_order_state()

    torch.testing.assert_close(offset, torch.zeros(1, 2))
    torch.testing.assert_close(height, torch.tensor([[HEIGHT]]))
    torch.testing.assert_close(velocity, torch.zeros(1, 3))
    # The whole design rests on this: at rest the proxy is exactly zero, so
    # the reward's optimum sits where the true optimum does even though the
    # per-link positions are body origins rather than link centres of mass.
    torch.testing.assert_close(momentum, torch.zeros(1, 3))


def test_angular_momentum_matches_the_analytic_cross_product():
    mass, speed = 16.2764, 0.25
    task = _lumped(0.0, (speed, 0.0, 0.0), mass=mass)
    _, _, _, momentum = task._compute_reduced_order_state()

    # L = m * r x v with r = (0, 0, h) and v = (speed, 0, 0) gives +m*h*speed
    # about y, then normalized by m * h * sqrt(g * h) -- so the mass and the
    # height both cancel and only speed / sqrt(g * h) survives.
    expected = speed / math.sqrt(9.81 * HEIGHT)
    assert momentum[0, 1].item() == pytest.approx(expected, rel=1e-5)
    assert momentum[0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert momentum[0, 2].item() == pytest.approx(0.0, abs=1e-7)


def test_tipping_forward_raises_momentum_about_the_contact_line():
    """Sign check against d/dt L_p = (c - p) x m g.

    Gravity on a centre of mass at +x produces +y angular momentum about the
    contact line, so a robot that has started to tip forward must read a
    positive y component. Getting this backwards would invert the entire
    reward while leaving every magnitude test passing.
    """
    forward = _lumped(0.05, (0.1, 0.0, 0.0))
    backward = _lumped(-0.05, (-0.1, 0.0, 0.0))

    assert forward._compute_reduced_order_state()[3][0, 1].item() > 0.0
    assert backward._compute_reduced_order_state()[3][0, 1].item() < 0.0


def test_reduced_order_state_is_invariant_to_yaw():
    """Rotating the whole scene about z must not change what the critic sees.

    The rewards are rotation-invariant norms and would pass this trivially;
    the observations are vectors and would not, which is the only reason the
    support frame exists.
    """
    feet = torch.tensor([[[0.0, HALF_SPAN, 0.0], [0.0, -HALF_SPAN, 0.0]]])
    body = torch.tensor([[[0.03, 0.0, HEIGHT]]])
    body_vel = torch.tensor([[[0.12, -0.04, 0.0]]])
    foot_vel = torch.zeros(1, 2, 3)

    def state_at(yaw):
        cos, sin = math.cos(yaw), math.sin(yaw)
        rot = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        positions = torch.cat((body, feet), dim=1) @ rot.T
        velocities = torch.cat((body_vel, foot_vel), dim=1) @ rot.T
        task = _task(
            masses=(16.0, 0.04, 0.04), positions=positions, velocities=velocities
        )
        task.front_feet_indices = torch.tensor([1, 2])
        task._update_support_frame()
        return task._compute_reduced_order_state()

    for name, upright, turned in zip(
        ("offset", "height", "velocity", "momentum"),
        state_at(0.0),
        state_at(1.0),
    ):
        torch.testing.assert_close(upright, turned, rtol=1e-5, atol=1e-6, msg=name)


def test_support_frame_is_right_handed_with_x_along_the_nose():
    """FL is index 0 of the canonical foot order, so FL - FR points at +y."""
    feet = torch.tensor([[[0.0, HALF_SPAN, 0.022], [0.0, -HALF_SPAN, 0.022]]])
    task = _task(masses=(0.04, 0.04), positions=feet)
    task.front_feet_indices = torch.tensor([0, 1])
    task._update_support_frame()

    torch.testing.assert_close(task._support_anchor, torch.tensor([[0.0, 0.0, 0.022]]))
    # Rows are x then y: x along the nose, y along the contact line.
    torch.testing.assert_close(
        task._support_rot, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    )


def test_support_anchor_tracks_the_feet_rather_than_staying_where_they_started():
    """The pivot is live. A pinned pivot would report an offset for a robot
    that is balanced over its real support, and the only way to null that
    offset is to lean off the feet."""
    task = _task(masses=(0.04, 0.04), positions=torch.zeros(1, 2, 3))
    task.front_feet_indices = torch.tensor([0, 1])
    task._rigid_body_pos[:] = torch.tensor(
        [[[0.0, HALF_SPAN, 0.022], [0.0, -HALF_SPAN, 0.022]]]
    )
    task._update_support_frame()
    torch.testing.assert_close(task._support_anchor[0, :2], torch.zeros(2))

    # Both feet slide 7 cm forward; the pivot must follow them.
    task._rigid_body_pos[:, :, 0] += 0.07
    task._update_support_frame()
    assert task._support_anchor[0, 0].item() == pytest.approx(0.07)


def test_degenerate_foot_separation_does_not_produce_nan():
    """Crossed or coincident feet must not poison the observation vector.

    The hip range is +/-1.047 rad, which is enough abduction to bring the feet
    together; a nan here would reach actor.obs and trip the finite-observation
    assertion in the task-registry smoke test non-deterministically.
    """
    feet = torch.zeros(1, 2, 3)
    task = _task(masses=(0.04, 0.04), positions=feet)
    task.front_feet_indices = torch.tensor([0, 1])
    task._update_support_frame()

    assert torch.isfinite(task._support_rot).all()


def _gravity_at_lean(radians, axis=2):
    """projected_gravity for a trunk leaned ``radians`` from the handstand.

    axis 2 is a pitch about the contact line (over the toes), axis 1 a roll
    along it. At zero this is the target (1, 0, 0), and the chord from the
    target is 2 sin(radians / 2) either way.
    """
    vector = [math.cos(radians), 0.0, 0.0]
    vector[axis] = math.sin(radians)
    return tuple(vector)


def _reward_task(
    offset=(0.0, 0.0),
    height=HEIGHT,
    momentum=(0.0, 0.0, 0.0),
    drift=0.0,
    com_vel=(0.0, 0.0, 0.0),
    gravity=None,
):
    task = Go2Stand.__new__(Go2Stand)
    task.cfg = Go2StandCfg()
    task.num_envs = 1
    task.device = "cpu"
    target = Go2StandCfg.reward_settings.target_projected_gravity
    task._target_projected_gravity = torch.tensor([target])
    task.projected_gravity = torch.tensor([list(gravity if gravity else target)])
    task.com_offset = torch.tensor([list(offset)])
    task.com_height = torch.tensor([[height]])
    task.com_vel = torch.tensor([list(com_vel)])
    task.support_momentum = torch.tensor([list(momentum)])
    # One foot sitting still, the other displaced by ``drift``.
    task.front_feet_indices = torch.tensor([0, 1])
    task._front_feet_anchor = torch.zeros(1, 2, 2)
    task._rigid_body_pos = torch.zeros(1, 2, 3)
    task._rigid_body_pos[0, 1, 0] = drift
    return _tolerances(task)


def test_every_reward_peaks_at_exactly_one_at_the_goal():
    task = _reward_task()
    for reward in (
        task._reward_balance_momentum,
        task._reward_balance_offset,
        task._reward_com_height,
        task._reward_com_vel_xy,
        task._reward_front_feet_drift,
        task._reward_balance_yaw,
        task._reward_balance_attitude,
        task._reward_balance_alive,
    ):
        assert reward().item() == pytest.approx(1.0)
        assert reward().shape == (1,)


def test_vertical_com_velocity_is_free_but_horizontal_is_not():
    """The reduced-order model's own split. Raising and lowering the centre of
    mass is the control authority the paper studies, so it must not be priced;
    horizontal travel means the robot is relocating rather than balancing."""
    speed = Go2StandCfg.reward_settings.com_vel_tolerance

    rising = _reward_task(com_vel=(0.0, 0.0, 4.0 * speed))
    assert rising._reward_com_vel_xy().item() == pytest.approx(1.0)

    for direction in ((speed, 0.0, 0.0), (0.0, speed, 0.0)):
        moving = _reward_task(com_vel=direction)._reward_com_vel_xy()
        # Isotropic in the horizontal plane: stepping is equally wrong in any
        # direction, unlike the offset and momentum terms.
        assert moving.item() == pytest.approx(
            math.exp(-4.0) - 0.2 * math.log1p(1.0), rel=1e-5
        )


def test_tolerances_are_anisotropic_and_not_transposed():
    """The axis roles are OPPOSITE for displacement and rotation.

    A displacement along x leaves the contact line, where there is no support
    polygon -- dangerous. A rotation about x is roll, which the 0.284 m foot
    separation arrests -- benign. Transposing the two tolerance vectors would
    invert both judgements while leaving every magnitude test passing.
    """
    magnitude = 0.05

    off_x = _reward_task(offset=(magnitude, 0.0))._reward_balance_offset()
    off_y = _reward_task(offset=(0.0, magnitude))._reward_balance_offset()
    assert off_x.item() < off_y.item(), "offset across the contact line must cost more"

    mom_x = _reward_task(momentum=(magnitude, 0.0, 0.0))._reward_balance_momentum()
    mom_y = _reward_task(momentum=(0.0, magnitude, 0.0))._reward_balance_momentum()
    assert mom_y.item() < mom_x.item(), "momentum about the contact line must cost more"


def test_foot_drift_reports_the_worst_foot_not_the_average():
    """A pivot about one planted foot sweeps the other through twice the
    distance a mean would report, halving the cost of the exact behaviour
    this term exists to stop."""
    one_foot_moved = _reward_task(drift=0.06)._reward_front_feet_drift()
    both_still = _reward_task(drift=0.0)._reward_front_feet_drift()
    assert one_foot_moved.item() < both_still.item()

    # The mean would be 0.03 and score distinctly better than the max of 0.06.
    tolerance = Go2StandCfg.reward_settings.foot_drift_tolerance
    error = 0.06 / tolerance
    expected = math.exp(-4 * error**2) - 0.2 * math.log1p(error)
    assert one_foot_moved.item() == pytest.approx(expected, rel=1e-5)


def test_front_feet_support_reaches_one_only_when_both_feet_carry_their_share():
    task = Go2Stand.__new__(Go2Stand)
    task.front_feet_indices = torch.tensor([0, 1])
    task._backend = SimpleNamespace(link_mass=torch.tensor([[8.0, 8.0]]))
    # Half the body weight on each foot is a fully loaded support pair.
    half_weight = 0.5 * 9.81 * 16.0
    task.contact_forces = torch.zeros(1, 2, 3)
    task.contact_forces[0, :, 2] = half_weight
    assert task._reward_front_feet_support().item() == pytest.approx(1.0)

    # One foot carrying everything while the other hops is a full failure,
    # not the 0.5 a mean would report.
    task.contact_forces[0, 1, 2] = 0.0
    assert task._reward_front_feet_support().item() == pytest.approx(0.0)


@pytest.mark.parametrize("magnitude", [0.02, 0.05, 0.2])
def test_offset_and_momentum_rewards_are_sign_symmetric(magnitude):
    # rel=1e-5 rather than the default: exp amplifies a float32 input
    # difference by about 6x here, which is enough to break an exact compare.
    positive = _reward_task(offset=(magnitude, 0.0))._reward_balance_offset()
    negative = _reward_task(offset=(-magnitude, 0.0))._reward_balance_offset()
    assert positive.item() == pytest.approx(negative.item(), rel=1e-5)

    up = _reward_task(momentum=(0.0, magnitude, 0.0))._reward_balance_momentum()
    down = _reward_task(momentum=(0.0, -magnitude, 0.0))._reward_balance_momentum()
    assert up.item() == pytest.approx(down.item(), rel=1e-5)


def test_height_reward_is_two_sided():
    """Overshoot is as bad as undershoot -- the pose is already 0.5 mm from
    its kinematic ceiling, so there is nothing above the target to reach for."""
    short = _reward_task(height=HEIGHT - 0.05)._reward_com_height()
    tall = _reward_task(height=HEIGHT + 0.05)._reward_com_height()
    assert short.item() == pytest.approx(tall.item(), rel=1e-5)
    assert short.item() < 1.0


def test_yaw_is_costed_by_its_own_term_and_cannot_swamp_the_balance_signal():
    """Yaw is costed, and by a separate term rather than a third component of
    balance_momentum's norm.

    The separation is the point: a norm's gradient splits as v_i/||v||, and
    yaw momentum is excited so much more easily than pitch (I_zz = 0.136 vs
    I_yy = 6.48 kg m^2) that it reaches 97% of a combined norm within 0.1 s of
    exploration, leaving the balance signal with almost none of its own.
    """
    spinning = _reward_task(momentum=(0.0, 0.0, 0.01))
    assert spinning._reward_balance_yaw().item() < 1.0, "yaw must not be free"

    # ...and it must leave balance_momentum completely alone.
    assert spinning._reward_balance_momentum().item() == pytest.approx(1.0)

    # Symmetric: spinning either way costs the same.
    widdershins = _reward_task(momentum=(0.0, 0.0, -0.01))._reward_balance_yaw()
    assert spinning._reward_balance_yaw().item() == pytest.approx(
        widdershins.item(), rel=1e-5
    )


def test_yaw_tolerance_bites_at_the_rate_the_policy_actually_turns():
    """Size the tolerance against the operating range, not against noise.

    An earlier value was chosen to stay off the clamp under untrained random
    actions. That put its knee at a yaw rate of 1.24 rad/s while a trained
    policy wobbles at roughly 0.15 -- so the term scored near its ceiling
    while the robot visibly rotated, and its weight was a dead knob.

    Measured |L_z|/N from a trained rollout, which is what these anchor to.
    """
    median, p90, p99 = 0.00026, 0.00082, 0.00153

    scores = [
        _reward_task(momentum=(0.0, 0.0, v))._reward_balance_yaw().item()
        for v in (median, p90, p99)
    ]
    assert scores[0] > 0.85, "a quiet robot must not be punished"
    assert 0.3 < scores[1] < 0.75, "the typical wobble must cost something real"
    assert scores[2] < 0.3, "the tail must be clearly bad"

    # A tolerance this tight means early training runs far out on the tail --
    # 0.0129 is e = 6.45 -- which is exactly where a clamp at e = 3 would hand
    # the policy a constant. On the shared logarithmic tail it still sees a
    # slope, so the term can teach from the noisy regime rather than only
    # scoring it.
    tolerance = Go2StandCfg.reward_settings.yaw_momentum_tolerance
    error = 0.0129 / tolerance
    noisy = _reward_task(momentum=(0.0, 0.0, 0.0129))._reward_balance_yaw()
    assert noisy.item() == pytest.approx(
        math.exp(-4 * error**2) - 0.2 * math.log1p(error), abs=1e-6
    )
    worse = _reward_task(momentum=(0.0, 0.0, 0.0258))._reward_balance_yaw()
    assert worse.item() < noisy.item() - 0.1, "the tail must still push at e = 6"


def test_drift_gradient_survives_many_tolerances_of_displacement():
    """Drift must never flatten into a constant.

    It accumulates monotonically against a reset-planted anchor, so it reaches
    any clamp within about half a second and stays there for the rest of the
    episode. A constant is invisible to PPO -- GAE cancels it and advantages
    are variance-normalized -- so a clamped term's weight would stop meaning
    anything. Hence the logarithmic tail; against a clamp at e = 3 every
    assertion below is an equality.
    """
    scores = [
        _reward_task(drift=d)._reward_front_feet_drift().item() for d in (0.3, 0.5, 1.0)
    ]
    assert scores[0] > scores[1] > scores[2], "gradient must survive past 3 tolerances"

    # And the push is still real, not a rounding artefact.
    assert scores[0] - scores[1] > 0.05
    assert scores[1] - scores[2] > 0.1


def test_dense_rewards_are_bounded_below_so_falling_beats_nothing():
    """Every tail is bounded, so riding a fall down is never worth less than
    terminating.

    A terminating step costs only -weight * dt = -1e-4, so an unbounded tail
    would make finishing the fall the best available action -- and the only
    gradient an untrained policy has.
    """
    far = _reward_task(
        offset=(100.0, 0.0),
        height=-100.0,
        momentum=(100.0, 0.0, 100.0),
        drift=100.0,
        com_vel=(100.0, 0.0, 0.0),
        gravity=(-1.0, 0.0, 0.0),  # fully inverted: the largest chord there is
    )
    # One shaping function, so one floor: the e = 40 guard of the logarithmic
    # tail, reached here by every unbounded term at once.
    floor = math.exp(-4 * 40.0**2) - 0.2 * math.log1p(40.0)
    unbounded = (
        "balance_momentum",
        "balance_offset",
        "com_height",
        "com_vel_xy",
        "balance_yaw",
        "front_feet_drift",
    )
    for name in unbounded:
        assert getattr(far, f"_reward_{name}")().item() == pytest.approx(
            floor, abs=1e-6
        )

    # balance_attitude is the exception and has its OWN floor: its argument is
    # a chord on the unit sphere, so it cannot exceed 2.0 however badly the
    # robot is oriented, and it bottoms well above the e = 40 guard.
    worst_chord = 2.0 / Go2StandCfg.reward_settings.attitude_tolerance
    attitude_floor = math.exp(-4 * worst_chord**2) - 0.2 * math.log1p(worst_chord)
    assert far._reward_balance_attitude().item() == pytest.approx(
        attitude_floor, abs=1e-6
    )
    assert attitude_floor > floor, "the chord bound must be the binding one"

    # The alive bonus must outweigh every floor at once, or the per-step
    # total goes negative and terminating becomes the cheaper option.
    # front_feet_support is a smoothstep in [0, 1] and cannot go negative, so
    # it does not enter. Summed from the config rather than hardcoded, so
    # adding a term or retuning a weight fails here rather than in a GPU run.
    weights = Go2StandRunnerCfg().critic.reward.weights
    worst = (
        floor
        * sum(getattr(weights, name) for name in unbounded if hasattr(weights, name))
        + attitude_floor * weights.balance_attitude
    )
    assert weights.balance_alive + worst > 0.0
    # com_height is off: an absent key, not a zero, because this weights class
    # does not inherit. If it is ever re-enabled it joins the sum above.
    assert not hasattr(weights, "com_height")


def _slip_task(linear, angular, quaternion=(0.0, 0.0, 0.0, 1.0), load=1.0):
    """A Go2Stand carrying just the two front feet, moving as given.

    ``load`` is each foot's share of body weight: 1.0 is a fully planted
    support pair, 0.0 is a foot in the air.
    """
    task = Go2Stand.__new__(Go2Stand)
    task.cfg = Go2StandCfg()
    task.num_envs = 1
    task.device = "cpu"
    task.front_feet_indices = torch.tensor([0, 1])
    task.foot_slip = torch.zeros(1, 2)
    task._foot_sphere_offset = torch.tensor(Go2StandCfg.asset.foot_sphere_offset)
    task._foot_contact_drop = torch.tensor([0.0, 0.0, -Go2StandCfg.asset.foot_radius])
    task._rigid_body_quat = torch.tensor([[list(quaternion)] * 2])
    task._rigid_body_lin_vel = torch.tensor([[list(linear)] * 2])
    task._rigid_body_ang_vel = torch.tensor([[list(angular)] * 2])
    # _foot_load normalizes by share * g * total mass, so with share=0.5 a
    # foot carrying half of 16 kg reads a full 1.0.
    task._backend = SimpleNamespace(link_mass=torch.tensor([[8.0, 8.0]]))
    task.contact_forces = torch.zeros(1, 2, 3)
    task.contact_forces[0, :, 2] = load * 0.5 * 9.81 * 16.0
    return task


def test_pure_rolling_reads_as_no_slip():
    """The reason slip is measured at the contact point and not at the body
    origin. A 22 mm sphere rolling without sliding still moves its origin at
    radius x omega, which an origin-velocity measure would report as slip."""
    rate = 3.0
    radius = Go2StandCfg.asset.foot_radius
    # omega about +y with the contact below gives the contact point
    # -rate*radius in x, so an origin moving at +rate*radius rolls exactly.
    task = _slip_task(linear=(rate * radius, 0.0, 0.0), angular=(0.0, rate, 0.0))
    task._update_foot_slip()

    origin_speed = torch.linalg.norm(task._rigid_body_lin_vel[0, 0, :2]).item()
    assert origin_speed == pytest.approx(rate * radius, rel=1e-5)
    assert origin_speed > 0.06, "the test is vacuous if the origin is not moving"
    torch.testing.assert_close(task.foot_slip, torch.zeros(1, 2), atol=2e-4, rtol=0)


def test_a_sliding_foot_reports_its_own_speed():
    task = _slip_task(linear=(0.12, -0.05, 0.0), angular=(0.0, 0.0, 0.0))
    task._update_foot_slip()
    expected = math.hypot(0.12, 0.05)
    torch.testing.assert_close(
        task.foot_slip, torch.full((1, 2), expected), rtol=1e-5, atol=0
    )


def test_vertical_foot_motion_is_not_slip():
    """Lifting a foot is not sliding; only the horizontal component counts."""
    task = _slip_task(linear=(0.0, 0.0, 0.8), angular=(0.0, 0.0, 0.0))
    task._update_foot_slip()
    torch.testing.assert_close(task.foot_slip, torch.zeros(1, 2), atol=1e-6, rtol=0)


def test_slip_is_clipped_so_a_crash_cannot_spike_the_observation():
    """Observations are not normalized here, so an unbounded value reaches the
    network raw. A crashing robot hits 7.6 m/s against a 0.58 m/s p99."""
    task = _slip_task(linear=(50.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    task._update_foot_slip()
    cap = Go2StandCfg.asset.foot_slip_max
    torch.testing.assert_close(task.foot_slip, torch.full((1, 2), cap), rtol=0, atol=0)


def test_an_airborne_foot_reports_no_slip():
    """The gate, and the reason it exists.

    A foot in the air has no contact point, but the contact-point formula
    still evaluates one and a swinging foot produces a large value. Ungated,
    the observation saturated its clip for 1% of samples at 34x the true
    loaded mean -- telling the policy the floor was giving way every time it
    picked a foot up.
    """
    airborne = _slip_task(linear=(0.4, 0.0, 0.0), angular=(0.0, 0.0, 0.0), load=0.0)
    airborne._update_foot_slip()
    torch.testing.assert_close(airborne.foot_slip, torch.zeros(1, 2), atol=1e-7, rtol=0)

    # The same motion with the foot planted is real slip.
    planted = _slip_task(linear=(0.4, 0.0, 0.0), angular=(0.0, 0.0, 0.0), load=1.0)
    planted._update_foot_slip()
    assert planted.foot_slip[0, 0].item() == pytest.approx(0.4, rel=1e-5)


def test_slip_scales_smoothly_with_how_much_weight_the_foot_carries():
    """A smoothstep gate, so nothing jumps at lift-off -- a hard contact
    threshold would put a discontinuity in an actor observation."""
    speeds = [
        _slip_task(linear=(0.4, 0.0, 0.0), angular=(0.0, 0.0, 0.0), load=f)
        for f in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    for task in speeds:
        task._update_foot_slip()
    values = [t.foot_slip[0, 0].item() for t in speeds]
    assert values == sorted(values), "gate must be monotone in load"
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(0.4, rel=1e-5)
    # Strictly between the ends, not a disguised step function.
    assert 0.0 < values[2] < 0.4


# ------------ attitude ----------------


def test_only_the_attitude_term_can_tell_a_handstand_from_a_lean():
    """The regression test for the bug this term exists to fix.

    A robot standing MOTIONLESS at 15 degrees of lean, with its mass arranged
    so the centre of mass sits over the contact line, is indistinguishable
    from a handstand to every reduced-order term: all the velocities are zero
    so the momentum vanishes, and the offset vanishes by construction. A
    policy trained 5000 iterations without an attitude term found exactly
    that pose and held it 99.2% of the time.

    (front_feet_support is not exercised here -- it needs contact forces the
    fixture does not build -- but it too is at ceiling for a planted robot.)
    """
    lean = _reward_task(gravity=_gravity_at_lean(math.radians(15.0)))

    for blind in (
        lean._reward_balance_momentum,
        lean._reward_balance_offset,
        lean._reward_com_vel_xy,
        lean._reward_balance_yaw,
        lean._reward_front_feet_drift,
    ):
        assert blind().item() == pytest.approx(1.0), (
            "a reduced-order term noticed a lean it cannot see"
        )

    assert lean._reward_balance_attitude().item() < 0.0, (
        "the attitude term must price the lean every other term misses"
    )

    # And the weighted totals must now differ, which is the whole point.
    weights = Go2StandRunnerCfg().critic.reward.weights
    upright = _reward_task()
    assert (
        weights.balance_attitude * lean._reward_balance_attitude().item()
        < weights.balance_attitude * upright._reward_balance_attitude().item()
    )


def test_attitude_is_blind_to_rotation_about_the_gravity_axis():
    """Spinning about the trunk's own long axis leaves projected gravity at
    the target, so this term must read exactly 1.0 -- yaw is balance_yaw's
    job and the two must not double-charge the same mode."""
    spun = _reward_task(gravity=(1.0, 0.0, 0.0))
    assert spun._reward_balance_attitude().item() == pytest.approx(1.0)


def test_attitude_is_sign_symmetric_and_isotropic():
    """Over the toes and over the heels cost the same, and so does the same
    tilt sideways: the term is a pose measure, not a balance measure, and a
    handstand leaning left is as wrong as one leaning forward."""
    angle = math.radians(8.0)
    scores = [
        _reward_task(gravity=_gravity_at_lean(s * angle, axis=a))
        ._reward_balance_attitude()
        .item()
        for a in (1, 2)
        for s in (1.0, -1.0)
    ]
    assert max(scores) - min(scores) < 1e-6


def test_attitude_does_not_constrain_centre_of_mass_height():
    """Height is the control authority the reduced-order model is about, which
    is why com_height is unweighted. This term is a function of projected
    gravity alone, so it must be flat in height."""
    scores = {
        height: _reward_task(height=height)._reward_balance_attitude().item()
        for height in (0.30, 0.45, HEIGHT, 0.62)
    }
    assert max(scores.values()) - min(scores.values()) < 1e-9


def test_attitude_tolerance_bites_at_the_lean_the_policy_actually_held():
    """Calibration, anchored on measurement rather than taste.

    The policy trained without this term held a median 15.1 degrees of lean
    and was within 5 degrees of vertical 0.8% of the time. The tolerance is
    sized so that a tight hold is near ceiling, the measured lean is clearly
    negative, and the band between them carries real gradient.
    """
    score = lambda d: (  # noqa: E731
        _reward_task(gravity=_gravity_at_lean(math.radians(d)))
        ._reward_balance_attitude()
        .item()
    )

    assert score(1.0) > 0.75, "a tight handstand must sit near ceiling"
    assert 0.3 < score(2.0) < 0.6, "the near field must carry the precision"
    assert score(15.1) < -0.2, "the measured lean must be clearly bad"
    # Monotone, and still pushing at the lean it has to escape from.
    assert score(5.0) > score(10.0) > score(15.1) > score(25.0)
    assert score(10.0) - score(20.0) > 0.05, "the tail must still pull it in"
