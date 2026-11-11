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


def _reward_task(offset=(0.0, 0.0), height=HEIGHT, momentum=(0.0, 0.0, 0.0), drift=0.0):
    task = Go2Stand.__new__(Go2Stand)
    task.cfg = Go2StandCfg()
    task.num_envs = 1
    task.device = "cpu"
    task.com_offset = torch.tensor([list(offset)])
    task.com_height = torch.tensor([[height]])
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
        task._reward_front_feet_drift,
        task._reward_balance_alive,
    ):
        assert reward().item() == pytest.approx(1.0)
        assert reward().shape == (1,)


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
    expected = math.exp(-4 * (0.06 / tolerance) ** 2) - 0.2 * (0.06 / tolerance)
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


def test_yaw_momentum_is_deliberately_uncosted():
    """Gravity exerts no torque about the vertical, so yaw is outside the
    model. Documented as a known gap: a slowly spinning handstand scores
    full marks."""
    spinning = _reward_task(momentum=(0.0, 0.0, 5.0))
    assert spinning._reward_balance_momentum().item() == pytest.approx(1.0)


def test_dense_rewards_are_bounded_below_so_falling_beats_nothing():
    """The clamp that removes the suicide incentive.

    Unbounded, the linear tail makes riding a fall down cost more per step
    than terminating (-weight * dt = -1e-4), and an untrained policy's only
    gradient would point at falling faster.
    """
    far = _reward_task(
        offset=(100.0, 0.0), height=-100.0, momentum=(100.0, 0.0, 0.0), drift=100.0
    )
    floor = math.exp(-4 * 3.0**2) - 0.6
    for reward in (
        far._reward_balance_momentum,
        far._reward_balance_offset,
        far._reward_com_height,
        far._reward_front_feet_drift,
    ):
        assert reward().item() == pytest.approx(floor, abs=1e-6)

    # The alive bonus must outweigh every shaped floor at once, or the
    # per-step total goes negative and terminating becomes the cheaper
    # option. front_feet_support is a smoothstep in [0, 1] and cannot go
    # negative, so it does not enter.
    weights = Go2StandRunnerCfg().critic.reward.weights
    shaped = (
        weights.balance_momentum
        + weights.balance_offset
        + weights.com_height
        + weights.front_feet_drift
    )
    assert weights.balance_alive + floor * shaped > 0.0
