"""Energy manifolds for the handstand curriculum, without a simulator.

The constants come from the task config, so these tests fail if a measured
value is edited without re-measuring.
"""

import math

import pytest
import torch

from gym.envs.go2.go2stand_config import Go2StandCfg
from gym.envs.go2.go2stand_curriculum import GRAVITY, HandstandPendulum

STAGES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@pytest.fixture
def pendulum():
    return HandstandPendulum.from_config(Go2StandCfg())


def test_goal_state_has_no_energy_deficit(pendulum):
    zero = torch.zeros(1)
    assert pendulum.energy_deficit(zero, zero).item() == pytest.approx(0.0, abs=1e-6)
    assert pendulum.goal_energy == pytest.approx(
        pendulum.mass * GRAVITY * pendulum.length
    )


def test_energy_is_conserved_along_a_free_pendulum_rollout(pendulum):
    """The test that catches a wrong inertia, a wrong length, or a sign error.

    Energy is conserved under the uncontrolled dynamics, which is the whole
    reason it can label a reachability layer. If the kinetic term used m r^2
    instead of the true inertia about the contact line, the drift here would be
    9% rather than numerical noise.
    """
    # theta_ddot = (m g r / I) sin(theta): the INVERTED pendulum, so the
    # gravity torque drives the lean away from upright.
    rate = pendulum.mass * GRAVITY * pendulum.length / pendulum.inertia

    def derivative(state):
        return torch.stack((state[1], rate * torch.sin(state[0])))

    for theta0, theta_dot0 in ((0.05, 0.0), (0.3, -1.0), (-0.2, 0.6), (0.5, 1.5)):
        state = torch.tensor([theta0, theta_dot0], dtype=torch.double)
        start = pendulum.energy(state[0], state[1])
        dt = 1e-4
        for _ in range(5000):  # half a second
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * dt * k1)
            k3 = derivative(state + 0.5 * dt * k2)
            k4 = derivative(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        drift = abs(pendulum.energy(state[0], state[1]).item() - start.item())
        assert drift < 1e-6, f"energy drifted {drift:.2e} J from {theta0}, {theta_dot0}"


def test_separatrix_is_the_zero_deficit_curve(pendulum):
    theta = torch.tensor([0.05, 0.2, 0.4, 0.6])
    deficit = pendulum.energy_deficit(theta, pendulum.separatrix_rate(theta))
    torch.testing.assert_close(deficit, torch.zeros_like(deficit), atol=1e-5, rtol=0)


@pytest.mark.parametrize("epsilon", STAGES)
def test_rate_band_endpoints_sit_exactly_on_the_budget(pendulum, epsilon):
    theta = torch.linspace(-0.6, 0.6, 25)
    lo, hi = pendulum.rate_band(theta, epsilon)
    # The upper edge always costs a full budget; the lower edge does too
    # wherever it has not been clipped at standstill.
    torch.testing.assert_close(
        pendulum.energy_deficit(theta, hi),
        torch.full_like(theta, -epsilon),
        atol=1e-4,
        rtol=0,
    )
    unclipped = lo > 0.0
    torch.testing.assert_close(
        pendulum.energy_deficit(theta[unclipped], lo[unclipped]),
        torch.full((int(unclipped.sum()),), epsilon),
        atol=1e-4,
        rtol=0,
    )


@pytest.mark.parametrize("epsilon", STAGES)
def test_samples_respect_the_budget_and_the_lean_cap(pendulum, epsilon):
    generator = torch.Generator().manual_seed(0)
    theta, theta_dot = pendulum.sample(epsilon, 20000, generator=generator)

    assert pendulum.energy_deficit(theta, theta_dot).abs().max() <= epsilon + 1e-4
    cap = min(pendulum.max_lean_at_rest(epsilon), pendulum.lean_cap)
    assert theta.abs().max() <= cap + 1e-6


def test_layers_are_nested_and_grow(pendulum):
    """Each stage must contain the one before it, or it is not a curriculum."""
    caps = [pendulum.max_lean_at_rest(e) for e in STAGES]
    assert caps == sorted(caps)
    assert caps[0] < caps[-1]
    # The final stage is the attitude termination, by construction.
    assert caps[-1] < pendulum.lean_cap
    assert caps[-1] > 0.9 * pendulum.lean_cap


def test_far_samples_must_swing_inward(pendulum):
    """The paper's eq. (5). Beyond the lean a budget can recover from rest,
    an outward swing never comes back whatever its energy says -- so the sign
    of the rate stops being free."""
    epsilon = 0.5
    generator = torch.Generator().manual_seed(0)
    theta, theta_dot = pendulum.sample(
        epsilon, 4000, generator=generator, lean_cap=pendulum.lean_cap
    )

    lo, _ = pendulum.rate_band(theta, epsilon)
    forced = lo > 0.0
    assert forced.any(), "lean cap too small to exercise the rule"
    assert (theta[forced] * theta_dot[forced] < 0.0).all()

    # Inside the recoverable lean the sign is free, so both appear.
    free = ~forced
    assert (theta_dot[free] > 0).any() and (theta_dot[free] < 0).any()


def test_root_state_at_the_goal_reproduces_the_nominal_pose(pendulum):
    state = pendulum.to_root_state(torch.zeros(1), torch.zeros(1))[0]
    torch.testing.assert_close(
        state[:3], torch.tensor(Go2StandCfg.init_state.pos), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        state[3:7], torch.tensor(Go2StandCfg.init_state.rot), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(state[7:], torch.zeros(6), atol=1e-7, rtol=0)


def test_root_state_is_a_rigid_rotation_about_the_contact_line(pendulum):
    theta = torch.tensor([-0.4, -0.1, 0.0, 0.25, 0.6])
    state = pendulum.to_root_state(theta, torch.zeros_like(theta))
    pivot = torch.tensor([pendulum.pivot_x, 0.0, 0.0])

    # A rotation preserves the distance from the pivot...
    radius = torch.linalg.norm(state[:, :3] - pivot, dim=1)
    torch.testing.assert_close(radius, radius[0].expand_as(radius), rtol=1e-5, atol=0)
    # ...and turns the body by exactly theta, about +y only. Columns 3:7 are
    # the quaternion [x, y, z, w], so a pure pitch leaves 3 and 5 at zero.
    torch.testing.assert_close(state[:, 3], torch.zeros_like(theta), atol=1e-7, rtol=0)
    torch.testing.assert_close(state[:, 5], torch.zeros_like(theta), atol=1e-7, rtol=0)
    pitch = 2.0 * torch.atan2(state[:, 4], state[:, 6])
    torch.testing.assert_close(pitch, 0.5 * math.pi + theta, atol=1e-5, rtol=0)


def test_positive_lean_carries_the_body_over_the_toes(pendulum):
    """Sign convention. Getting this backwards would make every stage start
    the robot leaning the wrong way, which no magnitude test would notice."""
    forward = pendulum.to_root_state(torch.tensor([0.3]), torch.zeros(1))[0]
    assert forward[0] > pendulum.pivot_x
    assert forward[2] < Go2StandCfg.init_state.pos[2]


def test_root_velocity_matches_the_rigid_rotation(pendulum):
    theta = torch.tensor([-0.3, 0.2, 0.5])
    theta_dot = torch.tensor([1.1, -0.7, 2.0])
    state = pendulum.to_root_state(theta, theta_dot)
    pivot = torch.tensor([pendulum.pivot_x, 0.0, 0.0])
    arm = state[:, :3] - pivot

    # Both blocks are world frame, so v = omega x arm holds directly.
    torch.testing.assert_close(state[:, 10], torch.zeros_like(theta), atol=1e-7, rtol=0)
    torch.testing.assert_close(state[:, 11], theta_dot, atol=1e-6, rtol=0)
    torch.testing.assert_close(state[:, 12], torch.zeros_like(theta), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        torch.linalg.norm(state[:, 7:10], dim=1),
        theta_dot.abs() * torch.linalg.norm(arm, dim=1),
        rtol=1e-5,
        atol=0,
    )
    # Velocity of a rotating body is perpendicular to its radius.
    dot = (state[:, 7:10] * arm).sum(dim=1)
    torch.testing.assert_close(dot, torch.zeros_like(dot), atol=1e-6, rtol=0)


def test_balance_margin_rejects_outward_swings_outright(pendulum):
    """The paper's eq. (5) is a precondition for eq. (7): a centre of mass
    moving away from the contact line cannot be balanced by height control at
    any margin."""
    outward = pendulum.balance_margin(torch.tensor([0.3]), torch.tensor([1.0]))
    assert outward.item() == -math.inf

    # Swinging inward fast enough is within the paper's region; barely moving
    # is not, because height control cannot arrest a near-static lean at all.
    assert pendulum.balance_margin(torch.tensor([0.3]), torch.tensor([-3.0])) > 0.0
    assert pendulum.balance_margin(torch.tensor([0.3]), torch.tensor([-0.05])) < 0.0
