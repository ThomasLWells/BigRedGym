"""Energy manifolds for the handstand curriculum, without a simulator.

The constants come from the task config, so these tests fail if a measured
value is edited without re-measuring.
"""

import math

import pytest
import torch

from gym.envs.go2.go2stand import Go2Stand
from gym.envs.go2.go2stand_config import Go2StandCfg
from gym.envs.go2.go2stand_curriculum import GRAVITY, HandstandPendulum

# The real schedule, not a stand-in, so every parametrized test below runs on
# the stages training will actually see.
STAGES = HandstandPendulum.from_config(Go2StandCfg()).schedule(
    Go2StandCfg.curriculum.start_energy,
    Go2StandCfg.curriculum.ratio,
    Go2StandCfg.curriculum.max_lean_step,
)


@pytest.fixture
def pendulum():
    return HandstandPendulum.from_config(Go2StandCfg())


def _curriculum_task(num_envs=100):
    """Go2Stand with only the state ``_score_curriculum`` reads.

    No simulator: the gate is bookkeeping over the two termination flags, and
    that is exactly the part worth testing in isolation.
    """
    task = Go2Stand.__new__(Go2Stand)
    task.cfg = Go2StandCfg()
    task.device = "cpu"
    task.num_envs = num_envs
    task._pendulum = HandstandPendulum.from_config(task.cfg)
    task._curriculum_schedule = STAGES
    task.curriculum_stage = 0
    task.curriculum_success_rate = 0.0
    task._curriculum_episodes = 0.0
    task._apply_curriculum_stage()
    task.timed_out = torch.zeros(num_envs, dtype=torch.bool)
    task.terminated = torch.zeros(num_envs, dtype=torch.bool)
    task.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    return task


def _finish_episodes(task, episodes, success_fraction, also_terminate=False):
    """Retire whole batches of episodes at a fixed success rate.

    ``also_terminate`` sets BOTH flags on the successes, the case a naive gate
    scores wrongly.
    """
    mask = torch.ones(task.num_envs, dtype=torch.bool)
    wins = int(round(success_fraction * task.num_envs))
    for _ in range(episodes // task.num_envs):
        task.episode_length_buf.fill_(100)
        task.timed_out[:] = torch.arange(task.num_envs) < wins
        task.terminated[:] = (
            torch.ones(task.num_envs, dtype=torch.bool)
            if also_terminate
            else torch.arange(task.num_envs) >= wins
        )
        task._score_curriculum(mask)


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


# ------------ the schedule ----------------


def test_schedule_steps_are_bounded_in_both_energy_and_lean(pendulum):
    """The two growth laws, each doing the job the other cannot.

    Geometric growth alone turns into 6-degree jumps at the top, where energy
    is no longer quadratic in lean; a fixed lean step alone would make the
    early stages absurdly fine, since 2.5 degrees from upright is only 0.09 J.
    Together they bound both, which is the whole claim of the schedule.
    """
    assert STAGES[0] == pytest.approx(Go2StandCfg.curriculum.start_energy)
    assert list(STAGES) == sorted(STAGES), "a curriculum must grow"

    leans = [pendulum.max_lean_at_rest(e) for e in STAGES]
    energy_steps = [b - a for a, b in zip(STAGES, STAGES[1:])]
    lean_steps = [b - a for a, b in zip(leans, leans[1:])]

    ratio = Go2StandCfg.curriculum.ratio
    step = Go2StandCfg.curriculum.max_lean_step
    assert max(lean_steps) <= step + 1e-9, "the lean cap must bind at the top"
    for before, jump in zip(STAGES, energy_steps):
        assert jump <= (ratio - 1.0) * before + 1e-9, "geometric bound must hold"

    # And the early stages are where the cap does NOT bind, or the geometric
    # law is doing nothing.
    assert lean_steps[0] < 0.5 * step

    # Ends at the attitude termination rather than at a chosen count: one more
    # stage would lean past it and terminate on its first step.
    assert pendulum.max_lean_at_rest(ratio * STAGES[-1]) > pendulum.lean_cap


def test_schedule_refuses_settings_that_cannot_terminate(pendulum):
    """A ratio of 1.0 with no lean headroom is an infinite loop, and it would
    hang at env construction rather than fail a test."""
    with pytest.raises(ValueError, match="ratio"):
        pendulum.schedule(0.05, 1.0, 0.04)
    with pytest.raises(ValueError, match="max_lean_step"):
        pendulum.schedule(0.05, 1.5, 0.0)
    with pytest.raises(ValueError, match="outside"):
        pendulum.schedule(50.0, 1.5, 0.04)


@pytest.mark.parametrize("epsilon", STAGES)
def test_every_spawn_state_has_reward_gradient(pendulum, epsilon):
    """The defect the whole shaping change exists to fix.

    The curriculum spawns the robot at its stage's lean cap, so the shaped
    terms must still have a slope *there* -- otherwise the robot starts each
    episode in a flat region and PPO sees nothing to follow. Against a clamp
    at e = 3 the two values below are equal from the 19-degree stage on, and
    this assertion is what catches that.
    """
    task = Go2Stand.__new__(Go2Stand)
    task.cfg = Go2StandCfg()
    task.device = "cpu"

    cap = min(pendulum.max_lean_at_rest(epsilon), pendulum.lean_cap)
    # The offset a spawn at the cap produces, in tolerance units: the capture
    # point is r sin(theta) from the contact line, across it (the tight axis).
    offset = pendulum.length * math.sin(cap)
    error = torch.tensor(
        [offset / Go2StandCfg.reward_settings.support_offset_tolerance[0]]
    )

    here = task._shaped_log_tail(error)
    twice_as_bad = task._shaped_log_tail(2.0 * error)
    assert here.item() > twice_as_bad.item(), "spawn sits in a flat region"
    # Not merely unequal -- the push has to be worth something next to the
    # 7.0 alive bonus the policy is already collecting.
    assert here.item() - twice_as_bad.item() > 0.01


@pytest.mark.parametrize("epsilon", STAGES)
def test_root_state_round_trips_back_into_the_same_energy_layer(pendulum, epsilon):
    """Decode the written root_states the way the env's own state does.

    The sampler works in (theta, theta_dot) but what reaches the simulator is
    a quaternion and a world angular velocity. If that encoding were wrong --
    a half-angle, a transposed axis, a scalar-first quaternion -- the spawned
    states would silently leave S(epsilon) and the stage label would be a lie.
    """
    generator = torch.Generator().manual_seed(0)
    theta, theta_dot = pendulum.sample(epsilon, 4000, generator=generator)
    state = pendulum.to_root_state(theta, theta_dot)

    # Scalar-last, pitch-only: theta = 2 atan2(q_y, q_w) - pi/2.
    decoded = 2.0 * torch.atan2(state[:, 4], state[:, 6]) - pendulum.base_pitch
    decoded_rate = state[:, 11]
    torch.testing.assert_close(decoded, theta, atol=1e-5, rtol=0)
    torch.testing.assert_close(decoded_rate, theta_dot, atol=1e-6, rtol=0)

    deficit = pendulum.energy_deficit(decoded, decoded_rate).abs()
    assert deficit.max() <= epsilon + 1e-3


# ------------ stage advancement ----------------


def test_a_fall_on_the_horizon_step_is_not_a_success():
    """timed_out and terminated are NOT mutually exclusive -- nothing clears
    terminated when the horizon fires. A gate reading the timeout alone scores
    every such fall as a success, and would walk the curriculum forward on a
    robot that is falling over."""
    task = _curriculum_task()
    _finish_episodes(task, 4000, success_fraction=1.0, also_terminate=True)

    assert task.curriculum_success_rate == pytest.approx(0.0)
    assert task.curriculum_stage == 0


def test_advancement_needs_both_a_good_rate_and_enough_episodes():
    """Derived from the config, not hardcoded: these constants are sized in
    episodes but spent in iterations, and the conversion has already been
    wrong once."""
    floor = Go2StandCfg.curriculum.min_episodes

    # Batches well under the floor, so the count can be probed either side.
    perfect = _curriculum_task(num_envs=25)
    total = int(floor) // 2
    _finish_episodes(perfect, total, success_fraction=1.0)
    assert perfect.curriculum_stage == 0, "advanced before the episode floor"

    while perfect.curriculum_stage == 0 and total < 20 * floor:
        _finish_episodes(perfect, perfect.num_envs, success_fraction=1.0)
        total += perfect.num_envs
    assert perfect.curriculum_stage == 1, "a perfect policy never cleared stage 0"
    assert total >= floor, "cleared the stage on fewer than min_episodes"

    # A policy that holds half its episodes never clears the stage, however
    # long it runs -- the rate gate, not the episode floor.
    mediocre = _curriculum_task()
    _finish_episodes(mediocre, 20 * int(floor), 0.5)
    assert mediocre.curriculum_stage == 0
    assert mediocre.curriculum_success_rate == pytest.approx(0.5, abs=0.05)


def test_stage_advances_one_at_a_time_and_stops_at_the_last():
    """Monotone, single-step, and clamped: running off the end of the schedule
    would be an IndexError inside a reset, mid-training."""
    task = _curriculum_task()
    trace = [task.curriculum_stage]
    for _ in range(400):
        _finish_episodes(task, task.num_envs, success_fraction=1.0)
        trace.append(task.curriculum_stage)

    assert trace == sorted(trace), "the stage must never go backwards"
    assert max(b - a for a, b in zip(trace, trace[1:])) <= 1
    assert task.curriculum_stage == len(STAGES) - 1, "never reached the top"
    assert task._curriculum_epsilon == STAGES[-1]

    # The termination threshold follows the stage, and only ever upward.
    assert task._max_attitude_error > Go2StandCfg.termination.max_attitude_error
    task.curriculum_stage = 0
    task._apply_curriculum_stage()
    assert task._max_attitude_error == Go2StandCfg.termination.max_attitude_error
