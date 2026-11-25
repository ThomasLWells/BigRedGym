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
    task.curriculum_stage_max = 0
    task.curriculum_window_rate = 0.0
    task.curriculum_reviews = 0
    task.curriculum_review_episodes = 0.0
    task._window_successes = 0.0
    task._window_episodes = 0.0
    task._curriculum_stage_step = 0
    task.common_step_counter = 0
    task.timed_out = torch.zeros(num_envs, dtype=torch.bool)
    task.terminated = torch.zeros(num_envs, dtype=torch.bool)
    task.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    task._curriculum_fresh = torch.zeros(num_envs, dtype=torch.bool)
    task._curriculum_reported = torch.zeros(num_envs, dtype=torch.bool)
    task._apply_curriculum_stage()
    return task


# Steps the real runner advances each env by per iteration: num_steps_per_env
# (rollout_size / num_envs = 8) times n_policy_steps (1 at a 100 Hz actor).
STEPS_PER_ITERATION = 8


def _finish_episodes(
    task, episodes, success_fraction, also_terminate=False, steps=STEPS_PER_ITERATION
):
    """Retire whole batches of episodes at a fixed success rate.

    ``steps`` advances the env's clock per batch, which the review's step floor
    reads; pass 0 to deliver episodes with no time passing at all, which is how
    the dwell is tested. ``also_terminate`` sets BOTH flags on the successes,
    the case a naive gate scores wrongly.
    """
    mask = torch.ones(task.num_envs, dtype=torch.bool)
    wins = int(round(success_fraction * task.num_envs))
    for _ in range(episodes // task.num_envs):
        task.common_step_counter += steps
        task.episode_length_buf.fill_(100)
        task.timed_out[:] = torch.arange(task.num_envs) < wins
        task.terminated[:] = (
            torch.ones(task.num_envs, dtype=torch.bool)
            if also_terminate
            else torch.arange(task.num_envs) >= wins
        )
        task._score_curriculum(mask)
        # The env marks each reset env as part of the open window, after
        # scoring; mirror that ordering here.
        task._curriculum_fresh |= mask


def _run_reviews(task, reviews, success_fraction, limit=5000):
    """Drive the task until ``reviews`` more reviews have fired.

    Tests say how many REVIEWS they want rather than how many episodes,
    because a review needs both floors satisfied and the episode count alone
    does not determine when one fires.
    """
    target = task.curriculum_reviews + reviews
    for _ in range(limit):
        if task.curriculum_reviews >= target:
            return
        _finish_episodes(task, task.num_envs, success_fraction)
    raise AssertionError(f"only {task.curriculum_reviews} reviews fired, want {target}")


def _run_realistic(task, iterations, success_rate, horizon=1000, fail_steps=50, seed=0):
    """Drive the gate with episodes whose LENGTH depends on their outcome.

    The simple driver above retires every env on the same step, which cannot
    show the censoring bias: in a real rollout a fall resolves in under a
    second and a hold takes the full ten, so a window counting episodes rather
    than envs over-samples failures by that ratio. Any test about what the
    rate *reads* has to use this one.
    """
    n = task.num_envs
    generator = torch.Generator().manual_seed(seed)
    phase = torch.randint(0, horizon, (n,), generator=generator)
    holds = torch.rand(n, generator=generator) < success_rate
    rates = []
    for _ in range(iterations):
        for _ in range(STEPS_PER_ITERATION):
            task.common_step_counter += 1
            phase += 1
            done = phase >= torch.where(holds, horizon, fail_steps)
            task.episode_length_buf = phase.clone()
            task.timed_out[:] = done & holds
            task.terminated[:] = done & ~holds
            seen = task.curriculum_reviews
            task._score_curriculum(done)
            if task.curriculum_reviews > seen:
                rates.append(task.curriculum_window_rate)
            task._curriculum_fresh |= done
            phase[done] = 0
            holds = torch.where(
                done, torch.rand(n, generator=generator) < success_rate, holds
            )
    return rates


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

    assert task.curriculum_window_rate == pytest.approx(0.0)
    assert task.curriculum_stage == 0


@pytest.mark.parametrize("true_rate", (0.9, 0.6, 0.3))
def test_the_window_rate_is_unbiased_when_failures_resolve_early(true_rate):
    """The estimator test, and the one that is easiest to get wrong.

    A fall ends an episode in under a second while a hold runs the full ten,
    so a window that counted EPISODES would let a failing env report twenty
    times for every once a succeeding one does. Measured against a true 0.9
    policy such a window reads 0.68, and 0.40 on the first review after a
    stage change -- which demotes a perfectly competent policy. Counting one
    trial per env removes the bias; this pins that it stays removed.
    """
    task = _curriculum_task(num_envs=400)
    rates = _run_realistic(task, 400, true_rate)

    assert rates, "no review fired"
    for rate in rates:
        assert rate == pytest.approx(true_rate, abs=0.08), (
            f"window read {rate:.3f} for a true rate of {true_rate}"
        )


def test_a_competent_policy_is_never_demoted():
    """The cascade regression. A policy holding 90% of its episodes sits above
    demote_rate, so a stage it can do must never be taken away from it -- the
    failure mode a censored window would produce on every single review."""
    task = _curriculum_task(num_envs=400)
    task.curriculum_stage = 6
    task._apply_curriculum_stage()

    _run_realistic(task, 600, success_rate=0.9)
    assert task.curriculum_stage >= 6, "demoted a policy that was holding 90%"


def test_promotion_cannot_outrun_the_step_floor():
    """Supplying unlimited episodes with the clock stopped must promote
    nothing: a window is a span of training, not a pile of samples."""
    task = _curriculum_task()
    _finish_episodes(task, 200 * task.num_envs, 1.0, steps=0)

    assert task.curriculum_reviews == 0, "reviewed with no time elapsed"
    assert task.curriculum_stage == 0


def test_each_env_contributes_one_trial_per_window():
    """The mechanism behind the unbiasedness: re-reporting envs are ignored
    until the window closes, so the sample is the population, once each."""
    task = _curriculum_task()
    for _ in range(4):
        _finish_episodes(task, task.num_envs, success_fraction=1.0)
        assert task._window_episodes <= task.num_envs, "an env reported twice"
    assert task._curriculum_reported.all(), "every env should have reported"

    _run_reviews(task, 1, success_fraction=1.0)
    assert task.curriculum_review_episodes <= task.num_envs
    assert task.curriculum_review_episodes >= (
        Go2StandCfg.curriculum.review_fraction * task.num_envs
    )
    # And the window reopens clean, or the next review inherits this one.
    assert not task._curriculum_reported.any()


def test_a_failing_stage_is_demoted_and_never_below_zero():
    """The point of the exercise. Without demotion the 2000-iteration run
    spent 1762 of its iterations above a frontier it could not hold."""
    task = _curriculum_task()
    task.curriculum_stage = 8
    task._apply_curriculum_stage()

    _run_reviews(task, 1, success_fraction=0.0)
    assert task.curriculum_stage < 8, "a collapsing policy was never demoted"

    # And the floor holds: demotion must not index off the front of the
    # schedule any more than promotion may run off the end.
    _run_reviews(task, 40, success_fraction=0.0)
    assert task.curriculum_stage == 0
    assert task._curriculum_epsilon == STAGES[0]
    # The high-water mark remembers where it had got to.
    assert task.curriculum_stage_max == 8


def test_the_dead_band_holds_a_stage_indefinitely():
    """Between demote_rate and promote_rate nothing moves. This is what stops
    a policy hovering near a threshold from ratcheting across it on noise."""
    stages = Go2StandCfg.curriculum
    middling = 0.5 * (stages.demote_rate + stages.promote_rate)

    task = _curriculum_task()
    task.curriculum_stage = 5
    task._apply_curriculum_stage()

    _finish_episodes(task, 400 * task.num_envs, middling)
    assert task.curriculum_reviews > 10, "the test never reviewed"
    assert task.curriculum_stage == 5, "the dead band did not hold"


def test_episodes_already_running_when_a_window_opens_are_not_scored():
    """Two bugs at once, which is why the rule is "started after the window
    opened" rather than merely "ended during it".

    Statistically, in-flight episodes are the inspection paradox: a hold
    occupies twenty times the wall-clock of a fall, so most running episodes
    are holds whatever the policy's actual success rate. Operationally, they
    were spawned under the stage the curriculum has just left -- which is the
    stale credit that promoted thirteen stages in twenty-five iterations.
    """
    task = _curriculum_task()
    task.curriculum_stage = 6
    task._apply_curriculum_stage()

    # Nothing has reset since the window opened, so nothing is `fresh`.
    mask = torch.ones(task.num_envs, dtype=torch.bool)
    for _ in range(40):
        task.common_step_counter += STEPS_PER_ITERATION
        task.episode_length_buf.fill_(100)
        task.timed_out.fill_(False)
        task.terminated.fill_(True)
        task._score_curriculum(mask)

    assert task._window_episodes == 0, "scored episodes the window does not own"
    assert task.curriculum_stage == 6, "cascaded on episodes it did not own"


def test_stage_moves_one_at_a_time_and_stops_at_the_last():
    """Single-step in both directions, and clamped: running off the end of the
    schedule would be an IndexError inside a reset, mid-training."""
    task = _curriculum_task()
    trace = [task.curriculum_stage]
    for _ in range(3000):
        _finish_episodes(task, task.num_envs, success_fraction=1.0)
        trace.append(task.curriculum_stage)

    assert max(abs(b - a) for a, b in zip(trace, trace[1:])) <= 1
    assert task.curriculum_stage == len(STAGES) - 1, "never reached the top"
    assert task.curriculum_stage_max == len(STAGES) - 1
    assert task._curriculum_epsilon == STAGES[-1]

    # The termination threshold follows the stage, and ONLY EVER UPWARD: it is
    # driven by the high-water mark, so a demotion does not tighten it. If it
    # did, every env sitting between the old and new thresholds would be killed
    # by the rule change -- taking a termination penalty and a truncated return
    # for a fall that never happened.
    loosest = task._max_attitude_error
    assert loosest > Go2StandCfg.termination.max_attitude_error
    task.curriculum_stage = 0
    task._apply_curriculum_stage()
    assert task._max_attitude_error == loosest, "the threshold tightened on demotion"
