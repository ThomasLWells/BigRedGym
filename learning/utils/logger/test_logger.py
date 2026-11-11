from types import SimpleNamespace

import torch

from learning.utils import Logger


def test_logger_aggregates_finished_episodes_and_iteration_values():
    logger = Logger()
    logger.initialize(
        num_envs=3,
        episode_dt=0.1,
        total_iterations=100,
        device="cpu",
    )
    logger.register_rewards(["first", "second"])

    rewards = {
        "first": torch.tensor([5.0, 5.0, 5.0]) * 0.1,
        "second": torch.tensor([3.0, 0.0, 2.0]) * 0.1,
    }
    for step in range(10):
        logger.log_rewards(rewards)
        logger.finish_step(torch.tensor([False, step == 9, step == 9]))

    averages = logger.reward_logs.get_average_rewards()
    torch.testing.assert_close(averages["first"], torch.tensor(5.0))
    torch.testing.assert_close(averages["second"], torch.tensor(1.0))
    torch.testing.assert_close(logger.reward_logs.get_average_time(), torch.tensor(1.0))

    target = SimpleNamespace(value=3.0)
    logger.register_category("algorithm", target, ["value"])
    logger.log_all_categories()
    assert logger.iteration_logs.get_all_logs("algorithm") == {"value": 3.0}


def test_reward_histograms_cover_only_finished_episodes():
    logger = Logger()
    logger.initialize(num_envs=3, episode_dt=0.1, total_iterations=10, device="cpu")
    logger.register_rewards(["first"])

    # Nothing has finished yet, so there is no distribution to report. The
    # buffers are zero-filled at this point and would otherwise log a spike
    # at the origin for every term.
    assert logger.reward_histograms() == {}

    rewards = {"first": torch.tensor([5.0, 5.0, 5.0]) * 0.1}
    for step in range(10):
        logger.log_rewards(rewards)
        logger.finish_step(torch.tensor([False, step == 9, step == 9]))

    histograms = logger.reward_histograms()
    # One per reward term plus "time", the episode lengths.
    assert set(histograms) == {"rewards_hist/first", "rewards_hist/time"}
    # Two episodes finished, so only the first two buffer slots are sampled --
    # not the remaining 98 zeros.
    assert sum(histograms["rewards_hist/first"].histogram) == 2


def test_reward_histograms_survive_a_near_constant_term():
    """A constant reward averages to its weight in every episode, leaving the
    samples one float32 ulp apart. numpy bins in the input's own dtype, so 64
    float32 edges across one ulp collapse into duplicates and np.histogram
    raises "Too many bins for data range" -- which killed a GPU run from
    inside the logger. Exactly-equal samples never tripped this, because numpy
    widens that range by +/-0.5, which is why the original test missed it.
    """
    logger = Logger()
    logger.initialize(num_envs=2, episode_dt=0.1, total_iterations=10, device="cpu")
    logger.register_rewards(["alive"])

    logger.reward_logs.finished_episode_count = 2
    buffer = logger.reward_logs.finished_episodes["alive"]
    buffer[0] = 5.5
    buffer[1] = torch.nextafter(torch.tensor(5.5), torch.tensor(6.0))
    assert buffer[0] != buffer[1], "the samples must differ, or numpy copes anyway"

    histograms = logger.reward_histograms()
    assert sum(histograms["rewards_hist/alive"].histogram) == 2


def test_reward_histograms_skip_non_finite_terms_instead_of_raising():
    """A NaN reward is worth finding out about, but not by crashing the
    logger mid-run -- the scalar average still carries it to wandb."""
    logger = Logger()
    logger.initialize(num_envs=2, episode_dt=0.1, total_iterations=10, device="cpu")
    logger.register_rewards(["broken", "fine"])

    logger.reward_logs.finished_episode_count = 2
    logger.reward_logs.finished_episodes["broken"][0] = float("nan")
    logger.reward_logs.finished_episodes["fine"][0] = 1.0
    logger.reward_logs.finished_episodes["fine"][1] = 2.0

    histograms = logger.reward_histograms()
    assert "rewards_hist/broken" not in histograms
    assert "rewards_hist/fine" in histograms
