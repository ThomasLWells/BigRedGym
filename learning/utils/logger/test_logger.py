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
