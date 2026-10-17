from types import SimpleNamespace

import torch
import pytest

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg


def test_contact_strength_uses_each_environments_applied_total_mass():
    task = Go2Trot.__new__(Go2Trot)
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(2, 4, 3)
    task.contact_forces[:, :, 2] = 98.1
    task._backend = SimpleNamespace(link_mass=torch.tensor([[5.0, 5.0], [10.0, 10.0]]))

    strength = task._foot_contact_strength()

    torch.testing.assert_close(strength[0], torch.ones(4))
    torch.testing.assert_close(strength[1], torch.full((4,), 0.5))


@pytest.mark.parametrize(
    "targets, expected",
    [
        ([0.2, -0.2, 0.2], -0.16),  # Alternation remains visible.
        ([0.4, 0.2, 0.0], -0.08),  # Equal contributions for a constant rate.
        ([0.2, 0.2, 0.2], 0.0),  # Constant targets have no cost.
    ],
)
def test_action_rate_sums_one_step_and_normalized_two_step_costs(targets, expected):
    task = Go2Trot.__new__(Go2Trot)
    task.num_actuators = 12
    task.dof_pos_history = torch.tensor([[q for q in targets for _ in range(12)]])
    assert task._reward_action_rate().item() == pytest.approx(expected)


def test_yaw_tracking_uses_velocity_error_not_its_square():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    task.commands = torch.zeros(3, 4)
    task.base_ang_vel = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]
    )
    # Keep the existing 2.5 rad/s error scale; square the error only once.
    torch.testing.assert_close(
        task._reward_tracking_ang_vel(),
        torch.tensor(
            [1.0, torch.exp(torch.tensor(-0.16)), torch.exp(torch.tensor(-0.16))]
        ),
    )


def test_reference_reward_tracks_measured_joints_with_normalized_symmetric_error():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    # Include a passive joint and permute actuator order.
    task.actuated_dof_indices = torch.tensor([2, 0])
    task.default_dof_pos = torch.tensor([[0.4, 10.0, -0.6]])
    task.gait_reference = torch.tensor([[0.2, -0.1]])
    task.scales = {"dof_pos": torch.tensor([2.0, 1.0, 0.5])}
    task.dof_pos = torch.tensor([[0.3, -99.0, -0.4]])
    assert task._reward_reference_traj().item() == pytest.approx(1.0)
    for sign in [-1, 1]:
        task.dof_pos[0, 0] = 0.3 + sign * 1.0
        task.dof_pos[0, 2] = -0.4 + sign * 0.25
        expected = (
            torch.exp(torch.tensor(-0.25 / task.cfg.reward_settings.tracking_sigma))
            - 0.1
        )
        assert task._reward_reference_traj().item() == pytest.approx(expected.item())


def test_base_height_tracking_peaks_at_the_commanded_height_symmetrically():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    # task.scales = {"base_height": 0.3}
    scale = task.cfg.reward_settings.base_height_tracking_scale
    # One commanded height, approached from below, exactly, and from above.
    task.commands = torch.zeros(3, 4)
    task.commands[:, 3] = 0.30
    task.base_height = torch.tensor([[0.24], [0.30], [0.36]])

    reward = task._reward_tracking_base_height()

    # Per-environment scalar: a base_height left as [N, 1] would broadcast
    # against the command to [N, N] here instead.
    assert reward.shape == (3,)
    # Exact tracking is the optimum, and the term is a bonus in [0, 1].
    assert reward[1].item() == pytest.approx(1.0)
    assert torch.all(reward > 0.0)
    # Two-sided, unlike _reward_min_base_height: overshooting the command
    # costs exactly as much as falling short of it.
    assert reward[0].item() == pytest.approx(reward[2].item(), rel=1e-5)
    expected = torch.exp(
        torch.tensor(-((0.06 / scale) ** 2) / task.cfg.reward_settings.tracking_sigma)
    )
    assert reward[0].item() == pytest.approx(expected.item(), rel=1e-5)
    assert reward[0].item() < reward[1].item()
