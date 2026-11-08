"""Go2 balancing in a nose-down handstand on its front feet.

The goal pose puts the trunk vertical with the rear legs in the air, so
gravity in the body frame points along +x (see Go2StandCfg). That makes two
of Go2's rewards wrong rather than merely misweighted -- `orientation`
rewards a level trunk, which is maximally violated here, and
`min_base_height` saturates above its target instead of holding one -- so
both are overridden below. The velocity-tracking terms have no meaning in a
handstand and are zero-weighted in the runner config instead.
"""

import torch

from gym.envs.go2.go2 import Go2
from gym.utils.sampling import masked_update


class Go2Stand(Go2):
    def _init_buffers(self):
        super()._init_buffers()

        # Front and rear feet are taken from the layout's canonical foot order
        # rather than hardcoded indices, so a layout change cannot silently
        # swap which pair is expected to carry the robot.
        foot_names = self.robot_layout.body_groups["feet"]
        self.front_feet_indices = self.feet_indices.index_select(
            0, self._foot_subset(foot_names, ("FL", "FR"))
        )
        self.rear_feet_indices = self.feet_indices.index_select(
            0, self._foot_subset(foot_names, ("RL", "RR"))
        )

        self._target_projected_gravity = torch.tensor(
            [self.cfg.reward_settings.target_projected_gravity],
            dtype=torch.float,
            device=self.device,
        )

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        # dof_pos_target is a residual: _compute_torques adds default_dof_pos
        # to it. LeggedRobot._reset_idx seeds it with default_dof_pos itself,
        # which would make the PD target twice the support posture on the
        # first step after every reset. Go2Trot._reset_idx corrects the same
        # thing; zero means "hold the posture you were reset into".
        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        masked_update(
            self.dof_pos_history,
            self.dof_pos_target.tile(1, 3),
            reset_mask,
        )

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        # End the episode once the attitude is past recovery. The inherited
        # check only fires when the trunk actually touches down, long after
        # the fall is decided, so without this most of the rollout is spent
        # sampling a robot that is already on the ground.
        attitude_error = torch.norm(
            self.projected_gravity - self._target_projected_gravity, dim=1
        )
        self.terminated |= attitude_error > self.cfg.termination.max_attitude_error

    def _foot_subset(self, foot_names, prefixes):
        return torch.tensor(
            [i for i, name in enumerate(foot_names) if name.startswith(prefixes)],
            dtype=torch.long,
            device=self.device,
        )

    def _foot_load(self, indices):
        """Upward foot force as a fraction of body weight, smoothed to [0, 1]."""
        load = torch.clamp(
            self.contact_forces[:, indices, 2]
            / (9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)),
            min=0.0,
            max=1.0,
        )
        return load.square() * (3.0 - 2.0 * load)

    # ------------ overridden Go2 rewards ----------------

    def _reward_orientation(self):
        """Track the handstand attitude instead of penalizing a non-flat base.

        Go2's version drives projected_gravity[:2] to zero -- a level trunk --
        which is exactly what this task must not do. The squared exponential
        is paired with a linear term because on its own it saturates at zero
        once the attitude error exceeds a few tolerances, leaving the policy
        with no gradient telling it which way to rotate.
        """
        error = (
            torch.norm(self.projected_gravity - self._target_projected_gravity, dim=1)
            / self.cfg.reward_settings.orientation_tolerance
        )
        return self._sqrdexp(error) - 0.2 * error

    def _reward_min_base_height(self):
        """Two-sided tracking of the handstand's base height.

        Go2's version clamps the error at zero, so any height at or above the
        target scores full marks. Here overshooting means driving the front
        legs toward their extension limit -- the pose tops out at 0.604 m --
        which is as bad as standing short, so both directions are penalized.
        The linear term serves the same purpose as in _reward_orientation.
        """
        error = (
            self.base_height.squeeze(1) - self.cfg.reward_settings.base_height_target
        ) / self.cfg.reward_settings.base_height_tracking_scale
        return self._sqrdexp(error) - 0.2 * error.abs()

    # ------------ handstand-specific rewards ----------------

    def _reward_front_feet_support(self):
        """Both support feet loaded at once.

        A mean over the pair would let one foot carry everything while the
        other hops; the minimum makes a single lifted foot a full failure,
        which is what keeps the support stance from stepping.
        """
        return self._foot_load(self.front_feet_indices).amin(dim=1)

    def _reward_front_feet_slip(self):
        """Penalize horizontal motion of the support feet -- they stay planted.

        A linear penalty rather than a squared exponential, so the gradient is
        constant at every magnitude: a sqrdexp would go flat exactly when a
        foot is sliding badly, which is when the signal is needed most.
        """
        velocity = self._rigid_body_lin_vel[:, self.front_feet_indices, :2]
        return -torch.mean(torch.norm(velocity, dim=-1), dim=1)

    def _reward_rear_feet_air(self):
        """Penalize load on the rear feet -- they have to leave the ground."""
        return -self._foot_load(self.rear_feet_indices).mean(dim=1)
