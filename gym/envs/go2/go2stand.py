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
            0, self._prefix_subset(foot_names, ("FL", "FR"))
        )
        self.rear_feet_indices = self.feet_indices.index_select(
            0, self._prefix_subset(foot_names, ("RL", "RR"))
        )

        # Columns of the full-DOF tensors belonging to each leg pair. Resolved
        # through actuated_dof_indices rather than used directly, because the
        # actuated list may be a subset of the canonical DOF order.
        self.front_dof_indices = self.actuated_dof_indices.index_select(
            0, self._prefix_subset(self.actuated_dof_names, ("FL", "FR"))
        )
        self.rear_dof_indices = self.actuated_dof_indices.index_select(
            0, self._prefix_subset(self.actuated_dof_names, ("RL", "RR"))
        )

        self._target_projected_gravity = torch.tensor(
            [self.cfg.reward_settings.target_projected_gravity],
            dtype=torch.float,
            device=self.device,
        )

        # Horizontal position each support foot was planted at, refreshed on
        # reset. Drift from it is the slip measure -- see _reward_front_feet_slip.
        self._front_feet_anchor = torch.zeros(
            self.num_envs,
            len(self.front_feet_indices),
            2,
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
        # Re-anchor the support feet wherever the reset put them.
        masked_update(
            self._front_feet_anchor,
            self._rigid_body_pos[:, self.front_feet_indices, :2],
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

    def _prefix_subset(self, names, prefixes):
        """Positions in ``names`` whose entry starts with one of ``prefixes``."""
        return torch.tensor(
            [i for i, name in enumerate(names) if name.startswith(prefixes)],
            dtype=torch.long,
            device=self.device,
        )

    def _foot_load(self, indices, share=1.0):
        """Upward foot force as a fraction of its expected share of body weight.

        ``share`` is how much of the robot's weight one of these feet carries
        when the pose is held. Normalizing by the total instead caps a
        two-footed support term at half its weight, since neither foot can
        read more than 0.5. The division has to happen before the smoothstep,
        which is why this is a parameter rather than a factor on the result.
        """
        load = torch.clamp(
            self.contact_forces[:, indices, 2]
            / (share * 9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)),
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

    def _reward_dof_near_home(self):
        """Hold the support posture -- front legs only.

        Go2's version anchors all twelve joints to default_dof_pos, which is a
        direct penalty on moving the rear legs at all. They are the reaction
        mass this task wants to balance with, so they are split out into
        _reward_rear_dof_near_home at a much lower weight. Restricting from
        twelve joints to six halves this term's ceiling, so its weight is
        unchanged to keep the front anchoring exactly as strong as before.
        """
        error = (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
        return torch.sum(
            self._sqrdexp(error.index_select(1, self.front_dof_indices)), dim=1
        )

    # ------------ handstand-specific rewards ----------------

    def _reward_rear_dof_near_home(self):
        """Weak pull returning the rear legs to the extended posture.

        Weak, not absent: curling the rear legs draws the centre of mass down
        toward the feet, which is genuinely more stable, so with no pull at
        all the policy has an incentive to bunch them up permanently rather
        than extend between corrections.
        """
        error = (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
        return torch.sum(
            self._sqrdexp(error.index_select(1, self.rear_dof_indices)), dim=1
        )

    def _reward_front_feet_support(self):
        """Both support feet loaded at once.

        A mean over the pair would let one foot carry everything while the
        other hops; the minimum makes a single lifted foot a full failure,
        which is what keeps the support stance from stepping. Two feet carry
        the robot, so each one's share is half the body weight.
        """
        return self._foot_load(self.front_feet_indices, share=0.5).amin(dim=1)

    def _reward_front_feet_drift(self):
        """Penalize the support feet moving from where they were planted.

        Displacement rather than velocity, because the feet are 22 mm spheres:
        a foot rotating about a stationary contact point moves its body origin
        at radius x omega with nothing actually sliding, and that rotation is
        the ankle strategy this task depends on (2 rad/s reads as 0.044 m/s of
        false slip). Drift separates the two -- rolling is bounded by the
        radius, sliding accumulates.

        Linear rather than a squared exponential so the gradient is constant
        at every magnitude; a sqrdexp would go flat once a foot has already
        skated far, which is when the signal is needed most.
        """
        drift = (
            self._rigid_body_pos[:, self.front_feet_indices, :2]
            - self._front_feet_anchor
        )
        return -torch.mean(torch.norm(drift, dim=-1), dim=1)

    def _reward_front_feet_slip(self):
        """Penalize support-foot speed, as a damper on top of the drift term.

        Kept at a low weight: drift alone permits fast oscillation around the
        anchor, but this measure cannot distinguish sliding from the foot
        rolling in place, so it should not be the primary signal.
        """
        velocity = self._rigid_body_lin_vel[:, self.front_feet_indices, :2]
        return -torch.mean(torch.norm(velocity, dim=-1), dim=1)

    def _reward_rear_feet_air(self):
        """Penalize load on the rear feet -- they have to leave the ground."""
        return -self._foot_load(self.rear_feet_indices).mean(dim=1)
