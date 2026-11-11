"""Go2 balancing in a nose-down handstand, rewarded through a reduced-order model.

Standing on the two front feet makes the support polygon a line segment, so
rotation about that line is underactuated: the robot *is* the 2D variable-height
inverted pendulum of Koolen, Posa & Tedrake, "Balance control using center of
mass height variation" (Humanoids 2016).

The governing identity, for a stationary contact point p and whole-body centre
of mass c:

    d/dt L_p = (c - p) x m g

The contact cannot exert a torque about the contact line -- the feet are 22 mm
spheres -- so this holds for the whole articulated robot, not just a point mass.
Therefore L_p held at zero means the centre of mass is over the contact line,
which is the definition of balanced. One scalar subsumes lean angle, lean rate,
and "am I about to fall".

The identity is exact only for a stationary p, and p is tracked live here, so
a sliding foot adds an unmodelled -m * pdot x cdot. That is a deliberate
trade: pinning p would keep the identity exact but make the reward *wrong*
after any drift, pointing the robot away from the support it is actually
standing on. Keeping p live and charging for drift separately
(_reward_front_feet_drift) bounds pdot instead, which bounds the error term.

That is the core of the reward, plus a horizontal offset term to supply a
direct position gradient and a horizontal centre-of-mass speed term to close
the one loophole L_p leaves open (the rear legs can be swung to null L_p while
the body translates underneath). Those three are functions of the reduced-order
state alone: none of them mentions a joint, a foot, or a posture, so the policy
is free to choose how it balances.

Note what is *not* costed: vertical centre-of-mass motion. Raising and lowering
the centre of mass is the authority the paper studies, so charging for it would
be charging for the mechanism. A height target exists on the class but is
zero-weighted -- see _reward_com_height.

Two foot terms sit alongside them -- drift and contact -- because the pivot is
tracked live rather than pinned at reset, so the balance terms deliberately do
not notice the feet moving. And a constant alive bonus keeps the per-step total
positive, without which riding a fall down costs more than terminating does.

In particular the paper's assumption that centroidal angular momentum stays zero
-- which forces the ground reaction force through the centre of mass -- is
deliberately *not* imposed here. Only the total about the contact point is
rewarded, so the policy may swing the rear legs as a flywheel to cancel L_p
during a recovery and gets paid for doing it. What it cannot do is hold L_p at
zero that way, because the identity above has no term it can fake.
"""

import math

import torch

from gym.envs.go2.go2 import Go2
from gym.utils.sampling import masked_update

GRAVITY = 9.81


class Go2Stand(Go2):
    def _init_buffers(self):
        super()._init_buffers()

        # The support pair is taken from the layout's canonical foot order
        # rather than hardcoded indices, so a layout change cannot silently
        # swap which feet are expected to carry the robot.
        foot_names = self.robot_layout.body_groups["feet"]
        self.front_feet_indices = self.feet_indices.index_select(
            0, self._prefix_subset(foot_names, ("FL", "FR"))
        )

        self._target_projected_gravity = torch.tensor(
            [self.cfg.reward_settings.target_projected_gravity],
            dtype=torch.float,
            device=self.device,
        )

        # The pendulum's pivot, tracked live on the support feet -- see
        # _update_support_frame for why it is not captured at reset.
        self._support_anchor = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        # Where each support foot was planted, captured at reset. This is the
        # one thing that does stay pinned: it is the reference for foot drift,
        # which is a genuine "how far has this foot moved" question rather than
        # part of the pendulum's state.
        self._front_feet_anchor = torch.zeros(
            self.num_envs,
            len(self.front_feet_indices),
            2,
            dtype=torch.float,
            device=self.device,
        )
        # Rows are the support frame's x and y axes in world coordinates: y
        # along the contact line, x perpendicular to it.
        #
        # This does nothing for the rewards -- they are rotation-invariant
        # norms, so a yaw rotation cannot change them. It is here for the
        # *observations*: com_offset, com_vel and support_momentum are vectors,
        # and in world axes they would leak absolute yaw into the critic the
        # moment a curriculum stage widens root_pos_range's yaw off [0, 0].
        self._support_rot = torch.zeros(
            self.num_envs, 2, 2, dtype=torch.float, device=self.device
        )

        self.com_offset = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.com_height = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.com_vel = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.support_momentum = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        # Per-axis tolerances, in the support frame. The axis roles are
        # OPPOSITE for a displacement and a rotation, which is the easiest
        # thing here to transpose by accident:
        #
        #   offset along x (perpendicular to the contact line) is the knife
        #     edge -- there is no support polygon at all in that direction;
        #   offset along y stays inside the 0.142 m half-width, so it is
        #     comparatively harmless;
        #   momentum about x is roll, which the 0.284 m foot separation
        #     arrests;
        #   momentum about y is pitch about the contact line -- the
        #     underactuated mode, and the whole point of the task.
        #
        # So tight-x/loose-y for offset, and loose-x/tight-y for momentum.
        # Built here rather than read from the config each call because
        # reward_settings is read directly, not through class_to_dict, so a
        # bare Python list would not broadcast.
        self._offset_tolerance = torch.tensor(
            self.cfg.reward_settings.support_offset_tolerance,
            dtype=torch.float,
            device=self.device,
        )
        self._momentum_tolerance = torch.tensor(
            self.cfg.reward_settings.angular_momentum_tolerance,
            dtype=torch.float,
            device=self.device,
        )

        self._update_support_frame()
        self._update_reduced_order_state()

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
        # Re-plant the drift reference wherever the reset put the feet. The
        # backend has already run a forward pass and re-synced the assembled
        # states by this point (mj_forward + _copy_env_state_from_mujoco on
        # CPU, mjw.forward + _sync_assembled_states on warp), so the foot
        # positions read here are genuinely post-reset.
        masked_update(
            self._front_feet_anchor,
            self._rigid_body_pos[:, self.front_feet_indices, :2],
            reset_mask,
        )
        # Refresh here rather than in _refresh_base_observations: that runs
        # earlier inside super()._reset_idx(), and it is the only caller, so
        # doing it here covers the reset path with the post-reset state.
        self._update_support_frame()
        self._update_reduced_order_state()

    def _post_decimation_step(self):
        super()._post_decimation_step()
        # Control rate, alongside projected_gravity. super() comes first
        # because _push_robots writes root_states and re-syncs the rigid-body
        # tensors, which moves everything read below. These are written in
        # place into buffers allocated once, never refreshed lazily from a
        # property getter -- that pattern previously left GPU training reading
        # frozen observations for an entire run.
        self._update_support_frame()
        self._update_reduced_order_state()

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        # End the episode once the attitude is past recovery. The inherited
        # check only fires when the trunk actually touches down, long after
        # the fall is decided, so without this most of the rollout is spent
        # sampling a robot that is already on the ground. This is a
        # sample-efficiency gate, not a shaping term -- nothing rewards
        # attitude directly.
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

    # ------------ reduced-order state ----------------

    def _update_support_frame(self):
        """Track the pendulum's pivot and axes on the live support feet.

        Live, not captured at reset. A pivot frozen at reset gives an actively
        wrong gradient once the feet have moved: the robot can be balanced over
        its real support and still read a large offset, and the only way to
        reduce that offset is to lean away from where it is actually standing,
        which is to say, to fall. Foot motion is charged for by
        _reward_front_feet_drift instead, which measures it directly.
        """
        feet = self._rigid_body_pos.index_select(1, self.front_feet_indices)
        self._support_anchor[:] = feet.mean(dim=1)

        # y along the contact line, from the right foot to the left one. The
        # feet are listed in the layout's canonical order, so index 0 is FL.
        span = feet[:, 0, :2] - feet[:, 1, :2]
        # Guarded divide, not a bare one: the hip range is +/-1.047 rad, which
        # is enough abduction to bring the feet together, and a nan here would
        # reach the critic's observation vector.
        y_axis = span / torch.clamp(torch.norm(span, dim=1, keepdim=True), min=1e-6)
        # x is y rotated -90 degrees in the horizontal plane, which points along
        # the robot's nose for the canonical FL/FR ordering.
        x_axis = torch.stack((y_axis[:, 1], -y_axis[:, 0]), dim=1)
        self._support_rot[:] = torch.stack((x_axis, y_axis), dim=1)

    def _to_support_frame(self, vectors):
        """Express world vectors in the support frame, leaving z alone."""
        horizontal = torch.einsum("nij,nj->ni", self._support_rot, vectors[:, :2])
        return torch.cat((horizontal, vectors[:, 2:3]), dim=1)

    def _compute_reduced_order_state(self):
        """Centre of mass and angular momentum about the support anchor.

        Returned in the support frame, as (offset, height, velocity, momentum).

        Link positions and velocities are those of each body's *origin* rather
        than its centre of mass, and the per-link spin term sum(I_i w_i) is
        left out. Measured against an exact centroidal computation, the result
        is within 2% once |L| exceeds about 1 kg m^2/s, and both the proxy and
        the truth are exactly zero when the robot is static -- so the reward's
        optimum sits where the true one does.

        The one place it bites is height and horizontal offset at the goal:
        the proxy reads the centre of mass 8.9 mm high and 2.2 mm forward of
        the truth. The height bias is absorbed by measuring com_height_target
        with this same proxy. The 2.2 mm is in the knife-edge direction and is
        accepted: a converged policy holds the true centre of mass that far
        behind the contact line, a standing 0.35 Nm, which is 0.037 error
        units against support_offset_tolerance.
        """
        mass = self._backend.link_mass.unsqueeze(-1)
        total_mass = self._backend.link_mass.sum(dim=1, keepdim=True)

        relative = self._rigid_body_pos - self._support_anchor.unsqueeze(1)
        com = (mass * relative).sum(dim=1) / total_mass
        com_vel = (mass * self._rigid_body_lin_vel).sum(dim=1) / total_mass
        momentum = (mass * torch.cross(relative, self._rigid_body_lin_vel, dim=-1)).sum(
            dim=1
        )

        # Normalize by m * h * sqrt(g * h): the angular momentum of the whole
        # robot tipping at the pendulum's own natural rate, so the number is
        # dimensionless and O(1) at the edge of recoverability. Built from the
        # live per-environment mass, so mass randomization cannot move the
        # reward's zero.
        height_target = self.cfg.reward_settings.com_height_target
        momentum = momentum / (
            total_mass * height_target * math.sqrt(GRAVITY * height_target)
        )

        com = self._to_support_frame(com)
        return (
            com[:, :2],
            com[:, 2:3],
            self._to_support_frame(com_vel),
            self._to_support_frame(momentum),
        )

    def _update_reduced_order_state(self):
        offset, height, velocity, momentum = self._compute_reduced_order_state()
        self.com_offset[:] = offset
        self.com_height[:] = height
        self.com_vel[:] = velocity
        self.support_momentum[:] = momentum

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
            / (share * GRAVITY * self._backend.link_mass.sum(dim=1, keepdim=True)),
            min=0.0,
            max=1.0,
        )
        return load.square() * (3.0 - 2.0 * load)

    # ------------ rewards ----------------

    def _shaped(self, error):
        """Squared exponential with a bounded linear tail.

        The linear companion is not decoration: _sqrdexp is exp(-4e^2) here,
        whose derivative is numerically zero past about e = 1.5, which would
        leave a falling robot with no gradient telling it which way to correct.

        The clamp is load-bearing for a different reason. Unbounded, the tail
        makes a marginal state worth *less* than a terminating one: on a
        terminating step the dense weights are masked off and only
        termination_weight fires, costing -weight * dt = -1e-4, while riding a
        fall down costs far more per step. Early in training, when no policy
        can hold the pose yet, that is the only gradient available and it
        points at falling faster. Clamping bounds each term at -0.6, and
        _reward_balance_alive then pays for the rest. LeggedRobot's
        _reward_dof_vel_limits clips for the same reason.
        """
        return self._sqrdexp(error) - 0.2 * error.clamp(max=3.0)

    def _reward_balance_alive(self):
        """Constant, and the only term that is not about the pendulum.

        Sized so the per-step total stays strictly positive everywhere in the
        reachable state space: the four shaped terms bottom out at
        -0.6 * (3 + 2 + 1 + 2) = -4.8 weighted, against this term's 5.5
        (front_feet_support is a smoothstep in [0, 1] and cannot go negative).
        At the measured termination boundary they sum to about -4.45, so
        surviving scores +1.05 against -1e-4 for terminating.

        It says nothing about *how* to balance, so it does not re-specify the
        solution the way the twelve terms it replaces did.
        """
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_balance_momentum(self):
        """Angular momentum about the support anchor -> 0. The balance signal.

        Horizontal components only, and with per-axis tolerances. Momentum
        about y -- the contact line -- is the exact reduced-order coordinate
        and the underactuated mode, so it gets the tight tolerance. Momentum
        about x is roll, which the 0.284 m foot separation actually arrests,
        so it gets a looser one. Note this is the opposite assignment to
        _reward_balance_offset; see _init_buffers.

        Yaw is left uncosted -- gravity exerts no torque about the vertical,
        so it is outside the model entirely.
        """
        error = torch.norm(
            self.support_momentum[:, :2] / self._momentum_tolerance, dim=1
        )
        return self._shaped(error)

    def _reward_balance_offset(self):
        """Hold the centre of mass over the support anchor.

        Redundant with the momentum term only in the limit: dL/dt = (c - p) x mg
        means a steady lean shows up there eventually, but weakly and late.
        This supplies the position gradient directly.

        Per-axis, and the opposite way round to _reward_balance_momentum: a
        displacement along x leaves the contact line, where there is no
        support polygon at all, so that is the tight axis; a displacement
        along y stays inside the 0.142 m half-width and is comparatively
        harmless.

        The x tolerance is sized against the momentum term's so neither
        dominates: one momentum tolerance is dL = 3.4 kg m^2/s, i.e.
        cdot = dL / (m h) = 0.36 m/s, which at the pendulum's natural rate
        w0 = sqrt(m g h / I) = 3.82 rad/s is 0.094 m of capture-point offset.
        0.06 is two-thirds of that, keeping the two terms within a factor 1.6.
        """
        error = torch.norm(self.com_offset / self._offset_tolerance, dim=1)
        return self._shaped(error)

    def _reward_com_height(self):
        """Two-sided tracking of the centre-of-mass height above the feet.

        Currently zero-weighted and therefore never called -- holding a height
        does not itself make the pose more stable, which is what the task is
        for right now. Kept because re-enabling it is a one-line weight
        change, and because the physics argues it is not urgently needed: for
        an inverted pendulum the divergence rate is sqrt(m g h / I), and
        folding the rear legs cuts I faster than h, taking the robot from
        3.78 rad/s to roughly 5 rad/s. Crouching makes balancing *harder*, so
        the policy is already paid not to do it by every other term.

        Note cfg.reward_settings.com_height_target stays load-bearing while
        this is off: _compute_reduced_order_state uses it as the pendulum
        length in the angular-momentum normalizer.

        Replaces both min_base_height and the old rear-leg posture anchor:
        curling the rear legs draws the centre of mass down, so extension is
        paid for directly rather than through a joint-space target.

        The tolerance is three times looser than the base-height tracker it
        replaces. Varying centre-of-mass height is the control authority the
        paper is about, so a tight tracker would clamp away the one mechanism
        the reduced-order model actually studies -- but exp(-4e^2) is sharp,
        so "loose" here still means a few centimetres costs something.
        """
        error = (
            self.com_height.squeeze(1) - self.cfg.reward_settings.com_height_target
        ) / self.cfg.reward_settings.com_height_tolerance
        return self._sqrdexp(error) - 0.2 * error.abs().clamp(max=3.0)

    def _reward_com_vel_xy(self):
        """Penalize horizontal centre-of-mass speed, leaving vertical free.

        The split is the reduced-order model's own: vertical motion is the
        control authority the paper is about -- the robot raises and lowers
        its centre of mass to arrest a lean -- while horizontal motion is the
        divergent coordinate, and sustained horizontal travel means the robot
        is relocating rather than balancing.

        Not redundant with _reward_balance_momentum, which is the point.
        Angular momentum about the pivot can be driven to zero by swinging the
        rear legs against the trunk, which leaves the body free to translate
        underneath with L_p still reading ~0. That is the flywheel loophole,
        and this closes it. Nor is it redundant with _reward_front_feet_drift:
        drift is an accumulated displacement, this is a rate, so it fires at
        the start of a lunge rather than after it has finished.

        Isotropic, unlike the two balance terms. Stepping is equally wrong in
        any direction, and the directional part of the task is already carried
        by the anisotropic offset and momentum tolerances.
        """
        error = (
            torch.norm(self.com_vel[:, :2], dim=1)
            / self.cfg.reward_settings.com_vel_tolerance
        )
        return self._shaped(error)

    def _reward_front_feet_drift(self):
        """Displacement of each support foot from where it was planted.

        This is what the pivot no longer does. With the pendulum's pivot
        tracking the live feet, nothing in the balance terms notices a foot
        sliding, so foot motion is charged for directly here -- which is also
        the honest way round, because it penalizes drift instead of corrupting
        the balance signal with it.

        Max over the pair, not mean: a pivot about one planted foot sweeps the
        other through twice the distance the mean would report, halving the
        cost of exactly the behaviour this exists to stop.

        Displacement rather than velocity because the feet are 22 mm spheres.
        A foot rotating about a stationary contact point moves its body origin
        at radius x omega with nothing actually sliding, and that rotation is
        the ankle strategy the task depends on -- 2 rad/s reads as 0.044 m/s
        of entirely false slip. Drift separates the two: rolling is bounded by
        the radius, sliding accumulates.
        """
        drift = (
            self._rigid_body_pos[:, self.front_feet_indices, :2]
            - self._front_feet_anchor
        )
        error = (
            torch.amax(torch.norm(drift, dim=-1), dim=1)
            / self.cfg.reward_settings.foot_drift_tolerance
        )
        return self._shaped(error)

    def _reward_front_feet_support(self):
        """Both support feet loaded at once.

        This keeps the reduced-order model's own precondition true. With a
        foot off the ground the contact point the momentum is measured about
        is fictional, and in free flight the model does not apply at all --
        so without this, extending the legs explosively to raise the centre of
        mass is an unpriced way to score the height term.

        The minimum over the pair, not the mean: a mean would let one foot
        carry everything while the other hops, which is the stepping strategy
        the paper sets aside and this task is meant to exclude. Two feet carry
        the robot, so each one's share is half the body weight.
        """
        return self._foot_load(self.front_feet_indices, share=0.5).amin(dim=1)
