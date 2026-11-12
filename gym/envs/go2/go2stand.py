"""Go2 balancing in a nose-down handstand, rewarded through a reduced-order model.

Standing on the two front feet makes the support polygon a line segment, so
rotation about that line is underactuated: the robot *is* the 2D variable-height
inverted pendulum of Koolen, Posa & Tedrake, "Balance control using center of
mass height variation" (Humanoids 2016).

The governing identity, for contact point p and whole-body centre of mass c:

    d/dt L_p = (c - p) x m g

The contact exerts no torque about the contact line -- the feet are 22 mm
spheres -- so this holds for the whole articulated robot, not just a point mass.
L_p held at zero therefore means the centre of mass is over the contact line,
which is the definition of balanced: one scalar subsuming lean angle, lean rate
and "am I about to fall". It is exact only for a stationary p, and p is tracked
live, so a sliding foot adds an unmodelled -m * pdot x cdot term;
_reward_front_feet_drift bounds pdot, which bounds that error.

Three terms are functions of the reduced-order state alone -- momentum, offset,
horizontal centre-of-mass speed -- and name no joint, foot or posture, so the
policy is free to choose *how* it balances. Two foot terms and a yaw term cover
what the model cannot see, and a constant alive bonus keeps the per-step total
positive.

Two things are deliberately uncosted. Vertical centre-of-mass motion, because
raising and lowering the centre of mass is the authority the paper studies and
charging for it would be charging for the mechanism. And centroidal angular
momentum: the paper assumes it stays zero, which forces the ground reaction
force through the centre of mass, and relaxing that lets the policy swing the
rear legs as a flywheel to cancel L_p during a recovery. What the flywheel
cannot do is *hold* L_p at zero, because the identity above has no term it can
fake.
"""

import math

import torch

from gym.envs.go2.go2 import Go2
from gym.utils.sampling import masked_update

GRAVITY = 9.81


class Go2Stand(Go2):
    def _init_buffers(self):
        super()._init_buffers()

        # Resolved through the layout's canonical foot order rather than
        # hardcoded, so a layout change cannot silently swap which pair is
        # expected to carry the robot.
        foot_names = self.robot_layout.body_groups["feet"]
        self.front_feet_indices = self.feet_indices.index_select(
            0, self._prefix_subset(foot_names, ("FL", "FR"))
        )

        self._target_projected_gravity = torch.tensor(
            [self.cfg.reward_settings.target_projected_gravity],
            dtype=torch.float,
            device=self.device,
        )

        # The pendulum's pivot, tracked live -- see _update_support_frame.
        self._support_anchor = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        # Where each support foot was planted, pinned at reset. The reference
        # for foot drift, which is a "how far has this foot moved" question
        # rather than part of the pendulum's state.
        self._front_feet_anchor = torch.zeros(
            self.num_envs,
            len(self.front_feet_indices),
            2,
            dtype=torch.float,
            device=self.device,
        )
        # Rows are the support frame's x and y axes in world coordinates: y
        # along the contact line, x perpendicular to it. This buys the rewards
        # nothing -- they are rotation-invariant norms. It exists for the
        # *observations*: com_offset, com_vel and support_momentum are vectors,
        # and in world axes they would leak absolute yaw into the critic as
        # soon as a curriculum stage widens root_pos_range's yaw off [0, 0].
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

        # Per-axis tolerances in the support frame. The axis roles are
        # OPPOSITE for a displacement and a rotation, which is the easiest
        # thing here to transpose by accident:
        #
        #   offset along x crosses the contact line, where there is no support
        #     polygon at all -- the knife edge, so tight;
        #   offset along y stays inside the 0.142 m half-width -- loose;
        #   momentum about x is roll, which the 0.284 m foot separation
        #     arrests -- loose;
        #   momentum about y is pitch about the contact line, the
        #     underactuated mode and the whole point of the task -- tight.
        #
        # Built as tensors here because reward_settings is read directly
        # rather than through class_to_dict, so a bare list would not
        # broadcast.
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
        # dof_pos_target is a residual that _compute_torques adds
        # default_dof_pos to, and LeggedRobot._reset_idx seeds it *with*
        # default_dof_pos -- which would make the PD target twice the support
        # posture on the first step after every reset. Zero means "hold the
        # posture you were reset into".
        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        masked_update(
            self.dof_pos_history,
            self.dof_pos_target.tile(1, 3),
            reset_mask,
        )
        # Re-plant the drift reference. The backend has already run a forward
        # pass and re-synced the assembled states, so these foot positions are
        # genuinely post-reset.
        masked_update(
            self._front_feet_anchor,
            self._rigid_body_pos[:, self.front_feet_indices, :2],
            reset_mask,
        )
        # Here rather than in _refresh_base_observations, which runs earlier
        # inside super() and is its only caller; doing it here covers the
        # reset path with post-reset state.
        self._update_support_frame()
        self._update_reduced_order_state()

    def _post_decimation_step(self):
        super()._post_decimation_step()
        # Control rate, alongside projected_gravity. super() first, because
        # _push_robots writes root_states and re-syncs the rigid-body tensors
        # that everything below reads. Written in place into buffers allocated
        # once: refreshing lazily from a property getter instead would leave
        # GPU training reading stale values for a whole run.
        self._update_support_frame()
        self._update_reduced_order_state()

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        # End the episode once the attitude is past recovery. The inherited
        # check only fires when the trunk touches down, long after the fall is
        # decided, so without this most of the rollout samples a robot that is
        # already on the ground. A sample-efficiency gate, not a shaping term:
        # no reward mentions attitude.
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

        Live rather than pinned at reset, because a pinned pivot gives an
        actively wrong gradient once the feet have moved: the robot can be
        balanced over its real support and still read a large offset, and the
        only way to null that offset is to lean off the feet. Foot motion is
        charged for by _reward_front_feet_drift, which measures it directly.
        """
        feet = self._rigid_body_pos.index_select(1, self.front_feet_indices)
        self._support_anchor[:] = feet.mean(dim=1)

        # y along the contact line, right foot to left; index 0 is FL in the
        # layout's canonical order.
        span = feet[:, 0, :2] - feet[:, 1, :2]
        # Guarded divide: the hip range is +/-1.047 rad, enough abduction to
        # bring the feet together, and a nan here reaches the critic's
        # observation vector.
        y_axis = span / torch.clamp(torch.norm(span, dim=1, keepdim=True), min=1e-6)
        # x is y rotated -90 degrees in the horizontal plane, which points
        # along the robot's nose for the canonical FL/FR ordering.
        x_axis = torch.stack((y_axis[:, 1], -y_axis[:, 0]), dim=1)
        self._support_rot[:] = torch.stack((x_axis, y_axis), dim=1)

    def _to_support_frame(self, vectors):
        """Express world vectors in the support frame, leaving z alone."""
        horizontal = torch.einsum("nij,nj->ni", self._support_rot, vectors[:, :2])
        return torch.cat((horizontal, vectors[:, 2:3]), dim=1)

    def _compute_reduced_order_state(self):
        """Centre of mass and angular momentum about the support anchor.

        Returned in the support frame, as (offset, height, velocity, momentum).

        Link positions and velocities are each body's *origin* rather than its
        centre of mass, and the per-link spin term sum(I_i w_i) is left out.
        Against an exact centroidal computation that is within 2% once |L|
        exceeds about 1 kg m^2/s, and both the proxy and the truth are exactly
        zero when the robot is static, so the reward's optimum sits where the
        true one does.

        Where it bites is at the goal: the proxy reads the centre of mass
        8.9 mm high and 2.2 mm forward of the truth. The height bias cancels
        because com_height_target is measured with this same proxy. The 2.2 mm
        lies in the knife-edge direction and is accepted -- a converged policy
        holds the true centre of mass that far behind the contact line, a
        standing 0.35 Nm, worth 0.037 error units.
        """
        mass = self._backend.link_mass.unsqueeze(-1)
        total_mass = self._backend.link_mass.sum(dim=1, keepdim=True)

        relative = self._rigid_body_pos - self._support_anchor.unsqueeze(1)
        com = (mass * relative).sum(dim=1) / total_mass
        com_vel = (mass * self._rigid_body_lin_vel).sum(dim=1) / total_mass
        momentum = (mass * torch.cross(relative, self._rigid_body_lin_vel, dim=-1)).sum(
            dim=1
        )

        # Normalize by m * h * sqrt(g * h), the angular momentum of the whole
        # robot tipping at the pendulum's natural rate, so the number is
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
        when the pose is held. Normalizing by the total instead would cap a
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

        The linear companion carries the far field: _sqrdexp is exp(-4e^2)
        here, whose derivative is numerically zero past about e = 1.5, leaving
        a falling robot with no gradient telling it which way to correct.

        The clamp bounds each term at -0.6 so that riding a fall down is never
        worth less than terminating. A terminating step masks off the dense
        weights and costs only -weight * dt = -1e-4, so an unbounded tail
        would make "finish the fall quickly" the best available action -- and
        early in training, when nothing can hold the pose, the only gradient
        there is. _reward_balance_alive pays for the bounded remainder.
        """
        return self._sqrdexp(error) - 0.2 * error.clamp(max=3.0)

    def _shaped_log_tail(self, error):
        """Squared exponential with a tail that never goes flat.

        For quantities that accumulate monotonically within an episode, a
        clamped tail is fatal: once past the clamp the term is a constant, and
        a constant is invisible to PPO at any weight -- GAE cancels it as soon
        as the critic adapts, and advantages are variance-normalized on top of
        that. Foot drift is the one such quantity here, so raising its weight
        would otherwise do nothing at all.

        log1p keeps the far field bounded in practice while never zeroing the
        gradient, which stays at -0.2/(1+e). This is *weaker* than the clamped
        tail in the near field (-1.33 against -4.00 per metre at 0.10 m of
        drift); the point is not a bigger push but one that survives, so that
        the weight is a working knob.

        The clamp at e = 40 is a floor guard at 2 m of drift, not a shaping
        choice -- com_vel_xy dominates long before that. It bounds the term at
        -0.743, which is what _reward_balance_alive's sum is computed against.
        """
        return self._sqrdexp(error) - 0.2 * torch.log1p(error.clamp(max=40.0))

    def _reward_balance_alive(self):
        """Constant, and the only term that is not about the pendulum.

        Sized so the per-step total stays strictly positive everywhere in the
        reachable state space. The four _shaped terms bottom out at
        -0.6 * (3 + 2 + 1 + 1) = -4.2 weighted, and front_feet_drift on the
        logarithmic tail bottoms at -0.743 * 2.0 = -1.49, for a floor of -5.69
        against this term's 6.0 (front_feet_support is a smoothstep in [0, 1]
        and cannot go negative). The constraint to preserve when retuning:

            balance_alive >= 0.6 * sum(clamped term weights)
                             + 0.743 * front_feet_drift

        Being constant, it says nothing about *how* to balance, so it adds no
        shaping of its own.
        """
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_balance_momentum(self):
        """Angular momentum about the support anchor -> 0. The balance signal.

        Horizontal components, with per-axis tolerances. Momentum about y --
        the contact line -- is the exact reduced-order coordinate and the
        underactuated mode, so it gets the tight tolerance; about x is roll,
        which the 0.284 m foot separation arrests, so it gets a looser one.
        Note this is the opposite assignment to _reward_balance_offset; see
        _init_buffers.

        Yaw is costed by _reward_balance_yaw rather than by joining this norm.
        """
        error = torch.norm(
            self.support_momentum[:, :2] / self._momentum_tolerance, dim=1
        )
        return self._shaped(error)

    def _reward_balance_yaw(self):
        """Angular momentum about the vertical -> 0.

        Gravity exerts no torque about the vertical, so yaw sits outside the
        reduced-order model, and projected_gravity is yaw-invariant so the
        attitude termination is blind to a spin. Nothing else can see it.

        A separate term rather than a third component of
        _reward_balance_momentum's norm, and that separation is load-bearing.
        A norm's gradient splits between its components as v_i/||v||, so the
        largest component owns it -- and yaw is excited far more easily than
        pitch, I_zz about the contact line being 0.136 kg m^2 against
        I_yy = 6.48. Under exploration-scale actions yaw momentum reaches 97%
        of a combined norm within 0.1 s and holds 72-76% through a fall, which
        would leave the balance signal with a fraction of its own gradient.
        Split out, each keeps all of it.
        """
        error = (
            self.support_momentum[:, 2].abs()
            / self.cfg.reward_settings.yaw_momentum_tolerance
        )
        return self._shaped(error)

    def _reward_balance_offset(self):
        """Hold the centre of mass over the support anchor.

        Overlaps the momentum term only in the limit: dL/dt = (c - p) x mg
        means a steady lean shows up there eventually, but weakly and late.
        This supplies the position gradient directly.

        Per-axis, and the opposite way round to _reward_balance_momentum: a
        displacement along x crosses the contact line, where there is no
        support polygon, so that is the tight axis; along y it stays inside
        the 0.142 m half-width and is comparatively harmless.

        The x tolerance is sized against the momentum term's so neither
        dominates. One momentum tolerance is dL = 3.4 kg m^2/s, i.e.
        cdot = dL / (m h) = 0.36 m/s, which at the pendulum's natural rate
        w0 = sqrt(m g h / I) = 3.82 rad/s is 0.094 m of capture-point offset;
        0.06 is two-thirds of that, keeping the two within a factor 1.6.
        """
        error = torch.norm(self.com_offset / self._offset_tolerance, dim=1)
        return self._shaped(error)

    def _reward_com_height(self):
        """Two-sided tracking of the centre-of-mass height above the feet.

        Zero-weighted, so never called. Holding a height does not itself make
        the pose more stable, and the physics already discourages the crouch
        it would guard against: for an inverted pendulum the divergence rate
        is sqrt(m g h / I), and folding the rear legs cuts I faster than h,
        taking the robot from 3.78 rad/s to roughly 5 rad/s. Kept because
        re-enabling it is a one-line weight change.

        cfg.reward_settings.com_height_target stays load-bearing while this is
        off -- _compute_reduced_order_state uses it as the pendulum length in
        the angular-momentum normalizer.

        The tolerance is loose, because varying centre-of-mass height is the
        control authority the paper is about and a tight tracker would clamp
        away the one mechanism it studies. exp(-4e^2) is sharp, though, so
        "loose" still means a few centimetres costs something.
        """
        error = (
            self.com_height.squeeze(1) - self.cfg.reward_settings.com_height_target
        ) / self.cfg.reward_settings.com_height_tolerance
        return self._sqrdexp(error) - 0.2 * error.abs().clamp(max=3.0)

    def _reward_com_vel_xy(self):
        """Penalize horizontal centre-of-mass speed, leaving vertical free.

        The split is the reduced-order model's own: vertical motion is the
        control authority -- the robot raises and lowers its centre of mass to
        arrest a lean -- while horizontal motion is the divergent coordinate,
        and sustained horizontal travel means the robot is relocating rather
        than balancing.

        It closes the flywheel loophole, which is why it is not redundant with
        _reward_balance_momentum: angular momentum about the pivot can be
        driven to zero by swinging the rear legs against the trunk, leaving
        the body free to translate underneath with L_p still reading ~0. Nor
        is it redundant with _reward_front_feet_drift -- drift is accumulated
        displacement, this is a rate, so it fires at the start of a lunge
        rather than after it has finished.

        Isotropic, unlike the two balance terms: stepping is equally wrong in
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

        The pivot tracks the live feet, so nothing in the balance terms
        notices a foot sliding; foot motion is charged for here instead. That
        is the honest way round -- it prices drift rather than corrupting the
        balance signal with it.

        Max over the pair, not mean: a pivot about one planted foot sweeps the
        other through twice the distance a mean would report, halving the cost
        of exactly the behaviour this exists to stop.

        Displacement rather than velocity, because the feet are 22 mm spheres.
        A foot rotating about a stationary contact point moves its body origin
        at radius x omega with nothing sliding, and that rotation is the ankle
        strategy the task depends on -- 2 rad/s reads as 0.044 m/s of entirely
        false slip. Drift separates the two: rolling is bounded by the radius,
        sliding accumulates.

        The log tail is what keeps that accumulation from saturating the term
        into a constant -- see _shaped_log_tail.
        """
        drift = (
            self._rigid_body_pos[:, self.front_feet_indices, :2]
            - self._front_feet_anchor
        )
        error = (
            torch.amax(torch.norm(drift, dim=-1), dim=1)
            / self.cfg.reward_settings.foot_drift_tolerance
        )
        return self._shaped_log_tail(error)

    def _reward_front_feet_support(self):
        """Both support feet loaded at once.

        Keeps the reduced-order model's own precondition true: with a foot off
        the ground the contact point the momentum is measured about is
        fictional, and in free flight the model does not apply at all. Without
        this, extending the legs explosively is an unpriced way to move the
        centre of mass.

        Minimum over the pair, not mean: a mean would let one foot carry
        everything while the other hops, which is the stepping strategy the
        paper sets aside and this task excludes. Two feet carry the robot, so
        each one's share is half the body weight.
        """
        return self._foot_load(self.front_feet_indices, share=0.5).amin(dim=1)
