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
policy is free to choose *how* it balances. Two foot terms, a yaw term and an
attitude term cover what the model cannot see, and a constant alive bonus keeps
the per-step total positive.

The attitude term is there because the identity above is blind to a lean. It
pins where the centre of mass sits relative to the contact line and says
nothing about the orientation of the trunk carrying it. On a rigid pendulum
those coincide; on a twelve-joint robot the rear legs are exactly the
redundancy that satisfies the centre-of-mass condition at ANY trunk angle, so
a motionless lean scores the full goal total. A policy trained without it
learned precisely that, holding a 15 degree lean 99.2% of the time. The term
is a posture preference the reduced-order model does not have, and it is
deliberately on the trunk rather than on any joint -- naming the rear legs
would kill the flywheel the task exists to study.

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
from gym.envs.go2.go2stand_curriculum import HandstandPendulum
from gym.utils.sampling import masked_update
from gym.utils.torch_quat import quat_apply

GRAVITY = 9.81


class Go2Stand(Go2):
    # Picked up by the runner's logger so the stage trace reaches vitals.jsonl
    # and wandb. Without it a curriculum run is uninterpretable: the one thing
    # worth knowing is which lean the policy stalled at.
    #
    # stage_max against stage is how far the curriculum overshot before
    # correcting, i.e. whether the hysteresis is earning its keep.
    # review_episodes is a self-check on the gate itself: it should read about
    # cfg.curriculum.review_episodes, and a much larger number means episodes
    # are arriving in bursts far bigger than the rollout budget implies.
    # curriculum_hold_error is the headline for a run: the mean attitude error
    # of the episodes that SURVIVED, in chord units, directly comparable to
    # cfg.curriculum.hold_attitude_error. Read it next to
    # curriculum_survival_rate -- with no survivors it reads 0.0, which is the
    # empty mean and not a perfectly upright robot. And the gap between
    # curriculum_survival_rate (the OLD criterion, survival alone) and
    # curriculum_window_rate (survival AND upright) is literally what fraction
    # of this policy's success is leaning.
    log_categories = {
        "stand": [
            "curriculum_stage",
            "curriculum_stage_max",
            "curriculum_window_rate",
            "curriculum_survival_rate",
            "curriculum_hold_error",
            "curriculum_reviews",
            "curriculum_review_episodes",
        ]
    }

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
        # Where each support foot is standing. The reference for foot drift,
        # which is a "how far has this foot moved" question rather than part
        # of the pendulum's state. NOT simply pinned at reset: it follows the
        # feet until the handstand is achieved and freezes there -- see
        # _settle_drift.
        self._front_feet_anchor = torch.zeros(
            self.num_envs,
            len(self.front_feet_indices),
            2,
            dtype=torch.float,
            device=self.device,
        )
        self._drift_settled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
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
        # Sliding speed at each support foot's contact point. Unlike the rest
        # of these, this one is an ACTOR observation: the policy has to handle
        # low-friction ground, and nothing else it can see reveals that the
        # ground is slippery -- see _update_foot_slip.
        self.foot_slip = torch.zeros(
            self.num_envs,
            len(self.front_feet_indices),
            dtype=torch.float,
            device=self.device,
        )
        # Contact point relative to the foot body origin: the collision
        # sphere's centre, rotated into world each step, then dropped by its
        # radius. Stored unrotated here.
        self._foot_sphere_offset = torch.tensor(
            self.cfg.asset.foot_sphere_offset, dtype=torch.float, device=self.device
        )
        self._foot_contact_drop = torch.tensor(
            [0.0, 0.0, -self.cfg.asset.foot_radius],
            dtype=torch.float,
            device=self.device,
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

        # reset_to_curriculum draws its joint jitter from the same tensors
        # reset_to_range does, but _init_buffers only builds those for that one
        # mode -- so build them here when the curriculum is driving.
        if self.cfg.init_state.reset_mode == "reset_to_curriculum":
            self.initialize_ranges_for_initial_conditions()

        self._pendulum = HandstandPendulum.from_config(self.cfg)
        stages = self.cfg.curriculum
        self._curriculum_schedule = self._pendulum.schedule(
            stages.start_energy, stages.ratio, stages.max_lean_step
        )
        # The difficulty itself is this one index: the sampler generates states
        # from the budget directly, so there are no range tensors to keep in
        # sync and no way for the config and the live distribution to disagree.
        # Everything else here is the review bookkeeping.
        self.curriculum_stage = 0
        self.curriculum_stage_max = 0
        self.curriculum_window_rate = 0.0
        self.curriculum_survival_rate = 0.0
        self.curriculum_hold_error = 0.0
        self.curriculum_reviews = 0
        self.curriculum_review_episodes = 0.0
        self._window_successes = 0.0
        self._window_survivals = 0.0
        self._window_hold_error = 0.0
        self._window_episodes = 0.0
        self._curriculum_stage_step = 0
        # A window samples each env's first episode to START after the window
        # opened: `fresh` marks an env that has begun such an episode and
        # `reported` one that has already contributed. See _score_curriculum
        # for why both halves are needed.
        self._curriculum_fresh = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._curriculum_reported = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Attitude error accumulated over the scored TAIL of each episode, and
        # the number of steps in it. Surviving the horizon is not evidence of
        # a handstand; this is what turns it into evidence. See
        # _accumulate_hold_quality.
        self._hold_error_sum = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._hold_error_steps = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # max_episode_length comes from _parse_cfg, which LeggedRobot.__init__
        # runs before _init_buffers; it is a float from np.ceil.
        self._hold_start_step = int(
            math.ceil(
                (1.0 - self.cfg.curriculum.hold_fraction) * self.max_episode_length
            )
        )
        self._apply_curriculum_stage()

        self._update_support_frame()
        self._update_reduced_order_state()
        self._update_foot_slip()

    # ------------ curriculum ----------------

    def _apply_curriculum_stage(self):
        """Cache what depends on the stage index, including termination.

        The last stage leans to within 0.017 rad of the configured attitude
        threshold, so a sample there would terminate on its first outward step.
        The threshold therefore follows the stage -- but only ever upward, so
        every stage below roughly 20 degrees of lean keeps the configured 0.6
        exactly and nothing about the established behaviour moves.

        Note the units. ``max_attitude_error`` is a CHORD on the unit sphere
        (it compares projected_gravity against its target), not an angle, so
        permitting a lean of phi takes 2 sin(phi/2) rather than phi.
        """
        # Maintained here rather than in _change_stage so the invariant holds
        # however the stage was set -- play.py pins it directly from
        # --curriculum_stage, and a stale high-water mark would misreport how
        # far a run had overshot.
        self.curriculum_stage_max = max(
            self.curriculum_stage_max, self.curriculum_stage
        )
        self._curriculum_epsilon = self._curriculum_schedule[self.curriculum_stage]
        self._curriculum_lean_cap = min(
            self._pendulum.max_lean_at_rest(self._curriculum_epsilon),
            self._pendulum.lean_cap,
        )
        # Driven by the HIGH-WATER mark, so the threshold only ever loosens.
        # If it tracked the live stage, a demotion would tighten it and envs
        # sitting between the old and new values would terminate on the next
        # step -- a termination penalty and a truncated return for a fall that
        # never happened.
        #
        # REVISIT THIS. Its original justification was that no reward term
        # mentioned attitude, so there was nothing to buy by tightening back
        # down. _reward_balance_attitude deletes that premise: the termination
        # and the reward are now two statements about the same quantity, and
        # this lets them disagree by 36%. One excursion to stage 15 pinned the
        # threshold at 40.8 degrees of permitted lean for the rest of a 5000
        # iteration run. It is kept for now only because the reward, not the
        # termination, is meant to hold the pose -- if the policy still leans,
        # tightening this is the next lever.
        #
        # The cost of reverting is also smaller than this comment used to
        # imply: termination_margin is 0.25 rad against a largest one-stage
        # lean step of 0.0436, so a one-stage demotion cannot kill a freshly
        # spawned env at any stage -- only envs already past 34 degrees, which
        # are overwhelmingly falling anyway.
        widest = self._curriculum_schedule[self.curriculum_stage_max]
        allowed = min(
            min(self._pendulum.max_lean_at_rest(widest), self._pendulum.lean_cap)
            + self.cfg.curriculum.termination_margin,
            math.pi,
        )
        self._max_attitude_error = max(
            self.cfg.termination.max_attitude_error,
            2.0 * math.sin(0.5 * allowed),
        )

    def _change_stage(self, delta):
        """Move the curriculum one stage and start the new one clean."""
        self.curriculum_stage += delta
        # Evidence gathered at the stage just left says nothing about the new
        # one, in either direction.
        self._open_window()
        self._apply_curriculum_stage()

    def _open_window(self):
        """Start a fresh review window.

        Nobody is `fresh`: episodes already in flight were begun before this
        window opened and are not part of its sample.
        """
        self._window_successes = 0.0
        self._window_survivals = 0.0
        self._window_hold_error = 0.0
        self._window_episodes = 0.0
        self._curriculum_fresh.fill_(False)
        self._curriculum_reported.fill_(False)
        self._curriculum_stage_step = self.common_step_counter

    def _score_curriculum(self, reset_mask):
        """Review the stage periodically and move it up or down.

        Called at the TOP of ``_reset_idx``, for two reasons: the flags, the
        episode counter and the hold accumulators are still those of the
        episode that just ended, and a stage change then takes effect on the
        very envs resetting on this step rather than a policy step later.

        A stage is solved when episodes **survive the horizon AND hold the
        handstand** -- see the `held` conjunct below. Survival alone was the
        original criterion and it is what certified a policy that balanced at
        a steady 15 degree lean through fifteen stages, because a motionless
        lean survives indefinitely. Both rates are logged, and the gap between
        them is what fraction of the policy's success is leaning.

        A *review* rather than a running average. The previous version kept an
        EMA that was reset to zero on every promotion, which made its own rise
        time -- not the policy's competence -- the thing pacing the curriculum;
        and an EMA reset to zero is not an unbiased estimate of anything.

        The sample is **each env's first episode to START after the window
        opened**, exactly once per env. Both halves of that are needed, and
        each guards against a bias in the opposite direction.

        Counting episodes instead of envs is length-biased toward FAILURES.
        Episode length correlates with outcome by construction -- a fall
        resolves in under a second, a hold takes the full ten -- so a failing
        env reports twenty times for every once a succeeding one does. Against
        a true 90% policy that reads 0.68, and 0.40 on the first review after a
        stage change, which would demote it.

        Counting each env's *in-flight* episode is length-biased toward
        SUCCESSES, by the same ratio working the other way: holds occupy
        twenty times more wall-clock than falls, so at any instant most envs
        are inside one. That is the inspection paradox, and against a true 30%
        policy it reads 0.95 -- which would promote it. Hence `fresh`: an
        episode already running when the window opened is not part of the
        sample.

        Because a window opens on every stage change, this also keeps out
        episodes spawned at a stage the curriculum has left -- the stale credit
        that let the previous run clear thirteen stages in twenty-five
        iterations on the strength of stage-0 episodes.

        Demotion is the point of the exercise. Without it the 2000-iteration
        run spent 1762 iterations -- 88% of itself -- above a frontier it could
        not hold, with no way back down.
        """
        # An env that has not run is not an episode. The startup reset() calls
        # _reset_idx with `terminated` still all-True from construction, which
        # would otherwise book num_envs failures before a single step.
        finished = (
            reset_mask
            & (self.episode_length_buf > 0)
            & self._curriculum_fresh
            & ~self._curriculum_reported
        )
        count = int(finished.sum())
        if count:
            # timed_out and terminated are NOT mutually exclusive -- nothing
            # clears terminated when the horizon fires -- so a fall on the last
            # step of an episode has both set, and gating on the timeout alone
            # would score it a success.
            survived = finished & self.timed_out & ~self.terminated
            # ...and surviving is still not succeeding. A motionless lean
            # survives indefinitely, which is how a policy holding 15 degrees
            # was promoted through fifteen stages. The episode must also have
            # HELD the pose: mean attitude error over the scored tail, within
            # cfg.curriculum.hold_attitude_error.
            #
            # Multiplied through rather than divided, so there is no
            # divide-by-zero to guard. The degenerate steps == 0 case reads
            # 0 <= 0 and passes, which is unreachable: an episode that never
            # got past the halfway mark cannot have timed out.
            held = self._hold_error_sum <= (
                self.cfg.curriculum.hold_attitude_error * self._hold_error_steps
            )
            self._window_successes += float((survived & held).sum())
            self._window_survivals += float(survived.sum())
            self._window_hold_error += float(
                (
                    self._hold_error_sum[survived]
                    / self._hold_error_steps[survived].clamp(min=1.0)
                ).sum()
            )
            self._window_episodes += count
            self._curriculum_reported |= finished

        stages = self.cfg.curriculum
        elapsed = self.common_step_counter - self._curriculum_stage_step
        if (
            self._window_episodes < stages.review_fraction * self.num_envs
            or elapsed < stages.min_review_steps
        ):
            return

        self.curriculum_window_rate = self._window_successes / self._window_episodes
        self.curriculum_survival_rate = self._window_survivals / self._window_episodes
        self.curriculum_hold_error = self._window_hold_error / max(
            self._window_survivals, 1.0
        )
        self.curriculum_reviews += 1
        self.curriculum_review_episodes = self._window_episodes
        self._open_window()

        # One unbiased window of the whole population decides it; the dead band
        # between the two rates is the hysteresis, and at this sample size it
        # is twenty-five standard errors wide.
        if (
            self.curriculum_window_rate >= stages.promote_rate
            and self.curriculum_stage + 1 < len(self._curriculum_schedule)
        ):
            self._change_stage(1)
        elif (
            self.curriculum_window_rate <= stages.demote_rate
            and self.curriculum_stage > 0
        ):
            self._change_stage(-1)

    def reset_to_curriculum(self, reset_mask):
        """Spawn on the current stage's energy manifold.

        A reset mode of its own rather than a widening of ``root_pos_range``,
        because the lean, the lean rate and the base position are correlated --
        the position is a function of the lean, and the admissible rate band is
        a function of both the lean and the budget. Independent per-axis boxes
        cannot express that, so they would spawn states off the manifold.
        """
        # The joint jitter and its range tensors, unchanged from what trained
        # before; the root pose it also samples is then discarded, the write
        # below covering all 13 components.
        self.reset_to_range(reset_mask)

        theta, theta_dot = self._pendulum.sample(
            self._curriculum_epsilon,
            self.num_envs,
            lean_cap=self._curriculum_lean_cap,
            device=self.device,
        )
        masked_update(
            self.root_states,
            self._pendulum.to_root_state(theta, theta_dot),
            reset_mask,
        )

    def _reset_idx(self, reset_mask):
        self._score_curriculum(reset_mask)
        super()._reset_idx(reset_mask)
        # These episodes begin now, so they belong to the open window -- AFTER
        # _score_curriculum has had its chance to close one and open another,
        # which is what keeps a window's sample to episodes spawned under the
        # stage it is judging.
        self._curriculum_fresh |= reset_mask
        # Same ordering: _score_curriculum has already read these for the
        # episode that just ended. Without the clear, a failed episode's lean
        # leaks into the next one and the hold criterion becomes a random walk.
        self._hold_error_sum.masked_fill_(reset_mask, 0.0)
        self._hold_error_steps.masked_fill_(reset_mask, 0.0)
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
        # Re-plant the drift reference and un-latch it, so the new episode may
        # step itself upright again. The backend has already run a forward
        # pass and re-synced the assembled states, so these foot positions are
        # genuinely post-reset. Without the un-latch, one episode's settled
        # anchor would charge the next episode for its recovery.
        masked_update(
            self._front_feet_anchor,
            self._rigid_body_pos[:, self.front_feet_indices, :2],
            reset_mask,
        )
        self._drift_settled.masked_fill_(reset_mask, False)
        # Here rather than in _refresh_base_observations, which runs earlier
        # inside super() and is its only caller; doing it here covers the
        # reset path with post-reset state.
        self._update_support_frame()
        self._update_reduced_order_state()
        self._update_foot_slip()

    def _pre_decimation_step(self):
        """Bound the commanded PD target before anything reads it.

        The actor is an unsquashed Gaussian and nothing between it and the
        torques bounds its output, so the action is limited only by whatever
        the network happens to produce. With no actuation cost in the reward
        that turned into a policy commanding 132 rad of joint target -- more
        than twenty revolutions of a joint with about two of travel -- which
        drove joints to 173 rad/s against the calf's 15.7 rad/s rating and
        pinned torque at its clip 89.5% of the time.

        Here rather than in the reward, because a weight is a preference and
        this is a fact about the robot. Here rather than in _compute_torques,
        because this runs once per control step before the decimation loop
        and before dof_pos_history records the target -- so the torques, the
        action-rate penalties and the observation all see the same clipped
        value rather than disagreeing about what was commanded.
        """
        super()._pre_decimation_step()
        limit = self.cfg.control.max_dof_pos_target
        self.dof_pos_target.clamp_(-limit, limit)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        # Control rate, alongside projected_gravity. super() first, because
        # _push_robots writes root_states and re-syncs the rigid-body tensors
        # that everything below reads. Written in place into buffers allocated
        # once: refreshing lazily from a property getter instead would leave
        # GPU training reading stale values for a whole run.
        self._update_support_frame()
        self._update_reduced_order_state()
        self._update_foot_slip()
        # After _update_support_frame, so the foot positions it plants are
        # this step's. Before the terminations, which read the drift it sets.
        self._settle_drift()
        self._accumulate_hold_quality()

    def _accumulate_hold_quality(self):
        """Average the attitude error over the tail of each episode.

        The curriculum's evidence that a stage is actually solved. super()
        has already incremented episode_length_buf and refreshed
        projected_gravity, so this reads post-step attitude against a 1-based
        step index.

        Only the tail counts. The robot SPAWNS at the stage's lean cap -- 21
        degrees at stage 13, 34 at stage 18 -- so averaging the whole episode
        would charge each stage for its own spawn transient, and charge the
        hard stages most. Scoring the tail is stage-independent and asks the
        question that matters: a recovery that takes five seconds is not a
        handstand.
        """
        scoring = (self.episode_length_buf >= self._hold_start_step).float()
        self._hold_error_sum += self._attitude_error() * scoring
        self._hold_error_steps += scoring

    def _foot_drift(self):
        """Metres the furthest support foot has travelled from its anchor.

        Max over the pair, not mean: a pivot about one planted foot sweeps the
        other through twice the distance a mean would report, halving the cost
        of exactly the behaviour this exists to measure.

        One definition for the reward and the termination, so the two cannot
        drift apart. What the anchor means is set by _settle_drift.
        """
        travel = (
            self._rigid_body_pos[:, self.front_feet_indices, :2]
            - self._front_feet_anchor
        )
        return torch.amax(torch.norm(travel, dim=-1), dim=1)

    def _settle_drift(self):
        """Let the robot step itself upright, then hold it where it landed.

        Foot drift is measured from ``_front_feet_anchor``. Until the trunk is
        actually upright that anchor FOLLOWS the feet, so drift reads ~0 and
        stepping costs nothing -- neither in the reward nor in the drift
        termination. The moment the handstand is achieved the anchor freezes
        where the feet are standing, and every centimetre after that is
        charged.

        This is what makes the drift termination safe. Recovering from a 29
        degree spawn genuinely requires stepping -- the energy has to go
        somewhere -- and a flat drift limit would kill exactly the recovery
        the curriculum exists to teach. Separating "getting there" from
        "staying there" lets both be priced correctly.

        The latch is PERMANENT for the episode. Releasing it whenever the
        trunk tipped would hand the policy an obvious loophole given this
        task's history: tip briefly to re-plant the anchor, then step freely.
        A policy that instead never becomes upright is already charged by
        _reward_balance_attitude and already fails the curriculum's hold
        criterion, so refusing to settle is not an escape either.
        """
        settling = ~self._drift_settled
        # Re-plant for everyone not yet settled, INCLUDING those crossing the
        # threshold on this very step, so the anchor lands exactly where the
        # handstand was achieved rather than a step late.
        masked_update(
            self._front_feet_anchor,
            self._rigid_body_pos[:, self.front_feet_indices, :2],
            settling,
        )
        self._drift_settled |= settling & (
            self._attitude_error() <= self.cfg.reward_settings.drift_settle_attitude
        )

    def _attitude_error(self):
        """Chord on the unit sphere between the trunk's gravity direction and
        the handstand's.

        One definition for the reward, the termination and the curriculum's
        hold criterion, so the three cannot drift apart. A lean of phi reads
        2 sin(phi/2): 0.0872 at 5 degrees, 0.2611 at 15, 0.600 at 34.9.
        """
        return torch.norm(
            self.projected_gravity - self._target_projected_gravity, dim=1
        )

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        # End the episode once the attitude is past recovery. The inherited
        # check only fires when the trunk touches down, long after the fall is
        # decided, so without this most of the rollout samples a robot that is
        # already on the ground. A sample-efficiency gate, deliberately far
        # looser than _reward_balance_attitude, which is what actually holds
        # the pose upright.
        #
        # Stage-dependent, not the raw config value -- see
        # _apply_curriculum_stage.
        self.terminated |= self._attitude_error() > self._max_attitude_error

        # Walking is not a handstand. Pricing it was not enough: a policy
        # trained with drift at weight 2.0 and nothing else held 1.2 degrees
        # of lean and perfect balance while covering 57 cm from a motionless
        # start, because surviving paid more than the drift cost.
        #
        # Measured from the SETTLED anchor, so this cannot fire while the
        # robot is still righting itself -- see _settle_drift.
        self.terminated |= self._foot_drift() > self.cfg.termination.max_foot_drift

    def _reward_orientation(self):
        """Inherited from Go2 and WRONG here -- refuse it rather than let it
        be reached for.

        Go2's version rewards a FLAT base, so in a handstand it is maximized
        by lying down: it would pay the policy to fall over. It is unweighted
        today, but adding a correctly-targeted attitude term makes this the
        obvious thing for a future reader to enable.
        """
        raise NotImplementedError(
            "Go2._reward_orientation targets a flat base and rewards falling "
            "over in a handstand; use balance_attitude instead."
        )

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

    def _update_foot_slip(self):
        """Sliding speed at each support foot's contact point.

        The feet are 22 mm spheres, so the body origin moves under pure
        rolling with nothing actually sliding. The contact point is what
        slides: on flat ground it sits directly below the sphere centre, and
        for a rigid body

            v_contact = v_origin + omega x (p_contact - p_origin)

        which cancels the rolling component exactly and leaves true slip.

        This exists because the actor is otherwise blind to the ground. It
        sees no friction and no contact force, so across a 0.8-1.0 friction
        range it can only learn one policy that averages over the draw --
        measured, domain randomization accounts for 81% of the spread in
        foot drift between episodes. Slip is the one proprioceptive-adjacent
        quantity that reveals which world this episode is in.

        Gated by foot load, which makes this "sliding while carrying weight"
        rather than "the contact point is moving". The distinction is not
        pedantic: a foot in the air has no contact point, but the formula
        above still evaluates one, and a swinging foot produces a large
        value. Ungated, the observation saturated its clip for 1% of samples
        -- 34x the true loaded mean -- so the policy was being told the floor
        was giving way whenever it picked a foot up.

        The gate is the smoothstep from _foot_load, so an unloaded foot
        contributes nothing, a lightly loaded one contributes in proportion,
        and nothing jumps at lift-off.

        Clipped as well: observations here are not normalized, and a crashing
        robot reaches several m/s against an operating range two orders
        smaller.
        """
        feet = self.front_feet_indices
        quaternion = self._rigid_body_quat[:, feet]
        offset = (
            quat_apply(
                quaternion,
                self._foot_sphere_offset.expand(quaternion.shape[:-1] + (3,)),
            )
            + self._foot_contact_drop
        )
        velocity = self._rigid_body_lin_vel[:, feet] + torch.cross(
            self._rigid_body_ang_vel[:, feet], offset, dim=-1
        )
        speed = torch.norm(velocity[..., :2], dim=-1)
        loaded = self._foot_load(feet, share=0.5)
        self.foot_slip[:] = (speed * loaded).clamp(max=self.cfg.asset.foot_slip_max)

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

    def _shaped_log_tail(self, error):
        """Squared exponential with a tail that never goes flat.

        The one shaping function in this task; `error` is a magnitude, already
        divided by its tolerance, and must be non-negative (log1p diverges at
        -1). Every caller passes a norm or an abs.

        A clamped tail is fatal wherever the policy can sit past the clamp,
        because there the term is a constant, and a constant is invisible to
        PPO at any weight -- GAE cancels it as soon as the critic adapts, and
        advantages are variance-normalized on top of that. Two things put the
        policy there. Foot drift accumulates monotonically within an episode,
        so raising its weight under a clamp did nothing at all for 900
        iterations. And the curriculum *spawns* the robot at its stage's lean
        cap: by the 18.97-degree stage the offset error alone is e = 3.3, so a
        clamp at e = 3 would hand the robot zero gradient on precisely the
        terms meant to guide it home.

        log1p keeps the far field bounded in practice while never zeroing the
        gradient, which stays at -0.2/(1+e). This is *weaker* than the clamped
        tail in the near field (-1.33 against -4.00 per metre at 0.10 m of
        drift); the point is not a bigger push but one that survives, so that
        the weight is a working knob.

        The clamp at e = 40 is a floor guard -- 2 m of drift, or 46 degrees of
        lean -- not a shaping choice. It bounds the term at -0.743, which is
        what _reward_balance_alive's sum is computed against.
        """
        return self._sqrdexp(error) - 0.2 * torch.log1p(error.clamp(max=40.0))

    def _reward_balance_alive(self):
        """Constant, and the only term that is not about the pendulum.

        Sized so the per-step total stays strictly positive everywhere in the
        reachable state space. Every shaped term shares the one logarithmic
        tail and so bottoms out at -0.743 -- except balance_attitude, whose
        argument is a chord on the unit sphere and therefore cannot exceed
        2.0, capping its error at 22.9 and its floor at -0.635. So

            floor = -0.743 * (3 + 2 + 1 + 1 + 2) + -0.635 * 3 = -8.59

        against this term's 9.0 (front_feet_support is a smoothstep in [0, 1]
        and cannot go negative). The constraint to preserve when retuning:

            balance_alive >= 0.743 * sum(unbounded shaped weights)
                             + 0.635 * balance_attitude

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
        return self._shaped_log_tail(error)

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
        return self._shaped_log_tail(error)

    def _reward_dof_vel_limits(self):
        """Charge for joint speed past what the motor is rated for.

        Overridden rather than inherited. The base version clips the excess
        at 1 rad/s per joint, so past 0.9 * rating + 1 it is a CONSTANT --
        15.1 rad/s for the calf, against measured speeds of 173. A constant
        is invisible to PPO at any weight, and this task has already lost 900
        iterations to exactly that (see _shaped_log_tail). The term has to
        keep pulling where the behaviour actually is.

        The same logarithmic tail every shaped term here uses, for the same
        reason: bounded in practice, never flat. The clamp at 200 rad/s of
        excess is a floor guard, not a shaping choice -- it puts the term at
        -1.06 and nothing reaches it.

        Max over the joints, not mean, for the reason _reward_front_feet_drift
        takes the max over the pair: one joint at eleven times its rating is
        the fault, and averaging it against eleven well-behaved joints divides
        the signal by twelve. Measured on the blown-up policy the mean form
        reads -0.036 against the good policy's -0.001 -- a real ratio but
        both so close to zero that no sane weight makes it matter.

        Why this is needed on top of control.max_dof_pos_target: the action
        clip bounds what is COMMANDED, not what the joint does. At the calf's
        45.43 Nm against 0.0119 kg m^2 the motor delivers 3820 rad/s^2, which
        covers the whole 15.7 rad/s rating in 4.1 ms -- under half a control
        step. The torque limit therefore cannot bound speed at 100 Hz, and
        measured per-step accelerations reach 9221 rad/s^2, 2.4x what the
        strongest joint can produce, so contact and solver error add to it.
        The physically correct fix is a torque-speed curve in the actuator
        model; this is the cheap approximation of it, and being a reward
        rather than a constraint it can be outbid.
        """
        excess = (
            self.dof_vel.abs()
            - self.dof_vel_limits * self.cfg.reward_settings.soft_dof_vel_limit
        ).clamp(min=0.0, max=200.0)
        return -0.2 * torch.log1p(torch.amax(excess, dim=1))

    def _reward_balance_attitude(self):
        """Trunk upright -> the pose is a handstand, not merely a balance.

        The one term here that is NOT a function of the reduced-order state,
        and it exists because that state cannot see a lean. The identity the
        task is built on, d/dt L_p = (c - p) x mg, constrains where the centre
        of mass sits relative to the contact line and says nothing about the
        orientation of the trunk carrying it. For a RIGID pendulum those are
        the same statement -- balance_offset reads 0.55 at a 2 degree rigid
        lean and -0.26 at 15. For a twelve-joint robot they are not: the rear
        legs are exactly the redundancy that puts the centre of mass over the
        contact line at any trunk angle.

        So a robot standing motionless at any lean, with its mass correctly
        distributed, reads support_momentum exactly zero (every link velocity
        is zero), com_offset zero, com_vel zero, drift zero and support at
        ceiling -- the full goal total, while visibly not doing a handstand.
        A policy trained 5000 iterations found precisely that pose and held a
        15.1 degree lean 99.2% of the time.

        Measured on projected_gravity, which is already an ACTOR observation.
        That makes this the only reward whose quantity the policy can read
        directly -- com_offset and support_momentum are critic-side and have
        to be inferred through implicit forward kinematics -- so expect it to
        be learned faster than its weight alone suggests.

        Isotropic, unlike the offset and momentum tolerances. Those are
        anisotropic because the support polygon is a line; this is a pose
        quantity, not a balance quantity, and a handstand leaning sideways is
        as wrong as one leaning over the toes. If the policy starts trading
        roll for pitch, splitting the tolerance is the first refinement.
        """
        error = self._attitude_error() / self.cfg.reward_settings.attitude_tolerance
        return self._shaped_log_tail(error)

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
        return self._shaped_log_tail(error)

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
        return self._shaped_log_tail(error.abs())

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
        return self._shaped_log_tail(error)

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
        error = self._foot_drift() / self.cfg.reward_settings.foot_drift_tolerance
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
