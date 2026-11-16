"""Energy manifolds for the go2stand back-reachability curriculum.

The handstand is an inverted pendulum pivoting on the front-foot contact line,
so an initial state can be ranked by the work the legs must do to reach the
goal. For the variable-height model of Koolen, Posa & Tedrake the dynamics give

    dE/dt = m u (q . q_dot) = F_leg * r_dot

-- energy changes only when the leg length changes under load, so the leg is
the actuator that pumps it. That makes energy the natural difficulty currency.

A curriculum layer is the set

    S(eps) = { (theta, theta_dot) : |E(theta, theta_dot) - E*| <= eps }
    E(theta, theta_dot) = 0.5 I theta_dot^2 + m g r cos(theta)

with eps in joules. Because E is conserved under the *uncontrolled* dynamics,
eps labels an entire coasting trajectory rather than a single point -- which is
what makes this a reachability layer rather than an arbitrary box.

Two things energy alone does not say, both handled here:

Direction. A far lean moving outward at the separatrix rate also has zero
energy error, but it is swinging away and never returns. The paper's eq. (5),
x * x_dot < 0, excludes it; see ``sample``, where the rule activates exactly
when the rate band's lower bound leaves zero.

Sufficiency. Having the budget says nothing about whether the work can be
delivered before the ground or a joint stop intervenes. The paper's eq. (7),
z_crit > 0, is the exact condition for the height-control-only model; this
robot also has a rear-leg flywheel, so its true reachable set is strictly
larger. ``balance_margin`` is therefore a label to plot, never a filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

GRAVITY = 9.81


@dataclass(frozen=True)
class HandstandPendulum:
    """Reduced-order model of the Go2 balanced on its front feet.

    All of these are measured at the nominal pose, not derived -- see
    ``from_config`` for where they come from and what they mean.
    """

    mass: float
    length: float
    inertia: float
    pivot_x: float
    base_position: tuple[float, float, float]
    base_pitch: float
    lean_cap: float

    @classmethod
    def from_config(cls, cfg):
        init, reward = cfg.init_state, cfg.reward_settings
        return cls(
            mass=reward.handstand_mass,
            length=reward.handstand_pendulum_length,
            inertia=reward.handstand_inertia,
            pivot_x=reward.handstand_pivot_x,
            base_position=tuple(init.pos),
            base_pitch=0.5 * math.pi,
            lean_cap=cfg.termination.max_attitude_error,
        )

    # ------------ energy ----------------

    @property
    def goal_energy(self) -> float:
        """Energy of the handstand itself: upright and motionless."""
        return self.mass * GRAVITY * self.length

    def energy(self, theta, theta_dot):
        """Total energy of a rigid lean at ``theta`` turning at ``theta_dot``.

        The kinetic term uses the true inertia about the contact line, not
        m r^2 -- the rear legs are 4.7 kg out at 0.93 m and dominate the second
        moment, so the point-mass form is 9% low.
        """
        theta, theta_dot = torch.as_tensor(theta), torch.as_tensor(theta_dot)
        return 0.5 * self.inertia * theta_dot**2 + (
            self.mass * GRAVITY * self.length * torch.cos(theta)
        )

    def energy_deficit(self, theta, theta_dot):
        """Work the legs must supply, in joules. Negative means too much."""
        return self.goal_energy - self.energy(theta, theta_dot)

    def separatrix_rate(self, theta):
        """Rate at which a lean has exactly the handstand's energy.

        A state on this curve moving inward coasts to the goal with no work at
        all; it is the curve the rate bands are centred on.
        """
        theta = torch.as_tensor(theta)
        gap = self.mass * GRAVITY * self.length * (1.0 - torch.cos(theta))
        return torch.sqrt(2.0 * gap / self.inertia)

    def rate_band(self, theta, epsilon):
        """|theta_dot| admissible at this lean for a budget of ``epsilon``.

        Returns (lo, hi). ``lo`` leaves zero exactly when the lean is too far
        to recover from standstill on this budget -- past that point the state
        must already be moving, and ``sample`` makes it move inward.
        """
        theta = torch.as_tensor(theta)
        gap = self.mass * GRAVITY * self.length * (1.0 - torch.cos(theta))
        lo = torch.sqrt(torch.clamp(2.0 * (gap - epsilon) / self.inertia, min=0.0))
        hi = torch.sqrt(torch.clamp(2.0 * (gap + epsilon) / self.inertia, min=0.0))
        return lo, hi

    def max_lean_at_rest(self, epsilon: float) -> float:
        """Furthest lean this budget can recover from standstill.

        Also the default lean cap: it keeps a layer to a genuine neighbourhood
        of the goal, so the sets grow monotonically outward instead of
        admitting far-but-fast states from the very first stage.
        """
        argument = 1.0 - epsilon / (self.mass * GRAVITY * self.length)
        return math.pi if argument <= -1.0 else math.acos(argument)

    # ------------ sampling ----------------

    def sample(self, epsilon, count, generator=None, lean_cap=None):
        """Draw ``count`` states uniformly from S(epsilon).

        ``lean_cap`` defaults to ``max_lean_at_rest(epsilon)`` and is clipped
        to the attitude termination, beyond which a sample would terminate on
        its first step and teach nothing.
        """
        cap = self.max_lean_at_rest(epsilon) if lean_cap is None else lean_cap
        cap = min(cap, self.lean_cap)

        theta = (torch.rand(count, generator=generator) * 2.0 - 1.0) * cap
        lo, hi = self.rate_band(theta, epsilon)
        magnitude = lo + torch.rand(count, generator=generator) * (hi - lo)

        # Free sign where standstill is admissible; inward once it is not,
        # which is the paper's eq. (5) -- an outward swing at that distance
        # never comes back, whatever its energy says.
        free = torch.rand(count, generator=generator) < 0.5
        sign = torch.where(lo > 0.0, -torch.sign(theta), torch.where(free, -1.0, 1.0))
        return theta, magnitude * sign

    # ------------ mapping to the simulator ----------------

    def to_root_state(self, theta, theta_dot):
        """Rigid rotation of the nominal handstand about the contact line.

        Returns ``[count, 13]`` laid out as LeggedRobot expects: position,
        quaternion **scalar-last**, then linear and angular velocity **both in
        the world frame**. The backend rotates the angular part into body-local
        on write using the new quaternion, so world is what to hand it.

        The joints do not move -- this is the whole robot leaning as one body,
        which is why the energy above may use a single rigid inertia.
        """
        theta = torch.as_tensor(theta, dtype=torch.float)
        theta_dot = torch.as_tensor(theta_dot, dtype=torch.float)
        count = theta.shape[0]

        pivot = torch.tensor([self.pivot_x, 0.0, 0.0])
        offset = torch.tensor(self.base_position) - pivot
        cos, sin = torch.cos(theta), torch.sin(theta)

        # Rotation by +theta about +y carries a point at +z toward +x, so
        # positive theta leans the centre of mass over the toes.
        position = torch.stack(
            (
                pivot[0] + offset[0] * cos + offset[2] * sin,
                torch.full((count,), float(self.base_position[1])),
                pivot[2] - offset[0] * sin + offset[2] * cos,
            ),
            dim=1,
        )

        half = 0.5 * (self.base_pitch + theta)
        quaternion = torch.zeros(count, 4)
        quaternion[:, 1] = torch.sin(half)
        quaternion[:, 3] = torch.cos(half)

        angular = torch.zeros(count, 3)
        angular[:, 1] = theta_dot
        linear = torch.cross(angular, position - pivot, dim=1)
        return torch.cat((position, quaternion, linear, angular), dim=1)

    # ------------ diagnostics ----------------

    def balance_margin(self, theta, theta_dot):
        """The paper's z_crit (eq. 7), for states moving inward.

        Substituting the rigid lean x = r sin(theta), z = r cos(theta) into
        z - z_dot x / x_dot - g x^2 / (2 x_dot^2) collapses to

            z_crit = r / cos(theta) - g tan^2(theta) / (2 theta_dot^2)

        Positive means centre-of-mass height control alone can still balance
        the state. Negative does NOT mean unrecoverable here: the paper
        excludes the hip strategy, and this robot can swing its rear legs. A
        label, not a filter.
        """
        theta = torch.as_tensor(theta)
        theta_dot = torch.as_tensor(theta_dot)
        inward = theta * theta_dot < 0.0
        margin = self.length / torch.cos(theta) - (
            GRAVITY * torch.tan(theta) ** 2 / (2.0 * theta_dot**2)
        )
        return torch.where(inward, margin, torch.full_like(margin, -math.inf))
