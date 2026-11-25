"""Phase portrait of the go2stand curriculum's energy manifolds.

Each stage is the set of initial states whose energy is within a budget of the
handstand's:

    S(eps) = { (theta, theta_dot) : |E(theta, theta_dot) - E*| <= eps }

eps is the work the legs must supply, in joules, and it is conserved under the
uncontrolled dynamics -- so a stage labels whole coasting trajectories, not
isolated points. With no --stages this draws the schedule training will
actually run, generated from the config by HandstandPendulum.schedule.

    uv run --frozen scripts/plot_stand_curriculum.py \
        --output logs/curriculum/manifolds.png

Reading the figure: the dashed curve is the separatrix, where a state has
exactly the handstand's energy and coasts to it for free. The bands straddle
it. The red curve is the paper's z_crit = 0, drawn only in the two quadrants
where the centre of mass is moving back toward the contact line -- inside it,
height control alone suffices; outside, the rear-leg flywheel is doing the
work, which the paper's model excludes and this robot has.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from gym.envs.go2.go2stand_config import Go2StandCfg  # noqa: E402
from gym.envs.go2.go2stand_curriculum import HandstandPendulum  # noqa: E402


def print_schedule(pendulum, stages):
    print(
        f"\nE* = {pendulum.goal_energy:.3f} J   "
        f"(mass {pendulum.mass} kg, length {pendulum.length} m, "
        f"inertia {pendulum.inertia} kg m^2)"
    )
    cap = pendulum.lean_cap
    budget = pendulum.goal_energy * (1.0 - math.cos(cap))
    print(f"attitude termination caps the lean at {cap} rad = {budget:.2f} J\n")
    header = f"{'stage':>5} {'eps [J]':>8} {'d eps':>7} {'lean':>8} {'d lean':>7}"
    print(header + f" {'rate at rest':>13} {'rate at cap':>12}")
    previous_lean = previous_eps = None
    for index, epsilon in enumerate(stages):
        lean = pendulum.max_lean_at_rest(epsilon)
        _, upright = pendulum.rate_band(torch.zeros(1), epsilon)
        _, edge = pendulum.rate_band(torch.tensor([lean]), epsilon)
        # The two quantities the schedule exists to bound, so print them
        # rather than leaving them to be inferred from the column beside.
        step_eps = "" if previous_eps is None else f"{epsilon - previous_eps:7.3f}"
        step_lean = (
            ""
            if previous_lean is None
            else f"{math.degrees(lean - previous_lean):7.2f}"
        )
        print(
            f"{index:5d} {epsilon:8.3f} {step_eps:>7} "
            f"{math.degrees(lean):7.2f}d {step_lean:>7} "
            f"[0.00,{upright.item():5.2f}] [0.00,{edge.item():6.2f}]"
        )
        previous_lean, previous_eps = lean, epsilon


def draw(pendulum, stages, output):
    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    cap = pendulum.lean_cap
    theta = torch.linspace(-cap * 1.25, cap * 1.25, 801)

    colours = plt.cm.viridis(torch.linspace(0.15, 0.9, len(stages)).numpy())
    # Every stage is drawn, but a nineteen-entry legend would cover a third of
    # the figure, so only a few are named -- the bands nest, so the unnamed
    # ones read off the gradient between.
    spacing = max(1, len(stages) // 6)
    for index in reversed(range(len(stages))):
        epsilon = stages[index]
        lean = min(pendulum.max_lean_at_rest(epsilon), cap)
        inside = theta.abs() <= lean
        _, hi = pendulum.rate_band(theta[inside], epsilon)
        named = index % spacing == 0 or index == len(stages) - 1
        # The set is symmetric in theta_dot wherever standstill is admissible,
        # which is everywhere inside the default lean cap.
        axis.fill_between(
            theta[inside],
            -hi,
            hi,
            color=colours[index],
            alpha=0.55,
            label=f"stage {index}: {epsilon:.2f} J" if named else None,
            zorder=2,
        )

    separatrix = pendulum.separatrix_rate(theta)
    axis.plot(
        theta,
        separatrix,
        "k--",
        lw=1.2,
        zorder=4,
        label="separatrix (zero energy error)",
    )
    axis.plot(theta, -separatrix, "k--", lw=1.2, zorder=4)

    # z_crit > 0: where height control alone can still balance. Only defined
    # for inward motion, so it appears in two opposite quadrants.
    grid_t, grid_d = torch.meshgrid(
        torch.linspace(-cap * 1.25, cap * 1.25, 401),
        torch.linspace(-4.0, 4.0, 401),
        indexing="ij",
    )
    margin = pendulum.balance_margin(grid_t, grid_d)
    # NaN, not a finite sentinel, in the outward quadrants where the margin is
    # undefined: contour skips NaN but would happily trace the jump between a
    # sentinel and the real field, drawing a boundary along theta = 0 that
    # does not exist.
    margin = torch.where(torch.isinf(margin), torch.nan, margin)
    axis.contour(
        grid_t.numpy(),
        grid_d.numpy(),
        margin.numpy(),
        levels=[0.0],
        colors="crimson",
        linewidths=1.4,
        zorder=5,
    )
    axis.plot([], [], color="crimson", lw=1.4, label="z_crit = 0 (paper eq. 7)")

    for sign in (-1.0, 1.0):
        axis.axvline(sign * cap, color="0.35", lw=1.0, ls=":", zorder=3)
    axis.axvspan(cap, cap * 1.25, color="0.85", zorder=1)
    axis.axvspan(-cap * 1.25, -cap, color="0.85", zorder=1)
    for sign in (-1.0, 1.0):
        axis.text(
            sign * cap * 1.12,
            -3.4,
            "terminates\nimmediately",
            ha="center",
            va="center",
            fontsize=8,
            color="0.3",
        )

    axis.set_xlabel("lean from vertical  theta  [rad]")
    # "Inward" is a sign opposing theta, so it is the upper-left and
    # lower-right quadrants -- which is exactly where z_crit is drawn.
    axis.set_ylabel("lean rate  theta_dot  [rad/s]   (opposing theta = inward)")
    axis.set_title("go2stand curriculum: energy manifolds of the handstand pendulum")
    axis.set_xlim(-cap * 1.25, cap * 1.25)
    axis.set_ylim(-4.0, 4.0)
    axis.grid(alpha=0.25, zorder=0)
    axis.legend(loc="upper right", fontsize=8, framealpha=0.9)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    print(f"\nwrote {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stages",
        type=float,
        nargs="+",
        help="energy budgets in joules, ascending; default is the "
        "schedule generated from the config",
    )
    args = parser.parse_args(argv)

    cfg = Go2StandCfg()
    pendulum = HandstandPendulum.from_config(cfg)
    if args.stages:
        stages = tuple(args.stages)
    else:
        stages = pendulum.schedule(
            cfg.curriculum.start_energy,
            cfg.curriculum.ratio,
            cfg.curriculum.max_lean_step,
        )
    print_schedule(pendulum, stages)
    draw(pendulum, stages, args.output)


if __name__ == "__main__":
    main()
