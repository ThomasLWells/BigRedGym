"""Histogram each reward term of a training run against its own ceiling.

    uv run --frozen scripts/plot_run_rewards.py \
        --run logs/go2stand/Nov12_16-52-40_ --output logs/rewards.png

Values are divided by the weight the run actually trained with, read from the
run's own code snapshot under <run>/files/ rather than from the working tree --
a run is only interpretable against the config it used. Every term is then a
bare score with ceiling 1.0, so panels are directly comparable and the gap to
1.0 is the headroom in that term.

Each panel overlays the first and second half of training. That is the
convergence test: a converging term shifts toward its ceiling AND narrows, so
two overlapping distributions of equal width mean the term has stopped
learning, whatever the time series looks like.

Note what this cannot show. vitals.jsonl stores per-ITERATION values, each
already a time-average over the last 100 finished episodes, so these
distributions are over training, not over episodes. For the shape of a reward
across episodes -- bimodality, saturation spikes -- the per-episode
`rewards_hist/*` panels in wandb are the right instrument.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WEIGHT_PATTERN = re.compile(r"^\s{8,}([a-z_]+)\s*=\s*([0-9.eE+-]+)\s*$")
CLASS_PATTERN = re.compile(r"^\s*class\s+\w+\s*[(:]")


def read_weights(run: Path) -> dict[str, float]:
    """Reward weights from the run's own snapshot, not the working tree.

    The block ends at the next real `class` STATEMENT, not at the next
    occurrence of the word -- these configs carry long comments, and one that
    mentions "this weights class does not inherit" truncates a naive split
    partway through, yielding a subset of the weights and a report that looks
    entirely plausible.
    """
    config = run / "files/gym/envs/go2/go2stand_config.py"
    if not config.exists():
        raise SystemExit(f"no snapshot config at {config}")

    lines = config.read_text().splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "class weights:"), None
    )
    if start is None:
        raise SystemExit(f"no 'class weights:' block in {config}")

    weights = {}
    for line in lines[start + 1 :]:
        if CLASS_PATTERN.match(line):
            break
        found = WEIGHT_PATTERN.match(line)
        if found:
            weights[found[1]] = float(found[2])
    return weights


def check_complete(rows, weights):
    """Every logged reward term must have a parsed weight.

    vitals.jsonl names exactly the terms that survived zero-weight pruning, so
    a missing one means the parse dropped it -- and dividing by the wrong
    weight, or silently skipping a panel, is worse than failing here.
    """
    logged = {
        key.removeprefix("rewards/") for key in rows[0] if key.startswith("rewards/")
    } - {"total_rewards", "termination"}
    missing = logged - set(weights)
    if missing:
        raise SystemExit(
            f"logged terms with no weight in the snapshot: {sorted(missing)}"
        )


def load(run: Path):
    rows = [json.loads(line) for line in (run / "vitals.jsonl").open()]
    if not rows:
        raise SystemExit(f"{run}/vitals.jsonl is empty")
    return rows


def draw(run: Path, rows, weights, output: Path):
    half = len(rows) // 2
    panels = [(name, weights[name]) for name in weights if f"rewards/{name}" in rows[0]]
    panels += [("total_rewards", None), ("episode_time", None)]

    columns = 3
    figure, axes = plt.subplots(
        -(-len(panels) // columns), columns, figsize=(14.0, 9.0)
    )
    for axis, (name, weight) in zip(axes.flat, panels):
        key = "episode_time" if name == "episode_time" else f"rewards/{name}"
        values = np.array([r[key] for r in rows]) / (weight or 1.0)
        early, late = values[:half], values[half:]

        lo, hi = values.min(), values.max()
        span = max(hi - lo, 1e-9)
        bins = np.linspace(lo - 0.02 * span, hi + 0.02 * span, 45)
        axis.hist(
            early,
            bins=bins,
            color="#4C72B0",
            alpha=0.65,
            label=f"first half  {early.mean():.3f}",
        )
        axis.hist(
            late,
            bins=bins,
            color="#DD8452",
            alpha=0.65,
            label=f"second half {late.mean():.3f}",
        )

        if weight is not None:
            axis.axvline(1.0, color="crimson", lw=1.3, ls="--")
            axis.set_title(f"{name}   (weight {weight:g}, ceiling 1.0)", fontsize=10)
        else:
            axis.set_title(name, fontsize=10)
        axis.legend(fontsize=7, loc="upper left")
        axis.tick_params(labelsize=8)
        axis.set_yticks([])

    for axis in axes.flat[len(panels) :]:
        axis.axis("off")

    figure.suptitle(
        f"{run.name}: reward terms against their own ceilings "
        f"({len(rows)} iterations, dashed red = 1.0)",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=150)
    print(f"wrote {output}")


def report(rows, weights):
    """Rank the terms by how much reward each is leaving on the table."""
    half = len(rows) // 2
    present = {n: w for n, w in weights.items() if f"rewards/{n}" in rows[0]}

    gaps = {}
    print(
        f"\n{'term':>19} {'weight':>7} {'mean':>7} {'1st half':>9} "
        f"{'2nd half':>9} {'shortfall':>10}"
    )
    for name, weight in present.items():
        values = np.array([r[f"rewards/{name}"] for r in rows]) / weight
        # Weighted, so shortfalls are comparable across terms and sum to the
        # gap between the achieved total and the ceiling.
        gaps[name] = (1.0 - values[half:].mean()) * weight
        early, late = values[:half].mean(), values[half:].mean()
        print(
            f"{name:>19} {weight:7.1f} {values.mean():7.3f} "
            f"{early:9.3f} {late:9.3f} {gaps[name]:10.3f}"
        )

    total_gap = sum(gaps.values())
    final = np.array([r["rewards/total_rewards"] for r in rows])[half:].mean()
    print(
        f"\n  ceiling {sum(present.values()):.1f}   "
        f"second-half total {final:.3f}   shortfall {total_gap:.3f}"
    )
    ranked = sorted(gaps.items(), key=lambda kv: -kv[1])
    shares = [f"{n} {g / total_gap:.0%}" for n, g in ranked if g / total_gap > 0.03]
    print("  where the headroom is: " + ", ".join(shares))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    weights = read_weights(args.run)
    rows = load(args.run)
    check_complete(rows, weights)
    report(rows, weights)
    draw(args.run, rows, weights, args.output)


if __name__ == "__main__":
    main()
