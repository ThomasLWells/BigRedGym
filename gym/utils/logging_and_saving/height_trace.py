import matplotlib

matplotlib.use("Agg")  # see step 4 — never open a second GUI toolkit
import matplotlib.pyplot as plt
import numpy as np


class HeightTrace:
    """Commanded vs measured base height for one environment, over time."""

    def __init__(self, env, env_index=0):
        self.env = env
        self.env_index = env_index
        self.commanded = []
        self.measured = []

    def record(self):
        i = self.env_index
        self.commanded.append(self.env.commands[i, 3].item())
        self.measured.append(self.env.base_height[i, 0].item())

    def save(self, path="height_trace.png"):
        if not self.measured:
            return

        t = np.arange(len(self.measured)) * self.env.dt
        commanded = np.array(self.commanded)
        measured = np.array(self.measured)

        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        # The command is piecewise-constant — it only moves on an arrow press —
        # so draw it as steps, not as an interpolated line.
        ax.plot(
            t,
            commanded,
            color="#2a78d6",
            lw=2,
            drawstyle="steps-post",
            label="command",
        )
        ax.plot(t, measured, color="#eb6834", lw=2, label="actual")

        ax.set_xlabel("time (s)", color="#52514e")
        ax.set_ylabel("base height (m)", color="#52514e")
        ax.set_xlim(t[0], t[-1])
        ax.grid(True, lw=0.5, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="best")

        fig.tight_layout()
        fig.savefig(path)
        np.savetxt(
            path.replace(".png", ".csv"),
            np.column_stack([t, commanded, measured]),
            delimiter=",",
            header="t_s,command_m,actual_m",
            comments="",
        )
