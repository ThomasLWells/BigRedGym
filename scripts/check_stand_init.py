"""Check that Go2Stand's reset really puts the robot in the handstand.

No policy and no training: actions are left at zero, so the PD controller
holds exactly the posture the reset produced and the only thing acting on the
robot is physics. That isolates the initialization from the learning.

    uv run --frozen scripts/check_stand_init.py                 # headless
    uv run --frozen scripts/check_stand_init.py --no-headless   # with viewer

What to expect: the pose is a static equilibrium (the centre of mass sits over
the front-foot line) but an unstable one -- the support polygon is the segment
between two feet, so in pitch it is an inverted pendulum. A correct
initialization therefore holds for a while and then diverges smoothly. An
immediate collapse on step 1, or a height/attitude that starts far from the
configured target, is an initialization bug rather than physics.
"""

import argparse

import torch

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="go2stand")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    import gym.envs  # noqa: F401  (populates the task registry)

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.seed = train_cfg.seed = args.seed
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env(
        args.task,
        env_cfg,
        device=args.device,
        headless=args.headless,
        backend="mujoco",
    )

    target_height = env_cfg.reward_settings.base_height_target
    target_gravity = torch.tensor(
        env_cfg.reward_settings.target_projected_gravity, device=env.device
    )
    print(f"\ntarget base height   {target_height:.3f} m")
    print(f"target proj. gravity {target_gravity.tolist()}")
    print(f"\n{'t [s]':>7} {'height [m]':>18} {'attitude err [rad]':>20} {'alive':>7}")

    steps = int(args.seconds / env.dt)
    report_every = max(1, steps // 12)
    with torch.no_grad():
        for step in range(steps + 1):
            height = env.base_height.squeeze(1)
            attitude = torch.norm(env.projected_gravity - target_gravity, dim=1)
            alive = (~env.terminated).float().mean().item()
            if step % report_every == 0:
                print(
                    f"{step * env.dt:7.2f} "
                    f"{height.mean().item():9.3f} +/- {height.std().item():.3f} "
                    f"{attitude.mean().item():12.3f} +/- {attitude.std().item():.3f} "
                    f"{alive:7.0%}"
                )
            if step == steps:
                break
            # Actions stay at zero: dof_pos_target is a residual, so the PD
            # controller is commanded to hold default_dof_pos exactly.
            env.step()

    print(
        f"\nstep 0 is the initialization check: height should read "
        f"{target_height:.3f} m and attitude error ~0."
    )


if __name__ == "__main__":
    main()
