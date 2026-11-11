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
    target_com_height = env_cfg.reward_settings.com_height_target
    target_gravity = torch.tensor(
        env_cfg.reward_settings.target_projected_gravity, device=env.device
    )
    print(f"\ntarget base height   {target_height:.3f} m")
    print(f"target CoM height    {target_com_height:.3f} m  (above the support anchor)")
    print(f"target proj. gravity {target_gravity.tolist()}")
    print(
        f"\n{'t [s]':>7} {'base h [m]':>12} {'CoM h [m]':>12} "
        f"{'CoM off [m]':>12} {'|L_xy| [-]':>12} {'|L_z| [-]':>11} "
        f"{'att err':>10} {'alive':>7}"
    )

    steps = int(args.seconds / env.dt)
    report_every = max(1, steps // 12)
    with torch.no_grad():
        for step in range(steps + 1):
            attitude = torch.norm(env.projected_gravity - target_gravity, dim=1)
            alive = (~env.terminated).float().mean().item()
            if step % report_every == 0:
                # The reduced-order state the rewards are built from. At step 0
                # the offset and the momentum must both read ~0 -- that is the
                # check that the support anchor and frame are right, not just
                # that the posture is. Under zero actions the momentum should
                # then grow monotonically as the robot tips.
                offset = torch.norm(env.com_offset, dim=1)
                momentum = torch.norm(env.support_momentum[:, :2], dim=1)
                # Yaw is uncosted by every reward -- gravity exerts no torque
                # about the vertical, so it is outside the reduced-order model
                # and projected_gravity cannot see a spin either. Logged so a
                # run tells us whether exploration finds it.
                yaw_momentum = env.support_momentum[:, 2].abs()
                print(
                    f"{step * env.dt:7.2f} "
                    f"{env.base_height.squeeze(1).mean().item():12.4f} "
                    f"{env.com_height.squeeze(1).mean().item():12.4f} "
                    f"{offset.mean().item():12.4f} "
                    f"{momentum.mean().item():12.4f} "
                    f"{yaw_momentum.mean().item():11.4f} "
                    f"{attitude.mean().item():10.3f} "
                    f"{alive:7.0%}"
                )
            if step == steps:
                break
            # Actions stay at zero: dof_pos_target is a residual, so the PD
            # controller is commanded to hold default_dof_pos exactly.
            env.step()

    print(
        f"\nstep 0 is the initialization check: base height should read "
        f"{target_height:.3f} m, attitude error ~0, and both the CoM offset "
        f"and |L_xy| ~0."
        f"\nThe CoM height at step 0 is the measurement that "
        f"reward_settings.com_height_target should be set from "
        f"(currently {target_com_height:.3f} m)."
    )


if __name__ == "__main__":
    main()
