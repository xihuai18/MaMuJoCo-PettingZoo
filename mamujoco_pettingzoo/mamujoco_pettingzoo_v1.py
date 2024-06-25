import gymnasium as gym
from gymnasium_robotics import mamujoco_v1
from gymnasium_robotics.envs.multiagent_mujoco.obsk import get_parts_and_edges


def parallel_env(
    scenario: str,
    agent_conf: str | None,
    agent_obsk: int | None = 1,
    agent_factorization: dict[str, any] | None = None,
    local_categories: list[list[str]] | None = None,
    global_categories: tuple[str, ...] | None = None,
    render_mode: str | None = None,
    gym_env: gym.envs.mujoco.mujoco_env.MujocoEnv | None = None,
    **kwargs,
):
    env = mamujoco_v1.parallel_env(
        scenario=scenario,
        agent_conf=agent_conf,
        agent_obsk=agent_obsk,
        agent_factorization=agent_factorization,
        local_categories=local_categories,
        global_categories=global_categories,
        render_mode=render_mode,
        gym_env=gym_env,
        **kwargs,
    )
    if hasattr(env, "state_space"):
        from co_mas.wrappers import AgentStateParallelEnvWrapper

        env = AgentStateParallelEnvWrapper(env)

    aec_env = pettingzoo.utils.parallel_to_aec(env)
    aec_env = pettingzoo.utils.OrderEnforcingWrapper(aec_env)
    env = pettingzoo.utils.aec_to_parallel(aec_env)
    return env


__all__ = ["parallel_env", "get_parts_and_edges"]
