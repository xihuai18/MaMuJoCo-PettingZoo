from typing import List

import gymnasium as gym
import pettingzoo
from gymnasium_robotics import mamujoco_v1
from gymnasium_robotics.envs.multiagent_mujoco import MultiAgentMujocoEnv
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
    additional_wrappers: List[pettingzoo.utils.BaseParallelWrapper] = [],
    **kwargs,
) -> MultiAgentMujocoEnv:
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
    env.state_space = env.unwrapped.single_agent_env.observation_space
    if hasattr(env, "state_space") and hasattr(env, "state"):
        from co_mas.wrappers import AgentStateParallelEnvWrapper

        env = AgentStateParallelEnvWrapper(env)

    for wrapper in additional_wrappers:
        if issubclass(wrapper, pettingzoo.utils.BaseParallelWrapper):
            env = wrapper(env)
        elif issubclass(wrapper, pettingzoo.utils.BaseWrapper):
            from co_mas.wrappers import AECToParallelWrapper, ParallelToAECWrapper

            aec_env = ParallelToAECWrapper(env)
            aec_env = wrapper(aec_env)
            env = AECToParallelWrapper(aec_env)
        else:
            raise ValueError(f"Unknown wrapper type: {wrapper}")

    return env


__all__ = ["parallel_env", "get_parts_and_edges"]
