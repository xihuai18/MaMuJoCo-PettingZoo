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
    order_forcing: bool = True,
    agent_state: bool = True,
    additional_wrappers: List[pettingzoo.utils.BaseParallelWrapper] = [],
    **kwargs,
) -> MultiAgentMujocoEnv:
    """
    Parameters
    ----------
    order_forcing : bool
        Whether to use the OrderForcingParallelEnvWrapper.
    agent_state : bool
        Whether to use the AgentStateParallelEnvWrapper, if possible.
    """
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
    env.observation_spaces = gym.spaces.Dict(env.observation_spaces)
    env.action_spaces = gym.spaces.Dict(env.action_spaces)
    if agent_state and hasattr(env, "state_space") and hasattr(env, "state"):
        from co_mas.wrappers import AgentStateParallelEnvWrapper

        if not any(issubclass(wrapper, AgentStateParallelEnvWrapper) for wrapper in additional_wrappers):

            env = AgentStateParallelEnvWrapper(env)

    from co_mas.wrappers import OrderForcingParallelEnvWrapper

    if order_forcing and not any(
        issubclass(wrapper, OrderForcingParallelEnvWrapper) for wrapper in additional_wrappers
    ):
        env = OrderForcingParallelEnvWrapper(env)

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
