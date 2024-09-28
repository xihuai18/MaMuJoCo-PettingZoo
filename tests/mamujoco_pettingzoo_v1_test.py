from co_mas.test.parallel_api import parallel_api_test, sample_action
from loguru import logger

from mamujoco_pettingzoo import mamujoco_pettingzoo_v1

# API Tests

env = mamujoco_pettingzoo_v1.parallel_env("Ant", "4x2", None)
parallel_api_test(env, 400, True)

env1 = mamujoco_pettingzoo_v1.parallel_env("Ant", "4x2", None)
logger.info(f"agents:\n{env1.agents}")
logger.info(f"action_partitions:\n{env1.agent_action_partitions}")

# Seed Tests
obs1_list = []
obs1, info1 = env1.reset(seed=42)
obs1_list.append(obs1)

while True:
    obs1, _, terminated1, _, info1 = env1.step(
        {agent: sample_action(agent, obs1[agent], info1[agent], env1.action_space(agent)) for agent in env1.agents}
    )
    obs1_list.append(obs1)

    if any(terminated1.values()):
        break

env1.close()

env2 = mamujoco_pettingzoo_v1.parallel_env("Ant", "4x2", None)

obs2_list = []
obs2, info2 = env2.reset(seed=42)
obs2_list.append(obs2)

while True:
    obs2, _, terminated2, _, info2 = env2.step(
        {agent: sample_action(agent, obs2[agent], info2[agent], env2.action_space(agent)) for agent in env2.agents}
    )
    obs2_list.append(obs2)

    if any(terminated2.values()):
        break

env2.close()

for i, (obs1, obs2) in enumerate(zip(obs1_list, obs2_list)):
    assert all(obs1[agent] == obs2[agent] for agent in env1.agents), f"Observations at step {i} differ: {obs1} {obs2}"

logger.success("Seed test passed!")

# Wrapper Tests
from co_mas.wrappers import AutoResetParallelEnvWrapper, OrderForcingParallelEnvWrapper

env = mamujoco_pettingzoo_v1.parallel_env(
    "Ant",
    "4x2",
    additional_wrappers=[OrderForcingParallelEnvWrapper],
)

try:
    env.step({agent: env.action_space(agent).sample() for agent in env.agents})
except Exception as e:
    assert (
        str(e) == "Environment must be reset before stepping"
        or str(e) == "Cannot call env.step() before calling env.reset()"
    ), e
    logger.success("Order Forcing Test Passed!")
else:
    raise AssertionError("Expected reset error")


env = mamujoco_pettingzoo_v1.parallel_env(
    "Ant",
    "4x2",
    additional_wrappers=[OrderForcingParallelEnvWrapper, AutoResetParallelEnvWrapper],
)

obs, info = env.reset(seed=42)

while True:
    obs, _, terminated, _, info = env.step(
        {agent: sample_action(agent, obs[agent], info[agent], env.action_space(agent)) for agent in env.agents}
    )

    if all(terminated.values()):
        break

obs, _, terminated, _, info = env.step(
    {agent: sample_action(agent, obs[agent], info[agent], env.action_space(agent)) for agent in env.agents}
)

assert terminated != {agent: True for agent in env.agents}

logger.success("Auto Reset Test Passed!")

logger.success("Wrapper Test Passed!")
env.close()
