# Multi-agent MuJoCo Robotics Environment with PettingZoo Compatibility

MaMuJoCo-PettingZoo - Multi-agent MuJoCo robotics environment with Convenient Wrappers and Utilities. The origin environment can be found at https://github.com/Farama-Foundation/Gymnasium-Robotics.

## Installation

### PyPi from sources
```shell
pip install git+https://github.com/xihuai18/MaMuJoCo-PettingZoo.git
```

### Install from GitHub sources
```shell
git clone https://github.com/xihuai18/MaMuJoCo-PettingZoo.git
cd MaMuJoCo-PettingZoo
pip install -r requirements.txt
pip install .
```


## Test
```shell
python -c "from mamujoco_pettingzoo import mamujoco_pettingzoo_v1; env = mamujoco_pettingzoo_v1.parallel_env('Ant', '4x2'); print(env.reset(seed=42)); print(env.step({agent: env.action_space(agent).sample() for agent in env.agents}))"
```