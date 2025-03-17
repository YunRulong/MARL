# StarCraft

Pytorch implementations of the multi-agent reinforcement learning algorithms, including 
[IQL](https://arxiv.org/abs/1511.08779),
[QMIX](https://arxiv.org/abs/1803.11485), [VDN](https://arxiv.org/abs/1706.05296), 
[COMA](https://arxiv.org/abs/1705.08926), [QTRAN](https://arxiv.org/abs/1905.05408)(both **QTRAN-base** and **QTRAN-alt**),
[MAVEN](https://arxiv.org/abs/1910.07483), [CommNet](https://arxiv.org/abs/1605.07736), 
[DyMA-CL](https://arxiv.org/abs/1909.02790?context=cs.MA), and [G2ANet](https://arxiv.org/abs/1911.10715), 
which are the state of the art MARL algorithms. In addition, because CommNet and G2ANet need an external training algorithm, 
we provide **Central-V** and **REINFORCE** for them to training, you can also combine them with COMA.
We trained these algorithms on [SMAC](https://github.com/oxwhirl/smac), the decentralised micromanagement scenario of [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty).

## Corresponding Papers
- [IQL: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736)
- [From Few to More: Large-scale Dynamic Multiagent Curriculum Learning](https://arxiv.org/abs/1909.02790?context=cs.MA)
- [Multi-Agent Game Abstraction via Graph Attention Neural Network](https://arxiv.org/abs/1911.10715)
- [MAVEN: Multi-Agent Variational Exploration](https://arxiv.org/abs/1910.07483)

## Requirements
Use `pip install -r requirements.txt` to install the following requirements:

- matplotlib
- torch
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)
+ [pymarl](https://github.com/oxwhirl/pymarl)

## Quick Start

```shell
$ python main.py --map=3m --alg=qmix
```

Directly run the `main.py`, then the algorithm will start **training** on map `3m`. **Note** CommNet and G2ANet need an external training algorithm, so the name of them are like `reinforce+commnet` or `central_v+g2anet`, all the algorithms we provide are written in `./common/arguments.py`.

If you just want to use this project for demonstration, you should set `--evaluate=True --load_model=True`. 

The running of DyMA-CL is independent from others because it requires different environment settings, so we put it on another project. For more details, please read [DyMA-CL documentation](https://github.com/starry-sky6688/DyMA-CL).

## Replay

If you want to see the replay, make sure the `replay_dir` is an absolute path, which can be set in `./common/arguments.py`. Then the replays of each evaluation will be saved, you can find them in your path.

## 环境
多智能体环境使用自行编写的environment环境，存放于environment文件夹下，目前仅实现了多导弹对抗的部分，没有飞机。另外当前项目并没有完全查完错误，环境接口可以参考env的返回函数

## environment 文件夹介绍
    文件夹内共有两个文件一个是作为environment环境的主程序，另外一个作为定义导弹的类型的辅助程序。
    环境主程序中主要函数有init，reset，step，get_state,get_obs,这几个主要函数，init用于在程序初始化环境时将args中的参
    数传入env对象，reset函数用于在强化学习的每一个episode开始时重新将双方的智能体归位，step函数用于智能体模拟推演的迭
    代，get_state与get_obs的作用类似，都是在强化学习中用于返回状态的函数，前者用于返回全局状态，后者用于返回智能体的观测
    状态。
    
    具体的调用流程是，训练开始-》调用init，训练共重复n个episode，每一个episode开始时-》调用reset，在单个episode内进行推
    演时，每一步都需要调用观测的get_obs和get_state函数，用于得到当前状态，根据当前状态网络输出动作决策，传入step函数进行
    一次动作。相关函数的作用已经注释在environment.py里

    missile文件内已经写好了基础导弹的object类，如果要扩展导弹可以通过集成object类来实现，具体的导弹不同的特性可以在自己的step函数中定义，guidance类用于实现比例导引，tool类用于missile类内的一些常用的归一化方法。