# Optimizing Microrobot Tasks using Reinforcement Learning

This repository showcases my implementation of a simulator and reinforcement algorithm training environment for a micro-robot completing a task in a low-Reynolds fluid environment.

The simulated micro-robot was trained to perform a cell sorting task, using state-of-the-art reinforcement learning advancements, such as experience replay [1], a separate target network [2], a double deep q network [3], and a dueling deep q network [4].

## Libraries
* Python 3.5
* Tensorflow 1.x
* Numpy
* Bokeh
* Tkinter
* Tornado

## References

[1] Schaul, Tom, et al. "Prioritized experience replay." _arXiv preprint arXiv:1511.05952_ (2015).

[2] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." _Nature_ 518.7540 (2015): 529-533. 

[3] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep Reinforcement Learning with Double Q-Learning." _AAAI_. 2016.

[4] Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." _arXiv preprint arXiv:1511.06581_ (2015).
