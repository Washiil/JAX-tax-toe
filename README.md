# Vectorized Tic-Tac-Toe RL with JAX & Flax NNX

A high-performance Reinforcement Learning pipeline for Tic-Tac-Toe implemented entirely in JAX and Flax NNX. This project leverages policy gradient methods (REINFORCE) and self-play to train an agent to play perfect Tic-Tac-Toe by simulating thousands of games completely in parallel using pure functional transformations.

With a single run of this notebook not only will you understand how JAX operates and why its worth your time but you will also lose at Tic-Tac-Toe!


```
Please input a move (0-8): 4
-------------
| . | . | . |
-------------
| . | X | . |
-------------
| . | . | . |
-------------
Turn count: 1

Model is thinking...
-------------
| . | . | . |
-------------
| . | X | . |
-------------
| . | . | O |
-------------
```

## Why JAX for Reinforcement Learning?
Traditionally, RL environments (like OpenAI Gym/Farama Gymnasium) run sequentially on CPUs because game logic involves conditional branching. This creates a massive data-transfer bottleneck between the CPU and GPU.

- The entire Tic-Tac-Toe game mechanics (state, step function, invalid move masking, and win conditions) are written using array transformations and functional primitives to remaine pure and JAX compile-able.

- Instead of simulating games one by one, `jax.vmap` vectorizes the execution across an entire axis. We simulate independent games concurrently in a single batch.

- End-to-End JIT Compilation (`@nnx.jit`): The entire simulation, policy sampling, history gathering, and gradient update step are fused into a single highly optimized machine-code loop.

### Performance on CPU vs. GPU

Thanks to JAX, it runs seamlessly across varying hardware backends. I was able to test in various enviroments via Google Colab.

This "write once run anywhere" style made both development and "deployment" very easy allowing me to test and write code locally and deploy to a T4 GPU with a single click!

## Future Enhancements
While this project focuses perfectly on solving Tic-Tac-Toe, the architecture is intentionally generic. This general structure solves many RL problems and I hope to solve more in the future:

- Deep Q-Networks: Transition from a policy gradient method to a Value-based method by incorporating an experience replay buffer; can also be vectorized in JAX memory.

- Actor-Critic: Scale the architecture by having the network output both action probabilities and state-value estimations to stabilize gradient variances.

- Scaling to Connect-Four or Chess: Expand the state tensor and swap out the basic Multi-Layer Perceptron for a Convolutional Neural Network or a small Transformer block to handle bigger spatial dimensions.
