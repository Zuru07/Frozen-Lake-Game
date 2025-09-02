# FrozenLake with Spiking Neural Networks (SNN) and Q-Learning  

This project demonstrates the integration of **Spiking Neural Networks (SNNs)** with **Q-Learning** to solve the **FrozenLake-v1** environment from OpenAI Gymnasium.  
The implementation leverages the **BindsNET** library for spiking neural networks and standard reinforcement learning techniques for training an agent.  

---

## Project Overview  

- Environment: **FrozenLake-v1** from Gymnasium  
- Learning Algorithm: **Q-Learning**  
- Neural Model: **Spiking Neural Network (SNN)** with STDP-based learning rule (`PostPre`)  
- Encoding: Grid states are spike-encoded as inputs to the SNN  
- Monitoring: Spike activity of hidden neurons is observed with `Monitor`  

The SNN is not directly used to compute Q-values but acts as a biologically inspired feature processor for learning state representations while Q-learning updates the action-value function.  

---

## Code Structure  

### 1. Environment Setup  
- Creates a **FrozenLake-v1** environment (`is_slippery=False` for deterministic transitions).  
- 16 discrete states, 4 possible actions.  

### 2. Spiking Neural Network  
- **Input Layer**: 64 input neurons (one-hot encoded for environment states).  
- **Hidden Layer**: 10 LIF (Leaky Integrate-and-Fire) neurons.  
- **Connections**: Synaptic weights initialized randomly with STDP update rule (`PostPre`).  
- **Monitoring**: Records hidden neuron spikes for analysis.  

### 3. Q-Learning Integration  
- Standard Q-learning with:  
  - Discount factor `gamma = 0.99`  
  - Learning rate `alpha = 0.1`  
  - Epsilon-greedy exploration (`epsilon` decays each episode).  
- Q-table: `q_table[states, actions]` with shape (16, 4).  
- Rewards are logged for convergence plotting.  

### 4. Training  
- Runs for 100 episodes.  
- Updates Q-values after each step.  
- Exploration rate decays to encourage exploitation.  
- Logs rewards per episode for convergence analysis.  

### 5. Testing  
- After training, the agent is tested with a **greedy policy** (`argmax` of Q-values).  
- Environment is rendered to visualize performance.  

### 6. Visualization  
- Plots total rewards across training episodes to observe learning convergence.  

---

## Installation  

### Dependencies  
Install the required Python libraries:  

```bash
    pip install gymnasium
    pip install bindsnet
    pip install torch
    pip install matplotlib
```

### How to Run

1. Clone this repository or copy the code.

2. Run the script in Python:

```bash
    python frozenlake_snn_qlearning.py
```

3. Observe:

- Training logs (epsilon decay and episode rewards).
- FrozenLake agent performance during testing.
- Reward convergence plot.

### Example Output:

- Training logs every 10 episodes:

```bash 
Episode 0: Epsilon = 0.9950, Reward = 0
Episode 10: Epsilon = 0.9053, Reward = 1
...
Training completed!
```

- Rendered FrozenLake environment showing the trained agent’s moves.
- Reward vs. Episode convergence plot.

### Notes and Future Improvements

The SNN currently serves as a feature encoder; integration with Q-learning can be further enhanced.

Future work:
- Use spike-based policies instead of Q-tables.
- Explore alternative STDP rules and larger hidden layers.
- Try different environments (e.g., CartPole, MountainCar).

### References
- https://gymnasium.farama.org
- https://bindsnet-docs.readthedocs.io
- https://en.wikipedia.org/wiki/Q-learning