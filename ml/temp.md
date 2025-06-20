## Introduction to Reinforcement Learning (RL)

Reinforcement Learning (RL) is a powerful paradigm in machine learning that allows an agent to learn optimal actions in an environment by receiving rewards or penalties. While not yet as widely commercially applied as supervised learning, it is a significant pillar of ML research with exciting potential.

### Core Idea: Learning by Trial and Error with Rewards

RL is analogous to training a dog: you don't explicitly tell the dog every single action it should take. Instead, you give it positive feedback ("good dog!") for desired behaviors and negative feedback ("bad dog!") for undesired ones. The dog then learns to maximize "good dog" outcomes and minimize "bad dog" ones.

* **In RL, this feedback is formalized as a "reward function."** The agent's goal is to learn to choose actions that maximize its cumulative reward over time.

### Example 1: Autonomous Helicopter Flight

* **Task:** Control a radio-controlled helicopter to fly stably and perform aerobatic maneuvers (e.g., upside-down flight).
* **Challenge for Supervised Learning:** It's difficult to get a dataset of "state (helicopter's position, orientation, speed) $\rightarrow$ ideal action (joystick movements)." The "correct" action is often ambiguous and depends on future outcomes.
* **RL Approach:**
    * **State (s):** The helicopter's current position, orientation, speed, etc.
    * **Action (a):** How to move the control sticks.
    * **Reward Function:**
        * $+1$ every second it flies well.
        * Negative reward for flying poorly.
        * Very large negative reward (e.g., $-1000$) for crashing.
* **Outcome:** The RL algorithm figures out on its own how to manipulate the controls to maximize positive rewards (flying well) and avoid negative rewards (crashing), even learning complex stunts.

### Example 2: Robotic Dog Obstacle Course

* **Task:** Train a robotic dog to navigate and overcome obstacles to reach a goal.
* **Challenge for Supervised Learning:** Extremely difficult to program precise leg placements and movements for varied obstacles.
* **RL Approach:** Reward the robot dog for making progress toward the goal (e.g., moving left across the screen).
* **Outcome:** The robot automatically learns complex locomotion and obstacle traversal strategies simply by maximizing rewards.

### Key Concept: Tell it *What* to Do, Not *How* to Do It

* **Supervised Learning:** You provide specific $(x, y)$ pairs, telling the algorithm *exactly what the correct output is* for a given input.
* **Reinforcement Learning:** You only specify a **reward function**, telling the algorithm *what goal to achieve* or *what constitutes good performance*. The algorithm's job is to figure out the optimal sequence of actions to achieve that goal. This offers much greater flexibility in system design for complex control tasks.

### Applications of Reinforcement Learning:

* **Robotics:** Controlling robots for locomotion, manipulation, and navigation (e.g., landing a lunar lander in simulation, as you'll do in the lab).
* **Factory Optimization:** Rearranging factory layouts or processes to maximize throughput and efficiency.
* **Financial Stock Trading:** Optimizing trade execution (e.g., selling a large block of shares over time without moving the market price against you).
* **Game Playing:** Achieving superhuman performance in games like checkers, chess, Go, and many video games.

RL's power lies in its ability to learn complex behaviors in environments where explicit instruction is impossible or impractical, simply by providing a clear reward signal. The next video will formalize the RL problem and begin developing algorithms for it.

## Reinforcement Learning Formalism: The Mars Rover Example

To formalize Reinforcement Learning (RL), we use a simplified example inspired by the Mars rover. This introduces the core components of an RL problem: **state, action, reward, and next state**.

### The Mars Rover Problem Setup:

* **States ($S$):** The rover can be in any of 6 discrete positions (boxes 1-6). Its current position is its state.
    * Example: Rover starts in State 4.
* **Actions ($A$):** At each step, the rover can choose one of two actions:
    * Go Left
    * Go Right
* **Rewards ($R(S)$):** Rewards are associated with specific states, indicating their value.
    * State 1 (Very interesting surface): $R(S_1) = 100$
    * State 6 (Interesting surface): $R(S_6) = 40$
    * States 2, 3, 4, 5 (Less interesting): $R(S_2)=R(S_3)=R(S_4)=R(S_5)=0$
* **Terminal States:** State 1 and State 6 are "terminal states." Once the rover reaches them, the "day ends," and it stops receiving further rewards.

### Agent-Environment Interaction (Sequence of Events):

At every time step, the agent (rover) is in some state $S$.

1.  **Observe State ($S$):** The rover perceives its current location.
2.  **Choose Action ($A$):** The rover selects an action (Left or Right).
3.  **Receive Reward ($R(S)$):** The rover receives the reward associated with the state it *just left* (or sometimes, the state it *just arrived at*, but here, it's defined as the reward *at* the state $S$).
4.  **Transition to New State ($S'$):** As a result of the action, the rover moves to a new state $S'$.

### Example Sequences of States, Actions, and Rewards:

<img src="/metadata/mars_rover.png" width="500" />

* **Path 1: Start $S_4$, Go Left (optimal choice)**
    * $S_4 \xrightarrow{\text{L R=0}} S_3 \xrightarrow{\text{L, R=0}} S_2 \xrightarrow{\text{L, R=0}} S_1 \xrightarrow{\text{T, R=100}}$
    * Rewards obtained: $0, 0, 0, 100$

* **Path 2: Start $S_4$, Go Right**
    * $S_4 \xrightarrow{\text{R, R=0}} S_5 \xrightarrow{\text{R, R=0}} S_6 \xrightarrow{\text{T, R=40}}$
    * Rewards obtained: $0, 0, 40$

* **Path 3: Start $S_4$, Go Right, then Left (suboptimal)**
    * $S_4 \xrightarrow{\text{R, R=0}} S_5 \xrightarrow{\text{L, R=0}} S_4 \xrightarrow{\text{L, R=0}} S_3 \xrightarrow{\text{L, R=0}} S_2 \xrightarrow{\text{L, R=0}} S_1 \xrightarrow{\text{T, R=100}}$
    * Rewards obtained: $0, 0, 0, 0, 0, 100$ (This path wastes time, but eventually reaches the high reward).

### Reinforcement Learning Goal:

The RL algorithm's job is to find a function (a "policy") that maps from the current state $S$ to an action $A$ that maximizes the **total future rewards**. This function is what the helicopter, robot dog, or Mars rover needs to learn to behave optimally.

The next video will further define what we want the RL algorithm to do by introducing the concept of **"return"**, which quantifies the total future rewards.
