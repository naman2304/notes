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
3.  **Receive Reward** ($R(S)$): The rover receives the reward associated with the state it *just left* (or sometimes, the state it *just arrived at*, but here, it's defined as the reward *at* the state $S$).
4.  **Transition to New State ($S'$):** As a result of the action, the rover moves to a new state $S'$.

### Example Sequences of States, Actions, and Rewards:

<img src="/metadata/mars_rover.png" width="500" />

* **Path 1: Start $S_4$, Go Left (optimal choice)**
    * $S_4 \xrightarrow{\text{L R=0}} S_3 \xrightarrow{\text{L R=0}} S_2 \xrightarrow{\text{L R=0}} S_1 \xrightarrow{\text{T R=100}}$
    * Rewards obtained: $0, 0, 0, 100$

* **Path 2: Start $S_4$, Go Right**
    * $S_4 \xrightarrow{\text{R R=0}} S_5 \xrightarrow{\text{R R=0}} S_6 \xrightarrow{\text{T R=40}}$
    * Rewards obtained: $0, 0, 40$

* **Path 3: Start $S_4$, Go Right, then Left (suboptimal)**
    * $S_4 \xrightarrow{\text{R R=0}} S_5 \xrightarrow{\text{L R=0}} S_4 \xrightarrow{\text{L R=0}} S_3 \xrightarrow{\text{L R=0}} S_2 \xrightarrow{\text{L R=0}} S_1 \xrightarrow{\text{T R=100}}$
    * Rewards obtained: $0, 0, 0, 0, 0, 100$ (This path wastes time, but eventually reaches the high reward).

### Reinforcement Learning Goal:

The RL algorithm's job is to find a function (a "policy") that maps from the current state $S$ to an action $A$ that maximizes the **total future rewards**. This function is what the helicopter, robot dog, or Mars rover needs to learn to behave optimally.

The next video will further define what we want the RL algorithm to do by introducing the concept of **"return"**, which quantifies the total future rewards.

## Reinforcement Learning: The Return

In Reinforcement Learning (RL), the goal is to maximize **total future rewards**. However, rewards received sooner are generally preferred over rewards received later. This concept is captured by the **Return**, which is a discounted sum of future rewards.

### The Discount Factor ($\gamma$ - Gamma)

* **Purpose:** The discount factor ($\gamma$), a number between 0 and 1 (typically close to 1, e.g., 0.9, 0.99), makes the RL agent "impatient." It reduces the value of future rewards.
* **Interpretation:**
    * A reward received one step in the future is multiplied by $\gamma$.
    * A reward received two steps in the future is multiplied by $\gamma^2$.
    * A reward received $t$ steps in the future is multiplied by $\gamma^t$.
* **Benefit:** Getting rewards sooner results in a higher total return. This encourages the agent to achieve goals more quickly.
* **Financial Analogy:** $\gamma$ is like an interest rate or the time value of money. A dollar today is worth more than a dollar in the future.

### Formula for the Return ($G$)

If an agent goes through a sequence of states and receives rewards $R_1, R_2, R_3, \dots$ at each step (where $R_t$ is the reward received at time step $t$):

$$G = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 + \dots$$
This sum continues until a terminal state is reached.

### Example: Mars Rover (with $\gamma = 0.5$)

Let's revisit the Mars Rover example with states $S_1, \dots, S_6$ and rewards $R(S_1)=100, R(S_6)=40$, and $R(\text{other states})=0$.

* **Path: $S_4 \rightarrow S_3 \rightarrow S_2 \rightarrow S_1$**
    * Rewards: $R(S_4)=0, R(S_3)=0, R(S_2)=0, R(S_1)=100$
    * Return ($G$ from $S_4$): $0 + (0.5 \times 0) + (0.5^2 \times 0) + (0.5^3 \times 100) = 0 + 0 + 0 + (0.125 \times 100) = 12.5$

* **Path: $S_4 \rightarrow S_5 \rightarrow S_6$**
    * Rewards: $R(S_4)=0, R(S_5)=0, R(S_6)=40$
    * Return ($G$ from $S_4$): $0 + (0.5 \times 0) + (0.5^2 \times 40) = 0 + 0 + (0.25 \times 40) = 10$

Comparing these two paths, "Go Left" (Return 12.5) yields a higher total discounted reward than "Go Right" (Return 10), suggesting "Go Left" might be a better overall strategy.

### Impact of Negative Rewards:

The discount factor also influences how negative rewards are handled:
* If a negative reward occurs far in the future (multiplied by a small $\gamma^t$), its negative impact on the total return is reduced.
* This incentivizes the agent to **postpone negative rewards** as much as possible, which is often desirable in real-world applications (e.g., delaying a payment).

The next video will formalize the goal of an RL algorithm: to maximize this return.

## Reinforcement Learning: The Policy

This video formalizes how a reinforcement learning (RL) algorithm chooses actions by introducing the concept of a **policy ($\pi$)**.

### The Policy ($\pi$)

* **Definition:** A policy, denoted as $\pi$, is a **function** that maps any given **state ($s$)** to an **action ($a$)** that the RL agent should take in that state.
    * Think of it as the agent's strategy or behavior.
    * Notation: $\pi(s) = a$.
* **Goal of Reinforcement Learning:** The ultimate goal of an RL algorithm is to **find an optimal policy $\pi^*$** that tells the agent what action to take in *every possible state* to **maximize the expected cumulative return** (the total discounted sum of future rewards).

### Examples of Policies in the Mars Rover Context:

Different rules for choosing actions based on the rover's state are different policies:

1.  **"Always go for the nearer reward":** A policy where the rover decides to go Left if State 1 is closer, or Right if State 6 is closer.
2.  **"Always go for the larger reward":** A policy that might lead the rover to prioritize reaching State 1 (reward 100) even if it's further away, over State 6 (reward 40).
3.  **"Go Left unless one step away from lesser reward (State 6)":**
    * If in State 2, $\pi(S_2) = \text{Left}$.
    * If in State 3, $\pi(S_3) = \text{Left}$.
    * If in State 4, $\pi(S_4) = \text{Left}$.
    * If in State 5, $\pi(S_5) = \text{Right}$.

### Core Components of RL (Review):

* **State ($S$):** The current situation or configuration of the agent and its environment.
* **Action ($A$):** The choices the agent can make in a given state.
* **Reward ($R(S)$):** Immediate feedback (positive or negative) received from the environment for being in a particular state or taking a particular action.
* **Return ($G$):** The total discounted sum of future rewards that the agent expects to receive from a given state onwards. It captures the long-term value of a state-action sequence.
* **Policy ($\pi$):** The function that dictates the agent's behavior, mapping states to actions.

The next video will provide a quick review of these concepts before moving on to algorithms for finding optimal policies.

## Reinforcement Learning Formalism: Review and Generalization

This video reviews the core concepts of Reinforcement Learning (RL) using the Mars Rover example and generalizes them to other applications, introducing the formal term **Markov Decision Process (MDP)**.

### Core RL Concepts (Review):

* **State ($S$):** The complete description of the environment at a given time. (e.g., Rover's position, Helicopter's position/orientation/speed, Chess board configuration).
* **Action ($A$):** The choice an agent makes in a given state. (e.g., Go Left/Right, Move control sticks, Legal chess move).
* **Reward ($R(S)$):** Immediate feedback from the environment associated with a state. (e.g., 100 at State 1 for Mars Rover, +1 for winning a chess game, -1000 for a helicopter crash).
* **Discount Factor ($\gamma$ - Gamma):** A value between 0 and 1 (e.g., 0.5, 0.99) that discounts future rewards, making the agent "impatient" for sooner rewards.
* **Return ($G$):** The total discounted sum of future rewards received from a given state onwards: $G = R_1 + \gamma R_2 + \gamma^2 R_3 + \dots$.
* **Policy ($\pi$):** A function that maps a state ($S$) to an action ($A$), dictating the agent's behavior ($\pi(S) = A$). The goal of RL is to find the optimal policy.

### Generalization to Other Applications:

| Concept         | Mars Rover Example                                | Autonomous Helicopter                               | Chess Game Playing                                 |
| :-------------- | :------------------------------------------------ | :-------------------------------------------------- | :------------------------------------------------- |
| **State ($S$)** | 6 discrete positions                             | Position, orientation, speed                        | Position of all pieces on the board                |
| **Action ($A$)** | Go Left, Go Right                                | Move control sticks                                 | Possible legal moves                               |
| **Rewards ($R(S)$)** | 100 (State 1), 40 (State 6), 0 (others)         | +1 (flying well), -1000 (crash)                   | +1 (win), -1 (lose), 0 (tie)                       |
| **Discount Factor ($\gamma$)** | 0.5                                            | 0.99 (common)                                       | 0.99, 0.995, 0.999 (very close to 1)               |
| **Return ($G$)** | Sum of $\gamma^t R_t$                             | Sum of $\gamma^t R_t$                               | Sum of $\gamma^t R_t$                              |
| **Policy ($\pi$)** | $\pi(S)$ maps state to Left/Right              | $\pi(S)$ maps position to control stick movements | $\pi(S)$ maps board position to next legal move    |

### Markov Decision Process (MDP):

* The formalism encompassing states, actions, rewards, return, and policy is called a **Markov Decision Process (MDP)**.
* **"Markov" Property:** The future depends *only* on the current state and *not* on the sequence of events that led to that state ("the future depends only on where you are now, not on how you got here").
* **Agent-Environment Loop:**
    * Agent (Robot/Controller) is in State $S$.
    * Agent chooses Action $A$ (based on Policy $\pi$).
    * Environment reacts: Agent transitions to new State $S'$.
    * Agent receives Reward $R(S')$.
    * Loop continues.

The next step in developing an RL algorithm is to define and learn the **state-action value function**, a key quantity for finding optimal policies.

## Reinforcement Learning: The State-Action Value Function (Q-Function)

This video introduces the **state-action value function**, typically denoted as $Q(s, a)$, which is a key quantity that reinforcement learning algorithms aim to compute.

### Definition of the Q-Function:

* **$Q(s, a)$:** Represents the **expected cumulative return** (total discounted future rewards) if you start in state $s$, take action $a$ *once*, and then behave **optimally** thereafter (i.e., take actions that maximize future returns from that point).

* **Initial "Circular" Feeling:** The definition seems circular because "optimally" implies knowing the best policy already. However, specific RL algorithms (to be discussed later) can compute $Q(s,a)$ without prior knowledge of the optimal policy.

### Example: Mars Rover with Optimal Policy ($\gamma = 0.5$)

Let's assume the optimal policy is: Go Left from $S_2, S_3, S_4$, and Go Right from $S_5$.

* **Calculating $Q(S_2, \text{Right})$:**
    * Start at $S_2$, take action Right.
    * Sequence: $S_2 \xrightarrow{\text{R R=0}} S_3 \xrightarrow{\text{Optimal (Left), R=0}}        S_2         \xrightarrow{\text{Optimal (Left), R=0}} S_1 \xrightarrow{\text{T R=100}}$
    * Rewards: $0, 0, 0, 100$
    * Return: $0 + (0.5 \times 0) + (0.5^2 \times 0) + (0.5^3 \times 100) = 12.5$
    * So, $Q(S_2, \text{Right}) = 12.5$. (This value simply reports the outcome of that specific initial action followed by optimal play).

* **Calculating $Q(S_2, \text{Left})$:**
    * Start at $S_2$, take action Left.
    * Sequence: $S_2 \xrightarrow{\text{L R=0}} S_1 \xrightarrow{\text{T, R=100}}$
    * Rewards: $0, 100$
    * Return: $0 + (0.5 \times 100) = 50$
    * So, $Q(S_2, \text{Left}) = 50$.

### The Q-Function Table:

By calculating $Q(s,a)$ for all states $s$ and all actions $a$, you get a table (or function) like this:

| State ($s$) | Action: Left | Action: Right |
| :---------- | :----------- | :------------ |
| $S_1$       | 100          | 100           |
| $S_2$       | 50           | 12.5          |
| $S_3$       | 25           | 6.25          |
| $S_4$       | 12.5         | 10            |
| $S_5$       | 6.25         | 20            |
| $S_6$       | 40           | 40            |

(Note: For terminal states like $S_1$ and $S_6$, $Q(s,a)$ is simply the terminal reward, as no further actions are taken).

### Using the Q-Function to Determine Optimal Policy:

Once you have the $Q(s,a)$ values for all states and actions, finding the optimal policy is straightforward:

* **Optimal Policy** (π*(s)): For any given state $s$, the optimal action $a$ is the one that **maximizes** $Q(s,a)$.
    π*(s) = $$\underset{a}{{argmax}} Q(s, a)$$
    * **Intuition:** $Q(s,a)$ tells you the maximum return you can get by starting in $s$, taking $a$, and playing optimally afterward. To get the best overall return from state $s$, you should simply pick the action $a$ that leads to this maximum value.
    * **Example (from table):**
        * In $S_2$: $Q(S_2, \text{Left}) = 50$, $Q(S_2, \text{Right}) = 12.5$. Max is 50, so $\pi^*(S_2) = \text{Left}$.
        * In $S_5$: $Q(S_5, \text{Left}) = 6.25$, $Q(S_5, \text{Right}) = 20$. Max is 20, so $\pi^*(S_5) = \text{Right}$.

In summary, the Q-function provides a comprehensive "map" of the value of taking any action from any state, assuming subsequent optimal play. Once computed, it directly yields the optimal policy. The next video will likely focus on algorithms to compute this Q-function.

## Exploring the State-Action Value Function (Q-Function) with the Mars Rover

This video introduces an optional Jupyter Notebook lab that allows you to interactively explore how changes to the Mars Rover example's parameters affect the Q-function values and the optimal policy. The goal is to build intuition about the relationship between rewards, the discount factor, and optimal decision-making in Reinforcement Learning.

### Lab Setup:

* The lab uses helper functions to compute and visualize the optimal policy and Q-function values.
* **Default Parameters:**
    * Number of states: 6
    * Number of actions: 2 (Left/Right)
    * Terminal Rewards: State 1 = 100, State 6 = 40. Intermediate states = 0.
    * Discount Factor ($\gamma$): 0.5
    * Misstep Probability: 0 (ignored for now, implying deterministic actions).
* The initial output of $Q(s,a)$ will match the values discussed in the previous lecture.

### Interactive Exploration (Key Changes to Observe):

The lab encourages you to modify parameters and observe the resulting changes in $Q(s,a)$ and the optimal policy:

1.  **Changing Terminal Right Reward:**
    * **Example:** Update terminal reward at State 6 from 40 to 10.
    * **Observation:** $Q(s, \text{Right})$ values for states leading to State 6 will decrease significantly. The optimal policy will likely shift to strongly favor paths leading to State 1, even if they are longer. For instance, from $S_5$, the optimal policy might switch to "Go Left" to eventually reach $S_1$, rather than "Go Right" to reach the now low-reward $S_6$.

2.  **Changing Discount Factor ($\gamma$):**
    * **Increase $\gamma$ (e.g., from 0.5 to 0.9):**
        * **Effect:** The rover becomes **less impatient**. Rewards in the future are discounted less severely.
        * **Observation:** The agent becomes more willing to take longer paths to reach larger, more distant rewards. The $Q(s,a)$ values will generally increase for all paths that reach a reward. The optimal policy might shift to prefer longer paths to the 100-reward State 1, even from states closer to State 6.
    * **Decrease $\gamma$ (e.g., from 0.5 to 0.3):**
        * **Effect:** The rover becomes **extremely impatient**. Future rewards are heavily discounted.
        * **Observation:** The agent will prioritize immediate or very near rewards, even if they are smaller. The $Q(s,a)$ values will decrease significantly for longer paths. The optimal policy might shift to prefer the smaller 40-reward State 6 if it's closer, as the 100-reward State 1 is too heavily discounted by the time it's reached.

### Learning from the Lab:

By playing with the reward function and the discount factor $\gamma$, you can observe:
* How $Q(s,a)$ values change in response to these parameters.
* How the optimal return (the maximum $Q(s,a)$ from each state) changes.
* How the optimal policy adapts to prioritize either larger, distant rewards (higher $\gamma$) or smaller, immediate rewards (lower $\gamma$).

This hands-on exploration will strengthen your intuition about the fundamental trade-offs and decision-making processes in Reinforcement Learning. After completing the lab, you will be ready to discuss the **Bellman Equation**, a central concept in RL.

## Reinforcement Learning: The Bellman Equation

The **Bellman Equation** is a fundamental equation in Reinforcement Learning that allows us to compute the state-action value function, $Q(s,a)$. It breaks down the total return into an immediate reward and the discounted future optimal return from the next state.

### Notation:

* $S$: Current state
* $A$: Current action taken from state $S$
* $R(S)$: Reward received for being in state $S$
* $S'$: Next state reached after taking action $A$ from state $S$
* $A'$: Action taken from the next state $S'$
* $\gamma$ (Gamma): Discount factor (e.g., 0.5)

### The Bellman Equation:

The core Bellman equation states:
$$Q(S,A) = R(S) + \gamma \max_{A'} Q(S', A')$$

Simplisticly, this can be solved using dynamic programming. But for small problems too, the state space so large that solving this via DP becomes infeasible. And hence, we resort to Deep RL techniques (DQN algorithm for instance).

* **Intuition:**
    * $Q(S,A)$ is the total return if you start in state $S$, take action $A$, and then act optimally thereafter.
    * This total return can be decomposed into two parts:
        1.  **Immediate Reward:** $R(S)$ – The reward you get right away for being in state $S$.
        2.  **Discounted Future Optimal Return:** $\gamma \max_{A'} Q(S', A')$ – This represents the maximum possible return you can get from the *next state* ($S'$) you transition to, discounted by $\gamma$. The $\max_{A'}$ indicates that from $S'$, you will choose the action that maximizes your return from that point onward (behaving optimally).

### Example: Mars Rover (Applying Bellman Equation)

Let's verify the Bellman equation with our Mars Rover example ($\gamma = 0.5$).

* **Case 1: $Q(S_2, \text{Right})$**
    * Current State ($S$): $S_2$
    * Current Action ($A$): Right
    * Next State ($S'$): $S_3$
    * $R(S_2) = 0$
    * $Q(S_2, \text{Right}) = R(S_2) + \gamma \max_{A'} Q(S_3, A')$
    * $Q(S_2, \text{Right}) = 0 + 0.5 \times \max(Q(S_3, \text{Left}), Q(S_3, \text{Right}))$
    * Using values from the Q-table: $Q(S_2, \text{Right}) = 0 + 0.5 \times \max(25, 6.25)$
    * $Q(S_2, \text{Right}) = 0.5 \times 25 = 12.5$. (Matches previously calculated value).

* **Case 2: $Q(S_4, \text{Left})$**
    * Current State ($S$): $S_4$
    * Current Action ($A$): Left
    * Next State ($S'$): $S_3$
    * $R(S_4) = 0$
    * $Q(S_4, \text{Left}) = R(S_4) + \gamma \max_{A'} Q(S_3, A')$
    * $Q(S_4, \text{Left}) = 0 + 0.5 \times \max(Q(S_3, \text{Left}), Q(S_3, \text{Right}))$
    * Using values from the Q-table: $Q(S_4, \text{Left}) = 0 + 0.5 \times \max(25, 6.25)$
    * $Q(S_4, \text{Left}) = 0.5 \times 25 = 12.5$. (Matches previously calculated value).

* **Terminal States:** For a terminal state $S_{term}$, the Bellman equation simplifies to $Q(S_{term}, A) = R(S_{term})$, as there are no future states or rewards.

### Intuition of Bellman Equation:

The Bellman equation provides a recursive relationship: the value of being in a state $S$ and taking action $A$ is the immediate reward plus the discounted maximum value achievable from the subsequent state $S'$. This mathematical breakdown is key to developing algorithms for RL.

The next (optional) video will discuss **Stochastic Markov Decision Processes**, where actions can have random outcomes. Following that, we will develop an RL algorithm based on the Bellman equation.

## Stochastic Markov Decision Processes (Optional)

In some real-world applications, actions do not always have deterministic outcomes. This video generalizes the RL framework to model **stochastic (random) environments**, where actions lead to a *distribution* of next states.

### Stochastic Environment Example: Mars Rover

* **Deterministic (Previous Assumption):** If you command "go left" from $S_4$, you *always* end up in $S_3$.
* **Stochastic (New Model):** If you command "go left" from $S_4$:
    * 90% chance (0.9 probability) of successfully going left to $S_3$.
    * 10% chance (0.1 probability) of accidentally slipping and going right to $S_5$.
* This means the next state $S'$ is now a **random variable**, not fixed.

### Impact on Returns: Expected Return

When the environment is stochastic, the sequence of states and rewards is also random. Therefore, the "return" ($G$) from a given state and action becomes a random variable itself.

* **Goal Change:** We are no longer trying to maximize a single return. Instead, the goal of an RL algorithm in a stochastic environment is to maximize the **expected return** (or average return).
    * **Expected Return:** The average value of the sum of discounted rewards over many (e.g., millions) hypothetical trials of following a policy from a given state.
    * **Notation:** $E[R_1 + \gamma R_2 + \gamma^2 R_3 + \dots]$

### Impact on the Bellman Equation:

The Bellman equation is modified to account for the randomness of the next state:

$$Q(S,A) = R(S) + \gamma E_{S' \sim P(S'|S,A)} [\max_{A'} Q(S', A')]$$

* **Key Change:** The $\max_{A'} Q(S', A')$ term is now inside an **expected value operator** ($E[\dots]$).
* **Intuition:** The total return from $(S,A)$ is the immediate reward $R(S)$ plus the discounted *average* of the maximum future returns achievable from the possible next states $S'$ (weighted by their probabilities of occurrence $P(S'|S,A)$).

### Lab Exploration (Misstep Probability):

The optional lab allows you to experiment with a "misstep probability" for the Mars Rover, simulating a stochastic environment.

* **Misstep Probability:** This parameter (e.g., 0.1 for 10% chance of going opposite direction) quantifies the randomness.
* **Observation:** As the misstep probability increases, the $Q(s,a)$ values (and the optimal expected return) will generally **decrease** for all states and actions. This is because your control over the robot is diminished, leading to lower expected rewards.

### Transition to Continuous State Spaces:

This stochastic MDP framework applies to small, discrete state spaces. However, many practical RL problems involve much larger, even **continuous, state spaces**. The next video will generalize the RL framework to handle such continuous state spaces.

## Reinforcement Learning with Continuous State Spaces

While our simplified Mars Rover example used a discrete set of states (6 positions), most real-world robotic control applications, including the Lunar Lander project, operate in **continuous state spaces**. This means the state is represented by a vector of numbers, where each number can take on any value within a range.

### What is a Continuous State Space?

Instead of being in one of a few distinct positions, an agent's state is defined by measurements that are continuous, like positions, velocities, and orientations.

### Examples of Continuous State Spaces:

1.  **Mars Rover (Continuous Version):**
    * Instead of 6 discrete boxes, imagine the rover can be at *any* position along a 6-kilometer line (e.g., at 2.7 km, 4.8 km). Its position is a single continuous number.

2.  **Controlling a Car/Truck:**
    * The state of a car or truck is described by multiple continuous numbers:
        * **Position:** `x` (east-west), `y` (north-south)
        * **Orientation:** `theta` ($\theta$, angle it's facing, 0-360 degrees)
        * **Velocities:** `x_dot` (speed in x-direction), `y_dot` (speed in y-direction)
        * **Angular Velocity:** `theta_dot` ($\dot{\theta}$, how fast it's turning)
    * **Total State Vector:** A vector of 6 continuous numbers: $[x, y, \theta, \dot{x}, \dot{y}, \dot{\theta}]$.

3.  **Controlling an Autonomous Helicopter:**
    * The state of a helicopter requires even more continuous numbers to describe its complex movement:
        * **Position:** `x, y, z` (north-south, east-west, height above ground)
        * **Orientation (Euler Angles):** `phi` ($\phi$, roll - side-to-side tilt), `theta` ($\theta$, pitch - forward/backward tilt), `psi` ($\psi$, yaw - compass heading).
        * **Velocities:** `x_dot, y_dot, z_dot` (speed in each direction).
        * **Angular Velocities:** `phi_dot, theta_dot, psi_dot` (rates of change for roll, pitch, yaw).
    * **Total State Vector:** A vector of 12 continuous numbers. This complex vector is the input to the helicopter's control policy.

### Implication for Reinforcement Learning:

In a **continuous state reinforcement learning problem** (or **continuous state Markov Decision Process - MDP**):
* The state is no longer a small integer (e.g., 1 to 6).
* It's a **vector of numbers**, where each number can take on any real value within its valid range.

The upcoming practice lab will involve implementing an RL algorithm for a **simulated Lunar Lander**, which is another application with a continuous state space. This will provide practical experience with such problems.

## The Lunar Lander: A Continuous State RL Application

The Lunar Lander is a classic continuous state Reinforcement Learning (RL) problem where the goal is to safely land a simulated vehicle on a landing pad on the moon. This application is a fun video game scenario used by many RL researchers.

<img src="/metadata/lunar_lander.png" width="300" />

### The Task:

* Command a lunar lander to fire its thrusters to achieve a soft landing within a designated landing pad (between two yellow flags).
* Failure: Crashing on the moon's surface.
* Success: Soft landing on the pad.

### Actions (Discrete):

At each time step, the lander can choose one of four actions:
1.  **Nothing:** Inertia and gravity pull the lander down.
2.  **Left (Fire Left Thruster):** Pushes the lander to the right.
3.  **Main (Fire Main Engine):** Provides downward thrust (slows descent).
4.  **Right (Fire Right Thruster):** Pushes the lander to the left.

### State Space (Continuous):

The lander's state ($S$) is a vector of continuous and binary values, describing its current condition:
* **Position:**
    * $X$: Horizontal position (left/right).
    * $Y$: Vertical position (height).
* **Velocity:**
    * $\dot{X}$: Horizontal velocity.
    * $\dot{Y}$: Vertical velocity.
* **Orientation:**
    * $\theta$: Angle/tilt of the lander (how far it's tilted left/right).
    * $\dot{\theta}$: Angular velocity (how fast it's rotating).
* **Leg Grounded Status (Binary):**
    * $L$: 1 if the left leg is grounded (touching the surface), 0 otherwise.
    * $R$: 1 if the right leg is grounded (touching the surface), 0 otherwise.

This state vector is the input to the policy ($\pi$).

### Reward Function (Moderately Complex):

The reward function incentivizes desired behaviors and penalizes undesired ones:
* **Landing on Pad:** Reward between +100 and +140 (depends on how close to center).
* **Moving Towards/Away from Pad:** Positive reward for moving closer, negative for drifting away.
* **Crashing:** Large negative reward (e.g., -100).
* **Soft Landing (Non-Crash):** +100 reward.
* **Leg Grounded:** +10 reward for each leg (left or right) that touches the ground.
* **Fuel Consumption (Penalties):**
    * -0.3 for firing the main engine.
    * -0.03 for firing a left or right side thruster.

**Importance of Reward Function Design:** Crafting a good reward function is crucial for RL, as it tells the agent *what* to achieve without specifying *how*. For the Lunar Lander, it encourages safe, fuel-efficient landings on the pad.

### Goal of the Lunar Lander Problem:

* **Learn a Policy ($\pi$):** Find a function $\pi(S)$ that maps the current state $S$ to the optimal action $A$.
* **Maximize Return:** This policy should maximize the sum of discounted rewards (return), typically using a high discount factor (e.g., $\gamma = 0.985$) to value future rewards significantly.

This problem is a quintessential example of a continuous state RL application, which you will implement using deep learning in the practice lab. The next video will introduce Deep Reinforcement Learning.

## Deep Reinforcement Learning: The DQN Algorithm

This video introduces how to use a neural network to approximate the state-action value function, $Q(s,a)$, which is central to solving Reinforcement Learning problems like controlling the Lunar Lander. This approach is called **Deep Q-Network (DQN)**.

### Approximating the Q-Function with a Neural Network:

* **Input ($X$):** The neural network takes the combined **state ($s$) and action ($a$)** as input.
    * For Lunar Lander: State ($s$) is an 8-number vector (position, velocity, angle, angular velocity, leg grounded status).
    * Action ($a$) is a 4-number one-hot encoded vector (representing "nothing", "left", "main", "right" thruster).
    * Thus, the input $X$ to the neural network is a 12-number vector (8 state + 4 action).
* **Neural Network Architecture:** A typical setup might be a neural network with 64 units in the first hidden layer, 64 units in the second hidden layer, and a single output unit in the output layer.
* **Output ($Y$):** The neural network's output is the predicted $Q(s,a)$ value for the given state-action pair.

### Using the Trained Q-Network for Policy:

Once the neural network (Q-network) is trained to approximate $Q(s,a)$:

1.  When the Lunar Lander is in a given state $s$:
2.  Compute $Q(s,a)$ for *all* possible actions $a$ (e.g., $Q(s, \text{nothing}), Q(s, \text{left}), Q(s, \text{main}), Q(s, \text{right})$) by feeding each $(s, \text{one-hot}(a))$ pair into the neural network.
3.  Choose the action $a$ that yields the **highest $Q(s,a)$ value**. This action is the optimal action for that state according to the current Q-network.

### Training the Q-Network (Bellman Equation as Supervised Learning):

The core idea is to turn the Bellman equation into a supervised learning problem to train the neural network.

* **Bellman Equation:** $Q(S,A) = R(S) + \gamma \max_{A'} Q(S', A')$
* **Supervised Learning Setup:**
    * **Input ($X$):** The current state-action pair $(S,A)$.
    * **Target ($Y$):** The value on the right-hand side of the Bellman equation, which is $R(S) + \gamma \max_{A'} Q(S', A')$.
    * **Loss Function:** Typically, **mean squared error** (MSE) is used to minimize the difference between the neural network's prediction $Q(S,A)$ and the target $Y$.

### Data for Training the Q-Network:

1.  **Experience Collection:**
    * The Lunar Lander (or agent) interacts with the environment (initially by taking random actions, or actions based on a current, imperfect policy).
    * For each step, record the tuple $(S, A, R(S), S')$, representing: (current state, action taken, reward received, next state).
    * Store the most recent experiences in a **replay buffer** (e.g., 10,000 most recent tuples) to manage memory.

2.  **Creating Training Examples (X, Y pairs):**
    * For each tuple $(S, A, R(S), S')$ from the replay buffer:
        * **Input $X$ for the Neural Network:** Combine $S$ and $A$ into the 12-number input vector.
        * **Target $Y$ for Supervised Learning:** Calculate $Y = R(S) + \gamma \max_{A'} Q_{\text{current}}(S', A')$.
            * **Crucial:** The $Q_{\text{current}}(S', A')$ value is obtained from the *current* (or a recent snapshot of the) Q-network itself. This is what makes the process iterative.

3.  **Train the Neural Network:** Use these $(X, Y)$ pairs to train the neural network via supervised learning (e.g., gradient descent with MSE loss). The newly trained network, let's call it $Q_{\text{new}}$, should be a slightly better approximation of the true Q-function.

### The DQN Algorithm Loop:

1.  **Initialize Q-Network:** Randomly initialize the neural network's parameters. This is your initial (random) guess for the Q-function.
2.  **Repeat (for many iterations):**
    *  **Explore/Act:** Take actions in the environment (e.g., Lunar Lander simulator). Initially, actions might be random. Over time, they might be chosen based on the current Q-network (exploiting).
    *  **Store Experience:** Save the $(S, A, R(S), S')$ tuples to the replay buffer. Say 10000 of these.
    *  **Train the network:** Now do the following
        * For each of these say 10000 tuple, calculate the target $Y$ using the Bellman equation and the *current* Q-network's estimate for $Q(S', A')$.
        * Train the Q-network (using supervised learning) to predict these $(X, Y)$ pairs.
    *  **Update Q-Network:** The newly trained network becomes the "current" Q-network for the next iteration of experience collection and target calculation.

* **Convergence:** By iteratively improving the Q-function's estimate using the Bellman equation, the Q-network gradually converges to a good approximation of the true $Q(s,a)$, allowing the agent to learn optimal behavior.
* **DQN (Deep Q-Network):** This algorithm is called DQN because it uses "deep learning" (neural networks) to learn the "Q-function."

While this basic DQN works, refinements are needed for better performance. The next videos will cover these refinements.

## DQN Refinement 1: Improved Neural Network Architecture for Q-Function

The initial neural network architecture for DQN, which inputs `(state, action)` and outputs $Q(s,a)$, can be inefficient. A more efficient architecture is commonly used in most DQN implementations.

### Original Architecture (Inefficient):

* **Input:** State `s` (8 numbers) + one-hot encoded Action `a` (4 numbers) = 12 numbers.
* **Output:** Single number ($Q(s,a)$).
* **Problem:** To find the optimal action for a given state `s`, you'd have to run inference through the neural network **four separate times** (once for each possible action: `Q(s, nothing)`, `Q(s, left)`, `Q(s, main)`, `Q(s, right)`). This is computationally expensive.

### Improved Architecture (Efficient):

* **Input:** Only the **State `s`** (8 numbers).
* **Neural Network:** Same hidden layers (e.g., two layers of 64 units each).
* **Output Layer:** Has **four output units**, where each unit corresponds to the Q-value for one specific action.
    * Output 1: $Q(s, \text{nothing})$
    * Output 2: $Q(s, \text{left})$
    * Output 3: $Q(s, \text{main})$
    * Output 4: $Q(s, \text{right})$
* **Benefit:** Given a state `s`, you only need to run inference through the neural network **once**. This single forward pass simultaneously computes the Q-value for all possible actions in that state. You can then quickly pick the action that corresponds to the maximum Q-value.

### Efficiency for Bellman Equation Target Calculation:

This architecture also makes computing the target $Y$ for the Bellman equation more efficient: $Y = R(S) + \gamma \max_{A'} Q(S', A')$.
* When calculating $Q(S', A')$ for all $A'$ to find the maximum, the new architecture can get all those $Q(S', A')$ values in a single pass of $S'$ through the network, directly identifying the max.

This architectural change significantly improves the overall efficiency of the DQN algorithm and will be used in the practice lab. The next video will introduce another important refinement: the Epsilon-Greedy policy.

## DQN Refinement 2: Epsilon-Greedy Policy for Action Selection

While the DQN algorithm learns to approximate $Q(s,a)$, we need a strategy to select actions in the environment, especially during the learning process when the Q-function estimate is still imperfect. The **Epsilon-Greedy policy** is the most common approach for this.

### The Problem: Exploration vs. Exploitation

* **Exploitation (Greedy Action):** Always picking the action `a` that currently maximizes $Q(s,a)$ based on the neural network's current estimate. This leverages (exploits) what the agent has already learned.
    * **Downside:** If the Q-network's initial random initialization or early experiences lead it to believe a truly good action is bad, it might *never try it*, preventing it from discovering optimal strategies. It gets stuck in local optima of its own flawed beliefs.
* **Exploration (Random Action):** Randomly picking an action `a`. This allows the agent to try new things and gain more information about the environment, potentially discovering better actions it hadn't considered.
    * **Downside:** Random actions are often suboptimal and can lead to low returns, making the learning process inefficient or even dangerous (e.g., crashing the Lunar Lander).

### The Epsilon-Greedy Policy:

This policy balances exploration and exploitation using a parameter $\epsilon$ (epsilon):

* **With probability $1 - \epsilon$ (most of the time):** The agent takes a **greedy action** – it picks the action `a` that maximizes its current estimate of $Q(s,a)$. (Exploitation)
* **With probability $\epsilon$ (a small fraction of the time):** The agent picks an action `a` **randomly** from all possible actions. (Exploration)

* **Example:** If $\epsilon = 0.05$: 95% of the time, the agent exploits its current knowledge; 5% of the time, it explores randomly.

### Annealing Epsilon (Gradually Decreasing $\epsilon$):

A common and effective trick is to **start with a high $\epsilon$** (e.g., $\epsilon = 1.0$, taking actions completely randomly initially) and **gradually decrease it over time** (e.g., down to 0.01 or less).

* **Initial High $\epsilon$:** Encourages broad exploration early in training when the Q-function estimate is very poor. This helps the agent discover the environment's dynamics and potential rewards.
* **Gradually Decreasing $\epsilon$:** As the Q-function estimate improves, the agent transitions to exploiting its knowledge more often, converging towards optimal behavior.

### Finicky Nature of RL Hyperparameters:

* Compared to supervised learning, RL algorithms are often **more sensitive ("finicky")** to hyperparameter choices (like $\epsilon$, learning rate, discount factor).
* A slightly suboptimal choice of parameters in RL can lead to significantly worse performance (e.g., 10-100x slower learning or complete failure to learn), making tuning more challenging. However, good default values are often provided in practical labs.

The next (optional) video will discuss further refinements to the DQN algorithm, including mini-batching and soft updates, to improve training speed and stability.
