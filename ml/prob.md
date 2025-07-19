# Probability & Statistics for Machine Learning & Data Science

This course focuses on probability and statistics, essential for designing, interpreting, and adapting machine learning algorithms.

### Key Concepts and Applications

* **Bayes' Theorem:**
    * Calculates the probability of an event given certain conditions.
    * Example: Determining the true likelihood of having a rare illness after a positive test result, which can often be counter-intuitive.
    * Applied in spam detection: $P(\text{spam} | \text{words, features})$.

* **Maximum Likelihood Estimation (MLE):**
    * A method for training models by finding the model parameters that maximize the probability of observing the given data.
    * Explains the use of squared error in linear regression:
        * Squared error naturally arises when assuming data points are sampled from a Gaussian distribution.
        * The logarithm of the Gaussian likelihood contains a squared term.
    * Helps determine when squared error is appropriate and when alternative cost functions might be needed.

* **Gaussian Distribution:**
    * One of the most widely used probability distributions.
    * Its properties lead to the use of squared error in MLE when applied to certain data distributions.

* **Regularization:**
    * Techniques used to prevent overfitting in models.
    * L2 regularization (sum of squared parameters) can be derived from probabilistic assumptions, specifically a Gaussian prior distribution on the model coefficients.

* **Hypothesis Testing:**
    * Used to determine if a claim or hypothesis about a population is supported by sample data.
    * Examples: Testing the effectiveness of a new drug, evaluating the impact of a web page feature on viewership.
    * Provides precise definitions and understanding of terms like "confidence," "confidence interval," and "p-value" to ensure accurate scientific conclusions.
 
## Week 1: Introduction to Probability

* Probability measures the likelihood of an event occurring.
* It is expressed as a value between 0 and 1 (or 0% to 100%).

### Basic Probability Formula

* $$P(\text{Event}) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

### Example: Kids playing soccer

* **Scenario:** 10 kids in a school, 3 play soccer, 7 don't.
* **Problem:** What is the probability that a randomly picked kid plays soccer?
* **Solution:**
    * Favorable outcomes (kids who play soccer): 3
    * Total possible outcomes (total kids): 10
    * $$P(\text{soccer}) = \frac{3}{10} = 0.3$$
* **Terminology:**
    * **Event:** The specific outcome of interest (e.g., kid plays soccer).
    * **Sample Space:** The set of all possible outcomes (e.g., all 10 kids).

### Venn Diagrams and Probability

* A Venn diagram can visually represent the sample space and events.
* The entire rectangle represents the total population (sample space).
* Circles within the rectangle represent specific events.

### Experiment

* An **experiment** is any process that produces an uncertain outcome.
* **Examples:** Flipping a coin, rolling a die.

### Coin Flip Examples

#### Flipping one fair coin

* **Experiment:** Flipping a coin.
* **Possible outcomes:** Heads (H), Tails (T) - each with 50% probability.
* **Problem:** Probability of landing on heads.
* **Solution:**
    * Favorable outcomes: 1 (Heads)
    * Total outcomes: 2 (Heads, Tails)
    * $$P(\text{heads}) = \frac{1}{2} = 0.5$$

#### Flipping two fair coins

* **Experiment:** Flipping two coins.
* **Possible outcomes (Sample Space):** HH, HT, TH, TT (4 outcomes).
* **Problem:** Probability of both coins landing on heads (HH).
* **Solution:**
    * Favorable outcomes: 1 (HH)
    * Total outcomes: 4
    * $$P(\text{HH}) = \frac{1}{4} = 0.25$$

#### Flipping three fair coins

* **Experiment:** Flipping three coins.
* **Possible outcomes (Sample Space):** HHH, HHT, HTH, HTT, THH, THT, TTH, TTT (8 outcomes).
* **Problem:** Probability of all three coins landing on heads (HHH).
* **Solution:**
    * Favorable outcomes: 1 (HHH)
    * Total outcomes: 8
    * $$P(\text{HHH}) = \frac{1}{8} = 0.125$$

## Probability of Complement Event

* The complement of an event is the probability that the event *does not* occur.
* If the probability of event A occurring is $P(A)$, then the probability of event A not occurring (denoted as $P(A')$ or $P(\text{not A}))$ is given by the **Complement Rule**:

$P(A') = 1 - P(A)$

### Example: Soccer Players

* **Scenario:** 10 kids, 3 play soccer, 7 don't.
* **Event A:** A child plays soccer.
    * $P(\text{soccer}) = \frac{3}{10} = 0.3$
* **Event A':** A child does not play soccer.
    * $P(\text{not soccer}) = \frac{7}{10} = 0.7$
* Using the Complement Rule: $P(\text{not soccer}) = 1 - P(\text{soccer}) = 1 - 0.3 = 0.7$.

### Venn Diagram Representation

* The entire sample space represents all possible outcomes.
* The event A is represented by a circle within the sample space.
* The complement of event A ($A'$) is everything outside the circle of A within the sample space.

### Example: Flipping Three Coins

* **Scenario:** Flipping three coins.
* **Total possible outcomes (sample space):** 8
    * (HHH, HHT, HTH, THH, HTT, THT, TTH, TTT)
* **Event A:** Obtaining three heads (HHH).
    * $P(\text{three heads}) = \frac{1}{8}$
* **Event A':** Not obtaining three heads.
    * Using the Complement Rule: $P(\text{not three heads}) = 1 - P(\text{three heads}) = 1 - \frac{1}{8} = \frac{7}{8}$.

### Example: Rolling a Die

* **Scenario:** Rolling one die.
* **Total possible outcomes:** 6 (1, 2, 3, 4, 5, 6)
* **Event A:** Obtaining a 6.
    * $P(6) = \frac{1}{6}$
* **Event A':** Obtaining anything different than 6 (not 6).
    * Using the Complement Rule: $P(\text{not 6}) = 1 - P(6) = 1 - \frac{1}{6} = \frac{5}{6}$.
 

## Sum of Probabilities

The sum of probabilities is used to find the probability of one event *or* another event occurring.

### Disjoint Events

The sum of probabilities applies directly when events are **disjoint** (mutually exclusive), meaning they cannot occur at the same time.

* **Example (School Sports):**
    * Kids can only play one sport (soccer or basketball).
    * $P(\text{Soccer}) = 0.3$
    * $P(\text{Basketball}) = 0.4$
    * $P(\text{Soccer or Basketball}) = P(\text{Soccer}) + P(\text{Basketball}) = 0.3 + 0.4 = 0.7$
* **Venn Diagram Representation (Union):**
    * If A and B are disjoint events, then the probability of A union B ($$P(A \cup B)$$) is the sum of their individual probabilities:
        $P(A \cup B) = P(A) + P(B)$

### Examples with Dice Rolls

* **Example (Single Die Roll):**
    * **Event A:** Rolling an even number (2, 4, 6)
    * **Event B:** Rolling a five (5)
    * These events are disjoint.
    * $P(\text{Even number}) = 3/6$
    * $P(\text{Five}) = 1/6$
    * $P(\text{Even number or Five}) = P(\text{Even number}) + P(\text{Five}) = 3/6 + 1/6 = 4/6 = 2/3$

* **Example (Two Dice Roll - Sums):**
    * **Event A:** Sum of seven (e.g., (1,6), (2,5), (3,4), (4,3), (5,2), (6,1))
    * **Event B:** Sum of ten (e.g., (4,6), (5,5), (6,4))
    * These events are disjoint (a roll cannot sum to both seven and ten simultaneously).
    * $P(\text{Sum of Seven}) = 6/36$
    * $P(\text{Sum of Ten}) = 3/36$
    * $P(\text{Sum of Seven or Sum of Ten}) = 6/36 + 3/36 = 9/36 = 1/4$

* **Example (Two Dice Roll - Differences):**
    * **Event A:** Difference of two (e.g., (1,3), (2,4), (3,5), (4,6), (3,1), (4,2), (5,3), (6,4))
    * **Event B:** Difference of one (e.g., (1,2), (2,3), (3,4), (4,5), (5,6), (2,1), (3,2), (4,3), (5,4), (6,5))
    * These events are disjoint (a pair of rolls cannot have a difference of both one and two simultaneously).
    * $P(\text{Difference of Two}) = 8/36$
    * $P(\text{Difference of One}) = 10/36$
    * $P(\text{Difference of Two or Difference of One}) = 8/36 + 10/36 = 18/36 = 1/2$

 ## Sum of Probabilities for Joint Events

When events are not disjoint (they can occur at the same time), we need to adjust the sum of probabilities to avoid over-counting the overlapping outcomes.

### The Inclusion-Exclusion Principle

The formula for the probability of the union of two events A and B (i.e., A or B occurs) when they are not disjoint is:

$P(A \cup B) = P(A) + P(B) - P(A \cap B)$

* $P(A \cup B)$: Probability of A or B occurring.
* $P(A)$: Probability of A occurring.
* $P(B)$: Probability of B occurring.
* $P(A \cap B)$: Probability of both A and B occurring (the intersection). This term is subtracted because the outcomes in the intersection are counted twice when $P(A)$ and $P(B)$ are simply added.

### Examples

* **Example (School Sports - Not Disjoint):**
    * Kids can play multiple sports (soccer and/or basketball).
    * $P(\text{Soccer}) = 0.6$
    * $P(\text{Basketball}) = 0.5$
    * $P(\text{Soccer and Basketball}) = 0.3$
    * $P(\text{Soccer or Basketball}) = P(\text{Soccer}) + P(\text{Basketball}) - P(\text{Soccer and Basketball})$
        $= 0.6 + 0.5 - 0.3 = 0.8$

* **Example (Two Dice Roll - Sum and Difference):**
    * **Event A:** Sum of seven (6 outcomes: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1))
    * **Event B:** Difference of one (10 outcomes: (1,2), (2,3), (3,4), (4,5), (5,6), (2,1), (3,2), (4,3), (5,4), (6,5))
    * **Intersection ($A \cap B$):** Outcomes where sum is seven *and* difference is one.
        * (3,4) (Sum 7, Diff 1)
        * (4,3) (Sum 7, Diff 1)
        * There are 2 such outcomes.
    * $P(\text{Sum of Seven}) = 6/36$
    * $P(\text{Difference of One}) = 10/36$
    * $P(\text{Sum of Seven and Difference of One}) = 2/36$
    * $P(\text{Sum of Seven or Difference of One}) = P(\text{Sum of Seven}) + P(\text{Difference of One}) - P(\text{Sum of Seven and Difference of One})$
        $= 6/36 + 10/36 - 2/36 = 14/36 = 7/18$

### Relationship between Disjoint and Non-Disjoint Events

* The formula for joint events is a general case.
* For disjoint (mutually exclusive) events, the intersection $P(A \cap B)$ is 0 (since they cannot happen at the same time).
* Therefore, for disjoint events, the formula simplifies to:
    $P(A \cup B) = P(A) + P(B) - 0 = P(A) + P(B)$

## Independence of Events

* **Definition**: Two events are **independent** if the occurrence of one does not affect the probability of the other.
    * **Example**: Tossing a coin multiple times; each toss is independent.
    * **Non-example**: Moves in a chess game; each move affects subsequent moves.
* **Importance in Machine Learning**: Assuming independence can significantly simplify calculations and improve prediction models.

## Illustrative Examples of Independence

### School Soccer Preference

* **Scenario 1**: 100 kids, 50 like soccer, 50 don't. Randomly split into two rooms of 50 kids each.
    * **Best Estimate**: Roughly 25 kids in each room will like soccer due to random distribution.
* **Scenario 2**: 100 kids, 40 like soccer, 60 don't. Randomly split into Room 1 (30 kids) and Room 2 (70 kids).
    * **Probability of a kid liking soccer**: $P(S) = 0.4$ (40 out of 100).
    * **Probability of a kid being in Room 1**: $P(R1) = 0.3$ (30 out of 100).
    * **Expected number of kids in Room 1 who like soccer**: Since the split is random, the proportion of soccer lovers is expected to be maintained in each room.
        * Expected kids = $40\%$ of 30 = 12 kids.
    * **Mathematical Representation**: This implies the probability of a kid playing soccer AND being in Room 1 is the product of their individual probabilities.
        * $P(S \cap R1) = P(S) \times P(R1)$
        * $P(S \cap R1) = 0.4 \times 0.3 = 0.12$ (or 12%)

## The Product Rule for Independent Events

* For two independent events A and B, the probability of both A and B occurring (their intersection) is:
    * $P(A \cap B) = P(A) \times P(B)$
* This rule is applicable only when events are **independent**.

## Extending the Product Rule

### Coin Toss Example

* **Question**: What is the probability of tossing a fair coin 5 times and getting heads all 5 times?
* **Solution**: Each toss is independent, and the probability of getting a head on a single toss is $1/2$.
    * $P(\text{5 Heads}) = P(H_1) \times P(H_2) \times P(H_3) \times P(H_4) \times P(H_5)$
    * $P(\text{5 Heads}) = \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} = (\frac{1}{2})^5 = \frac{1}{32}$
* The product rule extends to multiple independent events: if events $E_1, E_2, \ldots, E_n$ are all independent, then $P(E_1 \cap E_2 \cap \ldots \cap E_n) = P(E_1) \times P(E_2) \times \ldots \times P(E_n)$.

### Dice Roll Example

* **Recall**: Probability of rolling a 6 on a single die is $1/6$.
* **Question 1**: What is the probability of rolling two dice and getting two 6s?
* **Solution**: Each die roll is independent.
    * $P(\text{6 on first die and 6 on second die}) = P(\text{6 on first die}) \times P(\text{6 on second die})$
    * $P(\text{6,6}) = \frac{1}{6} \times \frac{1}{6} = (\frac{1}{6})^2 = \frac{1}{36}$
* **Question 2**: What is the probability of rolling ten fair dice and getting ten 6s?
* **Solution**: All ten rolls are independent.
    * $P(\text{ten 6s}) = (\frac{1}{6})^{10}$ (a very small number)

The product rule is a powerful tool for calculating the probability of multiple events occurring, provided those events are independent.

## The Birthday Problem 🎂

The Birthday Problem explores the surprisingly high probability that, in a relatively small group of people, at least two individuals share the same birthday. It's often counter-intuitive.

### Problem Setup

* **Assumption**: There are 365 days in a year (no February 29th).
* **Goal**: Calculate the probability that, among a group of `n` people, at least two share a birthday.

### The Complement Approach

It's easier to calculate the probability that **no two people** share the same birthday, and then use the **complement rule** to find the probability of at least two sharing a birthday.

* $P(\text{at least two share birthday}) = 1 - P(\text{no two share birthday})$

### Calculating $P(\text{no two share birthday})$

Let's consider `n` people and the probability that all `n` people have different birthdays.

1.  **For the first person (n=1)**:
    * They can have a birthday on any day of the 365 days.
    * $P(\text{1 person has unique birthday}) = \frac{365}{365} = 1$

2.  **For the second person (n=2)**:
    * To have a different birthday from the first person, they must be born on one of the remaining 364 days.
    * $P(\text{2 people have unique birthdays}) = \frac{365}{365} \times \frac{364}{365}$

3.  **For the third person (n=3)**:
    * To have a different birthday from the first two, they must be born on one of the remaining 363 days.
    * $P(\text{3 people have unique birthdays}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365}$

4.  **Generalizing for 'n' people**:
    * The probability that all `n` people have different birthdays is the product of these fractions:

$$
P(\text{no two share birthday, for n people}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times \ldots \times \frac{(365 - n + 1)}{365}
$$

This can also be expressed using permutations:

$$
P(\text{no two share birthday, for n people}) = \frac{P(365, n)}{365^n} = \frac{365!}{(365-n)! \cdot 365^n}
$$

### Surprising Results

The probability of no two people sharing a birthday drops surprisingly quickly as the number of people increases.

* **n = 23**:
    * $P(\text{no two share birthday})$ is approximately **0.493**.
    * This means $P(\text{at least two share birthday}) = 1 - 0.493 = \mathbf{0.507}$.
    * **In a group of just 23 people, it's more likely than not (over 50% chance) that two people share the same birthday.**

* **n = 30**:
    * $P(\text{no two share birthday})$ is approximately **0.294**.
    * $P(\text{at least two share birthday}) = 1 - 0.294 = \mathbf{0.706}$.
    * This means there's roughly a **70% chance** that two people share a birthday.

* **n = 50**:
    * $P(\text{no two share birthday})$ is approximately **0.03**.
    * $P(\text{at least two share birthday}) = 1 - 0.03 = \mathbf{0.97}$.
    * Almost certain that two people share a birthday.

* **n = 366**:
    * By the Pigeonhole Principle, if there are 366 people and only 365 possible birthdays, at least two people **must** share a birthday.
    * $P(\text{no two share birthday}) = 0$.

### Visualizing the Probability Drop

The graph shows the probability of *no shared birthdays* on the y-axis against the *number of people* on the x-axis. It rapidly decreases, crossing the 0.5 mark at around 23 people, indicating a higher than 50% chance of a shared birthday. This demonstrates the non-intuitive nature of the birthday problem.

## Conditional Probability

**Conditional probability** calculates the likelihood of an event occurring **given that another event has already happened**. It updates our understanding of probabilities based on new information.

### Notation

The probability of event A happening given that event B has occurred is denoted as $P(A|B)$. The vertical bar "|" is read as "given that".

### Example: Two Coin Tosses 💰💰

* **Original Sample Space**: {HH, HT, TH, TT} - 4 equally likely outcomes.
* **Probability of two heads**: $P(\text{HH}) = \frac{1}{4}$.

#### Scenario 1: Probability of HH given the first coin is H

* **Given Information**: The first coin landed heads.
* **New Sample Space**: We now only consider outcomes where the first coin is heads: {HH, HT}. This reduces our sample space from 4 to 2 outcomes.
* **Favorable Outcome**: Only one outcome is HH.
* **Conditional Probability**: $P(\text{HH} | \text{1st is H}) = \frac{1}{2}$.
    * The probability changed from $1/4$ to $1/2$ because of the new information.

#### Scenario 2: Probability of HH given the first coin is T

* **Given Information**: The first coin landed tails.
* **New Sample Space**: We only consider outcomes where the first coin is tails: {TH, TT}.
* **Favorable Outcome**: There are no outcomes of HH in this new sample space.
* **Conditional Probability**: $P(\text{HH} | \text{1st is T}) = \frac{0}{2} = 0$.
    * The probability changed from $1/4$ to $0$ due to the strong condition.

## The General Product Rule

The **general product rule** links conditional probability with the probability of the intersection of two events. It applies whether the events are independent or dependent.

### Formula

For any two events A and B, the probability of both A and B occurring is:

$P(A \cap B) = P(A) \times P(B|A)$

This means the probability of A and B happening is the probability of A happening, multiplied by the probability of B happening **given that A has already happened**.

### Relation to Independent Events

* If events A and B are **independent**, then the occurrence of A does not affect the probability of B.
* In this special case, $P(B|A) = P(B)$.
* Substituting this into the general product rule, we get the product rule for independent events:
    $P(A \cap B) = P(A) \times P(B)$ (which we learned in the previous lesson).

## Example: Rolling Two Dice 🎲🎲

* **Sample Space**: 36 possible outcomes for rolling two dice (e.g., (1,1), (1,2), ..., (6,6)).

#### Scenario 1: Probability that the first die is 6 AND the sum is 10

* Let A = "first die is 6"
* Let B = "sum is 10"
* **Favorable outcome for $A \cap B$**: Only (6,4).
* **Direct Probability**: $P(A \cap B) = \frac{1}{36}$.

* **Using the General Product Rule**:
    * $P(A) = P(\text{first die is 6})$: There are 6 outcomes where the first die is 6 ((6,1), (6,2), (6,3), (6,4), (6,5), (6,6)). So, $P(A) = \frac{6}{36} = \frac{1}{6}$.
    * $P(B|A) = P(\text{sum is 10 | first die is 6})$:
        * Given the first die is 6, our new sample space is the 6 outcomes starting with 6: {(6,1), (6,2), (6,3), (6,4), (6,5), (6,6)}.
        * Among these, only (6,4) results in a sum of 10.
        * So, $P(B|A) = \frac{1}{6}$.
    * Now apply the product rule: $P(A \cap B) = P(A) \times P(B|A) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36}$. This matches the direct calculation.

#### Scenario 2: Probability that the sum is 10

* **Favorable outcomes**: (4,6), (5,5), (6,4) - 3 outcomes.
* **Probability**: $P(\text{sum is 10}) = \frac{3}{36} = \frac{1}{12}$.

#### Scenario 3: Probability that the sum is 10 GIVEN the first die is 6

* **Given Information**: The first die is 6.
* **New Sample Space**: {(6,1), (6,2), (6,3), (6,4), (6,5), (6,6)} - 6 outcomes.
* **Favorable Outcome for sum is 10**: Only (6,4).
* **Conditional Probability**: $P(\text{sum is 10 | 1st is 6}) = \frac{1}{6}$.
    * The probability changed from $1/12$ to $1/6$.

#### Scenario 4: Probability that the sum is 10 GIVEN the first die is 1

* **Given Information**: The first die is 1.
* **New Sample Space**: {(1,1), (1,2), (1,3), (1,4), (1,5), (1,6)} - 6 outcomes.
* **Favorable Outcome for sum is 10**: None of these outcomes result in a sum of 10 (max sum is $1+6=7$).
* **Conditional Probability**: $P(\text{sum is 10 | 1st is 1}) = \frac{0}{6} = 0$.
    * The probability changed from $1/12$ to $0$.

Conditional probability is a fundamental concept for understanding how new information impacts the likelihood of events.

## Dependent Events

* **Definition**: Two events are **dependent** if the outcome of one event influences the outcome of the other.
* **Example 1 (Soccer & Room Assignment)**:
    * 100 kids, 50 play soccer, 50 don't.
    * Two rooms (capacity 50 each): Room 1 has a World Cup TV, Room 2 has a movie TV.
    * **Observation**: Kids who like soccer are likely to choose Room 1.
    * **Conclusion**: Liking soccer and being in Room 1 are dependent events because the choice of room is influenced by whether a kid plays soccer.
    * **Contrast with Independent Events**: If kids were randomly assigned to rooms, the events would be independent.

## Conditional Probability and Intersections

* **Scenario**: 100 kids, 40 play soccer (S), 60 don't.
    * Among those who play soccer, 80% wear running shoes (R).
    * **Given**:
        * $P(S) = 0.4$
        * $P(\text{not } S) = 0.6$
        * $P(R | S) = 0.8$ (Probability of wearing running shoes given they play soccer)

* **Probability of Soccer AND Running Shoes ($P(S \cap R)$)**:
    * This is the probability that a kid plays soccer AND wears running shoes.
    * Formula: $P(S \cap R) = P(S) \times P(R | S)$
    * Calculation: $0.4 \times 0.8 = 0.32$
    * This means 32% of kids (32 out of 100) are estimated to play soccer and wear running shoes.

* **Scenario Extension**: Probability that a kid wears running shoes when they don't like soccer is 50%.
    * **Given**: $P(R | \text{not } S) = 0.5$ (Probability of wearing running shoes given they do not play soccer)

* **Probability of NOT Soccer AND Running Shoes ($P(\text{not } S \cap R)$)**:
    * Formula: $P(\text{not } S \cap R) = P(\text{not } S) \times P(R | \text{not } S)$
    * Calculation: $0.6 \times 0.5 = 0.3$
    * This means 30% of kids (30 out of 100) are estimated to not play soccer and wear running shoes.

## Probability Tree

A probability tree visually represents all possible scenarios and their probabilities.

* **Branches:**
    * **Plays Soccer (S):** $P(S) = 0.4$
        * Wears Running Shoes (R): $P(R|S) = 0.8 \Rightarrow P(S \cap R) = 0.4 \times 0.8 = 0.32$
        * Doesn't Wear Running Shoes (R'): $P(R'|S) = 0.2 \Rightarrow P(S \cap R') = 0.4 \times 0.2 = 0.08$
    * **Doesn't Play Soccer (S'):** $P(S') = 0.6$
        * Wears Running Shoes (R): $P(R|S') = 0.5 \Rightarrow P(S' \cap R) = 0.6 \times 0.5 = 0.30$
        * Doesn't Wear Running Shoes (R'): $P(R'|S') = 0.5 \Rightarrow P(S' \cap R') = 0.6 \times 0.5 = 0.30$

* **Sum of all probabilities**: $0.32 + 0.08 + 0.30 + 0.30 = 1.00$

## Visualizing Dependence vs. Independence

* **Independent Events**: Can be represented by non-crossing lines or separate groups where the split for one event doesn't affect the other.
* **Dependent Events**: Represented by lines that "cross" or influence each other, indicating that the outcome of one event impacts the likelihood of the other. The example of kids choosing rooms based on liking soccer demonstrates this dependency.

## Bayes' Theorem: Probability of Disease Given a Positive Test

Bayes' Theorem helps calculate the **probability of an event (A) occurring given that another event (B) has occurred**, represented as $P(A|B)$. It's widely used in machine learning for tasks like spam filtering and speech recognition.

### Example: Rare Disease Testing

Let's consider a scenario with a rare disease and a diagnostic test:

* **Population:** 1,000,000 people
* **Disease Prevalence:** 1 in 10,000 people are sick.
    * Sick individuals: $1,000,000 / 10,000 = 100$
    * Healthy individuals: $1,000,000 - 100 = 999,900$
* **Test Effectiveness:** 99% effective.
    * **For sick people:** 99% are correctly diagnosed as sick, 1% are misdiagnosed as healthy.
    * **For healthy people:** 1% are misdiagnosed as sick, 99% are correctly diagnosed as healthy.

**Problem:** You tested positive. What is the probability that you actually have the disease given this positive test result?

### Breakdown of Outcomes

Let's categorize the population based on their actual health status and test results:

* **Sick and Diagnosed Sick:**
    * $99\%$ of 100 sick people $= 0.99 \times 100 = 99$ people.
* **Sick and Misdiagnosed as Healthy:**
    * $1\%$ of 100 sick people $= 0.01 \times 100 = 1$ person.
* **Healthy and Misdiagnosed as Sick (False Positive):**
    * $1\%$ of 999,900 healthy people $= 0.01 \times 999,900 = 9,999$ people.
* **Healthy and Diagnosed Healthy:**
    * $99\%$ of 999,900 healthy people $= 0.99 \times 999,900 = 989,901$ people.

### Calculating the Probability

We are interested in the probability of being sick *given* a positive test result. This means we only consider the group of people who were **diagnosed as sick**.

* **Total people diagnosed sick:**
    * Sick and diagnosed sick: 99 people
    * Healthy and misdiagnosed as sick: 9,999 people
    * Total: $99 + 9,999 = 10,098$ people.

* **Probability of being sick given a positive test:**
    * $P(\text{Sick | Diagnosed Sick}) = \frac{\text{Number of Sick and Diagnosed Sick}}{\text{Total Number of Diagnosed Sick}}$
    * $P(\text{Sick | Diagnosed Sick}) = \frac{99}{10,098} \approx 0.0098$

This means there's less than a 1% chance you are actually sick even after testing positive, which can be counterintuitive. This is because the number of healthy people who get a false positive (9,999) is significantly higher than the number of truly sick people who test positive (99), due to the disease's rarity.

### Visualizing with a Tree Diagram

* **Total Population (1,000,000)**
    * **Sick (100 people)**
        * Diagnosed Sick: 99 (99%)
        * Diagnosed Healthy: 1 (1%)
    * **Healthy (999,900 people)**
        * Diagnosed Sick: 9,999 (1%)
        * Diagnosed Healthy: 989,901 (99%)

When you are diagnosed sick, you fall into one of two groups: the 99 truly sick people or the 9,999 healthy but misdiagnosed people. The vast majority of people with a positive test result are, in fact, healthy.

## Bayes' Theorem: The Formula

We want to calculate the **probability of being sick (A) given that you tested sick (B)**, or $P(A|B)$.

### Recall from Conditional Probability

The fundamental formula for conditional probability states:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Where:
* $P(A|B)$ is the probability of A given B.
* $P(A \cap B)$ is the probability of both A and B occurring (their intersection).
* $P(B)$ is the probability of B occurring.

### Breaking Down the Components

Let's define the events and their given probabilities:

* **A:** Being sick
    * $P(A) = 1/10,000 = 0.0001$
* **A':** Not being sick (healthy)
    * $P(A') = 1 - P(A) = 1 - 0.0001 = 0.9999$
* **B:** Being diagnosed sick (testing positive)
* **B|A:** Diagnosed sick given that you are sick (correct diagnosis)
    * $P(B|A) = 0.99$ (99% effectiveness)
* **B|A':** Diagnosed sick given that you are not sick (false positive)
    * $P(B|A') = 0.01$ (1% misdiagnosis)

### Calculating the Numerator: $P(A \cap B)$

The probability of being sick AND diagnosed sick ($P(\text{Sick} \cap \text{Diagnosed Sick})$) can be found using the conditional probability formula $P(A \cap B) = P(A) \cdot P(B|A)$:

$$P(A \cap B) = P(\text{Sick}) \cdot P(\text{Diagnosed Sick | Sick})$$

### Calculating the Denominator: $P(B)$

The probability of being diagnosed sick ($P(\text{Diagnosed Sick})$) involves two mutually exclusive scenarios:
1.  Being sick and correctly diagnosed sick.
2.  Being healthy and misdiagnosed as sick (false positive).

Since these are disjoint events (you cannot be both sick and not sick at the same time), we can sum their probabilities:

$$P(B) = P(\text{Diagnosed Sick} \cap \text{Sick}) + P(\text{Diagnosed Sick} \cap \text{Not Sick})$$

Using the conditional probability formula for each term:

$$P(B) = [P(\text{Sick}) \cdot P(\text{Diagnosed Sick | Sick})] + [P(\text{Not Sick}) \cdot P(\text{Diagnosed Sick | Not Sick})]$$

### Bayes' Theorem Formula

Substituting these into the main conditional probability formula, we get Bayes' Theorem:

$$P(A|B) = \frac{P(A) \cdot P(B|A)}{[P(A) \cdot P(B|A)] + [P(A') \cdot P(B|A')]}$$

### Plugging in the Numbers

Using the probabilities defined above:

$$P(\text{Sick | Diagnosed Sick}) = \frac{0.0001 \cdot 0.99}{(0.0001 \cdot 0.99) + (0.9999 \cdot 0.01)}$$

$$P(\text{Sick | Diagnosed Sick}) = \frac{0.000099}{0.000099 + 0.009999}$$

$$P(\text{Sick | Diagnosed Sick}) = \frac{0.000099}{0.010098} \approx 0.0098$$

This confirms the previous numerical example: even with a positive test, the probability of actually having the disease is less than 1%. This highlights the importance of Bayes' Theorem, especially when dealing with rare events and test accuracy.

## Bayes' Theorem: Spam Detection Example

This example demonstrates how Bayes' Theorem can be applied to classify emails as spam based on the presence of certain words.

### Scenario Setup

* **Total Emails:** 100
* **Spam Emails:** 20 (out of 100)
* **Non-Spam (Ham) Emails:** 80 (out of 100)
* **Feature:** Presence of the word "lottery".
    * Spam emails containing "lottery": 14 (out of 20 spam emails)
    * Non-spam emails containing "lottery": 10 (out of 80 non-spam emails)

**Problem:** What is the probability that an email is spam, given that it contains the word "lottery"? In other words, we want to find $P(\text{Spam | Lottery})$.

### Intuitive Approach

We are only interested in emails that contain the word "lottery".

* **Emails containing "lottery" that are spam:** 14
* **Emails containing "lottery" that are NOT spam:** 10
* **Total emails containing "lottery":** $14 + 10 = 24$

The probability of an email being spam given it contains "lottery" is the number of spam emails with "lottery" divided by the total number of emails with "lottery":

$$P(\text{Spam | Lottery}) = \frac{14}{24} = \frac{7}{12} \approx 0.583$$

### Applying Bayes' Theorem Formula

Let's define our events:
* **A:** Email is spam
* **A':** Email is not spam
* **B:** Email contains the word "lottery"

The Bayes' Theorem formula is:

$$P(A|B) = \frac{P(A) \cdot P(B|A)}{[P(A) \cdot P(B|A)] + [P(A') \cdot P(B|A')]}$$

Let's calculate each component:

* **P(A) - Prior Probability of Spam:**
    * This is the overall probability of an email being spam before considering any features.
    * $P(\text{Spam}) = \frac{\text{Number of Spam Emails}}{\text{Total Emails}} = \frac{20}{100} = 0.2$

* **P(A') - Probability of Not Spam:**
    * $P(\text{Not Spam}) = 1 - P(\text{Spam}) = 1 - 0.2 = 0.8$

* **P(B|A) - Probability of "lottery" given Spam:**
    * This is the probability that a spam email contains the word "lottery".
    * $P(\text{Lottery | Spam}) = \frac{\text{Spam Emails with "lottery"}}{\text{Total Spam Emails}} = \frac{14}{20} = 0.7$

* **P(B|A') - Probability of "lottery" given Not Spam:**
    * This is the probability that a non-spam email contains the word "lottery".
    * $P(\text{Lottery | Not Spam}) = \frac{\text{Non-Spam Emails with "lottery"}}{\text{Total Non-Spam Emails}} = \frac{10}{80} = 0.125$

Now, plug these values into the Bayes' Theorem formula:

$$P(\text{Spam | Lottery}) = \frac{P(\text{Spam}) \cdot P(\text{Lottery | Spam})}{[P(\text{Spam}) \cdot P(\text{Lottery | Spam})] + [P(\text{Not Spam}) \cdot P(\text{Lottery | Not Spam})]}$$

$$P(\text{Spam | Lottery}) = \frac{0.2 \cdot 0.7}{(0.2 \cdot 0.7) + (0.8 \cdot 0.125)}$$

$$P(\text{Spam | Lottery}) = \frac{0.14}{0.14 + 0.1}$$

$$P(\text{Spam | Lottery}) = \frac{0.14}{0.24} \approx 0.583$$

Both the intuitive approach and the formula-based approach yield the same result, demonstrating the application of Bayes' Theorem in practical scenarios like spam classification.

## Prior vs. Posterior Probabilities

In probability, especially with Bayes' Theorem, it's crucial to understand the concepts of prior and posterior probabilities.

* **Prior Probability ($$P(A)$$):** This is the initial probability of an event occurring *before* any new information or evidence is considered. It represents our baseline belief or knowledge about the event.
* **Event (E):** This is the new piece of information or evidence that becomes available.
* **Posterior Probability ($$P(A|E)$$):** This is the updated probability of the event occurring *after* considering the new information from the event. It reflects a more refined or accurate estimate based on the observed evidence.

The posterior probability is generally a better estimation than the prior because it incorporates additional relevant information.

### Examples

1.  **Spam Email Classification:**
    * Prior ($$P(\text{Spam})$$): The initial probability that an email is spam, without looking at its content. If 20 out of 100 emails are spam, then $P(\text{Spam}) = 20/100 = 0.2$ (or 20%). This is the base rate.
    * Event (E): The email contains the word "lottery".
    * Posterior ($$P(\text{Spam | Lottery})$$): The updated probability that an email is spam, *given* that it contains the word "lottery". In the previous example, this was calculated as $14/24 \approx 0.583$. This is a more accurate assessment because we used specific content information.

2.  **Medical Diagnosis:**
    * Prior ($$P(\text{Sick})$$): The overall prevalence of the disease in the population. For instance, if 1 in 10,000 people are sick, then $P(\text{Sick}) = 1/10,000 = 0.0001$.
    * Event (E): You test positive for the disease.
    * Posterior ($$P(\text{Sick | Diagnosed Positive})$$): The updated probability that you are sick, *given* that you tested positive. As seen, this was surprisingly low due to the rarity of the disease and the false positive rate.

3.  **Rolling Two Dice:**
    * Scenario: You roll two standard six-sided dice.
    * Prior ($$P(\text{Sum is 10})$$): The probability that the sum of the values is 10. The possible outcomes for a sum of 10 are (4,6), (5,5), (6,4). There are $6 \times 6 = 36$ total possible outcomes.
        * $P(\text{Sum is 10}) = 3/36 = 1/12$.
    * Event (E): The first die is a 6.
    * Posterior ($$P(\text{Sum is 10 | First Die is 6})$$): Now, considering the first die is a 6, the possible outcomes are (6,1), (6,2), (6,3), (6,4), (6,5), (6,6) - 6 possibilities. Among these, only (6,4) results in a sum of 10.
        * $P(\text{Sum is 10 | First Die is 6}) = 1/6$. The probability changes significantly with new information.

4.  **Coin Tosses:**
    * Scenario: You toss two fair coins.
    * Prior ($$P(\text{Both Heads})$$): The probability that both coins land on heads. The possible outcomes are HH, HT, TH, TT.
        * $P(\text{Both Heads}) = 1/4$.
    * Event (E): The first coin lands on heads.
    * Posterior ($$P(\text{Both Heads | First Coin is Heads})$$): Given the first coin is heads, the possible outcomes are HH, HT.
        * $P(\text{Both Heads | First Coin is Heads}) = 1/2$.
     
## Naïve Bayes Classifier

While a single word (like "lottery") can help classify spam, real-world spam detection uses many words. Combining the probabilities of multiple words using standard Bayes' Theorem can be problematic, leading to the need for the "Naïve Bayes" assumption.

### Challenges with Multiple Features (Words)

When we want to calculate the probability of an email being spam given multiple words (e.g., "lottery" and "winning"), say $P(\text{Spam | Lottery}, \text{Winning})$, directly applying Bayes' Theorem would require:

$$P(\text{Spam | Lottery}, \text{Winning}) = \frac{P(\text{Spam}) \cdot P(\text{Lottery}, \text{Winning | Spam})}{P(\text{Lottery}, \text{Winning})}$$

The challenge arises with the terms $P(\text{Lottery}, \text{Winning | Spam})$ and $P(\text{Lottery}, \text{Winning})$. These represent the probability of an email containing *both* "lottery" and "winning" given it's spam (or any email).
* **Data Scarcity:** If we have many words (e.g., 100 words), finding emails in our dataset that contain *all* 100 specific words becomes extremely rare, possibly resulting in zero counts (e.g., 0/0), which makes the calculation impossible.

### The Naïve Assumption

To overcome this, the **Naïve Bayes** classifier makes a simplifying assumption: **the features (words) are conditionally independent given the class (spam or not spam)**.

* **Conditional Independence:** This means that the probability of seeing one word in an email (e.g., "lottery") does not affect the probability of seeing another word (e.g., "winning"), assuming we already know if the email is spam or not.
* **Real-world Applicability:** While this assumption is often *not* strictly true in natural language (words are dependent; "good" often implies "morning"), it often yields surprisingly good results in practice, making Naïve Bayes a powerful and efficient algorithm.

### Naïve Bayes Formula for Multiple Features

Under the independence assumption, the probability of an intersection becomes the product of individual probabilities:

$P(\text{Word}_1, \text{Word}_2, \dots, \text{Word}_n | \text{Class}) = P(\text{Word}_1 | \text{Class}) \cdot P(\text{Word}_2 | \text{Class}) \cdot \dots \cdot P(\text{Word}_n | \text{Class})$

So, for the spam example with "lottery" and "winning":

$$P(\text{Spam | Lottery}, \text{Winning}) = \frac{P(\text{Spam}) \cdot P(\text{Lottery | Spam}) \cdot P(\text{Winning | Spam})}{[P(\text{Spam}) \cdot P(\text{Lottery | Spam}) \cdot P(\text{Winning | Spam})] + [P(\text{Not Spam}) \cdot P(\text{Lottery | Not Spam}) \cdot P(\text{Winning | Not Spam})]}$$

### Example Calculation with "Lottery" and "Winning"

Let's use the previous data and add new data for "winning":

* **Total Emails:** 100
* **Spam Emails:** 20
* **Non-Spam (Ham) Emails:** 80

**Prior Probabilities:**
* $P(\text{Spam}) = 20/100 = 0.2$
* $P(\text{Ham}) = 80/100 = 0.8$

**Conditional Probabilities (from previous and new data):**

* **For "lottery":**
    * $P(\text{Lottery | Spam}) = 14/20 = 0.7$
    * $P(\text{Lottery | Ham}) = 10/80 = 0.125$
* **For "winning":**
    * Spam emails containing "winning": 15 (out of 20 spam emails)
    * Non-spam emails containing "winning": 8 (out of 80 non-spam emails)
    * $P(\text{Winning | Spam}) = 15/20 = 0.75$
    * $P(\text{Winning | Ham}) = 8/80 = 0.1$

Now, apply the Naïve Bayes formula:

$$P(\text{Spam | Lottery, Winning}) = \frac{0.2 \cdot 0.7 \cdot 0.75}{(0.2 \cdot 0.7 \cdot 0.75) + (0.8 \cdot 0.125 \cdot 0.1)}$$

$$P(\text{Spam | Lottery, Winning}) = \frac{0.105}{0.105 + 0.01}$$

$$P(\text{Spam | Lottery, Winning}) = \frac{0.105}{0.115} \approx 0.913$$

The probability of an email being spam, given that it contains both "lottery" and "winning", is approximately 91.3%. This much higher probability demonstrates the power of combining multiple features using the Naïve Bayes assumption.

## Role of Probability in Machine Learning

Probability is a foundational concept in machine learning, underpinning various algorithms and applications. Machine learning often involves calculating probabilities to make predictions, classifications, or generate new data.

### Supervised Machine Learning and Conditional Probabilities

Many supervised machine learning tasks revolve around calculating **conditional probabilities** of an output given certain input features. The goal is to build a model that estimates $P(\text{Output} | \text{Features})$.

* **Spam Detection:**
    * **Goal:** Determine if an email is spam.
    * **Conditional Probability:** $P(\text{Spam} | \text{Words, Recipients, Attachments, ...})$
    * A classifier calculates this probability based on email features.

* **Sentiment Analysis:**
    * **Goal:** Determine if text expresses a happy or sad sentiment.
    * **Conditional Probability:** $P(\text{Happy} | \text{Words in text})$
    * The model learns to associate words with sentiments to predict the probability of a happy (or sad) sentiment.

* **Image Recognition (e.g., Cat Detector):**
    * **Goal:** Identify if an image contains a specific object (e.g., a cat).
    * **Conditional Probability:** $P(\text{Cat in Image} | \text{Pixels of Image})$
    * A trained model takes image pixels as input and outputs the probability of a cat being present. A high probability (e.g., 0.9) suggests a cat, while a low probability (e.g., 0.1) suggests no cat.

* **Medical Diagnosis:**
    * **Goal:** Predict a patient's health status.
    * **Conditional Probability:** $P(\text{Healthy} | \text{Symptoms, Demographics, Medical History, ...})$
    * A model uses patient data to estimate the likelihood of being healthy or having a particular condition.

### Generative Machine Learning and Maximizing Probabilities

Another significant area where probability is central is **generative machine learning**, a part of unsupervised learning. Here, the aim is to create new data that resembles real data. This is often achieved by **maximizing the probability** that the generated data has certain characteristics.

* **Image Generation (e.g., Human Faces):**
    * **Goal:** Generate realistic images (e.g., human faces) from random noise.
    * **Probability Maximization:** The model learns to generate pixel combinations that maximize the probability of forming a realistic human face, $P(\text{Realistic Face} | \text{Generated Pixels})$. Advanced models like StyleGAN can produce incredibly convincing synthetic faces.

* **Text Generation:**
    * **Goal:** Generate coherent and sensical text.
    * **Probability Maximization:** The model learns to arrange words in sequences that maximize the probability of forming grammatically correct and meaningful sentences or paragraphs, $P(\text{Sensical Text} | \text{Generated Words})$.

### Bayes' Theorem in the Context of Machine Learning

As shown in previous examples (spam detection and medical diagnosis), Bayes' Theorem provides a structured way to update probabilities.

* **Prior:** Initial probability ($P(A)$) based on general knowledge (e.g., overall spam rate).
* **Event:** New information or evidence ($E$) (e.g., presence of a specific word).
* **Posterior:** Updated probability ($P(A|E)$) that incorporates the new information, making the prediction more accurate.

In essence, machine learning algorithms often learn to estimate these probabilities from data, allowing them to make informed decisions or generate realistic content.

## Random Variables

* A **random variable** is a variable that can take on multiple values, unlike deterministic variables (e.g., $x=3$) that always have the same value.
* The values a random variable takes are associated with **uncertain outcomes** of an experiment.

### Examples of Random Variables

* **Coin Flip**: If $X$ is the number of heads in a single coin toss:
    * $X = 1$ if heads (with $P(X=1) = 0.5$)
    * $X = 0$ if tails (with $P(X=0) = 0.5$)
* **10 Coin Tosses**: If $X$ is the number of heads in 10 coin tosses, $X$ can range from 0 to 10.
    * $P(X=0)$ (all tails) $= (0.5)^{10}$.
    * $P(X=10)$ (all heads) $= (0.5)^{10}$.
    * The probability of getting a specific number of heads (e.g., $X=5$) is typically higher than extreme outcomes (e.g., $X=0$ or $X=10$).
* **Other Examples**:
    * Number of 1s when rolling multiple dice.
    * Number of sick patients in a group.
    * Wait time until the next bus arrives.
    * Height of a gymnast's jump.
    * Number of defective products in a shipment.
    * Amount of rain in a month.

### Types of Random Variables

There are two main types of random variables:

* **Discrete Random Variables**:
    * Can take a **countable number of values**. This means the possible values can be listed (e.g., 0, 1, 2, 3, ...).
    * Examples: Number of heads in coin flips, number of children in a family, number of times you flip a coin until you get heads (can be 1, 2, 3, ..., an infinite but countable list).
* **Continuous Random Variables**:
    * Can take an **infinite number of values** within an entire interval. The possible values cannot be listed.
    * Examples: Wait time (e.g., 1 minute, 1.01 minutes, 1.0001 minutes), height, temperature, amount of rain.

### Random vs. Deterministic Variables

* **Deterministic Variables**:
    * Always take the same value once defined (e.g., $x=2$, input of $f(x) = x^2$).
    * Associated with a **fixed outcome**.
* **Random Variables**:
    * Can take many values.
    * Associated with an **uncertain outcome**.
 
## Probability Distribution

A **probability distribution** shows all possible scenarios of an experiment and the probability of each scenario occurring. For a random variable, it maps each possible value of the random variable to its probability.

### Example: Tossing Three Coins

Consider flipping three coins, and our random variable $X$ is the **number of heads**.
* **Total possible outcomes**: $2^3 = 8$
    * **0 Heads (TTT)**: 1 way. $P(X=0) = 1/8$
    * **1 Head (HTT, THT, TTH)**: 3 ways. $P(X=1) = 3/8$
    * **2 Heads (HHT, HTH, THH)**: 3 ways. $P(X=2) = 3/8$
    * **3 Heads (HHH)**: 1 way. $P(X=3) = 1/8$

This can be visualized as a histogram, where the height of each bar represents the probability. 

### Example: Tossing Four Coins

Let $X$ be the number of heads in four coin tosses.
* **Total possible outcomes**: $2^4 = 16$
    * **0 Heads**: 1 way. $P(X=0) = 1/16$
    * **1 Head**: 4 ways. $P(X=1) = 4/16$
    * **2 Heads**: 6 ways. $P(X=2) = 6/16$
    * **3 Heads**: 4 ways. $P(X=3) = 4/16$
    * **4 Heads**: 1 way. $P(X=4) = 1/16$

## Probability Mass Function (PMF)

For a **discrete random variable** $X$, its **probability mass function (PMF)**, denoted as $p(x)$ or $P(X=x)$, gives the probability that the random variable $X$ takes on a specific value $x$.

### Requirements for a PMF

A function $p(x)$ can be a valid PMF if it satisfies these two conditions:
1.  **Non-negativity**: The probability of any value must be non-negative.
    $p(x) \ge 0$ for all possible values of $x$.
2.  **Summation to One**: The sum of all probabilities for all possible values of the random variable must equal 1.
    $\sum_{\text{all } x} p(x) = 1$

### Binomial Distribution

The patterns observed in the number of heads in a fixed number of coin tosses (like the examples above) are characteristic of a specific type of probability distribution called the **Binomial Distribution**. This distribution models the number of successes in a fixed number of independent Bernoulli trials (experiments with only two outcomes, like coin flips).

The **binomial distribution** is a fundamental discrete probability distribution. It's used to model the number of successes in a fixed number of independent trials, where each trial has only two possible outcomes (success or failure) and the probability of success remains constant.

## Binomial Coefficient

To calculate the probability of a specific number of successes, we first need to determine the number of ways those successes can occur. This is done using the **binomial coefficient**, often read as "n choose k".

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

* $n$ represents the **total number of trials** (e.g., coin tosses).
* $k$ represents the **number of successes** (e.g., number of heads).

This formula counts the number of unique combinations where you have $k$ successes in $n$ trials.

### Properties of the Binomial Coefficient

* $\binom{n}{k} = \binom{n}{n-k}$: Obtaining $k$ successes is equivalent to obtaining $(n-k)$ failures. For example, getting 2 heads in 5 flips is the same as getting 3 tails in 5 flips.

## Binomial Probability Mass Function (PMF)

The **probability mass function (PMF)** of a binomial distribution gives the probability of obtaining exactly $x$ successes in $n$ trials, where the probability of success in a single trial is $p$.

The PMF is given by:

$$P(X=x) = \binom{n}{x} p^x (1-p)^{n-x}$$

* $X$ is the random variable representing the number of successes.
* $x$ is the specific number of successes we are interested in (where $x = 0, 1, ..., n$).
* $n$ is the total number of trials.
* $p$ is the probability of success in a single trial.
* $(1-p)$ is the probability of failure in a single trial (often denoted as $q$).

We denote that a random variable $X$ follows a binomial distribution with parameters $n$ and $p$ as $X \sim B(n, p)$.

### Examples

#### Example 1: Probability of 2 Heads in 5 Coin Tosses

Let's find the probability of getting exactly 2 heads in 5 fair coin tosses.
Here, $n=5$ (number of tosses) and $p=0.5$ (probability of heads). We want to find $P(X=2)$.

1.  **Calculate the binomial coefficient**:

$$
\binom{5}{2} = \frac{5!}{2!(5-2)!} = \frac{5!}{2!3!} = \frac{5 \times 4 \times 3 \times 2 \times 1}{(2 \times 1)(3 \times 2 \times 1)} = \frac{120}{12} = 10
$$

There are 10 ways to get 2 heads in 5 tosses.

2.  **Calculate the probability of one specific sequence**:
    The probability of one specific sequence with 2 heads and 3 tails (e.g., HHTTT) is $p^2 (1-p)^{5-2} = (0.5)^2 (0.5)^3 = 0.25 \times 0.125 = 0.03125$.

3.  **Multiply to get the total probability**:
    $P(X=2) = 10 \times (0.5)^2 (0.5)^3 = 10 \times 0.03125 = 0.3125$.

#### Example 2: Probability of 3 Ones when rolling a die 5 times

Consider rolling a standard six-sided die 5 times and wanting to find the probability of getting exactly 3 ones.
This can be thought of as a biased coin flip: "success" is rolling a 1, and "failure" is rolling anything else.
* $n=5$ (number of rolls)
* $p=1/6$ (probability of rolling a 1)
* $x=3$ (number of ones we want)

Using the PMF:
$$P(X=3) = \binom{5}{3} \left(\frac{1}{6}\right)^3 \left(1-\frac{1}{6}\right)^{5-3}$$

$$P(X=3) = \binom{5}{3} \left(\frac{1}{6}\right)^3 \left(\frac{5}{6}\right)^2$$

First, calculate $\binom{5}{3}$:
$$\binom{5}{3} = \frac{5!}{3!(5-3)!} = \frac{5!}{3!2!} = \frac{120}{(6)(2)} = 10$$

Now, substitute back into the PMF:
$$P(X=3) = 10 \times \left(\frac{1}{6}\right)^3 \times \left(\frac{5}{6}\right)^2$$

$$P(X=3) = 10 \times \frac{1}{216} \times \frac{25}{36}$$

$$P(X=3) = \frac{10 \times 25}{216 \times 36} = \frac{250}{7776} \approx 0.03215$$

#### Example 3: Parameters for rolling a die 10 times and recording the number of 1s

If we roll a die 10 times and record the number of 1s, this is a binomial distribution.
* **Number of trials ($n$)**: 10 (since we roll the die 10 times).
* **Probability of success ($p$)**: $1/6$ (since the probability of rolling a 1 is $1/6$).

So, the parameters for this binomial distribution are $n=10$ and $p=1/6 \approx 0.1667$.

### Shape of the PMF

The shape of the binomial PMF depends on the value of $p$:
* If $p=0.5$ (e.g., a fair coin), the PMF will be **symmetrical**.
* If $p < 0.5$, the PMF will be **skewed to the left** (higher probabilities for smaller numbers of successes).
* If $p > 0.5$, the PMF will be **skewed to the right** (higher probabilities for larger numbers of successes).

The concept of **combinations**, specifically "n choose k," is fundamental to understanding the binomial distribution. It provides the number of ways to select $k$ items from a set of $n$ distinct items *without regard to the order* of selection.

## Derivation of the Binomial Coefficient

### Permutations (Ordered Selection)

When we select $k$ items from $n$ items and the **order matters**, these are called permutations.
* For the first selection, there are $n$ options.
* For the second selection, there are $n-1$ options (since one item has already been picked).
* For the third selection, there are $n-2$ options.
* ...
* For the $k$-th selection, there are $n-(k-1) = n-k+1$ options.

The total number of ordered ways to pick $k$ items from $n$ is the product:
$$n \times (n-1) \times (n-2) \times \dots \times (n-k+1)$$

This can be expressed using factorials as:
$$\frac{n!}{(n-k)!}$$

### Accounting for Repetitions (Unordered Selection)

The above formula counts sets of items multiple times if the order is different (e.g., picking item A then B is distinct from B then A). To get **unordered sets** (combinations), we need to divide by the number of ways to order the $k$ chosen items.

The number of ways to order $k$ distinct items is $k!$ (k-factorial), which is the product $k \times (k-1) \times \dots \times 1$.

### The Binomial Coefficient Formula

By dividing the number of ordered selections by the number of ways to order the $k$ selected items, we get the binomial coefficient:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

This formula represents the number of ways to choose $k$ elements from a set of $n$ elements in an unordered way.

### Special Case: Zero Factorial

* $0! = 1$ (by definition).
* This ensures that $\binom{n}{0} = \frac{n!}{0!(n-0)!} = \frac{n!}{1 \cdot n!} = 1$. This makes sense, as there's only one way to choose zero items from a set (by choosing nothing).

## Binomial Distribution with Biased Coins

When dealing with a **biased coin** (or any trial where the probability of "success" $p$ is not 0.5), the individual outcomes are no longer equally likely.

Consider flipping a coin $n$ times where:
* $P(\text{Heads}) = p$
* $P(\text{Tails}) = (1-p)$

If we want to find the probability of getting exactly $k$ heads:
1.  **Probability of a specific sequence**: For any specific sequence with $k$ heads and $(n-k)$ tails (e.g., HH...H T...T), the probability is $p^k \cdot (1-p)^{n-k}$.
2.  **Number of possible sequences**: We use the binomial coefficient $\binom{n}{k}$ to find all the different orderings of $k$ heads and $(n-k)$ tails.

Combining these, the **Probability Mass Function (PMF)** for the binomial distribution with a biased coin is:

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

This formula allows us to calculate the probability of observing exactly $k$ successes in $n$ trials, even when the probability of success $p$ is not 0.5.



For example, if you flip a coin 5 times with $P(\text{Heads})=0.3$ and $P(\text{Tails})=0.7$:
* $P(X=0 \text{ heads}) = \binom{5}{0} (0.3)^0 (0.7)^5 = 1 \times 1 \times (0.7)^5 = 0.16807$
* $P(X=1 \text{ head}) = \binom{5}{1} (0.3)^1 (0.7)^4 = 5 \times 0.3 \times 0.2401 = 0.36015$
* $P(X=2 \text{ heads}) = \binom{5}{2} (0.3)^2 (0.7)^3 = 10 \times 0.09 \times 0.343 = 0.3087$
* And so on.

As $p$ deviates from 0.5, the histogram (PMF plot) of the binomial distribution will become **asymmetrical** or **skewed**, reflecting the higher probability of outcomes biased towards the more likely result (heads or tails).

## Bernoulli Distribution

The **Bernoulli distribution** is a fundamental discrete probability distribution that models a single experiment with only two possible outcomes: **success** or **failure**. It has a single parameter, $p$, which represents the probability of success.

### Key Characteristics

* **Single Trial**: The experiment consists of only one trial.
* **Two Outcomes**: The outcome is either a "success" (typically denoted by 1) or a "failure" (typically denoted by 0).
* **Probability of Success ($p$)**: This is the probability that the outcome is a success.
* **Probability of Failure ($1-p$)**: This is the probability that the outcome is a failure.

### Examples

1.  **Coin Flip**:
    * **Experiment**: Flipping a single coin.
    * **Success**: Getting a head ($X=1$).
    * **Failure**: Getting a tail ($X=0$).
    * **Parameter $p$**: Probability of getting a head (e.g., $p=0.5$ for a fair coin).

2.  **Rolling a Die for a Specific Number**:
    * **Experiment**: Rolling a single six-sided die.
    * **Success**: Rolling a '1' ($X=1$).
    * **Failure**: Rolling any number other than '1' ($X=0$).
    * **Parameter $p$**: Probability of rolling a '1', which is $1/6$. So, $p=1/6$, and $1-p=5/6$.

3.  **Patient Health Status**:
    * **Experiment**: Observing a single patient.
    * **Success**: The patient is sick ($X=1$). (While "sick" might not conventionally be a "success", in the context of counting sick patients, it's the outcome we're interested in measuring).
    * **Failure**: The patient is healthy ($X=0$).
    * **Parameter $p$**: Probability that the patient is sick.

### Probability Mass Function (PMF)

The PMF of a Bernoulli distribution is given by:

$$
P(X=x) = \begin{cases}
p & \text{if } x=1 \\
1-p & \text{if } x=0
\end{cases}
$$

This can be written compactly as:
$$P(X=x) = p^x (1-p)^{1-x} \quad \text{for } x \in \{0, 1\}$$

You've hit on a crucial distinction in probability: **discrete vs. continuous distributions**. It's all about whether the possible values a random variable can take can be listed or not.

## Discrete Distributions

In **discrete distributions**, the possible outcomes of a random variable can be listed or counted. Think of them as distinct, separate points on a number line.

* **Example 1: Number of Heads**: If you toss a coin 3 times, the number of heads ($X$) can only be 0, 1, 2, or 3. You can clearly list these values.
* **Example 2: Population**: The number of people in a town can be 0, 1, 2, 3, up to millions. Even though it can be a large number, it's still countable.

For discrete distributions, we use a **Probability Mass Function (PMF)**, where the height of each bar in a histogram directly represents the probability of that specific outcome. The sum of the heights of all bars must equal 1.

## Continuous Distributions

In **continuous distributions**, the random variable can take on any value within a given interval. You cannot list all the possible outcomes because there are infinitely many, uncountably many, values between any two points.

* **Example 1: Waiting Time**: If you're waiting for a bus, the waiting time ($X$) could be 1 minute, 1.01 minutes, $\pi$ minutes (3.14159...), or any value within a range. You can't make a list of all possible waiting times.
* **Example 2: Height**: A person's height isn't just 1.70m or 1.71m; it can be 1.705m, 1.7056m, and so on.

### The Problem with Point Probabilities in Continuous Distributions

With continuous random variables, the probability of the variable taking on any **exact, single value is zero**.
* **Why?** Because there are infinitely many possible values. If each had a non-zero probability, the sum of all probabilities would vastly exceed 1.

### Probability Density Function (PDF)

Since we can't assign probabilities to single points, for continuous distributions, we talk about the probability of a random variable falling within a **range or interval**. This is where the **Probability Density Function (PDF)** comes in.

Imagine the histogram approach from discrete distributions, but with infinitely many, infinitesimally thin bars. As the width of the intervals shrinks to zero, the histogram becomes a smooth curve.

* **Area Under the Curve**: For a continuous distribution, the **area under the curve** of the PDF over a specific interval represents the probability that the random variable falls within that interval.
* **Total Area**: Just like the sum of probabilities in a discrete distribution, the **total area under the entire PDF curve must equal 1**. This signifies that the random variable must take some value within its entire possible range.

In essence, the height of the PDF curve itself at a given point doesn't represent probability directly (it's "probability density"), but rather indicates where values are more likely to fall. The actual probability comes from integrating (finding the area under) the curve over an interval.

## Probability Density Function (PDF)

For **continuous random variables**, we cannot talk about the probability of the variable taking on an **exact single value**, because this probability is always **zero**. Instead, we talk about the probability of the variable falling within a **specific interval**. This is where the **Probability Density Function (PDF)**, denoted by $f(x)$ or $f_X(x)$, comes into play.

### Intuition: From Histograms to Curves

Imagine a continuous variable, like the duration of a phone call, which can take any value (e.g., 2 minutes, 2.01 minutes, 2.0005 minutes).

1.  **Discretizing Intervals**: If we divide the total possible duration (say, 0 to 5 minutes) into small, equal-sized intervals (e.g., 1-minute intervals), we can create a histogram where the height of each bar represents the probability that the call falls within that interval. The **area** of each bar (height $\times$ width) is the probability.
    
2.  **Making Intervals Finer**: As we make these intervals smaller and smaller (e.g., 30-second, then 15-second intervals), the bars become thinner and more numerous. The probability for each individual narrow interval decreases, but the total area under all bars still sums to 1.
    
3.  **The Continuous Limit**: If we make the intervals infinitesimally small, the tops of the bars form a smooth curve. This curve is the **Probability Density Function (PDF)**.
    

### Calculating Probabilities with a PDF

For a continuous random variable $X$, the probability that $X$ falls within an interval $[a, b]$ is given by the **area under the PDF curve between $a$ and $b$**. Mathematically, this is calculated using **integration**:

$$P(a \le X \le b) = \int_{a}^{b} f(x) \,dx$$

* The area represents the accumulation of probability over that interval.
* The probability of $X$ being *exactly* a specific value (e.g., $P(X=2)$) is 0 because a single point has no width, and thus no area under the curve.

### Requirements for a Valid PDF

A function $f(x)$ can be considered a valid PDF if it satisfies the following conditions:

1.  **Non-negativity**: The function must be non-negative for all possible values of $x$. You cannot have negative probabilities.
    $f(x) \ge 0$ for all $x \in \mathbb{R}$
2.  **Total Area of One**: The total area under the entire curve of the PDF must be equal to 1. This means the random variable must take on *some* value within its entire range of possibilities.
    $\int_{-\infty}^{\infty} f(x) \,dx = 1$

### Summary: Discrete vs. Continuous

| Feature           | Discrete Random Variables                       | Continuous Random Variables                      |
| :---------------- | :---------------------------------------------- | :----------------------------------------------- |
| **Outcomes** | Finite or countably infinite list of values     | Any value within an interval (uncountably infinite) |
| **Probability of single value** | $P(X=x) > 0$                           | $P(X=x) = 0$                                 |
| **Function type** | **Probability Mass Function (PMF)**, $p(x)$   | **Probability Density Function (PDF)**, $f(x)$ |
| **Probability calculation** | Sum of $p(x)$ for specific values      | Area under $f(x)$ (using integration)            |
| **Visual representation** | Histogram (bar heights are probabilities) | Smooth curve (area under curve is probability) |

## Cumulative Distribution Function (CDF)

The **Cumulative Distribution Function (CDF)** provides the probability that a random variable will take a value less than or equal to a specific point. It's a more convenient way to calculate probabilities compared to finding areas under a curve.

### CDF for Discrete Distributions

* **Cumulative Probability**: Tells the probability an event happened before a reference point.
* For discrete distributions, the CDF is a **step function** with jumps.
    * The height of each jump corresponds to the **probability mass** at that specific value.
    * If a point is not a possible outcome, the CDF remains flat, indicating no probability accumulation at that point.
* **Example (Phone Call Duration)**:
    * Probability of a call between 0 and 1 minute is the value of the CDF at 1 minute.
    * Probability of a call between 0 and 2 minutes is the sum of probabilities for 0-1 and 1-2 minutes. This is represented by the CDF value at 2 minutes.

### CDF for Continuous Distributions

* For continuous distributions, the CDF is a **continuous and smooth function**.
* It's obtained by **integrating** the Probability Density Function (PDF) from the beginning up to a certain point.
* Since individual points in a continuous distribution have zero probability mass, there are no jumps in the CDF.
* The area under the PDF up to a certain point $x$ is equal to the height of the CDF at that point $x$.

### General Properties of CDF

The CDF, denoted as $F_X(x)$, is defined as the probability that a random variable $X$ is less than or equal to some value $x$:

$$F_X(x) = P(X \le x)$$

This function is defined for all real numbers from $-\infty$ to $+\infty$.

* **Range**: All CDF values are between 0 and 1, inclusive: $0 \le F_X(x) \le 1$. This is because it represents a probability.
* **Left Endpoint**: As $x$ approaches negative infinity, the CDF approaches 0: $\lim_{x \to -\infty} F_X(x) = 0$. This means there's no accumulated probability before any possible outcomes.
* **Right Endpoint**: As $x$ approaches positive infinity, the CDF approaches 1: $\lim_{x \to \infty} F_X(x) = 1$. This signifies that all possible probabilities have been accumulated.
* **Non-decreasing**: The CDF can never decrease. As $x$ increases, the accumulated probability can only stay the same or increase, never decrease (as probabilities are non-negative).
    * Mathematically, if $a < b$, then $F_X(a) \le F_X(b)$.

### Relationship between PDF/PMF and CDF

* **PDF (Probability Density Function)** for continuous variables and **PMF (Probability Mass Function)** for discrete variables are represented by lowercase $f$. They describe the probability distribution at individual points or intervals.
* **CDF (Cumulative Distribution Function)** is represented by uppercase $F$. It describes the accumulated probability up to a certain point.
* PDF/PMF is always positive, and the total area/sum of probabilities is 1.
* CDF starts at 0, ends at 1, and is always non-decreasing.

### Usage

Both PDF/PMF and CDF provide the same information but in different forms. The choice of which to use depends on the convenience for a given calculation or analysis.
