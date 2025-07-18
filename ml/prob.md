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
