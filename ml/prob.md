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
