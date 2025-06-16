## [Week 4] Introduction to Decision Trees

Decision trees are powerful and widely used machine learning algorithms, particularly popular for winning competitions, even if they receive less academic attention than neural networks. They are a valuable tool for classification and regression.

### Running Example: Cat Classification

* **Problem:** Classify if an animal is a cat or not, based on a few features.
* **Training Data (10 examples: 5 cats, 5 dogs):**
    * **Features (X):**
        * `Ear Shape`: (categorical: pointy, floppy)
        * `Face Shape`: (categorical: round, not round)
        * `Whiskers`: (categorical: present, absent)
    * **Target (Y):** (binary: cat=1, not cat=0)
    * Initially, all features are binary/categorical. Later, continuous features will be discussed.

### What is a Decision Tree?

<img src="/metadata/decision_tree.png" width="600" />

A decision tree is a model that looks like a flowchart. It consists of:

* **Nodes:** The ovals or rectangles in the tree.
* **Root Node:** The topmost node where the classification process begins.
* **Decision Nodes:** Oval-shaped nodes (excluding the leaf nodes). They contain a feature and direct the flow down the tree based on that feature's value (e.g., "Ear Shape? Pointy -> Left, Floppy -> Right").
* **Leaf Nodes:** Rectangular-shaped nodes at the bottom. They represent the final prediction (e.g., "Cat" or "Not Cat").

### How a Decision Tree Makes a Prediction:

Let's classify a new animal: (Ear Shape: pointy, Face Shape: round, Whiskers: present)

1.  **Start at Root Node:** (e.g., "Ear Shape?")
2.  **Follow Branch:** If "pointy", go down the left branch to the next node.
3.  **Evaluate Next Feature:** At the new node (e.g., "Face Shape?"), check the animal's face shape.
4.  **Continue Down Tree:** If "round", follow the branch to a leaf node.
5.  **Prediction:** The leaf node indicates the final classification (e.g., "Cat").

### Learning a Decision Tree:

* There are many possible decision trees for a given dataset.
* The job of the **decision tree learning algorithm** is to select the tree that performs well on the training data and ideally generalizes well to new, unseen data (cross-validation and test sets).

The next video will delve into how an algorithm learns to construct a specific decision tree from a training set.

## Building a Decision Tree: The Learning Process

Building a decision tree involves iteratively splitting the training data based on features, aiming to create "pure" (single-class) leaf nodes.

### Overall Process:

1.  **Choose Root Node Feature:**
    * Start with the entire training set (e.g., 10 cat/dog examples).
    * Use an algorithm (discussed later) to decide which feature (e.g., "Ear Shape") provides the "best" initial split for the root node.
2.  **Split Data:**
    * Divide the training examples into subsets based on the value of the chosen feature (e.g., 5 "pointy ears" examples to the left branch, 5 "floppy ears" examples to the right branch).
3.  **Recursively Build Sub-trees (Repeat for each branch):**
    * For each new subset of data:
        * **Choose Next Feature to Split:** Again, use an algorithm to select the best feature to split on *within that subset*. (e.g., for "pointy ears" subset, choose "Face Shape").
        * **Split Subset:** Divide the current subset further based on the new feature's values (e.g., 4 "round face" examples to the left, 1 "not round face" example to the right).
        * **Create Leaf Nodes (Stopping Condition):** If a subset becomes "pure" (all examples belong to a single class, e.g., all 4 "round face" examples are cats), create a leaf node making that prediction. If not pure, continue splitting.
    * This process is applied recursively to all branches (e.g., after the left branch is done, build the right branch starting from "floppy ears").

### Key Decisions in Decision Tree Learning:

Two crucial decisions are made at various steps:

1.  **How to Choose Which Feature to Split On at Each Node?**
    * **Goal: Maximize Purity (or Minimize Impurity).** You want to find a feature that, when used for a split, results in child nodes that are as "pure" as possible (i.e., contain mostly examples of a single class, like all cats or all dogs).
    * **Example:** A "cat DNA" feature (if it existed) would be ideal, as it would create perfectly pure (100% cat, 0% cat) subsets.
    * **Challenge:** With real features (ear shape, face shape, whiskers), the algorithm must compare how well each feature splits the data into purer subsets. The feature that leads to the greatest "purity" gain (or greatest "impurity" reduction) is chosen.
    * **Next Video:** The concept of **entropy** will be introduced to measure impurity.

2.  **When to Stop Splitting (Create a Leaf Node)?**
    * **Pure Nodes:** Stop when a node contains only examples of a single class (e.g., all cats, or all dogs). This is the most natural stopping point.
    * **Maximum Depth:** Limit the maximum allowed depth of the tree. (Depth 0 is the root node, Depth 1 for its children, etc.). This prevents the tree from becoming too large/unwieldy and helps **reduce overfitting**.
    * **Minimum Purity Improvement:** Stop if splitting a node yields only a very small improvement in purity (or a small decrease in impurity). This again helps keep the tree smaller and **reduce overfitting**.
    * **Minimum Number of Examples:** Stop splitting if the number of training examples at a node falls below a certain threshold. This also helps keep the tree smaller and prevent overfitting to tiny subsets of data.

Decision tree algorithms can feel complicated due to these various refinements developed over time. However, these pieces work together to create effective learning algorithms. The next video will formally define **entropy** as a measure of impurity, a core concept for choosing optimal splits.

## Entropy: Measuring Purity (or Impurity)

In decision tree learning, we need a way to quantify how "pure" a set of examples is at a given node. **Entropy** is a common measure of **impurity**.

### Definition of Entropy

Given a set of examples:
* Let $p_1$ be the **fraction of positive examples** (e.g., cats, label $y=1$).
* Let $p_0 = 1 - p_1$ be the **fraction of negative examples** (e.g., dogs, label $y=0$).

The **entropy** $H(p_1)$ is defined as:
$$H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)$$or equivalently:$$H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)$$

* **Logarithm Base:** $\log_2$ (logarithm to base 2) is conventionally used, making the maximum entropy value 1.
* **Convention for $0 \log_2(0)$:** By convention, $0 \log_2(0)$ is taken to be $0$.

### Intuition and Examples:

* **Completely Pure Set (All one class):**
    * If all examples are cats ($p_1 = 1$, $p_0 = 0$): $H(1) = -1 \log_2(1) - 0 \log_2(0) = -1 \times 0 - 0 = 0$.
    * If all examples are dogs ($p_1 = 0$, $p_0 = 1$): $H(0) = -0 \log_2(0) - 1 \log_2(1) = 0 - 1 \times 0 = 0$.
    * **Result:** Entropy is **0** when the set is perfectly pure (contains only one class). This signifies zero impurity.

* **Maximally Impure Set (50-50 Mix):**
    * If there's an equal mix of cats and dogs ($p_1 = 0.5$, $p_0 = 0.5$): $H(0.5) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = -0.5(-1) - 0.5(-1) = 0.5 + 0.5 = 1$.
    * **Result:** Entropy is **1** (its maximum value) when the set is maximally impure (a 50-50 mix of classes).

* **Intermediate Impurity:**
    * If $p_1 = 5/6 \approx 0.83$ (5 cats, 1 dog): $H(0.83) \approx 0.65$. (Less impure than 50-50, more impure than all one class).
    * If $p_1 = 2/6 \approx 0.33$ (2 cats, 4 dogs): $H(0.33) \approx 0.92$. (More impure than 5 cats/1 dog, closer to 50-50).

### Entropy Curve:

The entropy function $H(p_1)$ forms a curve that starts at 0 (for $p_1=0$), rises to a peak of 1 (for $p_1=0.5$), and then falls back to 0 (for $p_1=1$).

### Other Impurity Measures:

While entropy is common, other functions like the **Gini criteria (or Gini impurity)** also measure impurity similarly (from 0 to 1) and are used in decision trees. For simplicity, this course focuses on entropy.

Now that we have a way to measure impurity (entropy), the next video will show how to use it to decide which feature to split on at each node of a decision tree.
