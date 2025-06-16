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
$$H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)$$ or equivalently: $$H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)$$

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

<img src="/metadata/entropy_curve.png" width="300" />

The entropy function $H(p_1)$ forms a curve that starts at 0 (for $p_1=0$), rises to a peak of 1 (for $p_1=0.5$), and then falls back to 0 (for $p_1=1$).

### Other Impurity Measures:

While entropy is common, other functions like the **Gini criteria (or Gini impurity)** also measure impurity similarly (from 0 to 1) and are used in decision trees. For simplicity, this course focuses on entropy.

Now that we have a way to measure impurity (entropy), the next video will show how to use it to decide which feature to split on at each node of a decision tree.

## Information Gain: Choosing the Best Split

When building a decision tree, the primary criterion for deciding which feature to split on at a given node is to choose the feature that leads to the **greatest reduction in entropy (impurity)**. This reduction is called **information gain**.

### How to Calculate Information Gain:

Let's illustrate with our cat classification example:

1.  **Calculate Entropy at the Root Node (before any split):**
    * For the initial set of examples at the node (e.g., the root node with all 10 examples: 5 cats, 5 dogs).
    * $p_{1}^{\text{root}} = \text{Fraction of cats at root} = 5/10 = 0.5$.
    * $H(\text{root}) = \text{Entropy}(p_{1}^{\text{root}}) = \text{Entropy}(0.5) = 1$. (Maximum impurity).

2.  **For each candidate feature to split on (e.g., Ear Shape, Face Shape, Whiskers):**
    *  **Hypothetically split the data** based on that feature. This creates left and right sub-branches.
    *  **Calculate $p_1$ and Entropy for each sub-branch:**
        * **Ear Shape Split (Pointy vs. Floppy):**
            * Left (Pointy): 5 examples total, 4 cats. $p_{1}^{\text{left}} = 4/5 = 0.8$. $\text{Entropy}(0.8) \approx 0.72$.
            * Right (Floppy): 5 examples total, 1 cat. $p_{1}^{\text{right}} = 1/5 = 0.2$. $\text{Entropy}(0.2) \approx 0.72$.
        * **Face Shape Split (Round vs. Not Round):**
            * Left (Round): 7 examples total, 4 cats. $p_{1}^{\text{left}} = 4/7 \approx 0.57$. $\text{Entropy}(0.57) \approx 0.99$.
            * Right (Not Round): 3 examples total, 1 cat. $p_{1}^{\text{right}} = 1/3 \approx 0.33$. $\text{Entropy}(0.33) \approx 0.92$.
        * **Whiskers Split (Present vs. Absent):**
            * Left (Present): 4 examples total, 3 cats. $p_{1}^{\text{left}} = 3/4 = 0.75$. $\text{Entropy}(0.75) \approx 0.81$.
            * Right (Absent): 6 examples total, 2 cats. $p_{1}^{\text{right}} = 2/6 \approx 0.33$. $\text{Entropy}(0.33) \approx 0.92$.

    *  **Calculate Weighted Average Entropy of the Split:** This accounts for the proportion of examples going into each sub-branch.
        * **For Ear Shape:**
            * $w^{\text{left}} = 5/10 = 0.5$ (proportion of examples with pointy ears)
            * $w^{\text{right}} = 5/10 = 0.5$ (proportion of examples with floppy ears)
            * Weighted Entropy = $(0.5 \times \text{Entropy}(0.8)) + (0.5 \times \text{Entropy}(0.2)) = (0.5 \times 0.72) + (0.5 \times 0.72) = 0.72$.
        * **For Face Shape:**
            * Weighted Entropy = $(7/10 \times \text{Entropy}(4/7)) + (3/10 \times \text{Entropy}(1/3)) = (0.7 \times 0.99) + (0.3 \times 0.92) = 0.97$.
        * **For Whiskers:**
            * Weighted Entropy = $(4/10 \times \text{Entropy}(3/4)) + (6/10 \times \text{Entropy}(2/6)) = (0.4 \times 0.81) + (0.6 \times 0.92) = 0.88$.

    *  **Calculate Information Gain:**
        $$\text{Information Gain} = \text{Entropy}(\text{root}) - \text{Weighted Average Entropy of Split}$$
        * **For Ear Shape:** $1 - 0.72 = 0.28$.
        * **For Face Shape:** $1 - 0.97 = 0.03$.
        * **For Whiskers:** $1 - 0.88 = 0.12$.

### Choosing the Best Split:

* The feature with the **highest information gain** is chosen for the split.
* In this example, "Ear Shape" has the highest information gain (0.28), so it would be chosen as the root node feature.
* **Why Information Gain?** It directly quantifies how much a split reduces the overall impurity of the dataset, leading to purer child nodes.

$$\text{Information gain} = H(p_1^{\text{root}}) - \left( w^{\text{left}} H(p_1^{\text{left}}) + w^{\text{right}} H(p_1^{\text{right}}) \right)$$

### Role in Stopping Criteria:

* Information gain is also used in stopping criteria: If the information gain from any potential split is below a certain threshold, the algorithm might decide not to split further, creating a leaf node instead. This helps control tree size and prevent overfitting.

The next video will integrate this information gain calculation into the overall decision tree building algorithm.

## Building a Decision Tree: The Overall Process

Building a decision tree involves a recursive process of splitting nodes based on features that provide the most information gain, until predefined stopping criteria are met.

### Overall Algorithm:

1.  **Start at Root Node:** Begin with all training examples at the root node.
2.  **Select Best Split Feature:**
    * Calculate the **information gain** for every possible feature (e.g., "Ear Shape," "Face Shape," "Whiskers").
    * Choose the feature that yields the **highest information gain**.
3.  **Perform Split:**
    * Divide the dataset into subsets based on the values of the chosen feature.
    * Create corresponding child branches (e.g., "pointy ears" branch, "floppy ears" branch).
    * Send the relevant training examples down each branch.
4.  **Recursively Build Sub-trees:**
    * **Repeat the splitting process** for each new branch (child node) created. The process is the same as building the main tree, but applied to a subset of data.
5.  **Stop Splitting (Stopping Criteria):** Stop the recursive splitting process for a branch when any of the following criteria are met:
    * **Node Purity:** All examples in the node belong to a single class (entropy is 0). This becomes a leaf node with a clear prediction.
    * **Maximum Depth:** The tree (or current branch) reaches a pre-defined `maximum depth`. This prevents overly complex trees and reduces overfitting.
    * **Minimum Information Gain:** The information gain from any potential further split is below a specified `threshold`. This avoids trivial splits.
    * **Minimum Examples per Node:** The number of training examples in the current node falls below a certain `threshold`. This also prevents overfitting to tiny subsets.

### Illustration of the Process:

* **Root Node:** All 10 examples. "Ear Shape" is chosen (highest info gain).
    * Splits into "Pointy Ears" (5 examples) and "Floppy Ears" (5 examples) branches.
* **Left Branch ("Pointy Ears"):** Focus on these 5 examples.
    * **Check stopping criteria:** Not pure yet (mix of cats and dogs).
    * **Select Best Split Feature:** (e.g., "Face Shape" is chosen after calculating information gain within this subset).
    * Splits into "Round Face" (4 examples) and "Not Round Face" (1 example) sub-branches.
    * **"Round Face" sub-branch:** All 4 examples are cats. **Stopping criteria met (purity).** Create a "Cat" leaf node.
    * **"Not Round Face" sub-branch:** All 1 example is a dog. **Stopping criteria met (purity).** Create a "Not Cat" leaf node.
* **Right Branch ("Floppy Ears"):** Similarly, recursively build this branch.
    * (e.g., "Whiskers" is chosen).
    * Splits into "Whiskers Present" and "Whiskers Absent" sub-branches, which then become pure leaf nodes.

### Recursive Algorithm:

Decision tree building is a classic example of a **recursive algorithm** in computer science. The main function to build a tree calls itself to build sub-trees on smaller subsets of the data.

### Key Parameters:

* **Maximum Depth:** Controls tree size and complexity. Larger depth increases risk of overfitting. Can be tuned via cross-validation, but open-source libraries often have good defaults.
* **Information Gain Threshold:** Controls when to stop splitting.
* **Minimum Examples per Node:** Controls when to stop splitting.

Understanding these parameters and their impact on tree size and overfitting is crucial for effective decision tree usage. After building, predictions are made by traversing the tree from root to leaf based on a new example's features.

The next videos will explore handling features with more than two categorical values and continuous-valued features.

## Handling Categorical Features with Many Values: One-Hot Encoding

So far, our decision tree examples have used features with only two discrete values (e.g., pointy/floppy, round/not round). This video introduces **one-hot encoding** as a method to handle categorical features that can take on *more than two* discrete values.

### The Problem: Categorical Features with $>2$ Values

Consider the `Ear Shape` feature, which can be `pointy`, `floppy`, or `oval`. If we directly use this feature, a decision tree would create three branches from a single node. While decision trees can technically handle this, one-hot encoding offers an alternative, especially useful for other algorithms.

### Solution: One-Hot Encoding

One-hot encoding transforms a single categorical feature with $K$ possible values into $K$ new **binary (0 or 1) features**.

* **Process:**
    1.  Identify all unique values (categories) a feature can take.
    2.  Create a new binary feature for each unique value.
    3.  For any given example, set the binary feature corresponding to its category to `1` ("hot"), and all other new binary features for that original categorical feature to `0`.

* **Example: Ear Shape (Pointy, Floppy, Oval) -> 3 New Features:**
    * **Original:** `Ear Shape: pointy`
    * **One-Hot Encoded:** `Pointy_Ears: 1, Floppy_Ears: 0, Oval_Ears: 0`
    * **Original:** `Ear Shape: oval`
    * **One-Hot Encoded:** `Pointy_Ears: 0, Floppy_Ears: 0, Oval_Ears: 1`

* **Benefit:** Each new feature is binary (0 or 1), making it directly compatible with the decision tree learning algorithm we've already discussed, without further modification.

### Applicability Beyond Decision Trees:

* One-hot encoding is a **general technique** for handling categorical features and is **also crucial for neural networks, linear regression, and logistic regression**. These algorithms typically expect numerical inputs.
    * By converting `Ear Shape`, `Face Shape`, and `Whiskers` into a list of binary features (e.g., `[Pointy_Ears, Floppy_Ears, Oval_Ears, Is_Face_Round, Whiskers_Present]`), the data becomes suitable for input into models that require numerical features.

One-hot encoding allows decision trees (and other algorithms) to process categorical features with multiple values efficiently. The next video will address how decision trees handle **continuous-valued features** (numerical features that can take on any value).
