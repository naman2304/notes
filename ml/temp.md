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
