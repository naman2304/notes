# Unsupervised Learning, Recommenders, Reinforcement Learning
In this course we will learn about
* Unsupervised learning
  * Clustering
  * Anomaly Detection
* Recommender Systems
* Reinforcement Learning

## What is Clustering?

**Clustering** is an **unsupervised learning algorithm** that automatically finds relationships or similarities among data points, grouping them into "clusters."

### Clustering vs. Supervised Learning:

* **Supervised Learning:** Given input features (x) and **labeled target outputs (y)**. The algorithm learns to map x to y (e.g., classifying data into "X"s and "O"s, learning a decision boundary).
* **Unsupervised Learning (Clustering):** Given only **input features (x)**, with **no labels (y)**. The algorithm is not told the "right answer." Instead, it aims to discover inherent structure or patterns in the data by grouping similar points together. A dataset might appear as just a scatter plot of dots, and clustering finds the natural groupings within those dots.

### Applications of Clustering:

* **News Article Grouping:** Automatically organizing similar news articles together (e.g., "Panda stories").
* **Market Segmentation:** Identifying different groups of customers or learners with similar motivations (e.g., "skill-growers," "career developers," "AI updated learners" at deeplearning.ai).
* **DNA Data Analysis:** Grouping individuals with similar genetic expression patterns.
* **Astronomical Data Analysis:** Grouping celestial bodies to identify galaxies or other coherent structures in space.

The next video will introduce the most commonly used clustering algorithm: the **k-means algorithm**.

## K-Means Clustering Algorithm

The K-means algorithm is an iterative, unsupervised learning method used to partition unlabeled data points into K distinct clusters.

### Algorithm Steps:

1.  **Initialization:**
    * Randomly guess the initial locations of **K cluster centroids**. These are typically represented by points on the plot (e.g., a red cross and a blue cross for K=2 clusters). The initial guesses are often not accurate.

2.  **Iterative Process (Repeats until convergence):**  
    K-means repeatedly performs two main steps:  

    *  **Assign Points to Cluster Centroids:**    
        * For *each* data point in the dataset, determine which of the K cluster centroids it is **closest to** (e.g., using Euclidean distance).
        * Assign that data point to the cluster represented by its closest centroid. (This is visually represented by coloring the points red or blue based on their assigned centroid).

    *  **Move Cluster Centroids:**    
        * For *each* of the K clusters:    
            * Calculate the **mean (average) position** of *all* data points currently assigned to that cluster.
            * Move the cluster centroid to this newly calculated mean position. (The red cross moves to the average of all red points, the blue cross moves to the average of all blue points).

3.  **Convergence:**  
    The algorithm **converges** when repeated application of these two steps no longer results in:
    * Any data point changing its cluster assignment (i.e., no points change color).
    * Any cluster centroid significantly changing its location.
    At this point, the clusters are stable.

### Example Illustration (K=2):

* **Initial:** Randomly place 2 centroids (red and blue crosses).
* **Iteration 1 (Assign):** Color each data point red or blue based on which initial centroid it's closer to.
* **Iteration 1 (Move):** Move each centroid to the center of its newly assigned colored points.
* **Iteration 2 (Assign):** Re-color points based on the *new* centroid locations. Some points might change color.
* **Iteration 2 (Move):** Move centroids again to the new centers of their assigned colored points.
* **Repeat:** Continue assigning and moving until no points change color and centroids stabilize.

In this example, K-means successfully identifies two distinct groups of points. The next video will formalize these steps into a mathematical algorithm.

## K-Means Clustering Algorithm: Detailed Steps

This video formalizes the K-means clustering algorithm, outlining its steps for implementation.

### Algorithm Steps:

1.  **Random Initialization of K Cluster Centroids:**
    * Randomly choose `K` initial data points from your dataset (or random locations within the data range) to serve as the initial cluster centroids. These are denoted $\mu_1, \mu_2, \dots, \mu_K$.
    * Each $\mu_k$ is a vector with the same dimensionality (number of features) as your data points $x^{(i)}$.

2.  **Iterative Optimization (Repeat until convergence):**  

    *  **Assignment Step (Assign Points to Closest Centroid):**  
        * For each training example $x^{(i)}$ (for $i=1, \dots, m$):  
            * Calculate the distance between $x^{(i)}$ and *each* cluster centroid $\mu_k$. The standard distance metric is **Euclidean distance** (or L2 norm).
            * $c^{(i)} = \text{index of } k \text{ that minimizes } ||x^{(i)} - \mu_k||^2$
            * Assign $x^{(i)}$ to the cluster whose centroid $\mu_k$ is closest. ($c^{(i)}$ stores the index of the assigned cluster). Minimizing the squared distance is often more convenient computationally.

    *  **Update Step (Move Centroids to Mean of Assigned Points):**  
        * For each cluster centroid $\mu_k$ (for $k=1, \dots, K$):  
            * Set $\mu_k$ to the **mean (average) of all training examples $x^{(i)}$ that were assigned to cluster $k$** in the assignment step.
            * $\mu_k = \frac{1}{\text{count}(x^{(i)} \text{ s.t. } c^{(i)}=k)} \sum_{i: c^{(i)}=k} x^{(i)}$

### Handling Empty Clusters (Corner Case):

* If a cluster has **zero training examples assigned to it** in the assignment step, its centroid cannot be updated by taking an average.
* **Common solutions:**
    * **Eliminate the cluster:** Continue with fewer than K clusters.
    * **Re-initialize the centroid:** Randomly place the centroid in a new location (e.g., at another data point).

### K-Means Beyond Well-Separated Clusters:

K-means is useful even when data clusters are not clearly separated (i.e., data varies continuously):

* **Example: T-shirt Sizing (Height vs. Weight):** Even if height and weight data don't show distinct groups, K-means (e.g., with K=3 for S, M, L sizes) can partition the data. The centroids can then represent the "average" customer for each size, guiding optimal t-shirt dimensions.

The next video will delve into the cost function that K-means implicitly optimizes, providing deeper intuition and explaining why the algorithm converges.

## K-Means Clustering Algorithm: Optimizing a Cost Function

K-means is an optimization algorithm that minimizes a specific **cost function**, even though it doesn't use gradient descent. The algorithm's iterative steps are designed to progressively reduce this cost.

### K-Means Cost Function (Distortion Function)

* **Notation Review:**
    * $c^{(i)}$: Index ($1$ to $K$) of the cluster centroid to which training example $x^{(i)}$ is assigned.
    * $\mu_k$: Location (vector) of cluster centroid $k$.
    * $\mu_{c^{(i)}}$: The specific cluster centroid to which $x^{(i)}$ is assigned.
* **Cost Function $J$:**
    $$J(c^{(1)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K) = \frac{1}{m} \sum_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2$$
    * This is the **average squared Euclidean distance** between each training example $x^{(i)}$ and the centroid of the cluster ($\mu_{c^{(i)}}$) to which it has been assigned.
    * In literature, this is also called the **distortion function**.

### How K-Means Minimizes the Cost Function

The two iterative steps of K-means are designed to minimize $J$:

1.  **Assignment Step (Optimizing $c$ values, holding $\mu$ fixed):**
    * For each $x^{(i)}$, K-means assigns it to the $\mu_k$ that minimizes $||x^{(i)} - \mu_k||^2$.
    * This step **minimizes $J$ with respect to the cluster assignments $c^{(i)}$**, assuming the centroid locations $\mu_k$ are fixed. By assigning each point to its *closest* centroid, we are making $||x^{(i)} - \mu_{c^{(i)}}||^2$ as small as possible for each term in the sum, thus reducing the overall $J$.

2.  **Update Step (Optimizing $\mu$ values, holding $c$ fixed):**
    * For each cluster $k$, K-means sets $\mu_k$ to the **mean of all training examples $x^{(i)}$ that were assigned to cluster $k$** in the assignment step.
    * $$\mu_k = \frac{1}{\text{count}(x^{(i)} \text{ s.t. } c^{(i)}=k)} \sum_{i: c^{(i)}=k} x^{(i)}$$
    * It can be mathematically proven that choosing the mean of the assigned points for $\mu_k$ is the value that **minimizes $J$ with respect to the centroid locations $\mu_k$**, assuming the assignments $c^{(i)}$ are fixed.

### Convergence and Monitoring:

* Because each step of K-means explicitly reduces the cost function $J$ (or keeps it the same), the algorithm is **guaranteed to converge**. The cost function should *never* increase. If it does, there's a bug.
* **Monitoring Convergence:**
    * Plot the cost function $J$ as a function of iterations. It should consistently decrease.
    * Stop the algorithm when $J$ no longer decreases significantly (e.g., changes by less than a small threshold).
    * Alternatively, stop when the cluster assignments and centroid locations no longer change between iterations.

### Importance for Initialization:

Understanding the cost function is crucial for methods that improve K-means, such as using multiple random initializations to find better clusters, as discussed in the next video.

## K-Means Initialization and Multiple Random Starts

The very first step of K-means is to randomly initialize the K cluster centroids ($\mu_1, \dots, \mu_K$). However, this random initialization can significantly impact the final clustering.

### Initializing Cluster Centroids

* **Common Method:** Randomly pick `K` training examples from your dataset and set their locations as the initial $\mu_k$ values.
    * This ensures centroids start within the data distribution.
* **Constraint:** You should always choose $K \le m$ (number of clusters must be less than or equal to the number of training examples).

### The Problem of Local Optima

K-means aims to minimize the distortion cost function $J$. However, because $J$ is generally non-convex (even though each step of K-means reduces $J$), it can get stuck in **local minima**.

* Different random initializations can lead to different final clusterings.
* Some clusterings might be suboptimal, with higher $J$ values, even if K-means converged for that specific initialization.

### Solution: Multiple Random Initializations

To mitigate the problem of local optima and increase the chance of finding a good clustering, run K-means multiple times with different random initializations.

1.  **Algorithm for Multiple Initializations:**
    * **Repeat N times** (e.g., 50 to 1000 times):
        1.  Randomly initialize $\mu_1, \dots, \mu_K$ (by picking $K$ random training examples).
        2.  Run the K-means algorithm (assignment and update steps) to convergence for this initialization.
        3.  Compute the final distortion cost $J$ for the resulting clustering.
    * **Select Best Result:** Choose the clustering (the set of assignments $c^{(i)}$ and centroids $\mu_k$) that yielded the **lowest distortion cost $J$** among all $N$ runs.

2.  **Why it works:** By trying many starting points, you increase the probability that at least one of them will lead to the global minimum or a very good local minimum of the distortion function. The run with the lowest $J$ value is considered the best result.

This technique significantly improves the quality of clusters found by K-means compared to running it just once. The next video will address the challenge of choosing the optimal number of clusters, K.

## Choosing the Number of Clusters (K) in K-Means

The K-means algorithm requires the number of clusters, K, as an input. Deciding on the optimal K can be challenging, as it's often ambiguous, especially in unsupervised learning where there are no "right answers."

### Limitations of Objective Methods:

1.  **Ambiguity:** For many datasets, the "true" number of clusters is not clear from the data itself. Different interpretations (e.g., seeing 2 clusters vs. 4 clusters in the same data) can both be valid.

2.  **Elbow Method:** (Andrew Ng don't use this method himself because of the **Issue** stated below)
    * **Concept:** Run K-means for a range of K values, plot the distortion cost $J$ as a function of K. Look for an "elbow" point where the decrease in $J$ sharply slows down. This "elbow" suggests a good K.
    * **Issue:** Often, the plot does not have a clear, distinct elbow. $J$ might decrease smoothly, making it hard to pick a definitive K.
    * **Caveat:** Choosing K to simply *minimize* J is incorrect, as J will always decrease (or stay the same) as K increases, eventually reaching $J=0$ when $K=m$ (each point is its own cluster), which is unhelpful.

<img src="/metadata/elbow_method.png" width="700" />

### Practical Approach: Evaluate Based on Downstream Purpose

The most effective way to choose K is to evaluate how well the clustering results serve the **later, downstream purpose** for which the clustering is being performed.

* **Example: T-shirt Sizing:**
    * Run K-means with K=3 (Small, Medium, Large) and K=5 (XS, S, M, L, XL).
    * Evaluate the trade-off:
        * **More sizes (higher K):** Generally leads to a better fit for customers.
        * **Fewer sizes (lower K):** Reduces manufacturing, inventory, and shipping costs.
    * The best K is chosen based on this business trade-off, not solely on a statistical metric.

* **Example: Image Compression (Programming Exercise):**
    * K-means can be used to reduce the number of colors in an image.
    * The trade-off is between **image quality** (higher K means more colors, better quality) and **compression ratio** (lower K means fewer colors, more compression, smaller file size).
    * You would manually decide K by visually assessing the compressed image and comparing it to the file size reduction.

In practice, the choice of K is often guided by domain knowledge, practical constraints, and the ultimate goal of the application.

This concludes the discussion on K-means clustering. Next, we'll move to another important unsupervised learning algorithm: **anomaly detection**.

## Anomaly Detection: Identifying Unusual Events

**Anomaly detection** is an unsupervised learning algorithm that examines a dataset of **normal events** to learn what constitutes typical behavior, then flags or raises a "red flag" when it encounters an **unusual or anomalous event**.

### Example: Aircraft Engine Manufacturing

* **Problem:** Detect potential defects in newly manufactured aircraft engines.
* **Data:** Collect features (e.g., heat generated ($x_1$), vibration intensity ($x_2$)) from many **normal (non-defective)** aircraft engines that roll off the assembly line.
* **Goal:** Given a new engine's features ($x_{test}$), determine if it's "normal" or if something is "weird" and requires closer inspection.

<img src="/metadata/anomaly.png" width="500" />

### How Anomaly Detection Works (Density Estimation)

1.  **Model p(x) (Probability Distribution):**
    * The algorithm first learns a model for the **probability distribution of the features ($p(x)$)** from the unlabeled training set of normal examples.
    * This model estimates which regions of the feature space have **high probability** (where normal examples cluster) and which have **low probability** (sparse regions). Visually, this creates contours of probability density (e.g., concentric ellipses).

2.  **Anomaly Flagging:**
    * For a new test example $x_{test}$, compute its probability $p(x_{test})$ using the learned model.
    * If $p(x_{test})$ is **less than a small threshold $\epsilon$ (epsilon)**, then $x_{test}$ is flagged as an **anomaly**.
    * If $p(x_{test})$ is greater than or equal to $\epsilon$, it's considered normal.

### Applications of Anomaly Detection:

* **Fraud Detection:**
    * **Features (x):** User login frequency, web page visits, transaction counts, typing speed.
    * **Application:** Identifies unusual user activity or purchase patterns that may indicate fake accounts or financial fraud, triggering further human review or security checks (e.g., identity verification, CAPTCHA).
* **Manufacturing:**
    * **Features (x):** Measurements from manufactured units (e.g., aircraft engines, circuit boards, smartphones, motors).
    * **Application:** Detects units that behave strangely, indicating a potential defect that warrants closer inspection before shipping.
* **Monitoring Computers in Data Centers:**
    * **Features (x):** CPU load, memory usage, disk accesses per second, CPU load to network traffic ratio.
    * **Application:** Flags computers behaving unusually, potentially indicating hardware failure, network issues, or a security breach (e.g., hacking).

Anomaly detection is a widely used yet often understated tool for identifying suspicious or abnormal behavior across various domains. The next videos will explain how to build these algorithms using the **Gaussian distribution** to model $p(x)$.

## Gaussian (Normal) Distribution for Anomaly Detection

To apply anomaly detection, we use the **Gaussian distribution**, also known as the **normal distribution** or **bell-shaped curve**, to model the probability $p(x)$ of our data.

### Definition of Gaussian Distribution

For a single numerical variable $x$, its probability distribution under a Gaussian model is defined by two parameters:
* **Mean ($\mu$):** The center of the distribution.
* **Variance ($\sigma^2$):** The spread or width of the distribution. ($\sigma$ is the standard deviation).

The probability density function (PDF) is given by:
$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$

* **Interpretation:** This curve shows the likelihood of observing different values of $x$. High points on the curve indicate values of $x$ that are more probable; low points indicate less probable values.
* The area under the curve always sums to 1.
* **Effect of Parameters:**
    * Changing $\mu$ shifts the curve horizontally.
    * Changing $\sigma^2$ (or $\sigma$) changes the width (spread) of the curve. A smaller $\sigma$ means a taller, skinnier curve; a larger $\sigma$ means a shorter, wider curve.

### Estimating Parameters from Data

Given a dataset of $m$ training examples ($x^{(1)}, \dots, x^{(m)}$), we estimate $\mu$ and $\sigma^2$ using **maximum likelihood estimation**:

* **Estimate for Mean ($\hat{\mu}$):** The average of all training examples:
    $$\hat{\mu} = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$$
* **Estimate for Variance ($\hat{\sigma}^2$):** The average of the squared differences from the mean:
    $$\hat{\sigma}^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \hat{\mu})^2$$
    (Using $1/m$ instead of $1/(m-1)$ is common and makes little practical difference for large $m$).

### Anomaly Detection with a Single Feature

Once $\hat{\mu}$ and $\hat{\sigma}^2$ are estimated:
* **High $p(x)$:** Values of $x$ near $\hat{\mu}$ have high probability. These are considered normal.
* **Low $p(x)$:** Values of $x$ far from $\hat{\mu}$ have low probability. These are considered unusual or anomalous.

This approach works for a single feature. For practical anomaly detection, which usually involves multiple features, we need to extend this concept. The next video will discuss how to build a multi-feature anomaly detection algorithm using these Gaussian distribution principles.

## Anomaly Detection Algorithm (Multivariate Gaussian)

This video builds the anomaly detection algorithm for data with multiple features, utilizing the Gaussian distribution for each feature.

### Modeling p(x) for Multiple Features

Given a training set $x^{(1)}, \dots, x^{(m)}$, where each $x^{(i)}$ is a vector with $n$ features (e.g., $x_1$ for heat, $x_2$ for vibrations, up to $x_n$):

We model the probability of a feature vector $x = [x_1, x_2, \dots, x_n]$ as the **product of the probabilities of its individual features**:

$$p(x) = p(x_1) \times p(x_2) \times \dots \times p(x_n)$$

* This approach assumes that the features $x_1, \dots, x_n$ are **statistically independent**. Even if they are not perfectly independent, this model often works well in practice for anomaly detection.
* For each feature $x_j$, we model its probability $p(x_j)$ as a **Gaussian distribution** with its own mean $\mu_j$ and variance $\sigma_j^2$.
    $$p(x_j; \mu_j, \sigma_j^2) = \frac{1}{\sqrt{2\pi\sigma_j^2}} e^{-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}}$$

### Anomaly Detection System Steps:

1.  **Choose Features:** Select features $x_j$ that are indicative of anomalous examples.
2.  **Fit Parameters ($\mu_j, \sigma_j^2$):**
    For each feature $j=1, \dots, n$:  
    * Estimate $\mu_j$: The mean of feature $x_j$ across all $m$ training examples.  
        $$\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}$$
    * Estimate $\sigma_j^2$: The variance of feature $x_j$ across all $m$ training examples.  
        $$\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2$$
    * (These calculations can be vectorized to compute all $\mu_j$ and $\sigma_j^2$ simultaneously).
3.  **Compute $p(x)$ for New Examples:**
    For a new example $x_{test}$, calculate its overall probability $p(x_{test})$:  
    $$p(x_{test}) = \prod_{j=1}^{n} p(x_{test, j}; \mu_j, \sigma_j^2)$$  
    (This is the product of the probabilities of each individual feature $x_{test, j}$ using its corresponding estimated Gaussian distribution.)
4.  **Flag as Anomaly:**
    * If $p(x_{test}) < \epsilon$ (a small threshold), flag $x_{test}$ as an **anomaly**.
    * Otherwise, it is considered normal.

### Intuition:

This algorithm flags an example as anomalous if one or more of its features are unusually large or unusually small (i.e., fall into the low-probability tails of their respective Gaussian distributions). If even one $p(x_j)$ term in the product is very small, the overall $p(x)$ will be very small.

### Example: Heat and Vibration

If $x_1$ (heat) and $x_2$ (vibration) are modeled by their respective Gaussians, then $p(x_1, x_2)$ forms a 3D probability surface. Points in the high-density center have high $p(x)$, while points far out in the "tails" of the distribution have low $p(x)$. An example like $x_{test2}$ with an unusually low $p(x)$ would be flagged as anomalous, while $x_{test1}$ with a high $p(x)$ would not.

The next video will discuss how to choose the threshold $\epsilon$ and how to evaluate the performance of an anomaly detection system.

## Practical Tips for Anomaly Detection Systems

Developing an anomaly detection system benefits greatly from a method to **evaluate its performance with real numbers**. This allows for faster iteration and improvement.

### Evaluation Setup: Labeled Data (Optional but Recommended)

Even though anomaly detection is an unsupervised learning technique, having a **small number of labeled anomalous examples** ($y=1$) is extremely useful for evaluation. Most of your data will still be unlabeled normal examples ($y=0$).

* **Training Set:** Consists predominantly of **unlabeled normal examples** (`X_train`, implicitly $y=0$). This is where the $p(x)$ model (Gaussian distributions) is learned. (A few accidental anomalies in this set are usually okay).
* **Cross-Validation Set:** Contains a mix of **normal examples ($y=0$) and a small number of known anomalous examples ($y=1$)**. Used for tuning parameters (like $\epsilon$) and features.
* **Test Set:** (If enough anomalies are available) Contains a separate mix of **normal examples ($y=0$) and known anomalous examples ($y=1$)**. Used for a final, unbiased evaluation of the system's performance.

### Example: Aircraft Engine Dataset Split

* **Total Data:** 10,000 good engines ($y=0$), 20 flawed engines ($y=1$).
* **Training Set:** 6,000 good engines.
* **Cross-Validation Set:** 2,000 good engines, 10 flawed engines.
* **Test Set:** 2,000 good engines, 10 flawed engines.

### Tuning and Evaluation Process:

1.  **Train `p(x)`:** Fit the Gaussian distributions on the (unlabeled/assumed normal) training set.
2.  **Make Predictions:** For each example `x` in the cross-validation or test set, compute `p(x)`. Predict $y=1$ (anomaly) if $p(x) < \epsilon$, else predict $y=0$ (normal).
3.  **Evaluate:** Compare these predictions against the known labels in the cross-validation/test set.
    * Use metrics like **precision, recall, or F1-score**, which are more appropriate than simple accuracy for highly skewed datasets (where $y=1$ examples are rare). This helps assess how well the system identifies anomalies without too many false alarms.
    * **Tune $\epsilon$:** Adjust $\epsilon$ based on performance on the cross-validation set (e.g., raise $\epsilon$ if missing too many anomalies, lower $\epsilon$ if too many false positives).
    * **Tune Features:** Also use the cross-validation set to decide whether to add, remove, or transform features.

### Alternative for Very Few Anomalies:

If there are extremely few anomalous examples (e.g., only 2-3), you might combine the cross-validation and test sets into a single evaluation set. This provides more data for tuning but means you lack a truly unbiased final evaluation.

This evaluation framework makes the iterative process of improving an anomaly detection system much more efficient. The next video will compare anomaly detection with supervised learning and discuss when to choose one over the other.

## Anomaly Detection vs. Supervised Learning: Choosing the Right Tool

When faced with a highly skewed dataset (very few positive examples, many negative examples), deciding between anomaly detection and supervised learning can be subtle. The choice depends on the nature of the "positive" class.

### Anomaly Detection (Recommended when):

* **Very Small Number of Positive Examples:** Often 0-20 known positive (anomalous) examples. The model for $p(x)$ is primarily learned from the large number of negative (normal) examples. Positive examples are used mainly for cross-validation and evaluation.
* **Many Different Types of Anomalies:** You believe there are numerous ways for something to be anomalous, and future anomalies are likely to be *different* from the few you've seen so far.
    * **Intuition:** The algorithm models what "normal" looks like, and anything deviating significantly from this norm is flagged. It can detect "new" types of anomalies it has never seen.
    * **Example:** Detecting **financial fraud** (new fraud schemes emerge constantly).
    * **Example:** Detecting **previously unseen defects** in manufacturing (e.g., a brand new way an aircraft engine could fail).
    * **Example:** Monitoring **hacked machines** in a data center (hackers find novel ways to compromise systems). This applies to many **security-related applications**.

### Supervised Learning (Recommended when):

* **Sufficient Number of Positive Examples:** You have enough positive examples to enable the algorithm to learn *what the positive examples look like*. This implies that future positive examples will be *similar* to the ones seen in the training set.
    * **"Enough" is relative:** Even 20 positive examples might be sufficient in some simple cases, but generally more are better.
* **Predicting Known Categories:** You want to classify into categories that you have extensively observed and have representative data for.
    * **Example:** **Email spam detection.** While spam evolves, new spam is often similar enough to past spam for a supervised model to generalize effectively.
    * **Example:** Detecting **known and previously seen manufacturing defects** (e.g., common scratches on smartphones where you have many examples of scratched phones).
    * **Example:** **Weather prediction** (you constantly observe and label weather types).
    * **Example:** **Diagnosing specific, known diseases** based on symptoms (where you have a dataset of confirmed cases).

### Summary of Comparison:

| Feature/Condition             | Anomaly Detection               | Supervised Learning               |
| :---------------------------- | :------------------------------ | :-------------------------------- |
| **Number of Positive Examples** | Very small (0-20)               | Larger number (enough to learn from) |
| **Nature of Positive Class** | Many diverse types; future ones unknown | Similar to past examples; patterns stable |
| **What it Learns** | "What normal looks like"        | "What positive looks like"        |
| **Best for** | Novel anomalies, rare events    | Known classes, common occurrences |
| **Example Use Cases** | Fraud (new types), new defects, hacked systems | Spam, known defects, weather, specific diseases |

The choice between these two approaches hinges on your assumptions about the nature of the "anomalies" or "positive" class you're trying to detect. The next video will discuss practical tips for selecting features in anomaly detection.

## Tuning Features for Anomaly Detection

In anomaly detection, carefully choosing features is even more crucial than in supervised learning, because the algorithm learns primarily from unlabeled "normal" data and doesn't have labels to guide feature relevance.

### 1. Make Features Approximately Gaussian

* **Problem:** The anomaly detection algorithm models $p(x_j)$ for each feature $x_j$ using a Gaussian (normal) distribution. If a feature's actual distribution is highly non-Gaussian, this model might be a poor fit.
* **Diagnosis:** Plot a **histogram** of each feature to visually inspect its distribution.
* **Solution:** If a feature's histogram looks highly skewed (e.g., concentrated on one side, not bell-shaped), consider applying **transformations** to make it more Gaussian.
    * **Common Transformations:**
        * `log(x)` (or `log(x + C)` if $x$ can be 0 or negative, adding a small constant $C$).
        * `sqrt(x)` (or `x` to the power of `0.5`).
        * `x` to the power of `P` (where `P` could be `0.25`, `0.33`, etc.).
    * **Process:** Experiment with different transformations and visually check the resulting histogram until it looks more symmetric and bell-shaped.
* **Rule:** Apply the same transformation to your training, cross-validation, and test sets.

### 2. Error Analysis for Anomaly Detection

After training your anomaly detection model $p(x)$, if it's not performing well (e.g., failing to detect known anomalies in the cross-validation set), perform error analysis.

* **Problem:** The algorithm fails to flag an anomaly because $p(x)$ is high for that example, meaning it looks "normal" based on existing features. This usually happens because one or more features of that anomalous example take on values similar to normal examples.
* **Process:**
    1.  Identify misclassified anomalies (False Negatives) in your cross-validation set.
    2.  **Manually inspect** these examples.
    3.  **Brainstorm and create new features** that might help distinguish these particular anomalies from normal examples. The goal is to find features where the anomalous examples exhibit unusually large or small values.

* **Example (Fraud Detection):**
    * `x1`: Number of transactions (might be normal for a fraudulent user).
    * If you notice the anomalous user has an "insanely fast typing speed," create `x2 = typing_speed`. This new feature might make the anomaly stand out.
* **Example (Data Center Monitoring):**
    * Original features: `memory_use`, `disk_accesses`, `CPU_load`, `network_traffic`.
    * If a machine has high `CPU_load` but unusually low `network_traffic` (which is rare for a video streaming server), create a new feature like `x5 = CPU_load / network_traffic`. This ratio might spike for the anomalous machine, allowing it to be flagged.
    * Other combinations: `(CPU_load)^2 / network_traffic`.

### Summary of Development Process:

* Train the initial model.
* Examine anomalies that were *missed* (False Negatives) in the cross-validation set.
* Based on these missed anomalies, brainstorm and engineer new features that would highlight their anomalous nature.
* Add these new features and retrain the model.
* Repeat the process.

This systematic approach of inspecting errors and engineering new features is crucial for improving anomaly detection performance, especially for handling novel types of anomalies that might not be well-represented in initial feature sets.

## Recommender Systems: Introduction

Recommender systems are a commercially impactful machine learning application, driving significant sales and engagement on online platforms like e-commerce, streaming services, and food delivery apps.

### What is a Recommender System?

* **Goal:** To suggest items (movies, products, articles, restaurants) to users that they are likely to be interested in or rate highly.
* **Core Data:** A dataset showing user ratings or interactions with various items.

### Running Example: Movie Rating Prediction

<img src="/metadata/rec_sys.png" width="600" />

Consider a movie streaming website with users rating movies 0-5 stars.

* **Users:** Alice (1), Bob (2), Carol (3), Dave (4). Let $N_u$ be the number of users ($N_u = 4$).
* **Items (Movies):** "Love at last", "Romance forever", "Cute puppies of love", "Nonstop car chases", "Sword versus karate". Let $N_m$ be the number of movies ($N_m = 5$).
* **Ratings Data:**
    * Alice (User 1): Rated Movies 1, 2, 4, 5. Missing Movie 3.
    * Bob (User 2): Rated Movies 1, 3, 4, 5. Missing Movie 2.
    * Carol (User 3): Rated Movies 1, 3, 4, 5. Missing Movie 2.
    * Dave (User 4): Rated Movies 1, 2, 3, 4, 5. (All rated).

### Notation for Recommender Systems:

* $N_u$: Number of users.
* $N_m$: Number of items (movies).
* $r(i,j) = 1$: If user $j$ has rated movie $i$. ($r(i,j) = 0$ otherwise).
    * Example: $r(1,1)=1$ (Alice rated Movie 1), but $r(3,1)=0$ (Alice did not rate Movie 3).
* $y(i,j)$: The rating given by user $j$ to movie $i$.
    * Example: $y(3,2)=4$ (Bob rated Movie 3 as 4 stars).

### The Recommendation Problem:

The primary goal is to predict how users would rate movies they *have not yet rated* (the question marks in the table). Once these predictions are made, the system can recommend items with the highest predicted ratings to users.

The next video will begin developing an algorithm for this, initially assuming we have explicit features for the movies (e.g., whether they are romance or action movies). Later, we'll explore how to work without such explicit features.

## Recommender Systems: Collaborative Filtering (Feature-based)

This video introduces a method for building recommender systems by predicting user ratings, assuming we have pre-defined features for each item.

### Problem Setup: Movie Rating Prediction

* **Users:** $N_u$ users (e.g., Alice, Bob, Carol, Dave; $N_u=4$).
* **Items (Movies):** $N_m$ movies (e.g., "Love at Last", "Nonstop Car Chases"; $N_m=5$).
* **Ratings:** $y(i,j)$ is the rating given by user $j$ to movie $i$. Ratings are sparse (many movies are unrated, denoted by '?').
* **Notation:**
    * $r(i,j) = 1$ if user $j$ has rated movie $i$, $0$ otherwise.
    * $m(j)$: Number of movies rated by user $j$.

### Assuming Item Features

* We assume each movie $i$ has a feature vector $X^{(i)} = [X_1^{(i)}, X_2^{(i)}, \dots, X_n^{(i)}]$, where $n$ is the number of features (e.g., $n=2$ for "romance" and "action" levels).
    * Example: $X^{(1)} = [0.9, 0]$ (Love at Last: 0.9 romance, 0 action).
    * Example: $X^{(4)} = [0.1, 1.0]$ (Nonstop Car Chases: 0.1 romance, 1.0 action).

### Model: User-Specific Linear Regression

For each user $j$, we train a separate linear regression-like model to predict their ratings based on movie features:

* **Predicted Rating for movie $i$ by user $j$:** $\hat{y}(i,j) = \vec{w}^{(j)} \cdot X^{(i)} + b^{(j)}$
    * $\vec{w}^{(j)}$: Parameter vector (weights) specific to user $j$.
    * $b^{(j)}$: Bias parameter specific to user $j$.

### Cost Function for a Single User $j$

To learn $\vec{w}^{(j)}$ and $b^{(j)}$ for a specific user $j$, we minimize a regularized squared error cost function, considering only the movies they have rated:

$$J(\vec{w}^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i: r(i,j)=1} (\vec{w}^{(j)} \cdot X^{(i)} + b^{(j)} - y(i,j))^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (w_k^{(j)})^2$$

* The sum is only over movies $i$ that user $j$ has rated ($r(i,j)=1$).
* The term $\frac{1}{m(j)}$ (where $m(j)$ is the number of movies rated by user $j$) is often omitted for convenience in recommender systems, as it's a constant factor and doesn't change the optimal $\vec{w}^{(j)}, b^{(j)}$.
* The $\lambda$ term is for regularization to prevent overfitting.

### Overall Cost Function (for all Users)

To learn parameters for *all* users, we sum the individual user cost functions:

$$J(\vec{w}^{(1)}, \dots, \vec{w}^{(N_u)}, b^{(1)}, \dots, b^{(N_u)}) = \sum_{j=1}^{N_u} \left( \frac{1}{2} \sum_{i: r(i,j)=1} (\vec{w}^{(j)} \cdot X^{(i)} + b^{(j)} - y(i,j))^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (w_k^{(j)})^2 \right)$$

* This global cost function is minimized with respect to all user-specific parameters ($\vec{w}^{(j)}$ and $b^{(j)}$ for all $j$).
* This approach effectively trains $N_u$ separate linear regression models.

This method works well if we have detailed features for each item. The next video will address the more common scenario where these explicit item features are *not* available in advance.

## Collaborative Filtering: Learning Item Features

The previous video assumed we have pre-defined features for each movie. This video explores **Collaborative Filtering**, where the system can **learn these item features ($X^{(i)}$) directly from user ratings**, without them being provided in advance.

### Intuition:

* **From Users to Item Features:** If we know how different users rate an item, and we know those users' preferences (their $\vec{w}^{(j)}$ and $b^{(j)}$ parameters), we can infer what characteristics that item must have to explain those ratings.
    * Example: If Alice (who likes romance) and Bob (who likes romance) both rate Movie 1 highly, but Carol (who likes action) and Dave (who likes action) rate Movie 1 low, this suggests Movie 1 is a "romance" movie.

<img src="/metadata/colabb.png" width="500" />

### Step 1 (Conceptual): Learning Item Features ($X^{(i)}$) given User Parameters ($\vec{w}^{(j)}, b^{(j)}$)

* **Goal:** For a specific movie $i$, learn its feature vector $X^{(i)}$.
* **Cost Function for $X^{(i)}$:** We minimize the squared error between predicted ratings (using known user parameters) and actual ratings, summed over all users who rated movie $i$:
    $$J(X^{(i)}) = \frac{1}{2} \sum_{j: r(i,j)=1} (\vec{w}^{(j)} \cdot X^{(i)} + b^{(j)} - y(i,j))^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (X_k^{(i)})^2$$
    * The second term is for regularization to prevent overfitting on the item features.
* **Overall Cost for All Item Features:** Sum this over all movies $i=1, \dots, N_m$:
    $$J(X^{(1)}, \dots, X^{(N_m)}) = \sum_{i=1}^{N_m} \left( \frac{1}{2} \sum_{j: r(i,j)=1} (\vec{w}^{(j)} \cdot X^{(i)} + b^{(j)} - y(i,j))^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (X_k^{(i)})^2 \right)$$
    * Minimizing this learns the features for all movies.

### Step 2 (Conceptual): Learning User Parameters ($\vec{w}^{(j)}, b^{(j)}$) given Item Features ($X^{(i)}$)

* This is what we did in the previous video. For each user $j$, we learn their preferences $\vec{w}^{(j)}, b^{(j)}$ given the movie features $X^{(i)}$.

### Full Collaborative Filtering Algorithm:

The core idea is that we don't know *either* the user parameters *or* the item features initially. Collaborative filtering learns both simultaneously by minimizing a single, combined cost function.

* **Combined Cost Function:** This function sums over *all* user-movie pairs $(i, j)$ for which a rating exists ($r(i,j)=1$), and includes regularization for both user parameters ($\vec{w}^{(j)}$) and item features ($X^{(i)}$).
    $$J(\vec{w}^{(1..N_u)}, b^{(1..N_u)}, X^{(1..N_m)}) = \frac{1}{2} \sum_{(i,j): r(i,j)=1} (\vec{w}^{(j)} \cdot X^{(i)} + b^{(j)} - y(i,j))^2 + \frac{\lambda}{2} \sum_{j=1}^{N_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{N_m} \sum_{k=1}^{n} (X_k^{(i)})^2$$
    * The first sum is over all ratings we have observed.
    * The second sum regularizes all user preference parameters ($\vec{w}^{(j)}$).
    * The third sum regularizes all movie features ($X^{(i)}$).

* **Minimization (Gradient Descent):**
    * Initialize all user parameters ($\vec{w}^{(j)}, b^{(j)}$) and all movie features ($X^{(i)}$) randomly (e.g., small random numbers).
    * Repeatedly update all these parameters simultaneously using gradient descent (or Adam):
        * $\vec{w}^{(j)} \leftarrow \vec{w}^{(j)} - \alpha \frac{\partial}{\partial \vec{w}^{(j)}} J(\dots)$
        * $b^{(j)} \leftarrow b^{(j)} - \alpha \frac{\partial}{\partial b^{(j)}} J(\dots)$
        * $X^{(i)} \leftarrow X^{(i)} - \alpha \frac{\partial}{\partial X^{(i)}} J(\dots)$
    * By doing this, the algorithm "collaborates" by iteratively guessing user preferences and item features, leading to values that best explain the observed ratings.

**Why "Collaborative Filtering"?** It's called this because ratings from *multiple users* on the *same item* collaboratively help infer the item's features, and similarly, ratings from a *single user* on *multiple items* help infer that user's preferences. This collaboration between user data points helps predict ratings for other users and items.

The next video will explore collaborative filtering for binary labels (e.g., user likes/favors an item) instead of star ratings.

## Collaborative Filtering with Binary Labels

Many recommender systems deal with **binary labels** (e.g., user "liked" or "disliked" an item, "purchased" or "did not purchase"), rather than explicit star ratings. This video generalizes the collaborative filtering algorithm to this setting, similar to how linear regression was generalized to logistic regression.

### Binary Label Examples:

* **Online Shopping:**
    * 1: User purchased an item (after exposure).
    * 0: User did not purchase (after exposure).
    * ?: User was not exposed to the item.
* **Social Media:**
    * 1: User favorited/liked an item (after being shown it).
    * 0: User did not favorite/like (after being shown it).
    * ?: Item not yet shown to the user.
* **Implicit Engagement (e.g., streaming, content sites):**
    * 1: User spent $\ge 30$ seconds on an item.
    * 0: User spent $< 30$ seconds on an item.
    * ?: Item not yet shown.
* **Online Advertising:**
    * 1: User clicked on an ad.
    * 0: User did not click.
    * ?: Ad not shown.

**Common Interpretation:**
* **1:** User engaged after being shown the item (clicked, spent time, favorited, purchased).
* **0:** User did not engage after being shown the item.
* **?:** Item not yet shown.

### Generalizing the Algorithm:

Previously, we predicted $y(i,j) = \vec{w}^{(j)} \cdot X^{(i)} + b^{(j)}$ (similar to linear regression).
For binary labels, we generalize this using the **Sigmoid (logistic) function**, similar to logistic regression:

* **Predicted Probability of Liking/Engagement:**
    $$\hat{y}(i,j) = P(y(i,j)=1) = g(\vec{w}^{(j)} \cdot X^{(i)} + b^{(j)})$$
    where $g(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.
    * This model now predicts the probability that user $j$ will like/engage with item $i$.

### Modifying the Cost Function for Binary Labels:

Instead of the squared error cost, we use a cost function appropriate for binary classification: the **binary cross-entropy loss** (which was used for logistic regression and binary neural network classification).

* **Loss for a single $y(i,j)$:**
    $$L(\hat{y}(i,j), y(i,j)) = -y(i,j) \log(\hat{y}(i,j)) - (1 - y(i,j)) \log(1 - \hat{y}(i,j))$$
* **Overall Cost Function (for all parameters $\vec{w}, \vec{b}, X$):**
    $$J(\vec{w}, \vec{b}, X) = \sum_{(i,j): r(i,j)=1} L(\hat{y}(i,j), y(i,j)) + \frac{\lambda}{2} \sum_{j=1}^{N_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{N_m} \sum_{k=1}^{n} (X_k^{(i)})^2$$
    * The summation is over all user-item pairs where a rating/engagement exists ($r(i,j)=1$).
    * The $\lambda$ terms are for regularization of user parameters and item features.

By minimizing this modified cost function (e.g., using gradient descent), the algorithm learns optimal user preferences and item features for predicting binary engagement signals. This significantly expands the range of applications collaborative filtering can address.

The next video will discuss implementation tips and refinements for this algorithm.

## Collaborative Filtering: Mean Normalization

For recommender systems that predict explicit ratings (e.g., 0-5 stars), **mean normalization** is a pre-processing step that significantly improves algorithm efficiency and the quality of predictions for new users or items.

### The Problem: New Users with No Ratings

Consider a new user, Eve, who hasn't rated any movies. If we train the collaborative filtering algorithm without mean normalization:

* The regularization term in the cost function incentivizes small parameters ($\vec{w}^{(j)}$ and $b^{(j)}$).
* For Eve ($\vec{w}^{(5)}, b^{(5)}$), since she has no ratings, her parameters don't affect the squared error sum.
* Minimizing the regularization term for Eve results in $\vec{w}^{(5)} = [0, 0]$ and $b^{(5)} = 0$ (assuming default initialization).
* **Prediction for Eve:** The algorithm predicts all her movie ratings will be $\vec{w}^{(5)} \cdot X^{(i)} + b^{(5)} = 0$, which is unhelpful and unrealistic.

### Solution: Mean Normalization

Mean normalization aims to center the ratings for each movie around zero.

1.  **Calculate Mean Rating for Each Movie ($\mu_i$):**
    * For each movie $i$, compute the average rating $\mu_i$ from *only the users who have rated that movie*.
    * Example: Movie 1 (rated by Alice=5, Bob=5, Carol=0, Dave=0), so $(5+5+0+0)/4 = 2.5$ if counting all users.
    * $\mu = [2.5, 2.5, 2.0, 2.25, 1.25]$ for movies 1-5 respectively.

2.  **Normalize Ratings:**
    * For every observed rating $y(i,j)$, subtract the movie's average rating $\mu_i$:
        $y'(i,j) = y(i,j) - \mu_i$
    * If a user hasn't rated a movie, it remains unrated.
    * This creates a new matrix of normalized ratings $Y'$.

3.  **Train Algorithm on Normalized Ratings:**
    * Train the collaborative filtering algorithm (learning $\vec{w}^{(j)}, b^{(j)}$ and $X^{(i)}$) using these **normalized ratings $Y'$**.
    * The cost function will now use $((\vec{w}^{(j)} \cdot X^{(i)} + b^{(j)}) - y'(i,j))^2$.

4.  **Prediction with Mean Normalization:**
    * When predicting a rating for user $j$ on movie $i$, remember to **add back the movie's mean rating $\mu_i$**:
        $\hat{y}(i,j) = (\vec{w}^{(j)} \cdot X^{(i)} + b^{(j)}) + \mu_i$

### Benefits of Mean Normalization:

* **Better Predictions for New/Sparse Users:**
    * For a new user like Eve (who has rated no movies), $\vec{w}^{(5)}$ and $b^{(5)}$ will still likely be [0 0] and 0 (due to regularization).
    * However, her predicted rating for Movie 1 would now be $0 + \mu_1 = 2.5$.
    * **Result:** The algorithm predicts new users will give ratings equal to the movie's average, which is much more reasonable than predicting 0 stars. This acts as a good "default" prediction.
* **Faster Optimization:** Normalizing ratings to have a consistent average value (often zero) helps the optimization algorithm (like gradient descent) run more efficiently.
* **Improved Behavior for Unrated Items:** While normalizing rows is more critical for new users, normalizing columns (i.e., making users' average ratings zero) can help for predicting new items that few users have rated. However, normalizing rows (for users) is generally more impactful for collaborative filtering.

Mean normalization is an important implementation detail that makes collaborative filtering more robust and its predictions more reasonable, especially for users with very few or no ratings.

## Implementing Collaborative Filtering with TensorFlow (AutoDiff)

This video demonstrates how to implement the collaborative filtering algorithm using TensorFlow, leveraging its automatic differentiation (AutoDiff) capabilities. This allows for optimization without manually calculating gradients.

### Why Use TensorFlow for Collaborative Filtering?

  * **AutoDiff:** TensorFlow can automatically compute the derivatives of the cost function with respect to all parameters (user parameters $\\vec{w}^{(j)}, b^{(j)}$ and item features $X^{(i)}$). This eliminates the need for manual calculus.
  * **Powerful Optimizers:** Once derivatives are available, you can use advanced optimizers like Adam, which are often more efficient than basic gradient descent.

### AutoDiff (Automatic Differentiation) in TensorFlow:

  * **Concept:** TensorFlow's `tf.GradientTape` feature records all operations performed within its context. It then uses this recorded "tape" to automatically compute gradients (derivatives) of a target loss with respect to specified `tf.Variable`s.
  * **Example (Simple $J=(wx-1)^2$):**
    ```python
    import tensorflow as tf

    w = tf.Variable(3.0) # Declare w as a TensorFlow Variable to be optimized
    x = 1.0
    y = 1.0
    learning_rate = 0.01

    for _ in range(30): # Iterations
        with tf.GradientTape() as tape:
            f_x = w * x
            J = (f_x - y)**2 # Define the cost function

        # Automatically compute derivative dJ/dw
        dJ_dw = tape.gradient(J, w)

        # Apply update (TensorFlow Variables use assign_sub)
        w.assign_sub(learning_rate * dJ_dw)
    ```
      * `tf.Variable`: Marks parameters that TensorFlow should track for gradient computation.
      * `tf.GradientTape()`: Records operations.
      * `tape.gradient(J, w)`: Computes $\\frac{dJ}{dw}$.
      * `w.assign_sub()`: Updates the variable in place.

### Implementing Collaborative Filtering with AutoDiff:

The collaborative filtering cost function $J(\\vec{w}, \\vec{b}, X)$ is defined (as in the previous video). The optimization process involves repeatedly computing this cost and its gradients.

1.  **Define Optimizer:**
    ```python
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Or a different optimizer
    ```
2.  **Training Loop:**
    ```python
    for iteration in range(num_iterations):
        with tf.GradientTape() as tape:
            # Implement the FULL collaborative filtering cost function J here
            # J = sum_over_all_rated_pairs(L(y_hat, y)) + regularization_terms
            # Need to define y_norm, r, num_users, num_movies, lambda_val etc.
            # And compute predictions y_hat = w_j . x_i + b_j

        # Compute gradients of J with respect to ALL optimizable parameters (x, w, b)
        # Note: x, w, b would also be tf.Variable objects
        gradients = tape.gradient(J, [X_items_variable, W_users_variable, B_users_variable])

        # Apply gradients
        optimizer.apply_gradients(zip(gradients, [X_items_variable, W_users_variable, B_users_variable]))
    ```
      * The `apply_gradients` function efficiently updates all parameters.

### Why this approach instead of `model.compile`/`model.fit`?

  * The collaborative filtering algorithm's structure (where both item features $X^{(i)}$ and user parameters $\\vec{w}^{(j)}, b^{(j)}$ are learned simultaneously as "parameters" through Bellman-like equations) doesn't neatly map to TensorFlow's standard `tf.keras.layers.Dense` and `model.compile`/`model.fit` paradigm.
  * The `model.fit` recipe is designed for sequential layers in a typical neural network. Collaborative filtering has a custom cost function and parameter structure.
  * AutoDiff provides the flexibility to define any custom cost function and automatically get its gradients, making it a powerful tool for implementing algorithms beyond standard neural network architectures.

## Finding Related Items with Collaborative Filtering

The collaborative filtering algorithm not only predicts user ratings but also learns **feature vectors ($X^{(i)}$) for each item $i$ (e.g., movie)**. These learned features provide a natural way to find similar items.

### Learned Item Features:

* The algorithm automatically learns feature vectors $X^{(i)}$ for each item $i$ (e.g., $X^{(1)}$ for Movie 1, $X^{(2)}$ for Movie 2, etc.).
* While these features ($x_1, x_2, \dots$) might be abstract and hard for humans to directly interpret (e.g., "feature 1 doesn't clearly mean 'romance'"), collectively, they capture the essence of what each item is like based on user preferences.

### Finding Related Items (Similarity Search):

* **Method:** To find items similar to a given item $i$ (with features $X^{(i)}$), search for other items $k$ (with features $X^{(k)}$) that have a **small squared distance** between their feature vectors.
* **Formula (Squared Euclidean Distance):**
    $$\text{Distance}(X^{(k)}, X^{(i)})^2 = \sum_{l=1}^{n} (X_l^{(k)} - X_l^{(i)})^2$$
    (This is also written as $||X^{(k)} - X^{(i)}||^2$).
* **Process:** Identify the top 5 or 10 items $k$ with the smallest squared distance to $X^{(i)}$. These will be the most "related" items based on the learned features.
* **Application:** This is how many online shopping or streaming websites show "items similar to this one."

### Limitations of Collaborative Filtering:

Despite its power, collaborative filtering has limitations:

1.  **Cold Start Problem:**
    * **New Items:** It struggles to recommend **new items** that have very few or no ratings. Without ratings, the algorithm cannot learn meaningful feature vectors $X^{(i)}$ for these items.
    * **New Users:** It also struggles to make good recommendations for **new users** who have rated only a few or no items. (Mean normalization helps mitigate this, but a new user still lacks rich preference data).
    * **Reason:** Collaborative filtering relies on a sufficient amount of interaction data between users and items.

2.  **Lack of Side Information / Additional Features:**
    * Collaborative filtering doesn't naturally incorporate "side information" or external features about items or users that might be readily available.
    * **Item Side Info:** Genre, cast, budget, studio (for movies); description, brand, category (for products).
    * **User Side Info:** Demographics (age, gender, location), explicit preferences (e.g., "likes sci-fi"), implicit cues (e.g., IP address, device type, web browser used).
    * **Impact:** This side information can be highly predictive and useful, especially for cold start situations or when user interaction data is sparse. Collaborative filtering, in its pure form, doesn't utilize it.

The next video will introduce **content-based filtering algorithms**, which directly address these limitations by leveraging side information about items and users, and are widely used in commercial applications.

## Content-Based Filtering: A New Recommender System Approach

This video introduces **Content-Based Filtering (CBF)**, a different approach to recommender systems that explicitly leverages features of both users and items to find good matches, addressing limitations of Collaborative Filtering (CF).

### Collaborative Filtering vs. Content-Based Filtering:

* **Collaborative Filtering (CF):** Recommends items based on ratings from *similar users* (users who liked what you liked) or item similarities derived from shared ratings. Relies solely on user-item interaction data.
* **Content-Based Filtering (CBF):** Recommends items based on **features of the user** and **features of the item**, seeking a good match between them. It requires explicit descriptive attributes for users and items.

### Features in Content-Based Filtering:

CBF makes good use of diverse features for users and items:

* **User Features ($X_u^{(j)}$ for user $j$):**
    * **Demographics:** Age, gender, country (often one-hot encoded).
    * **Past Behaviors:** List of top movies watched, products purchased (e.g., a binary vector for the 1000 most popular movies).
    * **Aggregated Preferences:** Average rating per genre the user has given (e.g., average rating for romance movies, average for action movies). These features depend on user ratings.

* **Item Features ($X_m^{(i)}$ for movie $i$):**
    * **Metadata:** Year of release, genre(s), critic reviews (can be numerical scores or textual features).
    * **Aggregated User Feedback:** Average rating of the movie (overall, or per country, or per user demographic). These features depend on user ratings of the item.

### The Content-Based Filtering Model:

Given user features ($X_u^{(j)}$) and item features ($X_m^{(i)}$), the goal is to predict the rating $\hat{y}(i,j)$.

* **Mapping to Latent Vectors:** Instead of directly using the raw features $X_u^{(j)}$ and $X_m^{(i)}$ (which can be of different lengths), CBF often learns to map them into lower-dimensional, fixed-size **latent vectors**:
    * $\vec{v}_u^{(j)}$: A vector representing user $j$'s preferences/profile.
    * $\vec{v}_m^{(i)}$: A vector representing item $i$'s attributes/content.
    * **Crucial:** These latent vectors ($\vec{v}_u$ and $\vec{v}_m$) must have the **same dimension** (e.g., both are 32-dimensional vectors) to allow for a dot product.

* **Prediction Model:** The predicted rating is given by the dot product of the user's latent vector and the item's latent vector:
    $$\hat{y}(i,j) = \vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)}$$
    * **Intuition:** This dot product effectively measures how well a user's preferences ($\vec{v}_u$) align with an item's characteristics ($\vec{v}_m$). For example, if $\vec{v}_u$ contains a component for "likes romance" and $\vec{v}_m$ contains a component for "is a romance movie," their dot product reflects this match.

### Advantages over Collaborative Filtering:

* **Addresses Cold Start Problem:**
    * **New Items:** If a new item has good descriptive features ($X_m$), $\vec{v}_m$ can be computed even without many ratings, allowing it to be recommended immediately.
    * **New Users:** If a new user provides some demographic or preference features ($X_u$), $\vec{v}_u$ can be computed, enabling recommendations without a long rating history.
* **Leverages Side Information:** Directly incorporates rich descriptive data about users and items that CF struggles with.

The next video will delve into how these latent vectors $\vec{v}_u$ and $\vec{v}_m$ are actually computed.

## Content-Based Filtering with Deep Learning

Deep learning is a state-of-the-art approach for content-based filtering, allowing recommender systems to leverage rich user and item features effectively.

### Model Architecture: Two-Tower Neural Network

The core idea is to use two separate neural networks (often called "towers") that learn compact representations (latent vectors) for users and items:

1.  **User Network:**
    * **Input:** Raw features of the user ($X_u^{(j)}$) (e.g., age, gender, country, past purchase history, average ratings per genre).
    * **Architecture:** Typically a few dense neural network layers.
    * **Output:** A fixed-size latent vector $\vec{v}_u^{(j)}$ (e.g., 32 numbers) that succinctly describes the user's preferences.

2.  **Movie Network (Item Network):**
    * **Input:** Raw features of the movie/item ($X_m^{(i)}$) (e.g., year, genre, critic reviews, average user rating for that movie).
    * **Architecture:** Typically a few dense neural network layers.
    * **Output:** A fixed-size latent vector $\vec{v}_m^{(i)}$ (e.g., 32 numbers) that succinctly describes the item's attributes.

3.  **Prediction:** The predicted rating ($\hat{y}(i,j)$) is computed as the **dot product** of the user's latent vector and the item's latent vector:
    $$\hat{y}(i,j) = \vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)}$$
    * **Interpretation:** This dot product measures the similarity or "match" between the user's learned preferences and the item's learned characteristics.
    * **For Binary Labels:** If predicting "like/dislike," a Sigmoid function can be applied to the dot product: $\hat{y}(i,j) = g(\vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)})$.

### Training the Combined Network:

* **Single Cost Function:** Both the user network and the movie network are trained simultaneously using a **single cost function**.
    * **Loss:** For star ratings, it's typically the mean squared error (MSE) between the predicted dot product and the actual rating.
    $$J = \sum_{(i,j): r(i,j)=1} (\vec{v}_u^{(j)} \cdot \vec{v}_m^{(i)} - y(i,j))^2 + \text{regularization terms}$$
    * Regularization terms (e.g., L2 regularization) are added to the cost function to keep the parameters of both neural networks small, preventing overfitting.
* **Optimization:** Gradient descent (or Adam) is used to tune all parameters across both the user network and the movie network to minimize this combined cost function.

### Finding Similar Items with Learned Embeddings:

* After training, the learned latent vectors ($\vec{v}_m^{(i)}$) effectively act as "embeddings" or representations of the items.
* To find items similar to a given item $i$ (with vector $\vec{v}_m^{(i)}$), you can search for other items $k$ (with vector $\vec{v}_m^{(k)}$) that have the **smallest squared Euclidean distance** between their latent vectors: $|| \vec{v}_m^{(k)} - \vec{v}_m^{(i)} ||^2$.

### Advantages and Considerations:

* **Powerful:** This deep learning approach is used in many state-of-the-art commercial recommenders.
* **Feature Engineering:** Good feature engineering for the raw inputs ($X_u, X_m$) remains crucial, as it provides the raw material for the networks to learn from.
* **Scalability Challenge:** While powerful, this model can be computationally expensive to run predictions for a very large catalog of items, as you need to compute dot products for all possible user-item pairs.
* **Pre-computation:** Finding similar items can be pre-computed offline (e.g., overnight) for all items in the catalog, allowing fast retrieval when a user is Browse.

The next video will discuss practical issues and modifications to scale this content-based filtering approach for very large item catalogs.

## Scaling Recommender Systems: Retrieval and Ranking

Modern recommender systems often deal with catalogs containing thousands, millions, or even tens of millions of items (movies, ads, songs, products). Running a deep learning model to predict a user's preference for *every single item* in such a large catalog every time a user visits is computationally infeasible.

To address this, large-scale recommender systems typically employ a **two-step architecture: Retrieval and Ranking.**

### 1. Retrieval (Candidate Generation):

* **Goal:** Efficiently generate a **large list of plausible item candidates** (e.g., 100s or 1000s) from the entire catalog. This list should be broad enough to include most items the user might like, even if it also includes some irrelevant ones.
* **Characteristics:** This step prioritizes **speed and broad coverage** over extreme accuracy. It often uses simpler, pre-computed methods or less computationally intensive operations.
* **Examples of Retrieval Strategies:**
    * **Similarity Search:** For each of the last 10 movies the user watched, find the 10 most similar movies (using pre-computed item-item similarities, as discussed in previous videos). This can be a simple lookup.
    * **Genre/Category Favorites:** Add the top 10 most popular movies from the user's top 3 favorite genres.
    * **Geographic/Popularity Trends:** Add the top 20 most popular movies in the user's country.
    * **Other Heuristics:** Items trending locally, items related to user's explicitly stated interests, etc.
* **Output:** A combined list of candidate items (duplicates removed, already-watched/purchased items removed).

### 2. Ranking (Fine-Tuning and Personalization):

* **Goal:** Take the smaller list of plausible candidates from the retrieval step and **fine-tune their ranking** to select the very best items for the user.
* **Characteristics:** This step prioritizes **accuracy and personalization**. It can afford to use more complex, computationally intensive models.
* **Process:**
    * For *each* item in the retrieved candidate list:
        1.  Compute the user's latent vector ($\vec{v}_u$) by feeding the user's features ($X_u$) through the user network (one inference pass).
        2.  Access the item's pre-computed latent vector ($\vec{v}_m$).
        3.  Feed the user's latent vector and the item's latent vector into the final prediction part of the neural network (often just a dot product, or a small final layer) to compute the predicted rating or probability of engagement.
    * **Output:** A ranked list of items, ordered by their predicted rating for the user. The top few items are then displayed to the user.

### Optimizations:

* **Pre-computation:** Item latent vectors ($\vec{v}_m$) can be pre-computed offline for all items.
* **Single User Inference:** The user's latent vector ($\vec{v}_u$) needs to be computed only once per user session.
* **Reduced Scope for Ranking Model:** The full, accurate ranking model is only run on a *few hundreds* of items, not millions, making it feasible in real-time.

### Deciding Retrieval Quantity:

* **Trade-off:** Retrieving more items leads to potentially better overall recommendations (higher recall of good items), but increases the computational cost of the ranking step.
* **Optimization:** Use offline experiments to determine the optimal number of items to retrieve (e.g., 100, 500, 1000 items) by evaluating the impact on recommendation relevance.

This two-step approach allows recommender systems to deliver both **fast and accurate** recommendations even with extremely large catalogs. The next video will discuss ethical issues associated with recommender systems.

## Ethical Considerations in Recommender Systems

Recommender systems, while highly profitable, also have potential for problematic use cases that can negatively impact individuals and society. It's crucial to adopt an ethical approach when building them.

### Goals of Recommender Systems (and their potential downsides):

Recommender systems are configured to optimize various objectives, some of which can have unintended consequences:

1.  **Maximize User Satisfaction (e.g., 5-star ratings):** Recommending movies users are most likely to rate highly. (Generally innocuous).
2.  **Maximize Purchase Likelihood:** Recommending products users are most likely to buy. (Generally innocuous).
3.  **Maximize Ad Clicks (and Advertiser Bids):** Showing ads users are most likely to click, especially if the advertiser bids high.
    * **Problem:** This can amplify harmful businesses (e.g., highly exploitative payday loan companies). A profitable but exploitative business can outbid ethical ones, leading to a virtuous cycle for harm: more profit allows higher bids, leading to more traffic to harmful businesses.
    * **Amelioration:** Refuse to show ads from exploitative businesses (though defining "exploitative" is difficult).
4.  **Maximize Profit/Margin:** Recommending products that generate the most profit for the company, not necessarily the most relevant or useful for the user.
    * **Problem:** Lack of transparency for users. Users might assume recommendations are for their benefit, but they are serving the company's profit.
    * **Amelioration:** Be transparent with users about the criteria for recommendations (e.g., "sorted by highest profit").
5.  **Maximize User Engagement/Watch Time:** Recommending content that keeps users on the platform longer.
    * **Problem:** Can inadvertently amplify harmful content like conspiracy theories, hate speech, or toxicity, because such content is often highly engaging. This has been widely reported for social media and video platforms.
    * **Amelioration:** Actively filter out problematic content (hate speech, fraud, scams, violent content), though defining "problematic" is complex.

### General Ethical Considerations and Ameliorations:

* **Transparency:** Be transparent with users about why specific items are being recommended (e.g., "Recommended because it's popular," "Recommended to maximize our profit"). This builds trust.
* **Content Moderation:** Implement robust systems to filter out problematic, harmful, or incendiary content, even if it is highly engaging.
* **Bias Mitigation:** Ensure the system does not unfairly discriminate or amplify harmful stereotypes.
* **Refuse Harmful Businesses:** Companies should consider refusing to serve (e.g., advertise for) businesses that engage in exploitative or harmful practices.
* **Debate and Diverse Perspectives:** For challenging ethical dilemmas, invite open discussion and debate within the team, involving diverse perspectives.
* **Prioritize Societal Benefit:** Ultimately, strive to build systems that genuinely make society and people better off. If a project feels unethical, even if financially sound, consider walking away.

Recommender systems are powerful and lucrative, but their builders have a responsibility to consider potential harms and actively design for positive societal impact.

## Implementing Content-Based Filtering with TensorFlow

This video walks through the key code concepts for implementing a content-based filtering algorithm using TensorFlow, particularly highlighting its flexible API for building custom model structures.

### Model Architecture: Two Sequential Networks + Dot Product

The content-based filtering model uses two separate sequential neural networks (towers), one for users and one for items, whose outputs are then dot-producted.

1.  **User Network:**
    ```python
    user_nn = tf.keras.Sequential([
        # Example: two Dense hidden layers
        tf.keras.layers.Dense(units=128, activation='relu'), # Hidden layer 1
        tf.keras.layers.Dense(units=64, activation='relu'),  # Hidden layer 2
        # Output layer with 32 units for the user's latent vector
        tf.keras.layers.Dense(units=32, activation='relu') # Use ReLU or linear based on preference
    ])
    ```
2.  **Item Network (Movie Network):**
    ```python
    item_nn = tf.keras.Sequential([
        # Example: two Dense hidden layers
        tf.keras.layers.Dense(units=128, activation='relu'), # Hidden layer 1
        tf.keras.layers.Dense(units=64, activation='relu'),  # Hidden layer 2
        # Output layer with 32 units for the item's latent vector
        tf.keras.layers.Dense(units=32, activation='relu') # Use ReLU or linear based on preference
    ])
    ```
      * Both `user_nn` and `item_nn` use ReLU activation for hidden layers.
      * Their final dense layers both output 32 units, ensuring the latent vectors have the same dimension for the dot product.

### Combining Networks and Computing Output:

TensorFlow's Keras Functional API is used to define how inputs flow through these networks and are combined.

1.  **Define Inputs:**
    ```python
    user_features_input = tf.keras.layers.Input(shape=(user_features_dim,)) # Placeholder for user raw features
    item_features_input = tf.keras.layers.Input(shape=(item_features_dim,)) # Placeholder for item raw features
    ```
2.  **Compute Latent Vectors:**
    ```python
    vu = user_nn(user_features_input) # Pass user features through user network
    vm = item_nn(item_features_input) # Pass item features through item network
    ```
3.  **L2 Normalization (Important Refinement):**
      * **Purpose:** Normalizing the length (L2 norm) of the latent vectors $\\vec{v}\_u$ and $\\vec{v}\_m$ to 1 (i.e., making them unit vectors) often improves algorithm performance and stability.
      * **Implementation:**
        ```python
        vu_normalized = tf.linalg.normalize(vu, axis=1)[0] # Normalize along the feature axis
        vm_normalized = tf.linalg.normalize(vm, axis=1)[0] # Normalize along the feature axis
        ```
        *(Note: `tf.linalg.normalize` returns both normalized vector and its norm; we take the 0th element for the vector)*
4.  **Compute Dot Product:**
      * **Purpose:** The dot product of the normalized latent vectors gives the final predicted rating.
      * **Implementation:** Keras provides a special layer for this.
        ```python
        predicted_rating = tf.keras.layers.Dot(axes=1)([vu_normalized, vm_normalized]) # Dot product along the feature dimension
        ```
5.  **Define Overall Model:**
    ```python
    model = tf.keras.Model(inputs=[user_features_input, item_features_input], outputs=predicted_rating)
    ```
      * This tells Keras that the model takes two inputs (user features, item features) and produces one output (predicted rating).

### Training the Model:

  * **Cost Function:** The mean squared error (MSE) loss is typically used for continuous ratings.
    ```python
    model.compile(optimizer='adam', loss='mse') # Or other optimizers/losses
    ```
  * **Training:**
    ```python
    model.fit([X_users_train, X_items_train], y_ratings_train, epochs=...)
    ```

### Key Takeaways:

  * TensorFlow is flexible enough to implement custom model architectures like this two-tower network, not just simple sequential ones.
  * The `tf.keras.layers.Dot` layer is used for efficiently computing dot products between learned latent vectors.
  * **L2 normalization of latent vectors (`tf.linalg.normalize`) is a crucial refinement** that often improves stability and performance.
  * **Feature Engineering:** Designers often spend significant time carefully crafting the input features ($X\_u, X\_m$) for these content-based filtering algorithms.

## Principal Components Analysis (PCA): Dimensionality Reduction for Visualization

This video introduces **Principal Components Analysis (PCA)**, an unsupervised learning algorithm primarily used for **dimensionality reduction**, especially for **visualization** of high-dimensional data.

### What is PCA?

* **Goal:** To take a dataset with many features (e.g., 10, 50, 1000+) and reduce the number of features to a much smaller number (typically 2 or 3) while retaining as much important information as possible.
* **Primary Use:** Visualization. You can't plot data with more than 3 features, so PCA helps project it onto a 2D or 3D space for inspection.
* **Intuition:** PCA finds new axes (combinations of original features) along which the data varies the most.

### Examples of PCA in Action:

1.  **Car Data: Length (x1) vs. Width (x2)**
    * **Scenario:** $x_1$ (length) varies a lot, $x_2$ (width) varies very little (cars fit roads).
    * **PCA's Action:** PCA would effectively decide to keep $x_1$ and discard $x_2$, as $x_2$ provides little unique variation.
2.  **Car Data: Length (x1) vs. Wheel Diameter (x2)**
    * **Scenario:** $x_1$ (length) varies a lot, $x_2$ (wheel diameter) varies some.
    * **PCA's Action:** PCA would still likely prioritize $x_1$ as the most informative single feature if reducing to one dimension.
  
<img src="/metadata/pca.png" width="400" />

3.  **Car Data: Length (x1) vs. Height (x2)**
    * **Scenario:** Both $x_1$ (length) and $x_2$ (height) vary significantly, often correlated (longer cars tend to be taller).
    * **Challenge:** You can't just pick one (e.g., length) and discard the other (height) without losing significant information.
    * **PCA's Action:** PCA would find a **new axis (let's call it $Z$)** that captures the combined variation of length and height. This $Z$-axis would represent something like the "size" of the car. Data points' coordinates on this new $Z$-axis would effectively reduce two features to one, while preserving much of the original data's variability. This $Z$-axis is not a third dimension sticking out but a new orientation within the existing 2D plane.

4.  **3D Data on a "Pancake" (Manifold Learning):**
    * **Scenario:** You have 3 features ($x_1, x_2, x_3$), but the data points effectively lie on a 2D surface (like a thin pancake) embedded in 3D space.
    * **PCA's Action:** PCA can reduce these 3 features down to 2 new features ($Z_1, Z_2$) by finding the 2 dimensions of the "pancake" that capture most of the data's variance. This allows visualization on a 2D plot.

5.  **Country Development Data (50 features):**
    * **Scenario:** Data for many countries with 50 features each (GDP, per capita GDP, Human Development Index, life expectancy, etc.). Impossible to plot directly.
    * **PCA's Action:** PCA reduces these 50 features to 2 features ($Z_1, Z_2$).
        * **Interpretation (often possible post-hoc):** $Z_1$ might loosely correspond to "total country size/economy" (e.g., GDP), and $Z_2$ might correspond to "development level per person" (e.g., per capita GDP, HDI).
    * **Visualization:** Plotting countries on the $Z_1$ vs. $Z_2$ axes allows visual clustering of similar countries (e.g., large developed nations, small highly developed nations, smaller less developed nations, large less developed nations).

### Core Principle of PCA:

PCA finds new axes (called principal components) that are linear combinations of the original features. It selects the axes along which the data has the **largest variance**, thereby capturing the most "information" (or spread) in the data in fewer dimensions.

### Why Use PCA?

* **Visualization:** Makes high-dimensional data plottable (2D or 3D), helping data scientists understand its structure, patterns, and outliers.
* **Insight:** Can reveal hidden relationships or groupings in the data.
* **Data Exploration:** Helps identify funny or unexpected patterns in the dataset.

The next video will delve into the exact mechanism of how PCA works.

## Principal Components Analysis (PCA): How it Works

PCA is an unsupervised learning algorithm for **dimensionality reduction**, primarily used to reduce a large number of features to a smaller set (e.g., 2 or 3) for visualization and data understanding.

### Goal of PCA:

To find a new axis (or set of axes), called **principal components**, such that when the original data is projected onto this new axis, the projected data retains the **largest possible variance (spread)**. Maximizing variance means minimizing the information lost during dimensionality reduction.

### Preprocessing Steps:

Before applying PCA, features should be preprocessed:

1.  **Zero Mean Normalization:** Subtract the mean from each feature so that each feature has a mean of zero. This centers the data around the origin.
2.  **Feature Scaling:** If features are on very different scales (e.g., house size in sq ft vs. number of bedrooms), perform feature scaling (e.g., min-max scaling or z-score normalization) to ensure features contribute equally to variance calculation.

### Finding the Principal Component (for 2D to 1D Reduction):

* **Visualization:** Imagine plotting data points (e.g., car length $x_1$ vs. height $x_2$) in a 2D space.
* **Projection:** PCA aims to find a new axis ($Z$-axis) such that when each data point is **projected** onto this $Z$-axis (by drawing a perpendicular line from the point to the axis), the resulting projected points are as spread out as possible.
* **Optimal Axis:** The $Z$-axis that maximizes the variance of these projected points is called the **first principal component**. This axis captures the most significant direction of variation in the data.

### Projecting a Data Point onto the Principal Component:

* Let the first principal component be represented by a **unit vector** (length 1) in the direction of the $Z$-axis (e.g., $[0.71, 0.71]$ for a diagonal axis).
* To project a data point (e.g., $[2, 3]$ for $x_1, x_2$) onto this axis, compute the **dot product** of the data point's vector and the principal component unit vector.
    * Example: $[2, 3] \cdot [0.71, 0.71] = (2 \times 0.71) + (3 \times 0.71) = 3.55$. This $3.55$ is the new single coordinate for the data point on the $Z$-axis.

### Multiple Principal Components:

* If reducing to more than one dimension (e.g., 50 features to 3 features):
    * The **second principal component** is an axis perpendicular (at 90 degrees) to the first principal component, capturing the next largest amount of variance in the remaining data.
    * The **third principal component** is perpendicular to both the first and second, and so on.

### PCA vs. Linear Regression:

PCA is fundamentally different from Linear Regression:

* **PCA (Unsupervised):**
    * Has no target label $Y$. It only uses input features ($X_1, X_2, \dots$).
    * Treats all features equally, aiming to find new axes ($Z$-axes) that minimize the squared perpendicular distance from data points to the axis.
    * Goal: **Reduce dimensionality** by finding directions of maximum variance.
* **Linear Regression (Supervised):**
    * Has a target label $Y$.
    * Fits a line (or hyperplane) to predict $Y$ from $X$.
    * Goal: **Minimize squared vertical distances** (along the $Y$-axis) between predictions and actual $Y$ values.

### Reconstruction (Optional Step):

* Given a projected point on the $Z$-axis (e.g., $Z=3.55$), PCA can **reconstruct** an approximation of the original high-dimensional data.
* Formula: Multiply the $Z$ coordinate by the principal component unit vector (e.g., $3.55 \times [0.71, 0.71] = [2.52, 2.52]$).
* **Purpose:** Shows how much information is retained. The reconstructed point is the closest point on the principal component axis to the original data point.

**Summary:** PCA finds new axes (principal components) that capture the maximum variance in the data when projected. This reduces dimensionality, allowing for visualization and better understanding of high-dimensional datasets. The next video will show how to implement PCA using scikit-learn.

## Principal Components Analysis (PCA): Implementation and Applications

This video demonstrates how to implement PCA using the `scikit-learn` library and discusses its primary applications, focusing on its most common use today: visualization.

### Steps to Implement PCA with Scikit-learn:

1.  **Preprocessing (Optional but Recommended):**
    * **Mean Normalization:** PCA's `fit` function automatically handles mean normalization (subtracting the mean from each feature). You do not need to do this separately.
    * **Feature Scaling:** If your features have very different ranges (e.g., GDP in trillions vs. life expectancy in tens), it's crucial to perform feature scaling (e.g., standardization, Min-Max scaling) *before* applying PCA. This ensures features with larger scales don't disproportionately dominate the principal components.
2.  **Run PCA Algorithm ("Fit"):**
    * Use `sklearn.decomposition.PCA` and specify the number of principal components (`n_components`) you want (e.g., 1, 2, or 3 for visualization).
    * Call the `fit()` method on your preprocessed data `X` (e.g., `pca.fit(X)`). This step learns the principal components (new axes).
    * **Example (1 component):** `pca_1 = PCA(n_components=1).fit(X)`
3.  **Analyze Explained Variance Ratio:**
    * After fitting, check `pca.explained_variance_ratio_`. This tells you the percentage of the original data's variance (information) captured by each principal component.
    * **Example (2 components):** `[0.992, 0.008]` means the first component explains 99.2% of variance, and the second explains 0.8%.
4.  **Transform Data (Project):**
    * Use the `transform()` method to project your original data onto the newly found principal components (axes).
    * `Z = pca_1.transform(X)` (for 1 component). The output `Z` will have `n_components` features for each example.
    * Each data point is now represented by fewer numbers (its coordinates on the new axes).
5.  **Visualize:**
    * If `n_components` is 2 or 3, you can now plot the transformed data ($Z_1$ vs. $Z_2$ or $Z_1$ vs. $Z_2$ vs. $Z_3$) to visualize the high-dimensional data.

```python
X = np.array([[1,1], [2,1], [3,2], [-1,-1], [-2,-1], [-3,-2]])
pca_1 = PCA(n_components=1)
pca_1.fit(X)
pca_1.explained_variance_ratio_ # 0.992
X_trans_1 = pca_1.transform(X)
X_reduced_1 = pca.inverse_transform(X_trans_1) # array[[1.38, 2.22, 3.60, -1.38, -2.22, -3.60]])
```

### Applications of PCA:

1.  **Visualization (Most Common Use Today):**
    * **Purpose:** To reduce data dimensionality to 2 or 3 features, making it plottable and allowing data scientists to understand its structure, identify clusters, outliers, or trends in high-dimensional data.
    * **Example:** Visualizing countries based on 50 development features by reducing to 2 components.

2.  **Data Compression (Less Common Today):**
    * **Purpose:** To reduce storage space or network transmission costs by representing data with fewer features.
    * **Mechanism:** Reduce high-dimensional data (e.g., 50 features per car) to a smaller number of principal components (e.g., 10 features).
    * **Trend:** Less popular now due to cheaper storage and faster networks.

3.  **Speeding Up Supervised Learning (Historically More Common, Less So Now):**
    * **Purpose:** To reduce the number of features before feeding them to a supervised learning algorithm if a high number of features makes training too slow.
    * **Mechanism:** Reduce 1,000 features to 100 using PCA, then train your supervised model.
    * **Trend:** Less relevant with modern algorithms like deep learning, which can handle high-dimensional inputs efficiently without PCA preprocessing. PCA itself has computational cost.

**Key Takeaway:** While PCA has been used for compression and speedup in the past, its most prevalent and valuable application today is **visualization**, helping gain insights into complex, high-dimensional datasets.
