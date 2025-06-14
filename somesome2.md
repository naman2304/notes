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
