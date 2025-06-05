# Multiple Linear Regression: Beyond a Single Feature

This week, we enhance linear regression to handle **multiple input features**, significantly boosting its power.

## Model with Multiple Features

Previously, we used a single feature ($x_1$, house size) to predict price ($y$). Now, let's incorporate multiple features:
* $x_1$: Size
* $x_2$: Number of bedrooms
* $x_3$: Number of floors
* $x_4$: Age of home

We denote these features as $x_j$ (where $j=1$ to $n$, and $n$ is the total number of features; here $n=4$). A training example $x^{(i)}$ becomes a **vector** (list) of these $n$ features for the $i$-th example: $x^{(i)} = [x_1^{(i)}, x_2^{(i)}, x_3^{(i)}, x_4^{(i)}]$.

The linear regression model with multiple features becomes:
$f_{w,b}(x) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b$

* **Interpreting Parameters ($w_j$):** Each $w_j$ indicates how much the output ($y$) changes for a unit increase in feature $x_j$, holding other features constant. For example, if $w_1=0.1$ and $x_1$ is in sq ft, it means $0.1 \times \$1000 = \$100$ increase per sq ft. $b$ is the base price.

## Vectorized Notation for Multiple Linear Regression

For $n$ features, the model is $f_{w,b}(x) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$.

To simplify this, we use **vector notation**:
* **Parameters Vector ($\vec{w}$):** $\vec{w} = [w_1, w_2, \dots, w_n]$
* **Features Vector ($\vec{x}$):** $\vec{x} = [x_1, x_2, \dots, x_n]$

Using the **dot product**, the model can be written compactly as:
$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$

Where $\vec{w} \cdot \vec{x} = w_1x_1 + w_2x_2 + \dots + w_nx_n$.

This type of model is called **Multiple Linear Regression**. This is NOT multivariate regression (that's something else)

# Vectorization: Speeding Up Machine Learning Code

**Vectorization** is a crucial technique for implementing learning algorithms. It makes code both **shorter** and **significantly more efficient**, leveraging modern hardware like CPUs and GPUs.

## What is Vectorization?

It means performing operations on entire arrays (vectors) at once, rather than iterating through individual elements using loops.

### Example: Computing the Model's Prediction ($f_{w,b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$)

Let's assume we have $n=3$ features, so $\vec{w} = [w_1, w_2, w_3]$ and $\vec{x} = [x_1, x_2, x_3]$. In Python with NumPy, array indexing starts from 0 (e.g., `w[0]`, `x[0]`).

1.  **Non-Vectorized (Manual calculation):**
    ```python
    f = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b
    ```
    * **Issue:** Inefficient for large $n$, cumbersome to write.

2.  **Non-Vectorized (Using a for loop):**
    ```python
    f = 0
    for j in range(n): # j goes from 0 to n-1
        f += w[j]*x[j]
    f += b
    ```
    * **Issue:** Better for writing, but still computationally inefficient due to the explicit loop.

3.  **Vectorized (Using NumPy):**
    ```python
    import numpy as np
    f = np.dot(w, x) + b
    ```
    * **Benefits:**
        * **Shorter Code:** One line for the dot product.
        * **Faster Execution:** NumPy's `dot` function is highly optimized. Behind the scenes, it utilizes **parallel hardware** (CPU's SIMD instructions, GPU cores) to perform computations on multiple data elements simultaneously. This dramatically speeds up operations, especially for large vectors.

# How Vectorization Works: The Magic Behind the Speed

Vectorization dramatically speeds up machine learning code by leveraging your computer's hardware for parallel processing.

## Non-Vectorized (Sequential) Execution:

* A traditional `for` loop executes operations **one after another**.
* For example, in a loop calculating `w[j]*x[j]`, each multiplication and addition is done sequentially for `j=0`, then `j=1`, and so on. This is slow for large datasets.

## Vectorized (Parallel) Execution:

* **NumPy functions** (like `np.dot` for dot products) are **vectorized implementations**.
* **Behind the scenes, your computer's hardware (CPU with SIMD instructions, or GPU cores) can:**
    1.  Perform **multiple element-wise operations (e.g., multiplications)** **simultaneously in parallel**.
    2.  Use **specialized hardware** to efficiently perform subsequent operations (e.g., summing up results) much faster than sequential additions.

## Why Vectorization Matters for ML:

* **Speed:** Vectorized code runs **much faster** than non-vectorized code, especially for large datasets and complex models. This can reduce computation time from hours to minutes.
* **Scalability:** It's essential for **scaling learning algorithms** to the massive datasets prevalent in modern machine learning.
* **Conciseness:** It makes code **shorter and easier to read**.

## Example: Updating Multiple Parameters in Gradient Descent

For $n$ features, updating $w_j = w_j - \alpha \cdot d_j$ (where $d_j$ is the derivative for $w_j$):

* **Non-Vectorized (for loop):**
    ```python
    for j in range(n):
        w[j] = w[j] - 0.1 * d[j]
    ```
    * Executes each update sequentially.

* **Vectorized (NumPy):**
    ```python
    w = w - 0.1 * d # w and d are NumPy arrays
    ```
    * The computer performs all $n$ subtractions and multiplications **in parallel** in a single step using specialized hardware.

The accompanying optional lab introduces NumPy and demonstrates how vectorized code runs significantly faster than explicit loops. This foundational understanding of vectorization is key to efficient machine learning implementation.

# Gradient Descent for Multiple Linear Regression with Vectorization

This video consolidates **gradient descent**, **multiple linear regression**, and **vectorization** for efficient model training.

## Multiple Linear Regression Model (Vectorized)

* **Parameters:** Instead of individual $w_1, \dots, w_n$ and $b$, we represent them as a **vector $\vec{w}$ (of length $n$) and a scalar $b$**.
* **Model Equation:**
    $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$
    where $\vec{w} \cdot \vec{x}$ is the dot product.
* **Cost Function:** The cost function $J$ is now $J(\vec{w}, b)$, taking a vector and a scalar as input.

## Gradient Descent Update Rules (Vectorized)

The parameters $\vec{w}$ and $b$ are updated repeatedly and **simultaneously**:

* **For each $w_j$ (where $j=1 \dots n$):**
    $w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(x^{(i)}) - y^{(i)})x_j^{(i)}$
* **For $b$:**
    $b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(x^{(i)}) - y^{(i)})$

These derivatives are derived from the squared error cost function and extend naturally from the single-feature case.

## The Normal Equation (Alternative Method - Optional)

* The **Normal Equation** is an **alternative, non-iterative method** to find optimal $w$ and $b$ for **linear regression *only***. It uses advanced linear algebra to solve directly.
* **Disadvantages:**
    * **Not generalizable** to most other ML algorithms (e.g., logistic regression, neural networks).
    * **Slow for large number of features (n)**.
* While not recommended for manual implementation in most cases, some mature ML libraries might use it internally for linear regression. Gradient descent is generally preferred for its broader applicability and scalability.

**You now know how to implement multiple linear regression**, one of the most widely used learning algorithms. The next videos will cover practical tricks like feature scaling and choosing an appropriate learning rate to make this algorithm even more effective.
