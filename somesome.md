---

Welcome to Week 3! This week, we'll shift from **linear regression (predicting numbers)** to **classification (predicting categories)**, specifically **binary classification** (two output classes). Linear regression is generally **not suitable for classification problems**.

## Why Linear Regression Fails for Classification

Let's use the example of classifying a tumor as **malignant (1, positive class)** or **benign (0, negative class)** based on tumor size.

1.  **Initial Attempt with Linear Regression:**
    * If you plot tumor size (x) vs. label (y=0 or 1) and fit a straight line (linear regression), it might look reasonable for a small dataset.
    * You could then set a **threshold** (e.g., 0.5):
        * Predict y=0 if output $< 0.5$.
        * Predict y=1 if output $\ge 0.5$.
    * This creates a **decision boundary** (a vertical line) separating the classes.

<img src="/metadata/classification_motivation.png" width="600" />

2.  **The Problem with Outliers:**

    * If you add just one more training example (e.g., a very large tumor that is also malignant) far to the right, the **best-fit linear regression line will significantly shift**.
    * This shift causes the **decision boundary to also shift** to the right, leading to incorrect classifications for previously well-classified examples. A large tumor shouldn't change how smaller tumors are classified.

## Introducing Logistic Regression

* Linear regression can predict values outside [0, 1], which is problematic for binary classification where outputs are 0 or 1.
* **Logistic regression**, despite its name, is a classification algorithm designed to output values **always between 0 and 1**, avoiding the problems seen with linear regression. It's one of the most popular and widely used learning algorithms today.

The next video will delve into the concept of the decision boundary and introduce the logistic regression algorithm in detail. The optional lab will let you experience why linear regression often fails for classification.

## Logistic Regression: A Classification Algorithm

**Logistic regression** is a widely used classification algorithm, especially for **binary classification** problems where the output variable $y$ can only be one of two values (e.g., 0 or 1, No or Yes, False or True). By convention, we often use 0 for the "negative class" and 1 for the "positive class."

### The Sigmoid (Logistic) Function

Logistic regression uses the **Sigmoid function** (or logistic function) to map any real-valued input to an output between 0 and 1.

* **Formula:** $g(z) = \frac{1}{1 + e^{-z}}$
* **Properties:**
    * As $z \rightarrow \infty$, $g(z) \rightarrow 1$.
    * As $z \rightarrow -\infty$, $g(z) \rightarrow 0$.
    * When $z = 0$, $g(z) = 0.5$.
* This S-shaped curve is ideal for classification as it naturally outputs probabilities.

<img src="/metadata/sigmoid.png" width="600" />

### The Logistic Regression Model

The logistic regression model combines a linear function with the Sigmoid function:

1.  **Linear Part (z):** First, calculate a linear combination of input features and parameters, similar to linear regression:
    $z = \vec{w} \cdot \vec{x} + b$
2.  **Sigmoid Part (f(x)):** Pass this $z$ value through the Sigmoid function:
    $f_{\vec{w},b}(\vec{x}) = g(z) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$

This model takes input features $\vec{x}$ and outputs a value between 0 and 1.

### Interpreting the Output

The output of logistic regression, $f_{\vec{w},b}(\vec{x})$, is interpreted as the **predicted probability that the output class $y$ is 1 (positive class) given the input $\vec{x}$**.

* Example: If $f(\text{tumor size}) = 0.7$, it means there's a 70% chance the tumor is malignant.
* Since $y$ must be either 0 or 1, if $P(y=1|\vec{x}; \vec{w},b) = 0.7$, then $P(y=0|\vec{x}; \vec{w},b) = 1 - 0.7 = 0.3$ (30% chance of being benign). This is read as probability of y = 1 given input $\vec{x}$, parameters $\vec{w}$ and b.

While the term "regression" is in its name, logistic regression is fundamentally a **classification algorithm**.

## Logistic Regression: The Decision Boundary

Logistic regression predicts probabilities $f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b)$, where $g$ is the Sigmoid function. To classify, we set a **threshold**, typically 0.5:

* If $f_{\vec{w},b}(\vec{x}) \ge 0.5$, predict $y=1$.
* If $f_{\vec{w},b}(\vec{x}) < 0.5$, predict $y=0$.

### When does $f_{\vec{w},b}(\vec{x}) \ge 0.5$?

Recall that $g(z) \ge 0.5$ whenever $z \ge 0$. Since $z = \vec{w} \cdot \vec{x} + b$, the model predicts 
* $y=1$ when $\vec{w} \cdot \vec{x} + b \ge 0$, and
* $y=0$ when $\vec{w} \cdot \vec{x} + b < 0$.

### The Decision Boundary

The **decision boundary** is the line (or surface) where $\vec{w} \cdot \vec{x} + b = 0$. This is the point where the model is 50/50 on its prediction.

<img src="/metadata/decision_boundary.png" width="400" />

* **Example (Two Features):** If $x_1$ and $x_2$ are features, and parameters are $w_1=1, w_2=1, b=-3$, then the decision boundary is $1 \cdot x_1 + 1 \cdot x_2 - 3 = 0$, or $x_1 + x_2 = 3$.
    * Points where $x_1 + x_2 \ge 3$ predict $y=1$.
    * Points where $x_1 + x_2 < 3$ predict $y=0$.
    * This forms a **linear decision boundary** (a straight line).

### Non-Linear Decision Boundaries with Polynomial Features

Just like in linear regression, you can use **polynomial features** in logistic regression to create complex, **non-linear decision boundaries**.

<img src="/metadata/non_linear_boundary.png" width="250" />

* **Example: Circular Boundary**
    * Define $z = w_1x_1^2 + w_2x_2^2 + b$.
    * If $w_1=1, w_2=1, b=-1$, the decision boundary is $x_1^2 + x_2^2 - 1 = 0$, or $x_1^2 + x_2^2 = 1$, which is a circle.
    * Points outside the circle predict $y=1$; points inside predict $y=0$.
* By including higher-order polynomial terms (e.g., $x_1^2, x_1x_2, x_2^2$), logistic regression can define even more complex decision boundaries (e.g., ellipses, arbitrary shapes).

**Note:** If you only use linear features ($x_1, x_2, \dots$), the decision boundary will always be linear.

## Cost Function for Logistic Regression

The **cost function** measures how well a model's parameters fit the training data. For logistic regression, the **squared error cost function (used in linear regression) is not suitable** because it results in a **non-convex cost function** with many local minima, preventing gradient descent from reliably finding the global minimum.

Instead, a different **loss function** is used for logistic regression that makes the overall cost function **convex**.

### The Logistic Loss Function (for a single training example)

Let $f(\vec{x})$ be the logistic regression model's prediction (a probability between 0 and 1), and $y$ be the true label (0 or 1).

The loss, $L(f(\vec{x}), y)$, for a single training example is defined as:

* If $y = 1$: $L(f(\vec{x}), y) = -\log(f(\vec{x}))$
    * **Intuition:**
        * If $f(\vec{x})$ is close to 1 (correct prediction), loss is very small (near 0).
        * If $f(\vec{x})$ is close to 0 (incorrect prediction), loss is very large (approaching infinity), penalizing the model heavily. This incentivizes the model to predict values close to 1 when the true label is 1.

* If $y = 0$: $L(f(\vec{x}), y) = -\log(1 - f(\vec{x}))$
    * **Intuition:**
        * If $f(\vec{x})$ is close to 0 (correct prediction), loss is very small (near 0).
        * If $f(\vec{x})$ is close to 1 (incorrect prediction), loss is very large (approaching infinity), penalizing the model heavily. This incentivizes the model to predict values close to 0 when the true label is 0.

### Overall Cost Function

The **cost function $J(\vec{w}, b)$** for the entire training set is the average of these individual losses:

$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L(f_{\vec{w},b}(x^{(i)}), y^{(i)})$

This cost function is **convex**, ensuring that gradient descent can reliably converge to a single global minimum.

The next video will provide a more compact way to write this cost function and then discuss how to apply gradient descent to train the logistic regression model. The optional lab will visually demonstrate the difference between the non-convex squared error cost and the convex logistic loss cost.
