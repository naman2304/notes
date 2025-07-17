# Calculus for Machine Learning and Data Science

This course focuses on **calculus**, a fundamental mathematical tool for **machine learning** and **data science**.

### Why Calculus in Machine Learning?

* **Minimizing and Maximizing Functions:** A core task in machine learning is to create a **cost function** (or error function) and then **minimize** it to train a model. Calculus, specifically **derivatives** and **gradients**, are essential for this optimization.
* **Gradient Descent:** This widely used optimization algorithm involves taking small steps in the direction that reduces the cost function. The **gradient** precisely indicates this direction.
* **Understanding Optimizers:** Knowing calculus provides intuition into how optimizers like gradient descent work, moving beyond just treating them as "black boxes."
* **High-Dimensional Spaces:** While derivatives are often visualized as slopes in 1D or 2D, machine learning often deals with very high-dimensional data (e.g., 10,000 or a million dimensions). Calculus provides the tools to handle derivatives in these complex spaces.
    * **First derivative:** Provides the direction of the steepest ascent/descent.
    * **Second derivative (Hessian matrix):** Describes the curvature of the space, which can be useful for more advanced optimization methods.

### Key Concepts and Techniques

* **Derivatives:** Measure the rate at which a function changes.
* **Gradients:** A generalization of the derivative for multi-variable functions, pointing in the direction of the steepest ascent.
* **Newton's Method:** An advanced optimization technique that can be significantly faster than gradient descent in certain applications, especially when dealing with fewer parameters. It utilizes both first and second derivatives. It provides a valuable alternative in your "toolkit" of optimization methods.

## Week 1: Introduction to Derivatives and Their Role in Machine Learning

This week introduces the **derivative**, a fundamental concept in calculus, and its critical applications in machine learning.

### What is a Derivative?

* The derivative represents the **rate of change** of a function.
* **Velocity** is a classic example of a derivative, representing the rate of change of position over time (e.g., speedometer readings).

### Why are Derivatives Important in Machine Learning?

* **Optimization**: Derivatives are crucial for **optimizing functions**, specifically finding their maximum or minimum values.
* **Loss Function Minimization**: In machine learning, training a model often involves **minimizing a loss function**. This function quantifies how well a model fits the data, and finding its minimum value leads to the best possible model.

### Key Functions and Their Derivatives (to be covered):

* Constant functions
* Linear functions
* Quadratic polynomials
* Exponential functions
* Logarithmic functions

### Rules for Finding Derivatives:

* Sum Rule
* Product Rule
* Chain Rule
* Multiplication by Scalars

### Machine Learning Applications (Motivating Examples):

#### 1. Linear Regression (Predicting House Prices)

* **Problem**: Predict the price of a house based on the number of bedrooms.
* **Data**: A dataset of houses with varying bedrooms and their corresponding prices.
* **Model**: A **line** that best fits the data points (number of bedrooms vs. price).
* **Model Training**: An iterative process where the model adjusts the line to minimize the "distance" to the data points, aiming for the best prediction. This is an **optimization** problem.
* **Prediction**: Once trained, the model can predict the price for a new house (e.g., a 9-bedroom house).

#### 2. Classification / Sentiment Analysis (Alien Language)

* **Problem**: Classify an alien's mood (happy or sad) based on their utterances.
* **Data**: Sentences (e.g., "aack aack aack", "beep beep") and their corresponding moods.
* **Features**: Number of times specific words (e.g., "aack", "beep") appear in a sentence.
* **Model**: A **line** that **separates** the data points into different classes (e.g., happy vs. sad). This line acts as a decision boundary.
* **Model Training**: The model learns to position this line to effectively classify new inputs.
* **Prediction**: For a new alien sentence, the model predicts its mood based on which side of the line it falls.

### Core Mathematical Concepts in Machine Learning Training:

* **Gradients**
* **Derivatives**
* **Optimization**
* **Loss Functions** (e.g., Square Loss, Log Loss)
* **Gradient Descent**

These mathematical concepts underpin the "smart engine" of machine learning models, enabling them to learn and make predictions.

## Understanding Derivatives: Instantaneous Rate of Change

* **Derivative Defined**: A derivative measures the **instantaneous rate of change** of a function at a specific point.

    * **Analogy**: Think of a car's speedometer reading at a particular moment. This is the instantaneous velocity, which is the derivative of the distance function with respect to time.
    * **Contrast with Average Rate of Change**: Average rate of change is like calculating your average speed over an entire trip (total distance/total time). A derivative focuses on what's happening *right now*.

* **Example: Car's Velocity**

    * **Scenario**: A car travels on a straight road. We want to understand its velocity.
    * **Observation**: The car's velocity is unlikely to be constant. It speeds up, slows down, or even stops.

### From Average to Instantaneous Velocity

* **Initial Data**: Let's consider a scenario where we track the distance a car travels every 5 seconds.

    | Time (s) | Distance (m) |
    | :------- | :----------- |
    | 0        | 0            |
    | 5        | 50           |
    | 10       | 122          |
    | 15       | 202          |
    | 20       | 265          |

    * **Is the speed constant?** No.
        * From 10s to 15s (5-second interval): Distance traveled = $202 - 122 = 80$ meters.
        * From 15s to 20s (5-second interval): Distance traveled = $265 - 202 = 63$ meters.
        * Since the distance covered in the same time interval is different (80m vs 63m), the car's speed is not constant.

* **Average Velocity Calculation**:
    * We cannot determine the *exact* velocity at a specific point like 12.5 seconds with this coarse data.
    * However, we *can* calculate the average velocity over an interval.
    * **Average Velocity (10s to 15s)**:
        * This is equivalent to the **slope of the line** connecting the points (10, 122) and (15, 202) on a distance-time graph.
        * Formula for average velocity (or slope): $\frac{\text{Change in Distance}}{\text{Change in Time}}$
        * Average Velocity = $\frac{202 \text{ m} - 122 \text{ m}}{15 \text{ s} - 10 \text{ s}} = \frac{80 \text{ m}}{5 \text{ s}} = 16 \text{ m/s}$

* **Improving the Estimate**: To get a better estimate of the instantaneous velocity, we need to take measurements over **smaller time intervals**.

    * **Refined Data**: Let's say we now have data recorded every second:

        | Time (s) | Distance (m) |
        | :------- | :----------- |
        | 10       | 122          |
        | 11       | 138          |
        | 12       | 155          |
        | 13       | 170          |
        | 14       | 186          |
        | 15       | 202          |

    * **Better Estimate for 12.5 seconds**: We can now calculate the average velocity over a much smaller interval that includes 12.5 seconds, such as from 12 seconds to 13 seconds.
        * Average Velocity (12s to 13s) = $\frac{\text{Distance at 13s} - \text{Distance at 12s}}{\text{13s} - \text{12s}}$
        * Average Velocity = $\frac{170 \text{ m} - 155 \text{ m}}{13 \text{ s} - 12 \text{ s}} = \frac{15 \text{ m}}{1 \text{ s}} = 15 \text{ m/s}$
    * This 15 m/s value is a **much better estimate** for the instantaneous velocity at t = 12.5 seconds compared to the previous 16 m/s.

* **The Limit Concept (Leading to Derivative)**: The core idea of a derivative is to make these time intervals (or the "change in time") **infinitesimally small** – approaching zero. As the interval shrinks, the average velocity over that tiny interval approaches the true instantaneous velocity. This concept is formalized using limits in calculus.

## Instantaneous Velocity and the Derivative

While **average velocity** provides the rate of change over an interval, **instantaneous velocity** describes the rate of change at a single, specific point in time. This is where the **derivative** comes in.

### Estimating Instantaneous Velocity

* **The Goal**: To find the exact velocity at a point, say $t = 12.5$ seconds.
* **The Challenge**: Our available data, even if refined, only gives us distances at discrete time points. We can't directly measure change *at* a single instant.
* **The Approach: Approaching the Limit**:
    * We start by calculating the average velocity ($\frac{\Delta x}{\Delta t}$) over an interval that includes our point of interest. This average velocity represents the **slope of a secant line** connecting two points on the distance-time graph.
    * To get a better estimate, we make the time interval ($\Delta t$) **smaller and smaller**. As we choose points closer and closer to $t = 12.5$, the secant line's slope gets closer to the slope of the curve *at* $t = 12.5$.
    * Imagine this process continuing indefinitely, with the second point getting infinitesimally close to the first point.

### The Derivative: The Slope of the Tangent Line

* As the interval $\Delta t$ approaches zero, the secant line transforms into the **tangent line** at that specific point on the curve.
* The slope of this tangent line is precisely the **instantaneous rate of change**.
* In calculus, this instantaneous rate of change is called the **derivative**.
* We denote an infinitesimally small change in distance as $dx$ and an infinitesimally small change in time as $dt$.
* The derivative, then, is represented as $\frac{dx}{dt}$.
* **Key Concept**: The **derivative of a function at a point** is equal to the **slope of the tangent line to the function's graph at that particular point**.

## Derivative Property: Zero Slope at Extrema

A key property of derivatives is that they can help locate the **maximum or minimum points of a function**. These points are called **extrema**.

### Zero Velocity (Zero Derivative)

* Consider the car's distance-time graph. If the car stops moving, its **velocity is zero**.
* In the provided table, from 19 seconds to 20 seconds, the distance remains constant at 265 meters.
    * Change in distance ($\Delta x$) = $265 - 265 = 0$ meters.
    * Change in time ($\Delta t$) = $20 - 19 = 1$ second.
    * Average velocity = $\frac{\Delta x}{\Delta t} = \frac{0}{1} = 0$ meters/second.
* Geometrically, a constant distance over time is represented by a **horizontal line** on the distance-time graph.
* The **slope of a horizontal line is always zero**. Since the derivative represents the slope of the tangent line (instantaneous velocity), a zero derivative indicates the car is momentarily stopped.

### Identifying Extrema Using Derivatives

* **Trajectory Analysis**: Imagine a car's journey where it moves forward, stops, goes backward, stops, goes forward, etc.
* **Where is the velocity zero?**
    * The velocity of the car is zero at any point where the **tangent line to the curve is horizontal**. This means the derivative at these points is zero. These are the moments when the car is momentarily stopped.
* **Farthest Point (Maximum Distance)**:
    * On the trajectory graph, the point where the car is farthest from its starting point represents a **local maximum** distance.
    * Notice that at this point of maximum distance, the tangent line to the curve is also **horizontal**.
* **The Coincidence (or rather, a Rule)**: It's not a coincidence that the car stops (velocity is zero) at its farthest point from the origin.
    * If the car were still moving at its maximum distance point, it would either move further (meaning it wasn't the maximum) or start moving backward (meaning it just passed the maximum).
    * Therefore, at a **maximum or minimum point** of a function (an extremum), the instantaneous rate of change (the derivative) is **zero**. This means the **tangent line at that point will be horizontal**.

### Conclusion

* To find the **maximum or minimum values** of a function, a critical step is to find the points where its **derivative is zero**. These points are candidates for extrema.

## Derivative Notations
The derivative, representing the instantaneous rate of change, has two primary notations: **Leibniz's notation** and **Lagrange's notation**.

### Leibniz's Notation

* Recall that the slope of a secant line was initially calculated as $\frac{\Delta x}{\Delta t}$ (change in distance over change in time).
* As the interval became infinitesimally small, this ratio transformed into $\frac{dx}{dt}$. Here, $dx$ and $dt$ represent **infinitesimal (extremely small) changes** in distance and time, respectively.
* More generally, if $y$ is a function of $x$, the derivative is expressed as $\frac{dy}{dx}$.
* This can also be written as $\frac{d}{dx}f(x)$, where $\frac{d}{dx}$ is considered an **operator** that, when applied to a function $f(x)$, yields its derivative.

### Lagrange's Notation

* If your function is denoted as $f(x)$, its derivative is expressed as $\mathbf{f'(x)}$.
* The prime symbol ($'$) indicates the derivative of the function. For example, if $f(x)$ represents distance, then $f'(x)$ would represent instantaneous velocity.

Both notations are used interchangeably in mathematics, and the choice often depends on convenience or context within a problem.

## Derivatives of Basic Functions

### Constant Functions

* A **constant function** is a horizontal line, represented by $f(x) = c$, where $c$ is any constant number (e.g., $f(x) = 5$, $f(x) = -173.5$).
* **Derivative**: The derivative of a constant function is **zero**.
    * **Reasoning**: For any two points $(x_0, c)$ and $(x_1, c)$ on a horizontal line, the change in $y$ ($\Delta y$) is always $c - c = 0$.
    * The slope ($\frac{\Delta y}{\Delta x}$) is $\frac{0}{\Delta x} = 0$ (as long as $\Delta x \neq 0$).
    * Since the tangent line to a horizontal line is the line itself, its slope is always zero.
    * **Notation**: If $f(x) = c$, then $f'(x) = 0$ or $\frac{d}{dx}(c) = 0$.
    
### Linear Functions

* A **linear function** (a straight line that is not horizontal) has the general equation $f(x) = ax + b$, where $a$ is the slope and $b$ is the y-intercept.
* **Derivative**: The derivative of a linear function is simply its **slope**, $a$.
    * **Reasoning**: For any line, the tangent line at any point is the line itself. Therefore, the slope of the tangent line is always the same as the slope of the entire line.
    * **Mathematical Proof**:
        * Consider two points on the line: $(x, ax + b)$ and $(x + \Delta x, a(x + \Delta x) + b)$.
        * Calculate the slope ($\frac{\Delta y}{\Delta x}$):
        * As $\Delta x$ approaches zero, the slope remains $a$.
    * **Notation**: If $f(x) = ax + b$, then $f'(x) = a$ or $\frac{d}{dx}(ax + b) = a$.

$$
\frac{[a(x + \Delta x) + b] - [ax + b]}{\Delta x} = \frac{ax + a\Delta x + b - ax - b}{\Delta x} = \frac{a\Delta x}{\Delta x} = a
$$

### Key Takeaway
For a straight line, whether horizontal or sloped, the **derivative at any point is simply the slope of that line**, because the tangent line is always the line itself.

## Derivative of Quadratic Functions: $f(x) = x^2$

* The function $f(x) = x^2$ represents a parabola.
    * To the left of the y-axis (negative x-values), the slope of the tangent lines is negative.
    * To the right of the y-axis (positive x-values), the slope of the tangent lines is positive.
* The derivative is defined as the limit of $\frac{\Delta f}{\Delta x}$ as $\Delta x$ approaches 0.
    * Here, $\Delta f = (x + \Delta x)^2 - x^2$.

### Example: Finding the Derivative at a Specific Point ($x=1$)

Let's estimate the slope of the tangent line at $x=1$ (where $f(1) = 1^2 = 1$). We'll do this by calculating the slopes of secant lines with progressively smaller $\Delta x$ values.

1.  **$\Delta x = 1$**:
    * Points: $(1, 1)$ and $(1+1, (1+1)^2) = (2, 4)$.
    * $\Delta f = 4 - 1 = 3$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{3}{1} = 3$.

2.  **$\Delta x = 0.5$ (or $\frac{1}{2}$)**:
    * Points: $(1, 1)$ and $(1+0.5, (1+0.5)^2) = (1.5, 2.25)$.
    * $\Delta f = 2.25 - 1 = 1.25$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{1.25}{0.5} = 2.5$.

3.  **$\Delta x = 0.25$ (or $\frac{1}{4}$)**:
    * Points: $(1, 1)$ and $(1+0.25, (1+0.25)^2) = (1.25, 1.5625)$.
    * $\Delta f = 1.5625 - 1 = 0.5625$.
    * Slope = $\frac{0.5625}{0.25} = 2.25$.

4.  **Continuing this process**:
    * For $\Delta x = \frac{1}{8}$, the slope is $2.125$.
    * For $\Delta x = \frac{1}{16}$, the slope is $2.0625$.
    * For $\Delta x = \frac{1}{1000}$, the slope is $2.001$.

* **Observation**: As $\Delta x$ gets smaller, the slope of the secant line approaches **2**.
* **Conclusion**: The slope of the tangent line to $f(x) = x^2$ at $x=1$ is 2. Notice that $2 = 2 \times 1$. This hints at the general formula.

### Formal Calculation of the Derivative for $f(x) = x^2$

We want to find $\frac{df}{dx}$ by evaluating the limit of $\frac{\Delta f}{\Delta x}$ as $\Delta x \to 0$.

$$
\frac{\Delta f}{\Delta x} = \frac{f(x + \Delta x) - f(x)}{\Delta x} = \frac{(x + \Delta x)^2 - x^2}{\Delta x}
$$

$$
(x + \Delta x)^2 = x^2 + 2x\Delta x + (\Delta x)^2
$$

$$
\frac{\Delta f}{\Delta x} = \frac{(x^2 + 2x\Delta x + (\Delta x)^2) - x^2}{\Delta x}
$$
$$
\frac{\Delta f}{\Delta x} = \frac{2x\Delta x + (\Delta x)^2}{\Delta x}
$$
$$
\frac{\Delta f}{\Delta x} = \frac{\Delta x (2x + \Delta x)}{\Delta x}
$$
$$
\frac{\Delta f}{\Delta x} = 2x + \Delta x
$$

$$
\frac{df}{dx} = \lim_{\Delta x \to 0} (2x + \Delta x) = 2x + 0 = 2x
$$

* **Result**: If $f(x) = x^2$, then its derivative is $f'(x) = 2x$. This matches our observation that at $x=1$, the derivative was $2(1)=2$.

## Derivative of Cubic Functions: $f(x) = x^3$

* The function $f(x) = x^3$ represents a cubic curve.
* The derivative for $f(x) = x^3$ is given by the limit of $\frac{\Delta f}{\Delta x}$ as $\Delta x$ approaches 0, where $\Delta f = (x + \Delta x)^3 - x^3$.

### Example: Finding the Derivative at a Specific Point ($x=0.5$)

Let's estimate the slope of the tangent line at $x=0.5$ (where $f(0.5) = (0.5)^3 = 0.125 = \frac{1}{8}$). We'll calculate the slopes of secant lines with decreasing $\Delta x$ values.

1.  **$\Delta x = 1$**:
    * Points: $(0.5, 0.125)$ and $(0.5+1, (0.5+1)^3) = (1.5, (1.5)^3) = (1.5, 3.375)$.
    * $\Delta f = 3.375 - 0.125 = 3.25$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{3.25}{1} = 3.25$.

2.  **$\Delta x = 0.5$ (or $\frac{1}{2}$)**:
    * Points: $(0.5, 0.125)$ and $(0.5+0.5, (0.5+0.5)^3) = (1, 1^3) = (1, 1)$.
    * $\Delta f = 1 - 0.125 = 0.875$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{0.875}{0.5} = 1.75$.

3.  **$\Delta x = 0.25$ (or $\frac{1}{4}$)**:
    * $\Delta f = (0.5 + 0.25)^3 - (0.5)^3 = (0.75)^3 - 0.125 = 0.421875 - 0.125 = 0.296875$.
    * Slope = $\frac{0.296875}{0.25} = 1.1875$.

4.  **Continuing this process**:
    * For $\Delta x = \frac{1}{8}$, the slope is approximately $0.95$.
    * For $\Delta x = \frac{1}{16}$, the slope is approximately $0.85$.
    * For $\Delta x = \frac{1}{1000}$, the slope is approximately $0.752$.

* **Observation**: As $\Delta x$ gets smaller, the slope of the secant line tends to converge to **$0.75$**.
* **Conclusion**: The slope of the tangent line to $f(x) = x^3$ at $x=0.5$ is $0.75$. This value is equal to $3 \times (0.5)^2 = 3 \times 0.25 = 0.75$. This suggests the general derivative formula for $x^3$.

### Formal Calculation of the Derivative for $f(x) = x^3$

We want to find $\frac{df}{dx}$ by evaluating the limit of $\frac{\Delta f}{\Delta x}$ as $\Delta x \to 0$.

$$
\frac{\Delta f}{\Delta x} = \frac{f(x + \Delta x) - f(x)}{\Delta x} = \frac{(x + \Delta x)^3 - x^3}{\Delta x}
$$
$$
\frac{\Delta f}{\Delta x} = \frac{(x^3 + 3x^2\Delta x + 3x(\Delta x)^2 + (\Delta x)^3) - x^3}{\Delta x}
$$
$$
\frac{\Delta f}{\Delta x} = \frac{3x^2\Delta x + 3x(\Delta x)^2 + (\Delta x)^3}{\Delta x}
$$
$$
\frac{\Delta f}{\Delta x} = \frac{\Delta x (3x^2 + 3x\Delta x + (\Delta x)^2)}{\Delta x}
$$
$$
\frac{\Delta f}{\Delta x} = 3x^2 + 3x\Delta x + (\Delta x)^2
$$
$$
\frac{df}{dx} = \lim_{\Delta x \to 0} (3x^2 + 3x\Delta x + (\Delta x)^2) = 3x^2 + 3x(0) + (0)^2 = 3x^2
$$

* **Result**: If $f(x) = x^3$, then its derivative is $f'(x) = 3x^2$. This is consistent with our earlier observation at $x=0.5$.

## Derivative of Reciprocal Function: $f(x) = \frac{1}{x}$

The derivative is found by evaluating the limit of $\frac{\Delta f}{\Delta x}$ as $\Delta x$ approaches 0, where $\Delta f = (x + \Delta x)^{-1} - x^{-1}$.

### Example: Finding the Derivative at a Specific Point ($x=1$)

Let's estimate the slope of the tangent line at $x=1$ (where $f(1) = 1^{-1} = 1$).

1.  **$\Delta x = 1$**:
    * Points: $(1, 1)$ and $(1+1, (1+1)^{-1}) = (2, \frac{1}{2})$.
    * $\Delta f = \frac{1}{2} - 1 = -0.5$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{-0.5}{1} = -0.5$.

2.  **$\Delta x = 0.5$ (or $\frac{1}{2}$)**:
    * Points: $(1, 1)$ and $(1+0.5, (1+0.5)^{-1}) = (1.5, \frac{1}{1.5}) = (1.5, \frac{2}{3})$.
    * $\Delta f = \frac{2}{3} - 1 = -\frac{1}{3} \approx -0.333$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{-1/3}{1/2} = -\frac{2}{3} \approx -0.667$.

3.  **Continuing this process with smaller $\Delta x$**:
    * For $\Delta x = \frac{1}{4}$, the slope is approximately $-0.8$.
    * For $\Delta x = \frac{1}{8}$, the slope is approximately $-0.89$.
    * For $\Delta x = \frac{1}{16}$, the slope is approximately $-0.94$.
    * For $\Delta x = \frac{1}{1000}$, the slope is approximately $-0.999$.

* **Observation**: As $\Delta x$ gets smaller, the slope of the secant line approaches **$-1$**.
* **Conclusion**: The slope of the tangent line to $f(x) = x^{-1}$ at $x=1$ is $-1$. This value is equal to $-1 \times 1^{-2}$. This suggests the general formula.

### Formal Calculation of the Derivative for $f(x) = x^{-1}$

We want to find $\frac{df}{dx}$ by evaluating the limit of $\frac{\Delta f}{\Delta x}$ as $\Delta x \to 0$.

$$
\frac{\Delta f}{\Delta x} = \frac{(x + \Delta x)^{-1} - x^{-1}}{\Delta x}
$$

$$
\frac{\Delta f}{\Delta x} = \frac{\frac{1}{x + \Delta x} - \frac{1}{x}}{\Delta x} = \frac{\frac{x - (x + \Delta x)}{x(x + \Delta x)}}{\Delta x}
$$

$$
\frac{\Delta f}{\Delta x} = \frac{\frac{x - x - \Delta x}{x(x + \Delta x)}}{\Delta x} = \frac{-\Delta x}{x(x + \Delta x) \cdot \Delta x}
$$

$$
\frac{\Delta f}{\Delta x} = \frac{-1}{x(x + \Delta x)}
$$

$$
\frac{df}{dx} = \lim_{\Delta x \to 0} \left( \frac{-1}{x(x + \Delta x)} \right) = \frac{-1}{x(x + 0)} = \frac{-1}{x^2}
$$

* **Result**: If $f(x) = x^{-1}$, then its derivative is $f'(x) = -1/x^{-2}$.

## The Power Rule

Let's review the derivatives we've found for **power functions**:

* If $f(x) = x^2$, then $f'(x) = 2x^1$
* If $f(x) = x^3$, then $f'(x) = 3x^2$
* If $f(x) = x^{-1}$, then $f'(x) = -1x^{-2}$

**Pattern Observation**:
For a function of the form $f(x) = x^n$:
1.  The **exponent $n$** comes down as a multiplicative factor in front of $x$.
2.  The new exponent of $x$ is **$n-1$**.

This leads to the general **Power Rule for Derivatives**:
If $f(x) = x^n$, then $f'(x) = nx^{n-1}$.

### Examples of Power Rule Application:

* If $f(x) = x^{100}$, then $f'(x) = 100x^{100-1} = 100x^{99}$.
* If $f(x) = x^{-100}$, then $f'(x) = -100x^{-100-1} = -100x^{-101}$.

This rule is fundamental for differentiating any function that is a power of $x$.

## Inverse Functions: The "Undo" Operation

* An **inverse function** "undoes" what the original function does.
* **Analogy**: If a function $f$ puts a hat on a person, its inverse function $g$ takes the hat off.
* **Mathematical Example**:
    * If $f$ squares a number ($f(x) = x^2$), then its inverse $g$ must take the square root ($g(x) = \sqrt{x}$).
    * If you apply $f$ to $x$ and then apply $g$ to the result, you get back the original $x$: $g(f(x)) = x$. Similarly, $f(g(x)) = x$.
* **Notation**: If $f$ and $g$ are inverse functions, we write $g(x) = f^{-1}(x)$.
    * **Important**: $f^{-1}(x)$ **does not mean** $\frac{1}{f(x)}$. It's purely a notation for the inverse function.
* **Example for $f(x)=x^2$**:
    * If $f(x) = x^2$, then $f^{-1}(x) = \sqrt{x}$.
    * This holds for $x \ge 0$ (we consider only the positive square root to maintain a function).

## Graphical Relationship of Inverse Functions

* If a point $(a, b)$ lies on the graph of $f(x)$, then the point $(b, a)$ lies on the graph of its inverse $g(x) = f^{-1}(x)$.
* This means the graph of $g(x)$ is a **reflection of the graph of $f(x)$ across the line $y=x$**.

## Derivative of Inverse Functions

There is a powerful relationship between the derivative of a function and the derivative of its inverse.

* Consider a point $(x, f(x))$ on the graph of $f$.
* The slope of the tangent at this point is $f'(x) = \frac{df}{dx}$.
* On the inverse function's graph, the corresponding point is $(f(x), x)$. Let's denote $y = f(x)$. So the point is $(y, g(y))$.
* The slope of the tangent at this point on the inverse function's graph is $g'(y) = \frac{dg}{dy}$.

* **Derivation (Intuitive)**:
    * On the graph of $f(x)$, a small change $\Delta x$ horizontally corresponds to a small change $\Delta f$ vertically. The slope is $\frac{\Delta f}{\Delta x}$.
    * On the graph of the inverse $g(y)$, the roles are swapped: a small change $\Delta y$ horizontally (which was $\Delta f$ in the original function) corresponds to a small change $\Delta g$ vertically (which was $\Delta x$ in the original function). The slope is $\frac{\Delta g}{\Delta y}$.
    * Since $\Delta g$ corresponds to $\Delta x$ and $\Delta y$ corresponds to $\Delta f$:

$$
\frac{\Delta g}{\Delta y} = \frac{\Delta x}{\Delta f}
$$

As $\Delta x$ (and thus $\Delta y$) approaches 0, we get:

$$
\frac{dg}{dy} = \frac{dx}{df} = \frac{1}{\frac{df}{dx}}
$$

* **Inverse Function Theorem for Derivatives**:
    If $f$ and $g$ are inverse functions ($g = f^{-1}$), then the derivative of $g$ at a point $y$ is the reciprocal of the derivative of $f$ at the corresponding point $x$ (where $y = f(x)$). Expressed formally:

$$
g'(y) = \frac{1}{f'(x)} \quad \text{or} \quad \frac{dg}{dy} = \frac{1}{\frac{df}{dx}}
$$

### Examples with $f(x) = x^2$ and $g(y) = \sqrt{y}$

1.  **At the point (1,1)**:
    * For $f(x) = x^2$, $f'(x) = 2x$. So, $f'(1) = 2(1) = 2$.
    * The slope of the tangent to $f(x)$ at $(1,1)$ is 2.
    * For $g(y) = \sqrt{y}$, using the inverse derivative rule:
        * $g'(1) = \frac{1}{f'(1)} = \frac{1}{2}$.
    * The slope of the tangent to $g(y)$ at $(1,1)$ is $\frac{1}{2}$. This makes intuitive sense as the graphs are reflections.

2.  **At the point (2,4) for $f(x)$ and (4,2) for $g(y)$**:
    * For $f(x) = x^2$, $f'(2) = 2(2) = 4$.
    * The slope of the tangent to $f(x)$ at $(2,4)$ is 4.
    * For $g(y) = \sqrt{y}$, using the inverse derivative rule:
        * $g'(4) = \frac{1}{f'(2)} = \frac{1}{4}$.
    * The slope of the tangent to $g(y)$ at $(4,2)$ is $\frac{1}{4}$.

This relationship simplifies finding derivatives of inverse functions if the derivative of the original function is known.

Let's delve into the derivatives of **trigonometric functions**, starting with sine and cosine.

## Derivatives of Trigonometric Functions

### Derivative of Sine Function

* Consider the function $f(x) = \sin(x)$.
* Let's look at the slopes of the tangent lines at a few specific points on the sine curve:
    * At $x = \frac{\pi}{2}$ (peak): The tangent is horizontal, so its slope is **0**.
    * At $x = -\frac{\pi}{2}$ (trough): The tangent is horizontal, so its slope is **0**.
    * At $x = 0$: The tangent has a positive slope, which is **1**.
    * At $x = -\pi$: The tangent has a negative slope, which is **-1**.
    

* Now, let's compare these slopes with the values of the **cosine function** at the same points:
    * $\cos(\frac{\pi}{2}) = 0$
    * $\cos(-\frac{\pi}{2}) = 0$
    * $\cos(0) = 1$
    * $\cos(-\pi) = -1$

* **Observation**: The values of $\cos(x)$ perfectly match the slopes of $\sin(x)$ at these points.
* **Conclusion**: The derivative of $\sin(x)$ is $\cos(x)$.
    * If $f(x) = \sin(x)$, then $f'(x) = \cos(x)$ or $\frac{d}{dx}(\sin(x)) = \cos(x)$.

### Derivative of Cosine Function

* Now consider the function $f(x) = \cos(x)$.
* Let's look at the slopes of the tangent lines at some specific points on the cosine curve:
    * At $x = 0$ (peak): The tangent is horizontal, so its slope is **0**.
    * At $x = -\pi$ (trough): The tangent is horizontal, so its slope is **0**.
    * At $x = \frac{\pi}{2}$: The tangent has a negative slope, which is **-1**.
    * At $x = -\frac{\pi}{2}$: The tangent has a positive slope, which is **1**.
    
* Let's compare these slopes with the values of the **sine function** at the same points, but with a negative sign:
    * $-\sin(0) = 0$
    * $-\sin(-\pi) = 0$
    * $-\sin(\frac{\pi}{2}) = -1$
    * $-\sin(-\frac{\pi}{2}) = -(-1) = 1$

* **Observation**: The values of $-\sin(x)$ perfectly match the slopes of $\cos(x)$ at these points.
* **Conclusion**: The derivative of $\cos(x)$ is $-\sin(x)$.
    * If $f(x) = \cos(x)$, then $f'(x) = -\sin(x)$ or $\frac{d}{dx}(\cos(x)) = -\sin(x)$.

## Euler's Number (e)

**Euler's number (e)** is a fundamental mathematical constant, approximately **2.718281828**. It's an **irrational number**, meaning its decimal representation is non-repeating and non-terminating.

**Definition using a Limit Expression:**
One key way to define $e$ is through the limit of the expression $(1 + \frac{1}{n})^n$ as $n$ approaches infinity.
* For $n=1: (1 + \frac{1}{1})^1 = 2^1 = 2$
* For $n=10: (1 + \frac{1}{10})^{10} \approx 2.594$
* For $n=100: (1 + \frac{1}{100})^{100} \approx 2.705$
* For $n=1000: (1 + \frac{1}{1000})^{1000} \approx 2.717$
As $n$ gets larger, the value of $(1 + \frac{1}{n})^n$ converges to $e$.

## The Exponential Function $f(x) = e^x$

The number $e$ is particularly special because the **exponential function $f(x) = e^x$ is its own derivative**.
* If $f(x) = e^x$, then $f'(x) = e^x$ (or $\frac{d}{dx}(e^x) = e^x$).
This unique property makes $e^x$ appear extensively in various fields like science, statistics, and probability.

## Understanding 'e' through Compound Interest

A practical way to understand $e$ is through the concept of **compound interest**.
Imagine you invest $1 and aim to earn 100% annual interest.

* **Bank 1: 100% interest once a year** ($n=1$)
    * After 1 year: $1 + 100\%$ of $1 = 1 + 1 = 2$
    * This can be represented as $(1 + \frac{1}{1})^1 = 2$.

* **Bank 2: 50% interest twice a year** ($n=2$)
    * After 6 months: $1 + 50\%$ of $1 = 1 + 0.5 = 1.5$
    * After 1 year: $1.5 + 50\%$ of $1.5 = 1.5 + 0.75 = 2.25$
    * This can be represented as $(1 + \frac{1}{2})^2 = (1.5)^2 = 2.25$.
    * Here, the interest earned in the first 6 months also starts earning interest, which is called **accrued interest**. This is why Bank 2 yields more than Bank 1.

* **Bank 3: 33.3% interest three times a year** ($n=3$)
    * After 4 months: $1 + \frac{1}{3} = 1.333...$
    * After 8 months: $(1 + \frac{1}{3})^2 \approx 1.777...$
    * After 1 year: $(1 + \frac{1}{3})^3 \approx 2.370$
    * This shows that compounding more frequently (Bank 3 vs. Bank 2) yields even more money.

**Generalizing Compound Interest:**
If a bank offers 100% annual interest, compounded $n$ times a year, the total amount after one year (starting with 1 dollar) will be $$(1 + \frac{1}{n})^n$$.

* **Bank 12:** Compounded monthly ($n=12$)
    * Amount after 1 year: $(1 + \frac{1}{12})^{12} \approx 2.613$

* **Bank 365:** Compounded daily ($n=365$)
    * Amount after 1 year: $(1 + \frac{1}{365})^{365} \approx 2.7145$

* **Bank Infinity (Continuous Compounding):**
    Imagine a bank that compounds interest infinitely many times per year (e.g., every second, every millisecond, or continuously). This means $n \to \infty$. The amount of money you would have at the end of the year from this "Bank Infinity" is precisely the value of $e$.

$$
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e \approx 2.718281828
$$

Therefore, $e$ represents the maximum possible return on a 100% annual interest rate when compounded continuously. It signifies constant, continuous growth.


## Derivative of the Exponential Function: $f(x) = e^x$

The most fascinating property of the exponential function $f(x) = e^x$ is that its **derivative is the function itself**.

* **Property**: If $f(x) = e^x$, then $f'(x) = e^x$.
* In Leibniz's notation: $\frac{d}{dx}(e^x) = e^x$.
* This means that at any point $(x, e^x)$ on the graph of $y = e^x$, the **slope of the tangent line** at that point is equal to the function's value $e^x$ at that point.

### Numerical Verification at $x=2$

Let's numerically verify this property by calculating the slope of secant lines for $f(x) = e^x$ at $x=2$. The point on the graph is $(2, e^2 \approx 7.39)$. We expect the slope of the tangent at this point to also be approximately $7.39$.

1.  **$\Delta x = 1$**:
    * Points: $(2, e^2)$ and $(2+1, e^{2+1}) = (3, e^3)$.
    * $\Delta f = e^3 - e^2 \approx 20.09 - 7.39 = 12.7$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{12.7}{1} = 12.7$.

2.  **$\Delta x = 0.5$ (or $\frac{1}{2}$)**:
    * Points: $(2, e^2)$ and $(2+0.5, e^{2+0.5}) = (2.5, e^{2.5})$.
    * $\Delta f = e^{2.5} - e^2 \approx 12.18 - 7.39 = 4.79$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{4.79}{0.5} = 9.58$.

3.  **$\Delta x = 0.25$ (or $\frac{1}{4}$)**:
    * Points: $(2, e^2)$ and $(2+0.25, e^{2.25})$.
    * $\Delta f = e^{2.25} - e^2 \approx 9.49 - 7.39 = 2.1$.
    * Slope = $\frac{\Delta f}{\Delta x} = \frac{2.1}{0.25} = 8.4$.

4.  **Continuing with smaller $\Delta x$ values**:
    * For $\Delta x = \frac{1}{8}$, the slope is approximately $7.87$.
    * For $\Delta x = \frac{1}{16}$, the slope is approximately $7.62$.
    * For $\Delta x = \frac{1}{1000}$, the slope is approximately $7.39$.

* **Observation**: As $\Delta x$ approaches 0, the slope of the secant lines converges to approximately **7.39**.
* **Conclusion**: This numerical evidence strongly suggests that the slope of the tangent line to $e^x$ at $x=2$ is indeed $e^2 \approx 7.39$. This illustrates that for any point $x$, the slope of the tangent to $e^x$ is $e^x$ itself. 

This self-replicating property in its derivative is what makes $e^x$ so fundamental in modeling continuous growth and decay processes in nature and science.

## The Natural Logarithmic Function

* **Definition**: The natural logarithm of a number $x$, denoted as $\ln(x)$ (or $\log(x)$ in some contexts, but specifically base $e$ in this course), is the power to which $e$ must be raised to equal $x$.
    * If $e^k = x$, then $k = \ln(x)$.
    * **Example**: If $e^k = 3$, then $k = \ln(3)$.

* **Inverse Relationship**: The natural logarithm function is the **inverse** of the exponential function $e^x$.
    * If $f(x) = e^x$, then its inverse function is $f^{-1}(y) = \ln(y)$.
    * This inverse relationship means:
        * $e^{\ln(x)} = x$
        * $\ln(e^y) = y$
    

## Derivative of the Natural Logarithm Function

We will use the **inverse function derivative rule**: If $g(y) = f^{-1}(y)$, then $g'(y) = \frac{1}{f'(x)}$, where $y = f(x)$.

1.  **Identify the functions**:
    * Let $f(x) = e^x$.
    * Its derivative is $f'(x) = e^x$.
    * Its inverse is $g(y) = \ln(y)$. This is the derivative we want to find.

2.  **Apply the inverse function rule**:
    * $g'(y) = \frac{1}{f'(x)}$
    * We know $f'(x) = e^x$. So, $g'(y) = \frac{1}{e^x}$.

3.  **Express in terms of $y$**:
    * Since $y = f(x) = e^x$, we can substitute $y$ for $e^x$ in the denominator.
    * Therefore, $g'(y) = \frac{1}{y}$.

* **Conclusion**: The derivative of the natural logarithm function is $\frac{1}{y}$ (or $\frac{1}{x}$ if using $x$ as the independent variable).
    * If $f(x) = \ln(x)$, then $f'(x) = \frac{1}{x}$.
    * In Leibniz's notation: $\frac{d}{dx}(\ln(x)) = \frac{1}{x}$.

### Example Verification at $x=2$

* Consider the point $(2, e^2 \approx 7.39)$ on the graph of $f(x) = e^x$.
    * The slope of the tangent at this point is $f'(2) = e^2 \approx 7.39$.
* The corresponding point on the graph of $g(y) = \ln(y)$ is $(e^2, 2)$, or approximately $(7.39, 2)$.
    * Using the inverse derivative rule, the slope of the tangent at $(7.39, 2)$ should be $\frac{1}{\text{slope of } e^x \text{ at } x=2} = \frac{1}{e^2}$.
    * According to our derived formula, $g'(y) = \frac{1}{y}$. So, at $y = e^2$, $g'(e^2) = \frac{1}{e^2}$.
* This confirms that the derivative of $\ln(y)$ is $\frac{1}{y}$.

## Visually Identifying Non-Differentiable Functions

There are three primary visual indicators that a function is not differentiable at a specific point:

### Corners or Cusps

  * A function is **not differentiable** at a point where its graph has a sharp corner or a cusp.
  * **Reason**: At a corner or cusp, you cannot draw a unique tangent line. Multiple lines could appear to "touch" the curve at that point.
  * **Example**: The **absolute value function**, $f(x) = |x|$.
      * This function is defined as $x$ for $x \\ge 0$ and $-x$ for $x \< 0$.
      * At $x=0$, the graph forms a sharp corner. If you try to draw a tangent, it's not well-defined.
      * Therefore, $f(x) = |x|$ is **not differentiable at $x=0$**.

### Jump Discontinuities
  * A function is **not differentiable** at any point where it has a jump discontinuity.
  * **Reason**: If a function is discontinuous (you have to lift your pencil to draw it), you cannot draw any tangent line at the point of the jump. A function must be **continuous** at a point to be differentiable at that point.
  * **Example**: A **step function** or a **piecewise function** with a jump.
      * Consider a function defined as $f(x) = 2$ for $x \< -1$ and $f(x) = x+1$ for $x \\ge -1$.
      * At $x=-1$, there is a sudden "jump" in the function's value.
      * Therefore, this function is **not differentiable at $x=-1$**.

### Vertical Tangents
  * A function is **not differentiable** at a point where its tangent line is vertical.
  * **Reason**: The slope of a vertical line is undefined (it's "rise over run" where run is zero, leading to division by zero, or an "infinite" slope). A well-defined derivative requires a finite slope.
  * **Example**: The cubic root function, $f(x) = x^{1/3}$ (or $\\sqrt[3]{x}$).
      * At $x=0$, the graph has a tangent line that is perfectly vertical (along the y-axis).
      * Therefore, $f(x) = x^{1/3}$ is **not differentiable at $x=0$**.

## Summary of Non-Differentiable Cases

A function is **not differentiable** at points exhibiting any of these characteristics:

  * **Corners or Cusps** (sharp points)
  * **Jump Discontinuities** (breaks in the graph)
  * **Vertical Tangents** (where the slope is undefined)

If a function is differentiable over an entire interval, it means that the derivative exists for every single point in that interval, and none of these conditions are met.

## Derivatives: Essential Rules for Complex Functions

To differentiate complex functions, we build upon simple derivative rules using several key rules: **multiplication by a scalar**, **sum rule**, **product rule**, and **chain rule**.

### Multiplication by a Scalar Rule

If a function $f(x)$ is a constant $c$ multiplied by another function $g(x)$, i.e., $f(x) = c \cdot g(x)$, then its derivative $f'(x)$ is $c$ times the derivative of $g(x)$:

$$\frac{d}{dx}[c \cdot g(x)] = c \cdot \frac{d}{dx}[g(x)]$$

**Intuition:**

* Consider a function $y = x^2$. Its derivative is $2x$.
* Now consider $y = 2x^2$. This function is essentially the original $x^2$ function, but every y-value (and thus the graph) is stretched vertically by a factor of 2.
* When a function is stretched vertically by a factor of $c$, the "rise" component of any secant line (and eventually the tangent line) is also multiplied by $c$, while the "run" component remains the same.
* Since slope = rise/run, multiplying the rise by $c$ means the overall slope is also multiplied by $c$.
* Therefore, the derivative, which represents the slope of the tangent line, also gets multiplied by $c$.

**Example:**

If $f(x) = 4g(x)$, then $f'(x) = 4g'(x)$.

## Derivatives: Sum Rule

The **sum rule** in derivatives states that if a function $f(x)$ is the sum of two (or more) other functions, say $g(x)$ and $h(x)$, then the derivative of $f(x)$ is simply the sum of the derivatives of $g(x)$ and $h(x)$.

If $f(x) = g(x) + h(x)$, then:  
$$f'(x) = g'(x) + h'(x)$$

This rule can be extended to any number of functions:  
$$\frac{d}{dx}[f_1(x) + f_2(x) + \dots + f_n(x)] = \frac{d}{dx}[f_1(x)] + \frac{d}{dx}[f_2(x)] + \dots + \frac{d}{dx}[f_n(x)]$$

### Intuition: The Boat and Child Analogy

Imagine a child running inside a moving boat.
* Let $X_B$ be the distance the **boat** moves.
* Let $X_C$ be the distance the **child moves relative to the boat**.
* The **total distance** the child moves with respect to the earth, $X_{total}$, is $X_B + X_C$.

Now, let's consider **velocities** (which are derivatives of distance with respect to time):
* **Speed of the boat** ($V_B$) is the derivative of $X_B$ with respect to time.
* **Speed of the child relative to the boat** ($V_C$) is the derivative of $X_C$ with respect to time.
* The **total speed** of the child with respect to the earth ($V_{total}$) is the derivative of $X_{total}$ with respect to time.

Since distances add up ($X_{total} = X_B + X_C$), their rates of change (velocities) also add up: $V_{total} = V_B + V_C$.

This analogy highlights that if individual components of a quantity add up, then their rates of change (derivatives) also add up.

### Formal Explanation

Consider two functions, $f_1(x)$ and $f_2(x)$. Let their sum be $f(x) = f_1(x) + f_2(x)$.
The derivative of a function at a point represents the slope of its tangent line at that point. If we consider a small change $\Delta x$:

The change in $f(x)$ is $\Delta f = f(x + \Delta x) - f(x)$.  
Since $f(x) = f_1(x) + f_2(x)$, we can write:  
$\Delta f = (f_1(x + \Delta x) + f_2(x + \Delta x)) - (f_1(x) + f_2(x))$  
$\Delta f = (f_1(x + \Delta x) - f_1(x)) + (f_2(x + \Delta x) - f_2(x))$  
$\Delta f = \Delta f_1 + \Delta f_2$

Dividing by $\Delta x$:
$$\frac{\Delta f}{\Delta x} = \frac{\Delta f_1}{\Delta x} + \frac{\Delta f_2}{\Delta x}$$

Taking the limit as $\Delta x \to 0$:
$$\lim_{\Delta x \to 0} \frac{\Delta f}{\Delta x} = \lim_{\Delta x \to 0} \frac{\Delta f_1}{\Delta x} + \lim_{\Delta x \to 0} \frac{\Delta f_2}{\Delta x}$$
$$f'(x) = f_1'(x) + f_2'(x)$$

This demonstrates that the **slope of the sum of functions is the sum of their individual slopes**.

## Derivatives: Product Rule

The **product rule** is used when you need to find the derivative of a function that is the product of two or more other functions. If a function $f(x)$ is the product of two functions, $g(x)$ and $h(x)$, i.e., $f(x) = g(x) \cdot h(x)$, then its derivative $f'(x)$ is given by:

$$f'(x) = g'(x)h(x) + g(x)h'(x)$$

In simpler terms, it's the derivative of the first function times the second function (left alone) plus the first function (left alone) times the derivative of the second function.

### Intuition: Building a House Analogy

Imagine building a house with two walls, a side wall and a front wall, whose lengths are changing with time.
* Let $g(t)$ be the length of the **side wall** as a function of time.
* Let $h(t)$ be the length of the **front wall** as a function of time.
* The **area** of the house at any given time, $f(t)$, is the product of these lengths: $f(t) = g(t) \cdot h(t)$.

We want to find the **rate of change of the area** with respect to time, which is $f'(t)$.


Consider a small change in time, $\Delta t$. This leads to small changes in the lengths of the walls, $\Delta g$ and $\Delta h$.
The **change in area** ($\Delta f$) can be visualized as three new rectangles that are added to the original area $g(t)h(t)$:
1. A rectangle with dimensions $\Delta g \cdot h(t)$.
2. A rectangle with dimensions $g(t) \cdot \Delta h$.
3. A tiny rectangle with dimensions $\Delta g \cdot \Delta h$.

So, $\Delta f(t) = \Delta g(t)h(t) + g(t)\Delta h(t) + \Delta g(t)\Delta h(t)$.

To find the derivative, we divide by $\Delta t$ and take the limit as $\Delta t \to 0$:

$$\frac{\Delta f(t)}{\Delta t} = \frac{\Delta g(t)h(t)}{\Delta t} + \frac{g(t)\Delta h(t)}{\Delta t} + \frac{\Delta g(t)\Delta h(t)}{\Delta t}$$

As $\Delta t \to 0$:
* $\frac{\Delta g(t)}{\Delta t} \to g'(t)$
* $\frac{\Delta h(t)}{\Delta t} \to h'(t)$
* $\frac{\Delta g(t)\Delta h(t)}{\Delta t} \to 0$ (This term approaches zero much faster than the others because it involves the product of two small changes, $\Delta g$ and $\Delta h$, divided by only one $\Delta t$. As $\Delta t$ approaches zero, $\Delta g$ and $\Delta h$ also approach zero, making the numerator infinitesimally small compared to the denominator.)

Thus, the product rule emerges:

$$f'(t) = g'(t)h(t) + g(t)h'(t)$$

## Derivatives: The Chain Rule

The **chain rule** is a fundamental rule in calculus for differentiating composite functions. A **composite function** is a function within a function. If you have a function $y = f(u)$ where $u$ itself is a function of $x$, say $u = g(x)$, then $y$ is a composite function of $x$, i.e., $y = f(g(x))$. The chain rule provides a way to find the derivative of $y$ with respect to $x$.

### Leibniz Notation

In Leibniz notation, the chain rule is intuitive:

If $y = f(u)$ and $u = g(x)$, then the derivative of $y$ with respect to $x$ is:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

This can be extended for more layers of composition. For example, if $y = f(g(h(t)))$:
$$\frac{dy}{dt} = \frac{dy}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dt}$$

The notation itself suggests a "chain" of multiplications, where intermediate variables "cancel out" (conceptually).

### Lagrange (Prime) Notation

While Leibniz notation is clear, the Lagrange notation requires careful attention to the input of the functions:

If $f(t) = g(h(t))$, then:
$$f'(t) = g'(h(t)) \cdot h'(t)$$
Here, $g'(h(t))$ means the derivative of $g$ evaluated at $h(t)$, not just $g'(t)$.

For a three-level composition, $f(t) = f_1(f_2(f_3(t)))$:
$$f'(t) = f_1'(f_2(f_3(t))) \cdot f_2'(f_3(t)) \cdot f_3'(t)$$

### Intuition: The Mountain Drive Analogy

Imagine you are driving up a mountain.
* **Temperature (T)** changes with **height (h)**: This rate of change is $\frac{dT}{dh}$. (e.g., as you go higher, it gets colder).
* **Height (h)** changes with **time (t)**: This rate of change is $\frac{dh}{dt}$. (e.g., as time passes, you drive higher up the mountain).
* You want to find how **temperature (T)** changes with **time (t)**: This is $\frac{dT}{dt}$.

The chain rule states that if you know how T changes with h, and how h changes with t, you can find how T changes with t by multiplying these rates:

$$\frac{dT}{dt} = \frac{dT}{dh} \cdot \frac{dh}{dt}$$

### Deeper Understanding with Small Changes

Consider infinitesimal changes:
* A small change in time, $\Delta t$.
* This causes a small change in height, $\Delta h$.
* This change in height then causes a small change in temperature, $\Delta T$.

We can write the relationship between these small changes:  
$$\frac{\Delta T}{\Delta t} = \frac{\Delta T}{\Delta h} \cdot \frac{\Delta h}{\Delta t}$$

As $\Delta t \to 0$, then $\Delta h \to 0$ and $\Delta T \to 0$. In this limit, these ratios become derivatives:  
$$\lim_{\Delta t \to 0} \frac{\Delta T}{\Delta t} = \left(\lim_{\Delta h \to 0} \frac{\Delta T}{\Delta h}\right) \cdot \left(\lim_{\Delta t \to 0} \frac{\Delta h}{\Delta t}\right)$$  
$$\frac{dT}{dt} = \frac{dT}{dh} \cdot \frac{dh}{dt}$$

This illustrates how the chain rule "links" the rates of change through an intermediate variable.

## Derivatives in Machine Learning: Optimization

Derivatives are crucial in machine learning primarily for **optimization**, which involves finding the maximum or minimum value of a function.

### Optimization in Machine Learning

* In machine learning, the goal is to find the **best model** that fits a given dataset.
* This is achieved by minimizing an **error function** (also known as a cost or loss function) that quantifies how far the current model is from an ideal model.
* Minimizing the error function leads to the optimal model.

### Intuition: Finding the Coldest Spot in a Sauna

Consider a scenario where you're trying to find the coldest spot on a sauna bench using a thermometer:

* **Initial Movement:** If moving left makes it hotter, you'd move right. If moving right makes it colder, you'd continue in that direction.
* **Reaching the Coldest Spot:** You've found the coldest spot when any movement (left or right) results in a higher temperature.

### Derivatives and Slopes

* **Decreasing Temperature:** When moving towards a colder spot (decreasing function value), the **slope of the tangent** at that point is **negative** ($<0$).
* **Increasing Temperature:** When moving towards a hotter spot (increasing function value), the **slope of the tangent** at that point is **positive** ($>0$).
* **Minimum/Maximum Point:** At the coldest (minimum) or hottest (maximum) spot, the **slope of the tangent is zero** ($=0$).

### Local vs. Global Minima/Maxima

* A point where the derivative is zero is a **candidate** for a maximum or minimum.
* These candidates are called **local minima** or **local maxima**.
* The **absolute minimum** or **absolute maximum** among all local minima/maxima is called the **global minimum** or **global maximum**.
* In more complex functions, there might be multiple points where the derivative is zero, requiring evaluation of each candidate to find the global optimum.

### Key Takeaway for Differentiable Functions

For a function that is differentiable at every point, the candidates for maximum and minimum values are those points where the **derivative is zero**.

## Optimizing House Location for Power Line Connection

This example demonstrates optimization in a more complex scenario, which is crucial for understanding machine learning concepts like **squared error** in **linear regression** and **neural networks**.

### The Problem

Imagine you need to build a house and connect it to several existing **power lines**. The goal is to find the optimal location for the house that **minimizes the total cost of connection** to all power lines.

* **Cost Factor:** The cost of connecting to a power line is proportional to the **square of the distance** ($X^2$) between the house and that power line. This means longer cables are more expensive.

### Simple Case: One Power Line

If there's only **one power line**, the optimal location for the house to minimize connection cost is **directly on top of the power line**.

* **Distance:** If the house is at distance $X = 0$ from the power line.
* **Cost:** The cost of connection would be $0^2 = 0$, which is the minimum possible cost.

Here's a breakdown of optimizing house placement for power lines:

## Optimizing House Location with Two Power Lines

Let's consider two power lines, a blue one located at position $a$ from the origin and an orange one at position $b$ from the origin. Your house is to be built at position $x$.

### Defining the Cost Function

* **Distance to blue power line:** $|x - a|$
* **Distance to orange power line:** $|x - b|$

The cost of connecting to each power line is the square of the distance.
* **Cost for blue power line:** $(x - a)^2$
* **Cost for orange power line:** $(x - b)^2$ (Note: $(b - x)^2 = (x - b)^2$, so the order doesn't matter when squaring.)

The **total cost function** $C(x)$ is the sum of these individual costs:
$$C(x) = (x - a)^2 + (x - b)^2$$

### Visualizing the Cost

The cost can be visualized as the sum of the areas of two squares, where the side lengths of the squares are the distances to the power lines. 
Moving the house (changing $x$) alters the sizes of these squares, and thus the total area (total cost). Intuitively, placing the house in the middle seems to minimize this combined area.

### Minimizing the Cost Function Using Derivatives

The cost function $C(x)$ is a **quadratic function** that opens upwards (because the coefficient of $x^2$ is positive, specifically $2x^2$ when expanded). This means its minimum value occurs at the point where its derivative is zero.

1.  **Calculate the derivative of the cost function with respect to $x$:**

$$
\frac{dC}{dx} = \frac{d}{dx}((x - a)^2 + (x - b)^2)
$$
Using the chain rule:
$$
\frac{d}{dx}(x - a)^2 = 2(x - a) \cdot \frac{d}{dx}(x - a) = 2(x - a) \cdot 1 = 2(x - a)
$$   $$
\frac{d}{dx}(x - b)^2 = 2(x - b) \cdot \frac{d}{dx}(x - b) = 2(x - b) \cdot 1 = 2(x - b)
$$
So,
$$
\frac{dC}{dx} = 2(x - a) + 2(x - b)
$$

2.  **Set the derivative to zero to find the minimum:**

$$
2(x - a) + 2(x - b) = 0
$$

3.  **Solve for $x$:** Divide by 2:

$$
(x - a) + (x - b) = 0
$$
Expand:
$$
x - a + x - b = 0
$$
Combine terms:
$$
2x - a - b = 0
$$
Isolate $x$:
$$
2x = a + b
$$   $$
x = \frac{a + b}{2}
$$

### Conclusion

The optimal location for the house to minimize the total connection cost to two power lines is exactly at the **midpoint** between the two power lines.

## Optimizing House Location with Three Power Lines

Now, let's extend the problem to **three power lines**, located at positions $a$, $b$, and $c$ from the origin. The goal remains to minimize the total cost of connecting the house (at position $x$) to all three.

### Defining the Cost Function

* **Distance to power line 1:** $|x - a|$
* **Distance to power line 2:** $|x - b|$
* **Distance to power line 3:** $|x - c|$

The cost for each connection is the square of its distance. The **total cost function** $C(x)$ is the sum of these squared distances:

$$C(x) = (x - a)^2 + (x - b)^2 + (x - c)^2$$

### Visualizing the Cost (Area Perspective)

This cost function represents the sum of the areas of three squares. Each square's side length corresponds to the distance from the house to one of the power lines.  Minimizing the total cost is equivalent to minimizing the total area of these three squares.

### Minimizing the Cost Function Using Derivatives

Just like with two power lines, $C(x)$ is a quadratic function that opens upwards, meaning its minimum occurs where its derivative is zero.

1.  **Calculate the derivative of the cost function with respect to $x$:**

$$
\frac{dC}{dx} = \frac{d}{dx}((x - a)^2 + (x - b)^2 + (x - c)^2)
$$
Using the chain rule for each term:
$$
\frac{dC}{dx} = 2(x - a) + 2(x - b) + 2(x - c)
$$

2.  **Set the derivative to zero to find the minimum:**

$$
2(x - a) + 2(x - b) + 2(x - c) = 0
$$

3.  **Solve for $x$:**
    Divide the entire equation by 2:

$$
(x - a) + (x - b) + (x - c) = 0
$$
Expand:
$$
x - a + x - b + x - c = 0
$$
Combine terms:
$$
3x - a - b - c = 0
$$
Isolate $x$:
$$
3x = a + b + c
$$   $$
x = \frac{a + b + c}{3}
$$

### Conclusion

The optimal location for the house to minimize the total connection cost to three power lines is the **average** of the coordinates of the three power lines.

### Generalization to N Power Lines

This concept can be generalized to $n$ power lines located at coordinates $a_1, a_2, \ldots, a_n$. The total cost function would be:

$$C(x) = \sum_{i=1}^{n} (x - a_i)^2$$

The derivative, when set to zero, would yield the optimal $x$:

$$x = \frac{a_1 + a_2 + \ldots + a_n}{n}$$

This means the optimal location is the **average** (or **mean**) of all power line coordinates.

### Connection to Machine Learning

This cost function, which sums the squared differences between observed values ($a_i$) and a predicted value ($x$), is precisely the **squared error loss function**. This is a fundamental concept in machine learning, especially in:

* **Linear Regression**: Used to find the "best-fit" line that minimizes the sum of squared residuals between predicted and actual values.
* **Neural Networks**: Employed in various architectures for training and optimization.

This problem provides a tangible example of how minimizing a sum of squared errors leads to the mean, a principle that underpins many machine learning algorithms.

## Log Loss Explained with Coin Flips

* **Log Loss**: A crucial function in machine learning, particularly in classification problems. It is derived from the logarithm of probabilities.
* **Coin Flip Game Analogy**: To understand log loss, consider a game where you toss a coin 10 times and win if you get 7 heads followed by 3 tails. You can choose a biased coin.
    * **Goal**: Maximize the probability of winning.
    * **Coin Probabilities**:
        * Coin 1: P(Heads) = 0.7, P(Tails) = 0.3
        * Coin 2 (Fair): P(Heads) = 0.5, P(Tails) = 0.5
        * Coin 3: P(Heads) = 0.3, P(Tails) = 0.7
    * **Calculating Probabilities of Winning for Each Coin**:
        * **Coin 1**: $P(Win) = 0.7^7 \times 0.3^3 = 0.00222$
        * **Coin 2**: $P(Win) = 0.5^7 \times 0.5^3 = 0.00097$
        * **Coin 3**: $P(Win) = 0.3^7 \times 0.7^3 = 0.00008$
    * **Conclusion**: Coin 1 gives the highest probability of winning.

## Finding the Optimal Probability (p) using Calculus

* **Generalizing the Problem**: Let 'p' be the probability of heads and '1-p' be the probability of tails for any coin.
* **Probability of Winning Function**: $g(p) = p^7 \times (1-p)^3$
* **Maximizing g(p) using Derivatives**:
    * Take the derivative of $g(p)$ with respect to $p$ and set it to zero.
    * Using the **product rule**: $\frac{d}{dp}(uv) = u'v + uv'$
        * $u = p^7$, $u' = 7p^6$
        * $v = (1-p)^3$, $v' = 3(1-p)^2 \times (-1)$ (due to **chain rule**)
    * $\frac{dg}{dp} = 7p^6(1-p)^3 + p^7(3(1-p)^2(-1))$
    * Factor out common terms: $p^6(1-p)^2 [7(1-p) - 3p] = 0$
    * This simplifies to: $p^6(1-p)^2 (7 - 7p - 3p) = 0 \implies p^6(1-p)^2 (7 - 10p) = 0$
    * **Possible values for p**:
        * $p^6 = 0 \implies p = 0$ (Discarded: Cannot get 7 heads)
        * $(1-p)^2 = 0 \implies p = 1$ (Discarded: Cannot get 3 tails)
        * $7 - 10p = 0 \implies p = 0.7$
    * **Result**: The optimal probability of heads is $p = 0.7$, confirming Coin 1 was the best.

## The Power of Logarithms: Simplifying Optimization

* **Logarithm Trick**: Maximizing $g(p)$ is equivalent to maximizing $\log(g(p))$. This often simplifies calculations.
* **Applying Logarithm to g(p)**:
    * $G(p) = \log(g(p)) = \log(p^7 \times (1-p)^3)$
    * Using **logarithm properties**:
        * $\log(ab) = \log(a) + \log(b)$
        * $\log(a^b) = b \log(a)$
    * $G(p) = \log(p^7) + \log((1-p)^3) = 7\log(p) + 3\log(1-p)$
* **Maximizing G(p) using Derivatives**:
    * Take the derivative of $G(p)$ with respect to $p$ and set it to zero.
    * Recall: $\frac{d}{dp}(\log(p)) = \frac{1}{p}$
    * $\frac{dG}{dp} = 7 \left(\frac{1}{p}\right) + 3 \left(\frac{1}{1-p}\right) \times (-1)$ (due to **chain rule** for $1-p$)
    * $\frac{dG}{dp} = \frac{7}{p} - \frac{3}{1-p} = 0$
    * Solve for p: $\frac{7}{p} = \frac{3}{1-p} \implies 7(1-p) = 3p \implies 7 - 7p = 3p \implies 7 = 10p \implies p = 0.7$
* **Conclusion**: The logarithm approach yields the same optimal $p = 0.7$ but with much simpler derivative calculations.

## Log Loss Function

* **Definition**: In machine learning, we often use the **negative logarithm of the probability**, referred to as **log loss**.
* **Reason for Negation**: The logarithm of a probability (between 0 and 1) is typically a negative number. Taking the negative ensures the loss function is positive.
* **Minimization vs. Maximization**: Instead of maximizing the probability (or its logarithm), machine learning models typically **minimize the log loss**.
* **Common Use**: Log loss is widely used as a loss function in **classification problems** .

## Machine Learning and the Log Loss

The coin flip example demonstrates a core machine learning concept: **finding the best model for a dataset**.

* **Dataset**: The observed 10 coin flips (7 heads, 3 tails).
* **Model**: The coin, characterized by its probability 'p' of landing heads.
* **Goal**: Find the 'p' that makes the model (coin) most likely to produce the observed dataset. This is achieved by **minimizing the log loss**.

## Why Use Logarithms in Log Loss?

There are two primary reasons why logarithms are integral to the log loss function and other machine learning calculations:

### 1. Simplification of Derivatives

* **Product Rule Complexity**: Directly calculating the derivative of a product of many terms (like $p^7 \times (1-p)^3$) becomes incredibly complex as the number of terms increases. The product rule becomes iterative and messy.
* **Logarithm Simplification**: The logarithm transforms products into sums, i.e., $\log(ab) = \log(a) + \log(b)$. This is a powerful simplification because the derivative of a sum is simply the sum of the derivatives, which is much easier to compute.
    * For example, deriving $7\log(p) + 3\log(1-p)$ is significantly simpler than deriving $p^7(1-p)^3$. While denominators appear (due to $\frac{d}{dx}(\log x) = \frac{1}{x}$), this is a small price to pay for the overall simplification.

### 2. Handling Numerical Precision Issues

* **Product of Small Probabilities**: When dealing with many independent events, probabilities are multiplied. If you have a product of thousands of probabilities (each between 0 and 1), the resulting number can be extremely small, potentially causing **underflow** errors in computer systems. Computers have limitations in representing numbers very close to zero.
    * 
* **Logarithm to the Rescue**: Taking the logarithm of a very small number results in a large negative number. Computers are much better at handling a wide range of negative numbers than extremely small positive numbers. This prevents numerical instability and ensures calculations are accurate.
    * Therefore, whenever complex products are involved, especially with probabilities, **using logarithms is a standard practice to maintain numerical stability**.

## Conclusion

The use of logarithms in functions like log loss is a clever mathematical trick that simplifies derivative calculations and mitigates numerical precision issues, making complex optimization problems in machine learning tractable and robust.

## Week 2: Functions with Multiple Variables

This week introduces **functions with two or more variables**, extending concepts from single-variable calculus to higher dimensions.

### From Tangent Line to Tangent Plane

* **Single Variable Functions (e.g., $f(x) = x^2$):** The **derivative** at a point represents the **slope of the tangent line** at that point.
    * Example: For $f(x) = x^2$, at point (2, 4), the slope of the tangent line is 4.

* **Multi-Variable Functions (e.g., $f(x, y) = x^2 + y^2$):**
    * These functions have multiple inputs (e.g., $x, y$) and one output ($f(x, y)$ or $z$), requiring a **3D plot**. 
    * Tangents are no longer lines but **planes**, called **tangent planes**.

### Constructing a Tangent Plane

To find a tangent plane, we take "slices" of the 3D function:

1.  **Fix one variable:**
    * If we fix $y = \text{constant}$ (e.g., $y=4$), we get a 2D parabola in the XZ plane ($f(x, 4) = x^2 + 4^2$). We can then find the **tangent line** to this parabola with respect to $x$.
    * If we fix $x = \text{constant}$ (e.g., $x=2$), we get a 2D parabola in the YZ plane ($f(2, y) = 2^2 + y^2$). We can then find the **tangent line** to this parabola with respect to $y$.

2.  **Two Tangent Lines Form a Plane:** Any two intersecting lines define a unique plane. The two tangent lines obtained from fixing one variable at a time will form the **tangent plane**.

### Optimization and Gradient Descent

* Optimizing functions with two or more variables can be computationally complex.
* **Gradient Descent** is a method that will be introduced to speed up optimization processes for these complex functions.

## Partial Derivatives

Partial derivatives are a fundamental concept when dealing with functions of multiple variables. They represent the rate of change of a multivariable function with respect to one variable, while holding all other variables constant.

### Visualizing Partial Derivatives

Imagine a 3D plot of a function $f(x, y)$.

* If you **cut this 3D surface with a plane** where $y$ is held constant (e.g., $y=4$), the intersection forms a 2D curve (a red parabola in the example). 
* The **tangent line** to this 2D curve at a specific point represents the **partial derivative with respect to $x$** at that point.
* Similarly, if you cut the surface with a plane where $x$ is held constant, you get another 2D curve, and its tangent line represents the **partial derivative with respect to $y$**. 

### Calculating Partial Derivatives

To calculate a partial derivative, you follow these steps:

1.  **Treat all other variables as constants.**
2.  **Differentiate the function** using the standard rules of differentiation with respect to the variable of interest.

#### Example: $f(x, y) = x^2 + y^2$

* **Partial Derivative with respect to $x$ ($\frac{\partial f}{\partial x}$ or $f_x$):**
    * Treat $y$ as a constant.
    * So, $f(x, y) = x^2 + \text{constant}$.
    * $\frac{\partial f}{\partial x} = \frac{d}{dx}(x^2) + \frac{d}{dx}(\text{constant}) = 2x + 0 = 2x$.

* **Partial Derivative with respect to $y$ ($\frac{\partial f}{\partial y}$ or $f_y$):**
    * Treat $x$ as a constant.
    * So, $f(x, y) = \text{constant} + y^2$.
    * $\frac{\partial f}{\partial y} = \frac{d}{dy}(\text{constant}) + \frac{d}{dy}(y^2) = 0 + 2y = 2y$.

### Notation

For a function $f(x, y)$:

* Partial derivative with respect to $x$: $\frac{\partial f}{\partial x}$ or $f_x$.
* Partial derivative with respect to $y$: $\frac{\partial f}{\partial y}$ or $f_y$.

If a function has more than two variables (e.g., 10 variables), you can take a partial derivative with respect to each of those variables.

## Example: $f(x, y) = 3x^2y^3$

### Partial Derivative with Respect to $x$ ($\frac{\partial f}{\partial x}$ or $f_x$)

To find the partial derivative of $f(x, y)$ with respect to $x$:

1.  **Treat $y$ as a constant.** This means $y^3$ is considered a constant.
2.  **Differentiate the function** with respect to $x$ using standard differentiation rules.

$$\frac{\partial}{\partial x} (3x^2y^3)$$
$$= 3 \cdot y^3 \cdot \frac{\partial}{\partial x} (x^2)$$
$$= 3 \cdot y^3 \cdot (2x)$$
$$= 6xy^3$$

### Partial Derivative with Respect to $y$ ($\frac{\partial f}{\partial y}$ or $f_y$)

To find the partial derivative of $f(x, y)$ with respect to $y$:

1.  **Treat $x$ as a constant.** This means $x^2$ is considered a constant.
2.  **Differentiate the function** with respect to $y$ using standard differentiation rules.

$$\frac{\partial}{\partial y} (3x^2y^3)$$
$$= 3 \cdot x^2 \cdot \frac{\partial}{\partial y} (y^3)$$
$$= 3 \cdot x^2 \cdot (3y^2)$$
$$= 9x^2y^2$$

## The Gradient Vector ∇

The **gradient** is a way to condense all the partial derivatives of a multivariable function into a single vector. It's denoted by the **nabla symbol** ($\nabla$).

### Definition of the Gradient

For a function $f$ with multiple variables, the gradient $\nabla f$ is a vector containing all its partial derivatives.

* **For a function of two variables**, $f(x, y)$: The gradient is:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}
$$

* **For a function of 'n' variables**, $f(x_1, x_2, \dots, x_n)$: The gradient is:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

This means if a function has 17 variables, its gradient would be a vector with 17 entries, each corresponding to the partial derivative with respect to one of those variables.

### Gradient as a Description of the Tangent Plane

The gradient vector provides a good description of the **tangent plane** because its components are the slopes of the lines that form that tangent plane (the partial derivatives in each direction).

### Example: Finding the Gradient

Let's find the gradient of the function $f(x, y) = x^2 + y^2$.

1.  **Calculate the partial derivative with respect to $x$**:
    $\frac{\partial f}{\partial x} = 2x$

2.  **Calculate the partial derivative with respect to $y$**:
    $\frac{\partial f}{\partial y} = 2y$

3.  **Form the gradient vector**:

$$
\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}
$$

### Evaluating the Gradient at a Specific Point

To find the gradient at a specific point (e.g., $(2, 3)$ for $f(x, y) = x^2 + y^2$), simply substitute the coordinates of the point into the gradient vector.

For $f(x, y) = x^2 + y^2$ at the point $(2, 3)$:

$$\nabla f(2, 3) = \begin{bmatrix} 2(2) \\ 2(3) \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$

This vector [4, 6] represents the gradient of the function $f(x, y) = x^2 + y^2$ at the point $(2, 3)$.

## The Gradient and Optimization

The gradient is crucial for **minimizing (or maximizing) functions with multiple variables**, much like a single derivative is used for single-variable functions.

### Minimizing Single-Variable Functions

* For a function like $f(x) = x^2$, the minimum occurs where the **slope of the tangent line is zero**.
* We find this point by setting the derivative to zero:
    * $f'(x) = 2x$
    * Set $2x = 0$, which gives $x = 0$. This is the minimum point.

### Minimizing Multi-Variable Functions

* For a function like $f(x, y) = x^2 + y^2$, the minimum point is the lowest point on the 3D surface.
* At this minimum, the **tangent plane is parallel to the "floor"** (the XY-plane).
* This implies that the **slopes of all partial derivatives are zero** at this point.
    
* To find the minimum, we set **all partial derivatives equal to zero** and solve the resulting system of equations.
    * **Partial derivative with respect to $x$**: $\frac{\partial f}{\partial x} = 2x$
    * **Partial derivative with respect to $y$**: $\frac{\partial f}{\partial y} = 2y$
    * Set both to zero:
        * $2x = 0 \implies x = 0$
        * $2y = 0 \implies y = 0$
    * The solution is the point $(x, y) = (0, 0)$, which is the minimum.

* This principle extends to functions with many variables: to find minima or maxima, set all partial derivatives to zero and solve the system of equations. For example, for a function of 12 variables, you would set all 12 partial derivatives to zero.

## Multidimensional Optimization in a Sauna

* **Problem:** Find the coolest point in a 2D sauna, where temperature is a function of x and y coordinates, T(x, y).
* **Analogy to 1D:** Similar to the 1D sauna problem (Week 1), where movement was restricted to left/right, here you can move in any direction.
* **Goal:** Reach the point where taking any step makes you hotter. This corresponds to a minimum in the temperature function.

### Finding the Minimum Using Derivatives

* **Concept:** At a minimum (or maximum, or saddle point), the **tangent plane** to the temperature function is parallel to the floor. This means the **partial derivatives** with respect to both x and y are zero.
    * $\frac{\partial T}{\partial x} = 0$
    * $\frac{\partial T}{\partial y} = 0$
* **Example Temperature Function:**
    $T(x, y) = 85 - \frac{1}{90} (x^2(x-6)y^2(y-6))$
* **Calculating Partial Derivatives:**
    * To find $\frac{\partial T}{\partial x}$, treat y as a constant.
        $\frac{\partial T}{\partial x} = -\frac{1}{90} [ (3x^2 - 12x) y^2(y-6) ] = -\frac{1}{90} [ x(3x - 12) y^2(y-6) ]$
    * To find $\frac{\partial T}{\partial y}$, treat x as a constant.
        $\frac{\partial T}{\partial y} = -\frac{1}{90} [ x^2(x-6) (3y^2 - 12y) ] = -\frac{1}{90} [ x^2(x-6) y(3y-12) ]$

### Solving for Critical Points

* **Setting Partial Derivatives to Zero:**
    * $\frac{\partial T}{\partial x} = 0 \implies x = 0$ or $x = 4$ or $y = 0$ or $y = 6$
    * $\frac{\partial T}{\partial y} = 0 \implies x = 0$ or $x = 6$ or $y = 0$ or $y = 4$
* **Candidate Points (where both partial derivatives are zero):**
    * $(0, 0)$
    * $(0, 4)$
    * $(0, 6)$
    * $(4, 0)$
    * $(4, 4)$
    * $(6, 0)$
    * $(6, 6)$
* **Filtering Candidates:**
    * Many points on the edges (where x=0, x=6, y=0, y=6) typically correspond to local maxima (temperature = 85 in this example).
    * The point $(4, 4)$ is the global minimum, with a temperature of 73.6. This is the coldest point.
* **Conclusion:** Just like in 1D, setting the derivatives to zero provides candidate points for minima (or maxima). These candidates must then be evaluated to find the desired minimum.

## Two-Dimensional Power Line Problem & Linear Regression

This lesson extends the one-dimensional power line problem to two dimensions, introducing **linear regression** as a core machine learning concept. The goal is to find an optimal straight fiber line connection to minimize the total cost of connecting to three power lines.

### Problem Setup

* Power lines are at specific (x, y) coordinates.
* Connections to the fiber line are made by wires parallel to the y-axis.
* Cost of connection is the square of the length of the wire.
* The fiber line is represented by the equation $y = mx + b$.
* The objective is to find the optimal $m$ (slope) and $b$ (y-intercept) that minimize the total cost.

### Cost Function Derivation

Consider the three power line coordinates: $(1,2)$, $(2,5)$, and $(3,3)$.

* For a power line at $(x_i, y_i)$, the connection point on the fiber line will have coordinates $(x_i, mx_i + b)$.
* The length of the wire is $|(mx_i + b) - y_i|$.
* The cost for each power line is $((mx_i + b) - y_i)^2$.

Let's denote the total cost as $E(m,b)$.

* **Cost for (1,2):** $(m(1) + b - 2)^2 = (m + b - 2)^2$
* **Cost for (2,5):** $(m(2) + b - 5)^2 = (2m + b - 5)^2$
* **Cost for (3,3):** $(m(3) + b - 3)^2 = (3m + b - 3)^2$

The total cost function is the sum of these individual costs:

$$E(m,b) = (m + b - 2)^2 + (2m + b - 5)^2 + (3m + b - 3)^2$$

Expanding and combining like terms, the cost function becomes:

$$E(m,b) = 14m^2 + 3b^2 + 12mb - 42m - 20b + 38$$

### Minimizing the Cost Function

To find the optimal $m$ and $b$ that minimize $E(m,b)$, we use partial derivatives:

* Set the partial derivative of $E$ with respect to $m$ equal to 0.
* Set the partial derivative of $E$ with respect to $b$ equal to 0.

#### Partial Derivative with respect to $m$:

$$\frac{\partial E}{\partial m} = 28m + 12b - 42$$

#### Partial Derivative with respect to $b$:

$$\frac{\partial E}{\partial b} = 6b + 12m - 20$$

Setting both derivatives to zero gives a system of linear equations:

1.  $28m + 12b - 42 = 0$
2.  $12m + 6b - 20 = 0$

Solving this system of equations: $m = \frac{1}{2}$ and $b = \frac{7}{3}$.

### Minimum Cost and Linear Regression

* The minimum cost $E(\frac{1}{2}, \frac{7}{3}) \approx 4.167$.
* The optimal fiber line equation is $y = \frac{1}{2}x + \frac{7}{3}$. 

This problem is a classic example of **linear regression**, a fundamental machine learning technique used to find the best-fitting straight line (or hyperplane) through a set of data points.

### Future Considerations

Solving systems of equations can become computationally expensive with many variables. Future lessons will introduce **gradient descent** as a more efficient method for minimizing functions in such scenarios.

## Gradient Descent: An Iterative Optimization Approach

While derivatives and gradients help solve optimization problems, finding exact analytical solutions can quickly become complex, especially in higher dimensions. **Gradient descent** is an iterative and powerful method for minimizing (or maximizing) functions with many variables.

### The Challenge of Analytical Solutions

Consider the function $f(x) = e^x - \log(x)$.

To find its minimum analytically, we'd calculate its derivative, $f'(x) = e^x - \frac{1}{x}$, and set it to zero:
$$e^x - \frac{1}{x} = 0$$
This is equivalent to $e^x = \frac{1}{x}$. Solving this equation analytically for $x$ is difficult. The solution is approximately $x \approx 0.5671$, known as the **Omega constant** or Lambert W function evaluated at 1.

### Limitations of Brute-Force Exploration

One might consider an iterative approach:
1.  **Pick a random starting point** for $x$.
2.  **Explore nearby points** (e.g., move a small step left and right).
3.  **Choose the point that yields a smaller function value** (if minimizing).
4.  **Repeat** until moving in either direction results in a higher function value, indicating a local minimum.

While this approach works, it's inefficient and might not pinpoint the exact minimum precisely. Gradient descent offers a more refined and efficient way to navigate towards the minimum.

## The Idea Behind Gradient Descent

Imagine you're trying to find the lowest point in a valley while blindfolded. Instead of randomly exploring, you'd feel the slope under your feet and always take a small step in the steepest downhill direction. That's essentially what gradient descent does.

### Utilizing the Derivative (Slope)

* **If the slope is negative** (you're to the left of the minimum and going downhill), you need to move to the **right** (increase your $x$ value).
* **If the slope is positive** (you're to the right of the minimum and going uphill), you need to move to the **left** (decrease your $x$ value).

This observation leads to the core updating rule:
**New Point = Old Point - Slope**

Mathematically, if $x_0$ is the old point and $x_1$ is the new point, then:

$$x_1 = x_0 - f'(x_0)$$
where $f'(x_0)$ is the derivative (slope) of the function at $x_0$.

### Introducing the Learning Rate ($\alpha$)

Taking steps purely based on the slope can lead to problems, especially in steep areas of the function where the derivative is large. This could cause:
* **Large jumps:** You might overshoot the minimum and oscillate wildly.
* **Missing the minimum entirely:** You could jump past the minimum and get lost in another part of the function.

To control the step size and ensure a more stable convergence, we introduce a **learning rate**, denoted by $\alpha$ (alpha). This is a small positive number (e.g., 0.01) that scales the step taken in the direction of the negative gradient.

The refined update formula becomes:

$$x_{k+1} = x_k - \alpha \cdot f'(x_k)$$

where $x_k$ is the point at the current iteration $k$, and $x_{k+1}$ is the point for the next iteration.



### Benefits of the Learning Rate

The learning rate provides an added bonus:
* **Adaptive step size:** When the function is steep (derivative is large), the term $\alpha \cdot f'(x_k)$ will be relatively large, leading to a bigger step.
* **Smaller steps near minimum:** As you approach the minimum, the function becomes flatter, the derivative $f'(x_k)$ becomes smaller, resulting in smaller, more precise steps. This is analogous to a golfer hitting the ball hard when far from the hole but with precision when close.

NOTE: This assumption just breaks depending on the type of loss function.
*  Example: If loss function is say $x^2$ then it works, because farther off from minima slope is higher so we take relatively big steps, but closer off to minima slope is smaller so we take smaller steps and hence we are able to reach minima.
*  Example: If loss function is say $sqrt(|x|)$ then it don't work, because farther off from minima slope is smaller so we take relatively small steps, but closer off to minima slope is high (at 0 it approaches to infinity) so we take larger and larger steps -- hence we might oscillate and never reach minima.

To mitigate this we can use adaptive learning rate methods like Adam.

### The Gradient Descent Algorithm

Gradient descent is an iterative procedure:

1.  **Define the function** $f(x)$ you want to minimize.
2.  **Choose a learning rate** $\alpha$ (a small positive value, often between 0.001 and 0.1).
3.  **Choose an initial starting point** $x_0$.
4.  **Iterate** the updating step:
    For $k = 0, 1, 2, \dots$:
    $$x_{k+1} = x_k - \alpha \cdot f'(x_k)$$
5.  **Repeat** until convergence. Convergence occurs when the change in $x$ (or the function value $f(x)$) between iterations becomes very small, indicating you're close to the minimum.

This algorithm is very fast and can be iterated thousands of times, leading to solutions very close to the true minimums of functions.

### Example Iterations

Let's apply gradient descent to $f(x) = e^x - \log(x)$, where $f'(x) = e^x - \frac{1}{x}$.
Choose a starting point $x_0 = 0.05$ and a learning rate $\alpha = 0.005$.

#### Iteration 1:
* $f'(0.05) = e^{0.05} - \frac{1}{0.05} \approx 1.051 - 20 = -18.949$
* $x_1 = x_0 - \alpha \cdot f'(x_0) = 0.05 - 0.005 \cdot (-18.949)$
* $x_1 = 0.05 + 0.094745 = 0.144745$ (This is closer to the minimum compared to 0.05).

#### Iteration 2:
* $f'(0.144745) = e^{0.144745} - \frac{1}{0.144745} \approx 1.1557 - 6.909 \approx -5.7533$
* $x_2 = x_1 - \alpha \cdot f'(x_1) = 0.144745 - 0.005 \cdot (-5.7533)$
* $x_2 = 0.144745 + 0.0287665 \approx 0.1735115$

Notice that with gradient descent, you only need to *calculate* the derivative at each step; you never have to *solve* the equation $f'(x) = 0$ analytically, which is a significant improvement for complex functions.

## Challenges with Learning Rate

The learning rate ($\alpha$) determines the size of the steps taken during gradient descent.

* **Learning Rate Too Large:** If $\alpha$ is too large, the algorithm might **overshoot the minimum** repeatedly, causing the optimization process to diverge or oscillate wildly without ever converging to a stable solution. 

* **Learning Rate Too Small:** Conversely, if $\alpha$ is too small, each step will be tiny. This means the algorithm will take an **extremely long time to converge** to the minimum, or it might get stuck before reaching it within a practical timeframe. 

* **Finding the "Just Right" Learning Rate:** There's no single definitive method for picking the optimal learning rate. It's often problem-dependent and can require experimentation. This is an active area of research, with many advanced methods developed to dynamically adjust the learning rate during training (e.g., adaptive learning rate methods like Adam, RMSprop).

## The Local Minima Problem

Another challenge with gradient descent is the risk of getting stuck in a **local minimum** rather than finding the true **global minimum**.

* A **global minimum** is the lowest point across the entire function.
* A **local minimum** is the lowest point within a specific region, but not necessarily the absolute lowest point of the function.

If you start your gradient descent algorithm in a region that leads to a local minimum, it will converge there and won't explore other parts of the function that might contain the global minimum.

### Overcoming Local Minima

While there's no foolproof way to guarantee finding the global minimum, a common strategy to mitigate the local minima problem is to **run the gradient descent algorithm multiple times with different random starting points**. By initializing the algorithm from various locations, you increase the chances that at least one of these runs will converge to the global minimum or a very good local minimum.
