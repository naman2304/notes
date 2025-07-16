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
    * As $\Delta x$ (and thus $\Delta y$) approaches 0, we get:

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

***
Further YouTube videos:
* "Inverse functions and their derivatives youtube"
* "Derivative of inverse function formula explanation youtube"
