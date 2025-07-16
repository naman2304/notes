Appendix
* [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science)
  * [Linear Algebra for Machine Learning and Data Science](#linear-algebra-for-machine-learning-and-data-science)
  * Calculus for Machine Learning and Data Science
  * Probability & Statistics for Machine Learning & Data Science

---

## Importance of Mathematics in Machine Learning

Understanding the mathematics behind machine learning (ML) is crucial for:
* **Deeper understanding:** Grasping *how* and *why* ML algorithms work.
* **Customization and development:** Moving beyond off-the-shelf algorithms to build and customize models.
* **Algorithm selection:** Better judging when to apply specific techniques.
* **Debugging:** More effectively identifying and resolving issues in ML algorithms.
* **Innovation:** Potentially inventing new ML algorithms.

## Specialization Overview

This specialization consists of three courses:

### Course 1: Linear Algebra

* **Duration:** Four weeks.
* **Concepts covered:** Vectors, matrices, linear transformations, systems of equations, determinants.
* **Perspective:** Matrices are not just arrays of numbers; they can be viewed in deeper ways, like how a neural network uses matrix operations to transform, rotate, and warp data through multiple layers to arrive at a prediction.
* **Relevance to ML:** Many fundamental learning algorithms, including neural networks, are built upon matrix operations. A deeper intuition helps understand the "magic" of these algorithms.

### Course 2: Calculus

* **Concepts covered:** Focus on maximizing and minimizing functions.
* **Relevance to ML:**
    * The vast majority of learning algorithms are designed by creating a **cost function** and then **minimizing** it.
    * Understanding derivatives helps in tuning optimization algorithms (e.g., gradient descent, Adam) to perform better when they are not working as expected.
    * Demystifies optimizers that might initially seem obscure.

### Course 3: Probability and Statistics

* **Concepts covered:** Probability distributions, hypothesis testing, p-values.
* **Relevance to ML:**
    * Many ML models output probabilities.
    * **Maximum Likelihood Estimation (MLE):** A key concept where the goal is to find the scenario (model) that most likely generated the observed evidence (data). This involves maximizing the probability that the model generated the given data.
    * Gaining a deeper understanding of probability and statistics helps achieve a better mastery of ML algorithms.

## Practical Application

* Learners will not only study the math but also see it implemented in **Python code** through practical labs.
* Basic Python knowledge is assumed, with resources provided for those needing to get up to speed.
* The specialization progresses from a high school math level to core concepts relevant for ML and data science.

# Linear Algebra For Machine Learning And Data Science

This week introduces the concept of a **system of linear equations**, its representations, and the significance of **singular** and **non-singular systems** in linear algebra. Linear algebra is a fundamental and widespread mathematical field in machine learning.

### Applications of Linear Algebra in Machine Learning

Linear algebra is crucial for many ML applications, with **linear regression** being a prime example.

### Course Approach

This course prioritizes building a strong **mathematical foundation**. While machine learning examples will provide context, the primary goal is understanding the math. It's okay if you don't grasp all ML details; the focus is on the underlying mathematics.

### Linear Regression: A Machine Learning Example

* **Supervised Learning:** Linear regression is a supervised machine learning approach where you have collected data (inputs and outputs) and aim to discover relationships between them.
* **Example: Wind Turbine Power Prediction**
    * **Single Feature (Wind Speed):** If only wind speed ($x$) is used to predict power output ($y$), the relationship can be modeled by a line.
        $$y = mx + b$$  
        In machine learning, this is often written as:
        $$y = wx + b$$  
        where $w$ is the "**weight**" and $b$ is the "**bias**". The goal is to find the best $w$ and $b$ values to fit the data.
    * **Multiple Features (Wind Speed, Temperature, etc.):** When more features (e.g., wind speed $x_1$, temperature $x_2$) are considered, the equation expands:  
        $$y = w_1 x_1 + w_2 x_2 + b$$  
        Graphically, with two features, this forms a plane in 3D space.
        For $n$ features, the model becomes:  
        $$y = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b$$  
        The aim is to find the right values for the weights ($w_i$) and bias ($b$) to make accurate predictions, assuming a linear relationship.
* **Dataset Representation:**
    * For a single data record, the equation is:  
        $$y = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b$$  
        Here, $x_i$ and $y$ values are known, and the goal is to find $w_i$ and $b$.
    * For a dataset with $m$ records, each record provides an equation:  
        $$y^{(1)} = w_1 x_1^{(1)} + w_2 x_2^{(1)} + \ldots + w_n x_n^{(1)} + b$$  
        $$y^{(2)} = w_1 x_1^{(2)} + w_2 x_2^{(2)} + \ldots + w_n x_n^{(2)} + b$$  
        $$\vdots$$  
        $$y^{(m)} = w_1 x_1^{(m)} + w_2 x_2^{(m)} + \ldots + w_n x_n^{(m)} + b$$  
        The superscripts $(j)$ denote the $j^{th}$ example in the dataset and are *not* exponents.
* **System of Linear Equations:** Ideally, the values of weights and bias terms would solve all these equations simultaneously, or at least get as close as possible. This collection of equations forms a **system of linear equations**, a core topic for this week.

### From Long Form to Vector/Matrix Notation

The long-form equation can be represented more compactly using vectors and matrices:
* A **vector of weights** $w$ (containing $w_1, \ldots, w_n$).
* A **matrix of features** $X$ (where each row is a feature set $x^{(m)}$).
* A **vector of targets** $y$.
* The model then becomes (conceptually, details of transpose depend on vector definitions):  
    $$y = w X + b$$  
    (Note: The precise vector/matrix multiplication requires careful handling of transposes, but the core idea is compact representation).

### Systems of Linear Equations in Machine Learning

* Linear regression fundamentally represents the system as a **system of linear equations**.
* If a perfect set of $w$ and $b$ values exists that perfectly predicts $y$ given $x$, then this system could be solved **analytically** (with basic algebra), provided there are at least as many unique data records as there are unknowns ($w$'s and $b$).
* In practice, linear regression solves the system **empirically** (iteratively and approximately) by finding the best-fit linear solution.

### Week 1 Concepts and Challenge Questions

This week will cover:
* Systems of linear equations.
* Representation of systems using vectors and matrices.
* Manipulation of these systems to compute determinants and other characteristics.

**Challenge Scenario:** Suppose you have scores $a$ (Linear Algebra), $c$ (Calculus), and $p$ (Probability and Statistics).
1.  **Equation 1:** Linear Algebra score + Calculus score - Probability and Statistics score = 6  
    $$a + c - p = 6$$
2.  **Equation 2:** Linear Algebra score - Calculus score + 2 * Probability and Statistics score = 4  
    $$a - c + 2p = 4$$
3.  **Equation 3:** 4 * Linear Algebra score - 2 * Calculus score + Probability and Statistics score = 10  
    $$4a - 2c + p = 10$$

**Questions to Consider:**

1.  **Represent as a System of Linear Equations:** (Already done above).
2.  **ML Equivalents:**
    * **Weights ($w$):** Your scores $a, c, p$ (these are the consistent unknowns we're trying to find).
    * **Features ($x$):** The coefficients next to the scores in each equation (e.g., for the first equation: $1, 1, -1$).
    * **Target ($y$):** The numbers on the right side of the equal sign ($6, 4, 10$).
3.  **Singular or Non-Singular?** Do these equations contradict each other, or is there redundant information? Can this system be solved?
4.  **Matrix and Vector Representation:** Can you represent this system as a matrix and a vector?
5.  **Determinant:** Can you calculate the determinant of that matrix?

Answering these questions indicates readiness for the week's quiz and labs. If not, the week's materials will guide you step-by-step through solving such systems and understanding properties like singularity and the determinant.

## Introduction to Systems of Sentences (Analogy for Systems of Equations)

Understanding how sentences (information) combine is analogous to how equations combine.

### Types of Systems of Sentences

Based on information conveyed:

* **Complete System:** Contains as many unique pieces of information as sentences.
    * *Example:* "The dog is black." "The cat is orange."
* **Redundant System:** Contains fewer unique pieces of information than sentences, due to repetition.
    * *Example:* "The dog is black." "The dog is black."
* **Contradictory System:** Contains conflicting information.
    * *Example:* "The dog is black." "The dog is white."

### Terminology: Singular vs. Non-Singular

* **Non-Singular System:** A **complete** system. It provides the maximum possible information (as many pieces of information as sentences).
* **Singular System:** A system that is either **redundant** or **contradictory**. It provides less information than a non-singular system.

* **Rank:** A measure of how redundant a system is (to be covered later).

### Complex System Example

**System:**
1.  One of (dog, cat, bird) is red.
2.  One of (dog, cat) is orange.
3.  The dog is black.

**Solution:**
* From (3): Dog is black.
* From (2) and Dog's color: Cat is orange.
* From (1) and Dog/Cat colors: Bird is red.

**Analysis:**
* This system allows determining the color of all three animals (dog black, cat orange, bird red).
* It provides three distinct pieces of information from three sentences.
* Therefore, it is a **complete** and **non-singular** system.

## Linear Equations and Systems of Linear Equations

Equations are statements providing numerical information, fundamental to linear algebra.

### What is a Linear Equation?

* An equation where variables are only multiplied by scalars and then added or subtracted. A constant term is also allowed.
    * *Example:* $a + b = 10$, $2x - 3y + z = 5$.
* **Non-linear equations** include variables with powers ($x^2$), in exponents ($2^x$), multiplied together ($xy$), divided ($y/x$), or within non-linear functions ($\sin(x)$, $$log(x)$$).
* Linear algebra specifically focuses on **linear equations** due to their predictable and manipulable properties.

### Types of Solutions for Systems of Linear Equations

Similar to systems of sentences, systems of linear equations are classified by the nature of their solutions.

* **Unique Solution (Complete / Non-Singular):**
    * Each equation provides genuinely new and independent information.
    * The number of independent equations equals the number of unknowns.
    * Results in exactly one specific set of values for all variables.
    * *Example:* $a + b = 10$ and $a + 2b = 12$. These two distinct equations uniquely determine $a$ and $b$.

* **Infinitely Many Solutions (Redundant / Singular):**
    * At least one equation provides redundant information, meaning it can be derived from other equations.
    * The number of independent equations is less than the number of unknowns.
    * Results in an infinite set of solutions.
    * *Example:* $a + b = 10$ and $2a + 2b = 20$. The second equation offers no new information, as it's just twice the first.

* **No Solution (Contradictory / Singular):**
    * The equations directly contradict each other, making it impossible for any values of the variables to satisfy all equations simultaneously.
    * *Example:* $a + b = 10$ and $2a + 2b = 24$. These statements are inconsistent: if $a+b=10$, then $2a+2b$ must be $20$, not $24$.

## Visualizing Systems of Linear Equations

Linear equations can be visualized as geometric shapes:
* **Two variables:** Represented as **lines** in a 2D coordinate plane ($a$-axis, $b$-axis).
* **Three variables:** Represented as **planes** in 3D space ($a$-axis, $b$-axis, $c$-axis).
* **More variables:** Represent "hyperplanes" in higher dimensions (hard to visualize).

Visualizing helps understand solutions and singularity.

### Visualizing 2D Systems (Lines in a Plane)

A system of two linear equations in two variables is represented by two lines in the same plane.

* **Unique Solution (Complete / Non-Singular System):**
    * The two lines intersect at a **single, unique point**. This point's coordinates are the unique solution to the system.
    * *Example:* $a + b = 10$ and $a + 2b = 12$. These lines cross at $(8, 2)$.

* **Infinitely Many Solutions (Redundant / Singular System):**
    * The two lines are **identical (overlap)**. Every point on that shared line is a solution.
    * *Example:* $a + b = 10$ and $2a + 2b = 20$. Both equations represent the same line.

* **No Solution (Contradictory / Singular System):**
    * The two lines are **parallel and distinct** (they never intersect).
    * *Example:* $a + b = 10$ and $2a + 2b = 24$. These lines are parallel but offset.

### Visualizing 3D Systems (Planes in Space)

A system of linear equations in three variables is represented by planes in 3D space.

* **Equation $a+b+c=1$:** Forms a plane intersecting axes at $(1,0,0)$, $(0,1,0)$, $(0,0,1)$.
* **Equation $3a-5b+2c=0$:** Forms a plane that *must* pass through the origin $(0,0,0)$ because $(0,0,0)$ is a solution.

Intersections of planes represent solutions to 3D systems:

* **Unique Solution (Complete / Non-Singular System):**
    * All planes intersect at a **single, unique point**.
    * *Example:* $a+b+c=0$, $a+2b+c=0$, $a+b+2c=0$. All three planes intersect only at the origin $(0,0,0)$.

* **Infinitely Many Solutions (Redundant / Singular System):**
    * Planes intersect along a **line** (e.g., three planes passing through the same line).
    * Or, planes are **identical** (e.g., all equations represent the same plane).
    * In both cases, there are multiple points of intersection.
    * *Example 1 (intersecting at a line):* The system $a+b+c=0$, $a+2b+c=0$, $a+b+2c=0$ (if the third equation was, for instance, a linear combination of the first two, leading to a line of intersection).
    * *Example 2 (identical planes):* $a+b+c=0$, $2a+2b+2c=0$, $3a+3b+3c=0$. All three equations represent the exact same plane.

* **No Solution (Contradictory / Singular System):**
    * Planes have **no common intersection point**, potentially due to parallel planes or complex arrangements where no single point is on all planes.

## Singular vs. Non-Singular: Simplified Visualization

This section introduces a simplified way to determine if a system of linear equations is singular or non-singular, by focusing solely on the relationship between the lines/planes, independent of their absolute position.

### The Role of Constants

Consider the general form of a linear equation: $Ax + By = C$. The constant term $C$ determines where the line (or plane) is positioned in space.

* **Original Systems:**
    * System 1: $a + b = 10$, $a + 2b = 12$ (Unique Solution, Non-Singular)
    * System 2: $a + b = 10$, $2a + 2b = 20$ (Infinitely Many Solutions, Singular - Redundant)
    * System 3: $a + b = 10$, $2a + 2b = 24$ (No Solution, Singular - Contradictory)

### Setting Constants to Zero

If all constant terms ($C$) in a system of linear equations are set to zero, the resulting lines (or planes) will always pass through the **origin** $(0,0)$ (or $(0,0,0)$ in 3D). This is because $(0,0)$ (or $(0,0,0)$) becomes a solution to every equation if the constants are zero.

* **Transformed Systems (Constants Set to Zero):**
    * System 1 (transformed): $a + b = 0$, $a + 2b = 0$.
        * **Plot:** Still two distinct lines intersecting at a unique point (the origin).
        * **Status:** Still **non-singular** (unique solution).
    * System 2 (transformed): $a + b = 0$, $2a + 2b = 0$.
        * **Plot:** Still two identical lines passing through the origin.
        * **Status:** Still **singular** (infinitely many solutions - redundant).
    * System 3 (transformed): $a + b = 0$, $2a + 2b = 0$.
        * **Plot:** Transforms from distinct parallel lines to identical lines (overlapping at the origin).
        * **Status:** Changes from "no solution/contradictory" to "infinitely many solutions/redundant", but crucially, it **remains singular**.

### Conclusion on Singularity

* **Key Insight:** Setting the constants to zero **does not change whether a system is singular or non-singular**.
* The concepts of "complete," "redundant," and "contradictory" describe *why* a system has a certain number of solutions, but "singular" and "non-singular" are the overarching classifications.
* **Significance:** This simplification means that to determine singularity, we can always analyze systems where all equations pass through the origin. This makes geometric interpretation simpler, as we only need to consider whether the lines/planes are distinct and intersecting, identical, or non-intersecting in their fundamental orientation, irrespective of their shifted positions.
* For the rest of the course, **singularity** and **non-singularity** will be the primary terms used.

## Introducing the Matrix

Matrices are fundamental objects in linear algebra, arising naturally from the coefficients of systems of linear equations.

### From System of Equations to Matrix

* Since constants don't affect singularity, we can simplify systems by setting all constants to zero.
* A **matrix** is an array of numbers (coefficients) arranged in a rectangle.
* **Structure:**
    * Each **row** of the matrix corresponds to an equation.
    * Each **column** of the matrix corresponds to the coefficients of a specific variable.
* **Examples (2x2 Systems):**  
**System 1:** $a + b = 0$, $a + 2b = 0$

$$
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
$$

**System 2:** $a + b = 0$, $2a + 2b = 0$

$$
\begin{bmatrix}
1 & 1 \\
2 & 2
\end{bmatrix}
$$

### Matrix Singularity

* A matrix is classified as **singular** or **non-singular** based on the singularity of its corresponding system of linear equations.
* If the system has a **unique solution** (non-singular), its matrix is **non-singular**.
* *Example:* Following is non-singular because $a+b=0, a+2b=0$ has only $(0,0)$ as a solution.

$$
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
$$

* If the system has **infinitely many solutions** or **no solutions** (singular), its matrix is **singular**.
* *Example:* Following is singular because $a+b=0, 2a+2b=0$ has infinitely many solutions.

$$
\begin{bmatrix}
1 & 1 \\
2 & 2
\end{bmatrix}
$$

### 3x3 Systems and their Matrices

Similar principles apply to systems with more variables and larger matrices.

* **System 1 (Non-Singular):**
* Example: $a+b+c=0$, $a+2b+c=0$, $a+b+2c=0$.
* Unique solution $(0,0,0)$.
* Corresponding Matrix (non-singular):

$$
\begin{bmatrix}
1 & 1 & 1 \\
1 & 2 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

* **System 2 & 3 (Singular - after setting constants to zero):**
    * Example: $a+b+c=0$, $a+b+2c=0$, $a+b+3c=0$ (originally had different constants, but becomes this when constants are zero).
    * Leads to $c=0$ and $a=-b$, thus infinitely many solutions where $a+b=0, c=0$.
    * Corresponding Matrix (singular):

$$
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 2 \\
1 & 1 & 3
\end{bmatrix}
$$

* **System 4 (Singular):**
    * Example: $a+b+c=0$, $2a+2b+2c=0$, $3a+3b+3c=0$.
    * All equations are multiples of the first, leading to infinitely many solutions where $a+b+c=0$.
    * Corresponding Matrix (singular):

$$
\begin{bmatrix}
1 & 1 & 1 \\
2 & 2 & 2 \\
3 & 3 & 3
\end{bmatrix}
$$

**Note:** Quicker methods exist to determine matrix singularity without solving the system (e.g., using determinants, which will be covered later).

## Linear Dependence and Independence of Rows

This concept provides a direct way to determine if a matrix (and its corresponding system of linear equations) is singular or non-singular, without solving the system.

### Linear Dependence

A set of rows (or equations) is **linearly dependent** if at least one row can be expressed as a **linear combination** of the others, meaning that row does not provide new information. One row can be a sum or weighted sum of others.

* **Example 1:** For the system $a = 1$, $b = 2$, $a+b = 3$, the coefficient matrix (assuming constants are zero for singularity check) is following. Here, Row 3 is Row 1 + Row 2. This indicates linear dependence, implying the matrix is **singular**.

$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
1 & 1 & 0
\end{bmatrix}
$$

* **Example 2:** For following matrix, Row 2 is $2 \times$ Row 1, and Row 3 is $3 \times$ Row 1. This demonstrates linear dependencies. This implies the matrix is **singular**.

$$
\begin{bmatrix}
1 & 1 & 1 \\
2 & 2 & 2 \\
3 & 3 & 3
\end{bmatrix}
$$

* **Example 3:** For following matrix, Row 2 is the average of Row 1 and Row 3 ($2 \times \text{Row 2} = \text{Row 1} + \text{Row 3}$). This implies the rows are linearly dependent, and the matrix is **singular**.

$$
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 2 \\
1 & 1 & 3
\end{bmatrix}
$$

### Linear Independence

A set of rows (or equations) is **linearly independent** if no row can be expressed as a linear combination of the others. Each row provides unique information.

* **Example:** For the system $a + b = 0$, $a + 2b = 0$, the matrix is following. Here, Row 2 is not a multiple of Row 1. This implies the rows are linearly independent, and the matrix is **non-singular** (system has a unique solution).

$$
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
$$

### Summary of Linear Dependence/Independence and Singularity

* **Linearly Dependent Rows $\implies$ Singular Matrix** (System has infinite solutions or no solutions).
* **Linearly Independent Rows $\implies$ Non-Singular Matrix** (System has a unique solution).

The concept of linear dependence also applies to columns of a matrix, which also determines singularity. Techniques to easily verify linear independence will be covered later.

## Determinant of a Matrix

The **determinant** is a quick formula to determine if a matrix is singular or non-singular.
* If the determinant is **zero**, the matrix is **singular**.
* If the determinant is **non-zero**, the matrix is **non-singular**.

### Determinant of a 2x2 Matrix

For a 2x2 matrix with entries:

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

The determinant is calculated as: $det(A) = ad - bc$
* $ad$ is the product of the numbers in the **main diagonal**.
* $bc$ is the product of the numbers in the **antidiagonal**.

**Example:**

$$
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
$$

Determinant = $(1 \times 2) - (1 \times 1) = 2 - 1 = 1$. Since $1 \neq 0$, the matrix is non-singular.

$$
\begin{bmatrix}
1 & 2 \\
1 & 2
\end{bmatrix}
$$

Determinant = $(1 \times 2) - (2 \times 1) = 2 - 2 = 0$. Since $0$, the matrix is singular.

### Determinant of a 3x3 Matrix

For a 3x3 matrix:

$$
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
$$

The determinant is calculated by summing the products of the elements along the "main" diagonals (top-left to bottom-right) and subtracting the products of the elements along the "anti-diagonals" (top-right to bottom-left).

**Visualizing the diagonals:**
1.  **Main diagonals (add):**
    * $a \cdot e \cdot i$
    * $b \cdot f \cdot g$ (wrap around from top-right to bottom-left)
    * $c \cdot d \cdot h$ (wrap around from top-right to bottom-left)
2.  **Anti-diagonals (subtract):**
    * $c \cdot e \cdot g$
    * $a \cdot f \cdot h$ (wrap around from top-left to bottom-right)
    * $b \cdot d \cdot i$ (wrap around from top-left to bottom-right)

So, the determinant is: $det(A) = (aei + bfg + cdh) - (ceg + afh + bdi)$

**Example:**

$$
\begin{bmatrix}
1 & 1 & 1 \\
1 & 2 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

* Main diagonals:
    * $1 \times 2 \times 2 = 4$
    * $1 \times 1 \times 1 = 1$
    * $1 \times 1 \times 1 = 1$  
    Sum of main diagonals = $4 + 1 + 1 = 6$

* Anti-diagonals:
    * $1 \times 2 \times 1 = 2$
    * $1 \times 1 \times 1 = 1$
    * $1 \times 1 \times 2 = 2$  
    Sum of anti-diagonals = $2 + 1 + 2 = 5$

Determinant = $6 - 5 = 1$. Since $1 \neq 0$, the matrix is non-singular.

### Shortcut for Upper Triangular Matrices

For an **upper triangular matrix** (where all elements below the main diagonal are zero), the determinant is simply the product of the elements in the main diagonal.

**Example:**

$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 3
\end{bmatrix}
$$

Determinant = $1 \times 2 \times 3 = 6$.
Even if an upper triangular matrix has a zero on its main diagonal, its determinant will be zero, indicating it is singular.

**Example:**

$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

Determinant = $1 \times 2 \times 0 = 0$. This matrix is singular.

## Solving Systems of Linear Equations

The goal is to transform a system of linear equations into a "solved system" where each variable is isolated and its value is directly stated. This process works for non-singular systems with a unique solution.

### Basic Equation Manipulations

To solve a system of equations, you can perform the following operations, which maintain the equivalence of the system:

* **Multiplying an equation by a constant:** If $A = B$, then $kA = kB$.
    * Example: If $a + b = 10$, then $7(a + b) = 7(10) \implies 7a + 7b = 70$.
* **Adding two equations:** If $A = B$ and $C = D$, then $A + C = B + D$.
    * Example: If $a + b = 10$ and $2a + 3b = 26$, then $(a + b) + (2a + 3b) = 10 + 26 \implies 3a + 4b = 36$.

### Algorithm for Solving a 2x2 System

Let's consider a general system:  
$ax + by = c$  
$dx + ey = f$

The general strategy is to eliminate one variable from one of the equations, solve for the remaining variable, and then substitute back to find the value of the first variable.

**Steps:**

1.  **Normalize the coefficient of the first variable (optional but often helpful):**
    * Divide the first equation by the coefficient of the first variable (e.g., 'a').
    * Divide the second equation by the coefficient of the first variable (e.g., 'd'), if 'd' is not zero.
    * This makes the coefficient of the first variable '1' in both equations, simplifying the next step.
    * Example: For $5a + b = 17$ and $4a - 3b = 6$:
        * Divide first equation by 5: $a + 0.2b = 3.4$
        * Divide second equation by 4: $a - 0.75b = 1.5$

2.  **Eliminate the first variable from one equation:**
    * Subtract one of the normalized equations from the other. This will cancel out the first variable.
    * Example: Subtract $(a + 0.2b = 3.4)$ from $(a - 0.75b = 1.5)$:
        $(a - 0.75b) - (a + 0.2b) = 1.5 - 3.4$
        $-0.95b = -1.9$

3.  **Solve for the second variable:**
    * Divide the resulting equation by the coefficient of the second variable.
    * Example: $-0.95b = -1.9 \implies b = \frac{-1.9}{-0.95} = 2$.

4.  **Substitute the value of the second variable into one of the original (or normalized) equations:**
    * Solve for the first variable.
    * Example: Substitute $b=2$ into $a + 0.2b = 3.4$:
        $a + 0.2(2) = 3.4$
        $a + 0.4 = 3.4$
        $a = 3.4 - 0.4 = 3$.

**Special Case: Variable Already Eliminated**

If, during the normalization step, the coefficient of the variable you intend to eliminate is zero in one of the equations, it means that variable is already eliminated from that equation. Proceed directly to solve that equation for the present variable.

**Example:**
$5a + b = 17$
$0a + 3b = 6$ (or simply $3b = 6$)

* From the second equation, $3b = 6 \implies b = 2$.
* Substitute $b=2$ into the first equation: $5a + 2 = 17 \implies 5a = 15 \implies a = 3$.

## Solving Singular Systems of Linear Equations

The method of elimination can also identify singular systems, which do not have a unique solution. Singular systems fall into two categories: redundant (infinitely many solutions) and contradictory (no solutions).

### Redundant Systems (Infinitely Many Solutions)

A system is redundant if one or more equations are linear combinations of the others, meaning they provide no new information.

**Example:**
$a + b = 10$
$2a + 2b = 20$

**Applying the Elimination Method:**

1.  **Normalize coefficients (divide by coefficient of 'a'):**
    * Equation 1: $a + b = 10$
    * Equation 2: $a + b = 10$ (dividing $2a + 2b = 20$ by 2)

2.  **Eliminate 'a' from the second equation:**
    * Subtract Equation 1 from Equation 2:
        $(a + b) - (a + b) = 10 - 10$
        $0 = 0$

**Interpretation:**
* The result $0 = 0$ is a trivially true statement that provides no information about the variables.
* This indicates that the equations are linearly dependent and the system is **redundant**.
* There is no unique solution; instead, there are **infinitely many solutions**.

**Representing the Solution:**
Since $a + b = 10$ is the only independent equation, we can express one variable in terms of the other.
* Let $a = x$ (where $x$ can be any real number, representing a "degree of freedom").
* Then, $b = 10 - x$.
* The solutions form a line in a 2D plot.

### Contradictory Systems (No Solutions)

A system is contradictory if the equations are inconsistent, meaning they cannot all be true simultaneously.

**Example:**
$a + b = 10$
$2a + 2b = 24$

**Applying the Elimination Method:**

1.  **Normalize coefficients (divide by coefficient of 'a'):**
    * Equation 1: $a + b = 10$
    * Equation 2: $a + b = 12$ (dividing $2a + 2b = 24$ by 2)

2.  **Eliminate 'a' from the second equation:**
    * Subtract Equation 1 from Equation 2:
        $(a + b) - (a + b) = 12 - 10$
        $0 = 2$

**Interpretation:**
* The result $0 = 2$ is a **contradiction** (a false statement).
* This indicates that the equations are inconsistent and the system has **no solutions**.

## Solving Systems of Three Equations with Three Variables

Solving a system of three linear equations with three variables is an extension of the method used for two variables. The core idea is to progressively eliminate variables to reduce the system to a simpler form.

**Goal:** Transform the system into a "solved system" where each variable's value is directly given (e.g., $a = X$, $b = Y$, $c = Z$).

**General Strategy (Gaussian Elimination):**

1.  **Isolate the first variable (e.g., 'a') in the first equation, and eliminate it from the other equations.**
    * **Normalize the first column:** Divide each equation by the coefficient of the first variable ('a') in that equation. This ensures the leading coefficient of 'a' in each equation becomes 1.
    * **Eliminate 'a' from subsequent equations:** Subtract the (normalized) first equation from the second equation, and then from the third equation. This will result in new second and third equations that only contain the remaining two variables (e.g., 'b' and 'c').

    *Example (after step 1):*
    $a + \text{some } b + \text{some } c = \text{some constant}$
    $0a + \text{some } b' + \text{some } c' = \text{some constant}'$
    $0a + \text{some } b'' + \text{some } c'' = \text{some constant}''$

2.  **Solve the resulting 2x2 system for the remaining two variables (e.g., 'b' and 'c').**
    * The second and third equations now form a 2x2 system with variables 'b' and 'c'. Apply the same elimination method learned previously:
        * Normalize the leading coefficient of 'b' in the new second equation to 1.
        * Eliminate 'b' from the new third equation by subtracting the normalized second equation from it.
        * This will leave the third equation with only 'c' (or the final variable).

    *Example (after step 2):*
    $a + \text{some } b + \text{some } c = \text{some constant}$
    $0a + b + \text{some } c' = \text{some constant}'$
    $0a + 0b + \text{some } c'' = \text{some constant}''$

3.  **Back-substitution to find all variable values.**
    * **Solve for the last variable:** From the last equation, you can directly solve for the value of the last variable (e.g., 'c').
    * **Substitute back:** Substitute the known value of 'c' into the second equation to solve for 'b'.
    * **Substitute back again:** Substitute the known values of 'b' and 'c' into the first equation to solve for 'a'.

**Example Walkthrough (General Steps):**

Suppose you have a system like:  
Equation 1: $a + 2b - c = 5$  
Equation 2: $2a - b + 3c = 9$  
Equation 3: $3a + b - 2c = 4$

1.  **Isolate 'a' in the first equation, eliminate 'a' from others:**
    * (Already normalized for 'a' in Eq 1 if leading coefficient is 1)
    * Subtract (Eq 1 * 2) from Eq 2:  
        $(2a - b + 3c) - 2(a + 2b - c) = 9 - 2(5)$  
        $2a - b + 3c - 2a - 4b + 2c = 9 - 10$  
        $-5b + 5c = -1$ (New Eq 2')  

    * Subtract (Eq 1 * 3) from Eq 3:  
        $(3a + b - 2c) - 3(a + 2b - c) = 4 - 3(5)$  
        $3a + b - 2c - 3a - 6b + 3c = 4 - 15$  
        $-5b + c = -11$ (New Eq 3')  

    System now is:  
    $a + 2b - c = 5$  
    $-5b + 5c = -1$  
    $-5b + c = -11$  

2.  **Solve the 2x2 system (New Eq 2' and New Eq 3'):**
    * Let's work with:  
        $-5b + 5c = -1$  
        $-5b + c = -11$  

    * Normalize New Eq 2' by dividing by -5: $b - c = 1/5$
    * Subtract normalized New Eq 2' from New Eq 3':  
        $(-5b + c) - (-5b + 5c) = -11 - (-1)$  
        $-4c = -10$  
        $c = \frac{-10}{-4} = 2.5$  

3.  **Back-substitution:**
    * We found $c=2.5$.
    * Substitute $c=2.5$ into the second equation of the reduced 2x2 system (e.g., $-5b + 5c = -1$ or $b - c = 1/5$ from our generic example): $b = 2.7$
    * Substitute $b=2.7$ and $c=2.5$ into the original first equation:  
        $a + 2(2.7) - (2.5) = 5$  
        $a + 5.4 - 2.5 = 5$  
        $a + 2.9 = 5$  
        $a = 2.1$  

The final solution is $a=2.1, b=2.7, c=2.5$.

## Matrix Row Reduction (Gaussian Elimination)

Matrix row reduction, also known as Gaussian elimination, is a method of applying the same manipulations used to solve systems of linear equations directly to the rows of a matrix. The goal is to transform the matrix into a simplified form, from which valuable information can be extracted.

### Key Matrix Forms

**Row Echelon Form (REF)**

A matrix is in **Row Echelon Form** if it satisfies all of the following conditions:

1. **All nonzero rows are above any rows of all zeros.**
   (i.e., if a row is all zeros, it must be below any non-zero row.)

2. **The leading (first nonzero) entry of each nonzero row is strictly to the right of the leading entry of the row above it.**
   (This creates a staircase or "echelon" shape of leading entries moving to the right as you move down.)

3. **The entries below each leading entry (pivot) are all zero.**

$$
\begin{bmatrix}
1 & 2 & 0 & 3 \\
0 & 1 & 4 & -2 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}
$$


For a 2x2 matrix, the REF can have:
1.  **Two 1s on the diagonal:** Corresponds to a non-singular system with a unique solution.

$$
\begin{bmatrix}
1 & x \\
0 & 1
\end{bmatrix}
$$

2.  **One 1 on the diagonal:** Corresponds to a singular system with infinitely many solutions (if no contradiction arises).

$$
\begin{bmatrix}
1 & x \\
0 & 0
\end{bmatrix}
$$

3.  **Zero 1s on the diagonal:** Corresponds to a trivial system (all zeros) or a singular system.

$$
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
$$


**Reduced Row Echelon Form (RREF):**
  * It satisfies all the conditions of Row Echelon Form.
  * Additionally, all entries **above** and below the leading "ones" (pivots) are zero.

**Example (2x2 RREF):**

$$
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### Connecting Systems of Equations to Matrix Forms

When solving a system of linear equations, the steps correspond directly to row operations on the coefficient matrix:

* **Original System:**  
    $5a + b = 17$  
    $4a - 3b = 6$

    **Corresponding Coefficient Matrix:**

$$
\begin{bmatrix}
5 & 1 \\
4 & -3
\end{bmatrix}
$$

* **Intermediate System (after eliminating 'a' from the second equation):**
    $a + 0.2b = 3.4$ (or $5a + b = 17$)  
    $0a - 0.95b = -1.9$

    **Corresponding Matrix (Row Echelon Form):**
    (After appropriate row operations to get leading '1's and zeros below diagonal) (where $x$ would be $0.2$ and the lower right $1$ comes from normalizing $-0.95b = -1.9$ to $b=2$)

$$
\begin{bmatrix}
1 & x \\
0 & 1
\end{bmatrix}
$$

* **Solved System:**
    $a = 3$  
    $b = 2$

    **Corresponding Matrix (Reduced Row Echelon Form):**

$$
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### Row Echelon Form and System Singularity

The Row Echelon Form provides immediate insight into the nature of the system:

* **Non-singular system (unique solution):** The REF will have ones along the entire main diagonal (or more generally, a pivot in every column corresponding to a variable). This implies that a unique solution can be found.
    **Example (2x2):**

$$
\begin{bmatrix}
1 & x \\
0 & 1
\end{bmatrix}
$$

* **Singular system (redundant - infinitely many solutions):** The REF will have at least one row of all zeros (below the main diagonal) corresponding to a trivially true statement ($0=0$).
    **Example (2x2):**  
    Original System: $a + b = 10$, $2a + 2b = 20$  
    REF: This corresponds to $a+b=10$ and $0a+0b=0$.

$$
\begin{bmatrix}
1 & 1 \\
0 & 0
\end{bmatrix}
$$
    

* **Singular system (contradictory - no solutions):** The REF will have a row where all coefficients are zero, but the corresponding constant term is non-zero (e.g., $0 = 5$). This indicates a contradiction.

## Matrix Row Operations

Matrix row operations are the fundamental manipulations applied to the rows of a matrix during Gaussian elimination. A crucial property of these operations is that they **preserve the singularity or non-singularity of a matrix**. This means:
* If you apply them to a singular matrix, the resulting matrix is also singular.
* If you apply them to a non-singular matrix, the resulting matrix is also non-singular.

Let's consider an example matrix:

$$
\begin{bmatrix}
5 & 1 \\
4 & 3
\end{bmatrix}
$$

Its determinant is $(5 \times 3) - (1 \times 4) = 15 - 4 = 11$. Since the determinant is non-zero, this matrix is **non-singular**.

There are three types of elementary row operations:

### 1. Row Switching (Swapping Two Rows)

* **Operation:** Exchange the positions of two rows.
* **Effect on Determinant:** The determinant of the new matrix is the negative of the original determinant.
* **Preserves Singularity:** Yes. If the original determinant was 0, it remains 0 (because $-0 = 0$). If it was non-zero, it remains non-zero.

**Example:**
Original Matrix:

$$
\begin{bmatrix}
5 & 1 \\
4 & 3
\end{bmatrix}
$$

Determinant = 11.

Swap Row 1 and Row 2:

$$
\begin{bmatrix}
4 & 3 \\
5 & 1
\end{bmatrix}
$$

New Determinant = $(4 \times 1) - (3 \times 5) = 4 - 15 = -11$.
(Note: $-11$ is the negative of the original determinant, $11$.)

### 2. Multiplying a Row by a Non-Zero Scalar

* **Operation:** Multiply all elements in a single row by a non-zero constant ($k \neq 0$).
* **Effect on Determinant:** The determinant of the new matrix is $k$ times the original determinant.
* **Preserves Singularity:** Yes. Since $k \neq 0$:
    * If original determinant was 0, $k \times 0 = 0$.
    * If original determinant was non-zero, $k \times (\text{non-zero}) = (\text{non-zero})$.

**Example:**
Original Matrix:

$$
\begin{bmatrix}
5 & 1 \\
4 & 3
\end{bmatrix}
$$

Determinant = 11.

Multiply Row 1 by 10:

$$
\begin{bmatrix}
(5 \times 10) & (1 \times 10) \\
4 & 3
\end{bmatrix}
$$

$$
\begin{bmatrix}
50 & 10 \\
4 & 3
\end{bmatrix}
$$

New Determinant = $(50 \times 3) - (10 \times 4) = 150 - 40 = 110$.
(Note: $110 = 10 \times 11$, which is 10 times the original determinant.)

### 3. Adding a Multiple of One Row to Another Row

* **Operation:** Add a scalar multiple of one row to another row.
* **Effect on Determinant:** The determinant of the new matrix **remains unchanged**.
* **Preserves Singularity:** Yes. Since the determinant remains the same, if it was 0, it stays 0; if it was non-zero, it stays non-zero.

**Example:**
Original Matrix:

$$
\begin{bmatrix}
5 & 1 \\
4 & 3
\end{bmatrix}
$$

Determinant = 11.

Add Row 2 to Row 1 (R1 = R1 + R2):

$$
\begin{bmatrix}
(5+4) & (1+3) \\
4 & 3
\end{bmatrix}
$$

$$
\begin{bmatrix}
9 & 4 \\
4 & 3
\end{bmatrix}
$$

New Determinant = $(9 \times 3) - (4 \times 4) = 27 - 16 = 11$.
(Note: The determinant is the same as the original determinant.)

## The Rank of a Matrix

The **rank of a matrix** measures the "amount of information" a matrix carries, or equivalently, how much information its corresponding system of linear equations provides.

### Intuitive Understanding of Rank

Imagine a system of "sentences" (like equations):
* **System 1:** "The dog is black." and "The cat is orange."
    * Each sentence provides new, independent information.
    * Rank = 2 (two pieces of information).
* **System 2:** "The dog is black." and "The dog is black."
    * The second sentence is redundant; it provides no new information.
    * Rank = 1 (only one independent piece of information).
* **System 3:** "The dog." and "The dog." (incomplete sentences, no information about color)
    * Neither sentence provides useful information about the color.
    * Rank = 0 (zero pieces of information regarding the color).

### Rank in Systems of Linear Equations

The rank of a system of linear equations is the number of independent pieces of information it provides, which helps constrain the solution space.

* **System with a unique solution (Non-singular):**
    * Example: $a + b = 10$, $a + 2b = 12$
    * Each equation narrows down the solution. The first constrains solutions to a line, the second to a single point.
    * Rank = 2 (two independent pieces of information).

* **System with infinitely many solutions (Singular - Redundant):**
    * Example: $a + b = 10$, $2a + 2b = 20$
    * The second equation is a multiple of the first and provides no new information.
    * The solutions are constrained to a line (one degree of freedom).
    * Rank = 1 (one independent piece of information).

* **System with no solutions (Singular - Contradictory):**
    * Example: $a + b = 10$, $2a + 2b = 24$
    * These equations contradict each other, meaning they provide no consistent information leading to a solution.
    * Rank = 0 (zero consistent pieces of information, though this interpretation for contradictory systems can be nuanced).

### Defining the Rank of a Matrix

The **rank of a matrix is defined as the rank of its corresponding system of linear equations.**

* Matrix for System 1 (non-singular): Has Rank 2.
* Matrix for System 2 (singular, redundant): Has Rank 1.
* Matrix for System 3 (all zeros): Has Rank 0.

### Rank-Nullity Theorem (Relationship between Rank and Solution Space Dimension)

There's a fundamental relationship between the rank of a matrix and the dimension of its solution space (also known as the null space or kernel). For a matrix with 'n' columns (which corresponds to 'n' variables in the system):

$\text{Rank}(A) + \text{Dimension of Solution Space} = \text{Number of Columns (n)}$

For a 2x2 matrix (where n=2 variables):
$\text{Rank}(A) + \text{Dimension of Solution Space} = 2$

* **Non-singular Matrix (e.g., $a=0, b=0$ is the only solution for homogeneous system):**
    * Solution space dimension = 0 (a single point).
    * Rank = $2 - 0 = 2$.
    * Such a matrix has **full rank** (rank equals the number of rows/columns).

* **Singular Matrix (e.g., solutions form a line for homogeneous system):**
    * Solution space dimension = 1 (a line).
    * Rank = $2 - 1 = 1$.

* **Singular Matrix (e.g., solutions form a plane for homogeneous system - for a 2x2, this means all elements are zero):**
    * Solution space dimension = 2 (a plane).
    * Rank = $2 - 2 = 0$.

### Full Rank and Non-Singularity

A matrix is **non-singular if and only if it has full rank**.
* **Full rank** means the rank of the matrix is equal to the number of its rows (and columns, for square matrices).
* This implies that every equation in the corresponding system brings a new, independent piece of information, and there is no redundancy.

### Application: Image Compression

* Pixelated images can be represented as matrices where pixel intensities are numbers.
* The rank of this matrix is related to the amount of information (and storage space) needed for the image.
* **Singular Value Decomposition (SVD)** is a powerful technique that can reduce the rank of an image matrix, allowing for significant compression while maintaining visual quality. Lower-rank approximations often appear slightly blurrier but use much less storage.

## Rank of a Matrix

The rank of a matrix is a measure of its "non-singularity" and is closely related to the number of independent equations it represents.

### Rank and Systems of Linear Equations

For an $m \times n$ matrix (e.g., a $3 \times 3$ matrix representing a system of three equations with three unknowns), the rank is defined by the number of independent equations within the system.

* **Independent Equation:** An equation that provides genuinely new information and cannot be derived as a linear combination of the other equations in the system.

**Examples for a $3 \times 3$ system:**

* **System 1 (Rank 3):** All three equations are linearly independent. Each equation contributes new information.
    * This system has 3 independent pieces of information.
    * The matrix representing this system has **Rank 3**.
    * Geometrically, for a $3 \times 3$ system, a rank 3 matrix corresponds to three planes intersecting at a single point (a unique solution).

* **System 2 (Rank 2):** Two of the three equations are linearly independent, and the third can be derived from the other two (e.g., one equation is an average of the other two).
    * This system has 2 independent pieces of information.
    * The matrix representing this system has **Rank 2**.
    * Geometrically, this can correspond to three planes intersecting along a line (infinite solutions) or two parallel planes and one intersecting plane, etc.

* **System 3 (Rank 1):** Only one equation is linearly independent, and the other two are scalar multiples of the first.
    * This system has 1 independent piece of information.
    * The matrix representing this system has **Rank 1**.
    * Geometrically, this can correspond to three planes that are all the same, or parallel.

* **System 4 (Rank 0):** No equation provides any information about the variables (e.g., all coefficients are zero).
    * This system has 0 independent pieces of information.
    * The matrix representing this system has **Rank 0**. (This typically means the matrix is a zero matrix).

### Easier Calculation of Rank

Directly identifying independent equations can be challenging. A simpler method to calculate the rank involves transforming the matrix into its **row echelon form**. The number of non-zero rows in the row echelon form of a matrix is equal to its rank.

## Row Echelon Form and Rank

The row echelon form of a matrix is a simplified version obtained through elementary row operations. It provides valuable information about the matrix, particularly its rank.

### Obtaining Row Echelon Form through Row Operations

The goal of obtaining row echelon form is to create a matrix where:
* The first non-zero element (leading entry or pivot) in each row is a '1'.
* Each leading '1' is to the right of the leading '1' in the row above it.
* Rows consisting entirely of zeros are at the bottom.

**Steps and Examples:**

1.  **For a non-singular matrix (e.g., Rank 2):**

$$
Given A = \begin{bmatrix}
5 & 1 \\
4 & -3
\end{bmatrix}
$$

**Step 1:** Divide each row by its leftmost non-zero coefficient to make the leading entry '1'.
  * Row 1: Divide by 5 $\rightarrow$ [ 1 & 0.2 ]
  * Row 2: Divide by 4 $\rightarrow$ [ 1 & -0.75 ]

$$
\begin{bmatrix}
1 & 0.2 \\
1 & -0.75
\end{bmatrix}
$$

**Step 2:** Create zeros below the leading '1's.
  * Subtract Row 1 from Row 2 to make the bottom-left entry zero.
  * $R_2 \leftarrow R_2 - R_1$

$$
\begin{bmatrix}
1 & 0.2 \\
0 & -0.95
\end{bmatrix}
$$

**Step 3:** Make the leftmost non-zero coefficient of the second row a '1'.
   * Divide Row 2 by -0.95.

$$
\begin{bmatrix}
1 & 0.2 \\
0 & 1
\end{bmatrix}
$$

This is the row echelon form.

2.  **For a singular matrix (e.g., Rank 1):**

$$
Given A = \begin{bmatrix}
5 & 1 \\
10 & 2
\end{bmatrix}
$$

**Step 1:** Divide each row by its leftmost non-zero coefficient.

$$
\begin{bmatrix}
1 & 0.2 \\
1 & 0.2
\end{bmatrix}
$$

**Step 2:** Create zeros below the leading '1's.
   * Subtract Row 1 from Row 2.
   * $R_2 \leftarrow R_2 - R_1$

$$
\begin{bmatrix}
1 & 0.2 \\
0 & 0
\end{bmatrix}
$$

This is the row echelon form. Note the row of zeros.

3.  **For a very singular matrix (e.g., Rank 0):**

$$
Given A = \begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
$$

This matrix is already in row echelon form, as there are no non-zero coefficients to divide by.

$$
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
$$

### Connection Between Rank and Row Echelon Form

The rank of a matrix can be easily determined from its row echelon form:

* The **rank of a matrix is the number of ones on the diagonal of its row echelon form** (or more generally, the number of non-zero rows).

    * Matrix 1 (Rank 2) had two '1's on the diagonal.
    * Matrix 2 (Rank 1) had one '1' on the diagonal.
    * Matrix 3 (Rank 0) had zero '1's on the diagonal.

### Rank and Singularity

There's also a direct relationship between the row echelon form and whether a matrix is singular or non-singular:

* A matrix is **non-singular (invertible)** if and only if its row echelon form has **only ones on the main diagonal and no zeros**, meaning it is the identity matrix or an identity-like upper triangular matrix.
* A matrix is **singular (non-invertible)** if and only if its row echelon form contains at least one **row of all zeros** or a **zero on its main diagonal where a '1' would be expected for an identity matrix**.

## Row Echelon Form (REF) in general

### Definition

A matrix is in row echelon form if it satisfies the following conditions:

* All rows consisting entirely of zeros are at the bottom of the matrix.
* For each non-zero row, the first non-zero entry (called the pivot) is 1. (Note: In some textbooks, pivots can be any non-zero number, but for this class, we will ensure pivots are 1).
* For any two consecutive non-zero rows, the pivot of the lower row is strictly to the right of the pivot of the upper row.
* All entries in a column below a pivot are zero.

### Examples

Consider a system of equations:

$$
\begin{aligned}
ax + by + cz &= d \\
ey + fz &= g \\
hz &= i
\end{aligned}
$$

The corresponding augmented matrix in row echelon form would have a structure similar to:

$$
\begin{bmatrix}
1 & * & * & * \\
0 & 1 & * & * \\
0 & 0 & 1 & *
\end{bmatrix}
$$

where '*' represents any number (zero or non-zero).

### Identifying Pivots

Pivots are the first non-zero entries in each non-zero row of a matrix in row echelon form.

### Rank of a Matrix

The rank of a matrix is equal to the number of pivots (or non-zero rows) in its row echelon form.

* A matrix with 5 pivots has Rank 5.
* A matrix with 3 pivots has Rank 3.

### Row Operations to Achieve REF

The same row operations used to solve systems of equations can be applied to matrices to transform them into row echelon form:

* Swapping two rows.
* Multiplying a row by a non-zero scalar.
* Adding a multiple of one row to another row.

### Examples of Reducing Matrices to REF

* **Example 1:**

$$
\begin{bmatrix}
1 & 1 & 1 \\
1 & 2 & 3 \\
1 & 3 & 6
\end{bmatrix} {\text{ ---- R2 = R2 - R1, R3 = R3 - R1   ---->   }} \begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 2 \\
0 & 2 & 5
\end{bmatrix}
$$

* **Example 2 (Singular Matrix):**

$$
\begin{bmatrix}
1 & 2 & 3 \\
1 & 2 & 3 \\
1 & 2 & 3
\end{bmatrix} {\text{ ---- R2 = R2 - R1, R3 = R3 - R1   ---->   }} \begin{bmatrix}
1 & 2 & 3 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

In this case, the rank is 1, as there is only one pivot.

* **Example 3 (Another Singular Matrix):**

$$
\begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 7 \\
3 & 6 & 10
\end{bmatrix} {\text{ ----   R2 = R2 - 2R1, R3 = R3 - 3R1   ---->   }} \begin{bmatrix}
1 & 2 & 3 \\
0 & 0 & 1 \\
0 & 0 & 1
\end{bmatrix} {\text{----    R3 = R3 - R2   ---->   }} \begin{bmatrix}
1 & 2 & 3 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

In this case, the rank is 2, as there are two pivots.

## Reduced Row Echelon Form

The **reduced row echelon form (RREF)** is an extension of the row echelon form, representing a fully solved system of linear equations.

### Connection to Solving Systems of Equations

Consider the system:  
$5a + b = 17$  
$4a - 3b = 6$

This system can be solved to get $a = 3$ and $b = 2$.
This solution can be expressed as a system:  
$1a + 0b = 3$  
$0a + 1b = 2$

The corresponding matrix representation of this solved system is the reduced row echelon form.

$$
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### Characteristics of Reduced Row Echelon Form

* It must first be in **row echelon form**.
* Each **pivot** (the first non-zero entry in a row) must be a **one**.
* All entries **above** each pivot must be **zero**.

### Converting from Row Echelon Form to Reduced Row Echelon Form

The general method involves using each pivot (which should be 1) to eliminate all non-zero entries above it.

1.  **Ensure Pivots are One:** If the pivots in the row echelon form are not 1, divide each row by its leading coefficient to make the pivots equal to 1.
2.  **Clear Entries Above Pivots:** For each pivot (starting from the leftmost/topmost pivot and working downwards):
* Multiply the row containing the pivot by a suitable scalar.
* Subtract this modified row from the rows above to make the entries directly above the pivot zero.

**Example:**
To convert the following row echelon form matrix to reduced row echelon form:

$$
\\begin{bmatrix}
1 & 2 & -5 & 4 \\
0 & 1 & 3 & 0 \\
0 & 0 & 1 & 0
\\end{bmatrix}
$$

**Step 1: Eliminate '2' in R1C2**
Subtract 2 times Row 2 from Row 1 ($R\_1 \\leftarrow R\_1 - 2R\_2$):

$$
\\begin{bmatrix}
1 & 0 & -11 & 4 \\
0 & 1 & 3 & 0 \\
0 & 0 & 1 & 0
\\end{bmatrix}
$$


**Step 2: Eliminate '-11' in R1C3**
Add 11 times Row 3 to Row 1 ($R\_1 \\leftarrow R\_1 + 11R\_3$):

$$
\\begin{bmatrix}
1 & 0 & 0 & 4 \\
0 & 1 & 3 & 0 \\
0 & 0 & 1 & 0
\\end{bmatrix}
$$

**Step 3: Eliminate '3' in R2C3**
Subtract 3 times Row 3 from Row 2 ($R\_2 \\leftarrow R\_2 - 3R\_3$):

$$
\\begin{bmatrix}
1 & 0 & 0 & 4 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\\end{bmatrix}
$$

This final matrix is the reduced row echelon form.

### Rank of a Matrix

The **rank** of a matrix is equal to the number of **pivots** (or leading ones) in its row echelon form or reduced row echelon form.
  * A matrix with 5 pivots has a rank of 5.
  * A matrix with 3 pivots has a rank of 3.

## Gaussian Elimination Algorithm

**Gaussian elimination** is a classic algorithm used to solve systems of linear equations. It's essentially a formalized version of the elimination method.

### Augmented Matrix

To begin, you create an **augmented matrix**. This is formed by taking the coefficient matrix of your system of equations and adding an extra column on the right, which contains the constant values from the right-hand side of the equations. A vertical line typically separates the coefficients from the constants.

For a system like:  
$2a + 4b - 2c = 1$  
$2a + 7b + 1c = -2$  
$4a + 14b + 0c = -1$

The augmented matrix would be:

$$
\begin{bmatrix}
2 & 4 & -2 & | & 1 \\
2 & 7 & 1 & | & -2 \\
4 & 14 & 0 & | & -1
\end{bmatrix}
$$

Any row operations performed on the matrix (e.g., multiplying a row by a scalar, adding/subtracting rows) must also be applied to the constant column.

### The Process: From Augmented Matrix to Reduced Row Echelon Form

The goal of Gaussian elimination is to transform the augmented matrix into **reduced row echelon form (RREF)**. This involves two main phases:

1.  **Forward Elimination (to Row Echelon Form):**
    * **Select a Pivot:** Start with the top-left element as your first pivot.
    * **Set Pivot to 1:** Use row operations (e.g., multiply the row by the reciprocal of the pivot) to make the pivot equal to 1 (this is only applicable if this element is non zero). If this element is zero, then swap this row with the row which has non zero element below this pivot point -- and then multiply by reciprocal.
    * **Zero Out Elements Below Pivot:** Use row operations (e.g., subtract a multiple of the pivot row from rows below it) to make all elements directly below the pivot equal to 0.
    * **Repeat:** Move to the next pivot along the diagonal (the first non-zero element in the next row) and repeat the process until the matrix is in **row echelon form** (all pivots are 1s, and all elements below pivots are 0s).

2.  **Backward Substitution / Back Elimination (from Row Echelon Form to Reduced Row Echelon Form):**
    * **Work Upwards:** Starting from the last row (bottom-rightmost pivot), use each pivot (which should be 1) to eliminate (make zero) all non-zero entries **above** it in its column. This process is very similar to how you made elements below the pivot zero.
    * **Result:** Once this is complete, the coefficient part of the augmented matrix will be the **identity matrix** (1s on the diagonal, 0s everywhere else), and the constant column will contain the solutions for your variables.

### Singular Cases

Gaussian elimination can also identify if a matrix is **singular** (non-invertible), which implies issues with the system of equations.

* If, during forward elimination, you obtain a **row of all zeros** in the coefficient part of the augmented matrix:
    * **Infinitely Many Solutions:** If the corresponding constant value in that same row is also **zero** ($0a + 0b + 0c = 0$), then the system has **infinitely many solutions**. This equation is always true, implying a dependent system.
    * **No Solution (Contradictory):** If the corresponding constant value in that same row is **non-zero** ($0a + 0b + 0c = \text{non-zero constant}$), then the system has **no solutions**. This is a contradiction (0 cannot equal a non-zero number).

### Summary of Gaussian Elimination

1.  **Create Augmented Matrix:** Combine coefficients and constants.
2.  **Forward Elimination:** Transform the coefficient part into row echelon form (1s on diagonal, 0s below).
3.  **Backward Substitution:** Transform the row echelon form into reduced row echelon form (0s above pivots).
4.  **Read Solutions:** The constant column in the RREF will give the solutions for the variables.
5.  **Handle Singularities:** If a row of zeros appears in the coefficient matrix, check the corresponding constant to determine if there are no solutions or infinitely many solutions.

## Week 3: Vectors and Matrices - Foundations for Machine Learning

This week focuses on vectors and matrices, extending the understanding from systems of linear equations to fundamental algebraic operations and their applications in machine learning.

### Introduction to Vectors and Matrices

* Vectors and matrices are analogous to numbers, supporting operations like addition, multiplication, and potentially inversion.
* They are essential for representing and manipulating datasets in machine learning.

### Linear Regression Revisited with Vectors and Matrices

* **Dataset Representation:**
    * Features ($x_1, x_2, \dots, x_n$) for multiple examples are structured into a **feature matrix** ($X$). Each row represents an example.
    * Target values ($y$) form a **vector of targets**.
* **Linear Regression Model:**
    * The model seeks weights ($w_1, w_2, \dots, w_n$) for each feature and a bias ($b$).
    * The relationship is expressed as a system of linear equations.
    * Using vector and matrix notation, this simplifies to:
        $Y_{hat} = WX + b$
        * $W$ is a vector of weights.
        * $X$ is the feature matrix.
        * $b$ is the bias (a scalar or broadcasted vector).
        * $Y_{hat}$ is the vector of predicted target values.
* While real-world datasets aren't typically solved analytically like simple systems, this linear approximation is a reasonable starting point. Machine learning algorithms iteratively solve these systems to make predictions.

### Neural Networks: A Collection of Linear Models

* Neural networks are powerful models for representing **nonlinear systems**, but at their core, they are a large collection of linear models.
* **Structure:** Neural networks are organized into layers of interconnected "artificial neurons."
* **Input Layer:** Receives input features ($x_1, \dots, x_n$), which can be represented as a vector or matrix $X$.
* **Hidden Layers (and Output Layer):**
    * Each neuron in a layer receives inputs from the previous layer.
    * Inside each neuron, a linear model calculates a weighted sum of its inputs plus a bias:
        $Z = WX + b$
        * $W$ is a vector/matrix of weights specific to that neuron/layer.
        * $X$ (or $A_{\text{previouslayer}}$) is the input vector/matrix from the previous layer.
        * $b$ is the bias term.
    * The result $Z$ is then passed through an **activation function** to produce an output (e.g., $$A = \text{activationfunction}(Z)$$).
    * These outputs ($A$) then become the inputs for the next layer.
    * **Notation:** Superscripts in square brackets (e.g., $W^{[1]}, b^{[1]}, A^{[1]}$) are used to denote the layer number.
* **Matrix/Vector Operations are Key:** Instead of numerous individual equations, linear algebra (vectors, matrices, and tensors – higher-dimensional matrices) is used to efficiently compute the operations within and between layers.

**Key takeaway for this course:** The complexity of neural networks simplifies down to sequences of linear algebraic operations on vectors, matrices, and tensors. Understanding these operations is crucial for comprehending how ML models function.

## Vectors

A **vector** is a simpler array of numbers, typically represented as a single column. Vectors can be visualized as arrows in a multi-dimensional space.

### Components of a Vector

Vectors have two essential components:
1.  **Magnitude (Size/Length):** How long the arrow is.
2.  **Direction:** Which way the arrow points.

### Representing a Vector

A vector is a tuple of numbers. The number of coordinates determines the dimension of the space it lives in.

* **2D Vector:** A vector with two coordinates, e.g., $(4, 3)$, lives in a 2D plane. It represents an arrow starting from the origin and ending at the point $(4, 3)$.
* **3D Vector:** A vector with three coordinates, e.g., $(4, 3, 1)$, lives in a 3D space, pointing to the point $(4, 3, 1)$.
* **n-dimensional Vector:** A vector with $n$ coordinates lives in an $n$-dimensional space.

### Notations for Vectors

Vectors can be written:
* **Horizontally (Row Vector):** $(x_1, x_2, \dots, x_n)$
* **Vertically (Column Vector):**

$$
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

* Components are denoted by subscripts, e.g., $x_2$ for the second component.
* Vectors might be denoted with an arrow ($\vec{x}$) or bold font (**x**), but in this course, plain letters like 'x' are used.
* Square brackets $[x_1, x_2, \dots, x_n]$ or vertical column brackets are common, sometimes emphasizing that a vector can be seen as a skinny matrix.

### Magnitude (Norm) of a Vector

Several ways to measure the "size" or "length" of a vector, called norms, are used.

1.  **L1-Norm (Manhattan/Taxicab Distance):**
    * Analogous to walking along city blocks.
    * For a 2D vector $(a, b)$, the L1-norm is $|a| + |b|$.
    * For an n-dimensional vector $x = (x_1, x_2, \dots, x_n)$:  
        $$||x|| = \sum_{i=1}^{n} |x_i|$$

2.  **L2-Norm (Euclidean Distance):**
    * Analogous to flying in a straight line (helicopter distance).
    * Based on the Pythagorean theorem.
    * For a 2D vector $(a, b)$, the L2-norm is $\sqrt{a^2 + b^2}$.
    * For an n-dimensional vector $x = (x_1, x_2, \dots, x_n)$:  
        $$||x|| = \sqrt{\sum_{i=1}^{n} x_i^2}$$
    * **By default, when "norm" is mentioned without specification, it refers to the L2-norm.** This is because it represents the actual length of the arrow.

### Direction of a Vector

The direction of a vector is determined by the ratios of its coordinates.
* For a 2D vector $(a, b)$, the angle $\theta$ with the horizontal axis can be found using trigonometry:  
    $\tan(\theta) = \frac{b}{a}$  
    $\theta = \arctan(\frac{b}{a})$ (in radians or degrees).
* Vectors can have different magnitudes but point in the same direction if their components are proportional (e.g., $(4, 3)$ and $(2, 1.5)$ point in the same direction).

## Vector Operations

Just like numbers, vectors can be added, subtracted, and multiplied by scalars (numbers) to produce new vectors.

### 1. Vector Addition

* **Operation:** To add two vectors, sum their corresponding components.
* **Requirement:** Vectors must have the same number of components (same dimension).
* **Formal Definition:** If $x = (x_1, x_2, \dots, x_n)$ and $y = (y_1, y_2, \dots, y_n)$, then:
    $x + y = (x_1 + y_1, x_2 + y_2, \dots, x_n + y_n)$
* **Geometric Interpretation:** The sum vector is the diagonal of the parallelogram formed by the two original vectors when placed tail-to-tail at the origin.

**Example:**
If $u = (4, 1)$ and $v = (1, 3)$, then
$u + v = (4+1, 1+3) = (5, 4)$

### 2. Vector Subtraction

* **Operation:** To subtract two vectors, subtract their corresponding components.
* **Requirement:** Vectors must have the same number of components (same dimension).
* **Formal Definition:** If $x = (x_1, x_2, \dots, x_n)$ and $y = (y_1, y_2, \dots, y_n)$, then:
    $x - y = (x_1 - y_1, x_2 - y_2, \dots, x_n - y_n)$
* **Geometric Interpretation:** The difference vector $y - x$ (or $x - y$) represents the vector from the tip of $x$ to the tip of $y$ (or vice-versa). It's the other diagonal of the parallelogram formed by $x$ and $y$.

**Example:**
If $u = (4, 1)$ and $v = (1, 3)$, then
$v - u = (1-4, 3-1) = (-3, 2)$

### Vector Difference as Distance

The difference between two vectors is useful for calculating the "distance" or "dissimilarity" between them using norms:

* **L1 Distance:** The L1-norm of their difference.  
    $||x - y|| = \sum_{i=1}^{n} |x_i - y_i|$
* **L2 Distance:** The L2-norm of their difference.
    $||x - y|| = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$

**Example (Distances):**
If $x = (1, 5)$ and $y = (6, 2)$:
* $x - y = (1-6, 5-2) = (-5, 3)$
* L1 Distance = $||(-5, 3)|| = |-5| + |3| = 5 + 3 = 8$
* L2 Distance = $||(-5, 3)|| = \sqrt{(-5)^2 + 3^2} = \sqrt{25 + 9} = \sqrt{34} \approx 5.83$

### 3. Scalar-Vector Multiplication

* **Operation:** To multiply a vector by a scalar ($\lambda$), multiply each component of the vector by that scalar.
* **Formal Definition:** If $x = (x_1, x_2, \dots, x_n)$ and $\lambda$ is a scalar, then:
    $\lambda x = (\lambda x_1, \lambda x_2, \dots, \lambda x_n)$
* **Geometric Interpretation:**
    * If $\lambda > 1$, the vector is stretched.
    * If $0 < \lambda < 1$, the vector is shrunk.
    * If $\lambda < 0$, the vector is stretched/shrunk and its direction is reversed (reflected about the origin).

**Example:**
If $x = (1, 2)$ and $\lambda = 3$:
$\lambda x = 3(1, 2) = (3 \times 1, 3 \times 2) = (3, 6)$

If $x = (1, 2)$ and $\lambda = -2$:
$\lambda x = -2(1, 2) = (-2 \times 1, -2 \times 2) = (-2, -4)$

## The Dot Product

The **dot product** is a fundamental operation in linear algebra that combines two vectors to produce a single scalar value. It's a compact way to express calculations seen in systems of linear equations.

### Definition and Calculation

The dot product of two vectors is the sum of the products of their corresponding components.

**Example: Cost Calculation**
* **Quantities Vector (column):** (2 apples, 4 bananas, 1 cherry). Q = [2, 4, 1]
* **Prices Vector (column):** (Apples $3, Bananas $5, Cherries $2). P = [3, 5, 2]

To find the total cost, you multiply corresponding quantities by their prices and sum the results:  
$(2 \times 3) + (4 \times 5) + (1 \times 2) = 6 + 20 + 2 = 28$

This operation is the dot product. It's often written with the first vector as a row vector and the second as a column vector for clarity in matrix multiplication context.

**Formal Definition:**
Given two vectors $x = (x_1, x_2, \dots, x_n)$ and $y = (y_1, y_2, \dots, y_n)$ with the same number of components (dimension):
The dot product, denoted as $x \cdot y$ or $\langle x, y \rangle$, is:  
$x \cdot y = x_1 y_1 + x_2 y_2 + \dots + x_n y_n = \sum_{i=1}^{n} x_i y_i$

### Connection to the L2-Norm

A significant relationship exists between the dot product and the L2-norm:
The square of the L2-norm of a vector is equal to the dot product of the vector with itself.

$||x||_2^2 = x \cdot x = x_1^2 + x_2^2 + \dots + x_n^2$
Therefore, $||x||_2 = \sqrt{x \cdot x}$

**Example:**
For vector $(4, 3)$:
Dot product with itself: $(4 \times 4) + (3 \times 3) = 16 + 9 = 25$
L2-norm: $\sqrt{25} = 5$.

### Transpose Operation

The **transpose** operation converts rows into columns and columns into rows. It is denoted by a superscript $T$.

* **Matrix Transpose:**
    * To transpose a matrix, simply transpose each of its columns to become rows, or equivalently, swap rows and columns.
    * If a matrix $A$ has dimensions $m \times n$, its transpose $A^T$ will have dimensions $n \times m$.

$$
A = \begin{bmatrix}
2 & 3 \\
4 & 5 \\
1 & 2
\end{bmatrix}
$$
  
$$
A^T = \begin{bmatrix}
2 & 4 & 1 \\
3 & 5 & 2
\end{bmatrix}
$$

### Dot Product Notation with Transpose

In contexts where the dot product is expected to be a row vector multiplied by a column vector, the transpose is often used explicitly:
$x \cdot y = x^T y$

This notation emphasizes the matrix multiplication aspect, where a $1 \times n$ row vector multiplies an $n \times 1$ column vector to yield a $1 \times 1$ scalar (the dot product).

## Angle Between Vectors and Dot Product

The angle between two vectors is a crucial concept, and it has a direct relationship with the dot product.

### Orthogonal Vectors and Zero Dot Product

* Two vectors are **perpendicular** or **orthogonal** if the angle between them is 90 degrees ($\pi/2$ radians).
* **Key Property:** Two vectors are orthogonal if and only if their dot product is zero.

**Example:**
* Vector $u = (-1, 3)$
* Vector $v = (6, 2)$
* Dot product $u \cdot v = (-1 \times 6) + (3 \times 2) = -6 + 6 = 0$.
    Since the dot product is 0, these vectors are orthogonal.

### Dot Product Formula with Angle

The dot product of two vectors $u$ and $v$ is related to their magnitudes and the cosine of the angle $\theta$ between them:

$u \cdot v = |u| \ |v| \cos(\theta)$

Where:
* $|u|$ is the magnitude (L2-norm) of vector $u$.
* $|v|$ is the magnitude (L2-norm) of vector $v$.
* $\theta$ is the angle between vector $u$ and vector $v$.

**Derivations:**

* **Vector with itself:** If $u = v$, then $\theta = 0$, and $\cos(0) = 1$.
    $u \cdot u = |u| \ |u| \cos(0) = |u|^2 \times 1 = |u|^2$. This confirms the earlier definition of the L2-norm squared.
* **Vectors in the same direction:** If $u$ and $v$ are in the same direction, $\theta = 0$, and $\cos(0) = 1$.
    $u \cdot v = |u| \ |v|$.
* **Perpendicular vectors:** If $u$ and $v$ are perpendicular, $\theta = 90^\circ$, and $\cos(90^\circ) = 0$.
    $u \cdot v = |u| \ |v| \times 0 = 0$. This confirms the property of orthogonal vectors.

### Geometric Interpretation of Dot Product Sign

The sign of the dot product indicates the general direction of one vector relative to another:

* **Positive Dot Product ($u \cdot v > 0$):**
    * This occurs when $\cos(\theta) > 0$, meaning the angle $\theta$ is acute ($0^\circ \le \theta < 90^\circ$).
    * The vectors generally point in the same direction.
    * The projection of one vector onto the other has a positive length (points in the same direction as the projected-onto vector).

* **Zero Dot Product ($u \cdot v = 0$):**
    * This occurs when $\cos(\theta) = 0$, meaning $\theta = 90^\circ$.
    * The vectors are orthogonal (perpendicular).

* **Negative Dot Product ($u \cdot v < 0$):**
    * This occurs when $\cos(\theta) < 0$, meaning the angle $\theta$ is obtuse ($90^\circ < \theta \le 180^\circ$).
    * The vectors generally point in opposite directions.
    * The projection of one vector onto the other has a negative length (points in the opposite direction of the projected-onto vector).

**Visual Summary:**

<img src="/metadata/gp_dot.png" width="700" />

For a given vector $u$:
* Vectors orthogonal to $u$ lie on a line perpendicular to $u$ through the origin (dot product = 0).
* Vectors on the "side" of $u$ (acute angle) have a positive dot product.
* Vectors on the "opposite side" of $u$ (obtuse angle) have a negative dot product.

Geometric definition of a dot product is used in one of the applications - to evaluate **vector similarity**. In Natural Language Processing (NLP) words or phrases from vocabulary are mapped to a corresponding vector of real numbers. Similarity between two vectors can be defined as a cosine of the angle between them. When they point in the same direction, their similarity is 1 and it decreases with the increase of the angle. 

Then equation can be rearranged to evaluate cosine of the angle between vectors:

$\cos(\theta)=\frac{x \cdot y}{\lvert x\rvert \lvert y\rvert}\tag{3}$

Zero value corresponds to the zero similarity between vectors (and words corresponding to those vectors). Largest value is when vectors point in the same direction, lowest value is when vectors point in the opposite directions.

## Matrices as Linear Transformations

* Matrices can represent linear transformations, which map points from one space to another in a structured way.
* For a 2x2 matrix, this transforms points in a 2D plane to new points in another 2D plane.

### How a Linear Transformation Works

* Given a matrix $M$ and a column vector $v$ (representing a point's coordinates), the transformed point $v'$ is found by the matrix-vector multiplication: $v' = Mv$.
* **Example:** For a matrix

$$
\begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix}
$$

The origin (0, 0) always maps to the origin (0, 0) in a linear transformation.
Point (1, 0) maps to (3, 1):

$$
\begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix} \cdot \begin{bmatrix}
1 \\
0
\end{bmatrix} = \begin{bmatrix}
3 \\
1
\end{bmatrix}
$$

Point (0, 1) maps to (1, 2):

$$
\begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix} \cdot \begin{bmatrix}
0 \\
1
\end{bmatrix} = \begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

Point (1, 1) maps to (4, 3):

$$
\begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix} \cdot \begin{bmatrix}
1 \\
1
\end{bmatrix} = \begin{bmatrix}
4 \\
3
\end{bmatrix}
$$

### Basis and Tessellation

<img src="/metadata/mat_lt.png" width="700" />

* The initial square formed by points like (0,0), (1,0), (0,1), (1,1) is called a **basis**.
* This basis transforms into a parallelogram under the linear transformation.
* Both the original square and the transformed parallelogram tessellate (cover) the entire plane.
* A linear transformation can be viewed as a **change of coordinates**.

### Example of Change of Coordinates

<img src="/metadata/mat_lt_2.png" width="700" />

* To find where a point like (-2, 3) goes:
    * In the original plane, (-2, 3) means moving 2 units left and 3 units up from the origin.
    * In the transformed plane, we apply the same "steps" but using the new basis vectors (the transformed (1,0) and (0,1) vectors) as the new "unit steps."
    * So, -2 times the transformed (1,0) vector plus 3 times the transformed (0,1) vector gives the new point:

### Analogy: Buying Apples and Bananas

* Imagine a scenario:
    * Day 1: Buy 3 apples, 1 banana.
    * Day 2: Buy 1 apple, 2 bananas.
* If 'a' is the price of an apple and 'b' is the price of a banana, the total cost for each day can be represented by a matrix multiplication:

$$
\begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix} \cdot \begin{bmatrix}
a \\
b
\end{bmatrix}
$$

* This linear transformation maps a point (a, b) representing apple and banana prices to a point representing the total cost for Day 1 and Day 2.
* If a = $1 and b = $1, then the matrix transforms (1, 1) to (4, 3):

$$
\begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix} \cdot \begin{bmatrix}
1 \\
1
\end{bmatrix} = \begin{bmatrix}
4 \\
3
\end{bmatrix}
$$

Day 1 cost: $4. Day 2 cost: $3

* This illustrates how the matrix facilitates a change of coordinates from ingredient prices to daily expenses.

## Linear Transformation to Matrix

### Finding the Matrix from Transformed Basis Vectors

* To determine the matrix of a linear transformation, the key is to observe how the **basis vectors** are transformed.
* In a 2D space, the standard basis vectors are:

$$
\begin{bmatrix}
1 \\
0
\end{bmatrix} \text{unit vector along the x-axis}
$$

$$\begin{bmatrix}
0 \\
1
\end{bmatrix} \text{unit vector along the y-axis}
$$

* If a linear transformation maps:

$$
\text{The vector} \begin{bmatrix}
1 \\
0
\end{bmatrix} \text{to a new vector} \begin{bmatrix}
a \\
b
\end{bmatrix}
$$

$$
\text{The vector} \begin{bmatrix}
0 \\
1
\end{bmatrix} \text{to a new vector} \begin{bmatrix}
c \\
d
\end{bmatrix}
$$

* Then, the corresponding transformation matrix is constructed by using these transformed vectors as its columns:

$$
\begin{pmatrix}
a & c \\
b & d
\end{pmatrix}
$$

## Matrix-Matrix Multiplication

Matrix multiplication represents the combination of two linear transformations into a third one.

### Visualizing Matrix Multiplication as Combined Transformations

<img src="/metadata/combining_lt.png" width="700" />

Imagine two successive linear transformations:
* Transformation 1 (Matrix A): Maps basis vectors to new positions.
* Transformation 2 (Matrix B): Maps the output basis vectors of Transformation 1 to yet another set of positions.

The combined effect from the initial basis to the final positions is a single linear transformation, which corresponds to the product of the two matrices.

**Important Note on Order:** If Transformation 1 is represented by matrix $A$ and Transformation 2 by matrix $B$, the combined transformation is represented by $BA$. This is because the transformations are applied sequentially, with the matrix closer to the vector being applied first (e.g., $B(Av)$).

### Scaling
Horizontal scaling can be defined, for example, considering transformation of a vector(x,y) [1 0] into a vector [2 0] and keeping [0 1] to [0 1] only. So transformation matrix here will be

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix} \cdot \begin{bmatrix}
1 \\
0
\end{bmatrix} = \begin{bmatrix}
2 \\
0
\end{bmatrix}
$$

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix} \cdot \begin{bmatrix}
0 \\
1
\end{bmatrix} = \begin{bmatrix}
0 \\
1
\end{bmatrix}
$$

Solving this, we get 

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix} = \begin{bmatrix}
2 & 0 \\
0 & 1
\end{bmatrix}
$$

Using same way, transformation matrix for reflection about y-axis, reflection about x-axis, stretching etc can be derived.

### Rotation
To rotate a vector in the plane by an angle of $\theta$ (radians), the matrix related to this transformation is given by:

$$
M = \begin{bmatrix}
\cos \theta & - \sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
$$

## Calculating Matrix-Matrix Products

Matrix multiplication involves a series of dot products.

To find an element in the resulting matrix at position $(i, j)$ (row $i$, column $j$), you take the dot product of the $i$-th row of the first matrix and the $j$-th column of the second matrix.

**Example:**
To multiply matrix

$$
\text{A = } \begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix} \text{and B = } \begin{bmatrix}
2 & -1 \\
0 & 2
\end{bmatrix}
$$

$$
\begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix} \cdot \begin{bmatrix}
2 & -1 \\
0 & 2
\end{bmatrix} = \begin{bmatrix}
(3 \times 2) + (1 \times 0) & (3 \times -1) + (1 \times 2) \\
(1 \times 2) + (2 \times 0) & (1 \times -1) + (2 \times 2)
\end{bmatrix} = \begin{bmatrix}
6 + 0 & -3 + 2 \\
2 + 0 & -1 + 4
\end{bmatrix} = \begin{bmatrix}
6 & -1 \\
2 & 3
\end{bmatrix}
$$

**General Rule:**
If matrix A has dimensions $m \times n$ and matrix B has dimensions $n \times p$:
* The number of columns of the first matrix ($n$) must match the number of rows of the second matrix ($n$).
* The resulting matrix will have dimensions $m \times p$. Its number of rows comes from the first matrix ($m$), and its number of columns comes from the second matrix ($p$).

**Example with non-square matrices:**
Multiplying a $2 \times 3$ matrix by a $3 \times 4$ matrix will result in a $2 \times 4$ matrix.

$$
\text{A = } \begin{bmatrix}
3 & 1 & 4 \\
2 & -1 & 2
\end{bmatrix}
$$

$$
\text{B = } \begin{bmatrix}
3 & 0 & 5 & -1 \\
1 & 5 & 2 & 3 \\
-2 & 1 & 0 & 2
\end{bmatrix}
$$

The element at the bottom-left cell (row 2, column 1) of the resulting matrix is calculated by taking the dot product of the second row of A and the first column of B:
$(2 \times 3) + (-1 \times 1) + (2 \times -2) = 6 - 1 - 4 = 1$.

## The Identity Matrix

The identity matrix, denoted as $I$, plays a role in matrix multiplication similar to the number '1' in scalar multiplication.

* **Definition:** When multiplied by any other matrix $A$, the identity matrix leaves $A$ unchanged ($IA = AI = A$).
* **Linear Transformation:** The linear transformation corresponding to the identity matrix is one that leaves the space (e.g., the plane) intact. Every point is mapped precisely to itself.
* **Structure:** The identity matrix has '1's along its main diagonal (from top-left to bottom-right) and '0's everywhere else.

* For a $2 \times 2$ matrix:

$$
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

* For a $3 \times 3$ matrix:

$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

* **How it Works (Example with a vector):**
 When multiplied by a vector

$$
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} \cdot \begin{bmatrix}
a \\
b
\end{bmatrix} = \begin{bmatrix}
(1 \times a) + (0 \times b) \\
(0 \times a) + (1 \times b)
\end{bmatrix} = \begin{bmatrix}
a \\
b
\end{bmatrix}
$$

## The Inverse of a Matrix

The inverse of a matrix, denoted as $A^{-1}$, is analogous to the reciprocal of a number.

* **Definition:** For a square matrix $A$, its inverse $A^{-1}$ is a matrix such that when multiplied by $A$, it yields the identity matrix $I$.
    * $A \cdot A^{-1} = A^{-1} \cdot A = I$

* **Linear Transformation:** In terms of linear transformations, the inverse matrix $A^{-1}$ represents the transformation that "undoes" the job of the original matrix $A$. If $A$ transforms a plane in a certain way, $A^{-1}$ transforms it back to its original state.

### Finding the Inverse of a $2 \times 2$ Matrix

To find the inverse of a matrix, we solve a system of linear equations.

Given a matrix

$$
\text{A = } \begin{bmatrix}
p & q \\
r & s
\end{bmatrix} \text{and its inverse } A^{-1} \text{ = } \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

$$
\begin{bmatrix}
p & q \\
r & s
\end{bmatrix} \cdot \begin{bmatrix}
a & b \\
c & d
\end{bmatrix} = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

This expands into four linear equations:
* $pa + qc = 1$
* $pb + qd = 0$
* $ra + sc = 0$
* $rb + sd = 1$

Solving this system for $a, b, c, d$ will give the entries of the inverse matrix.

**Example 1:** Find the inverse of 

$$\begin{bmatrix}
3 & 1 \\
1 & 2
\end{bmatrix}
$$

The system of equations is:
1.  $3a + 1c = 1$
2.  $3b + 1d = 0$
3.  $1a + 2c = 0$
4.  $1b + 2d = 1$

Solving these equations yields: $a = \frac{2}{5}, b = -\frac{1}{5}, c = -\frac{1}{5}, d = \frac{3}{5}$.

$$
\text{So, } A^{-1} \text{ = } \begin{bmatrix}
\frac{2}{5} & -\frac{1}{5} \\
-\frac{1}{5} & \frac{3}{5}
\end{bmatrix}
$$

### When an Inverse Does Not Exist

Not all matrices have an inverse. This occurs when the system of linear equations derived from the inverse definition leads to a contradiction.

**Example:** Consider the matrix 

$$
\begin{bmatrix}
1 & 2 \\
2 & 4
\end{bmatrix}
$$

The system of equations for its inverse would be:
1.  $1a + 2c = 1$
2.  $1b + 2d = 0$
3.  $2a + 4c = 0$
4.  $2b + 4d = 1$

From equation (1), $a + 2c = 1$.
From equation (3), $2a + 4c = 0$, which simplifies to $2(a + 2c) = 0$, meaning $a + 2c = 0$.

This is a contradiction ($1 = 0$), indicating that no solution exists for $a, b, c, d$. Therefore, the matrix does not have an inverse.

## Invertible and Non-Invertible Matrices

Just like numbers, not all matrices have a multiplicative inverse.

* **Analogy to Numbers:**
    * Any non-zero number has a multiplicative inverse (e.g., the inverse of 5 is 1/5).
    * The number zero does *not* have a multiplicative inverse (there's no number you can multiply by zero to get 1).

* **Matrices:**
    * Some matrices have an inverse (called **invertible matrices**).
    * Some matrices do not have an inverse (called **non-invertible matrices**).

### The Rule for Matrix Invertibility

The rule for whether a matrix has an inverse is directly related to its **determinant** and whether it is **singular** or **non-singular**.

* **Non-Singular Matrices:**
    * Are also called **invertible matrices**.
    * Always have an inverse.
    * Their determinant is **non-zero** ($\det(A) \neq 0$).
    * This is analogous to non-zero numbers having an inverse.

* **Singular Matrices:**
    * Are also called **non-invertible matrices**.
    * Never have an inverse.
    * Their determinant is **zero** ($\det(A) = 0$).
    * This is analogous to zero not having an inverse.

**In summary:**

* **Non-singular matrix** ⇔ **Determinant ≠ 0** ⇔ **Inverse exists**
* **Singular matrix** ⇔ **Determinant = 0** ⇔ **Inverse does not exist**

## Machine Learning Application: Neural Networks

Neural networks, a powerful machine learning model, are fundamentally built upon matrices and matrix products.

### Simple Neural Network: Spam Classifier (Perceptron)

Let's consider a spam classification problem using a simple neural network, often called a perceptron.

* **Problem:** Classify emails as "spam" or "not spam" based on the frequency of certain words, e.g., "lottery" and "win."
* **Data Representation:** Each email can be represented as a vector (or a row in a matrix) where entries correspond to the count of specific words.
    * Example: `[count_of_lottery, count_of_win]`
* **Classifier Model:**
    * Assign a "score" (weight) to each word (e.g., `score_lottery`, `score_win`).
    * Calculate a sentence score: `(count_of_lottery * score_lottery) + (count_of_win * score_win)`.
    * Apply a **threshold**: If the sentence score is greater than the threshold, classify as "spam"; otherwise, "not spam."

**Example Walkthrough:**
Consider a model where `score_lottery = 1`, `score_win = 1`, and `threshold = 1.5`.

| Email | Lottery Count | Win Count | Spam (Actual) | Sentence Score (Dot Product) | Prediction (>1.5?) | Correct? |
| :---- | :------------ | :-------- | :------------ | :--------------------------- | :----------------- | :------- |
| 1     | 2             | 1         | Spam          | $(2 \times 1) + (1 \times 1) = 3$ | Spam               | Yes      |
| 2     | 0             | 1         | Not Spam      | $(0 \times 1) + (1 \times 1) = 1$ | Not Spam           | Yes      |

* This process of multiplying counts by scores and summing them is a **dot product**.
* If we have multiple emails (a dataset), we can represent the email word counts as a matrix and the word scores as a vector. The predictions for all emails can then be calculated simultaneously using **matrix-vector multiplication**.

$$
\begin{bmatrix}
2 & 1 \\
0 & 1 \\
\vdots & \vdots
\end{bmatrix} \cdot \begin{bmatrix}
1 \\
1
\end{bmatrix} = \begin{bmatrix}
(2 \times 1) + (1 \times 1) \\
(0 \times 1) + (1 \times 1) \\
\vdots
\end{bmatrix} = \begin{bmatrix}
3 \\
1 \\
\vdots
\end{bmatrix}
$$

The resulting vector contains the scores for each email, which are then compared against the threshold.

### Graphical Representation: Linear Separator

This type of classifier can be visualized graphically:
* Plot emails on a 2D plane: X-axis = "Lottery Count", Y-axis = "Win Count".
* The classification rule (score > threshold) defines a **line** that separates the "spam" points from the "not spam" points.
    * The equation of this line is `(score_lottery * X) + (score_win * Y) = threshold`.
    * One side of the line is the "positive region" (spam), and the other is the "negative region" (not spam).
* This is a **linear classifier**.

### Bias Term

Instead of a threshold, a **bias** term can be incorporated:
* The classification rule becomes: `(score_lottery * X) + (score_win * Y) - bias > 0`.
* This can be integrated into the matrix multiplication by:
    * Adding an extra column of `1`s to the input data matrix.
    * Adding the bias term as an extra entry in the model vector (e.g., `-threshold`).
* This allows checking if the result is positive or negative, rather than comparing to a threshold.

### The AND Perceptron

The "AND" logical operator can also be modeled as a single-layer neural network (perceptron).
* **Dataset:**
    | X | Y | AND (Output) |
    | :- | :- | :----------- |
    | 0 | 0 | 0            |
    | 0 | 1 | 0            |
    | 1 | 0 | 0            |
    | 1 | 1 | 1            |
* Using weights (scores) of `1` for X and `1` for Y, and a threshold of `1.5`:
    * `[X, Y] . [1, 1] = Score`
    * If `Score > 1.5`, output 1 (True); else output 0 (False).
* This perfectly replicates the AND logic.

### Perceptron Diagram

A common way to represent a perceptron is:
* **Inputs:** `x1, x2, ..., xn` (e.g., word counts).
* **Weights:** `w1, w2, ..., wn` (e.g., word scores).
* **Bias:** `b` (or ` -threshold`).
* **Node (Summation):** Calculates the weighted sum of inputs plus the bias: `z = (x1*w1 + x2*w2 + ... + xn*wn) + b`. This is a dot product plus a bias.
* **Activation Function:** Applies a non-linear function to `z` to produce the output. For a simple perceptron, this is often a step function:
    * Returns 1 (or "Yes") if `z >= 0`.
    * Returns 0 (or "No") if `z < 0`.

This structure highlights how linear algebra concepts like dot products and matrix multiplication are fundamental to neural networks.

