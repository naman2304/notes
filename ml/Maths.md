Appendix
* [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science)
  * Linear Algebra for Machine Learning and Data Science
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

Understood. I will revert to using `$` for inline equations and `$$` for display-block equations, without any HTML tags for centering, as GitHub's rendering environment for MathJax/LaTeX often overrides or ignores them.

Here are the notes:

## Week 1: Systems of Linear Equations

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
\begin{pmatrix}
1 & 1 \\
1 & 2
\end{pmatrix}
$$

**System 2:** $a + b = 0$, $2a + 2b = 0$

$$
\begin{pmatrix}
1 & 1 \\
2 & 2
\end{pmatrix}
$$

### Matrix Singularity

* A matrix is classified as **singular** or **non-singular** based on the singularity of its corresponding system of linear equations.
* If the system has a **unique solution** (non-singular), its matrix is **non-singular**.
* *Example:* Following is non-singular because $a+b=0, a+2b=0$ has only $(0,0)$ as a solution.

$$
\begin{pmatrix}
1 & 1 \\
1 & 2
\end{pmatrix}
$$

* If the system has **infinitely many solutions** or **no solutions** (singular), its matrix is **singular**.
* *Example:* Following is singular because $a+b=0, 2a+2b=0$ has infinitely many solutions.

$$
\begin{pmatrix}
1 & 1 \\
2 & 2
\end{pmatrix}
$$

### 3x3 Systems and their Matrices

Similar principles apply to systems with more variables and larger matrices.

* **System 1 (Non-Singular):**
* Example: $a+b+c=0$, $a+2b+c=0$, $a+b+2c=0$.
* Unique solution $(0,0,0)$.
* Corresponding Matrix (non-singular):

$$
\begin{pmatrix}
1 & 1 & 1 \\
1 & 2 & 1 \\
1 & 1 & 2
\end{pmatrix}
$$

* **System 2 & 3 (Singular - after setting constants to zero):**
    * Example: $a+b+c=0$, $a+b+2c=0$, $a+b+3c=0$ (originally had different constants, but becomes this when constants are zero).
    * Leads to $c=0$ and $a=-b$, thus infinitely many solutions where $a+b=0, c=0$.
    * Corresponding Matrix (singular):

$$
\begin{pmatrix}
1 & 1 & 1 \\
1 & 1 & 2 \\
1 & 1 & 3
\end{pmatrix}
$$

* **System 4 (Singular):**
    * Example: $a+b+c=0$, $2a+2b+2c=0$, $3a+3b+3c=0$.
    * All equations are multiples of the first, leading to infinitely many solutions where $a+b+c=0$.
    * Corresponding Matrix (singular):

$$
\begin{pmatrix}
1 & 1 & 1 \\
2 & 2 & 2 \\
3 & 3 & 3
\end{pmatrix}
$$

**Note:** Quicker methods exist to determine matrix singularity without solving the system (e.g., using determinants, which will be covered later).

## Linear Dependence and Independence of Rows

This concept provides a direct way to determine if a matrix (and its corresponding system of linear equations) is singular or non-singular, without solving the system.

### Linear Dependence

A set of rows (or equations) is **linearly dependent** if at least one row can be expressed as a **linear combination** of the others, meaning that row does not provide new information. One row can be a sum or weighted sum of others.

* **Example 1:** For the system $a = 1$, $b = 2$, $a+b = 3$, the coefficient matrix (assuming constants are zero for singularity check) is following. Here, Row 3 is Row 1 + Row 2. This indicates linear dependence, implying the matrix is **singular**.

$$
\begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
1 & 1 & 0
\end{pmatrix}
$$

* **Example 2:** For following matrix, Row 2 is $2 \times$ Row 1, and Row 3 is $3 \times$ Row 1. This demonstrates linear dependencies. This implies the matrix is **singular**.

$$
\begin{pmatrix}
1 & 1 & 1 \\
2 & 2 & 2 \\
3 & 3 & 3
\end{pmatrix}
$$

* **Example 3:** For following matrix, Row 2 is the average of Row 1 and Row 3 ($2 \times \text{Row 2} = \text{Row 1} + \text{Row 3}$). This implies the rows are linearly dependent, and the matrix is **singular**.

$$
\begin{pmatrix}
1 & 1 & 1 \\
1 & 1 & 2 \\
1 & 1 & 3
\end{pmatrix}
$$

### Linear Independence

A set of rows (or equations) is **linearly independent** if no row can be expressed as a linear combination of the others. Each row provides unique information.

* **Example:** For the system $a + b = 0$, $a + 2b = 0$, the matrix is following. Here, Row 2 is not a multiple of Row 1. This implies the rows are linearly independent, and the matrix is **non-singular** (system has a unique solution).

$$
\begin{pmatrix}
1 & 1 \\
1 & 2
\end{pmatrix}
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
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$

The determinant is calculated as: $det(A) = ad - bc$
* $ad$ is the product of the numbers in the **main diagonal**.
* $bc$ is the product of the numbers in the **antidiagonal**.

**Example:**

$$
\begin{pmatrix}
1 & 1 \\
1 & 2
\end{pmatrix}
$$

Determinant = $(1 \times 2) - (1 \times 1) = 2 - 1 = 1$. Since $1 \neq 0$, the matrix is non-singular.

$$
\begin{pmatrix}
1 & 2 \\
1 & 2
\end{pmatrix}
$$

Determinant = $(1 \times 2) - (2 \times 1) = 2 - 2 = 0$. Since $0$, the matrix is singular.

### Determinant of a 3x3 Matrix

For a 3x3 matrix:

$$
\begin{pmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{pmatrix}
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
\begin{pmatrix}
1 & 1 & 1 \\
1 & 2 & 1 \\
1 & 1 & 2
\end{pmatrix}
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
\begin{pmatrix}
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 3
\end{pmatrix}
$$

Determinant = $1 \times 2 \times 3 = 6$.
Even if an upper triangular matrix has a zero on its main diagonal, its determinant will be zero, indicating it is singular.

**Example:**

$$
\begin{pmatrix}
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 0
\end{pmatrix}
$$

Determinant = $1 \times 2 \times 0 = 0$. This matrix is singular.
