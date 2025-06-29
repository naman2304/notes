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
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix}
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
A = \begin{pmatrix}
2 & 3 \\
4 & 5 \\
1 & 2
\end{pmatrix}
$$
  
$$
A^T = \begin{pmatrix}
2 & 4 & 1 \\
3 & 5 & 2
\end{pmatrix}
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
