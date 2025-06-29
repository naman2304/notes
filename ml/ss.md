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
