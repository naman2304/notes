## Training Neural Networks: An Overview

Welcome to Week 2! This week focuses on **training neural networks**, building on last week's understanding of inference.

### Training Process in TensorFlow

Here's a high-level overview of the TensorFlow code for training a neural network:

1.  **Define the Model Architecture:**
    This step sets up the layers of your neural network sequentially. It's familiar from last week's inference discussion.
    * Example:
        ```python
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential([
            Dense(units=25, activation='sigmoid'), # Layer 1 (Hidden)
            Dense(units=15, activation='sigmoid'), # Layer 2 (Hidden)
            Dense(units=1, activation='sigmoid')   # Layer 3 (Output)
        ])
        ```
    * This defines how the network computes its output given an input.

2.  **Compile the Model:**
    This step configures the model for the training process. The key part here is specifying the **loss function**.
    * Example:
        ```python
        from tensorflow.keras.losses import BinaryCrossentropy
        model.compile(loss=BinaryCrossentropy())
        ```
    * `loss=BinaryCrossentropy()`: This tells TensorFlow to use the binary cross-entropy loss function, which is suitable for binary classification tasks. More details on this loss function will be covered in the next video.

3.  **Train (Fit) the Model:**
    This step executes the training algorithm, fitting the model to your data.
    * Example:
        ```python
        model.fit(X, Y, epochs=100) # X: input data, Y: true labels
        ```
    * `X, Y`: Your training dataset.
    * `epochs=100`: Specifies how many steps (iterations of the underlying optimization algorithm, like gradient descent) to run the training for.

### Why Understand Beyond the Code:

While these few lines of code are powerful, understanding what happens "behind the scenes" is crucial for:

* **Debugging:** Effectively identifying and fixing issues when your model doesn't perform as expected.
* **Problem Solving:** Developing a conceptual framework to approach complex machine learning challenges.

The next video will delve deeper into the details of these TensorFlow steps, starting with the `compile` function and the binary cross-entropy loss.

## Training Neural Networks in Detail (TensorFlow)

This video delves into the details of how TensorFlow trains neural networks, drawing parallels with the logistic regression training process from Course 1.

### Recap: Logistic Regression Training Steps

1.  **Specify the Model (f(x)):** Define how the output is computed from inputs (x) and parameters (w, b). For logistic regression, $f(x) = g(\vec{w} \cdot \vec{x} + b)$.
2.  **Specify Loss and Cost Functions:**
    * **Loss:** Measures performance on a single example. For binary classification, it was the binary cross-entropy loss: $L(f(x), y) = -y \log(f(x)) - (1-y) \log(1-f(x))$.
    * **Cost:** Average of the loss over the entire training set $J(W, B) = \frac{1}{m} \sum L(f(x^{(i)}), y^{(i)})$.
3.  **Minimize the Cost Function:** Use an optimization algorithm (e.g., gradient descent) to find optimal parameters (w, b) by repeatedly updating them based on the derivatives of J.

### Neural Network Training in TensorFlow: Three Steps

The same three conceptual steps apply to training neural networks in TensorFlow:

1.  **Specify the Model Architecture (using `tf.keras.Sequential`):**
    * This defines the layers (e.g., `Dense` layers with specific units and activation functions) and their connections. This fully specifies how the network computes its output $f(x)$ (or $a^{[L]}$) given an input $x$ and its parameters (all $W^{[l]}$ and $B^{[l]}$).

2.  **Compile the Model (Specifying Loss & Optimizer):**
    * This step tells TensorFlow *how* to train the model.
    * **Loss Function:** For binary classification (like handwritten digit recognition), the most common loss is `tf.keras.losses.BinaryCrossentropy()`. This is the same cross-entropy loss function used for logistic regression.
    * For **regression problems**, you would use a different loss, such as `tf.keras.losses.MeanSquaredError()`.
    * **Cost Function:** TensorFlow implicitly uses the average of the specified loss function over the entire training set as the cost function it aims to minimize.
    * Note: There is just one cost function for the entire neural network, not per neuron. It's for the final one neuron on the final layer to be precise.

3.  **Train the Model (using `model.fit()`):**
    * This is where the actual optimization happens.
    * `model.fit(X, Y, epochs=...)`: Takes your input training data `X` and ground truth labels `Y`.
    * `epochs`: Specifies the number of training iterations (similar to gradient descent steps).
    * **Under the Hood:** `model.fit()` uses an optimization algorithm (like gradient descent, or more advanced variants) to minimize the specified cost function. Crucially, it employs **backpropagation** to efficiently compute the necessary partial derivatives of the cost function with respect to all network parameters (all $W^{[l]}$ and $B^{[l]}$ across all layers).

### Evolution of ML Implementation: Libraries

Historically, developers implemented ML algorithms from scratch. However, modern deep learning libraries like TensorFlow and PyTorch are highly mature. They abstract away complex details like backpropagation, allowing developers to train powerful models with just a few lines of code. While using these libraries is standard practice, understanding the underlying principles (forward propagation, loss/cost functions, gradient descent, backpropagation) is vital for debugging and effectively utilizing these tools.

The next video will explore different **activation functions** that can be used in neural networks, moving beyond just the Sigmoid function to enhance model power.

## Beyond Sigmoid: Other Activation Functions

So far, we've primarily used the Sigmoid activation function, which outputs values between 0 and 1, suitable for probabilities. However, neural networks become much more powerful with other activation functions.

### The Need for Different Activation Functions

In the demand prediction example, modeling "awareness" as a binary 0/1 (or 0-1 probability) might be too restrictive. Awareness could be non-binary and potentially very high (e.g., viral). If we want an activation that can take on any non-negative value, Sigmoid isn't suitable.

### Rectified Linear Unit (ReLU)

* **Formula:** $g(z) = \max(0, z)$
* **Graph:** Outputs 0 for any negative input $z$, and outputs $z$ itself for any positive $z$.
* **Properties:** Allows activations to be any non-negative number, which is useful for modeling quantities that can grow indefinitely (e.g., "awareness" or "virality").
* **Common Name:** ReLU (pronounced "ray-loo"), short for Rectified Linear Unit. This is a very common choice in modern neural networks.

### Linear Activation Function

* **Formula:** $g(z) = z$
* **Properties:** The output is simply equal to the input $z$.
* **Usage:** Often used in the output layer of neural networks for **regression problems** (where you want to predict a continuous number rather than a probability or category).
* **Terminology:** Sometimes referred to as "no activation function" because it doesn't transform $z$ in a non-linear way, but in this course, it's called the "linear activation function."

### Summary of Common Activation Functions:

1.  **Sigmoid:** $g(z) = 1 / (1 + e^{-z})$ (Outputs 0 to 1; good for output layer probabilities in binary classification).
2.  **ReLU:** $g(z) = \max(0, z)$ (Outputs 0 to positive infinity; common in hidden layers, allows for non-linear relationships and avoids vanishing gradients).
3.  **Linear:** $g(z) = z$ (Outputs negative infinity to positive infinity; good for output layer in regression problems).

These three, along with the Softmax activation function (to be covered later), form the basis for building a wide variety of powerful neural networks. The next video will discuss how to choose among these different activation functions.

## Choosing Activation Functions

Selecting the right activation function for different layers is crucial for neural network performance.

### Output Layer Activation Function

The choice of output layer activation depends on the type of prediction:

* **Binary Classification (y is 0 or 1):**
    * **Sigmoid** is the most natural choice. It outputs a probability between 0 and 1, directly interpretable as P(y=1|x).
    * TensorFlow: `activation='sigmoid'`

* **Regression (y is a continuous number):**
    * **Linear** activation is recommended if y can be **positive or negative** (e.g., predicting stock price change).
    * TensorFlow: `activation='linear'`
    * **ReLU** activation is recommended if y can only be **non-negative** (e.g., predicting house prices, which can't be negative).
    * TensorFlow: `activation='relu'`

### Hidden Layer Activation Function

* **ReLU** is the **most common and recommended default choice** for hidden layers.
    * TensorFlow: `activation='relu'`

* **Why ReLU over Sigmoid for hidden layers?**
    1.  **Computational Efficiency:** ReLU (max(0,z)) is faster to compute than Sigmoid (which involves exponentiation).
    2.  **Avoids "Flatness" (Vanishing Gradients):**
        * Sigmoid is "flat" on both ends (for very large positive or negative z).
        * ReLU is "flat" only on one side (for negative z).
        * Flat regions mean very small gradients, which significantly slow down gradient descent and can hinder learning. ReLU's less severe flatness helps neural networks learn faster.

### Summary of Activation Function Choices:

* **Output Layer:**
    * Binary Classification: Sigmoid
    * Regression (positive/negative y): Linear
    * Regression (non-negative y): ReLU
* **Hidden Layers:** ReLU (default)

While other activation functions exist (e.g., tanh, LeakyReLU, Swish), ReLU is generally sufficient for most applications.

A fundamental question remains: Why do we need activation functions at all, especially non-linear ones? The next video will explain why simply using linear activation functions everywhere would make neural networks ineffective.

## Why Neural Networks Need Non-Linear Activation Functions

Neural networks fundamentally rely on **non-linear activation functions** (like Sigmoid, ReLU) to learn complex patterns. If every neuron in a neural network used only the **linear activation function** ($g(z) = z$), the entire network would collapse into a simple linear model (equivalent to linear or logistic regression), defeating the purpose of using a complex multi-layered structure.

### The Problem with All Linear Activations

Let's illustrate with a simple 2-layer network:
* **Input:** $x$
* **Layer 1:** 1 hidden unit ($a_1 = g(w_1x + b_1)$)
* **Layer 2 (Output):** 1 output unit ($a_2 = g(w_2a_1 + b_2)$)

If $g(z) = z$ for all activations:
1.  $a_1 = (w_1x + b_1)$
2.  $a_2 = w_2(a_1) + b_2 = w_2(w_1x + b_1) + b_2$
3.  Expanding: $a_2 = (w_2w_1)x + (w_2b_1 + b_2)$

If we set $W_{new} = w_2w_1$ and $B_{new} = w_2b_1 + b_2$, then $a_2 = W_{new}x + B_{new}$.

This shows that a neural network with two linear layers (or any number of linear layers) simply computes an output that is a **linear function of the input**, just like a single linear regression model. Adding more linear layers achieves nothing more than what a single linear layer could do. This is a consequence of the fact that a linear function of a linear function is itself a linear function.

### Generalization

* **All linear hidden layers + linear output layer:** The entire network becomes equivalent to a single **linear regression** model.
* **All linear hidden layers + logistic output layer:** The entire network becomes equivalent to a single **logistic regression** model.

### Conclusion: Rule of Thumb

* **Never use linear activation functions in the hidden layers of a neural network.**
* **Always use non-linear activation functions (like ReLU)** in hidden layers to enable the network to learn complex, non-linear relationships in the data.
* Linear activation functions are typically reserved for the **output layer of regression problems** (where the target value can be any real number).

This understanding is crucial for building powerful neural networks that can model complex real-world data. The next video will extend classification to handle multiple categories (multi-class classification).

## Multiclass Classification

**Multiclass classification** refers to classification problems where the output variable `y` can take on **more than two discrete categories** (not just 0 or 1).

### Examples of Multiclass Classification:

* **Handwritten Digit Recognition:** Classifying an image into one of 10 digits (0-9).
* **Disease Diagnosis:** Classifying a patient's condition into one of several possible diseases.
* **Visual Defect Inspection:** Classifying a manufactured product as having a scratch, discoloration, or chip defect.

### Data Representation for Multiclass Classification:

* For binary classification, we typically estimated $P(y=1 | x)$.
* For multiclass classification, with, say, K classes, we want to estimate:
    * $P(y=1 | x)$ (probability of being class 1)
    * $P(y=2 | x)$ (probability of being class 2)
    * ...
    * $P(y=K | x)$ (probability of being class K)

### Decision Boundaries in Multiclass Classification:

<img src="/metadata/multiclass.png" width="400" />

Unlike binary classification, which learns a single decision boundary to separate two classes, multiclass classification algorithms learn **multiple decision boundaries** that divide the input space into several regions, one for each class.

The next video will introduce the **Softmax Regression algorithm**, a generalization of logistic regression, which is designed to handle multiclass classification problems. Following that, we'll see how to integrate Softmax Regression into a neural network for multiclass classification.

## Softmax Regression: Multiclass Classification

Softmax regression is a generalization of logistic regression, extending binary classification to **multiclass classification** (where $y$ can take on more than two discrete values).

### From Logistic to Softmax Regression

Recall logistic regression calculates $z = \vec{w} \cdot \vec{x} + b$, then $a = g(z) = \frac{1}{1 + e^{-z}}$, interpreted as $P(y=1 | x)$. We implicitly knew $P(y=0 | x) = 1 - P(y=1 | x)$.

To generalize, imagine logistic regression as computing two "activations":
* $a_1 = P(y=1 | x)$
* $a_0 = P(y=0 | x) = 1 - a_1$
where $a_1 + a_0 = 1$.

### Softmax Regression Model

For $K$ possible output classes ($y \in \{1, 2, \dots, K\}$):

1.  **Linear Scores (z_j):** For each class $j$, compute a linear score:
    $z_j = \vec{w}_j \cdot \vec{x} + b_j$
    (Here, $\vec{w}_j$ and $b_j$ are the parameters for class $j$).
2.  **Softmax Activation (a_j):** Convert these scores into probabilities using the softmax function:
    $a_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$

* **Interpretation:** $a_j$ is the model's estimate of the probability that $y$ is equal to class $j$, given input $x$.
* **Property:** By construction, $\sum_{j=1}^{K} a_j = 1$. The sum of probabilities for all classes always equals 1.
* **Generalization:** If $K=2$, softmax regression reduces to (a slightly re-parameterized version of) logistic regression.

### Cost Function for Softmax Regression

The loss function for a single training example with true label $y$ and predicted probabilities $a_1, \dots, a_K$ is based on **negative log-likelihood**:

* **Loss $L(\vec{a}, y)$:** If the true label for an example is $y=j$, the loss is $-\log(a_j)$.
    * **Intuition:**
        * If $a_j$ (the predicted probability for the *correct* class) is close to 1, then $-\log(a_j)$ is close to 0 (low loss).
        * If $a_j$ is small (meaning the model was not confident about the correct class), then $-\log(a_j)$ is large (high loss). This penalizes the model for being uncertain or wrong about the true class.
* **Overall Cost Function $J(\vec{W}, \vec{B})$:** This is the average of the loss function over all $m$ training examples:
    $J(\vec{W}, \vec{B}) = \frac{1}{m} \sum_{i=1}^{m} L(\vec{a}^{(i)}, y^{(i)})$
    (where $\vec{W}$ and $\vec{B}$ denote all parameters across all classes).

By minimizing this cost function, softmax regression learns parameters to accurately predict probabilities for each class. The next step is to integrate this into a neural network for multi-class classification.

## Multi-class Classification with Softmax Output Layer

To build a neural network for **multi-class classification**, we integrate the Softmax regression model into the network's **output layer**.

### Network Architecture for Multi-class

For handwritten digit recognition (0-9, 10 classes):
* Input (X): Image pixels.
* Hidden Layer 1: 25 units (ReLU activation).
* Hidden Layer 2: 15 units (ReLU activation).
* **Output Layer:** 10 units (for classes 0-9), using **Softmax activation function**.

### Forward Propagation with Softmax Output

1.  **Hidden Layers (A1, A2):** Computations in hidden layers 1 and 2 proceed exactly as before (e.g., $a_j^{[l]} = g(w_j^{[l]} \cdot a^{[l-1]} + b_j^{[l]})$ with ReLU).
2.  **Output Layer (A3):**
    * For each of the 10 output units, compute a linear score $z_j^{[3]}$:
        $z_j^{[3]} = \vec{w}_j^{[3]} \cdot \vec{a}^{[2]} + b_j^{[3]}$ (where $\vec{a}^{[2]}$ is the activation vector from the previous hidden layer).
    * Then, apply the **Softmax activation function** to all $z_j^{[3]}$ values simultaneously to get the probabilities $a_j^{[3]}$:
        $a_j^{[3]} = \frac{e^{z_j^{[3]}}}{\sum_{k=1}^{10} e^{z_k^{[3]}}}$
    * The vector $\vec{a}^{[3]} = [a_1^{[3]}, \dots, a_{10}^{[3]}]$ represents the estimated probabilities for each of the 10 classes.

### Key Characteristic of Softmax Activation:

* Unlike Sigmoid, ReLU, or Linear functions (which operate element-wise on their input $z$), the **Softmax activation function operates on *all* $z$ values simultaneously**. Each output probability $a_j$ depends on *all* input scores $z_1, \dots, z_K$.

### TensorFlow Implementation (High-Level):

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=25, activation='relu'),
    tf.keras.layers.Dense(units=15, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax') # 10 units for 10 classes
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy() # For integer labels (0, 1, ..., 9)
)

model.fit(X_train, Y_train, epochs=...)
```

* **`units=10` in last Dense layer:** Specifies 10 output neurons, one for each class.
* **`activation='softmax'`:** Applies the Softmax function to convert raw outputs into probabilities.
* **`loss=tf.keras.losses.SparseCategoricalCrossentropy()`:** This is the appropriate loss function for multi-class classification where true labels (Y) are integers (e.g., 0, 1, 2...9). "Sparse" indicates that each example belongs to exactly one category.

### Important Caveat: Recommended TensorFlow Usage

While the code above works, there's a more numerically stable and recommended way to implement the Softmax output layer in TensorFlow, which will be covered in the next video. You should **not use the `activation='softmax'` directly in the last layer** when using `SparseCategoricalCrossentropy` loss, as it handles the Softmax computation internally for better numerical precision.

## Improved Softmax Implementation: Numerical Stability

The previous Softmax implementation, while conceptually correct, can suffer from **numerical round-off errors** in computers due to floating-point precision limitations. This video explains a more robust way to implement Softmax in TensorFlow.

### The Problem: Numerical Instability

Calculations involving very small or very large numbers (e.g., `exp(z)` where `z` is very large or very small) can lead to precision loss.
* **Example:** Calculating `2/10000` directly is more accurate than `(1 + 1/10000) - (1 - 1/10000)`, where intermediate terms are calculated before subtraction.

Similarly, in Softmax, `e^z_j` can be very large, or the sum in the denominator can be very small/large, leading to `a_j` values that are slightly off.

### Solution: `from_logits=True`

TensorFlow provides a more numerically stable way to compute the Softmax probabilities and the cross-entropy loss together, by avoiding the explicit intermediate calculation of `a_j`.

1.  **Modify Model Architecture:**
    * The **final `Dense` layer** should now use a **`linear` activation function** (or no activation function specified, as linear is the default). This means the output of the last layer will be the raw `z` values (often called **logits**).
    * Example:
        ```python
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=25, activation='relu'),
            tf.keras.layers.Dense(units=15, activation='relu'),
            tf.keras.layers.Dense(units=10, activation='linear') # Output layer: 10 units, linear activation
        ])
        ```

2.  **Modify Compile Step:**
    * When compiling the model, specify the loss function (e.g., `SparseCategoricalCrossentropy`) with the argument **`from_logits=True`**.
    * Example:
        ```python
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        ```

3.  **Actually getting the result:**
    * Example:
        ```python
        # Train
        model.fit(X, Y, epoch=100)
        
        # Have to do extra step to actually get a1, a2, ...
        logits = model(x_input)         # this actually outputs z1, z2, ... 
        f_x = tf.nn.softmax(logits)     # thats how we then get a1, a2, ...
        ```

**How it works:**
* By setting `from_logits=True`, you tell TensorFlow that the output of your last layer is the raw `z` values (logits), *before* the Softmax activation.
* TensorFlow then **internally and simultaneously computes the Softmax probabilities and the cross-entropy loss** in a single, numerically optimized step. This avoids potential precision issues that arise from calculating $a_j$ explicitly and then passing those potentially rounded $a_j$ values to the loss function.
* Effectively, rather than calculating $a_j$ explicitly and passing it to loss function, we just calculate the `z` and let tensorflow rearrange all the values in cost function (so that probably some terms cancel out OR/AND there is less numerical precision errors)

**Important Note for Binary Classification (Logistic Regression):**
The same principle applies to binary classification. For a logistic regression output layer, you would also use `activation='linear'` in the `Dense` layer and `tf.keras.losses.BinaryCrossentropy(from_logits=True)` in `model.compile()`.

```python
# Train
model.fit(X, Y, epoch=100)

# Have to do extra step to actually get a1, a2
logits = model(x_input)         # this actually outputs z1, z2
f_x = tf.nn.sigmoid(logits)     # thats how we then get a1, a2
```
This recommended implementation is more robust and accurate, though it might make the conceptual flow (z -> a -> loss) slightly less explicit in the code.

The next video will introduce another type of classification problem: **multi-label classification**.

## Multi-label Classification

**Multi-label classification** is a type of classification problem where a single input can have **multiple associated labels (outputs)** simultaneously. This is distinct from multi-class classification, where the input belongs to only one of many possible categories.

### Examples of Multi-label Classification:

* **Self-Driving Cars / Driver Assistance:** Given an image, identify if it contains:
    * A car (Yes/No)
    * A bus (Yes/No)
    * A pedestrian (Yes/No)
    For a single image, the output is a vector of binary labels, e.g., `[1, 0, 1]` (Car present, Bus absent, Pedestrian present).
* **Image Tagging:** Tagging an image with multiple concepts like "beach," "sunset," "people."

### Building a Neural Network for Multi-label Classification:

There are two main approaches:

1.  **Separate Networks:** You could train a completely separate binary classification neural network for each label (e.g., one network for "car present," one for "bus present," etc.). This is a valid, though potentially less efficient, approach.

2.  **Single Neural Network with Multiple Output Units:**
    * **Architecture:** The neural network typically has an input layer, one or more hidden layers, and an **output layer with multiple units**, where each unit corresponds to one possible label.
    * **Output Layer:** For a problem with $K$ labels, the output layer will have $K$ neurons. Each of these $K$ output neurons will use a **Sigmoid activation function**.
    * **Prediction:** The output `a` will be a vector of $K$ probabilities, e.g., `a = [P(car), P(bus), P(pedestrian)]`. Each element represents the probability of that specific label being true, independently of the others.
    * Example: For the car/bus/pedestrian problem, the output layer would have 3 neurons, each with Sigmoid activation.

This unified approach allows the network to learn shared features in its hidden layers that are useful for predicting all labels simultaneously.

The next video will introduce advanced optimization algorithms that can train neural networks faster than basic gradient descent.

## Optimizing Neural Network Training: The Adam Algorithm

Gradient descent is a foundational optimization algorithm, but modern neural networks benefit from more advanced optimizers that can train models much faster. The **Adam algorithm (Adaptive Moment Estimation)** is a popular choice that automatically adjusts learning rates.

### Limitations of Basic Gradient Descent

* **Slow Convergence (Small Alpha):** If the learning rate ($\alpha$) is too small, gradient descent takes tiny steps, leading to very slow convergence, especially in elongated cost function landscapes. You might wish $\alpha$ would automatically increase.
* **Oscillation/Divergence (Large Alpha):** If $\alpha$ is too large, gradient descent might overshoot the minimum, oscillating back and forth or even diverging, causing the cost to increase. You'd wish $\alpha$ would automatically decrease.

### How Adam Algorithm Helps

The Adam algorithm intelligently adapts the learning rate for each parameter automatically:

* **Per-Parameter Learning Rates:** Unlike basic gradient descent which uses a single global $\alpha$, Adam maintains a **separate learning rate for every single parameter** ($w_j$ and $b$) in the model.
* **Adaptive Adjustment:**
    * If a parameter's updates consistently move in roughly the same direction (like in the "too small $\alpha$" scenario), Adam **increases its specific learning rate**, accelerating convergence in that dimension.
    * If a parameter's updates oscillate back and forth (like in the "too large $\alpha$" scenario), Adam **decreases its specific learning rate**, dampening oscillations and promoting smoother convergence.

### Implementing Adam in TensorFlow

Using Adam is straightforward in TensorFlow:

```python
import tensorflow as tf

# ... (model definition using Sequential API) ...

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # or other loss
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001) # Specify Adam optimizer
)

model.fit(X_train, Y_train, epochs=...)
```

* You pass an `optimizer` argument to `model.compile()`.
* `tf.keras.optimizers.Adam()` is the Adam optimizer.
* It takes a `learning_rate` parameter (e.g., `0.001`), which serves as an initial/default global learning rate. Although Adam adapts, it's still worth experimenting with this initial value (e.g., trying factors of 10 like 0.01, 0.001, 0.0001) to find the best performance.

### Benefits of Adam:

* **Faster Convergence:** Typically converges much faster than basic gradient descent.
* **Robustness:** More robust to the initial choice of learning rate compared to traditional gradient descent, as it adapts automatically.
* **Industry Standard:** Adam is a de facto standard and widely used by practitioners for training neural networks today.

The next videos will explore other advanced neural network concepts, starting with alternative layer types.
