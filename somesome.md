## Welcome to Week 3: Practical Advice for Building ML Systems

This week focuses on **practical strategies for building effective machine learning systems**. While you've learned powerful algorithms like linear/logistic regression and neural networks, efficient development hinges on making good decisions about where to invest your time. I've seen teams spend months on approaches that a more skilled team could have done in weeks.

### The Problem: What to Try Next?

When your model (e.g., regularized linear regression for housing prices) performs poorly (makes large errors), there are many potential next steps:

* **More Data:** Collect more training examples.
* **Fewer Features:** Try a smaller subset of existing features.
* **Additional Features:** Find and add new, relevant features.
* **Polynomial Features:** Add non-linear transformations of existing features (e.g., $x^2$, $x_1x_2$).
* **Regularization Parameter ($\lambda$):** Adjust $\lambda$ (decrease if too much bias, increase if too much variance).

The key to efficiency is knowing which of these options will be most fruitful.

### Diagnostics: Guiding Your ML Project

This week will teach you about **diagnostics**: tests you can run to gain insights into what is or isn't working with your learning algorithm.

* **Purpose of Diagnostics:** To provide guidance on where to invest your time and effort to improve performance. For example, a diagnostic can tell you if spending weeks or months collecting more data is truly worthwhile.
* **Time Investment:** Diagnostics themselves take time to implement, but they can save far more time in the long run by preventing misguided efforts.

We'll begin by learning how to properly evaluate the performance of your machine learning algorithm.

## Evaluating Model Performance

Systematic evaluation of a machine learning model's performance is crucial for understanding its effectiveness and guiding improvements.

### The Need for Systematic Evaluation

When models become complex (e.g., using many features beyond what can be plotted), visual inspection of the fit is no longer sufficient to identify problems like overfitting. We need numerical metrics. **Note**: even after using regularization term in cost function, model can overfit -- hence we do following all things to avoid overfitting and choosing correct model-parameters (w and b) and hyperparameters (polynomial degree, lambda, architecture like # of layers or/and # of neurons per layer)

### Splitting the Dataset

To evaluate effectively, split your dataset into two subsets:

1.  **Training Set (e.g., 70% of data):** Used to fit (train) the model's parameters ($w, b$). Denoted as $({x_{train}^{(i)}}, {y_{train}^{(i)}})$ for $i = 1, \dots, m_{\text{train}}$.
2.  **Test Set (e.g., 30% of data):** Used to evaluate the model's performance on unseen data. Denoted as $({x_{test}^{(i)}}, {y_{test}^{(i)}})$ for $i = 1, \dots, m_{\text{test}}$.

### Evaluation for Regression Problems (Squared Error)

$$J(w,b) = \frac{1}{2m_{train}} \sum_{i=1}^{m_{train}} (f_{w,b}(x_{train}^{(i)}) - y_{train}^{(i)})^2  + \frac{\lambda}{2m_{train}} \sum_{j=1}^{n} w_j^2$$

After training the model on the training set by minimizing $J(w, b)$ (the cost function including regularization), evaluate its performance using:

* **Test Error ($$J_{test}(w,b)$$):** Average squared error on the test set.

    $$J_{test}(w,b) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}} (f_{w,b}(x_{test}^{(i)}) - y_{test}^{(i)})^2$$
    * **Crucially, this formula does NOT include the regularization term ($\lambda \sum w_j^2$)**, as regularization is part of the *training objective*, not the *performance metric*.
* **Training Error ($$J_{train}(w,b)$$):** Average squared error on the training set.

    $$J_{train}(w,b) = \frac{1}{2m_{train}} \sum_{i=1}^{m_{train}} (f_{w,b}(x_{train}^{(i)}) - y_{train}^{(i)})^2$$
    * Again, this also does NOT include the regularization term.

**Example of Overfitting Detection:** If a model has very low $J_{\text{train}}$ (e.g., near zero, indicating a perfect fit on training data) but a high $J_{\text{test}}$, it signifies overfitting.

### Evaluation for Classification Problems (Logistic Loss / Misclassification Error)

$$J(w,b) = \frac{-1}{m_{train}} \sum_{i=1}^{m_{train}} [y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)}) \log(1-f_{\vec{w},b}(\vec{x}^{(i)}))] + \frac{\lambda}{2m_{train}} \sum_{j=1}^{n} w_j^2$$

For classification, after training by minimizing the regularized logistic cost function, you can evaluate using:

1.  **Logistic Loss ($$J_{test} / J_{train}$$):** Compute the average logistic loss on the test/training sets, similar to regression but using the logistic loss formula.
2.  **Misclassification Error (More Common):**
    * **For each example:** Make a binary prediction $\hat{y}$ (e.g., $\hat{y}=1$ if $f(x) \ge 0.5$, else $\hat{y}=0$).
    * **Test Error ($J_{\text{test}}$):** The fraction of examples in the test set where $\hat{y} \ne y_{\text{actual}}$.
    * **Training Error ($J_{\text{train}}$):** The fraction of examples in the training set where $\hat{y} \ne y_{\text{actual}}$.

Splitting data into training and test sets provides a systematic way to measure a model's performance and is a foundational step in model selection. The next video will build on this to automate model selection.

## Model Selection: Using a Cross-Validation Set

To automatically choose the best model (e.g., polynomial degree, neural network architecture), we refine the data splitting strategy beyond just training and test sets.

### The Flaw of Using Only a Test Set for Model Selection

* If you train different models (e.g., linear, quadratic, cubic) on the training set and then pick the one with the lowest error on the *test set*, your reported test error will be an **optimistic estimate** of the true generalization error.
* This is because you've used the test set *itself* to select a model, effectively "fitting" the model selection parameter (like polynomial degree $d$) to the test set, similar to how training fits $w$ and $b$ to the training set.

### Three-Way Data Split for Model Selection

To perform unbiased model selection, split your dataset into three subsets:

1.  **Training Set (e.g., 60% of data):** Denoted $X_{\text{train}}, Y_{\text{train}}$, with $m_{\text{train}}$ examples. Used **only for fitting the model's parameters** ($w, b$).
2.  **Cross-Validation Set (or Validation Set, Dev Set) (e.g., 20% of data):** Denoted $X_{\text{cv}}, Y_{\text{cv}}$, with $m_{\text{cv}}$ examples. Used **only for choosing hyperparameters** (e.g., polynomial degree $d$, neural network architecture, regularization parameter $\lambda$).
    * The name "cross-validation" means it's used to "cross-check" the validity of different models.
    * This is also used to early stop the training set from overfitting on fixed set of hyperparameters. While training a model (with fixed hyperparameters), at each epoch, we calculate cross-validation loss too. If it is increasing consistently we stop as we might be overfitting the data. 
3.  **Test Set (e.g., 20% of data):** Denoted $X_{\text{test}}, Y_{\text{test}}$, with $m_{\text{test}}$ examples. Used **only for the final, unbiased estimate** of the chosen model's generalization error on completely unseen data.

### Model Selection Procedure:

Let's say you are choosing among $D$ different models (e.g., polynomials of degree $d=1$ to $d=10$):

1.  **Train each model:** For each candidate model (e.g., for each polynomial degree $d$):
    * Fit its parameters ($w, b$) by minimizing the cost function on the **training set only**.
    * This yields parameters $w^{(d)}, b^{(d)}$ for each model $d$.

2.  **Evaluate on Cross-Validation Set:** For each trained model:
    * Compute its error on the **cross-validation set** ($$J_{cv}(w^{(d)}, b^{(d)})$$).
    * For regression, this is the average squared error on $X_{cv}, Y_{cv}$.
    * For classification, this is usually the misclassification error (fraction of errors) on $X_{cv}, Y_{cv}$.

3.  **Choose the Best Model:** Select the model (e.g., degree $d^*$) that has the **lowest error on the cross-validation set**. This is your chosen model.

4.  **Estimate Generalization Error:**
    * Report the error of the *chosen model* ($w^{(d*)}, b^{(d*)}$) on the **test set** ($$J_{test}(w^{(d*)}, b^{(d*)})$$).
    * Since the test set was not used for training parameters or for model selection, this $J_{test}$ provides a fair and unbiased estimate of how well your final model will perform on new data.

### Importance of This Procedure:

* **Unbiased Evaluation:** Prevents overly optimistic estimates of generalization error.
* **Systematic Model Selection:** Provides a clear, data-driven way to choose between different model complexities or architectures.
* **Best Practice:** Considered standard best practice in machine learning for any project involving model selection.

This refined evaluation technique is crucial for building robust and generalizable machine learning systems. It sets the stage for using powerful diagnostics like bias and variance, which will be discussed next.

## Diagnosing Bias and Variance

After training a machine learning model, it rarely performs perfectly on the first try. The key to improvement is diagnosing *why* it's not performing well. Looking at **bias** and **variance** helps guide your next steps.

### Bias vs. Variance Visualized (1D Example)

Recall the housing price prediction example with a single feature $x$:

* **High Bias (Underfitting):** (e.g., fitting a straight line, $d=1$)
    * The model is too simple; it doesn't capture the underlying pattern.
    * **Characteristic:** Both training error ($J_{\text{train}}$) and cross-validation error ($J_{\text{cv}}$) will be **high**. The model doesn't even fit the data it trained on very well.

* **High Variance (Overfitting):** (e.g., fitting a 4th-order polynomial, $d=4$)
    * The model is too complex; it fits the training data (including noise) too perfectly but fails to generalize.
    * **Characteristic:** Training error ($J_{\text{train}}$) will be **low** (model performs great on seen data), but cross-validation error ($J_{\text{cv}}$) will be **much higher than** $J_{\text{train}}$.

* **Just Right:** (e.g., fitting a quadratic polynomial, $d=2$)
    * The model fits the underlying pattern well without overfitting.
    * **Characteristic:** Both $J_{\text{train}}$ and $J_{\text{cv}}$ will be **low** and relatively close to each other.

### Diagnosing Bias and Variance Systematically

For models with many features that are hard to visualize:

* **High Bias Indicator:** $J_{\text{train}}$ is high. (This means the model can't even fit the training data adequately).
* **High Variance Indicator:** $J_{\text{cv}}$ is much greater than $J_{\text{train}}$. (This means the model fits the training data well but struggles with unseen data).

### Bias-Variance Trade-off Curve (as a function of polynomial degree $d$)

<img src="/metadata/bias_variance.png" width="500" />

When plotting $J_{\text{train}}$ and $J_{\text{cv}}$ against the polynomial degree $d$ (or model complexity):

* **$J_{\text{train}}$:** As $d$ increases (model complexity increases), $J_{\text{train}}$ generally **decreases**, approaching zero for very high degrees, as the model becomes capable of perfectly fitting the training data.
* **$J_{\text{cv}}$:**
    * For **low $d$ (simple models)**, $J_{\text{cv}}$ is high (high bias).
    * As $d$ increases, $J_{\text{cv}}$ first **decreases** to a minimum (the "just right" spot).
    * For **high $d$ (complex models)**, $J_{\text{cv}}$ then **increases** again (high variance) because the model overfits.

### High Bias and High Variance Simultaneously (Rare but Possible)

In some complex scenarios (especially with neural networks), a model can exhibit both high bias and high variance.

* **Indicator:** $J_{\text{train}}$ is high, AND $J_{\text{cv}}$ is much greater than $J_{\text{train}}$.
* **Intuition:** The model might underfit some parts of the input space while overfitting others, resulting in overall poor performance on both training and cross-validation sets, with a significant gap between them.

Knowing how to diagnose high bias vs. high variance (or both) provides crucial guidance on what actions to take to improve your model's performance. Next, we'll see how regularization affects bias and variance.

## Regularization, Bias, and Variance

This video explains how the **regularization parameter $\lambda$ (lambda)** affects the bias and variance of a learning algorithm, guiding its optimal selection. We'll use a 4th-order polynomial model with regularization as an example.

### Impact of $\lambda$ on Model Fit:

* **Large $\lambda$ (e.g., $\lambda = 10000$): High Bias (Underfitting)**
    * The regularization term heavily penalizes large parameters, forcing most $w_j$ values to be very close to zero.
    * The model approximates $f(x) \approx b$ (a constant horizontal line).
    * **Result:** The model significantly underfits the data. Both $J_{\text{train}}$ and $J_{\text{cv}}$ will be **high**.

* **Small $\lambda$ (e.g., $\lambda = 0$): High Variance (Overfitting)**
    * No regularization is applied. The model fits the training data almost perfectly, potentially capturing noise.
    * **Result:** The model overfits. $J_{\text{train}}$ will be **low**, but $J_{\text{cv}}$ will be **much higher than** $J_{\text{train}}$.

* **"Just Right" $\lambda$ (Intermediate Value): Optimal Fit**
    * A balanced $\lambda$ allows the model to fit the underlying patterns well without overfitting the noise.
    * **Result:** Both $J_{\text{train}}$ and $J_{\text{cv}}$ will be **low** and relatively close to each other.

### Choosing Optimal $\lambda$ using Cross-Validation

The procedure for choosing $\lambda$ is similar to selecting the polynomial degree:

1.  **Define a set of candidate $\lambda$ values:** Try a wide range, often increasing by factors (e.g., $0, 0.01, 0.02, 0.04, \dots, 10$).
2.  **For each $\lambda$ value:**
    * Train the model's parameters ($w, b$) by minimizing the regularized cost function on the **training set**.
    * Compute the **cross-validation error ($J_{\text{cv}}$)** for this trained model.
3.  **Select the best $\lambda$:** Choose the $\lambda$ value that results in the **lowest $J_{\text{cv}}$**.
4.  **Report generalization error:** Evaluate the final chosen model (with its $w, b$ trained using the optimal $\lambda$) on the untouched **test set** ($J_{\text{test}}$) to get an unbiased estimate of its true performance.

### $\lambda$ vs. Model Complexity ($d$) Curve Comparison

* **Plot $J_{\text{train}}$ vs. $\lambda$:** As $\lambda$ increases, the penalty for large $w_j$ increases. This forces $w_j$ to be smaller, making it harder for the model to fit the training data perfectly. Thus, $J_{\text{train}}$ will generally **increase** as $\lambda$ increases.
* **Plot $J_{\text{cv}}$ vs. $\lambda$:**
    * For very small $\lambda$ (left side), $J_{\text{cv}}$ is high due to **overfitting (high variance)**.
    * As $\lambda$ increases, $J_{\text{cv}}$ decreases to a minimum.
    * For very large $\lambda$ (right side), $J_{\text{cv}}$ is high due to **underfitting (high bias)**.
    * This curve typically has a "U" or "V" shape, with the minimum indicating the "just right" $\lambda$.

This diagram is somewhat a "mirror image" of the $J_{\text{train}}$ and $J_{\text{cv}}$ versus polynomial degree $d$ plot, but both illustrate how cross-validation helps find the optimal model complexity parameter.

The next video will discuss how to interpret whether $J_{\text{train}}$ and $J_{\text{cv}}$ values are "high" or "low" by establishing a baseline performance.

## Diagnosing Bias and Variance with Baseline Performance

When training a machine learning model, it's rare for it to work perfectly on the first try. Diagnosing whether the problem is **high bias (underfitting)** or **high variance (overfitting)** is crucial for deciding the next steps. This often involves comparing training error ($J_{\text{train}}$) and cross-validation error ($J_{\text{cv}}$) against a **baseline level of performance**.

### Establishing a Baseline Level of Performance

* This is the error level you can **reasonably hope** your learning algorithm can achieve.
* **Common Baselines:**
    * **Human-level performance:** Especially useful for unstructured data (audio, images, text) where humans excel. For example, if even humans make 10.6% error in transcribing noisy speech, expecting an algorithm to do much better than that is unrealistic.
    * **Performance of competing algorithms:** A previous implementation or a competitor's system.
    * **Prior experience/Guesswork:** Based on similar problems.

### Diagnosing High Bias vs. High Variance with a Baseline:

Let's use a speech recognition example:
* **Baseline (Human-level performance):** 10.6% error.

1.  **Example 1: High Variance Problem**
    * $J_{\text{train}} = 10.8\%$
    * $J_{\text{cv}} = 14.8\%$
    * **Analysis:**
        * **Baseline vs. Training Error Gap:** $10.8\% - 10.6\% = 0.2\%$. This small gap indicates the model is doing quite well on the training set, close to human performance. So, **bias is low**.
        * **Training vs. Cross-Validation Error Gap:** $14.8\% - 10.8\% = 4.0\%$. This is a significant gap, meaning the model performs much worse on unseen data.
        * **Conclusion:** This is primarily a **high variance (overfitting)** problem.

2.  **Example 2: High Bias Problem**
    * $J_{\text{train}} = 15.0\%$
    * $J_{\text{cv}} = 15.5\%$
    * **Analysis:**
        * **Baseline vs. Training Error Gap:** $15.0\% - 10.6\% = 4.4\%$. This large gap indicates the model is struggling even with the training data, performing significantly worse than what's achievable.
        * **Training vs. Cross-Validation Error Gap:** $15.5\% - 15.0\% = 0.5\%$. This small gap suggests it's not overfitting significantly.
        * **Conclusion:** This is primarily a **high bias (underfitting)** problem.

3.  **Example 3: Both High Bias and High Variance (Possible but Less Common)**
    * $J_{\text{train}} = 15.0\%$
    * $J_{\text{cv}} = 19.7\%$
    * **Analysis:**
        * **Baseline vs. Training Error Gap:** $15.0\% - 10.6\% = 4.4\%$. High bias.
        * **Training vs. Cross-Validation Error Gap:** $19.7\% - 15.0\% = 4.7\%$. High variance.
        * **Conclusion:** The algorithm suffers from **both high bias and high variance**.

### Summary of Diagnostic Logic:

* **High Bias:** Indicated by a large gap between **Baseline Performance** and **$J_{\text{train}}$**.
* **High Variance:** Indicated by a large gap between **$J_{\text{train}}$** and **$J_{\text{cv}}$**.

This systematic approach provides a more accurate diagnosis of your model's issues, especially for complex tasks where perfect performance (0% error) is unrealistic. This diagnosis then guides your next steps in improving the model. The next video will introduce another useful diagnostic tool: the **learning curve**.

## Learning Curves: Diagnosing Bias and Variance with Data Size

Learning curves plot the performance of a learning algorithm (training error $J_{\text{train}}$ and cross-validation error $J_{\text{cv}}$) as a function of the **training set size ($m_{\text{train}}$)**. They are a powerful diagnostic tool.

### General Learning Curve Behavior

<img src="/metadata/learn_curvess.png" width="500" />

* **$J_{\text{cv}}$ (Cross-Validation Error):** As $m_{\text{train}}$ increases, $J_{\text{cv}}$ generally **decreases**. More data typically leads to a better, more generalizable model, thus lower error on unseen data.
* **$J_{\text{train}}$ (Training Error):** As $m_{\text{train}}$ increases, $J_{\text{train}}$ generally **increases**. With very little data, a model (especially a complex one) can easily fit all points perfectly (or nearly perfectly), resulting in low training error. As more examples are added, it becomes harder for the model to fit every single training example perfectly, so the average training error increases.
* **Relationship:** $J_{\text{cv}}$ will typically be higher than $J_{\text{train}}$ because the parameters are optimized on the training set.

### Learning Curves for High Bias (Underfitting)

<img src="/metadata/lc_1.png" width="500" />

* **Scenario:** The model is too simple (e.g., fitting a linear function to non-linear data).
* **Learning Curve Shape:**
    * $J_{\text{train}}$ starts low (for very small $m_{\text{train}}$) but quickly **flattens out at a high error value**.
    * $J_{\text{cv}}$ starts high (for small $m_{\text{train}}$) and also **flattens out at a high error value**, typically close to $J_{\text{train}}$.
    * Both $J_{\text{train}}$ and $J_{\text{cv}}$ remain high and close to each other.
    * There will be a significant gap between these flattened curves and the **baseline performance** (e.g., human-level error).
* **Key Insight:** If a learning algorithm has **high bias**, **getting more training data will NOT significantly help** improve its performance. The model is fundamentally too simple to learn the underlying patterns, regardless of how much data it sees.

### Learning Curves for High Variance (Overfitting)

<img src="/metadata/lc_22.png" width="500" />

* **Scenario:** The model is too complex (e.g., fitting a 4th-order polynomial to limited data).
* **Learning Curve Shape:**
    * $J_{\text{train}}$ starts very low (often near zero for small $m_{\text{train}}$) and slowly **increases** as $m_{\text{train}}$ grows.
    * $J_{\text{cv}}$ starts very high (for small $m_{\text{train}}$) and **decreases** as $m_{\text{train}}$ grows.
    * There is a **large gap between $J_{\text{train}}$ and $J_{\text{cv}}$**. This large gap is the signature of high variance.
    * The model might perform unrealistically well on the training set, potentially even better than human-level performance.
* **Key Insight:** If a learning algorithm suffers from **high variance**, **getting more training data is very likely to help**. As $m_{\text{train}}$ increases, $J_{\text{cv}}$ should continue to decrease and approach $J_{\text{train}}$, leading to better generalization.

### Practical Considerations for Plotting Learning Curves:

* **Method:** Train models on increasing subsets of your available training data (e.g., 100, 200, 300, ..., 1000 examples if you have 1000 total). Plot $J_{\text{train}}$ and $J_{\text{cv}}$ for each subset size.
* **Computational Cost:** Training multiple models can be computationally expensive, so this diagnostic isn't always performed.
* **Mental Model:** Even without plotting, having a mental picture of these learning curves can help you diagnose whether your algorithm has high bias or high variance.

This understanding of learning curves complements the previous diagnosis methods by showing how performance scales with data. The next video will apply these diagnostic insights to common machine learning problems.

## Practical Debugging: Addressing High Bias and High Variance

When your machine learning model performs poorly, diagnosing whether it has **high bias (underfitting)** or **high variance (overfitting)** provides a roadmap for improvement. This diagnosis helps you decide which techniques to apply to fix the problem.

Let's revisit common strategies in the context of fixing bias vs. variance:

### Strategies for High Bias (Underfitting)

High bias means the model performs poorly even on the training set. The model is too simple or lacks sufficient flexibility to capture the underlying patterns.

1.  **Get Additional Features:**
    * **Helps High Bias:** Yes. Providing more relevant information to the model can give it the necessary input to learn the underlying patterns that it couldn't capture before (e.g., adding bedrooms, floors, age to house price prediction).
2.  **Add Polynomial Features:**
    * **Helps High Bias:** Yes. Creating non-linear transformations of existing features (e.g., $x^2, x_1x_2$) increases the model's complexity and flexibility, allowing it to fit non-linear patterns.
3.  **Decrease Regularization Parameter (Lambda, $\lambda$):**
    * **Helps High Bias:** Yes. A smaller $\lambda$ reduces the penalty on parameter size, giving the model more freedom to fit the training data better and capture more complex relationships.

### Strategies for High Variance (Overfitting)

High variance means the model performs well on the training set but poorly on unseen data. The model is too complex or has too much flexibility.

1.  **Get More Training Examples:**
    * **Helps High Variance:** Yes. More data allows the complex model to learn more general patterns instead of just memorizing noise, making it less prone to overfitting a small dataset.
2.  **Try a Smaller Set of Features:**
    * **Helps High Variance:** Yes. Reducing the number of features simplifies the model, limiting its flexibility and reducing its ability to overfit. This is useful if many features are redundant or irrelevant.
3.  **Increase Regularization Parameter (Lambda, $\lambda$):**
    * **Helps High Variance:** Yes. A larger $\lambda$ heavily penalizes large parameter values, forcing the model to be smoother and less "wiggly," thus reducing overfitting.

| Strategy                       | Helps Fix High Bias (Underfitting) | Helps Fix High Variance (Overfitting) |
| :----------------------------- | :--------------------------------- | :------------------------------------ |
| **Get More Training Examples** | No (by itself)                     | Yes                                   |
| **Add / Remove Features** | Add few features                                | Remove few features |
| **Add / Remove Polynomial Features** | Add poly features                                | Remove poly features |
| **Lambda ($\lambda$)** | Decrease                                | Increase|

### Important Considerations:

* **Reducing Training Set Size:** Do NOT reduce training set size to fix high bias. While it might lower training error (by making it easier to fit a small set), it will worsen generalization and increase cross-validation error, exacerbating overall performance issues.
* **Bias-Variance is Foundational:** The concepts of bias and variance are fundamental to machine learning diagnostics and will guide your decision-making throughout your ML career. It's a concept that takes practice to master.

The next video will apply these crucial bias and variance concepts to the context of neural network training.

## Neural Networks and the Bias-Variance Trade-off

Neural networks, especially when combined with large datasets, offer a new perspective on addressing high bias and high variance, moving beyond the traditional bias-variance trade-off dilemma.

### Traditional Bias-Variance Trade-off (Pre-Neural Networks)

* **Dilemma:** Machine learning engineers often had to balance model complexity (e.g., polynomial degree, regularization parameter $\lambda$) to avoid both high bias (underfitting, model too simple) and high variance (overfitting, model too complex). You had to find a "just right" spot where $J_{\text{cv}}$ was minimized.

### Neural Networks: A New Approach

Large neural networks offer a way to reduce bias and variance more independently, with caveats.

* **Large Neural Networks as "Low Bias Machines":**
    * If you make a neural network large enough (more hidden layers, more neurons per layer), it can almost always fit your training data very well, achieving low $J_{\text{train}}$. This means they are inherently good at reducing bias, provided the training set isn't excessively enormous (which would make training computationally infeasible).

### Recipe for Building Accurate Neural Networks (when applicable):

<img src="/metadata/bias_variance_nn.png" width="600" />

This recipe works well for applications where you have access to sufficient data and computational power:

1.  **Reduce Bias (Fit Training Set Well):**
    * **Question:** Does the model do well on the training set? (Is $J_{\text{train}}$ low, e.g., comparable to human-level performance?)
    * **If $J_{\text{train}}$ is high (High Bias):**
        * **Action:** Use a **bigger neural network** (more hidden layers, more hidden units per layer).
        * **Repeat:** Keep increasing network size until $J_{\text{train}}$ is acceptably low.
    * *(This step leverages the "low bias machine" property of large neural networks.)*

2.  **Reduce Variance (Generalize Well):**
    * **Question:** Does the model do well on the cross-validation set? (Is $J_{\text{cv}}$ not much higher than $J_{\text{train}}$?)
    * **If $J_{\text{cv}}$ is much higher than $J_{\text{train}}$ (High Variance):**
        * **Action:** **Get more data.**
        * **Repeat:** Collect more data and retrain until $J_{\text{cv}}$ is closer to $J_{\text{train}}$.
    * *(This addresses overfitting by providing more examples for the complex network to generalize from.)*

3.  **Iterate:** Continue looping between steps 1 and 2 until the model performs well on both the training and cross-validation sets.

### Limitations:

* **Computational Expense:** Training larger neural networks requires significant computational power (often GPUs). Beyond a certain point, training becomes infeasible.
* **Data Availability:** Getting more data isn't always possible.

### Neural Network Size and Regularization:

* **Larger Networks are Often Better (with Regularization):** A very large neural network, when appropriately regularized, will typically perform as well or better than a smaller one.
    * **Caveat:** The primary "cost" of a larger network is increased computational time for training and inference.
* **Regularization for Neural Networks:**
    * The regularization term in the cost function for neural networks is: $\frac{\lambda}{2m} \sum_{\text{all weights } W} W^2$.
    * Typically, only the weight parameters ($W$) are regularized, not the bias parameters ($b$).
* **TensorFlow Implementation:** You add `kernel_regularizer=tf.keras.regularizers.l2(lambda_value)` to your `Dense` layers. You can choose different $\lambda$ values for different layers, though often a single $\lambda$ is used for all weights.

```python
# Unregularized
layer1 = Dense(units=25, activation="relu")
layer2 = Dense(units=10, activation="relu")
layer3 = Dense(units=1, activation="sigmoid")

# Regularized
layer1 = Dense(units=25, activation="relu", kernel_regularizer=L2(0.01))
layer2 = Dense(units=10, activation="relu", kernel_regularizer=L2(0.01))
layer3 = Dense(units=1, activation="sigmoid", kernel_regularizer=L2(0.01))
```

### Key Takeaways:

1.  **It almost never hurts to use a larger neural network (performance-wise), provided it's properly regularized.** The main trade-off is computational cost.
2.  **Large neural networks are often "low bias machines":** They excel at fitting complex functions, meaning you are often fighting **variance problems** (overfitting) rather than bias problems when using large enough networks.

This shift in thinking, enabled by deep learning and big data, has profoundly impacted how ML practitioners approach bias and variance. The next video will integrate all these ideas into a practical development workflow for ML systems.

## Machine Learning Development Process: An Iterative Loop

Developing a machine learning system is an iterative process. It rarely works perfectly on the first try. The key to efficiency is making good decisions about "what to do next" to improve performance.

### The Iterative Loop:

1.  **Architecture Design:**
    * Choose your machine learning model (e.g., linear regression, neural network).
    * Decide on data representation and features.
    * Pick initial hyperparameters.
2.  **Implement & Train Model:**
    * Train the model on your chosen data.
3.  **Run Diagnostics:**
    * **Bias/Variance Analysis:** Check if the model is underfitting (high bias) or overfitting (high variance) using training and cross-validation errors.
    * **Error Analysis:** (Discussed next video) Examine misclassified examples.
4.  **Decide Next Steps (Based on Diagnostics):**
    * Make a bigger neural network?
    * Adjust regularization parameter ($\lambda$)?
    * Collect more data?
    * Add/subtract features?
    * Improve existing features?
5.  **Iterate:** Go back to step 1 with the refined architecture/data and repeat the loop until desired performance is achieved.

### Example: Building an Email Spam Classifier

* **Problem:** Classify emails as spam ($y=1$) or non-spam ($y=0$).
* **Features ($x$):** A common approach is to use a "bag-of-words" representation.
    * Create a dictionary of the top 10,000 common English words.
    * For each email, create a 10,000-dimensional feature vector where $x_j=1$ if word $j$ appears in the email, else $x_j=0$. (Alternatively, $x_j$ could be the word count).
* **Model:** Train a classification algorithm (e.g., logistic regression, neural network) on these features.

### Ideas for Improvement & the Role of Diagnostics:

After an initial model, you'll have many ideas for improvement. Choosing the most promising path is critical for speeding up the project.

* **Tempting Idea: Collect More Data (e.g., Honeypot projects):**
    * Creating fake email addresses to collect known spam emails.
    * **Diagnostic Insight:** Bias/variance analysis tells you if more data will actually help. If your model has high bias, collecting more data won't be fruitful. If it has high variance, more data can help immensely.
* **Develop More Sophisticated Features:**
    * **Email Routing:** Features based on the email's travel path (email headers).
    * **Email Body Features:** More advanced text processing (e.g., treating "discounting" and "discount" as the same word; detecting deliberate misspellings like "w@tches").

Diagnostics (like bias/variance and error analysis) provide the empirical guidance to determine which of these ideas are most likely to improve performance, saving significant time and resources. The next video will detail error analysis.

## Error Analysis: Diagnosing Specific Mistakes

Error analysis is a crucial diagnostic tool, second only to bias and variance analysis, for understanding *why* your learning algorithm is making mistakes and guiding your next steps for improvement.

### The Process:

1.  **Identify Misclassified Examples:** Get a set of examples that your algorithm misclassified from your **cross-validation set** (e.g., if you have 500 CV examples and misclassify 100, focus on those 100).
    * **Sampling:** If the number of misclassified examples is very large (e.g., 1000 out of 5000), randomly sample a manageable subset (e.g., 100-200 examples) for manual review.
2.  **Manual Inspection and Categorization:**
    * **Manually look through each misclassified example.**
    * **Categorize** them by common themes, properties, or traits of the errors. These categories can be overlapping.
    * **Count** how many errors fall into each category.

### Example: Spam Classifier Error Analysis

Suppose you misclassified 100 spam emails on your CV set. You might categorize them as:

* **Pharmaceutical spam:** 21 emails (trying to sell drugs)
* **Deliberate misspellings:** 3 emails (e.g., "w@tches")
* **Unusual email routing:** 7 emails (suspicious paths through servers)
* **Phishing emails:** 18 emails (trying to steal passwords)
* **Embedded image spam:** 15 emails (spam message in an image)

### Insights from Error Analysis:

* **Prioritization:** The counts clearly show where the biggest problems lie.
    * In this example, pharmaceutical spam (21%) and phishing emails (18%) are major issues.
    * Deliberate misspellings (3%) are a much smaller problem.
* **Resource Allocation:** This guides where to invest your time. Spending significant time building a complex algorithm to detect deliberate misspellings might only fix 3 out of 100 errors, leading to minimal overall impact. Error analysis helps avoid such low-impact efforts.
* **Inspiration for Solutions:** Specific error categories can inspire targeted solutions:
    * **Pharmaceutical spam:**
        * Collect more *pharmaceutical-specific* spam data.
        * Engineer new features related to drug names or pharmaceutical product names.
    * **Phishing emails:**
        * Collect more *phishing-specific* email data.
        * Analyze URLs in emails; develop features to detect suspicious URLs.

### Comparison with Bias/Variance Analysis:

* **Bias/Variance Analysis:** Tells you *if* you should get more data (high variance) or try more features/complex model (high bias).
* **Error Analysis:** Tells you *what kind* of data to get, or *what kind* of features to build, or *what specific aspect* of the problem to focus on.

### Limitations:

* Error analysis is easier and more effective for tasks where **humans are good at the task** and can readily identify why the algorithm made a mistake (e.g., understanding email content).
* It's harder for tasks where human intuition is poor (e.g., predicting ad clicks).

Error analysis, combined with bias/variance diagnostics, forms a powerful duo for systematically improving machine learning systems, potentially saving months of fruitless work. The next video will discuss efficient ways to acquire more data when needed.

## Adding Data: Strategies for Machine Learning Applications

Having more data is almost always beneficial for machine learning algorithms, especially when dealing with high variance. However, collecting data can be slow and expensive. This video shares techniques for efficiently acquiring or creating more data.

### 1. Targeted Data Collection

* **Problem:** Collecting "more data of everything" is costly and slow.
* **Solution:** Use **error analysis** to identify specific subsets of data where your model performs poorly (e.g., pharmaceutical spam, phishing emails). Then, focus efforts on collecting **more data of *only* those specific types**.
    * **Example:** For spam classification, if error analysis shows many mistakes on pharmaceutical spam, ask human annotators to specifically find and label more pharmaceutical-related emails from a large pool of unlabeled data.
* **Benefit:** More efficient use of resources, leading to a higher impact on performance for a lower cost.

### 2. Data Augmentation

* **Concept:** Artificially increasing the size of your training dataset by applying domain-specific distortions or transformations to existing training examples. The transformations should preserve the original label.
* **Applications:**
    * **Images (e.g., OCR, A-Z letter recognition):** Rotate, enlarge, shrink, change contrast, flip (if semantically valid, e.g., 'A' flipped is still 'A', but 'b' flipped is not 'd'). You can also use more advanced techniques like elastic warping grids.
    * **Audio (e.g., Speech Recognition):** Add realistic background noise (crowd, car), apply audio distortions (e.g., simulate bad phone connection).
* **Key Principle:** The distortions/changes made to the data should be **representative of the types of noise or variability expected in the *test set***. Adding purely random or unrepresentative noise (e.g., per-pixel random noise to images) is generally not helpful.

### 3. Data Synthesis

* **Concept:** Generating entirely new training examples from scratch, rather than just modifying existing ones. This often requires deep domain knowledge and specialized code.
* **Applications:** Most common in **computer vision**.
    * **Example: Photo OCR (reading text from images):** Instead of taking photos of text, you can:
        * Programmatically render text using various fonts, colors, backgrounds, and contrasts from a computer's text editor.
        * This can create a vast, diverse, and realistic synthetic dataset.
* **Benefit:** Can generate extremely large amounts of data, providing a significant boost to algorithm performance.
* **Challenge:** Can be computationally intensive and requires effort to ensure the synthetic data is realistic enough.

### Model-Centric vs. Data-Centric AI Development

* Historically, ML research focused on the **model-centric approach**: holding the data fixed and spending effort on improving the code/algorithm/model. This has led to highly effective algorithms (linear regression, neural networks, decision trees).
* However, sometimes a **data-centric approach** is more fruitful: holding the code/algorithm fixed and focusing on **engineering the data** used by the algorithm. This includes targeted collection, data augmentation, and data synthesis.
* A data-centric focus can be a very efficient way to improve algorithm performance.

### Beyond Data Addition: Transfer Learning

For applications with **very limited data**, a technique called **transfer learning** can provide a huge performance boost. This involves taking a model pre-trained on a **different, often larger, but related task** and adapting it for your specific application. This is discussed in the next video.

## Transfer Learning: Leveraging Pre-trained Models

**Transfer learning** is a powerful technique for applications with limited data, allowing you to leverage knowledge (parameters) gained from training on a different, often much larger, but related task. This is a very frequently used technique.

### How Transfer Learning Works:

1.  **Supervised Pre-training (or download pre-trained model):**
    * Find a neural network already trained on a **very large dataset** (e.g., 1 million images of cats, dogs, cars, people across 1000 classes).
    * This network learns a good set of parameters for its hidden layers (e.g., $W^{[1]}, b^{[1]}, \dots, W^{[4]}, b^{[4]}$). These layers learn to detect generic, low-level to mid-level features (edges, corners, basic shapes, object parts).
    * You can either train this large network yourself or, more commonly, **download a pre-trained model** (parameters) from researchers who have already published them online.

2.  **Fine-tuning on Your Specific Task:**
    * **Copy the pre-trained network's hidden layers:** Take all layers except the final output layer (e.g., layers 1-4).
    * **Replace the output layer:** Discard the original output layer (e.g., the 1000-unit output layer for cats/dogs) and replace it with a new, smaller output layer tailored to your specific task (e.g., a 10-unit output layer for 0-9 digit recognition).
    * **Initialize new output layer parameters:** The parameters for this new output layer (e.g., $W^{[5]}, b^{[5]}$) are typically initialized randomly.
    * **Train (Fine-tune):** Use an optimization algorithm (gradient descent or Adam) on your *smaller, specific dataset* (e.g., handwritten digits). There are two main options for training:
        * **Option 1: Train Only Output Layer:** Keep the parameters of the copied hidden layers ($W^{[1]}, b^{[1]}, \dots, W^{[4]}, b^{[4]}$) fixed, and only train the new output layer parameters ($W^{[5]}, b^{[5]}$) from scratch. Recommended for very small datasets.
        * **Option 2: Train All Parameters:** Initialize the copied hidden layer parameters with the pre-trained values, and train *all* parameters ($W^{[1]}, b^{[1]}, \dots, W^{[5]}, b^{[5]}$) on your specific dataset. Recommended if your dataset is a bit larger.

### Why Transfer Learning Works (Intuition):

* **Generic Feature Detectors:** The early and mid-layers of a neural network trained on a large, diverse dataset learn to detect general, reusable features (e.g., edges, corners, basic curves, object parts).
* **Transferability:** These generic features are highly useful for many other related tasks. For example, edge detectors learned from cat images are also useful for digit recognition.
* **Better Starting Point:** The network starts with powerful feature extractors, requiring less data and training time to adapt to the new task compared to training from scratch.

### Limitations/Restrictions:

* **Input Type Must Be the Same:** The pre-trained network must have been trained on the **same type of input data** as your application.
    * Images $\rightarrow$ pre-trained on images.
    * Audio $\rightarrow$ pre-trained on audio.
    * Text $\rightarrow$ pre-trained on text.
* **Amount of Data:** Transfer learning is most impactful when your specific application dataset is not very large (e.g., a few dozens to thousands of examples).

### Real-World Examples:

Advanced techniques like GPT-3, BERT, and ImageNet-pretrained models are prime examples of successful transfer learning (supervised pre-training and fine-tuning). These large, publicly available models allow anyone to build high-performing systems even with smaller custom datasets. Transfer learning embodies the spirit of open sharing in the ML community, allowing collective progress.

## The Full Cycle of a Machine Learning Project

Building a valuable machine learning system involves more than just training a model. It encompasses a full cycle from scoping to deployment and maintenance.

### The Project Cycle:

1.  **Project Scoping:**
    * **Define the problem:** Clearly decide what specific task the ML system will address (e.g., speech recognition for voice search on mobile phones).
2.  **Data Collection:**
    * Determine what data is needed (e.g., audio clips and their text transcripts for speech recognition).
    * Perform the work to acquire, clean, and prepare this data.
3.  **Model Training & Iteration:**
    * **Train the model:** Implement and train your chosen ML model.
    * **Diagnostics:** Almost always, the first trained model won't be good enough. Use diagnostics like **error analysis** and **bias/variance analysis** to understand its shortcomings.
    * **Iterate on Improvement:** Based on diagnostics, decide on next steps (e.g., collect more data, specifically targeted data like speech in car noise using augmentation; adjust model architecture or hyperparameters; add/remove features).
    * This loop (train $\rightarrow$ diagnose $\rightarrow$ improve) is repeated multiple times until the model's performance is satisfactory.
4.  **Deployment in Production:**
    * Make the trained model available for real users. This often involves integrating it into a larger software system.
    * **Example (Speech Recognition):**
        * The ML model is hosted on an **inference server**.
        * A mobile app records user audio and makes an **API call** to the inference server.
        * The inference server runs the ML model to generate the text transcript and returns it to the app.
    * This step requires **software engineering** to ensure reliability, efficiency, and scalability (handling millions of users).
5.  **Monitoring & Maintenance:**
    * **Continuous Monitoring:** Track the system's performance in the production environment.
    * **Logging Data:** Log input data ($x$) and predictions ($\hat{y}$) (with user privacy/consent). This data is vital for:
        * **System Monitoring:** Detecting performance degradation (e.g., speech recognition accuracy dropping due to new celebrity names or elections causing shifts in search terms).
        * **Model Updates:** Providing new training data to retrain and update the model when its performance degrades.
    * This ensures the system remains high-performing over time.

### MLOps (Machine Learning Operations):

* This is a growing field focused on the practices and tools for **systematically building, deploying, and maintaining ML systems**.
* It encompasses ensuring reliability, scalability, efficient resource usage, logging, monitoring, and enabling continuous model updates.

Training a high-performing model is critical, but deploying and maintaining it effectively requires additional considerations and potentially specialized MLOps practices, especially for large-scale applications. The next video will discuss the ethics of building ML systems.

## Ethics in Machine Learning: Fairness, Bias, and Responsible AI

Machine learning algorithms impact billions of people globally, making ethical considerations, fairness, and bias crucial in their development and deployment.

### Unacceptable Biases in ML Systems:

History has unfortunately seen widely publicized examples of biased ML systems:

* **Hiring Tools:** Discrimination against women.
* **Face Recognition:** Higher misidentification rates for dark-skinned individuals, particularly in matching to criminal mugshots.
* **Bank Loan Approvals:** Biased decisions discriminating against certain subgroups.
* **Reinforcing Stereotypes:** Algorithms can inadvertently reinforce negative stereotypes (e.g., search results for professions not showing diverse representation), potentially discouraging individuals.

### Adverse Use Cases of ML:

Beyond bias, there are deliberate malicious uses:

* **Deepfakes:** Generating fake videos or audio without consent/disclosure.
* **Spread of Toxic/Incendiary Speech:** Social media algorithms optimizing for engagement can inadvertently promote harmful content.
* **Bots for Fake Content:** Generating fake comments or political propaganda.
* **Fraud/Harmful Products:** ML used by fraudsters (e.g., in financial fraud) or for creating harmful products.

**Ethical Imperative:** It is crucial to **not build ML systems that have a negative societal impact**. If asked to work on an unethical application, it's advised to decline.

### General Guidance for Ethical ML Development:

While there's no simple "ethical checklist," here are suggestions for building more fair, less biased, and more ethical systems:

1.  **Assemble a Diverse Team:**
    * **Benefit:** Diverse teams (in terms of gender, ethnicity, culture, background, etc.) are collectively better at brainstorming and identifying potential harms, especially to vulnerable groups, before deployment.
2.  **Literature Search for Standards/Guidelines:**
    * **Action:** Research existing industry standards or ethical guidelines relevant to your specific application (e.g., financial industry standards for fair loan approval systems). These emerging standards can inform your work.
3.  **Audit the System Prior to Deployment:**
    * **Action:** After training, and *before* deployment, conduct an audit to specifically measure performance across identified dimensions of potential harm (e.g., check for bias against specific genders or racial groups).
    * **Goal:** Identify and fix any problems before the system impacts users.
4.  **Develop a Mitigation Plan (and Monitor for Harm Post-Deployment):**
    * **Action:** Plan for what to do if harm occurs (e.g., roll back to an older, fairer system).
    * **Continuous Monitoring:** Keep monitoring for unexpected harms even after deployment to trigger mitigation plans quickly if issues arise.
    * **Example:** Self-driving car teams have pre-planned mitigation strategies for accidents.

**Conclusion:** Ethics, fairness, and bias are serious considerations in ML. While some projects have more severe ethical implications (e.g., loan approvals vs. coffee roasting optimization), all practitioners should strive to:
* Debate these issues.
* Spot problems early.
* Fix them before they cause harm.

This collective responsibility is vital to ensure ML systems benefit society. The course will now transition to optional videos on handling skewed datasets, a common practical challenge in ML.
