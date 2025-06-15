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
