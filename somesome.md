## Diagnosing Bias and Variance with Baseline Performance

When training a machine learning model, it's rare for it to work perfectly on the first try. Diagnosing whether the problem is **high bias (underfitting)** or **high variance (overfitting)** is crucial for deciding the next steps. This often involves comparing training error ($$J_{\text{train}}$$) and cross-validation error ($$J_{\text{cv}}$$) against a **baseline level of performance**.
