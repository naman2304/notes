## Chapter 1: The Machine Learning Landscape

### What Is Machine Learning?
* Machine Learning is the science and art of programming computers so they can learn from data.
* **A More General Definition**: [Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed. (Arthur Samuel, 1959)
* **An Engineering-Oriented Definition**: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. (Tom Mitchell, 1997)
    * **Example: Spam Filter**:
        * **Task (T)**: To flag spam for new emails.
        * **Experience (E)**: The training data, which consists of example spam and non-spam emails.
        * **Performance Measure (P)**: The ratio of correctly classified emails, also known as **accuracy**.
* The system uses a **training set** to learn. Each example in this set is a **training instance** or **sample**.

### Why Use Machine Learning?
* **Traditional Approach vs. ML Approach**:
    * **Traditional**: You study the problem (e.g., spam), notice patterns (e.g., words like "free," "credit card"), write explicit rules for each pattern, evaluate, and repeat. This leads to a long list of complex, hard-to-maintain rules.
    * **ML Approach**: The ML algorithm automatically learns which words and phrases are good predictors of spam by detecting patterns in example emails. The resulting program is shorter, easier to maintain, and often more accurate.
* **Adapting to Change**: An ML-based spam filter can automatically notice that spammers have started using "For U" instead of "4U" and adapt without manual intervention. A traditional system would need a new rule to be written.
* **Complex Problems with No Known Algorithm**: For problems like speech recognition, the best solution is to use an ML algorithm that learns from many example recordings.
* **Getting Insights (Data Mining)**: ML algorithms can be inspected to see what they have learned. For instance, a spam filter can reveal the best predictors of spam, leading to a better understanding of the problem.
* **Summary of When to Use ML**:
    * Problems requiring a lot of hand-tuning or long lists of rules.
    * Complex problems for which no good solution exists with a traditional approach.
    * Environments that fluctuate, requiring the system to adapt to new data.
    * Gaining insights about complex problems and large datasets.

### Types of Machine Learning Systems
ML systems can be classified based on:
* The amount of human supervision during training (Supervised, Unsupervised, Semisupervised, Reinforcement Learning).
* Whether they can learn incrementally on the fly (Online vs. Batch learning).
* How they generalize, by comparing new data to known data or by building a predictive model (Instance-based vs. Model-based learning).

#### Supervised/Unsupervised Learning
* **Supervised Learning**: The training data fed to the algorithm includes the desired solutions, called **labels**.
    * **Classification**: A typical task is to classify instances into discrete classes. A spam filter is a good example; it's trained with emails and their class (spam or ham).
    * **Regression**: The task is to predict a target numeric value. For example, predicting the price of a car given features like mileage, age, and brand. The features are called **predictors**, and the value is the label.
    * **Key Supervised Algorithms**: k-Nearest Neighbors, Linear Regression, Logistic Regression, Support Vector Machines (SVMs), Decision Trees, and Neural Networks.
* **Unsupervised Learning**: The training data is unlabeled. The system tries to learn without a teacher.
    * **Clustering**: The goal is to detect groups of similar instances. For example, clustering blog visitors to understand different user segments.
    * **Anomaly Detection**: The goal is to detect unusual instances that deviate from the norm, like detecting defective products or credit card fraud.
    * **Dimensionality Reduction**: The goal is to simplify the data by reducing the number of features without losing too much information. This is often done to fight the "curse of dimensionality" and can also be used for data visualization. **Feature extraction** is a common technique where correlated features are merged.
    * **Association Rule Learning**: The goal is to discover interesting relations between attributes in large datasets. For example, discovering that customers who buy barbecue sauce also tend to buy potato chips.
    * **Key Unsupervised Algorithms**: K-Means, DBSCAN (for clustering); Principal Component Analysis (PCA), Kernel PCA (for dimensionality reduction).
* **Semisupervised Learning**: Deals with data that is partially labeled; some instances have labels, but many do not.
    * **Example**: Google Photos automatically clusters photos of the same person (unsupervised part). Once you provide a single label (a name) for that person, it can label every photo of that person (supervised part).
* **Reinforcement Learning (RL)**: The learning system, called an **agent**, can observe the **environment**, select and perform **actions**, and get **rewards** (or penalties) in return. It must learn the best strategy, called a **policy**, to get the most reward over time.
    * **Example**: AlphaGo learned its winning policy by analyzing millions of games and then playing against itself.

#### Batch and Online Learning
* **Batch Learning (Offline Learning)**: The system is trained using all available data at once. It cannot learn incrementally. To learn about new data, it must be retrained from scratch on the full dataset. This is time and resource-intensive.
* **Online Learning**: The system is trained incrementally by feeding it data instances sequentially, either individually or in small groups called **mini-batches**. It's great for systems that need to adapt to change rapidly or for handling huge datasets that can't fit in one machine's memory (**out-of-core learning**).
    * A key parameter is the **learning rate**, which controls how fast the system adapts to new data. A high rate means rapid adaptation but also a tendency to forget old data.

#### Instance-Based Versus Model-Based Learning
This is about how the system generalizes to new instances it has never seen before.
* **Instance-Based Learning**: The system learns the examples by heart and then generalizes to new cases by comparing them to the learned examples using a **similarity measure**.
    * **Example**: k-Nearest Neighbors. To classify a new instance, it finds the most similar training instances and uses their labels to make a prediction.
* **Model-Based Learning**: A model is built from a set of examples, and then that model is used to make predictions.
    * **Process**:
        1.  **Model Selection**: Select a model type (e.g., a linear model).
        2.  **Training**: The algorithm finds the model parameters that best fit the training data. This usually involves minimizing a **cost function**.
        3.  **Inference**: Use the trained model to make predictions on new cases.
    * **Example**: Using a country's GDP per capita to predict life satisfaction. A linear model can be expressed as:
        $life\_satisfaction = \theta_0 + \theta_1 \times \text{GDPpercapita}$
        The parameters $\theta_0$ (bias) and $\theta_1$ (weight) are tuned during training to minimize the error between the model's predictions and the actual training data.

### Main Challenges of Machine Learning
The two main things that can go wrong are "bad algorithm" and "bad data."

#### Bad Data
* **Insufficient Quantity of Training Data**: Most ML algorithms need thousands of examples to work properly, and millions for complex tasks like image recognition. The "unreasonable effectiveness of data" states that for complex problems, more data often matters more than better algorithms.
* **Nonrepresentative Training Data**: The training data must be representative of the new cases you want to generalize to. If not, the model will not generalize well.
    * **Sampling Noise**: Occurs when the sample is too small.
    * **Sampling Bias**: Occurs when the sampling method is flawed, even with large samples (e.g., the 1936 Literary Digest poll that incorrectly predicted Landon would win against Roosevelt).
* **Poor-Quality Data**: If your data is full of errors, outliers, and noise, it will make it harder for the system to detect underlying patterns. Data cleaning is a significant part of a data scientist's work.
* **Irrelevant Features**: The system will only learn if the training data contains enough relevant features and not too many irrelevant ones. This is where **feature engineering** comes in, which includes:
    * **Feature Selection**: Selecting the most useful features.
    * **Feature Extraction**: Combining existing features to create a more useful one.
    * Creating new features by gathering new data.

#### Bad Algorithm
* **Overfitting the Training Data**: This happens when the model is too complex relative to the amount and noisiness of the data. The model performs well on the training data but does not generalize well to new instances.
    * **Solutions**:
        * Simplify the model (fewer parameters or features).
        * Gather more training data.
        * Reduce the noise in the data.
        * Apply **regularization**, which constrains a model to make it simpler and reduce the risk of overfitting. This is controlled by a **hyperparameter**.
* **Underfitting the Training Data**: This is the opposite of overfitting and occurs when your model is too simple to learn the underlying structure of the data.
    * **Solutions**:
        * Select a more powerful model with more parameters.
        * Feed better features to the algorithm (feature engineering).
        * Reduce the constraints on the model (e.g., reduce the regularization hyperparameter).

### Testing and Validating
* **Training and Test Sets**: To know how well a model will generalize, you split your data into a **training set** and a **test set**. You train on the training set and evaluate on the test set. The error rate on the test set is the **generalization error**.
* **Hyperparameter Tuning and Validation Set**: Tuning hyperparameters on the test set is a mistake, as it will lead to overfitting the test set. A common solution is to have a second holdout set called the **validation set** (or **dev set**). You train multiple models with various hyperparameters on the reduced training set (full training set minus the validation set) and select the model that performs best on the validation set. After finding the best model, you retrain it on the full training set (including the validation set) to get the final model.
* **Data Mismatch**: If the data you have for training is not representative of the data that will be used in production, you might create a **train-dev set**. This is a part of the training set that is held out. If the model performs well on the training set but poorly on the train-dev set, it's overfitting. If it performs well on both but poorly on the validation set, there is a data mismatch between your training data and your validation/test data.

### No Free Lunch Theorem
* This theorem states that if you make no assumptions about the data, then there is no reason to prefer one model over any other. There is no model that is a priori guaranteed to work better on all problems. The only way to know which model is best for a given task is to evaluate a few reasonable models.
