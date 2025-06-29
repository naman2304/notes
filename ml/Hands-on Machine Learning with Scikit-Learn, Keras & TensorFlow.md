## Chapter 1: The Machine Learning Landscape

### What Is Machine Learning?
* Machine Learning is the science and art of programming computers so they can learn from data.
* A more engineering-oriented definition is: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.
* The `training set` is the set of examples the system uses to learn, and each example is a `training instance`.

### Why Use Machine Learning?
* **Simplifying Complex Problems**: For problems where traditional solutions would require extensive fine-tuning or long lists of rules, an ML algorithm can often simplify the code and perform better.
* **Solving Intractable Problems**: For complex issues where a traditional approach yields no good solution, ML techniques might find one.
* **Adapting to Fluctuating Environments**: An ML system can be designed to adapt to new data automatically.
* **Gaining Insights**: ML algorithms can be applied to large datasets to discover patterns that were not immediately apparent, a process known as `data mining`.

### Types of Machine Learning Systems
Machine Learning systems can be classified based on:
* The amount and type of supervision during training.
* Whether they can learn incrementally on the fly.
* How they generalize from data (comparing to known data points or building a predictive model).

#### Supervised/Unsupervised Learning
* **Supervised Learning**: The training data fed to the algorithm includes the desired solutions, called `labels`.
    * `Classification`: A typical task where the model is trained to predict a class (e.g., spam or not spam).
    * `Regression`: A task to predict a target numeric value given a set of `predictors` (features).
* **Unsupervised Learning**: The training data is unlabeled, and the system tries to learn without a teacher.
    * `Clustering`: Algorithms try to detect groups of similar instances.
    * `Anomaly detection`: Aims to detect unusual instances that deviate from the norm. `Novelty detection` is similar but assumes the training set is "clean".
    * `Dimensionality reduction`: Simplifies data by merging correlated features into one (`feature extraction`) without losing too much information.
    * `Association rule learning`: Discovers interesting relations between attributes in large amounts of data.
* **Semisupervised Learning**: Can deal with data that's partially labeled, using plenty of unlabeled instances and few labeled ones.
* **Reinforcement Learning**: A learning system, called an `agent`, observes an `environment`, selects and performs `actions`, and gets `rewards` or `penalties`. The goal is to learn the best strategy, called a `policy`, to maximize rewards over time.

#### Batch and Online Learning
* **Batch Learning**: The system is trained using all available data at once. To learn from new data, it must be retrained from scratch on the full dataset. This is also known as `offline learning`.
* **Online Learning**: The system is trained incrementally by feeding it data instances sequentially, either individually or in small groups called `mini-batches`. This is ideal for systems that need to adapt to changing data rapidly. The `learning rate` is an important parameter that controls how fast the system adapts.

#### Instance-Based Versus Model-Based Learning
* **Instance-Based Learning**: The system learns the examples by heart and generalizes to new cases using a similarity measure to compare them to the learned examples.
* **Model-Based Learning**: A model is built from a set of examples and then used to make predictions. This involves three main steps:
    * `Model selection`: Selecting a type of model (e.g., linear model).
    * `Training`: The learning algorithm finds the model parameter values that minimize a `cost function` or maximize a `utility function`.
    * `Inference`: Applying the trained model to make predictions on new cases.

### Main Challenges of Machine Learning
The two main challenges are "bad algorithm" and "bad data".

#### Insufficient Quantity of Training Data
* Most Machine Learning algorithms require a large amount of data to work properly.

#### Nonrepresentative Training Data
* The training data must be representative of the new cases you want to generalize to.
* This can occur due to `sampling noise` (nonrepresentative data as a result of chance) or `sampling bias` (flawed sampling method).

#### Poor-Quality Data
* If training data is full of errors, outliers, and noise, it will be harder for the system to detect underlying patterns. Data cleaning is a crucial step.

#### Irrelevant Features
* The training data must contain enough relevant features and not too many irrelevant ones.
* `Feature engineering` is the process of creating a good set of features and includes:
    * `Feature selection`: Selecting the most useful features.
    * `Feature extraction`: Combining existing features to produce a more useful one.

#### Overfitting the Training Data
* Overfitting means the model performs well on the training data but does not generalize well to new instances.
* It happens when the model is too complex relative to the amount and noisiness of the data.
* Solutions include simplifying the model, gathering more data, or reducing noise in the data.
* `Regularization` is the process of constraining a model to make it simpler and reduce the risk of overfitting. This is controlled by a `hyperparameter`.

#### Underfitting the Training Data
* Underfitting occurs when your model is too simple to learn the underlying structure of the data.
* Solutions include selecting a more powerful model, using better features, or reducing the constraints on the model.

### Testing and Validating
* Data is split into a `training set` and a `test set`. The model is trained on the training set and evaluated on the test set to estimate the `generalization error`.

#### Hyperparameter Tuning and Model Selection
* A `validation set` (or `dev set`) is held out from the training set to evaluate several candidate models and select the best one. This avoids `data snooping` bias.
* This process is known as `holdout validation`. To avoid imprecise evaluations with small validation sets or training on a much smaller dataset, `repeated cross-validation` can be used.

#### Data Mismatch
* If the training data is not perfectly representative of the data that will be used in production, a `train-dev set` is used.
* If the model performs poorly on the train-dev set, it has overfit the training set. If it performs well on the train-dev set but poorly on the validation set, the issue is a data mismatch.
