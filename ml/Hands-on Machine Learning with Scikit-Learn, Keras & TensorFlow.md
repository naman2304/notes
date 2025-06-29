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
        $lifeSatisfaction = \theta_0 + \theta_1 \times \text{GDPperCapita}$  
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
* **Hyperparameter Tuning and Validation Set**:
  * Tuning hyperparameters on the test set is a mistake, as it will lead to overfitting the test set.
  * To avoid this, introduce a validation set (also called a dev set).
  * Data is split into:
    * Training set: used to train models.
    * Validation (or dev) set: used to tune hyperparameters.
    * Test set: used to evaluate final model performance.
  * Tuning Process
    * Set aside part of the training data as the validation set.
    * Train multiple models with different hyperparameter settings on the reduced training set (excluding validation set).
    * Evaluate each model on the validation set.
    * Select the model with the best performance on the validation set.
  * Final Model Training
    * Retrain the selected best model using the entire training data (including the validation set).
    * Evaluate this final model on the test set for unbiased performance estimation.
* **Data Mismatch**: If the data you have for training is not representative of the data that will be used in production, you might create a **train-dev set**. This is a part of the training set that is held out. If the model performs well on the training set but poorly on the train-dev set, it's overfitting. If it performs well on both but poorly on the validation set, there is a data mismatch between your training data and your validation/test data.

### No Free Lunch Theorem
* This theorem states that if you make no assumptions about the data, then there is no reason to prefer one model over any other. There is no model that is a priori guaranteed to work better on all problems. The only way to know which model is best for a given task is to evaluate a few reasonable models.

## Chapter 2: End-to-End Machine Learning Project

Look at the Big Picture

### Frame the Problem
* The first step is to understand the business objective. The model's output is not the end goal, but a component of a larger system.
* **Pipeline**: A sequence of data processing components is called a **data pipeline**. Components in ML systems often run asynchronously, making the system robust.
* **Problem Framing for Housing Price Prediction**:
    * **Supervised Learning**: The task is supervised as we are given labeled training examples (each district has a median housing price).
    * **Regression Task**: We are asked to predict a value (price). Specifically, it's a **multiple regression** problem as we use multiple features, and a **univariate regression** problem as we predict a single value per district.
    * **Batch Learning**: There is no continuous flow of data, and the dataset is small enough to fit in memory, so a plain batch learning approach is suitable.

### Select a Performance Measure
* A typical performance measure for regression tasks is the **Root Mean Square Error (RMSE)**. It gives more weight to large errors.

    $$
    RMSE(\mathbf{X}, h) = \sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(h(\mathbf{x}^{(i)}) - y^{(i)}\right)^2}
    $$

    * $m$ is the number of instances in the dataset.
    * $\mathbf{x}^{(i)}$ is a vector of all the feature values of the $i^{th}$ instance.
    * $y^{(i)}$ is the label (the desired output value) for that instance.
    * $\mathbf{X}$ is a matrix containing all the feature values of all instances.
    * $h$ is the system's prediction function, also called a **hypothesis**. $h(\mathbf{x}^{(i)})$ is the predicted value, noted as $\hat{y}^{(i)}$.
* If there are many outliers, you might prefer the **Mean Absolute Error (MAE)**.

    $$
    MAE(\mathbf{X}, h) = \frac{1}{m}\sum_{i=1}^{m}\left|h(\mathbf{x}^{(i)}) - y^{(i)}\right|
    $$

* **Norms**: Both RMSE and MAE are ways to measure the distance between the vector of predictions and the vector of target values.
    * RMSE corresponds to the Euclidean norm, or $l_2$ norm. It is more sensitive to outliers.
    * MAE corresponds to the Manhattan norm, or $l_1$ norm.

## Get the Data

### Take a Quick Look at the Data Structure
* `pandas.read_csv()`: Loads the data into a DataFrame.
* `.head()`: Shows the top five rows.
* `.info()`: Provides a quick description of the data, including the total number of rows, each attribute's type, and the number of non-null values.
    * Useful for spotting missing values. For example, `total_bedrooms` has missing values.
* `.value_counts()`: For categorical attributes (like `ocean_proximity`), this shows the categories and how many districts belong to each.
* `.describe()`: Shows a summary of the numerical attributes (count, mean, std, min, max, and percentiles).
* `.hist()`: Plots a histogram for each numerical attribute. Histograms help understand the data's distribution.
    * **Key Observations from Histograms**:
        * Some attributes are capped (e.g., `housing_median_age`, `median_house_value`). Capping the target attribute can be a problem as the model may learn that prices never go beyond that limit.
        * Attributes have very different scales, which will require feature scaling.
        * Many histograms are **tail-heavy**, meaning they extend much farther to one side. This may require transformations (e.g., computing their logarithm) to get more bell-shaped distributions.

### Create a Test Set
* It's crucial to create a test set and put it aside *before* inspecting the data further to avoid **data snooping bias**. Your brain might spot patterns in the test set, leading you to choose a model that is biased toward those patterns, resulting in an overly optimistic evaluation.
* **Random Sampling**: For large datasets, picking instances randomly is generally fine.
* **Stratified Sampling**: When a dataset is not large enough, random sampling can introduce significant sampling bias. To avoid this, use stratified sampling. The population is divided into homogeneous subgroups called **strata**, and the right number of instances is sampled from each stratum to ensure the test set is representative of the overall population.
    * **Example**: The `median_income` is a very important attribute. To ensure the test set represents the various income categories, you can create an income category attribute and sample based on that to maintain the same proportions in the test set as in the full dataset.

## Discover and Visualize the Data to Gain Insights

### Visualizing Geographical Data
* For geographical data, a scatterplot of latitude and longitude is a good start.
* Setting the `alpha` option to a smaller value (e.g., 0.1) can help visualize the density of data points.
* Advanced plots can convey more information. For instance, using the radius of each circle to represent population (`s` parameter) and color to represent price (`c` parameter) can reveal that housing prices are strongly related to location and population density.

### Looking for Correlations
* The **standard correlation coefficient** (Pearson's r) measures the linear correlation between attributes. It ranges from -1 to 1.
    * Use the `.corr()` method on the DataFrame.
    * **Important**: The correlation coefficient only measures *linear* relationships. It may completely miss non-linear relationships.
* The `pandas.plotting.scatter_matrix()` function plots every numerical attribute against every other. It's a great way to spot correlations visually. The diagonal plots histograms of each attribute.

### Experimenting with Attribute Combinations
* Before feeding data to an ML algorithm, try combining attributes to create new, more useful features.
* **Example**:
    * `total_rooms` in a district is not very useful. `rooms_per_household` is more informative.
    * `bedrooms_per_room` can be more telling than `total_bedrooms`.
    * These new attributes often show a stronger correlation with the target variable (`median_house_value`).

## Prepare the Data for Machine Learning Algorithms
It's best to write functions for data preparation to make the process reproducible.

### Data Cleaning
* Most ML algorithms can't work with missing features.
* **Options for handling missing values** (e.g., in `total_bedrooms`):
    1.  Get rid of the corresponding instances (rows). `housing.dropna(subset=["total_bedrooms"])`
    2.  Get rid of the whole attribute (column). `housing.drop("total_bedrooms", axis=1)`
    3.  Set the values to some value (zero, mean, median). `housing["total_bedrooms"].fillna(median, inplace=True)`
* Scikit-Learn provides `SimpleImputer` to handle missing values. It's preferable because it can store the computed median/mean and apply it to the test set and new data.

### Handling Text and Categorical Attributes
* Most ML algorithms work with numbers, so text and categorical attributes need to be converted.
* **From Text Categories to Numbers**:
    * `OrdinalEncoder`: Maps each category to a different integer. This is risky because ML algorithms will assume that two nearby values are more similar than two distant values (e.g., category 0 is more similar to 1 than to 4), which is not true for a feature like `ocean_proximity`.
    * **One-Hot Encoding**: A better solution for nominal categorical attributes. It creates one binary attribute per category. Only one attribute is "hot" (1) at a time, while the others are "cold" (0). Use `OneHotEncoder`. The output is a SciPy sparse matrix to save memory.

### Custom Transformers
* For custom cleanup or combining attributes, you can create your own transformers that work with Scikit-Learn pipelines.
* Create a class and implement three methods: `fit()`, `transform()`, and `fit_transform()`. By inheriting from `BaseEstimator` and `TransformerMixin`, you get `fit_transform()` and methods for hyperparameter tuning for free.

### Feature Scaling
* ML algorithms generally don't perform well when input numerical attributes have very different scales.
* Two common methods are:
    * **Min-max scaling (Normalization)**: Values are shifted and rescaled to range from 0 to 1. Sensitive to outliers. Use `MinMaxScaler`.
    * **Standardization**: Subtracts the mean value and divides by the standard deviation. Does not bound values to a specific range but is much less affected by outliers. Use `StandardScaler`.
* **Important**: Fit scalers to the training data *only*, then use them to transform the training set, the test set, and new data.

### Transformation Pipelines
* Scikit-Learn's `Pipeline` class helps with sequences of transformations. It takes a list of name/estimator pairs.
* The `ColumnTransformer` is even more useful. It allows you to apply different transformations to different columns. You can apply a pipeline of transformations for numerical columns and a different one (e.g., `OneHotEncoder`) for categorical columns in a single step.

## Select and Train a Model

### Training and Evaluating on the Training Set
* After data preparation, you can train a model.
* A **Linear Regression** model might show that the model is **underfitting** the data (the error is high).
* A more powerful model, like a **DecisionTreeRegressor**, might achieve a perfect score on the training data (RMSE of 0). This is a clear sign of severe **overfitting**.

### Better Evaluation Using Cross-Validation
* A better way to evaluate models is using Scikit-Learn's **K-fold cross-validation** feature.
* It splits the training set into K folds, then trains and evaluates the model K times, picking a different fold for evaluation each time and training on the other K-1 folds.
* This provides not only a performance estimate but also a measure of how precise that estimate is (the standard deviation).
* Using cross-validation, the `DecisionTreeRegressor` is shown to perform worse than the `LinearRegression` model, confirming it overfit the data.
* A `RandomForestRegressor`, which is an **Ensemble** model, works much better.

## Fine-Tune Your Model

### Grid Search
* To find the best combination of hyperparameters, use Scikit-Learn’s `GridSearchCV`.
* You tell it which hyperparameters to experiment with and what values to try, and it uses cross-validation to evaluate all possible combinations.

### Randomized Search
* When the hyperparameter search space is large, `RandomizedSearchCV` is preferable.
* Instead of trying all combinations, it evaluates a given number of random combinations, which is more efficient.

### Analyze the Best Models and Their Errors
* Inspect the best models to gain insights. For example, `RandomForestRegressor` can indicate the relative importance of each feature via `feature_importances_`. This might lead to dropping less useful features.

### Evaluate Your System on the Test Set
* After fine-tuning, you evaluate the final model on the test set to estimate the generalization error.
* **Important**: Do not tweak your model after this step, as you would start overfitting the test set. You can compute a 95% confidence interval for the generalization error.

## Launch, Monitor, and Maintain Your System
* Get your solution ready for production (polish code, write tests, etc.).
* Deploy the model.
* Write monitoring code to check live performance and trigger alerts if it drops. Models can "rot" over time as data evolves.
* Retrain your models on fresh data regularly, automating the process as much as possible.
