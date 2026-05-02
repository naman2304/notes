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

## Evaluating Skewed Datasets: Precision and Recall (Optional)

When the ratio of positive to negative examples in a binary classification problem is highly skewed (e.g., 99.5% negative, 0.5% positive), standard **classification accuracy** can be a misleading metric.

### The Problem with Accuracy on Skewed Data:

* **Example: Rare Disease Detection**
    * Suppose only 0.5% of patients have a rare disease ($y=1$).
    * A "dumb" algorithm that *always predicts $y=0$* (no disease) would achieve 99.5% accuracy.
    * This high accuracy is deceptive, as the algorithm never correctly identifies any positive cases, making it useless in practice.
* **Issue:** High accuracy doesn't guarantee a useful model if the model simply ignores the rare class.

### Solution: Precision and Recall

<img src="/metadata/precision_recall.png" width="300" />

For skewed datasets, **precision** and **recall** are more informative metrics. To define them, we use a **confusion matrix**:

| Actual Class \ Predicted Class | Predicted 1 (Positive) | Predicted 0 (Negative) |
| :----------------------------- | :--------------------- | :--------------------- |
| **Actual 1 (Positive)** | True Positive (TP)     | False Negative (FN)    |
| **Actual 0 (Negative)** | False Positive (FP)    | True Negative (TN)     |

* **True Positive (TP):** Actual = 1, Predicted = 1 (Correctly identified positive)
* **False Negative (FN):** Actual = 1, Predicted = 0 (Missed positive, Type II error)
* **False Positive (FP):** Actual = 0, Predicted = 1 (Incorrectly identified positive, Type I error)
* **True Negative (TN):** Actual = 0, Predicted = 0 (Correctly identified negative)

#### Metrics Definitions:

1.  **Accuracy:** "proportion of all classifications that were correct, whether positive or negative"
    $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
    * Avoid for imbalanced datasets.
    
3.  **Recall (true positive rate):** "Of all that were *actually positive*, what fraction did we *correctly detect*?" (we want this ideally to be 1)
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
    * High recall means the model finds most of the actual positive cases. (Minimizes false negatives).
    * Use when false negatives are more expensive than false positives.

4.  **False positive rate:** "probability of false alarm" (we want this ideally to be 0)
    $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
    * High recall means the model finds most of the actual positive cases. (Minimizes false negatives).
    * Use when false positives are more expensive than false negatives.
   
5.  **Precision:** "Of all that we *predicted as positive*, what fraction were *actually positive*?" (we want this ideally to be 1)
    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
    * High precision means when the model predicts positive, it's usually correct. (Minimizes false positives).
    * Use when it's very important for positive predictions to be accurate.


Precision improves as false positives decrease, while recall improves when false negatives decrease. But increasing the classification threshold tends to decrease the number of false positives and increase the number of false negatives, while decreasing the threshold has the opposite effects. As a result, precision and recall often show an inverse relationship, where improving one of them worsens the other.

### Example Calculation:

Assume a CV set of 100 examples:
* TP = 15
* FP = 5
* FN = 10
* TN = 70
* (Total Actual Positives = TP + FN = 25; Total Actual Negatives = FP + TN = 75)

So:
* **Precision:** $\frac{15}{15 + 5} = \frac{15}{20} = 0.75$ (75%)
* **Recall:** $\frac{15}{15 + 10} = \frac{15}{25} = 0.60$ (60%)

### Detecting "Dumb" Algorithms:

* If an algorithm always predicts $y=0$, then $TP=0$ and $FP=0$.
    * Precision becomes undefined (0/0), but is usually treated as 0.
    * Recall becomes $0 / (0 + FN) = 0$.
* Both precision and recall would be 0, clearly indicating a useless algorithm despite potentially high accuracy.

By looking at both precision and recall, you can ensure your model is both accurate in its positive predictions and effective at finding most of the true positive cases, making it genuinely useful for skewed datasets. The next video will discuss the trade-off between precision and recall.

## Precision-Recall Trade-off and F1-Score (Optional)

In binary classification with skewed datasets, there's often a **trade-off between precision and recall**. Ideally, we want both to be high.

* **High Precision:** When the model predicts positive, it's highly likely to be correct. (Minimizes false positives).
* **High Recall:** The model finds most of the actual positive cases. (Minimizes false negatives).

### The Trade-off with Thresholding:

Logistic regression outputs a probability $f(x)$ between 0 and 1. We typically use a threshold (e.g., 0.5) to convert this probability into a binary prediction ($\hat{y}=1$ if $f(x) \ge \text{threshold}$, else $\hat{y}=0$).

* **Raising the Threshold (e.g., from 0.5 to 0.7 or 0.9):**
    * **Impact:** The model predicts $y=1$ only when it's more confident.
    * **Result:** **Higher Precision**, but **Lower Recall**. (Fewer false positives, but more missed true positives). This is preferred if false positives are very costly (e.g., unnecessary invasive medical procedures).

* **Lowering the Threshold (e.g., from 0.5 to 0.3):**
    * **Impact:** The model is more willing to predict $y=1$.
    * **Result:** **Lower Precision**, but **Higher Recall**. (More false positives, but fewer missed true positives). This is preferred if missing true positives is very costly (e.g., untreated severe diseases).

By adjusting this threshold, you can balance the trade-off. Plotting a Precision-Recall curve (precision vs. recall for different thresholds) helps visualize this trade-off and manually pick a suitable operating point.

<img src="/metadata/pr_thres.png" width="300" />

### Combining Precision and Recall: The F1-Score

When comparing multiple algorithms, and they have different precision/recall values (e.g., Algorithm 2: high precision, low recall; Algorithm 3: low precision, high recall), it's hard to decide which is "best." A single metric combining them is useful.

* **Simple Average (P+R)/2:** This is generally **NOT recommended**. It can be misleading, as an algorithm with very high recall but very low precision (e.g., predicting positive always) might get a deceptively high average score.
* **F1-Score:** The most common way to combine precision and recall into a single metric. It is the **harmonic mean** of precision (P) and recall (R).
    $$F1 = \frac{2 \times P \times R}{P + R}$$
    * **Alternatively calculated as:** $F1 = \frac{1}{\frac{1}{2} \left( \frac{1}{P} + \frac{1}{R} \right)}$
    * **Property:** The F1-score gives more emphasis to the lower of the two values (precision or recall). If either P or R is very low, F1 will also be low, indicating that the algorithm is not useful.
    * **Usage:** You can choose the algorithm or threshold that maximizes the F1-score as an automated way to balance precision and recall.

This concludes the practical tips for building ML systems. Next week, we'll cover Decision Trees, another powerful ML algorithm.

## Handling Class Imbalance

### Upsampling (Minority Class Oversampling)

* **Method:** Increase the number of samples in the minority class by duplicating existing ones.
* **Pros:**
    * No information loss from the original minority samples.
    * Helps the model learn patterns specific to the minority class.
* **Cons:**
    * Risk of **overfitting** to the duplicated samples.

### Downsampling (Majority Class Undersampling)

* **Method:** Reduce the number of samples in the majority class to match the size of the minority class.
* **Pros:**
    * Reduces training time significantly.
    * Generally less prone to overfitting than upsampling.
* **Cons:**
    * **Potential loss of important information** from the discarded majority class samples.
    * Lower data efficiency, as you're effectively throwing data away.

## Imbalanced Datasets

* A dataset is **imbalanced** when one label (**majority class**) is much more common than another (**minority class**).
* An imbalanced dataset can cause problems if there are too few minority class examples for the model to train on effectively, leading to poor predictions for the minority class.
* The severity of imbalance is categorized by the percentage of data belonging to the minority class:
    * **Mild**: 20-40%
    * **Moderate**: 1-20%
    * **Extreme**: less than 1%

### Downsampling and Upweighting

* **Downsampling** and then **upweighting** are techniques used to address imbalanced datasets by adjusting the training data to give the model more exposure to the minority class.
* The general process is:
    1.  **Downsample the majority class**: Reduce the number of majority class examples in the training set to a smaller, more balanced ratio relative to the minority class. This helps during training as within mini batches, it is more likely now to see positive labels (say if we choose mini batch size to be 50)
    2.  **Upweight the downsampled class**: Increase the **example weight** of the downsampled majority class examples. This makes each of these examples more important to the model during loss calculation, effectively "undoing" the downsampling for the purpose of the final prediction while retaining the benefits of a balanced training set.
* The example weight should be equal to the downsampling factor. For example, if you downsample by a factor of 10, the example weight should be 10.
* This combination helps to reduce **prediction bias** and ensures that mini-batches contain enough minority class examples for effective training.

#### Example
A model detecting a rare disease.
* **Dataset**: 990 people without the disease (majority class), 10 people with the disease (minority class).
* **Model Behavior**: A model might learn to always predict "no disease" and achieve 99% accuracy. This is a form of **prediction bias** because it fails to correctly identify the rare, but important, disease cases.

This two-step process is used to force the model to pay more attention to the rare data.

##### 1. Downsampling the Majority Class

* **Action**: We reduce the number of examples from the majority class in our training data.
* **Example**: From the 990 "no disease" cases, we only select a random subset, like 40 examples.
* **Result**: Our new, smaller training set is more balanced, with a ratio of 10 "disease" cases to 40 "no disease" cases (a 1:4 ratio instead of 1:99). This makes the minority class more visible during training.

##### 2. Upweighting the Downsampled Class

* **Action**: We assign a higher **example weight** to the downsampled majority class examples. The weight is proportional to the factor by which we downsampled. This "weight" is completely diff from weight as model parameters. 
* **Example**: We reduced the majority class from 990 to 40, a factor of roughly 25. So, we give each of the 40 "no disease" examples a weight of 25.
* **Result**: This ensures that while the model sees a balanced number of examples during training, the overall impact of the majority class on the model's loss calculation is still proportional to its true prevalence in the real world. This helps to reduce prediction bias and prevents the model from overcorrecting and over-predicting the minority class.

### Rebalancing Ratios

* The ideal rebalancing ratio is a hyperparameter that needs to be tuned through experimentation.
* The optimal ratio depends on the batch size, the original imbalance ratio, and the size of the training set.
* A key goal is to ensure that each training **mini-batch** contains multiple examples of the minority class to enable proper learning.

### Precision & Recall

* These metrics are crucial for **evaluating** model performance in the presence of class imbalance; they do not directly solve the imbalance.
* They are used instead of raw accuracy:
    * **Precision:** Of all instances predicted as positive, how many were actually correct positives?
    * **Recall:** Of all actual positive instances, how many were correctly detected by the model?

## [Week 4] Introduction to Decision Trees

Decision trees are powerful and widely used machine learning algorithms, particularly popular for winning competitions, even if they receive less academic attention than neural networks. They are a valuable tool for classification and regression.

### Running Example: Cat Classification

* **Problem:** Classify if an animal is a cat or not, based on a few features.
* **Training Data (10 examples: 5 cats, 5 dogs):**
    * **Features (X):**
        * `Ear Shape`: (categorical: pointy, floppy)
        * `Face Shape`: (categorical: round, not round)
        * `Whiskers`: (categorical: present, absent)
    * **Target (Y):** (binary: cat=1, not cat=0)
    * Initially, all features are binary/categorical. Later, continuous features will be discussed.

### What is a Decision Tree?

<img src="/metadata/decision_tree.png" width="600" />

A decision tree is a model that looks like a flowchart. It consists of:

* **Nodes:** The ovals or rectangles in the tree.
* **Root Node:** The topmost node where the classification process begins.
* **Decision Nodes:** Oval-shaped nodes (excluding the leaf nodes). They contain a feature and direct the flow down the tree based on that feature's value (e.g., "Ear Shape? Pointy -> Left, Floppy -> Right").
* **Leaf Nodes:** Rectangular-shaped nodes at the bottom. They represent the final prediction (e.g., "Cat" or "Not Cat").

### How a Decision Tree Makes a Prediction:

Let's classify a new animal: (Ear Shape: pointy, Face Shape: round, Whiskers: present)

1.  **Start at Root Node:** (e.g., "Ear Shape?")
2.  **Follow Branch:** If "pointy", go down the left branch to the next node.
3.  **Evaluate Next Feature:** At the new node (e.g., "Face Shape?"), check the animal's face shape.
4.  **Continue Down Tree:** If "round", follow the branch to a leaf node.
5.  **Prediction:** The leaf node indicates the final classification (e.g., "Cat").

### Learning a Decision Tree:

* There are many possible decision trees for a given dataset.
* The job of the **decision tree learning algorithm** is to select the tree that performs well on the training data and ideally generalizes well to new, unseen data (cross-validation and test sets).

The next video will delve into how an algorithm learns to construct a specific decision tree from a training set.

## Building a Decision Tree: The Learning Process

Building a decision tree involves iteratively splitting the training data based on features, aiming to create "pure" (single-class) leaf nodes.

### Overall Process:

1.  **Choose Root Node Feature:**
    * Start with the entire training set (e.g., 10 cat/dog examples).
    * Use an algorithm (discussed later) to decide which feature (e.g., "Ear Shape") provides the "best" initial split for the root node.
2.  **Split Data:**
    * Divide the training examples into subsets based on the value of the chosen feature (e.g., 5 "pointy ears" examples to the left branch, 5 "floppy ears" examples to the right branch).
3.  **Recursively Build Sub-trees (Repeat for each branch):**
    * For each new subset of data:
        * **Choose Next Feature to Split:** Again, use an algorithm to select the best feature to split on *within that subset*. (e.g., for "pointy ears" subset, choose "Face Shape").
        * **Split Subset:** Divide the current subset further based on the new feature's values (e.g., 4 "round face" examples to the left, 1 "not round face" example to the right).
        * **Create Leaf Nodes (Stopping Condition):** If a subset becomes "pure" (all examples belong to a single class, e.g., all 4 "round face" examples are cats), create a leaf node making that prediction. If not pure, continue splitting.
    * This process is applied recursively to all branches (e.g., after the left branch is done, build the right branch starting from "floppy ears").

### Key Decisions in Decision Tree Learning:

Two crucial decisions are made at various steps:

1.  **How to Choose Which Feature to Split On at Each Node?**
    * **Goal: Maximize Purity (or Minimize Impurity).** You want to find a feature that, when used for a split, results in child nodes that are as "pure" as possible (i.e., contain mostly examples of a single class, like all cats or all dogs).
    * **Example:** A "cat DNA" feature (if it existed) would be ideal, as it would create perfectly pure (100% cat, 0% cat) subsets.
    * **Challenge:** With real features (ear shape, face shape, whiskers), the algorithm must compare how well each feature splits the data into purer subsets. The feature that leads to the greatest "purity" gain (or greatest "impurity" reduction) is chosen.
    * **Next Video:** The concept of **entropy** will be introduced to measure impurity.

2.  **When to Stop Splitting (Create a Leaf Node)?**
    * **Pure Nodes:** Stop when a node contains only examples of a single class (e.g., all cats, or all dogs). This is the most natural stopping point.
    * **Maximum Depth:** Limit the maximum allowed depth of the tree. (Depth 0 is the root node, Depth 1 for its children, etc.). This prevents the tree from becoming too large/unwieldy and helps **reduce overfitting**.
    * **Minimum Purity Improvement:** Stop if splitting a node yields only a very small improvement in purity (or a small decrease in impurity). This again helps keep the tree smaller and **reduce overfitting**.
    * **Minimum Number of Examples:** Stop splitting if the number of training examples at a node falls below a certain threshold. This also helps keep the tree smaller and prevent overfitting to tiny subsets of data.

Decision tree algorithms can feel complicated due to these various refinements developed over time. However, these pieces work together to create effective learning algorithms. The next video will formally define **entropy** as a measure of impurity, a core concept for choosing optimal splits.

## Entropy: Measuring Purity (or Impurity)

In decision tree learning, we need a way to quantify how "pure" a set of examples is at a given node. **Entropy** is a common measure of **impurity**.

### Definition of Entropy

Given a set of examples:
* Let $p_1$ be the **fraction of positive examples** (e.g., cats, label $y=1$).
* Let $p_0 = 1 - p_1$ be the **fraction of negative examples** (e.g., dogs, label $y=0$).

The **entropy** $H(p_1)$ is defined as:
$$H(p_1) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)$$ or equivalently: $$H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)$$

* **Logarithm Base:** $\log_2$ (logarithm to base 2) is conventionally used, making the maximum entropy value 1.
* **Convention for $0 \log_2(0)$:** By convention, $0 \log_2(0)$ is taken to be $0$.

### Intuition and Examples:

* **Completely Pure Set (All one class):**
    * If all examples are cats ($p_1 = 1$, $p_0 = 0$): $H(1) = -1 \log_2(1) - 0 \log_2(0) = -1 \times 0 - 0 = 0$.
    * If all examples are dogs ($p_1 = 0$, $p_0 = 1$): $H(0) = -0 \log_2(0) - 1 \log_2(1) = 0 - 1 \times 0 = 0$.
    * **Result:** Entropy is **0** when the set is perfectly pure (contains only one class). This signifies zero impurity.

* **Maximally Impure Set (50-50 Mix):**
    * If there's an equal mix of cats and dogs ($p_1 = 0.5$, $p_0 = 0.5$): $H(0.5) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = -0.5(-1) - 0.5(-1) = 0.5 + 0.5 = 1$.
    * **Result:** Entropy is **1** (its maximum value) when the set is maximally impure (a 50-50 mix of classes).

* **Intermediate Impurity:**
    * If $p_1 = 5/6 \approx 0.83$ (5 cats, 1 dog): $H(0.83) \approx 0.65$. (Less impure than 50-50, more impure than all one class).
    * If $p_1 = 2/6 \approx 0.33$ (2 cats, 4 dogs): $H(0.33) \approx 0.92$. (More impure than 5 cats/1 dog, closer to 50-50).

### Entropy Curve:

<img src="/metadata/entropy_curve.png" width="300" />

The entropy function $H(p_1)$ forms a curve that starts at 0 (for $p_1=0$), rises to a peak of 1 (for $p_1=0.5$), and then falls back to 0 (for $p_1=1$).

### Other Impurity Measures:

While entropy is common, other functions like the **Gini criteria (or Gini impurity)** also measure impurity similarly (from 0 to 1) and are used in decision trees. For simplicity, this course focuses on entropy.

Now that we have a way to measure impurity (entropy), the next video will show how to use it to decide which feature to split on at each node of a decision tree.

## Information Gain: Choosing the Best Split

When building a decision tree, the primary criterion for deciding which feature to split on at a given node is to choose the feature that leads to the **greatest reduction in entropy (impurity)**. This reduction is called **information gain**.

### How to Calculate Information Gain:

Let's illustrate with our cat classification example:

1.  **Calculate Entropy at the Root Node (before any split):**
    * For the initial set of examples at the node (e.g., the root node with all 10 examples: 5 cats, 5 dogs).
    * $p_{1}^{\text{root}} = \text{Fraction of cats at root} = 5/10 = 0.5$.
    * $H(\text{root}) = \text{Entropy}(p_{1}^{\text{root}}) = \text{Entropy}(0.5) = 1$. (Maximum impurity).

2.  **For each candidate feature to split on (e.g., Ear Shape, Face Shape, Whiskers):**
    *  **Hypothetically split the data** based on that feature. This creates left and right sub-branches.
    *  **Calculate $p_1$ and Entropy for each sub-branch:**
        * **Ear Shape Split (Pointy vs. Floppy):**
            * Left (Pointy): 5 examples total, 4 cats. $p_{1}^{\text{left}} = 4/5 = 0.8$. $\text{Entropy}(0.8) \approx 0.72$.
            * Right (Floppy): 5 examples total, 1 cat. $p_{1}^{\text{right}} = 1/5 = 0.2$. $\text{Entropy}(0.2) \approx 0.72$.
        * **Face Shape Split (Round vs. Not Round):**
            * Left (Round): 7 examples total, 4 cats. $p_{1}^{\text{left}} = 4/7 \approx 0.57$. $\text{Entropy}(0.57) \approx 0.99$.
            * Right (Not Round): 3 examples total, 1 cat. $p_{1}^{\text{right}} = 1/3 \approx 0.33$. $\text{Entropy}(0.33) \approx 0.92$.
        * **Whiskers Split (Present vs. Absent):**
            * Left (Present): 4 examples total, 3 cats. $p_{1}^{\text{left}} = 3/4 = 0.75$. $\text{Entropy}(0.75) \approx 0.81$.
            * Right (Absent): 6 examples total, 2 cats. $p_{1}^{\text{right}} = 2/6 \approx 0.33$. $\text{Entropy}(0.33) \approx 0.92$.

    *  **Calculate Weighted Average Entropy of the Split:** This accounts for the proportion of examples going into each sub-branch.
        * **For Ear Shape:**
            * $w^{\text{left}} = 5/10 = 0.5$ (proportion of examples with pointy ears)
            * $w^{\text{right}} = 5/10 = 0.5$ (proportion of examples with floppy ears)
            * Weighted Entropy = $(0.5 \times \text{Entropy}(0.8)) + (0.5 \times \text{Entropy}(0.2)) = (0.5 \times 0.72) + (0.5 \times 0.72) = 0.72$.
        * **For Face Shape:**
            * Weighted Entropy = $(7/10 \times \text{Entropy}(4/7)) + (3/10 \times \text{Entropy}(1/3)) = (0.7 \times 0.99) + (0.3 \times 0.92) = 0.97$.
        * **For Whiskers:**
            * Weighted Entropy = $(4/10 \times \text{Entropy}(3/4)) + (6/10 \times \text{Entropy}(2/6)) = (0.4 \times 0.81) + (0.6 \times 0.92) = 0.88$.

    *  **Calculate Information Gain:**
        $$\text{Information Gain} = \text{Entropy}(\text{root}) - \text{Weighted Average Entropy of Split}$$
        * **For Ear Shape:** $1 - 0.72 = 0.28$.
        * **For Face Shape:** $1 - 0.97 = 0.03$.
        * **For Whiskers:** $1 - 0.88 = 0.12$.

### Choosing the Best Split:

* The feature with the **highest information gain** is chosen for the split.
* In this example, "Ear Shape" has the highest information gain (0.28), so it would be chosen as the root node feature.
* **Why Information Gain?** It directly quantifies how much a split reduces the overall impurity of the dataset, leading to purer child nodes.

$$\text{Information gain} = H(p_1^{\text{root}}) - \left( w^{\text{left}} H(p_1^{\text{left}}) + w^{\text{right}} H(p_1^{\text{right}}) \right)$$

### Role in Stopping Criteria:

* Information gain is also used in stopping criteria: If the information gain from any potential split is below a certain threshold, the algorithm might decide not to split further, creating a leaf node instead. This helps control tree size and prevent overfitting.

The next video will integrate this information gain calculation into the overall decision tree building algorithm.

## Building a Decision Tree: The Overall Process

Building a decision tree involves a recursive process of splitting nodes based on features that provide the most information gain, until predefined stopping criteria are met.

### Overall Algorithm:

1.  **Start at Root Node:** Begin with all training examples at the root node.
2.  **Select Best Split Feature:**
    * Calculate the **information gain** for every possible feature (e.g., "Ear Shape," "Face Shape," "Whiskers").
    * Choose the feature that yields the **highest information gain**.
3.  **Perform Split:**
    * Divide the dataset into subsets based on the values of the chosen feature.
    * Create corresponding child branches (e.g., "pointy ears" branch, "floppy ears" branch).
    * Send the relevant training examples down each branch.
4.  **Recursively Build Sub-trees:**
    * **Repeat the splitting process** for each new branch (child node) created. The process is the same as building the main tree, but applied to a subset of data.
5.  **Stop Splitting (Stopping Criteria):** Stop the recursive splitting process for a branch when any of the following criteria are met:
    * **Node Purity:** All examples in the node belong to a single class (entropy is 0). This becomes a leaf node with a clear prediction.
    * **Maximum Depth:** The tree (or current branch) reaches a pre-defined `maximum depth`. This prevents overly complex trees and reduces overfitting.
    * **Minimum Information Gain:** The information gain from any potential further split is below a specified `threshold`. This avoids trivial splits.
    * **Minimum Examples per Node:** The number of training examples in the current node falls below a certain `threshold`. This also prevents overfitting to tiny subsets.

### Illustration of the Process:

* **Root Node:** All 10 examples. "Ear Shape" is chosen (highest info gain).
    * Splits into "Pointy Ears" (5 examples) and "Floppy Ears" (5 examples) branches.
* **Left Branch ("Pointy Ears"):** Focus on these 5 examples.
    * **Check stopping criteria:** Not pure yet (mix of cats and dogs).
    * **Select Best Split Feature:** (e.g., "Face Shape" is chosen after calculating information gain within this subset).
    * Splits into "Round Face" (4 examples) and "Not Round Face" (1 example) sub-branches.
    * **"Round Face" sub-branch:** All 4 examples are cats. **Stopping criteria met (purity).** Create a "Cat" leaf node.
    * **"Not Round Face" sub-branch:** All 1 example is a dog. **Stopping criteria met (purity).** Create a "Not Cat" leaf node.
* **Right Branch ("Floppy Ears"):** Similarly, recursively build this branch.
    * (e.g., "Whiskers" is chosen).
    * Splits into "Whiskers Present" and "Whiskers Absent" sub-branches, which then become pure leaf nodes.

### Recursive Algorithm:

Decision tree building is a classic example of a **recursive algorithm** in computer science. The main function to build a tree calls itself to build sub-trees on smaller subsets of the data.

### Key Parameters:

* **Maximum Depth:** Controls tree size and complexity. Larger depth increases risk of overfitting. Can be tuned via cross-validation, but open-source libraries often have good defaults.
* **Information Gain Threshold:** Controls when to stop splitting.
* **Minimum Examples per Node:** Controls when to stop splitting.

Understanding these parameters and their impact on tree size and overfitting is crucial for effective decision tree usage. After building, predictions are made by traversing the tree from root to leaf based on a new example's features.

The next videos will explore handling features with more than two categorical values and continuous-valued features.

## Handling Categorical Features with Many Values: One-Hot Encoding

So far, our decision tree examples have used features with only two discrete values (e.g., pointy/floppy, round/not round). This video introduces **one-hot encoding** as a method to handle categorical features that can take on *more than two* discrete values.

### The Problem: Categorical Features with $>2$ Values

Consider the `Ear Shape` feature, which can be `pointy`, `floppy`, or `oval`. If we directly use this feature, a decision tree would create three branches from a single node. While decision trees can technically handle this, one-hot encoding offers an alternative, especially useful for other algorithms.

### Solution: One-Hot Encoding

One-hot encoding transforms a single categorical feature with $K$ possible values into $K$ new **binary (0 or 1) features**.

* **Process:**
    1.  Identify all unique values (categories) a feature can take.
    2.  Create a new binary feature for each unique value.
    3.  For any given example, set the binary feature corresponding to its category to `1` ("hot"), and all other new binary features for that original categorical feature to `0`.

* **Example: Ear Shape (Pointy, Floppy, Oval) -> 3 New Features:**
    * **Original:** `Ear Shape: pointy`
    * **One-Hot Encoded:** `Pointy_Ears: 1, Floppy_Ears: 0, Oval_Ears: 0`
    * **Original:** `Ear Shape: oval`
    * **One-Hot Encoded:** `Pointy_Ears: 0, Floppy_Ears: 0, Oval_Ears: 1`

* **Benefit:** Each new feature is binary (0 or 1), making it directly compatible with the decision tree learning algorithm we've already discussed, without further modification.

### Applicability Beyond Decision Trees:

* One-hot encoding is a **general technique** for handling categorical features and is **also crucial for neural networks, linear regression, and logistic regression**. These algorithms typically expect numerical inputs.
    * By converting `Ear Shape`, `Face Shape`, and `Whiskers` into a list of binary features (e.g., `[Pointy_Ears, Floppy_Ears, Oval_Ears, Is_Face_Round, Whiskers_Present]`), the data becomes suitable for input into models that require numerical features.

One-hot encoding allows decision trees (and other algorithms) to process categorical features with multiple values efficiently. The next video will address how decision trees handle **continuous-valued features** (numerical features that can take on any value).

## Handling Continuous-Valued Features in Decision Trees

This video explains how to adapt decision trees to work with features that are **continuous values** (i.e., numbers that can take on any value within a range), such as an animal's weight.

### The Problem: Continuous Features

* Our previous examples used categorical features with a limited number of discrete values (e.g., "pointy" or "floppy").
* For a continuous feature like `Weight` (in pounds), we can't create a branch for every possible weight.

### Solution: Threshold-Based Splitting

When considering a continuous-valued feature for a split:

1.  **Identify Candidate Thresholds:**
    * Sort all training examples by the value of that continuous feature.
    * Consider the **midpoints** between consecutive unique feature values as candidate thresholds for splitting.
    * Example: If weights are [7, 8, 9, 13, 14, ...], candidate thresholds could be 7.5, 8.5, 11, 13.5, etc.
    * For $m$ training examples, there will be at most $m-1$ unique candidate thresholds.

2.  **Evaluate Each Candidate Threshold:**
    * For each candidate threshold `T` (e.g., `weight <= T`):
        * **Split the data:** Divide the examples at the current node into two subsets: those where `feature <= T` (left branch) and those where `feature > T` (right branch).
        * **Calculate Information Gain:** Compute the Information Gain for this specific split, using the entropy formula for the left and right subsets, weighted by the proportion of examples in each.
            * Example: For `Weight <= 7.5`: Information Gain $\approx 0.24$.
            * Example: For `Weight <= 8.5`: Information Gain $\approx 0.61$.
            * Example: For `Weight <= 11`: Information Gain $\approx 0.40$.
            * Example: For `Weight <= 13.5`: Information Gain $\approx 0.32$.

3.  **Select the Best Continuous Split:**
    * Choose the threshold that yields the **highest Information Gain** for that continuous feature.

4.  **Overall Feature Selection:**
    * Compare this maximum Information Gain from the continuous feature (e.g., `Weight`) to the maximum Information Gain from all other discrete/categorical features (e.g., `Ear Shape`, `Face Shape`, `Whiskers`).
    * The feature (whether categorical or a specific threshold for a continuous feature) that gives the **overall highest Information Gain** is chosen to split that node.

### Another Example

Consider a node in a decision tree with three features:
* $f1$: A continuous feature with sorted thresholds $t_1, t_2, t_3, \dots, t_n$.
* $f2$: A discrete feature.
* $f3$: A discrete feature.

At this node, we determined that splitting on $f1$ at threshold $t_3$ ($f1 <= t_3$ vs. $f1 > t_3$) yields the highest Information Gain.

After the split, the data from this node is partitioned into two subsets, forming two new subtrees:

1.  **For the Left Subtree (where $f1 \le t_3$):**
    * **Feature choices available for the *next* split:**
        * **$f1$:** Only the thresholds $t_1, t_2$ are now relevant for $f1$. All examples in this subtree already satisfy $f1 \le t_3$.
        * **$f2$:** This discrete feature is still available.
        * **$f3$:** This discrete feature is still available.

2.  **For the Right Subtree (where $f1 > t_3$):**
    * **Feature choices available for the *next* split:**
        * **$f1$:** Only the thresholds $t_4, t_5, \dots, t_n$ are now relevant for $f1$. All examples in this subtree already satisfy $f1 > t_3$.
        * **$f2$:** This discrete feature is still available.
        * **$f3$:** This discrete feature is still available.

In essence, for continuous features, the range of available thresholds narrows in descendant nodes based on the split made by their parent. Discrete features remain fully available for all subsequent splits.

### Summary for Continuous Features:

* At every node, when considering a continuous feature, iterate through different possible thresholds.
* For each threshold, perform the standard Information Gain calculation.
* If a continuous feature (with its optimal threshold) provides the best Information Gain compared to all other discrete features, then split the node using that continuous feature and its optimal threshold.

This mechanism allows decision trees to effectively leverage numerical features for improved classification. The next (optional) video will generalize decision trees to **regression trees** for predicting numerical values.

## Regression Trees: Decision Trees for Predicting Numbers (Optional)

This video generalizes decision trees to solve **regression problems**, where the goal is to predict a continuous numerical output (Y), such as an animal's weight.

### Structure of a Regression Tree

* **Decision Nodes:** Same as classification trees, they split data based on features.
* **Leaf Nodes:** Unlike classification trees which predict a category, leaf nodes in a regression tree predict a **numerical value**. This value is typically the **average (mean)** of the target variable (Y) for all training examples that fall into that leaf node during training.
    * Example: If a leaf node has animals with weights `[7.2, 7.6, 8.3, 10.2]`, it will predict `8.35` (the average) for any new animal reaching this node.

<img src="/metadata/dt_reg.png" width="600" />

### How to Build a Regression Tree: Splitting Criteria

When building a regression tree, instead of maximizing reduction in entropy, we aim to maximize reduction in variance.

* **Variance:** A statistical measure of how widely a set of numbers varies from their mean. A lower variance means the numbers are more tightly clustered, indicating higher "purity" for regression.

**Process for Choosing a Split (e.g., at the Root Node):**

<img src="/metadata/dt_reg_split.png" width="700" />

1.  **Calculate Initial Variance:** Compute the variance of Y for all examples at the current node (e.g., variance of all 10 animal weights = 20.51). This is $V_{\text{root}}$.

2.  **For each candidate feature to split on (e.g., Ear Shape, Face Shape, Whiskers):**

    *  **Hypothetically split the data** based on that feature, creating child nodes (subsets).
    *  For each child node:
        * Calculate the variance of the Y values (weights) within that specific child node. (e.g., for "pointy ears" subset: variance $\approx 1.47$; for "floppy ears" subset: variance $\approx 21.87$).
        * Calculate $w^{\text{left}}$ and $w^{\text{right}}$ (the fraction of examples going to the left/right child nodes).
    *   **Calculate the Weighted Average Variance of the Split:** This is similar to weighted average entropy.
        * $$Variance_{split} = w^{left} \times Variance_{left} + w^{right} \times Variance_{right}$$
        * Example (Ear Shape): $(5/10 \times 1.47) + (5/10 \times 21.87) = 11.67$.
        * Example (Face Shape): (weights for split) * (variance values) $\approx 19.87$.
        * Example (Whiskers): (weights for split) * (variance values) $\approx 14.29$.

3.  **Calculate Reduction in Variance:** Instead of just comparing weighted variances, we calculate the reduction:
    $$\text{Reduction in Variance} = V_{\text{root}} - V_{\text{split}}$$
    * Example (Ear Shape): $20.51 - 11.67 = 8.84$.
    * Example (Face Shape): $20.51 - 20.51 = 0.64$.
    * Example (Whiskers): $20.51 - 14.29 = 6.22$.

### Choosing the Best Split:

* The feature that gives the **largest Reduction in Variance** is chosen. In the example, "Ear Shape" (8.84) provides the largest reduction.

### Recursive Process:

* Once a split is chosen, the process recursively continues on the resulting subsets of data until stopping criteria are met (similar to classification trees, e.g., max depth, min examples per node).

This adaptation allows decision trees to effectively solve regression problems by finding splits that reduce the spread of the target variable's values. The next video will discuss **ensemble methods** of decision trees.

## Tree Ensembles: Building Robust Decision Trees

A single decision tree can be highly sensitive to small changes in the training data, leading to different tree structures and predictions. To make the algorithm more robust and accurate, we use **tree ensembles**, which are collections of multiple decision trees.

**This is the same problem of overfitting -- in neural network we solve it by various generalization techniques like decreasing size of neural network, or increasing lambda or increasing more training data. Here in decision trees, generalization is done by limiting max depth or doing random forest**.

### The Problem: Sensitivity of Single Trees

* **Example:** In our cat classification, changing just one training example's features (e.g., a specific cat's ear shape changed from floppy to pointy) can cause the optimal root node split to change (e.g., from `Ear Shape` to `Whiskers`).
* **Consequence:** This single change at the root propagates down, leading to an entirely different subsequent tree structure and potentially different predictions. This sensitivity makes a single tree less reliable.

### The Solution: Tree Ensembles

* **Concept:** Instead of training just one decision tree, train a "bunch" or "collection" of slightly different decision trees.
* **Prediction:** For a new test example, run it through all trees in the ensemble. Each tree makes its own prediction.
    * For classification, the final prediction is determined by a **majority vote** among all the trees.
    * For regression, the final prediction would be the average of all tree predictions.
* **Benefit:** The ensemble averages out the individual trees' sensitivities and errors, making the overall algorithm more stable, robust, and generally more accurate. No single tree's "vote" (prediction) holds absolute sway.

### How to Create Diverse Trees in an Ensemble?

The challenge is how to generate multiple, plausible, yet slightly different decision trees from the same dataset. This is a key step that will be covered in upcoming videos.

The next video will introduce a statistical technique called **sampling with replacement**, which is crucial for building these tree ensembles.

## Sampling with Replacement

**Sampling with replacement** is a statistical technique crucial for building tree ensembles. It allows us to create multiple, slightly different training sets from an original dataset.

### How it Works (Analogy with Tokens):

Imagine a bag with four colored tokens (red, yellow, green, blue).

1.  **Pick a token:** Reach into the bag and draw one token (e.g., green).
2.  **Record and Replace:** Record the token drawn, then **put it back into the bag**.
3.  **Repeat:** Shake the bag and repeat the process (pick, record, replace) for a desired number of times (e.g., 4 times).

* **Outcome:** The sequence of drawn tokens might be: green, yellow, blue, blue.
    * Notice: Some tokens might be selected multiple times (e.g., blue appears twice).
    * Notice: Some tokens might not be selected at all (e.g., red was not picked).
* **Importance of Replacement:** If tokens were *not* replaced, drawing 4 tokens from a bag of 4 would always yield the exact same set of 4 tokens. Replacement ensures variability in the sampled sequence.

### Application to Building Tree Ensembles:

* **Goal:** Create multiple "random training sets" that are similar to, but distinct from, the original training set.
* **Process:**
    1.  Imagine your original training set (e.g., 10 cat/dog examples) as items in a theoretical "bag."
    2.  To create one new random training set:
        * **Sample:** Randomly select one training example from the original set.
        * **Replace:** Put that selected example *back* into the original set (the "bag").
        * **Repeat:** Perform this sampling and replacement process `m` times, where `m` is the size of your *original* training set (e.g., 10 times to get a new set of 10 examples).

* **Outcome:** The newly created training set (also of size `m`) will:
    * Likely contain some original examples multiple times.
    * Likely omit some original examples entirely.
* **Benefit:** This process generates multiple, slightly varied training sets. Each of these new sets can then be used to train a different decision tree, leading to the diverse trees needed for an ensemble.

The next video will demonstrate how this technique is used to build the ensemble of trees.

## Tree Ensembles: Random Forest Algorithm

Single decision trees are highly sensitive to small data changes. **Tree ensembles** build multiple trees to create a more robust and accurate model. The **Random Forest algorithm** is a powerful example of a tree ensemble.

### 1. Bagging (Bagged Decision Trees)

The first step to building an ensemble is using **Bagging**, which stands for Bootstrap Aggregating, a technique that leverages **sampling with replacement**.

* **Process:**
    * Given an original training set of size $M$ (e.g., 10 examples).
    * **Repeat $B$ times** (e.g., $B=100$ times, typical range 64-128):
        1.  Create a new training set of size $M$ by **sampling with replacement** from the original training set. This new set will have some original examples repeated and some omitted.
        2.  Train a full decision tree on this newly sampled training set.
    * This generates $B$ different, plausible (but slightly varied) decision trees.
* **Prediction:** For a new test example, pass it through all $B$ trees.
    * For classification: The final prediction is determined by a **majority vote** among the $B$ trees.
    * For regression: The final prediction is the average of all $B$ tree predictions.
* **Benefit:** Averaging (or voting) across multiple trees makes the overall algorithm less sensitive to the peculiarities of any single tree and more robust to small changes in the original training data. Increasing $B$ (number of trees) generally improves performance initially, but eventually leads to diminishing returns in accuracy (while increasing computation time).

### 2. Random Forest: Improving on Bagged Trees

Bagged decision trees can sometimes still create very similar trees, especially near the root, if a single feature is overwhelmingly the best split. Random Forest adds a modification to further diversify the trees:

Repeat B times
* Do sample with replacement and get M training data.
* **Key Idea:** At every node during the decision tree training process:
    * Instead of considering all $N$ available features for the best split, randomly select a **subset of $K$ features** (where $K < N$). Note that we are choosing subset of features (N) to make the split decision, not choosing subset of training data (M).
    * The algorithm then chooses the best split only from this random subset of $K$ features.
* **Typical $K$ Choice:** When $N$ is large (dozens to hundreds of features), a common choice for $K$ is $\sqrt{N}$.
* **Benefit:** This additional randomization forces the individual trees to be even more diverse. If the absolute best feature is not in the random subset, the tree is forced to explore other, potentially good, splits. When these more diverse trees are combined via voting, the ensemble's overall accuracy and robustness are further improved.

### Why Random Forest is Robust:

The combination of sampling with replacement (bagging) and random feature subsets at each split (random forest) makes the algorithm highly robust. It averages over many slightly different trees, each trained on slightly different data and exploring different feature combinations, making the final prediction much less sensitive to specific data points or choices.

## XGBoost: Boosted Decision Trees

While Random Forest is a powerful tree ensemble, **XGBoost (Extreme Gradient Boosting)** is a further refinement that has become the de facto standard for highly competitive machine learning tasks and many commercial applications. It improves upon bagged decision trees by focusing on examples where previous trees performed poorly.

### Boosting: The Core Idea

Boosting is inspired by the concept of "deliberate practice" in education. Instead of training new trees independently (like in bagging), boosting trains trees sequentially, with each new tree **focusing more attention on the examples that the *previously trained trees* misclassified or struggled with**.

* **Process (Conceptual):**
    1.  **Initial Tree:** Train the first decision tree on the sampled with replacement of the training set.
    2.  **Evaluate Performance:** Go back to the **original training set** and see which examples this first tree misclassified.
    3.  **Weighted Sampling/Focus:** When training the *next* tree in the ensemble, adjust the sampling probability (or assign higher weights) to the misclassified examples. This makes the new tree "pay more attention" to those difficult examples.
    4.  **Repeat:** Continue this process for $B$ trees, with each new tree in the sequence learning from the mistakes of the previous ensemble.

* **Benefit:** This iterative focus on "hard" examples allows the ensemble to learn more quickly and achieve higher accuracy, often outperforming simple bagging.

### XGBoost Features and Advantages:

* **Extreme Gradient Boosting:** XGBoost is a specific, highly optimized implementation of gradient boosting. It's known for its speed and efficiency.
* **Weighted Examples (instead of re-sampling):** Unlike traditional bagging that uses physical re-sampling with replacement, XGBoost typically assigns **different weights to different training examples**. Misclassified examples receive higher weights, effectively boosting their influence on the next tree's training. This is more computationally efficient.
* **Built-in Regularization:** XGBoost includes internal regularization techniques to prevent overfitting, making it robust even for complex problems.
* **Default Settings:** It has good default criteria for splitting nodes and stopping tree growth.
* **Highly Competitive:** XGBoost is a top-performing algorithm in machine learning competitions (e.g., Kaggle) and frequently wins alongside deep learning models.
* **Versatility:** Can be used for both **classification** (XGBClassifier) and **regression** (XGBRegressor).

### Implementing XGBoost:

XGBoost is complex to implement from scratch, so practitioners almost universally use its open-source library:

```python
import xgboost as xgb

# For classification
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, ...)
# For regression
# model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, ...)

model.fit(X_train, Y_train)
predictions = model.predict(X_new)
```

XGBoost offers a highly effective and robust solution for decision tree ensembles, often providing state-of-the-art performance for structured data problems.

### Comparing Random Forest vs. XGBoost:
* **Random Forest primarily reduces **Variance (Overfitting)**.**
    * **How:** By training many independent trees on bootstrapped (sampled with replacement) subsets of the data, and then averaging their predictions (for regression) or taking a majority vote (for classification). Each individual tree might have high variance (prone to overfitting its specific subset of data), but by averaging many slightly different high-variance trees, the noise and overfitting tendencies tend to cancel each other out, leading to a much more stable and generalized overall prediction.
    * **Bias:** A single, deep decision tree often has low bias. Since Random Forests build many such trees (even if on subsets), the overall bias of a Random Forest is typically similar to or slightly higher than that of a single deep tree, but the reduction in variance is its main benefit.

* **XGBoost primarily reduces **Bias (Underfitting)**.**
    * **How:** It builds trees sequentially, with each new tree explicitly trying to correct the *errors* (residuals) made by the previous ensemble of trees. By repeatedly focusing on the "hard" examples that the model isn't getting right, it progressively learns more complex patterns and reduces the systematic error (bias) of the overall model.
    * **Variance:** While its primary goal is bias reduction, XGBoost also incorporates regularization techniques (like L1/L2 regularization on weights, tree pruning, shrinkage) that help control variance and prevent it from overfitting. Without these regularization components, a pure boosting algorithm could be very prone to overfitting.

**In summary:**

* **Random Forest:** Good at fixing **Variance (Overfitting)**.
* **XGBoost:** Good at fixing **Bias (Underfitting)**, while also having strong mechanisms to control variance.

## Neural Networks vs. Decision Trees (and Tree Ensembles)

Both decision trees (and their ensembles like Random Forest and XGBoost) and neural networks are powerful and effective machine learning algorithms. The choice between them often depends on the type of data and application.

### Decision Trees and Tree Ensembles (e.g., XGBoost)

**Pros:**

* **Tabular/Structured Data:** Highly effective and often competitive with neural networks on data that fits well into a spreadsheet format (e.g., housing prices, customer data). This includes both classification and regression tasks with categorical or continuous features.
* **Fast to Train:** Generally much faster to train than large neural networks. This allows for quicker iteration through the ML development loop.
* **Interpretability (Single Small Tree):** A single, small decision tree can be human-interpretable, allowing understanding of the decision logic. (However, interpretability decreases significantly with large trees or ensembles of many trees).
* **Strong Performance:** Algorithms like XGBoost are highly competitive and have won many machine learning competitions. **If using decision trees, Andrew Ng always use XGBoost -- it's that good**.

**Cons:**

* **Unstructured Data:** Not recommended for unstructured data like images, video, audio, or raw text. They struggle to extract meaningful features from such data directly.
* **Computational Cost (Ensembles):** While faster than large NNs, tree ensembles are more computationally expensive than single decision trees. If computational budget is extremely constrained, a single tree might be preferred.

### Neural Networks (Deep Learning)

**Pros:**

* **Versatile Data Types:** Works well on all types of data, including:
    * **Tabular/Structured Data:** Often competitive with tree ensembles.
    * **Unstructured Data:** **Preferred algorithm** for images, video, audio, and text. Excels at learning complex features directly from raw unstructured input.
    * **Mixed Data:** Can handle applications with both structured and unstructured components.
* **Transfer Learning:** A huge advantage for applications with limited data. Pre-trained neural networks (e.g., from ImageNet, BERT) can be fine-tuned on smaller, custom datasets, achieving high performance.
* **Multi-Model Integration:** It can be easier to combine and jointly train multiple neural networks in complex systems compared to multiple decision trees. This is because they can all be trained end-to-end using gradient descent.

**Cons:**

* **Slower to Train:** Large neural networks can take a long time to train, slowing down the iterative development cycle.
* **Less Interpretable (Generally):** Large neural networks are often considered "black boxes" due to their complex, non-linear computations, making it harder to understand their exact decision-making process.

### Conclusion:

* For **tabular/structured data**, both tree ensembles (like XGBoost) and neural networks are strong contenders, and you might try both to see which performs better. XGBoost is often a default choice due to its speed and performance.
* For **unstructured data** (images, audio, text), **neural networks are overwhelmingly the preferred and more powerful choice**.
* The rise of faster computing and transfer learning has significantly boosted the applicability and performance of neural networks across a wide range of problems.

This concludes the course on Advanced Learning Algorithms. You've now learned about both neural networks and decision trees, along with practical tips for building effective ML systems. The next course will cover unsupervised learning.

## Data Leakage in Machine Learning

Data leakage occurs when your model gains access to information during training that it wouldn't legitimately have at the time of making predictions (inference), leading to overly optimistic performance during development but poor real-world results.

### 1. Classical ML Data Leakage (Feature Leakage)

* **Definition:** The model inadvertently uses future or otherwise unavailable information from the training data.
* **Key Points:**
    * Most common in tabular or traditional machine learning tasks.
    * Results in deceptively high training and validation accuracy during development.
    * Leads to significantly worse performance when deployed in the real world.
    * Often caused by errors in feature engineering or improper data splitting.
* **Examples:**
    * Using the target variable itself (e.g., a `loan_status` label that indicates if a loan defaulted, being present as an input feature for predicting default).
    * Incorporating information that only becomes known *after* the prediction point (e.g., using `account_closed` as a feature to predict fraud before the account actually closes).
    * Incorrectly splitting time-series data without respecting chronological order, or allowing test data to influence training data (test data contamination).
* **Prevention:**
    * Rigorously ensure that all features used for training are strictly those that would be available at the exact moment of inference.
    * Implement time-aware train/test splits for time-series data to prevent future information leakage.
    * Carefully audit features that show an unusually high correlation with the target variable, as this can be a red flag for leakage.

### 2. LLM Data Leakage (Training Data Memorization)

* **Definition:** A Large Language Model (LLM) memorizes and directly reproduces specific content (which may be sensitive, copyrighted, or from evaluation datasets) from its vast training data.
* **Key Points:**
    * A significant concern in the development and deployment of LLMs and other foundation models.
    * Raises serious issues related to user privacy, intellectual property (copyright), and the validity of benchmark evaluations.
    * The model might output private personal identifiable information (PII), confidential data like API keys, or exact questions/answers from standard test sets.
* **Examples:**
    * Generating user-specific PII (e.g., email addresses, phone numbers) in response to a prompt.
    * Providing verbatim answers to benchmark questions (e.g., from datasets like MMLU) that it encountered during training, inflating its perceived performance.
    * Reproducing copyrighted literary works or segments of code without attribution.
* **Prevention:**
    * **Data Deduplication:** Remove near-identical documents from the training dataset to reduce opportunities for memorization.
    * **PII Filtering:** Employ regular expressions or automated detection systems to identify and remove sensitive personal data from training corpora.
    * **Benchmark Exclusion:** Explicitly exclude known public benchmark datasets from the training data to ensure fair evaluation.
    * **Red Teaming & Auditing:** Proactively test and audit the deployed model by intentionally crafting prompts designed to elicit potential data leaks.
    * **Differential Privacy:** Explore advanced cryptographic techniques, though implementing differential privacy at the scale of LLM training remains a challenging research area.
