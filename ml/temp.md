**Appendix**  
[Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production)

---

## Machine Learning in Production: Beyond the Model

Welcome to the course on Machine Learning in Production! This course focuses on the critical steps needed to take a trained ML model and successfully deploy, monitor, and maintain it in a real-world production environment to create maximum value.

### The Challenge: From Proof-of-Concept (PoC) to Production

* You might successfully train a model (e.g., a neural network to detect scratches on phones) that performs well on your test set. This is a great milestone (a "proof of concept" or PoC).
* However, achieving a valuable **production deployment** often involves significant additional work and challenges beyond just the ML model itself. This is often called the "PoC to Production Gap."

### Example: Automated Visual Defect Inspection for Phones

* **Goal:** Use computer vision to detect defects (scratches, cracks) on phones moving down a manufacturing line.
* **Deployment Pattern:**
    * **Edge Device:** A device within the factory runs inspection software.
    * **Camera:** The software controls a camera to take pictures of phones.
    * **Prediction Server:** The inspection software sends these images (via API calls) to a prediction server.
    * **ML Model:** The prediction server hosts the trained ML model, which processes the image and returns a prediction (e.g., "defective" or "acceptable," possibly with bounding boxes).
    * **Control Decision:** The inspection software then uses this prediction to make a control decision (e.g., pass the phone or divert it for rework).
    * *(Note: Prediction servers can be in the Cloud or at the Edge itself, common in manufacturing for robustness against internet outages.)*

### What Can Go Wrong in Production (Beyond Test Set Performance)?

Even a model that does well on a holdout test set can fail in production:

* **Concept Drift / Data Drift:** The real-world data distribution changes over time compared to the training data.
    * **Example:** Factory lighting conditions change, making production images much darker than those in the training set. The model, trained on bright images, performs poorly on dark images.
    * **Consequence:** The model's performance degrades in the live environment.

### The Scope of a Production ML System

The ML model code (the trained neural network) is often a very small fraction (e.g., 5-10%) of the total code needed for a production ML system. Much more software engineering is required for:

* **Data Management:** Data collection, data verification, feature extraction.
* **Serving Infrastructure:** Prediction servers, API interfaces, managing computation resources.
* **Monitoring:** Logging inputs and predictions, system health, detecting data/concept drift.
* **Maintenance:** Mechanisms for updating models in response to performance degradation or data changes.

This course will guide you through all these pieces, providing a framework for organizing the full lifecycle of an ML project from start to finish. The next video will detail this full lifecycle.

## The Machine Learning Project Lifecycle

Building a machine learning system involves a structured, iterative lifecycle. This framework helps in planning, executing, and minimizing surprises throughout the project.

### Major Steps of an ML Project:

<img src="/metadata/ml_project_lifecycle.png" width="700" />

1.  **Scoping:**
    * **Define the project:** Determine the specific problem to solve with ML.
    * **Define X and Y:** Clearly identify what the input data ($X$) will be and what the target output ($Y$) is.

2.  **Data Collection:**
    * **Acquire Data:** Gather the necessary raw data.
    * **Define Data:** Establish clear definitions for what constitutes your data.
    * **Establish Baseline:** Determine a reasonable baseline performance target (e.g., human-level performance or a simple heuristic).
    * **Label & Organize:** Annotate or label the data, and organize it effectively. (Best practices for this are covered later).

3.  **Model Training:**
    * **Select & Train Model:** Choose an appropriate ML model (e.g., neural network, decision tree) and train its parameters on the collected data.
    * **Error Analysis:** Analyze the model's mistakes (e.g., on a cross-validation set) to understand specific failure modes.
    * **Iterative Improvement:** This is often an iterative loop. Insights from error analysis may lead to:
        * Updating the model (e.g., changing architecture, hyperparameters).
        * Returning to data collection (e.g., acquiring more data or specific types of data).
    * **Final Audit:** Before deployment, conduct a final check and audit to ensure the system's performance is good enough and reliable for the application.

4.  **Deployment:**
    * **Deploy in Production:** Put the trained model into a live environment where users can interact with it.
    * **Write Software:** Develop all necessary surrounding software infrastructure (APIs, servers, logging, etc.).
    * **Monitor System:** Continuously track the system's performance and the characteristics of the incoming data.
    * **Maintain System:** Address issues as they arise (e.g., performance degradation due to data drift).

5.  **Maintenance Loop (Post-Deployment):**
    * After initial deployment, ongoing maintenance is crucial. This often triggers a loop:
        * More error analysis on live data.
        * Retrain the model with new data.
        * Potentially update the dataset with live production data.
        * Deploy an updated model.

This framework is widely applicable across various ML domains (computer vision, audio, structured data, etc.) and helps minimize surprises by outlining all essential steps from initial concept to sustained operation.

## The Machine Learning Project Lifecycle: A Speech Recognition Example

This video illustrates the full machine learning project lifecycle using the example of building a speech recognition system for voice search.

### 1. Scoping: Defining the Project

* **Define Goal:** E.g., speech recognition for voice search on mobile phones.
* **Estimate Key Metrics:** Identify critical performance indicators for the specific problem. For speech recognition, these include:
    * **Accuracy:** How correctly is speech transcribed (e.g., word error rate)?
    * **Latency:** How long does it take to transcribe speech?
    * **Throughput:** How many queries per second can the system handle?
* **Estimate Resources:** Guesstimate time, compute, budget, and project timeline. (More on scoping in Week 3).

### 2. Data Stage: Acquiring and Organizing Data

* **Define Data:** Establish clear conventions for data labeling and formatting.
    * **Challenge: Label Consistency:** For audio like "Um, today's weather", acceptable transcriptions might vary ("Um, today's weather", "Today's weather", "[noise] today's weather"). Inconsistent labeling across annotators can confuse the model. Spotting and correcting such inconsistencies (e.g., standardizing on one convention) is crucial for performance.
    * **Other Data Definition Questions:** Silence duration before/after clips, volume normalization (especially for clips with varying loudness).
* **Establish Baseline:** Determine human-level performance or a competitor's performance to set realistic goals.
* **Label & Organize Data:** Annotate raw data (audio) with ground truth labels (transcripts).
* **Data-Centric AI Mindset:** Unlike academic research that often holds datasets fixed, in production, you actively **edit and improve data quality** based on needs, rather than just fixing code.

### 3. Modeling Stage: Training and Improving the Model

<img src="/metadata/research_prod.png" width="600" />

* **Inputs to Training:** Code (algorithm/architecture), Hyperparameters, Data.
* **Product Team Focus (Data-centric AI):** While academic research often focuses on optimizing code/algorithms for fixed data, product teams often find it more effective to:
    * **Hold Code Fixed** (e.g., use an open-source implementation).
    * **Focus on Optimizing Data and Hyperparameters.**
* **Error Analysis:** After initial training, perform error analysis to identify specific shortcomings of the model (e.g., misclassifications due to background noise, specific accents).
* **Iterative Data Improvement:** Use insights from error analysis to systematically improve the data (e.g., targeted data collection for specific noise types like car noise, using data augmentation to simulate it). This is more efficient than generic data collection.

### 4. Deployment Stage: Putting the Model into Production

* **Deployment Pattern (Example: Mobile Voice Search):**
    * **Edge Device (Smartphone):** Runs local software (e.g., a Voice Activity Detection (VAD) module).
    * **VAD:** A simple algorithm (possibly ML-based) that detects when a user is speaking and extracts only the relevant audio segment.
    * **Prediction Server (Cloud):** The VAD-filtered audio clip is sent via API call to a prediction server (often in the cloud).
    * **ML Model:** The prediction server hosts the speech recognition model, transcribes the audio.
    * **Frontend:** The transcript and search results are sent back to the mobile phone's frontend code for display.
* **Software Engineering:** Significant software engineering is needed to build inference servers, APIs, manage scaling for millions of users, ensure reliability, and optimize computational cost.

### 5. Monitoring & Maintenance Stage: Sustaining Performance

* **Continuous Monitoring:** Track real-time performance of the deployed system.
* **Data/Concept Drift:** Be prepared for the data distribution to change over time.
    * **Example:** A speech system trained on adult voices might degrade if more young individuals (teenagers, children) start using it, as their voices sound different. This is "concept drift" or "data drift."
* **Actionable Insights:** Monitoring helps spot such drifts.
* **Model Updates:** When performance degrades due to drift, trigger retraining (potentially with newly collected production data) and deploy an updated model.

The full ML project lifecycle emphasizes that building a valuable ML system is a continuous process of definition, data work, iterative modeling, deployment, and ongoing maintenance in the face of evolving real-world conditions.

## Course Overview: Machine Learning in Production

This course will guide you through the machine learning project lifecycle, but in a non-linear fashion, starting from deployment and working backward, to highlight the practical considerations for valuable production systems.

### Course Structure:

* **Rest of This Week (Week 1): Deployment**
    * Focus on the most important ideas and techniques for deploying machine learning models into production environments.
* **Week 2: Modeling (Data-Centric Approach)**
    * Learn new strategies for systematically improving model performance using a **data-centric AI approach**. This emphasizes optimizing the data rather than just the code/model.
* **Week 3: Data & Scoping**
    * **Data:** Best practices for defining, establishing baselines for, labeling, and organizing data in a systematic and efficient manner.
    * **Scoping (Optional):** Tips on defining effective machine learning projects.
* **Final Project (Optional):** A hands-on project that follows the full ML project lifecycle from scoping to deployment.

### MLOps (Machine Learning Operations):

* Throughout the course, you'll learn about **MLOps**, an emerging discipline comprising tools and principles for systematically supporting the entire ML project lifecycle, especially the modeling, data, and deployment phases.
* MLOps aims to streamline previously manual and slow processes, making ML development more efficient and robust.

The course begins by diving into the crucial concepts and ideas for deploying machine learning systems.

## Challenges and Considerations in ML Deployment

Deploying a machine learning model is a crucial step to deliver value, but it comes with significant challenges, broadly categorized as statistical (ML-specific) and software engineering issues.

### 1. ML/Statistical Challenges: Concept Drift and Data Drift

These challenges arise when the data distribution or the underlying relationship between input and output changes after the model has been deployed.

* **Concept Drift:** The relationship between $X$ (inputs) and $Y$ (targets) changes. The "concept" being learned by the model changes.
    * **Example:** Before COVID-19, many online purchases from a new user might flag fraud; after COVID-19, such behavior became normal, so the concept of "fraudulent purchase pattern" changed.
    * **Example:** Housing prices ($Y$) increase over time due to inflation for the same house size ($X$).
* **Data Drift:** The distribution of the input features $X$ changes, even if the mapping from $X$ to $Y$ remains the same.
    * **Example:** In a factory, lighting conditions change, making images captured for defect inspection darker. The input distribution ($X$) for images shifts.
    * **Example:** A speech recognition system initially used by adults sees an increasing number of young users (teenagers, children); the distribution of voices ($X$) changes.
* **Impact:** Both types of drift cause the model's performance to degrade in production.
* **Response:** Monitor for these changes and update/retrain the model (often with new, representative data) to adapt. Drift can be gradual (e.g., language evolution) or sudden (e.g., COVID-19 impact on consumer behavior).

### 2. Software Engineering Challenges: Designing the Prediction Service

Implementing the prediction service (the component that takes $X$ and returns $\hat{Y}$) involves numerous design choices:

* **Real-time vs. Batch Predictions:**
    * **Real-time:** Needs immediate response (e.g., speech recognition, self-driving cars, typically sub-second latency).
    * **Batch:** Can process data periodically (e.g., nightly patient record analysis). This affects software complexity significantly.
* **Deployment Location (Cloud vs. Edge vs. Browser):**
    * **Cloud:** Prediction server hosted in a remote data center (e.g., many web services). Offers scalability and centralized management.
    * **Edge:** Prediction server runs on a device physically close to the data source (e.g., in a factory, car, or smartphone). Essential when internet connectivity is unreliable or low latency is critical.
    * **Web Browser:** Some models can run directly in the user's browser, enabling offline functionality and very low latency.
* **Compute & Memory Resources:**
    * Consider the available CPU, GPU, and RAM on the deployment hardware. A model trained on powerful GPUs might need to be compressed or optimized for deployment on less powerful edge devices.
* **Latency & Throughput (QPS - Queries Per Second):**
    * **Latency:** Time taken for a single prediction request (e.g., 300ms for speech recognition within a 500ms user budget).
    * **Throughput:** Number of prediction requests handled per unit time (e.g., 1000 QPS). Requires sufficient hardware and optimized code.
* **Logging:**
    * Log inputs ($X$) and predictions ($\hat{Y}$) from production (with user consent/privacy). This data is invaluable for future analysis, debugging, retraining, and detecting drift.
* **Security & Privacy:**
    * Design appropriate levels of security and privacy based on data sensitivity and regulatory requirements (e.g., high requirements for patient health records).

### The Deployment Milestone: Only Halfway There

* A first successful model deployment is a significant achievement but often only marks about **halfway** to a truly valuable, sustained production system.
* The "second half" involves continuous monitoring, maintenance, adapting to data changes, and iterative improvements in the live environment. This is often where the most challenging and valuable lessons are learned.
* **MLOps (Machine Learning Operations)** is the emerging discipline focused on standardizing practices and tools to manage this entire lifecycle, ensuring reliable, scalable, and maintainable ML systems.

The next video will explore common design patterns for deploying machine learning models.

## ML Deployment Patterns and Degrees of Automation

Deploying a machine learning model requires careful planning beyond just turning it on. This involves selecting appropriate deployment patterns and deciding the optimal degree of automation.

### Common Deployment Use Cases:

1.  **New Product/Capability:** Offering an ML-powered service for the first time (e.g., a new speech recognition service).
    * **Pattern:** Typically starts with a **small traffic rollout** and **gradual ramp-up with monitoring**.
2.  **Automating/Assisting a Human Task:** Replacing or aiding human effort with ML (e.g., AI inspecting phones for scratches, previously done by humans).
    * **Pattern:** Often uses **shadow mode deployment** initially.
3.  **Updating an Existing ML System:** Replacing an older ML model with a newer, hopefully better, one.
    * **Pattern:** Often involves **gradual ramp-up with monitoring** and strong **rollback** capabilities.

### Recurring Themes in Deployment:

* **Gradual Ramp-up with Monitoring:** Instead of full traffic immediately, send a small percentage of traffic to the new system, monitor its performance, and gradually increase traffic as confidence grows.
* **Rollback Capability:** Design the system to easily revert to the previous version if the new deployment encounters issues.

### Key Deployment Patterns:

1.  **Shadow Mode Deployment:**
    * **Concept:** The new ML model runs in parallel with the existing system (human or old ML model), but its outputs are **not used for live decisions**.
    * **Purpose:** Collect data on how the new model performs in a real-world environment, compare its outputs to the existing system's decisions, and verify its accuracy offline without any risk.
    * **Example:** New AI defect inspection system runs, but factory decisions are still based on human inspectors.

2.  **Canary Deployment:**
    * **Concept:** Roll out the new ML model to a **small, controlled fraction of live traffic** (e.g., 5-10%).
    * **Purpose:** Detect problems early with minimal impact. If issues arise, they affect only a small subset of users/operations.
    * **Example:** 5% of smartphone pictures are inspected by the AI, while 95% go to the human.

3.  **Blue/Green Deployment:**
    * **Concept:** Run two identical production environments: "Blue" (the old version) and "Green" (the new version).
    * **Process:** All traffic is initially routed to Blue. When ready, the router is **instantly (or gradually) switched** to send traffic to Green.
    * **Advantage:** Provides an **easy and fast rollback**. If issues occur in Green, traffic can be immediately rerouted back to Blue.

### Degrees of Automation (Spectrum):

ML deployment is not binary (fully automated or not at all); it exists on a spectrum of automation:

1.  **No Automation (Human Only System):** All decisions made by humans.
2.  **Shadow Mode:** AI predicts, but human makes all decisions. AI output is only for monitoring/evaluation.
3.  **AI Assistance:** AI influences the human's decision-making process (e.g., highlights areas of interest in a UI) but the human still makes the final decision. UI design is critical here.
4.  **Partial Automation:** AI makes decisions if it's highly confident. If confidence is low, it defers to a human. (e.g., If AI is sure a phone is fine/defective, it decides. If unsure, it sends to human reviewer). Human judgment in these edge cases can also be valuable feedback for future training.
5.  **Full Automation:** AI makes every single decision without human intervention. (Common in large-scale consumer internet applications like web search where human intervention is infeasible per query).

* **Common Path:** Many applications start with lower degrees of automation (human in the loop) and gradually move towards higher automation as AI performance and confidence improve.
* **Human-in-the-Loop Deployments:** AI Assistance and Partial Automation are examples of "human-in-the-loop" systems, which are often the optimal design point, especially outside large-scale consumer internet services.

The next video will delve into the critical aspect of **monitoring** ML systems in production.
