Appendix
* [AI Engineering: Building Applications with Foundation Models]()

---

*   **Key Problems:**
    *   Should an AI application be built?
    *   How to evaluate AI applications and AI outputs (using AI to evaluate AI)?
    *   Causes, detection, and mitigation of **hallucinations**.
    *   Best practices for **prompt engineering**.
    *   How **RAG** (Retrieval-Augmented Generation) works and its strategies. Retrieval-augmented generation (RAG) applications are built upon retrieval technology that has powered search and recommender systems since long before the term RAG was coined.
    *   What an **agent** is, how to build and evaluate one.
    *   When and when not to **finetune** a model.
    *   Data requirements and quality validation.
    *   Methods to make models faster, cheaper, and more secure.
    *   Establishing a **feedback loop** for continuous application improvement.
*   **Comparison of the book with "Designing Machine Learning Systems" (DMLS):**
    *   **DMLS:** Focuses on applications with **traditional ML models**, involving tabular data annotations, feature engineering, and model training.
    *   **AI Engineering (AIE):** Focuses on applications with **foundation models**, involving prompt engineering, context construction, and parameter-efficient finetuning.
    *   **Relationship:** AIE can be a companion to DMLS. Both are self-contained and modular. A real-world system often combines both traditional ML and foundation models.
* **Lindy's Law:** Assuming a technology's future life expectancy is proportional to its current age (if it's been around, it will likely continue).

## Chapter 1: Introduction to Building AI Applications with Foundation Models

**I. Overview of AI Engineering in the Era of Foundation Models**

*   **Scale of AI Post-2020:** The period after 2020 is characterized by the **massive scale** of AI models, such as ChatGPT, Google's Gemini, and Midjourney. These models consume a significant portion of the world's electricity and risk exhausting publicly available internet data for training.
*   **Focus of the Chapter:** This chapter introduces **foundation models** as a key catalyst for AI engineering, discusses successful **AI use cases**, and outlines the **new AI stack**, including how the role of an AI engineer differs from a traditional ML engineer.
*   **Core Transformation:** Foundation models (LLMs and LMMs) have transformed AI from an esoteric discipline into a powerful development tool, lowering the barriers for building AI products, even for those without prior AI experience.

**II. The Rise of AI Engineering**

*   **Evolutionary Path:** AI engineering emerged from the culmination of decades of technology advancements, tracing a path from early **language models** (1950s) to **large language models**, then to **foundation models**, and finally to **AI engineering**.

**III. From Language Models to Large Language Models**

*   **Language Models (LMs):**
    *   **Definition:** Encode statistical information about language, indicating the probability of a word appearing in a given context.
    *   **Example:** Given "My favorite color is __", an English LM would predict "blue" more often than "car".
    *   **Historical Roots:** Concepts from Claude Shannon's 1951 paper "Prediction and Entropy of Printed English" are still used today.
*   **Tokens and Tokenization:**
    *   **Token:** The basic unit of a language model; can be a character, a word, or a part of a word (e.g., "-tion").
    *   **Tokenization:** The process of breaking original text into tokens.
    *   **Example:** GPT-4 breaks "I can’t wait to build AI applications" into nine tokens, splitting "can’t" into "can" and "’t".
    *   **Conversion Rate:** For GPT-4, 100 tokens are approximately 75 words.
    *   **Vocabulary:** The set of all tokens a model can work with (e.g., Mixtral 8x7B: 32,000; GPT-4: 100,256).
    *   **Advantages of Tokens (over words/characters):**
        1.  Break words into meaningful components (e.g., "cooking" into "cook" and "ing").
        2.  Reduce vocabulary size, enhancing model efficiency.
        3.  Help process unknown words (e.g., "chatgpting" into "chatgpt" and "ing").
*   **Types of Language Models:**
    *   **Masked Language Model:** Predicts missing tokens using context from *both* sides (e.g., BERT). Less common for text generation.
    *   **Autoregressive Language Model:** Predicts the *next* token in a sequence. This type is the primary choice for **generative AI** and text generation. (Unless specified, "language model" in the book refers to this type).
*   **Completion Capability:** Autoregressive models are powerful for tasks framed as completion, such as translation, summarization, coding, and classification.
    *   **Example (Classification):** Prompt "Question: Is this email likely spam? Here’s the email: <email content>\nAnswer:" could be completed with "Likely spam".
    *   **Limitation:** Completion differs from conversational engagement; a model might respond with another question rather than an answer. "Post-Training" (Chapter 2) addresses this.
*   **Self-supervision:** This breakthrough allowed language models to scale.
    *   **Mechanism:** Models infer labels directly from input data, bypassing the need for explicit human labeling.
    *   **Example:** The sentence "I love street food." generates six training samples, where each word is predicted from the preceding context.
    *   **Special Tokens:** `<BOS>` (beginning of sequence) and `<EOS>` (end of sequence) markers are used for multiple sequences; `<EOS>` helps models know when to stop generating responses.

**IV. From Large Language Models to Foundation Models**

*   **Foundation Models Defined:** Include both Large Language Models (LLMs) and Large Multimodal Models (LMMs).
*   **Broad Capabilities:** Can perform a wide array of tasks like summarization, translation, and Q&A.
*   **Adapting Models:** Even powerful out-of-the-box models might need adaptation to meet specific needs (e.g., capturing a brand's voice in product descriptions).
*   **Common AI Engineering Adaptation Techniques:**
    1.  **Prompt Engineering:** Crafting detailed instructions and examples (e.g., for desired product descriptions).
    2.  **Retrieval-Augmented Generation (RAG):** Connecting the model to an external database (e.g., customer reviews) to provide supplementary context.
    3.  **Finetuning:** Further training the model on a dataset of high-quality, task-specific examples.
*   **Efficiency:** Adapting existing foundation models is significantly easier and faster than building models from scratch (e.g., 10 examples/one weekend vs. 1 million examples/six months). This reduces AI application development costs and time to market.

**V. From Foundation Models to AI Engineering**

*   **Definition:** **AI engineering** is the process of building applications on top of foundation models.
*   **Terminology Choice:** The term "AI engineering" is used to differentiate it from "ML engineering," as foundation models introduce new opportunities and challenges that traditional ML engineering doesn't fully capture. (ML engineering is still a broader, encompassing term).

**VI. Foundation Model Use Cases**

*   **Value Generation:** AI applications can reduce costs, improve process efficiency, drive growth, and accelerate innovation. For 7% of executives surveyed by Gartner in 2023, **business continuity** was a motivation for adopting generative AI.
*   **Occupational Exposure to AI (Eloundou et al., 2023):**
    *   A task is "exposed" if AI can reduce its completion time by ≥50%.
    *   **High Exposure (near 100%):** Interpreters and translators, tax preparers, web designers, writers, mathematicians, financial quantitative analysts.
    *   **No Exposure:** Cooks, stonemasons, athletes.
*   **Application Categories (from 205 open-source GitHub repositories):** Writing, Education, Information Aggregation, Data Organization, Workflow Automation, Customer Support, Product Copilots, and Agents.
    *   Many applications can span multiple categories (e.g., a bot for companionship and information aggregation).
*   **Enterprise Adoption Trend:** Enterprises favor lower-risk, internal-facing applications (e.g., internal knowledge management) over external ones (e.g., customer support chatbots) to minimize risks related to data privacy, compliance, and potential catastrophic failures.
*   **Key Use Case Examples:**
    *   **Writing:** Auto-correct, auto-completion, generating emails, essays, and ad copy. AI is helpful for brainstorming and automating parts of writing tasks due to its high volume, tediousness, and user tolerance for mistakes.
    *   **Education:** Summarizing textbooks, generating personalized lecture plans, adapting materials to different learning styles (e.g., auditory learners, code-to-math translation, roleplaying for language learning). Duolingo found AI most helpful in **lesson personalization**.
    *   **Information Aggregation & Distillation:** Processing and summarizing documents, emails, and messages. 74% of generative AI users use it for this. **Instacart's "Fast Breakdown" template** used AI to summarize meeting notes, emails, and Slack conversations into facts, open questions, and action items.
    *   **Workflow Automation:** Automating daily tasks (e.g., booking restaurants, trip planning, refunds) and enterprise tasks (e.g., lead management, invoicing, data entry). AI can also **synthesize data** to improve models.
    *   **Agents:** AI systems that can **plan and use external tools** (e.g., search engines, phone calls, calendars) to accomplish complex tasks like booking a restaurant. This area shows immense potential for productivity gains and is a central topic in Chapter 6.

**VII. Planning AI Applications**

*   **Initial Step:** Before building, especially for commercial purposes, carefully consider the "why" and "how." Building a cool demo is easy; creating a profitable product is hard.
*   **Use Case Evaluation:**
    *   **Business Continuity:** Assess if *not* adopting AI poses an existential threat to the business (e.g., in financial analysis, advertising, document processing).
*   **AI Product Defensibility:** Crucial for standalone AI products.
    *   **Challenge of Low Entry Barrier:** What's easy for you to build is easy for competitors.
    *   **Risk of Subsumption:** If underlying foundation models expand their capabilities (e.g., ChatGPT improving PDF parsing), your application layer might become obsolete.
    *   **Competitive Advantages:**
        *   **Technology:** Often similar across companies with foundation models.
        *   **Distribution:** Usually favors large companies.
        *   **Data:** Startups first to market can build a **"data flywheel"** by gathering user usage data to continuously improve their products, creating a competitive moat.
*   **Setting Expectations:** Define clear, measurable goals for success.
    *   **Business Metrics:** Automation rate (e.g., for customer support chatbots), messages processed, response time, human labor saved, and customer satisfaction.
    *   **Usefulness Thresholds (ML Metrics):**
        *   **Quality Metrics:** Measure response quality.
        *   **Latency Metrics:** Time to First Token (TTFT), Time Per Output Token (TPOT), total latency. Acceptable latency varies by use case.
        *   **Cost Metrics:** Cost per inference request.
        *   **Other Metrics:** Interpretability and fairness.
*   **Milestone Planning:**
    *   Start by evaluating existing models to understand their capabilities.
    *   **"Last Mile Challenge":** Initial rapid progress (e.g., 80% desired experience in one month) can lead to underestimation of time for final improvements (e.g., reaching 95% might take four more months, addressing kinks and hallucinations). Each subsequent percentage gain becomes increasingly challenging.
*   **Maintenance:** An important consideration for long-term application health.

**VIII. The AI Engineering Stack**

*   **Focus on Fundamentals:** Prioritize fundamental building blocks over rapidly changing tools/techniques.
*   **Evolution from ML Engineering:** AI engineering is an evolution of ML engineering, with many enduring principles remaining applicable.
*   **Three Layers of the AI Stack:**
    1.  **Application Development (Top Layer):** Includes evaluation, prompt engineering, and AI interface.
    2.  **Model Development:** Includes modeling and training, dataset engineering, and inference optimization.
    3.  **Infrastructure (Bottom Layer):** Includes model serving, data/compute management, and monitoring.
        *   Saw less growth in 2023 compared to other layers, as core infrastructural needs remain similar despite new models.

**IX. AI Engineering Versus ML Engineering**

*   **Core Distinction:** AI engineering emphasizes **adapting and evaluating existing models** rather than developing them from scratch, distinguishing it from traditional ML engineering.
*   **Model Adaptation Techniques:**
    *   **Prompt-based techniques (e.g., prompt engineering):** Adapt a model *without updating its weights* (by providing instructions and context). Easier, requires less data.
    *   **Finetuning:** Involves *updating model weights* (making changes to the model itself). More complex, requires more data, but can significantly improve quality, latency, and cost. Can unlock tasks the model wasn't explicitly trained for.
*   **Changes in Model Development (Table 1-4):**
    *   **Modeling and Training:** ML knowledge is "nice-to-have" for FMs, not a "must-have" as in traditional ML.
    *   **Dataset Engineering:** Shifts from feature engineering (tabular data for traditional ML) to **data deduplication, tokenization, context retrieval, and quality control** (unstructured data for FMs). Data quality and dataset engineering are critical differentiators.
    *   **Inference Optimization:** Important for both, even *more* so for FMs.
*   **Changes in Application Development (Table 1-6):**
    *   **Evaluation:** **More important** for foundation models due to their open-ended nature and expanded capabilities, making traditional ground-truth comparisons difficult. (e.g., Gemini Ultra's MMLU score changed from 83.7% to 90.04% with a different prompt).
    *   **Prompt Engineering and Context Construction:** **Important** for FMs to elicit desirable behaviors from input (without changing weights) and provide necessary context/tools.
    *   **AI Interface:** **Important** for FMs, moving beyond traditional interfaces to include plug-ins, add-ons, voice-based, or embodied (AR/VR) interactions. Also facilitates new ways of collecting user feedback.
*   **Shift Towards Full-Stack Engineering:** AI engineering, with its emphasis on application development and interfaces, increasingly resembles full-stack development. This shift is reflected in growing support for JavaScript APIs alongside Python. The workflow now often starts with building a product demo first, then investing in data and models if the product shows promise. AI engineers are more involved in product decisions.
