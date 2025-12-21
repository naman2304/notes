Appendix
* [Agentic AI by Andrew Ng](https://www.deeplearning.ai/courses/agentic-ai/?utm_campaign=agentic-ai-launch&utm_medium=headband&utm_source=dlai-homepage)

---
## Introduction to Agentic AI

### The Rise of Agentic AI
- The term "agentic" was coined by Andrew Ng to describe a significant and growing trend in AI application development.
- While the term has been overused by marketers, leading to considerable hype, the number of genuinely valuable applications using agentic workflows has also grown rapidly.
- This course focuses on the best practices for building practical and effective Agentic AI applications.

### Applications of Agentic Workflows
Agentic workflows are currently used to build a wide range of applications, including:
- **Customer Support Agents:** Automating and enhancing customer service interactions.
- **Deep Research:** Assisting in writing deeply insightful research reports.
- **Legal Document Processing:** Analyzing and processing complex legal texts.
- **Medical Diagnosis:** Suggesting possible medical diagnoses based on patient input.
- Many projects, including those on Andrew Ng's teams, would be impossible without these workflows.

### The Key to Building Effective Agents
- The primary difference between effective and less effective builders of agentic systems is the implementation of a **disciplined development process**.
- This process is critically focused on **evaluations (evals) and error analysis**.
- Mastering this skill is one of the most valuable in AI today, opening up numerous job opportunities and the ability to create advanced software.

## What is Agentic AI?

### Traditional LLM Usage vs. Agentic Workflows

-   **Traditional Prompting:**
    -   This is like asking an LLM to perform a task, such as writing an essay, in a single, linear pass from start to finish.
    -   Analogy: Forcing a human to write an essay from the first word to the last without ever using the backspace key.
    -   While LLMs perform surprisingly well under this constraint, it's not how humans (or AIs) do their best work.

-   **Agentic Workflow:**
    -   This is an **iterative process** that breaks a large task into multiple smaller steps.
    -   Example (Writing an Essay):
        1.  **Outline:** Generate an essay outline.
        2.  **Research:** Conduct web research based on the outline.
        3.  **Draft:** Write a first draft using the research.
        4.  **Reflect & Revise:** Read the draft, identify areas for improvement or further research, and revise it.
        5.  This cycle of thinking, researching, and revising can be repeated.
    -   **Outcome:** An agentic workflow may take longer but delivers a **much better work product**.

### Defining an Agentic AI Workflow

-   An agentic AI workflow is a process where an LLM-based application executes **multiple steps** to complete a complex task.
-   It often involves using an LLM for different sub-tasks and integrating with external tools (e.g., web search APIs).
-   A potential step is including a **human-in-the-loop** for tasks like reviewing key facts.

### The Core Skill: Task Decomposition

-   A key skill for building agentic systems is learning how to **decompose a complex task into smaller, executable steps**.
-   Mastering this allows you to build workflows for a huge range of exciting applications.

### Course Project: The Research Agent

-   The course will feature a running example: building a **research agent**.
-   **Example Task:** "How do I build a new rocket company to compete with SpaceX?"
-   **Agent's Process:**
    1.  **Plan:** Outlines the research steps, including what to search for.
    2.  **Search:** Calls a web search engine to gather information.
    3.  **Synthesize:** Ranks and synthesizes findings from multiple sources.
    4.  **Outline:** Drafts a report outline.
    5.  **Review:** An "editor agent" reviews the draft for coherence.
    6.  **Generate:** Creates a final, comprehensive report in markdown.
-   This multi-step, deep-thinking process results in a more thoughtful report than a single prompt could produce.

### The Spectrum of Autonomy

-   Agentic AI workflows can range from simple, multi-step processes to highly complex, autonomous systems.
-   The degree of autonomy is an important factor when designing and building different types of applications.

## The Spectrum of Agentic Autonomy

### Why the Term "Agentic"?

-   The AI community was engaged in a debate over the precise definition of an "agent," leading to disagreements about whether a system qualified as a "true agent."
-   Andrew Ng introduced the term **"agentic"** to reframe the discussion.
-   Using "agentic" as an adjective acknowledges that systems can possess agent-like qualities to **different degrees**, rather than being a binary (is/is not an agent) classification.
-   This shift in terminology helps the community focus on the practical work of building these systems rather than debating definitions.

### Notational Convention in this Course

-   **Red Boxes**: Represent user input (e.g., a query, an input document).
-   **Gray Boxes**: Indicate a call to a Large Language Model (LLM).
-   **Green Boxes**: Denote actions carried out by other software, such as:
    -   Calling a web search API.
    -   Executing code to fetch a website's content.
    -   This is often referred to as **tool use**.


### Levels of Agent Autonomy

The degree to which an agent operates autonomously exists on a spectrum.

#### Less Autonomous Agents
-   **Process:** Follows a **fully deterministic sequence of steps** that are hard-coded by the programmer.
-   **Decision-Making:** The primary autonomy lies in the text the LLM generates, not in the workflow itself.
-   **Example (Essay on Black Holes):**
    1.  LLM generates search queries (gray box).
    2.  Hard-coded step to call a web search API (green box).
    3.  Hard-coded step to fetch web pages (green box).
    4.  LLM writes the essay using the fetched content (gray box).
-   **Value:** These systems are highly valuable and are being built for many businesses today.

#### More Autonomous Agents
-   **Process:** The LLM determines the sequence of steps dynamically. The workflow is not predetermined.
-   **Decision-Making:** The LLM decides *which* tools to use, *how* to use them, and whether to *iterate* or *reflect*.
-   **Example (Essay on Black Holes):**
    1.  The LLM decides whether to search the web, news sources, or academic archives like arXiv.
    2.  The LLM chooses to call the web search tool.
    3.  The LLM decides how many web pages to fetch.
    4.  The LLM might write a draft, then decide to reflect and go back to fetch more information before producing the final output.
-   **Characteristics:** These agents are less predictable, harder to control, and are an area of active research.

### Summary of the Spectrum

| Category | Characteristics | Control & Predictability |
| :--- | :--- | :--- |
| **Less Autonomous** | Steps are predetermined and hard-coded by an engineer. | High |
| **Semi-Autonomous** | Can make some decisions, such as choosing from a predefined set of tools. | Medium |
| **Highly Autonomous** | Makes many decisions, including the sequence of steps, and may even create new tools. | Low |

## Benefits of Agentic Workflows

### 1. Enhanced Performance
Agentic workflows allow for the effective completion of tasks that were previously impossible with standard zero-shot prompting.
* **The "Human Eval" Benchmark (Coding Tasks):**
    * **GPT-3.5 (Zero-shot):** ~40% accuracy when asked to write code directly.
    * **GPT-4 (Zero-shot):** ~67% accuracy.
    * **Agentic Workflow Impact:** Wrapping GPT-3.5 in an agentic workflow (e.g., prompting it to write code, reflect, and improve) results in performance that often exceeds the zero-shot performance of GPT-4.
* **Key Insight:** The performance improvement gained by implementing an agentic workflow on an older model often dwarfs the improvement gained by simply upgrading to the next generation of the model.

### 2. Parallelism
While agentic workflows may take longer than a single direct prompt, they can execute specific sub-tasks much faster than a human by running processes in parallel.
* **Sequential vs. Parallel:** A human must research sequentially (read one page, then the next). An agent can parallelize data gathering.
* **Example (Essay on Black Holes):**
    1.  Three LLM instances run in parallel to generate search terms.
    2.  Multiple web search queries are executed simultaneously.
    3.  The agent identifies and downloads multiple web pages (e.g., 9 pages) at the same time.
    4.  All gathered data is fed into an LLM for synthesis.

### 3. Modularity
Agentic workflows allow developers to break a system into individual components, making it easy to swap tools or models to optimize performance.
* **Swapping Tools:** You can choose different tools for specific steps.
    * *Search Engines:* Google, Bing, DuckDuckGo, Tavily, You.com (u.com).
    * *Specialized Search:* Swapping a general web search for a news-specific search engine to find the latest breakthroughs.
* **Swapping Models:** You do not need to use the same LLM for every step. You can route different tasks to different models or providers based on which one performs best for that specific action (e.g., planning vs. drafting).

## Examples of Agentic AI Applications

### 1. Invoice Processing (Structured Process)
This is a common business task involving the extraction of specific data from documents.

* **Goal:** Extract key fields (Biller Name, Address, Amount Due, Due Date) from a PDF invoice and record them in a database.
* **Workflow:**
    1. **Input:** Invoice PDF.
    2. **Tool Use:** Call a PDF-to-text API to convert the document into a format the LLM can ingest (e.g., Markdown).
    3. **LLM Decision:** Verify if the document is actually an invoice.
    4. **Extraction:** LLM extracts the required fields.
    5. **Tool Use:** Call an API/Database tool to update records.
* **Characteristics:** This is considered an **easier** workflow because there is a clear, standard procedure to follow step-by-step.

### 2. Basic Customer Order Inquiry (Structured Process)
An agent designed to handle specific, predictable customer questions, such as order status.

* **Goal:** Respond to a customer email regarding a specific order.
* **Workflow:**
    1. **Input:** Customer email.
    2. **LLM Analysis:** Extract order details and customer name.
    3. **Tool Use:** Query the orders database to retrieve status.
    4. **LLM Action:** Draft an email response.
    5. **Tool Use (Human-in-the-loop):** Place the draft in a queue for human review before sending.
* **Characteristics:** Similar to invoice processing, this follows a clear, pre-defined path.

### 3. General Customer Service Agent (Dynamic/Harder)
An agent designed to respond to *any* customer query, where the steps cannot be hard-coded in advance.

* **Scenario A (Inventory Check):** "Do you have black jeans or blue jeans?"
    * The agent must plan a sequence: Check Black Inventory \rightarrow Check Blue Inventory \rightarrow Synthesize Answer.

* **Scenario B (Returns):** "I'd like to return the beach towel."
    * The agent must decide the logic: Verify purchase \rightarrow Check Return Policy (e.g., is it within 30 days? is it unused?) \rightarrow If valid, issue packing slip \rightarrow Update database to "Return Pending".

* **Characteristics:** This is **harder** because the LLM must **plan** the sequence of steps itself ("solve as you go") rather than following a fixed flowchart.

### 4. Computer Use Agents (Cutting Edge/Hardest)
Agents that interact directly with a computer interface (web browser) to perform tasks.

* **Example:** Checking flight availability on United Airlines.
    * The agent navigates the website, clicks buttons, and fills text fields.
    * *Resilience:* In the video example, when the agent struggled with the United site, it autonomously navigated to Google Flights to find the data, then returned to the United site to confirm.

* **Current Limitations:**
    * Agents often struggle if pages load slowly or have complex layouts.
    * Currently not reliable enough for mission-critical applications.
    * Remains an area of active research.

**Key Skill:** The most important skill for building these systems is **Task Decomposition**—the ability to look at a complex workflow and break it down into discrete, executable steps.

## Task Decomposition

### The Core Philosophy of Decomposition
The central challenge in Agentic AI is transforming complex human or business activities into discrete steps that an agentic workflow can execute.

* **The Feasibility Check:** When breaking a task into steps, the primary question to ask for each step is: *"Can this be done by an LLM, a short piece of code, a function call, or a tool?"*
* **Handling Complexity:** If a specific step is too complex for an LLM to perform reliably, reflect on how a human would handle it. Humans typically break complex tasks down (e.g., they don't write an essay in one breath; they outline, research, draft, and revise).

### Iterative Refinement Example: Research Agent

**Level 1: Direct Generation**

* **Method:** Prompting an LLM to "Write an essay on Topic X" in one go.
* **Limitation:** For complex topics, this often results in surface-level points and lacks deep insight.

**Level 2: Initial Decomposition**

* **Step 1:** Write an essay outline. (*Feasible for LLM*)
* **Step 2:** Generate search terms and search the web. (*Feasible for LLM + Search Tool*)
* **Step 3:** Write the essay based on search results. (*Feasible for LLM*)
* **Outcome:** Improved depth, but the result might feel "disjointed" (e.g., the beginning may not align consistently with the end).

**Level 3: Advanced Decomposition (Solving for Quality)**  
If the output is still not satisfactory (e.g., disjointedness), decompose the "Write Essay" step further:

* **Step 3a:** Write a first draft.
* **Step 3b (Reflection):** Read the draft and identify parts needing revision. (*Feasible for LLM*)
* **Step 3c:** Revise the draft based on the critique.
* **Conclusion:** Breaking a single step into three (Draft \rightarrow Critique \rightarrow Revise) mimics the human writing process and yields a richer, more coherent result.

### Examples of Task Decomposition

**1. Customer Order Inquiries**

* **Step 1 (Extract):** Identify the sender, the items ordered, and the order number from the email. (*Feasible for LLM*)
* **Step 2 (Retrieve):** Generate a database query to pull relevant records (shipping status, dates). (*Feasible for LLM + Database Tool*)
* **Step 3 (Respond):** Draft and send an email response using the retrieved info. (*Feasible for LLM + Email API*)

**2. Invoice Processing**

* **Step 1:** Extract required fields (Biller, Address, Date, Amount) from the text-converted invoice. (*Feasible for LLM*)
* **Step 2:** Verify the extraction and call a function to save the data into a database. (*Feasible for LLM + Database Tool*)

### The Building Blocks of Agentic Workflows
When designing a workflow, view the system as a collection of available components to be sequenced:

**AI Models**

* **LLMs:** Core engines for text generation, reasoning, and decision-making.
* **Multimodal Models:** For processing images or audio.
* **Specialized Models:** Tools for specific tasks like PDF-to-text conversion, text-to-speech, or image analysis.

**Software Tools & APIs**

* **External APIs:** Web search, voice search, real-time weather, email sending, calendar access.
* **Retrieval (RAG):** Tools to search and retrieve relevant text from large databases.
* **Code Execution:** Tools allowing the LLM to write and run code (e.g., Python) for calculations or data processing.

### Summary Strategy for Building Workflows
1. **Analyze the Task:** Observe how a human or business performs the task and identify the high-level discrete steps.
2. **Verify Feasibility:** For each step, ask if it can be implemented with an LLM or an available tool (API/Function).
3. **Decompose Further:** If the answer is "No," ask *"How would a human do this step?"* and break that specific step down into smaller sub-steps that are feasible.
4. **Iterate:** Expect to build an initial workflow, evaluate it, and then refine the decomposition (adding steps like "critique" or "plan") to improve performance.

## Evaluations (Evals) for Agentic Workflows  
### The Importance of Discipline  
* One of the biggest predictors of success in building agentic workflows is the ability to drive a **disciplined evaluation process**.
* Building effective agents requires not just coding the workflow, but systematically measuring and improving it.

### The Development Strategy  
* **Build First, Evaluate Later:** It is nearly impossible to anticipate every potential error in advance.  
* **Manual Discovery:** Start by building the agent, then manually inspect outputs to find specific behaviors you dislike.
* *Example:* You might find your customer service agent is awkwardly mentioning competitors ("Unlike RivalCo, we make returns easy...").
* **Iterative Loop:** Once an undesirable behavior is identified, build a specific metric to track it and work to eliminate it.

### Types of Evaluations  
#### 1. Objective Evals (Code-Based)  
* Used for "black and white" criteria where the presence or absence of a feature is clear.
* **Method:** Write standard code (e.g., Python scripts) to search the text.
* **Example:** **Competitor Mentions**.
* If you have a list of competitors, write code to count the frequency of their names appearing in the agent's output.

#### 2. Subjective Evals (LLM-as-a-Judge)
* Used for free-text outputs where quality is nuanced (e.g., "Is this a high-quality research report?").
* **Method:** Use a separate, often stronger, LLM to read the output and assign a score (e.g., 1 to 5).
* **Limitation:** LLMs can be inconsistent with simple scalar ratings (1-5). While useful as a first pass, more advanced techniques are often required for accuracy.

### Scope of Evaluations (Preview)
* **End-to-End Evals:** Measuring the quality of the *final* output produced by the entire agent.
* **Component-Level Evals:** Measuring the quality of a *single step* within the workflow.
* **Error Analysis (Traces):** Reading through the **intermediate outputs** (traces) of every step to pinpoint exactly where the reasoning or execution failed.

## Key Design Patterns for Agentic Workflows
Building effective agentic workflows involves sequencing building blocks using established patterns. There are four major design patterns that help structure these complex systems.

### 1. Reflection  
Reflection involves the agent examining its own outputs to identify improvements, rather than just generating a final result in one pass.

* **Process:**
    1. **Generate:** The LLM produces an initial output (e.g., Python code).
    2. **Critique:** The system prompts the LLM (or a separate agent) to check the work for correctness, style, and efficiency.
    3. **Iterate:** The LLM uses this critique to generate a vastly improved version (v2, v3, etc.).
* **External Feedback:** Reflection can also incorporate external signals, such as running generated code to check for execution errors, which are then fed back to the LLM to fix bugs.
* **The Critique Agent:** You can simulate a second agent specifically prompted with a persona (e.g., "Your role is to critique code") to review the work of the first agent.

### 2. Tool Use  
Tool use allows LLMs to interact with the outside world by calling functions or "tools" to perform specific tasks that pure text generation cannot handle.

* **Functionality:** The LLM decides *when* to call a tool and *what* arguments to pass to it.
* **Examples:**
    * **Web Search:** To answer questions about current events or product reviews (e.g., "Best coffee maker").
    * **Code Execution:** To solve math problems or perform data analysis by writing and running code (e.g., calculating compound interest).
    * **Productivity:** Interfacing with email, calendars, or databases.
    * **Perception:** Processing images or audio.

### 3. Planning  
Planning enables the LLM to autonomously decide the sequence of steps required to complete a complex objective, rather than following a hard-coded workflow defined by the developer.

* **Mechanism:** The agent breaks down a high-level request into a specific sequence of actions or API calls.
* **Example (HuggingGPT):**
    * *User Request:* "Generate an image where a girl is reading a book and the pose is the same as the boy in this image, then describe it."
    * *Agent's Plan:*
        1. Call Pose Determination Model (to analyze the boy).
        2. Call Image Generation Model (to create the girl's image).
        3. Call Image-to-Text Model (to describe the new image).
        4. Call Text-to-Speech Model (to read the description).
* **Trade-off:** Planning agents are more powerful and flexible but can be harder to control and more unpredictable than fixed workflows.

### 4. Multi-Agent Collaboration
This pattern simulates a human organization where multiple specialized agents collaborate to accomplish a complex task.

* **Concept:** Instead of one generalist agent, you instantiate multiple agents with distinct roles (personas).
* **Example (ChatDev):** A virtual software company composed of a CEO agent, a Programmer agent, a Tester agent, and a Designer agent working together to build software.
* **Example (Marketing Brochure):**
    * **Researcher Agent:** Gathers facts.
    * **Marketer Agent:** Write the copy based on research.
    * **Editor Agent:** Reviews and polishes the text.
* **Benefit:** Research shows that multi-agent systems often produce better outcomes for complex tasks (like writing biographies or playing games) compared to single agents, though they can be harder to orchestrate.

## The Reflection Design Pattern
### Concept
* **Human Analogy:** Just as humans improve their work by reviewing and correcting their first drafts (e.g., fixing typos or clarifying ambiguous dates in an email), LLMs can improve their outputs through a similar iterative process.
* **Mechanism:**
    1. **Drafting:** Prompt an LLM to generate an initial output (e.g., Email v1 or Code v1).
    2. **Reflection:** Pass this initial output back to an LLM (either the same model or a different one) with a prompt asking it to critique, check for bugs, or suggest improvements.
    3. **Revision:** The LLM generates an improved version (e.g., Email v2 or Code v2) based on the reflection.



### Model Selection strategies
* **Same vs. Different Models:** You can use the same LLM for both drafting and reflection, or swap models to leverage specific strengths.
* **Reasoning Models:** "Reasoning" or "Thinking" models are often superior at finding bugs and logical errors. A common pattern is to use a standard model for the first draft and a specialized reasoning model for the reflection/critique step.

### Enhancing Reflection with External Feedback
* **The Power of External Data:** Reflection is significantly more powerful when it incorporates **new, external information** rather than just relying on the LLM to "think" about its own output in isolation.
* **Code Execution Example:**
    * Instead of simply asking the LLM to review the code, **execute the code**.
    * Capture the **output** and any **error messages/logs** (e.g., syntax errors).
    * Feed this execution data back to the LLM during the reflection step.
    * This concrete feedback allows the LLM to diagnose issues accurately and produce a much better second version compared to reflection without execution.


### Key Takeaway
* **Performance:** Reflection is not a "magic bullet" that guarantees 100% accuracy, but it reliably offers a modest performance boost.
* **Design Principle:** Whenever possible, design the workflow to ingest **additional external information** (like code execution results or tool outputs) into the reflection step to maximize its effectiveness.

## Direct Generation vs. Reflection Workflows  
### Direct Generation (Zero-Shot Prompting)  
Direct generation involves prompting the LLM to produce an answer immediately without any examples or intermediate steps.

* **Zero-Shot Prompting:** The term "zero-shot" refers to providing zero examples of the desired input-output pair in the prompt.
* **Examples:**
    * "Write an essay about black holes."
    * "Write a Python function to calculate compound interest."
* **Contrast:** "One-shot" or "Few-shot" prompting includes one or more examples to guide the model.

### Performance Comparison  
Studies have consistently shown that reflection improves performance over direct generation across a variety of tasks.

* **Research Findings:** A paper by Madaan et al. demonstrates that for various models (GPT-3.5, GPT-4), using reflection (indicated by dark bars in the study's data) yields significantly higher performance scores compared to zero-shot prompting (light bars).

### When to Use Reflection  
Reflection is particularly helpful in specific scenarios where errors are subtle or require verification:

* **Structured Data Generation:**
    * Validating formatting for HTML or complex JSON structures with deep nesting, where syntax errors are common.
* **Instruction Sequences:**
    * Ensuring completeness in step-by-step guides (e.g., "How to brew the perfect cup of tea"), where models might skip a step.
* **Domain Name Generation (Real-World Example):**
    * Andrew Ng's team used reflection to filter domain names for startups.
    * **The Issue:** Initial generation might produce names that are hard to pronounce or have unintended negative meanings in other languages.
    * **The Fix:** A reflection step checks specific criteria: "Is it easy to pronounce?" and "Are there negative connotations?"

### Writing Effective Reflection Prompts  
To maximize the benefit of reflection, prompts should be specific and directive.

* **Tips:**
* **Explicit Instruction:** Clearly state that the model should "review" or "reflect" on the first draft.
* **Define Criteria:** Provide specific rubrics for the critique.
    * *Domain Names:* "Check if easy to pronounce", "Check for negative connotations."
    * *Emails:* "Check the tone", "Verify all facts/dates against context."
* **Learning Strategy:** A good way to improve prompt engineering skills is to download open-source software and read the prompts written by experienced developers.
