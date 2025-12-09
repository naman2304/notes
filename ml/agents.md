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
