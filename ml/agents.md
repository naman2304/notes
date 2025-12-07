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
