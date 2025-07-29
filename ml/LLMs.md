Appendix
* Andrej Karpathy videos
    * [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) 
    * [How I use LLMs](https://www.youtube.com/watch?v=EWvNQjAaOHw)
 
---

## Deep Dive into LLMs like ChatGPT

This document provides concise notes on the construction, training, and capabilities of Large Language Models (LLMs) like ChatGPT, based on Andrej Karpathy's "Deep Dive into LLMs like ChatGPT" video.

The process of building LLMs like ChatGPT involves multiple sequential stages:

### 1. Pre-training Stage

This stage is most computationally expensive.

#### 1.1 Acquiring Internet Knowledge

* **Data Collection & Processing:**
    * Goal: Obtain a huge quantity of high-quality and diverse text from publicly available internet sources.
    * Starting Point: **Common Crawl** (indexes billions of webpages since 2007).
    * **Filtering Steps:**
        * **URL Filtering:** Removal of undesirable domains (malware, spam, adult, racist sites).
        * **Text Extraction:** Isolating pure text from raw HTML (removing navigation, CSS, etc.).
        * **Language Filtering:** Keeping specific languages (e.g., FineWeb is 65%+ English).
        * **Deduplication:** Removing redundant text.
        * **PII Removal:** Filtering out personally identifiable information (addresses, SSNs).
    * And then we get a dataset like **FineWeb** which is just text (44 terabytes of text).
* **Text Representation: Tokenization:**
    * Neural networks require 1D sequences of symbols from a finite set.
    * **Goal:** Trade off symbol size (vocabulary) and sequence length to make sequences shorter and have more symbols.
    * **Byte-Pair Encoding (BPE):** An algorithm that identifies frequently occurring consecutive bytes/symbols and groups them into new symbols. This reduces sequence length while increasing vocabulary size.
    * **Tokens:** These symbols are called tokens. GPT-4 uses **100,277** possible tokens.
    * **Tokenization Process:** Converting raw text into a sequence of tokens. Example: "Hello World" can be two tokens.
    * Result: The FineWeb dataset, after tokenization, is about **15 trillion tokens**.

#### 1.2 Neural Network Training (The "Heavy Lifting")

* **Objective:** Model the statistical relationships of how tokens follow each other in a sequence.
* **Process:**
    * Take "windows" of tokens (e.g., 8,000 tokens for modern models) from the massive token sequence (so if choose 8000, then this is called **context length** of model)
    * **Input:** The preceding tokens in a window (the "context").
    * **Output:** The neural network predicts the probability of each possible token in the vocabulary coming next.
    * **Training Loop:**
        * Initially, the neural network's parameters are randomly set, leading to random predictions.
        * The model knows the "correct" next token (the label) from the dataset.
        * The network's parameters are iteratively adjusted (updated) to increase the probability of the correct next token and decrease the probability of incorrect ones. This is the **training** process.
        * This happens in parallel across large batches of tokens.
* **Neural Network Internals: The Transformer Architecture:**
    * Input tokens are mixed with billions of parameters (weights) through a giant mathematical expression (these expressions are choosen such that they can be optimizable, parallelizable)
    * Parameters are like "knobs" that are tuned during training.
    * The **Transformer** is a common and effective architecture (e.g., GPT-2 had 1.5 billion parameters).
    * Information flows through layers of simple mathematical operations (layer norms, matrix multiplications, softmaxes).
    * These are "stateless" functions; they don't have memory like biological neurons.

#### 1.3 Pre-training Examples & Cost

* **GPT-2 (OpenAI, 2019):**
    * 1.5 billion parameters.
    * Max context length: 1,024 tokens (tiny by modern standards).
    * Trained on ~100 billion tokens (small compared to 15 trillion for FineWeb).
    * Original training cost: ~$40,000 (2019).
    * Reproducible today for ~$100-$600 due to better datasets, faster hardware (GPUs) and better software.
* **Llama 3 (Meta, 2024):**
    * Much larger and more modern.
    * **405 billion parameters.**
    * Trained on **15 trillion tokens.**
* **Compute Requirements:**
    * Training requires massive computational resources (GPUs, data centers).
    * NVIDIA H100 GPUs are highly sought after for LLM training.
    * These machines collaborate to predict the next token on vast datasets.

### 2. Post-training Stage: Turning LLMs into Assistants

The pre-trained "base model" is an internet text simulator. The parameters of the model are like a zip file of the internet (lossy compression of the internet). To become a helpful assistant, it undergoes post-training.

**Note**: pre-training takes a lot of time (say 3 months), but post training takes very less time (say 3 hours).

#### 2.1 Supervised Fine-tuning (SFT): Imitating Human Experts

* **Goal:** Shift the model's behavior from generating raw internet documents to responding to questions in a conversational manner.
* **Data Source:** Manually curated datasets of conversations between a human and an assistant.
    * Humans (labelers) create prompts and write ideal assistant responses (ScaleAI startup does this for example)
    * Labeling instructions (hundreds of pages) guide human labelers to be "helpful, truthful, and harmless."
    * **InstructGPT (OpenAI, 2022):** Early example of fine-tuning on human-labeled conversations.
    * **Modern Data:** Mostly synthetic conversations generated by LLMs, then potentially edited by humans (e.g., UltraChat). These datasets can have millions of conversations.
* **Training:** Continue training the base model on this new conversation dataset using the exact same algorithm.
* **Outcome:** The model learns to imitate the persona of a helpful, truthful, and harmless assistant. It statistically aligns with what human labelers would provide.

#### 2.2 Tokenization of Conversations

* Conversations, as structured objects, are encoded into 1D sequences of tokens.
* **Special Tokens:** New tokens (e.g., `IM_START`, `IM_END`, `IM_SEP` for GPT-4o) are introduced to denote turns and roles (user, assistant).
* These special tokens, interspersed with text, teach the model the conversational structure.

#### 2.3 LLM Psychology: Emergent Cognitive Effects

Important: Knowledge in the parameters == vague recollection. Knowledge in the context window == working memory.

* **Hallucinations:** LLMs "make stuff up" because they imitate the confident style of answers in their training data, even when they lack factual knowledge. They're not doing real-time research.
    * **Mitigation 1: Knowledge-Based Refusal:** Add examples to the training data where the model is taught to explicitly say "I don't know" when it genuinely lacks information. This allows the model to learn to express uncertainty. Meta interrogates models to find knowledge boundaries and trains accordingly.
    * **Mitigation 2: Tool Use (Web Search, Code Interpreter):**
        * Models can be taught to emit special tokens (e.g., `SEARCH_START`, `SEARCH_END`) to call external tools.
        * The program pauses generation, executes the tool (e.g., performs a web search), and injects the retrieved text into the model's context window (its "working memory").
        * This provides direct, verifiable information, reducing reliance on vague recollections from pre-training.
        * Similarly, a **Code Interpreter** tool allows the model to write and execute code, leveraging its output for calculations (e.g., for math problems), which is more reliable than "mental arithmetic."
* **Models Need Tokens to Think:**
    * Each token generation involves a finite, relatively small amount of computation. All tokens fed into the neural network takes roughly the same amount of compute to output the next token.
    * **Distribute Reasoning:** Models perform better when their reasoning is spread out over many tokens (e.g., showing intermediate steps in a math problem) rather than trying to cram all computation into a single token.
    * This is why ChatGPT often shows step-by-step solutions for math problems – it's not for you, but for the model to "think."
* **Counting Issues:** LLMs are generally poor at direct counting because they see "tokens" (chunks of text) not individual characters, and counting within a single token is computationally intensive. Tool use (e.g., Python code to count characters in a string) is a workaround.
* **Spelling Issues:** Similar to counting, spelling tasks are difficult because models operate on tokens, not individual characters. Using a code interpreter for string manipulation is more reliable.
* **Swiss Cheese Model of Capabilities:** LLMs are incredibly adept in many domains but can exhibit arbitrary failures in seemingly simple tasks (e.g., "9.11 is bigger than 9.9"). These are "holes in the Swiss cheese" that are not fully understood.
* **Knowledge of Self:** LLMs do not possess a persistent "self" or consciousness. They are stateless token tumblers that restart with each conversation. Their "identity" (e.g., "I am ChatGPT by OpenAI") is learned from training data where such statements are common on the internet. This can be hardcoded by developers using specific training examples or "system messages" at the start of conversations.

### 3. Post Training Stage: Reinforcement Learning (RL): Practice Problems and Emergent Thinking

This is the newest and most actively researched stage, pushing models beyond mere imitation. With this, the model becomes a thinking or reasoning model. Examples of RL (thinking/reasoning models): DeepSeek R1, GPT o1, o3. The GPT 2/3/4/4o should be thought as SFT models.

#### 3.1 Motivation: Going Beyond Imitation

* **Analogy to School:**
    * Pre-training: Reading expository material (background knowledge).
    * SFT: Looking at worked solutions from human experts (imitating).
    * **RL:** Doing practice problems (discovering solutions through trial and error).
* **Problem with SFT:** Humans don't always know the *optimal* way for an LLM to solve a problem. An LLM's cognition is different; what's easy for a human might be a "too big a leap" for a model in a single token.
* **RL Solution:** Let the LLM discover its own optimal token sequences that reliably lead to correct answers.

#### 3.2 How RL Works (Verifiable Domains)

* **Trial and Error:**
    * For a given prompt (e.g., a math problem), generate many different "candidate solutions."
    * **Verify Solutions:** Automatically check if each solution reaches the correct answer (e.g., by checking a boxed answer, or using an LLM judge).
    * **Reinforcement:** Solutions that lead to correct answers are "reinforced" (made more likely in future generations) by updating the model's parameters.
    * The model "practices" and learns for itself what token sequences work best.
* **DeepSeek-R1 Paper:** A recent paper that publicly detailed RL fine-tuning for LLMs.
    * Showed significant improvements in mathematical problem-solving accuracy.
    * **Emergent Thinking (Chain of Thought):** A key qualitative finding was that models learned to generate very long solutions, **incorporating self-correction, re-evaluation, and different approaches** ("Wait, let me check my math again..."). This resembles internal monologue and cognitive strategies, emerging directly from the RL process without explicit human instruction.
    * This "thinking" or "reasoning" leads to higher accuracy.
* **Analogy to AlphaGo:**
    * RL allows models to surpass human performance by discovering strategies unknown to humans (e.g., AlphaGo's "Move 37" in Go).
    * Unlike imitation, RL is not constrained by human limitations.
    * In LLMs, this could mean discovering new problem-solving analogies, thinking strategies, or even new "languages" for internal computation.

#### 3.3 RL in Unverifiable Domains (RLHF - Reinforcement Learning from Human Feedback)

* **Challenge:** For tasks like creative writing (jokes, poems, summarization), there's no single "correct" answer to automatically verify. Human judgment is needed, but direct human evaluation at scale is impossible.
* **RLHF Solution: Indirection:**
    1. **Reward Model:** Train a separate neural network (the "reward model") to **simulate human preferences**. This emits a reward score (say 0 to 1)
    2. **Human Feedback:** Humans are asked to *order* different generated solutions from best to worst for a given prompt (easier than direct scoring).
    3. **Reward Model Training:** The reward model is trained to align its scores with these human orderings (it's knobs are adjusted slightly so that the score it generates now aligns more closely with human ordering preference)
    4. **RL with Simulator:** Once the reward model is trained, the main LLM can be fine-tuned via RL using the *simulated* scores from the reward model. This is automatic and scalable.
* **Upsides of RLHF:**
    * Enables RL in unverifiable domains.
    * Empirically leads to improved model performance.
    * Leverages the human ability to *discriminate* (order) rather than *generate* (create ideal responses), which is often easier.
* **Downsides of RLHF:**
    * **Lossy Simulation:** The reward model is a simulation and may not perfectly reflect human judgment.
    * **Gaming the Reward Model:** RL is extremely good at finding ways to "game" the reward model, leading to nonsensical but highly-scored outputs (adversarial examples for which reward score somehow comes very high). This limits how long RLHF can be run before the model's performance degrades.
    * RLHF is more of a "fine-tune" or "little improvement" rather than a path to indefinite, magical improvement like RL in verifiable domains (AlphaGo, coding, maths)

### 4. Inference (Generating New Data)

* Also called test time compute.
    * Start with a "prefix" of tokens.
    * The model outputs a probability distribution for the next token.
    * A token is sampled from this distribution (like flipping a biased coin).
    * The sampled token is appended to the prefix, and the process repeats.
    * **Stochasticity:** Models are stochastic; they don't always reproduce verbatim training data. They generate "remixes" based on statistical patterns.
    * **Example:** When using ChatGPT, it's just doing inference (parameters are fixed, no training is happening).
    * Special tokens introduced in post training (e.g., `IM_START`, `IM_END`, `IM_SEP` for GPT-4o) are used here so that model knows to behave like an assistant
 
### 5. Future Capabilities and Where to Find LLMs

#### 5.1 Future Capabilities

* **Multimodality:** LLMs will natively handle audio (hearing, speaking) and images (seeing, painting) by tokenizing these modalities and integrating them into the existing token-based framework.
* **Agents:** LLMs will move beyond single-task completion to coherent, error-correcting execution of long-running jobs, requiring human supervision (human-to-agent ratios).
* **Pervasiveness and Computer Usage:** LLMs will become more integrated into tools and capable of taking actions on your behalf (e.g., controlling keyboard/mouse, as seen in early "operator" models).
* **Test-Time Training (Research Area):** Models currently have fixed parameters after training. Future research may explore how models can continue to learn and update their parameters during inference, similar to how human brains learn and adapt (e.g., during sleep). This is crucial for very long-running, multimodal tasks.

#### 5.2 Where to Find and Use LLMs

* **Proprietary Models (e.g., OpenAI, Google Gemini):** Access via their respective websites (e.g., chat.openai.com, gemini.google.com).
* **Open-Weights Models (e.g., DeepSeek, Llama):**
    * **Inference Providers:** Platforms like **together.ai** host many state-of-the-art open models for interaction.
    * **Base Models:** Hyperbolic is mentioned as a good place to access Llama 3.1 base models.
    * **Local Deployment:** Smaller, distilled versions of models can be run locally on personal computers (e.g., using **LM Studio** on a MacBook).

#### 5.3 Key Takeaways

* **LLMs as Tools:** Treat LLMs as powerful tools, not infallible entities.
* **Verify Work:** Always check and verify their output, especially for factual information, calculations, or critical tasks.
* **Swiss Cheese Model:** Be aware of their unpredictable shortcomings and "holes" in their capabilities.
* **Stochastic Nature:** They are statistical systems, so results can vary even with the same prompt.
* **Use for Inspiration:** Leverage them for brainstorming, first drafts, and acceleration of work.
* **Ethical Use:** Understand that they are simulations of human behavior based on training data and labeling instructions.

---

## How I Use LLMs

This document summarizes Andrej Karpathy's "How I use LLMs" video, focusing on practical applications, available settings, and his personal usage strategies.

### 1. The LLM Ecosystem Landscape

* **Incumbent:** ChatGPT by OpenAI (released 2022) - most popular, feature-rich due to being established.
* **Big Tech Offerings:** Gemini (Google), Claude (Anthropic), Copilot (Microsoft), Grok (xAI).
* **Other Players:** DeepSeek (China), Mistral (France).
* **Tracking LLMs:** Chatbot Arena, LMSys Leaderboard (ELO scores, evaluations).

### 2. Core Interaction: Text and Tokens

* **Basic Interaction:** Input text, get text back. LLMs excel at writing (haikus, poems, emails, etc.).
* **Under the Hood:**
    * All interactions (user queries, model responses) are converted into **tokens** (small text chunks).
    * **TickTokenizer:** A tool to visualize how text is tokenized (e.g., GPT-4o has ~200,000 possible tokens).
    * **Conversation Format:** Special tokens (e.g., user/assistant start/end) structure the conversation in a 1D token sequence.
    * **Context Window:** This 1D token sequence serves as the model's "working memory."
        * **Resetting Context:** Start a "new chat" to wipe the context window, preventing distraction from irrelevant past tokens and saving computational cost/time.
        * **Precious Resource:** Treat tokens in the context window as a valuable resource; keep it concise and relevant.

### 3. Understanding the LLM "Entity"

* **Self-Contained "Zip File":** An LLM is fundamentally a self-contained neural network (e.g., 1TB file, 1 trillion parameters).
* **Knowledge (Pre-training):**
    * Learned from reading the entire internet (lossy, probabilistic compression).
    * Knowledge is often *out of date* due to the high cost and infrequency of pre-training (knowledge cut-off).
    * More frequent information on the internet leads to better "recollection."
* **Personality/Style (Post-training/SFT):**
    * Programmed by human labelers curating conversation datasets, teaching the model to act as a helpful assistant.
* **No Default Tools:** By default, LLMs do not have access to calculators, web browsers, or other tools. They rely solely on their internal "zip file" knowledge.

### 4. Practical Usage: Knowledge-Based Queries

* **Good for:** Common, non-recent, low-stakes information (e.g., caffeine content in coffee, ingredients of common medicines).
* **Caveats:**
    * Not guaranteed to be accurate; always verify with primary sources if stakes are high.
    * Relies on the model's "vague recollection" of pre-trained data.

### 5. Model Tiers and Selection

* **Paid Tiers:** Companies offer different models at various pricing tiers (e.g., ChatGPT Free/Plus/Pro, Claude Haiku/Sonnet/Opus).
* **Model Intelligence:** Larger, more expensive models generally offer:
    * More world knowledge.
    * Better writing/creativity.
    * Less hallucination.
    * Access to advanced features (thinking models, specific tools).
* **"LLM Council":** Karpathy often asks multiple LLMs the same question for diverse perspectives (e.g., travel advice).

### 6. "Thinking Models" (Reinforcement Learning - RL)

* **Concept:** Models specifically fine-tuned with Reinforcement Learning (RL) on math/code problems, leading to emergent "thinking strategies" (inner monologue, backtracking, re-evaluation).
* **Benefit:** Higher accuracy on complex problems (math, code, reasoning).
* **Trade-off:** Slower response times (can take minutes as the model "thinks").
* **Examples:**
    * **OpenAI:** Models starting with "O" (e.g., O1 Pro Mode) are thinking models.
    * **Grok:** Has a "Think" button to activate reasoning mode.
    * **DeepSeek-R1:** (Hosted on Perplexity) explicitly shows detailed "thoughts."
    * **Google Gemini:** Some experimental "thinking" models available.
* **Usage Strategy:** Start with faster, non-thinking models for simple queries. Switch to thinking models for difficult problems where accuracy is paramount.

### 7. Tool Use: Extending LLM Capabilities

LLMs can be equipped with tools, allowing them to interact with external systems.

* **Internet Search:**
    * **Purpose:** Access recent information or niche knowledge not in the model's pre-trained data.
    * **Mechanism:** Model emits a special "search" token; the application pauses, performs a web search, and injects retrieved text into the context window for the model to process.
    * **Examples:** ChatGPT's "Search the web" feature, Perplexity.ai (specializes in search), Grok (often auto-searches).
    * **Usage:** For current events, changing information (e.g., stock prices, product offerings), or trending topics.
    * **Caveat:** Always verify citations; hallucinations can still occur.
* **Deep Research (e.g., ChatGPT Pro, Perplexity, Grok Deep Search):**
    * Combines internet search with extended thinking.
    * Spends minutes (e.g., 5-10 mins) researching multiple sources, reading papers, and synthesizing a comprehensive report.
    * **Usage:** Detailed comparisons (e.g., web browsers, supplements), complex historical events, or scientific concepts.
    * **Caveat:** Still a "first draft"; verify information, especially for critical applications.
* **File Uploads (e.g., Claude, ChatGPT):**
    * Upload PDFs, text files, or screenshots.
    * Model "reads" the content (often by converting it to text and loading into context window).
    * **Usage:** Summarizing research papers, "reading" books collaboratively (asking questions, clarifying concepts), analyzing personal data (e.g., blood test results – with caution for medical advice).
    * **Convenience:** Screenshot tools (e.g., Cmd+Shift+4 on Mac) make it easy to upload parts of the screen.
* **Python Interpreter / Code Execution (e.g., ChatGPT Advanced Data Analysis, Claude Artifacts):**
    * Model writes code (e.g., Python, JavaScript) to solve problems.
    * The application runs the code and feeds the output back to the model.
    * **Benefits:** Highly accurate calculations (avoiding LLM "mental arithmetic" errors), data visualization, simple app prototyping.
    * **ChatGPT Advanced Data Analysis:** Acts as a "junior data analyst," can plot data, fit trend lines, etc.
    * **Claude Artifacts:** Generates executable code (e.g., React apps, Mermaid diagrams) directly in the browser. Useful for interactive flashcards, custom tools, or conceptual diagrams.
    * **Caution:** Always scrutinize generated code; LLMs can still make subtle assumptions or errors.
* **Code Editors Integration (e.g., Cursor):**
    * Dedicated apps (Cursor, VS Code extensions) integrate LLMs directly into development environments.
    * LLM has full context of the codebase (multiple files).
    * **"Vibe Coding":** Giving high-level commands and letting the LLM (e.g., Claude via API) write/modify code, debug, add features.
    * Much faster and more efficient than copying code to web-based LLMs.

### 8. Multimodal Interaction: Beyond Text

* **Audio Input/Output:**
    * **Fake Audio (Text-to-Speech/Speech-to-Text):** Using separate models to transcribe speech to text for LLM input, or read LLM text output aloud. (e.g., Super Whisper for desktop, ChatGPT app's microphone icon).
    * **True Audio (Advanced Voice Mode):** LLM natively processes audio tokens. It can "hear" and "speak" directly (e.g., ChatGPT's advanced voice mode, Grok app's voice mode).
        * Enables natural conversation, voice modulation (e.g., Yoda voice, pirate voice), and basic sound imitation (though often limited).
        * **Usage:** For natural interaction, hands-free operation, or creative voice applications.
    * **Podcast Generation (e.g., Google NotebookLM):** Upload sources (text, PDF) and generate custom audio podcasts on niche topics.
* **Image Input/Output:**
    * **Image Input:** LLMs can "see" images (by tokenizing them into patches) and answer questions about their content.
        * **Usage:** Analyzing nutrition labels, interpreting blood test results (with medical caution), explaining visual memes.
        * **Strategy:** Often good to first ask the model to transcribe relevant text from the image to verify its "reading" accuracy.
    * **Image Output (Text-to-Image):** LLMs can generate images from text prompts (e.g., OpenAI DALL-E 3).
        * **Mechanism:** LLM creates a caption, which is sent to a separate image generation model.
        * **Usage:** Content creation (e.g., YouTube thumbnails, icons), visual summaries.
* **Video Input/Output:**
    * **Video Input (Advanced Voice Mode in Mobile Apps):** Point camera at objects, and the LLM can describe/interpret what it "sees" in real-time (likely processing images frames from the video).
        * **Usage:** Identifying objects, explaining concepts related to visual environment.
    * **Video Output:** Rapidly evolving tools (e.g., RunwayML Gen-2, Pika, Stable Video Diffusion) can generate videos from text prompts.

### 9. Quality of Life Features

* **ChatGPT Memory:**
    * Allows the LLM to "remember" information about you across conversations.
    * Memory is recorded as text snippets in a separate database, prepended to future conversations.
    * Can be explicitly invoked ("Can you please remember this?") or triggered automatically.
    * **Benefit:** Model becomes more personalized and relevant over time.
    * **Management:** Users can view, edit, and delete memories.
* **Custom Instructions:**
    * Globally modify ChatGPT's behavior, tone, or specific preferences (e.g., "don't be an HR business partner," preferred Korean formality).
    * Applied to all new conversations.
* **Custom GPTs:**
    * Pre-configured LLM "assistants" for specific tasks.
    * Essentially saved prompts (often few-shot prompts with examples).
    * **Usage:** Language learning (e.g., Korean vocabulary extractor, detailed translator), recurring task automation.
    * Saves time by not having to re-type instructions for repetitive tasks.

### 10. Conclusion and Recommendations

* **Rapidly Evolving Landscape:** The LLM ecosystem is dynamic, with new features and models emerging constantly.
* **ChatGPT as Default:** A strong incumbent and good starting point.
* **Experimentation:** Explore different LLM providers and pricing tiers to find what works best for specific needs.
* **Key Considerations:**
    * **Model Tier:** Influences knowledge, creativity, and hallucination rates.
    * **Thinking Models (RL):** Essential for complex reasoning tasks.
    * **Tool Availability:** Internet search, code interpreters, file uploads, etc., greatly enhance capabilities.
    * **Multimodality:** Audio, images, video input/output for more natural interaction.
    * **Quality of Life Features:** Memory, custom instructions, custom GPTs for personalization and efficiency.
* **LLMs as Tools:** Use them as powerful tools for inspiration, first drafts, and acceleration.
* **Verify and Be Responsible:** Do not blindly trust LLM output; always verify information, especially for high-stakes tasks. They will still hallucinate or make errors due to their inherent nature.
