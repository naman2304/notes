Appendix
* Andrej Karpathy videos
    * [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) 
    * [How I use LLMs](https://www.youtube.com/watch?v=EWvNQjAaOHw)
 * 3Blue1Brown
    * [Neural Network playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

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

---

## 3Blue1Brown Neural Network playlist

### Video 1: Neural Networks: The Structure

#### The Recognition Challenge
* Our brains recognize varying handwritten digits effortlessly.
* Programming a computer to do this (e.g., classifying 28x28 pixel images) is **difficult** with traditional methods.
* **Neural Networks (NNs)** offer a solution: they're mathematical structures that **learn** to identify patterns.

#### What is a Neural Network?
* Inspired by the brain, an NN is a **function** mapping inputs to outputs.
* **Learning:** The process of a computer automatically finding optimal internal settings (parameters) to solve a problem.
* **Goal:** Understand the **structure** of a basic NN for handwritten digit recognition.

#### Neurons & Layers
* **Neuron:** A unit holding a **number (activation)**, typically between 0 and 1. Higher activation means "more active."
* **Layers:** Neurons are organized into layers.
    * **Input Layer:** Represents raw data. For a 28x28 image, 784 neurons, each holding a pixel's grayscale value (0=black, 1=white).
    * **Output Layer:** Represents the network's prediction. For digits 0-9, 10 neurons; the highest activation indicates the predicted digit.
    * **Hidden Layers:** Intermediate layers (e.g., two layers with 16 neurons each in this example). They process information, abstracting features from the input.
* **Flow:** Activations in one layer determine activations in the next.

#### Hierarchical Feature Detection (The "Hope")
* NNs are hoped to learn **hierarchical features**:
    * **Early Layers:** Detect simple features like **edges**.
    * **Middle Layers:** Combine edges into **sub-components** (e.g., loops, lines).
    * **Later Layers:** Combine sub-components to recognize complete digits.
* This layered abstraction is useful for many AI tasks (e.g., speech recognition: sounds → syllables → words).

#### How Neurons Activate: Weights & Biases
* A neuron's activation in a layer is determined by:
    1.  **Weighted Sum:** Sum of (previous layer's neuron activations $\times$ their respective **weights**).
        * **Weights (W):** Numerical strength of connection. They define *what* pattern a neuron detects.
            * *Example:* For an edge detector, positive weights on pixels forming the edge, negative on surrounding pixels. 
    2.  **Bias (b):** A number added to the weighted sum. It controls the threshold for a neuron to become active (e.g., how high the sum needs to be).
    3.  **Activation Function:*
        * **Sigmoid ($\sigma$):** "Squishes" the result into the 0-1 range. $\sigma(z) = 1 / (1 + e^{-z})$. Maps any real number $z$ to a value between 0 and 1.
        * **ReLU (Rectified Linear Unit):** $\text{ReLU}(z) = \max(0, z)$. Common in modern NNs, often easier to train than sigmoid.

#### Network Parameters & Learning
* A network has many **weights and biases** (parameters). This example has ~13,000.
* **Learning:** The process where the computer **finds the best values** for these parameters, allowing the network to solve the given problem (e.g., recognizing digits accurately).
* Manually setting these parameters is impractical; thus, learning algorithms are essential.

#### Mathematical Representation
* Using **linear algebra** simplifies computation:
    * Activations: **vectors**.
    * Weights between layers: **matrices**.
    * Biases: **vectors**.
* The transition from one layer's activations ($a$) to the next ($a'$) is:
    $a' = \sigma(Wa + b)$
    * This compact notation is efficient for coding and computation.
* An entire NN is a complex, multi-layered function with many parameters, which is why it can solve complex tasks.

Here are the super crisp notes for the second 3blue1brown neural network video:

### Video 2: Neural Networks: How They Learn

#### Recap: Network Structure
* **Goal:** Handwritten digit recognition (28x28 pixels).
* **Input Layer:** 784 neurons (pixel values 0-1).
* **Hidden Layers:** Two layers, 16 neurons each (arbitrary choice).
* **Output Layer:** 10 neurons (0-9 digits), brightest activation is prediction.
* **Neuron Activation:** Based on a **weighted sum** of previous layer's activations, plus a **bias**, passed through an **activation function** (e.g., sigmoid/ReLU).
* **Parameters:** ~13,000 adjustable **weights** and **biases**. These define network behavior.

#### The Learning Problem
* **Goal:** Adjust 13,000 weights/biases to improve performance on **training data**.
* **Training Data:** Images of digits + their correct labels (e.g., MNIST dataset).
* **Generalization:** Hope that learning on training data helps classify **new, unseen images**.
* **Learning is Optimization:** It's essentially finding the minimum of a function.

#### Cost Function
* **Purpose:** Measures how "bad" the network's current performance is.
* **Calculation:** For a training example, sum of squared differences between network's output activations and desired (correct) output activations.
    * *Example:* If target is '3' (output neuron 3 = 1, others = 0), and network outputs (0.1, 0.2, 0.9, ...), the cost is high.
* **Overall Cost:** Average cost over *all* training examples.
* **Input:** The 13,000 weights and biases.
* **Output:** A single number representing "lousiness."
* **Goal of Learning:** **Minimize** this cost function.

#### Gradient Descent: The Learning Algorithm
* **Analogy:** Finding the lowest point in a multi-dimensional "valley."
* **Concept:** Start anywhere, then repeatedly take small steps in the **steepest downhill direction**.
* **Gradient:** In multivariable calculus, the gradient is a vector that points in the direction of **steepest ascent**.
    * **Negative Gradient:** Points in the direction of **steepest descent** (downhill). Its magnitude indicates steepness.
* **Process:**
    1.  Initialize weights/biases randomly.
    2.  Calculate the negative gradient of the cost function at the current parameter values.
    3.  Adjust weights/biases in the direction of the negative gradient (take a small "step downhill").
    4.  Repeat steps 2 & 3.
* **Outcome:** Converges to a **local minimum** of the cost function. (Not guaranteed global minimum).
* **Benefit of Continuous Activations:** Allows for a "smooth" cost function, making gradient descent feasible.
* **"Learning" means:** Minimizing this cost function using gradient descent.

#### Backpropagation (Next Video's Topic)
* The efficient algorithm for computing the **gradient** of the cost function. This is the core of NN learning.

#### Network Performance & Limitations
* **Simple NN Performance:** This basic network (two 16-neuron hidden layers) achieves ~96% accuracy on unseen digits. With tweaks, up to 98%.
* **Unexpected Hidden Layer Behavior:**
    * **Initial Hope:** Hidden layers would detect clear features (edges, loops).
    * **Reality:** The weights connecting layers often look "random," not clearly defined edges. The network finds a successful local minimum, but not necessarily by forming human-interpretable features.
* **Network "Confidence" Issues:**
    * If fed random noise, the network still confidently outputs a digit, showing it doesn't understand "uncertainty" or how to *generate* digits.
    * It's trained only on clean, centered digits and optimized for confidence, not common-sense understanding.

#### Modern Context
* This "plain vanilla" network is an older technology (1980s-90s) [multilayer perceptron]
* It's a necessary **foundation** for understanding more advanced, powerful modern variants.

### Video 3: Neural Networks: Backpropagation

#### Recap: Network & Cost
* **Network Structure:** 784 input neurons (28x28 pixel image), two 16-neuron hidden layers, 10 output neurons (digits 0-9).
* **Parameters:** ~13,000 **weights** and **biases** that define network behavior.
* **Cost Function:** Measures network "lousiness." Calculated as the average squared difference between network output and desired output over all training examples.
* **Learning Goal:** Find weights and biases that **minimize this cost function** using **gradient descent**.
    * **Gradient:** A vector indicating the direction of steepest *increase* in cost.
    * **Negative Gradient:** The direction of steepest *decrease* in cost ("downhill"). Its components tell you how sensitive the cost is to each weight/bias change (which "nudge" gives "most bang for buck").

#### What is Backpropagation?
* **Backpropagation (Backprop):** An algorithm to **efficiently compute this negative gradient** of the cost function with respect to all weights and biases.
* It determines how each weight and bias should be adjusted to decrease the cost.

#### Intuitive Walkthrough (for a Single Training Example)
Let's consider one input image (e.g., a "2") where the network currently outputs random activations.

1.  **Output Layer Desired Changes:**
    * We want the "2" output neuron's activation to increase (towards 1) and all other output neurons' activations to decrease (towards 0).
    * The **size of desired nudges** is proportional to how far off the current activation is from its target. (e.g., if a neuron is already close to 0, its desired decrease is small).

2.  **Influencing the Output Neuron (e.g., the '2' neuron):**
    * A neuron's activation is $\sigma(\sum (\text{weight} \times \text{prevactivation}) + \text{bias})$.
    * To increase this output neuron's activation, we have three avenues:
        * **Increase its bias.**
        * **Increase its incoming weights:** Especially for connections from *bright* neurons in the previous layer, as these have a larger impact on the weighted sum ("more bang for your buck"). This relates to **Hebbian theory**: "neurons that fire together, wire together."
        * **Change previous layer's activations:** Make activations connected by *positive* weights brighter, and those by *negative* weights dimmer. Again, changes proportional to weight strength are most effective.

3.  **Propagating Backwards:**
    * We can't directly change previous layer activations.
    * However, the "desired changes" for the previous layer's activations from the output layer's perspective become the **"error signals"** for that previous layer.
    * Each neuron in the hidden layer receives error signals from *all* connected neurons in the next layer. These signals are summed up (proportionally to weights) to determine the "total desired nudge" for that hidden neuron.
    * This process **recursively applies** the same logic: once we have desired nudges for a layer's activations, we determine how their *own* incoming weights and biases should change to achieve those nudges. This "backwards propagation" gives the algorithm its name.

#### Combining Training Examples: Stochastic Gradient Descent (SGD)
* Each training example suggests a slightly different set of nudges for weights and biases.
* To get a full gradient descent step:
    1.  **True Gradient Descent (ideal but slow):** Compute the "desired nudges" (gradient contribution) for *every single training example*, then **average** them all.
    2.  **Stochastic Gradient Descent (SGD) (practical):**
        * Randomly **shuffle** all training data.
        * Divide into small **mini-batches** (e.g., 100 examples).
        * For each mini-batch, compute the "desired nudges" and make a gradient step.
        * This is an **approximation** of the true gradient but is much faster computationally.
        * *Analogy:* "A drunk man stumbling aimlessly down a hill but taking quick steps," rather than a "carefully calculating man taking slow, exact steps."
* **Result:** Repeatedly applying SGD steps across mini-batches will converge to a **local minimum** of the cost function, making the network perform well on training data.

#### Key Takeaway
* Backpropagation is the method to figure out exactly how much and in what direction to adjust *each* of the network's thousands of weights and biases to reduce the overall error.
* **Requirement:** Needs ample **labeled training data** (like the MNIST database for digits).

### Video 4: Neural Networks: Backpropagation (Calculus) 📐

#### Goal
* Understand **how the chain rule is applied** in neural networks to calculate gradients.
* This is crucial for **gradient descent** – minimizing the cost function.

#### Simple Case: Single Neuron Per Layer
* **Network:** Input $\rightarrow$ Neuron $(L-1) \rightarrow$ Neuron $(L)$ (Output).
* **Variables:**
    * $a^{(L-1)}$: Activation of neuron in layer $L-1$.
    * $a^{(L)}$: Activation of neuron in layer $L$.
    * $y$: Desired output for $a^{(L)}$ for a given training example.
    * $C_0 = (a^{(L)} - y)^2$: Cost for this single training example.
    * $w^{(L)}$: Weight connecting $a^{(L-1)}$ to $a^{(L)}$.
    * $b^{(L)}$: Bias for $a^{(L)}$.
    * $z^{(L)} = w^{(L)}a^{(L-1)} + b^{(L)}$: Weighted sum before activation function.
    * $a^{(L)} = \sigma(z^{(L)})$ (where $\sigma$ is the sigmoid or ReLU function).

* **Dependency Chain:**
  ```
      w^L
           \
    a^(L-1)  --> z^L --> a^L --> C_0
           /
      b^L
  ```
* **Goal:** Find $\frac{\partial C_0}{\partial w^{(L)}}$ (how sensitive $C_0$ is to $w^{(L)}$). This tiny nudge to $w^{(L)}$ causes some nudge to $z^{(L)}$ which in turn causes some nudge to $a^{(L)}$ which directly influences the $C_0$

#### The Chain Rule in Action
* The chain rule allows us to break down derivatives through the dependency chain:
    $\frac{\partial C_0}{\partial w^{(L)}} = \frac{\partial C_0}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial w^{(L)}}$

* **Calculating Each Term:**
    1.  **$\frac{\partial C_0}{\partial a^{(L)}} = 2(a^{(L)} - y)$**: Measures how far off the output is. Larger difference means greater sensitivity.
    2.  **$\frac{\partial a^{(L)}}{\partial z^{(L)}} = \sigma'(z^{(L)})$**: The derivative of the activation function (e.g., derivative of sigmoid).
    3.  **$\frac{\partial z^{(L)}}{\partial w^{(L)}} = a^{(L-1)}$**: The sensitivity depends on the activation of the *previous* neuron. A brighter previous neuron means changes to its incoming weight have a larger impact (connects to "neurons that fire together, wire together" idea).

* **Bias Sensitivity ($\frac{\partial C_0}{\partial b^{(L)}}$):**
    * Very similar to weight sensitivity. Only $\frac{\partial z^{(L)}}{\partial b^{(L)}}$ changes, which is simply $1$.
    * So, $\frac{\partial C_0}{\partial b^{(L)}} = \frac{\partial C_0}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot 1$.

* **Propagating Backwards to Previous Activations:**
    * $\frac{\partial C_0}{\partial a^{(L-1)}} = \frac{\partial C_0}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial a^{(L-1)}}$
    * Here, $\frac{\partial z^{(L)}}{\partial a^{(L-1)}} = w^{(L)}$.
    * This term ($\frac{\partial C_0}{\partial a^{(L-1)}}$) tells us how sensitive the cost is to the activation of the *previous* layer. This "error signal" is then used to calculate derivatives for weights/biases in *earlier* layers, iteratively moving backward.

#### Generalizing to Multiple Neurons Per Layer
* The core chain rule application remains the same.
* **Notation:** Indices added to specify neurons within a layer.
    * $a_j^{(L)}$: Activation of $j^{th}$ neuron in layer $L$.
    * $y_j$: Desired output for $a_j^{(L)}$.
    * $C_0 = \sum_j (a_j^{(L)} - y_j)^2$: Cost is sum of squared differences over all output neurons.
    * $w_{jk}^{(L)}$: Weight connecting $k^{th}$ neuron in layer $L-1$ to $j^{th}$ neuron in layer $L$.
    * $z_j^{(L)} = \sum_k (w_{jk}^{(L)} a_k^{(L-1)}) + b_j^{(L)}$.
* **Derivatives still look similar:** $\frac{\partial C_0}{\partial w_{jk}^{(L)}}$ follows the chain rule for that specific weight's path.
* **Key Difference:** How a neuron in layer $L-1$ influences the cost.
    * **$\frac{\partial C_0}{\partial a_k^{(L-1)}}$:** This neuron ($a_k^{(L-1)}$) now influences **multiple neurons** in layer $L$ ($a_0^{(L)}, a_1^{(L)}, \dots$).
    * Therefore, its sensitivity to cost is the **sum of its influences through all paths** to the next layer's neurons:
        $\frac{\partial C_0}{\partial a_k^{(L-1)}} = \sum_j \left( \frac{\partial C_0}{\partial a_j^{(L)}} \cdot \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} \cdot \frac{\partial z_j^{(L)}}{\partial a_k^{(L-1)}} \right)$
        * Where $\frac{\partial z_j^{(L)}}{\partial a_k^{(L-1)}} = w_{jk}^{(L)}$.

#### Conclusion
* Backpropagation computes all partial derivatives of the cost function with respect to every weight and bias.
* These derivatives form the **gradient vector**, which dictates the adjustments in **gradient descent** to minimize the network's cost.
* The process involves repeatedly applying the chain rule backward through the network, leveraging the "error signals" from later layers.


### Video 5: Large Language Models (LLMs) & Transformers

#### What is an LLM?
* A **Large Language Model** is a sophisticated mathematical function that **predicts the next word** in any given text.
* It assigns a **probability** to all possible next words, not a single certain word.
* **Chatbot Interaction:** LLMs complete dialogue by repeatedly predicting the next word of an AI assistant's response.
* **Non-Deterministic Output:** While the model itself is deterministic, randomness in selecting less likely words makes the output feel more natural and often produces different answers for the same prompt.

#### LLM Training (Pre-training)
* **Data:** LLMs are trained on **enormous amounts of text**, typically from the internet. (e.g., GPT-3 training data would take a human over 2600 years to read).
* **Parameters/Weights:** An LLM's behavior is determined by **hundreds of billions** of these continuous values. These are the "dials" being tuned.
    * No human manually sets these; they start randomly (gibberish output).
* **Refinement Process:**
    1.  The model is fed training text with the **last word removed**.
    2.  Its prediction for that missing word is compared to the **true last word**.
    3.  **Backpropagation** (the algorithm from previous videos) is used to **tweak all parameters**. This makes the model more likely to predict the correct word and less likely to predict others.
* **Scale of Computation:** Training large LLMs is **mind-boggling**. It can take over **100 million years** for a supercomputer performing a billion operations per second.
* **Result:** After trillions of examples, the model not only predicts accurately on training data but also makes **reasonable predictions on unseen text**.

#### Reinforcement Learning with Human Feedback (RLHF)
* **Pre-training Goal:** Auto-completing random text.
* **Chatbot Goal:** Being a good AI assistant.
* **RLHF addresses this gap:**
    * Human workers **flag unhelpful or problematic predictions**.
    * Their corrections further **adjust model parameters**, making it more likely to give user-preferred responses.

#### Transformers & Parallelization
* **GPUs:** Special computer chips optimized for **parallel operations** are crucial for the massive scale of LLM training.
* **Pre-2017 Models:** Processed text **one word at a time** (difficult to parallelize).
* **Transformers (Introduced 2017 by Google):**
    * A new model architecture that processes text **all at once, in parallel**.
    * **Word Embeddings:** Each word is converted into a **long list of numbers** (a vector). This encodes its meaning and allows numerical processing.
    * **Attention Mechanism:** The unique feature of Transformers. It allows these numerical word representations to "talk to one another" and **refine their meanings based on context** (e.g., "bank" in "river bank"). This happens in parallel. 
    * **Feed-Forward Neural Networks:** Also included, providing additional capacity to store language patterns.
    * **Iterative Flow:** Data flows through many iterations of attention and feed-forward networks, enriching the word embeddings.
    * **Final Prediction:** A final function on the last word's vector (influenced by all context) produces probabilities for the next word.

#### Emergent Behavior
* The specific, fluent, and useful behavior of LLMs is an **emergent phenomenon** from the tuning of billions of parameters during training.
* This makes it **challenging to interpret exactly why** an LLM makes specific predictions.

### Video 6: GPT & Transformers: The Core Mechanics 🛠️

#### What do GPT Initials Mean?
* **G**enerative: These models **generate** new text.
* **P**retrained: Models learn from a **massive amount of data** (e.g., internet text). "Pre" implies further fine-tuning is possible.
* **T**ransformer: A **specific neural network architecture** that is the core invention behind the current AI boom.

#### Transformer Applications
* Originally for **language translation** (Google, 2017).
* Now used for:
    * **Text generation** (like ChatGPT).
    * **Text-to-speech** and **speech-to-text**.
    * **Text-to-image** (e.g., DALL-E, Midjourney).

#### How LLMs Generate Text (Next Word Prediction)
* **Core Task:** Predict the **next word** (or token) given previous text.
* **Prediction Form:** A **probability distribution** over all possible next words/chunks.
* **Generation Process:**
    1.  Give the model an **initial snippet (seed text)**.
    2.  Model predicts probabilities for the next token.
    3.  **Randomly sample** a token from this distribution (allowing less likely words for naturalness).
    4.  **Append** the sampled token to the text.
    5.  **Repeat** the process with the new, longer text.
* **Scale Matters:** Small models (e.g., GPT-2) generate less coherent text; much larger models (e.g., GPT-3) generate sensible, even creative, stories.

#### High-Level Transformer Data Flow
1.  **Tokenization:** Input text (or image/sound) is broken into **tokens** (words, sub-words, or common character combos).
2.  **Embedding:** Each token is converted into a **vector** (a list of numbers) that encodes its meaning. If you think of these vectors as coordinates in some high dimensionsal space, words with similar meanings tend to land on vectors that are close to each other in that space.
3.  **Attention Block:** These vectors "talk to each other" in parallel. They update their values (meanings) based on **context** (e.g., "model" in "machine learning model" vs. "fashion model"). This figures out relevance and updates meanings.
4.  **Multi-Layer Perceptron (MLP) / Feed-Forward Layer:** Vectors pass through this independently (in parallel). Here the vectors don't talk to each other. It gives the model extra capacity to learn and store language patterns.
5.  **Repeat:** Steps 3 and 4 (Attention and MLP) are **repeated many times** (multiple blocks/layers).
6.  **Final Prediction:** The last vector in the sequence (now enriched with full context) is used to produce a **probability distribution** for the next token.

#### Deep Learning Fundamentals for Transformers
* **Machine Learning:** Using data to determine a model's behavior, rather than explicit coding.
    * *Example:* Linear Regression finds best-fit line parameters (slope, intercept) from data.
* **Deep Learning:** A class of models (like NNs) that **scale remarkably well** with data and parameters.
* **Key Format Requirements for Scalable Training (Backpropagation):**
    1.  **Input/Layers as Arrays (Tensors):** All data and intermediate layers are formatted as arrays of real numbers.
    2.  **Weights for Weighted Sums:** Model parameters are mostly **weights**, which interact with data through **weighted sums**. (Often seen as **matrix multiplications**).
        * GPT-3 has 175 billion weights, organized into ~28,000 matrices.
    3.  **Data vs. Weights:** Crucial distinction:
        * **Weights (blue/red):** The "brains," learned during training, determine behavior.
        * **Data (gray):** The specific input being processed for a given run.

<img src="/metadata/gpt3.png" width="700" />

#### Initial & Final Steps in Detail

##### 1. Token Embedding
* **Vocabulary:** The model has a fixed list of all possible tokens (e.g., 50,257 for GPT-3).
* **Embedding Matrix ($W_E$):** The first set of weights. Each column in this matrix represents a **vector for a specific token**.
    * Initially random, learned during training.
    * **High-Dimensional Space:** Word embeddings are high-dimensional vectors (e.g., GPT-3 uses 12,288 dimensions).
    * **Semantic Meaning:** During training, models learn embeddings where **directions in the space carry meaning**.
        * *Example:* "woman minus man" vector is similar to "queen minus king" vector. 
        * This allows for "vector math" to find related words (e.g., Italy - Germany + Hitler $\approx$ Mussolini).
* **Parameter Count:** GPT-3's embedding matrix adds ~617 million parameters.

<img src="/metadata/embed_mat.png" width="700" />

This embedding matrix is also learned during the training. Then during inference, we just lookup the relevant column vectors (from the embedding matrix) for the prompt tokens and feed those vectors to the model to generate output.

##### 2. Context & Positional Encoding
* **Context Size:** Transformers process a fixed number of vectors at a time (e.g., 2048 for GPT-3). This limits the amount of text the model can consider for prediction.
* **Positional Encoding:** Vectors also encode information about the **position** of the word in the sequence (to be covered later).
* **Soaking in Context:** The network's primary goal is to enrich these initial word vectors with context from the entire input sequence.

##### 3. Unembedding Matrix & Softmax
* **Prediction:** Only the **last vector** in the processed sequence is used to predict the next token. (Though in training, all vectors simultaneously predict their immediate next token for efficiency).
* **Unembedding Matrix ($W_U$):** Another weight matrix that maps the final context-rich vector to a list of 50,000+ values (one for each token in the vocabulary).
    * Adds another ~617 million parameters. Total parameters now slightly over 1 billion.
* **Softmax Function:**
    * **Purpose:** Converts an arbitrary list of numbers (called **logits**) into a valid **probability distribution** (values between 0 and 1, summing to 1).
    * **Mechanism:** $P(i) = e^{z_i} / \sum_k e^{z_k}$ (where $z_i$ are the inputs/logits).
    * **Effect:** Larger input values get probabilities closer to 1; smaller values get probabilities closer to 0. It's "soft" because similar large values still get significant weight. Sum of all the values is equal to 1.
    * **Temperature (T):** An optional constant in softmax.  $P(i) = e^{z_i / T} / \sum_k e^{z_k / T}$ (where $z_i$ are the inputs/logits and T is the temperature).
        * **Higher T:** We give more weight to lower values, which makes the distribution more uniform (more "creative" or "random" output).
        * **Lower T:** We give more weight to higher values, which makes the largest values dominate more aggressively (more "predictable" output). For T=0, the probability of token with highest $z_i$ is 1, and everything else is 0.
        * *Example:* Generating a story with T=0 gives a predictable, trite outcome; higher T gives more original but potentially nonsensical results.
        * Usually max temperature that APIs allow is 2 (because beyond that model starts spitting out nonsensical things)

#### Foundation for Attention
* Understanding **word embeddings**, **softmax**, how **dot products measure similarity**, and the dominance of **matrix multiplication with tunable parameters** are essential prerequisites for grasping the **attention mechanism**, the true heart of Transformers.

### Video 7: Transformers: The Attention Mechanism ✨

#### Context Recap
* **Goal:** Predict the next word (or token) in a text sequence.
* **Input:** Text broken into **tokens** (we'll pretend they're words for simplicity).
* **Initial Step:** Each token is converted into a **high-dimensional vector (embedding)**.
    * **Key Idea:** Directions in this embedding space correspond to **semantic meaning** (e.g., gender, geographical origin).
* **Transformer's Aim:** Refine these initial embeddings to bake in **rich contextual meaning**.

#### Why Attention? The Need for Context
* **Problem:** Initial token embeddings are context-free. "Mole" in "American shrew mole," "mole of carbon," and "biopsy of the mole" all start with the **same** initial embedding.
* **Solution:** Attention allows embeddings to exchange information and update their meanings based on **surrounding context**.
    * *Example:* "Mole" should be adjusted to mean "animal," "chemical unit," or "skin growth" based on its context.
    * *Example:* "Tower" after "Eiffel" should be updated to mean "Eiffel Tower" (associated with Paris, France, steel) and if "miniature" is added, its "tallness" aspect should diminish.
* **Broader Goal:** Attention moves information across distant parts of the sequence, allowing the final token's embedding (which predicts the next word) to incorporate all relevant context from the entire input.

#### Single Head of Attention: The Computations
"a fluffy blue creature roamed the verdant forest" when changed to embeddings is say 'E1 E2 E3 E4 E5 E6 E7 E8". Assume, motivation here is to have nouns be enriched with adjective info ("creature" with "fluffy" and "blue"; "forest" with "verdant").

Attention involves three types of learned matrices (weights) that transform the embeddings:

1.  **Queries (Q):**  
    <img src="/metadata/attn1.png" width="500" />
    * Each input embedding ($e$) is multiplied by a **Query Matrix ($W_Q$)** to produce a **query vector ($q$)**.
    * *Conceptual Role:* Like asking a question about the word's context (e.g., a noun "creature" asking "Are there any adjectives preceding me?").
    * Dimension of $q$ is smaller than $e$ (e.g., 128 for GPT-3).

2.  **Keys (K):**
    * Each input embedding ($e$) is multiplied by a **Key Matrix ($W_K$)** to produce a **key vector ($k$)**.
    * *Conceptual Role:* Like providing an answer to a query (e.g., an adjective "fluffy" answering "Yes, I'm an adjective in this position.").
    * Same dimension as $q$.

3.  **Dot Products & Attention Pattern:**  
    <img src="/metadata/attn2.png" width="700" />
    * **Measure Alignment:** The **dot product** of each **Query** with every **Key** is computed. This measures how well each key "answers" each query, indicating relevance.
    * *Example:* High dot product between "creature" (query) and "fluffy" (key) means "fluffy" is highly relevant to "creature."
    * **Attention Scores:** These dot products form a grid of scores.
    * **Normalization (Softmax):** Each column of scores (representing how relevant all *other* words are to a *given* word) is normalized using **Softmax**. This turns scores into probabilities (0-1, sum to 1). This resulting grid is the **attention pattern**.
        * **Attention(Q, K, V) =** $\text{Softmax}(\frac{QK^T}{\sqrt{d_k}})$ * V
        * where $Q$ is the matrix of all queries, $K^T$ is the transpose of the matrix of all keys, and $\sqrt{d_k}$ is for numerical stability.  this $d_k$ is dimension of key vector

4.  **Masking (for Causal LLMs like GPT):**
    <img src="/metadata/mask.png" width="700" />
    * During training, transformers predict words sequentially. To prevent "cheating," **later tokens are prevented from influencing earlier tokens**.
    * This is done by setting dot products for "future" tokens to **negative infinity** *before* softmax. After softmax, these become 0.

Note that this matrix size is product of context length -- that's why it's very challenging for AI Labs to scale context window. Other techniques being used here: Sparse Attention Mechanisms, Blockwise Attention, Linformer, Reformer, Ring attention, Longformer, Adaptive Attention Span.

5.  **Values (V) & Updating Embeddings:**
    <img src="/metadata/val_mat.png" width="700" />
    * Each input embedding ($e$) is multiplied by a **Value Matrix ($W_V$)** to produce a **value vector ($v$)**.
    * *Conceptual Role:* If a word is relevant, what information should it "add" to the target word's embedding?
    * **Updating:** For each target embedding (column in the attention pattern):
        * Take a **weighted sum** of all other words' **value vectors**, using the attention pattern probabilities as weights.
        * This sum is the $\Delta e$ (change) to be added to the original embedding.
        * The new, refined embedding is $e + \Delta e$.
    * **Value Matrix Factoring:** It looks like value matrix dimension is context length * context length (12288 * 12288) but in practice, $W_V$ is often factored into two smaller matrices (Value Down, Value Up) to save parameters, especially in multi-head attention -- so two matrices of 12288 * 128 and 128 * 12288 sizes. So it looks like value up * value down * e = v

#### Multi-Headed Attention
 <img src="/metadata/multi_attn.png" width="700" />
 
* **Concept:** Instead of just one attention calculation, **many "attention heads" run in parallel**, each with its own distinct $W_Q, W_K,$ and $W_V$ matrices.
* **Purpose:** Allows the model to learn **many different types of contextual relationships** simultaneously (e.g., one head for adjective-noun, another for verb-object, another for sentiment).
* **Combination:** Each head produces a proposed change to the embedding. These proposed changes are **summed together** and added to the original embedding.
* **Parameter Count (GPT-3 example):**
    * GPT-3 uses **96 attention heads** per block.
    * Each head: ~6.3 million parameters (for $W_Q, W_K, W_{V\_down}, W_{V\_up}$).
    * Total per block: $96 \times 6.3 \text{ million} \approx 600 \text{ million parameters}$.
    * GPT-3 has **96 such blocks** (layers), leading to almost **58 billion parameters** for attention alone.

 <img src="/metadata/val_out.png" width="700" />
* We thought of this value matrix as two matrices of 12288 * 128 and 128 * 12288 sizes. But in papers, only second one is called value matrix. All the first one value matrix in a block are concatenated, and it's called output matrix.

#### Other Components & Scaling
* **Multi-Layer Perceptrons (MLPs):** Data also flows through MLP blocks *between* attention blocks. These account for the majority of the remaining parameters in LLMs (to be discussed later).
* **Context Size:** The number of tokens a transformer can process at once ($N^2$ complexity for attention). For GPT-3, it's 2048 tokens. This is a major bottleneck for long conversations.
* **Parallelizability:** Attention is highly parallelizable, which allows for the massive scaling of models using **GPUs**. This ability to scale is a key reason for the recent success of deep learning.

Some self-attention mechanisms are bidirectional, meaning that they calculate relevance scores for tokens preceding and following the word being attended to. For example, in Figure 3, notice that words on both sides of it are examined. So, a bidirectional self-attention mechanism can gather context from words on either side of the word being attended to. By contrast, a unidirectional self-attention mechanism can only gather context from words on one side of the word being attended to. Bidirectional self-attention is especially useful for generating representations of whole sequences, while applications that generate sequences token-by-token require unidirectional self-attention. For this reason, encoders use bidirectional self-attention, while decoders use unidirectional.

### Video 8: Transformers: Multi-Layer Perceptrons (MLPs) 🧠

#### MLP's Role in Transformers
* **MLPs** are the other major component of a transformer, alongside **Attention**.
* While Attention handles **context**, MLPs are thought to be the primary place where **facts and knowledge are stored**.
* The computation is relatively simple, but the interpretation of what it's doing is complex.
* This section will use a toy example: storing the fact that "Michael Jordan plays basketball."

#### The Core MLP Operation
 <img src="/metadata/tran_mlp.png" width="700" />
An MLP block processes each vector from the sequence **independently and in parallel** (they don't "talk" to each other here). The operation on a single vector ($E$) is as follows:

1.  **"Up" Projection (Questions):**
    * The vector $E$ is multiplied by a large **"up" projection matrix ($W_{\text{up}}$)**. A bias vector ($B_{\text{up}}$) is also added.
    * This matrix is filled with learned parameters.
    * **Conceptual Meaning:** Each **row** of $W_{\text{up}}$ can be thought of as a **"question"** about the input vector's features. A dot product between the row and the vector measures how much the vector aligns with that feature.
        * *Example:* A row could represent "Michael" + "Jordan." The dot product will be high if the vector $E$ encodes both names.
    * This matrix maps the vector to a much higher-dimensional "neuron" space. (e.g., in GPT-3, it's 4x the embedding dimension, or nearly 50,000 dimensions).

2.  **Activation Function (ReLU):**
    * The resulting high-dimensional vector is passed through a simple **non-linear function**.
    * A common choice is the **ReLU (Rectified Linear Unit)**: it sets all negative values to zero and leaves positive values unchanged.
    * **Conceptual Meaning:** This creates a clean "yes/no" or "AND gate" behavior.
        * *Example:* A neuron's value becomes positive **only if** the vector encodes both "Michael" and "Jordan," while staying zero otherwise.
    * The positive values in this vector are what are referred to as "active neurons."

3.  **"Down" Projection (Actions):**  
    * The activated neuron vector is multiplied by a second matrix, the **"down" projection matrix ($W_{\text{down}}$)**. A bias vector ($B_{\text{down}}$) is added.
    * This matrix maps the vector back down to the original embedding dimension.
    * **Conceptual Meaning:** Each **column** of $W_{\text{down}}$ is a vector that gets **added to the result if its corresponding neuron is active**.
        * *Example:* The column corresponding to our "Michael Jordan" neuron could be the "basketball" vector. If the neuron is active, this "basketball" knowledge is added to the output.

  <img src="/metadata/down_proj.png" width="700" />
  
4.  **Residual Connection:**
    * The final output of the MLP is **added to the original input vector ($E$)**.
    * The sum is the final vector flowing out of the MLP block.

#### Parameter Breakdown (GPT-3)
* **MLP Blocks** make up the majority of a transformer's parameters.
* **Up & Down Projection Matrices:**
    * $W_{\text{up}}$: ~12,288 columns (embedding dim) $\times$ ~50,000 rows (neuron dim) $\approx$ 604 million parameters.
    * $W_{\text{down}}$: Transposed, same number of parameters.
    * Total per MLP: ~1.2 billion parameters.
* **Total MLP Parameters:** GPT-3 has **96 MLP blocks**, for a total of **~116 billion parameters** devoted to MLPs.
* **Grand Total:** This accounts for ~2/3 of the total 175 billion parameters in GPT-3.

#### A Note on Superposition
<img src="/metadata/superpos.png" width="700" />

* **Traditional View:** We might assume each neuron represents a single, distinct feature (e.g., "Michael Jordan"). This would limit the number of features to the number of neurons.
* **Superposition Hypothesis:** It's more likely that models use **"nearly perpendicular" directions** to encode features.
    * In high dimensions (say N)
        * you can fit N vectors which are perpendicular to each other [all vectors at 90 degree to each other]
        * you can fit **exponentially more** "nearly perpendicular" vectors (say b/w 89 and 91 degrees) than strictly perpendicular ones.
    * This means a transformer can store **far more ideas** than the number of neurons it has.
    * **Result:** Individual features might not be a single neuron but a **combination of several neurons**, making the network harder to interpret but much more powerful and scalable. This could be a key reason why LLMs perform so well with size. 
