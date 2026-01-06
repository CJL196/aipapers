# 1. Bibliographic Information

## 1.1. Title
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## 1.2. Authors
The authors are Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, and Daya Guo. The affiliations listed are DeepSeek-AI, Tsinghua University, and Peking University. This indicates a strong collaboration between a prominent AI company and top-tier academic institutions in China, blending industrial engineering prowess with academic research rigor.

## 1.3. Journal/Conference
The paper was published on arXiv, a popular preprint server for academic papers in fields like physics, mathematics, computer science, and quantitative biology. As a preprint, it has not yet undergone a formal peer-review process for publication in a conference or journal. However, arXiv is a primary venue for the rapid dissemination of cutting-edge research in the AI community, and papers published there often have a significant and immediate impact.

## 1.4. Publication Year
The paper was first submitted to arXiv on February 5, 2024. The version analyzed here is v3, which was updated from the original.

## 1.5. Abstract
The abstract introduces `DeepSeekMath 7B`, a language model specializing in mathematical reasoning. It is built by continuing the pre-training of `DeepSeek-Coder-Base-v1.5 7B` on a massive 120 billion token dataset of math-related content sourced from Common Crawl, supplemented with natural language and code data. The model achieves a remarkable 51.7% accuracy on the challenging competition-level MATH benchmark without using external tools or voting, a result that approaches the performance of leading closed-source models like Gemini-Ultra and GPT-4. The paper attributes this success to two main innovations: (1) a sophisticated data pipeline for extracting high-quality math data from public web sources, and (2) a new reinforcement learning algorithm called **Group Relative Policy Optimization (GRPO)**. GRPO is presented as a memory-efficient variant of Proximal Policy Optimization (PPO) that enhances mathematical reasoning abilities.

## 1.6. Original Source Link
-   **Original Source Link:** `https://arxiv.org/abs/2402.03300`
-   **PDF Link:** `https://arxiv.org/pdf/2402.03300v3.pdf`
-   **Publication Status:** This is a preprint and has not been formally published in a peer-reviewed venue as of the time of this analysis.

# 2. Executive Summary

## 2.1. Background & Motivation
-   **Core Problem:** Mathematical reasoning remains a formidable frontier for Large Language Models (LLMs). Unlike natural language, mathematics demands strict logical consistency, symbolic manipulation, and multi-step structured reasoning, which are difficult for current models to master.
-   **Importance & Gap:** While state-of-the-art models like GPT-4 have shown impressive mathematical capabilities, they are proprietary and closed-source. This limits the broader research community's ability to study, build upon, and understand the mechanisms behind advanced AI reasoning. The open-source models available have historically lagged far behind in performance on complex mathematical benchmarks.
-   **Innovative Idea:** The paper's central hypothesis is that an open-source model can close this performance gap by focusing on two key areas. First, instead of relying on curated but limited academic datasets (like arXiv papers), they propose to systematically mine the vast, publicly available Common Crawl web data for high-quality mathematical content at an unprecedented scale. Second, they aim to develop a more resource-efficient reinforcement learning (RL) algorithm to further align the model for mathematical problem-solving, moving beyond standard supervised fine-tuning.

## 2.2. Main Contributions / Findings
The paper makes several significant contributions to the field of AI and mathematical reasoning:

1.  **The DeepSeekMath Corpus:** The authors constructed a massive, high-quality pre-training dataset of **120 billion tokens** of math-related content by developing an iterative filtering pipeline for Common Crawl data. This is significantly larger than previous open math datasets like OpenWebMath and the math portion of Minerva's training data.
2.  **State-of-the-Art Open-Source Models:** They release a series of models (`DeepSeekMath-Base`, `DeepSeekMath-Instruct`, and `DeepSeekMath-RL`) that set a new standard for mathematical reasoning in open LLMs. The final model, `DeepSeekMath-RL 7B`, achieves **51.7% accuracy** on the MATH benchmark (pass@1), becoming the first open-source model to cross the 50% threshold and rivaling closed-source giants.
3.  **Group Relative Policy Optimization (GRPO):** They introduce GRPO, a novel and efficient RL algorithm. By replacing the memory-intensive critic model of Proximal Policy Optimization (PPO) with a baseline derived from the average reward of a group of sampled responses, GRPO significantly reduces training resource requirements while effectively improving model performance.
4.  **Key Empirical Insights:**
    *   **Code Pre-training Boosts Math Reasoning:** The paper provides strong evidence that initializing from a code-trained model (`DeepSeek-Coder`) is more effective for mathematical tasks than starting from a general-purpose LLM.
    *   **The Surprising Ineffectiveness of arXiv Data:** Counter-intuitively, their experiments show that training on arXiv papers (a common practice for math models) did not lead to notable improvements on the evaluated benchmarks, suggesting that web data may be a more valuable resource.
    *   **A Unified Framework for RL Methods:** The paper presents a paradigm to understand and compare different alignment methods like Supervised Fine-Tuning (SFT), Rejection Sampling Fine-Tuning (RFT), Direct Preference Optimization (DPO), and PPO/GRPO, analyzing them based on their data source, reward function, and gradient update mechanism.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

To fully appreciate the paper's contributions, it's essential to understand the following concepts:

-   **Large Language Models (LLMs):** LLMs are advanced artificial intelligence models, most commonly based on the **Transformer architecture**. They are trained on enormous amounts of text data in a self-supervised manner (a process called **pre-training**), enabling them to understand and generate human-like language. Their ability to perform a wide range of tasks, including reasoning, stems from the patterns and knowledge learned during this phase.

-   **Three-Stage LLM Training Paradigm:** Modern high-performance LLMs are typically developed through a three-stage process:
    1.  **Pre-training:** The model learns general knowledge, grammar, and reasoning patterns from vast, unlabeled text corpora (e.g., the entire internet).
    2.  **Supervised Fine-Tuning (SFT):** The pre-trained model is further trained on a smaller, high-quality dataset of curated input-output pairs (e.g., question-answer pairs). This stage teaches the model to follow instructions and produce outputs in a desired format.
    3.  **Reinforcement Learning from Human Feedback (RLHF):** The model's behavior is further refined using reinforcement learning. A separate "reward model" is trained to score the quality of the LLM's outputs, often based on human preferences. The LLM is then optimized to generate outputs that maximize the score from this reward model. This paper's GRPO is an advanced algorithm used in this stage.

-   **Chain-of-Thought (CoT) and Program-of-Thought (PoT) Prompting:**
    -   **CoT:** A technique to elicit better reasoning from LLMs by prompting them to "think step-by-step." Instead of producing just the final answer, the model generates the intermediate reasoning steps it took to arrive at the solution. This often leads to more accurate results for complex problems.
    -   **PoT:** A variant of CoT where the model generates a program (e.g., in Python) to solve a problem. The program is then executed, and its output is used as the final answer. This is particularly effective for mathematical problems that require precise calculations, as it offloads the computation from the LLM to a reliable code interpreter.

-   **Proximal Policy Optimization (PPO):** PPO is a widely used reinforcement learning algorithm and a cornerstone of RLHF. It's an **actor-critic** method, meaning it uses two main components:
    -   **Actor (The Policy):** This is the LLM itself, which takes an action (generates a token) based on the current state (the input prompt and previously generated tokens).
    -   **Critic (The Value Function):** This is a separate model that estimates the expected future reward from a given state. Its role is to provide a baseline to judge how good the actor's action was. The "advantage" is calculated as the actual reward received minus this baseline estimate.
        PPO's key innovation is its objective function, which "clips" the policy update to prevent it from changing too drastically in a single step, ensuring more stable training. The standard PPO objective function is:
    \$
    L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon)\hat{A}_t\right) \right]
    \$
    where:
    -   $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the new and old policies.
    -   $\hat{A}_t$ is the estimated advantage at timestep $t$.
    -   $\varepsilon$ is a hyperparameter that defines the clipping range.
        Training the critic model requires significant memory, a problem that this paper's GRPO algorithm aims to solve.

## 3.2. Previous Works

-   **Minerva (Lewkowycz et al., 2022a):** A pioneering work from Google that demonstrated the power of domain-specific pre-training for mathematics. Minerva was based on the PaLM architecture and was further trained on a 118GB dataset of scientific papers from arXiv and web pages containing MathJax/LaTeX. This paper builds on Minerva's idea of domain-specific training but shifts the data focus from arXiv to the broader web (Common Crawl) and achieves superior results with a much smaller model (7B vs. 540B).

-   **Llemma (Azerbayev et al., 2023):** An open-source model for mathematics based on Code Llama. It was further trained on a dataset called `Proof-Pile-2`, which includes web math data (`OpenWebMath`), code, and scientific papers. DeepSeekMath outperforms Llemma, suggesting its data collection pipeline and training methodology are more effective.

-   **WizardMath (Luo et al., 2023):** This work enhanced models like Llama-2 and Mistral for math using a technique called `Evol-Instruct` (generating more complex instruction data) followed by PPO training. It is a key competitor in the space of RL-tuned math models. DeepSeekMath's RL approach with GRPO proves to be more effective.

-   **Rejection Sampling Fine-tuning (RFT) (Yuan et al., 2023a):** An alternative to complex RL algorithms. The idea is to generate multiple solutions for a given problem, use a simple verifier (e.g., checking the final answer) to identify the correct ones, and then fine-tune the model only on these correct solutions. It's a form of "offline" reinforcement.

-   **Direct Preference Optimization (DPO) (Rafailov et al., 2023):** A more recent and elegant alignment algorithm that bypasses the need for an explicit reward model. DPO directly optimizes the LLM on a dataset of preferred and rejected responses. The core idea is that the DPO loss function implicitly defines a reward and optimizes the policy. The loss is given by:
    \$
    \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
    \$
    where:
    -   $(x, y_w, y_l)$ is a data point with a prompt $x$, a "winner" (preferred) response $y_w$, and a "loser" (rejected) response $y_l$.
    -   $\pi_\theta$ is the policy being optimized.
    -   $\pi_{ref}$ is a frozen reference policy.
    -   $\beta$ is a temperature parameter.
    -   $\sigma$ is the sigmoid function.
        This paper includes DPO in its unified analysis of RL methods.

## 3.3. Technological Evolution
The approach to building mathematical LLMs has evolved rapidly:
1.  **General LLMs:** Early attempts used general-purpose LLMs (e.g., GPT-3) with clever prompting (like CoT) to solve math problems. Performance was limited.
2.  **Domain-Specific Pre-training:** Models like Minerva showed that continuing pre-training on math-heavy text (like arXiv papers) significantly boosts performance.
3.  **Instruction Tuning for Math:** Works like `MetaMath` focused on curating large-scale, high-quality instruction-following datasets specifically for math to improve performance through SFT.
4.  **Reinforcement Learning for Math:** Models like `WizardMath` applied RL algorithms like PPO to further refine the model's reasoning process, rewarding correct solution paths.
5.  **DeepSeekMath's Contribution:** This paper represents the next step, integrating several key advancements: starting from a strong code model, using a massive and novel web-based data source for pre-training, and introducing a more efficient, custom-built RL algorithm (GRPO).

## 3.4. Differentiation Analysis
DeepSeekMath's approach is innovative in several key ways compared to prior work:

-   **Data Sourcing Strategy:** While Minerva and Llemma relied heavily on academic sources like arXiv, DeepSeekMath makes a bold and successful bet on the general web (Common Crawl). Their meticulously engineered pipeline extracts a much larger volume of high-quality data, proving that the open web is a rich, underexploited resource for specialized domains.
-   **RL Algorithm Efficiency:** PPO is powerful but resource-hungry due to its critic model. GRPO is a pragmatic and effective innovation that removes the critic, making powerful RL techniques more accessible and scalable by reducing memory overhead.
-   **Foundation Model Choice:** The explicit choice to start from `DeepSeek-Coder` and the subsequent analysis provide some of the clearest evidence to date for the long-held hypothesis that training on code enhances abstract reasoning capabilities, at least for mathematics.
-   **Scale and Openness:** DeepSeekMath achieves performance competitive with proprietary models orders of magnitude larger, all while remaining fully open-source, thereby significantly advancing the capabilities of the entire research community.

# 4. Methodology

The paper's methodology can be broken down into three main stages: (1) large-scale math pre-training, (2) supervised fine-tuning on math instructions, and (3) reinforcement learning using the novel GRPO algorithm.

## 4.1. Math Pre-Training at Scale

The foundation of DeepSeekMath's success is the `DeepSeekMath Corpus`, a massive 120B-token dataset created from Common Crawl.

### 4.1.1. Data Collection and Decontamination

The authors designed an iterative pipeline to discover and filter high-quality mathematical web pages from the vast and noisy Common Crawl dataset. The process is illustrated in Figure 2 from the paper.

![Figure 2 | An iterative pipeline that collects mathematical web pages from Common Crawl.](images/2.jpg)
*该图像是图示，展示了从Common Crawl中收集数学相关网页的迭代流程，包含训练FastText模型、召回数学网页、发现数学相关域名及注释数学相关URL路径四个步骤，形成数学语料库并反馈回数学种子。*

The steps are as follows:
1.  **Initialization with a Seed Corpus:** The process begins with a small, high-quality seed dataset. The authors used `OpenWebMath` (13.6B tokens) as their initial positive examples.
2.  **Training a Classifier:** A `fastText` classifier is trained to distinguish between mathematical and non-mathematical web pages. It is trained on 500,000 positive examples from the seed corpus and 500,000 negative examples randomly sampled from Common Crawl. `fastText` is a lightweight library for efficient text classification and representation learning.
3.  **Recalling Math Pages:** The trained classifier is run over a 40-billion-page deduplicated subset of Common Crawl to "recall" or identify potential math-related pages.
4.  **Refinement and Iteration:** The core innovation lies in the iterative improvement of this process.
    -   **Domain Discovery:** The entire Common Crawl is grouped by domain (base URL). Domains with a high percentage (>10%) of pages recalled by the classifier are flagged as potentially math-rich (e.g., `mathoverflow.net`).
    -   **Human Annotation:** Within these math-rich domains, human annotators identify specific URL patterns corresponding to mathematical content (e.g., $mathoverflow.net/questions$).
    -   **Seed Corpus Expansion:** Pages matching these new patterns, which were not collected in the first pass, are added to the seed corpus as new positive examples.
    -   **Retraining:** The `fastText` classifier is retrained with this enriched and more diverse seed corpus, making it more powerful for the next iteration.
5.  **Convergence:** This iterative loop was repeated four times. The process was stopped when the fourth iteration collected 98% of the same data as the third, indicating that the discovery process had stabilized. The final result was **35.5 million web pages, totaling 120 billion tokens**.

### 4.1.2. Decontamination
To ensure fair evaluation, the training data was "decontaminated" to remove any overlap with standard benchmarks. Any text segment with a 10-gram that exactly matched a substring from evaluation sets (like `GSM8K`, `MATH`, `CMATH`) was removed.

### 4.1.3. Pre-training Process
The `DeepSeekMath-Base 7B` model was created by taking the pre-trained `DeepSeek-Coder-Base-v1.5 7B` model and continuing its training for an additional 500 billion tokens. The data mixture was:
-   **56% DeepSeekMath Corpus** (the web data collected above)
-   **4% AlgebraicStack** (mathematical code)
-   **10% arXiv** (scientific papers)
-   **20% Github code**
-   **10% Natural Language** (from Common Crawl)

    This mixture aimed to bolster mathematical ability while maintaining strong coding and general language skills.

## 4.2. Supervised Fine-Tuning (SFT)

After pre-training, the `DeepSeekMath-Base` model was instruction-tuned to create `DeepSeekMath-Instruct 7B`. This stage teaches the model to follow instructions and generate solutions in specific formats.

-   **SFT Data Curation:** A dataset of 776,000 examples was constructed from various sources, covering English and Chinese problems. The solutions were provided in multiple formats to teach the model diverse problem-solving strategies:
    -   **Chain-of-Thought (CoT):** Step-by-step natural language reasoning.
    -   **Program-of-Thought (PoT):** Python code solutions.
    -   **Tool-Integrated Reasoning:** A combination of text reasoning and calls to tools (like a Python interpreter).
-   **Training Process:** The model was trained for 500 steps with a batch size of 256 and a constant learning rate, using a context length of 4K tokens.

## 4.3. Reinforcement Learning with GRPO

The final and most innovative stage is applying reinforcement learning to further boost the model's mathematical reasoning. The authors introduce **Group Relative Policy Optimization (GRPO)**, a memory-efficient alternative to PPO.

The following figure (Figure 4 from the original paper) provides a high-level comparison of PPO and GRPO.

![Figure 4 | Demonstration of PPO and our GRPO. GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.](images/4.jpg)
*该图像是论文中图4的示意图，展示了传统PPO与本文提出的GRPO两种训练策略的流程对比。GRPO摈弃了价值模型，采用基于组得分的基线估计，显著降低了训练资源消耗。*

### 4.3.1. From PPO to GRPO
The derivation begins with the standard PPO algorithm used in RLHF.

-   **PPO Objective:** The goal of PPO is to maximize a surrogate objective function that encourages the policy (the LLM) to take actions that lead to higher rewards, while not deviating too much from the previous policy to maintain training stability. The objective function is given in Equation 1 of the paper:
    \$
    \mathcal { T } p r o  ( \theta ) = \mathbb { E } [ q \sim P ( Q ) , o \sim \pi \theta _ { o d d } ( O | q ) ] \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \operatorname* { m i n } \left[ \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o d d } } ( o _ { t } | q , o _ { < t } ) } A _ { t } , \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) } { \pi _ { \theta _ { o d d } } ( o _ { t } | q , o _ { < t } ) } , 1 - \varepsilon , 1 + \varepsilon \right) A _ { t } \right]
    \$
    -   $\pi_{\theta}$: The current policy (LLM) being trained.
    -   $\pi_{\theta_{old}}$: A copy of the policy from before the update, used as a stable reference.
    -   `q, o`: A question and an output (completion) sampled from the old policy.
    -   $A_t$: The **advantage** at token $t$. This measures how much better the chosen action was compared to the average action at that state. It is typically calculated using a **critic (value) model**.
    -   $\varepsilon$: A small hyperparameter (e.g., 0.2) that defines the clipping range, which prevents excessively large updates.

-   **The Problem with PPO:** The advantage $A_t$ relies on a critic model, which is usually another large neural network of a similar size to the policy model. Training and storing this critic model alongside the policy model, a reference model, and the reward model consumes a large amount of GPU memory, making the process expensive.

-   **GRPO's Innovation:** GRPO eliminates the need for a critic model. Instead of learning a value function, it estimates the baseline directly from the data. For a single question, it samples a **group** of $G$ different outputs. The reward for each output is calculated, and the **average reward of the group** is used as the baseline. The "advantage" for a given output is then its reward relative to this group average.

-   **GRPO Objective Function:** The objective function for GRPO, given in Equation 3, is a direct adaptation of the PPO objective, replacing the critic-based advantage with the group-relative advantage.
    \$
    \mathcal { J } _ { G R P O } ( \theta ) = \mathbb { E } [ q \sim P ( Q ) , \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { o d d } } ( O | q ) ] \\ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | o _ { i } | } \sum _ { t = 1 } ^ { | o _ { i } | } \left\{ \min \left[ \frac { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta _ { o d d } } ( o _ { i , t } | q , o _ { i , < t } ) } \hat { A } _ { i , t } , \mathrm{clip} \left( \frac { \pi _ { \theta } ( o _ { i , t } | q , o _ { i , < t } ) } { \pi _ { \theta _ { o d d } } ( o _ { i , t } | q , o _ { i , < t } ) } , 1 - \varepsilon , 1 + \varepsilon \right) \hat { A } _ { i , t } \right] - \beta \mathbb { D } _ { K L } [ \pi _ { \theta } || \pi _ { r e f } ] \right\}
    \$
    -   $\{o_i\}_{i=1}^G$: A group of $G$ outputs sampled from the old policy $\pi_{\theta_{old}}$ for the same question $q$.
    -   $\hat{A}_{i,t}$: The **group-relative advantage** for token $t$ of output $i$. This is calculated based on the rewards within the group, not a critic model.
    -   $\beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]$: A KL-divergence penalty term to prevent the trained policy $\pi_\theta$ from straying too far from a reference policy $\pi_{ref}$ (usually the initial SFT model). This helps maintain model quality and prevent reward hacking.

### 4.3.2. Supervision Types for GRPO

GRPO can be used with two different reward schemes:

-   **Outcome Supervision (OS):** A single reward is given at the very end of a generated solution (e.g., +1 if the final answer is correct, -1 if incorrect). This reward is normalized using the mean and standard deviation of all rewards in the group. The advantage $\hat{A}_{i,t}$ is the same for every token in a given output $o_i$.
-   **Process Supervision (PS):** A more fine-grained approach where a reward is given for each intermediate reasoning step. For example, in a math problem, each logical step can be evaluated as correct or incorrect. The advantage for a token $t$, $\hat{A}_{i,t}$, is calculated as the sum of all future (normalized) step-wise rewards from that token onwards. This provides a much stronger and more direct learning signal.

### 4.3.3. Iterative RL with GRPO

To prevent the policy model from "outgrowing" the fixed reward model, the authors employ an iterative training scheme (Algorithm 1).
1.  Sample outputs from the current policy model.
2.  Use these new outputs to generate new training data for the reward model (e.g., pairs of correct/incorrect steps).
3.  Continuously train the reward model on this new data, mixed with some historical data (a replay mechanism).
4.  Use the updated reward model to continue training the policy model with GRPO.
    This iterative loop ensures the reward model remains a useful guide for the ever-improving policy.

# 5. Experimental Setup

## 5.1. Datasets
The paper uses a comprehensive suite of benchmarks to evaluate the models' capabilities across various aspects of mathematical and general reasoning.

-   **English Mathematical Reasoning:**
    -   `GSM8K`: A dataset of grade-school math word problems.
    -   `MATH`: A challenging dataset of competition-level high school math problems covering algebra, geometry, number theory, etc. This is a key benchmark for advanced reasoning.
    -   `SAT`: Math problems from the official SAT college entrance exams.
    -   `OCW Courses`: Problems from MIT OpenCourseWare STEM courses.
    -   `MMLU-STEM`: The STEM-related subjects from the Massive Multitask Language Understanding benchmark, testing broad scientific knowledge.

-   **Chinese Mathematical Reasoning:**
    -   `MGSM-zh`: The Chinese version of GSM8K.
    -   `CMATH`: A dataset of Chinese elementary school math problems.
    -   `Gaokao-MathCloze` & `Gaokao-MathQA`: Math problems from the Chinese college entrance exam (Gaokao), which is notoriously difficult.

-   **Formal Mathematics:**
    -   `miniF2F`: A benchmark for formal theorem proving, requiring the model to generate a formal proof in the Isabelle proof assistant from an informal statement.

-   **General Reasoning and Code:**
    -   `BIG-Bench Hard (BBH)`: A subset of challenging BIG-Bench tasks that require multi-step reasoning.
    -   `HumanEval` & `MBPP`: Standard benchmarks for evaluating a model's ability to generate Python code from docstrings.

## 5.2. Evaluation Metrics

The paper uses several standard metrics to quantify model performance.

-   **Accuracy (Top-1 Accuracy):**
    1.  **Conceptual Definition:** This is the most straightforward metric. It measures the percentage of problems for which the model's single, most confident answer is correct. It is used for multiple-choice questions or problems with a unique, verifiable answer.
    2.  **Mathematical Formula:**
        \$
        \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
        \$
    3.  **Symbol Explanation:** N/A.

-   **Pass@k:**
    1.  **Conceptual Definition:** This metric is used to evaluate generative tasks like coding or complex problem-solving where there might be multiple valid solution paths. The model generates $k$ independent solutions for the same problem. The metric measures the probability that at least one of these $k$ solutions is correct. It assesses the model's ability to find a correct solution within a few attempts.
    2.  **Mathematical Formula:** An unbiased estimator for Pass@k is calculated as:
        \$
        \text{Pass}@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
        \$
    3.  **Symbol Explanation:**
        -   $n$: The total number of solutions generated for a problem (where $n \ge k$).
        -   $c$: The number of correct solutions among the $n$ generated samples.
        -   $k$: The number of solutions considered for the "pass" condition.

-   **Maj@k (Majority Voting):**
    1.  **Conceptual Definition:** This metric also involves generating $k$ solutions. However, instead of just checking if any are correct, it extracts the final answer from all $k$ solutions and takes the most frequent one (the "majority vote") as the model's final answer. The accuracy of this majority-voted answer is then calculated. This metric tests the robustness and consistency of the model's reasoning; a high Maj@k score suggests the model reliably converges on the correct answer.
    2.  **Mathematical Formula:** This is a procedural metric, but its result is an accuracy score calculated as:
        \$
        \text{Maj}@k\text{-Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{majority\_vote}(\text{solutions}_i) = \text{true\_answer}_i)}{N}
        \$
    3.  **Symbol Explanation:**
        -   $N$: Total number of problems in the test set.
        -   $\mathbb{I}(\cdot)$: An indicator function that is 1 if the condition is true, and 0 otherwise.
        -   $\text{solutions}_i$: The set of $k$ solutions generated for problem $i$.
        -   $\text{majority\_vote}(\cdot)$: A function that returns the most frequent final answer from the set of solutions.

## 5.3. Baselines
The paper compares DeepSeekMath against a wide range of strong baseline models, divided into two categories:

-   **Closed-Source Models:** These are powerful, proprietary models that are not publicly available for inspection or modification. They serve as the ultimate performance benchmark.
    -   `GPT-4` & `GPT-4 Code Interpreter`
    -   `Gemini Ultra` & `Gemini Pro`
    -   `Inflection-2`, `Grok-1`, `Baichuan-3`, `GLM-4`

-   **Open-Source Models:** These models are publicly available and represent the state-of-the-art in the open-source community.
    -   **General Models:** `DeepSeek-LLM-Chat 67B`, `Qwen 72B`, `Mistral 7B`, `SeaLLM-v2 7B`, `ChatGLM3 6B`.
    -   **Math-Enhanced Models:** `InternLM2-Math 20B`, `Math-Shepherd-Mistral 7B`, `WizardMath` series (7B & 70B), `MetaMath 70B`, `ToRA 34B`, `MAmmoTH 70B`, and `Llemma` (7B & 34B).

# 6. Results & Analysis

This section delves into the experimental results, providing strong evidence for the effectiveness of the paper's methods.

## 6.1. Core Results Analysis

### 6.1.1. Base Model Performance
The performance of the pre-trained `DeepSeekMath-Base 7B` model is a testament to the quality of the `DeepSeekMath Corpus` and the code-centric training strategy.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Size</th>
<th colspan="4">English Benchmarks</th>
<th colspan="3">Chinese Benchmarks</th>
</tr>
<tr>
<th>GSM8K</th>
<th>MATH</th>
<th>OCW</th>
<th>SAT</th>
<th>MMLU STEM</th>
<th>CMATH</th>
<th>Gaokao MathCloze</th>
<th>Gaokao MathQA</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9"><b>Closed-Source Base Model</b></td>
</tr>
<tr>
<td>Minerva</td>
<td>7B</td>
<td>16.2%</td>
<td>14.1%</td>
<td>7.7%</td>
<td>35.6%</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Minerva</td>
<td>62B</td>
<td>52.4%</td>
<td>27.6%</td>
<td>12.0%</td>
<td>-</td>
<td>53.9%</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Minerva</td>
<td>540B</td>
<td>58.8%</td>
<td>33.6%</td>
<td>17.6%</td>
<td>-</td>
<td>63.9%</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="9"><b>Open-Source Base Model</b></td>
</tr>
<tr>
<td>Mistral</td>
<td>7B</td>
<td>40.3%</td>
<td>14.3%</td>
<td>9.2%</td>
<td>71.9%</td>
<td>51.1%</td>
<td>44.9%</td>
<td>5.1%</td>
<td>23.4%</td>
</tr>
<tr>
<td>Llemma</td>
<td>7B</td>
<td>37.4%</td>
<td>18.1%</td>
<td>6.3%</td>
<td>59.4%</td>
<td>43.1%</td>
<td>43.4%</td>
<td>11.9%</td>
<td>23.6%</td>
</tr>
<tr>
<td>Llemma</td>
<td>34B</td>
<td>54.0%</td>
<td>25.3%</td>
<td>10.3%</td>
<td>71.9%</td>
<td>52.9%</td>
<td>56.1%</td>
<td>11.9%</td>
<td>26.2%</td>
</tr>
<tr>
<td><b>DeepSeekMath-Base 7B</b></td>
<td><b>7B</b></td>
<td><b>64.2%</b></td>
<td><b>36.2%</b></td>
<td><b>15.4%</b></td>
<td><b>84.4%</b></td>
<td><b>56.5%</b></td>
<td><b>71.7%</b></td>
<td><b>20.3%</b></td>
<td><b>35.3%</b></td>
</tr>
</tbody>
</table>

**Analysis:**
-   **Dominant Performance:** `DeepSeekMath-Base 7B` dramatically outperforms all other open-source base models, including `Llemma 34B`, which is nearly 5 times larger. On the difficult `MATH` benchmark, its score of 36.2% is over 10 absolute points higher than `Llemma 34B`'s 25.3%.
-   **Beating a Giant:** Most impressively, the 7B parameter `DeepSeekMath-Base` surpasses the 540B parameter `Minerva` model (36.2% vs. 33.6% on `MATH`). This strongly validates the paper's thesis that a superior data strategy and training methodology can be more important than sheer model size.
-   **Cross-Lingual Strength:** The model shows outstanding performance on Chinese benchmarks like `CMATH` and `Gaokao-MathQA`, a direct result of including high-quality multilingual data in the pre-training corpus, unlike previous English-centric efforts.

### 6.1.2. Instruct and RL Model Performance
The final models, `DeepSeekMath-Instruct` and `DeepSeekMath-RL`, set a new state-of-the-art for open-source mathematical reasoning.

The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Size</th>
<th colspan="2">English Benchmarks</th>
<th colspan="2">Chinese Benchmarks</th>
</tr>
<tr>
<th>GSM8K</th>
<th>MATH</th>
<th>MGSM-zh</th>
<th>CMATH</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6"><b>Chain-of-Thought Reasoning</b></td>
</tr>
<tr>
<td colspan="6"><i>Closed-Source Model</i></td>
</tr>
<tr>
<td>Gemini Ultra</td>
<td>-</td>
<td>94.4%</td>
<td>53.2%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>GPT-4</td>
<td>-</td>
<td>92.0%</td>
<td>52.9%</td>
<td>-</td>
<td>86.0%</td>
</tr>
<tr>
<td>... (other closed models) ...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<td colspan="6"><i>Open-Source Model</i></td>
</tr>
<tr>
<td>InternLM2-Math</td>
<td>20B</td>
<td>82.6%</td>
<td>37.7%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Qwen</td>
<td>72B</td>
<td>78.9%</td>
<td>35.2%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>WizardMath-v1.1</td>
<td>7B</td>
<td>83.2%</td>
<td>33.0%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>DeepSeek-LLM-Chat</td>
<td>67B</td>
<td>84.1%</td>
<td>32.6%</td>
<td>74.0%</td>
<td>80.3%</td>
</tr>
<tr>
<td>MetaMath</td>
<td>70B</td>
<td>82.3%</td>
<td>26.6%</td>
<td>66.4%</td>
<td>70.9%</td>
</tr>
<tr>
<td><b>DeepSeekMath-Instruct</b></td>
<td><b>7B</b></td>
<td><b>82.9%</b></td>
<td><b>46.8%</b></td>
<td><b>73.2%</b></td>
<td><b>84.6%</b></td>
</tr>
<tr>
<td><b>DeepSeekMath-RL</b></td>
<td><b>7B</b></td>
<td><b>88.2%</b></td>
<td><b>51.7%</b></td>
<td><b>79.6%</b></td>
<td><b>88.8%</b></td>
</tr>
<tr>
<td colspan="6"><b>Tool-Integrated Reasoning</b></td>
</tr>
<tr>
<td colspan="6"><i>Closed-Source Model</i></td>
</tr>
<tr>
<td>GPT-4 Code Interpreter</td>
<td>-</td>
<td>97.0%</td>
<td>69.7%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="6"><i>Open-Source Model</i></td>
</tr>
<tr>
<td>InternLM2-Math</td>
<td>20B</td>
<td>80.7%</td>
<td>54.3%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>DeepSeek-LLM-Chat</td>
<td>67B</td>
<td>86.7%</td>
<td>51.1%</td>
<td>76.4%</td>
<td>85.4%</td>
</tr>
<tr>
<td>ToRA</td>
<td>34B</td>
<td>80.7%</td>
<td>50.8%</td>
<td>41.2%</td>
<td>53.4%</td>
</tr>
<tr>
<td><b>DeepSeekMath-Instruct 7B</b></td>
<td><b>7B</b></td>
<td><b>83.7%</b></td>
<td><b>57.4%</b></td>
<td><b>72.0%</b></td>
<td><b>84.3%</b></td>
</tr>
<tr>
<td><b>DeepSeekMath-RL</b></td>
<td><b>7B</b></td>
<td><b>86.7%</b></td>
<td><b>58.8%</b></td>
<td><b>78.4%</b></td>
<td><b>87.6%</b></td>
</tr>
</tbody>
</table>

**Analysis:**
-   **New Open-Source SOTA:** `DeepSeekMath-RL 7B` achieves a groundbreaking **51.7%** on the `MATH` benchmark using Chain-of-Thought, making it the first open-source model to surpass the 50% barrier. This performance is remarkably close to `GPT-4` (52.9%) and `Gemini Ultra` (53.2%).
-   **Effectiveness of GRPO:** The jump from `DeepSeekMath-Instruct` (46.8%) to `DeepSeekMath-RL` (51.7%) on `MATH` is a significant **~5 point absolute improvement**. This demonstrates the power of the GRPO algorithm to further refine the model's reasoning abilities, even from a very strong starting point.
-   **Generalization of RL:** The RL training was only performed on CoT data from `GSM8K` and `MATH`. Despite this, performance improved across the board, including on Chinese benchmarks (`CMATH`: 84.6% -> 88.8%) and on tool-use tasks ($MATH+Python$: 57.4% -> 58.8%). This shows that the RL process enhanced the model's core reasoning capabilities in a generalizable way.

## 6.2. Ablation Studies / Parameter Analysis

The paper's "Discussion" section contains several crucial analyses that function as ablation studies.

### 6.2.1. Code Training Benefits Mathematical Reasoning
The experiments in Table 6 investigate the effect of code pre-training.

The following are the results from Table 6 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Training Setting</th>
<th colspan="3">Training Tokens</th>
<th colspan="3">w/o Tool Use</th>
<th colspan="2">w/ Tool Use</th>
</tr>
<tr>
<th>General</th>
<th>Code</th>
<th>Math</th>
<th>GSM8K</th>
<th>MATH</th>
<th>CMATH</th>
<th>GSM8K+Python</th>
<th>MATH+Python</th>
</tr>
</thead>
<tbody>
<tr>
<td>No Continual Training</td>
<td>−</td>
<td>−</td>
<td>−</td>
<td>2.9%</td>
<td>3.0%</td>
<td>12.3%</td>
<td>2.7%</td>
<td>2.3%</td>
</tr>
<tr>
<td colspan="9"><b>Two-Stage Training</b></td>
</tr>
<tr>
<td>Stage 1: General Training</td>
<td>400B</td>
<td>−</td>
<td>−</td>
<td>2.9%</td>
<td>3.2%</td>
<td>14.8%</td>
<td>3.3%</td>
<td>2.3%</td>
</tr>
<tr>
<td>Stage 2: Math Training</td>
<td></td>
<td>−</td>
<td>150B</td>
<td>19.1%</td>
<td>14.4%</td>
<td>37.2%</td>
<td>14.3%</td>
<td>6.7%</td>
</tr>
<tr>
<td>Stage 1: Code Training</td>
<td></td>
<td>400B</td>
<td></td>
<td>5.9%</td>
<td>3.6%</td>
<td>19.9%</td>
<td>12.4%</td>
<td>10.0%</td>
</tr>
<tr>
<td><b>Stage 2: Math Training</b></td>
<td></td>
<td>−</td>
<td>150B</td>
<td><b>21.9%</b></td>
<td><b>15.3%</b></td>
<td><b>39.7%</b></td>
<td><b>17.4%</b></td>
<td><b>9.4%</b></td>
</tr>
<tr>
<td colspan="9"><b>One-Stage Training</b></td>
</tr>
<tr>
<td>Math Training</td>
<td></td>
<td>−</td>
<td>150B</td>
<td>20.5%</td>
<td>13.1%</td>
<td>37.6%</td>
<td>11.4%</td>
<td>6.5%</td>
</tr>
<tr>
<td>Code & Math Mixed Training</td>
<td></td>
<td>400B</td>
<td>150B</td>
<td>17.6%</td>
<td>12.1%</td>
<td>36.3%</td>
<td>19.7%</td>
<td>13.5%</td>
</tr>
</tbody>
</table>

**Analysis:** The best results across almost all benchmarks are achieved by the two-stage process of **Code Training followed by Math Training**. This setting outperforms training on general data then math, or training on math alone. This provides strong empirical support for the claim that **code training improves a model's foundational reasoning abilities**, making subsequent math training more effective.

### 6.2.2. ArXiv Papers Seem Ineffective
In a counter-intuitive finding, the authors show that training on corpora dominated by arXiv papers (`MathPile`, `ArXiv-RedPajama`) brought no notable improvements and sometimes even degraded performance on the evaluated benchmarks compared to not doing any math training at all. This challenges the common wisdom in the field and highlights the superiority of their web-mined `DeepSeekMath Corpus`.

### 6.2.3. Analysis of Reinforcement Learning
The paper provides deep insights into why and how their RL approach works.

-   **Online vs. Offline Data (Figure 5):** The results show `Online RFT` (data sampled from the live policy) significantly outperforms `RFT` (data sampled once from the initial SFT model). This highlights the importance of **exploration**; as the model improves, it can generate more challenging and informative training data for itself.

    ![Figure 5 | Performance of the DeepSeekMath-Instruct 1.3B model, which was further trained using various methods, on two benchmarks.](images/5.jpg)
    *该图像是图表，展示了DeepSeekMath-Instruct 1.3B模型通过不同训练方法在GSM8K和MATH两个基准测试上的准确率随训练步骤变化的曲线。图中对比了RFT、Online RFT、GRPO+OS和GRPO+PS四种方法的表现，GRPO+PS在两个基准上均表现最佳。*

-   **Gradient Coefficient Matters (Figure 5):** $GRPO+OS$ (Outcome Supervision) outperforms `Online RFT`. Both use online data and binary rewards, but GRPO uses a model-based, scaled gradient coefficient that can penalize wrong answers and differentially weight correct ones, while RFT only reinforces correct answers with a uniform weight. This shows that a more nuanced update mechanism is superior. Furthermore, $GRPO+PS$ (Process Supervision) is the best, confirming that fine-grained, step-by-step rewards provide the most effective learning signal.

-   **Why RL Works (Figure 7):** The analysis of Pass@K vs. Maj@K provides a crucial insight.

    ![Figure 7 | The Maj $@ \\mathrm { K }$ and Pass $@ \\mathrm { K }$ of SFT and RL DeepSeekMath 7B on GSM8K and MATH (temperature 0.7). It was noted that RL enhances Maj $@ \\mathrm { K }$ but not Pass@K.](images/7.jpg)
    *该图像是图表，展示了DeepSeekMath 7B模型在GSM8K和MATH数据集上，使用SFT和RL训练方法时，Maj@K和Pass@K准确率随候选数量K变化的趋势。图中说明RL提升了Maj@K但对Pass@K作用不明显。*

The chart shows that after RL, the `Maj@K` score (robustness) increases significantly, while the `Pass@K` score (raw capability to find a correct answer) stays roughly the same. This suggests that RL doesn't necessarily teach the model new fundamental reasoning skills. Instead, it **reduces the model's uncertainty and aligns its probability distribution**, making it more confident and reliable in producing the correct answers it already "knows." It learns to favor correct reasoning paths over incorrect ones it might have explored during SFT.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper presents `DeepSeekMath`, a series of open-source language models that significantly advance the state-of-the-art in mathematical reasoning. The 7B model achieves performance on the challenging `MATH` benchmark that is competitive with top-tier closed-source models like GPT-4. This success is attributed to two primary contributions: (1) the creation of the `DeepSeekMath Corpus`, a massive 120B-token dataset of high-quality math content mined from the web, and (2) the development of **Group Relative Policy Optimization (GRPO)**, a memory-efficient and effective reinforcement learning algorithm. The work provides compelling evidence that large-scale, carefully curated web data is a highly valuable resource and that starting from a code-trained model boosts mathematical reasoning. The analysis of the RL process offers the valuable insight that its primary benefit may be in increasing the model's robustness rather than its raw capability.

## 7.2. Limitations & Future Work
The authors candidly acknowledge several limitations and outline future research directions:

-   **Limitations:**
    -   **Domain Gaps:** The model's performance on geometry and formal theorem proving, while strong, still lags behind the best closed models, possibly due to data selection bias in the training corpus.
    -   **Scale-Dependent Capabilities:** Unlike GPT-4, DeepSeekMath does not show significant performance gains from few-shot prompting compared to zero-shot, a limitation likely attributable to its smaller model scale (7B).
-   **Future Work:**
    -   **Data Pipeline Enhancement:** They plan to further refine their data collection pipeline to build even larger and more diverse high-quality corpora, potentially addressing the gaps in geometry and other areas.
    -   **More Effective RL:** The paper lays out a roadmap for improving RL based on their unified paradigm:
        1.  **Data Source:** Explore using out-of-distribution prompts and more advanced decoding strategies (e.g., tree-of-thought) to generate more challenging and diverse data for RL training.
        2.  **Algorithms:** Develop RL algorithms that are robust to noisy reward signals, moving towards a "weak-to-strong" generalization paradigm where a powerful model can be trained even with an imperfect reward model.
        3.  **Reward Function:** Improve the generalization and uncertainty-awareness of reward models and build higher-quality process-level reward models.

## 7.3. Personal Insights & Critique
-   **Inspirations:**
    -   **The Power of Open Data:** This paper is a powerful testament to the idea that with clever engineering and a solid methodology, publicly available data can be leveraged to create models that rival those built on proprietary datasets. The strategic pivot away from the well-trodden path of using arXiv to mining the "messy" but vast Common Crawl is a major takeaway.
    -   **Pragmatic Algorithm Design:** GRPO is a brilliant piece of pragmatic engineering. It identifies a key bottleneck in a theoretically powerful algorithm (PPO's memory usage) and proposes a simple, intuitive, and effective solution. This approach of tailoring algorithms to practical resource constraints is highly valuable.
    -   **Synergy of Code and Math:** The paper provides some of the strongest empirical evidence for the synergy between coding and mathematical reasoning. This insight can guide the development of future foundation models, suggesting that a strong grounding in the structured, logical language of code is a prerequisite for high-level abstract reasoning.

-   **Critique and Potential Issues:**
    -   **The arXiv Finding:** The conclusion that arXiv data is "ineffective" is provocative and well-supported by their experiments. However, this finding should be interpreted with caution. It's possible that the benefits of formal academic text are more subtle, domain-specific (e.g., for cutting-edge theoretical problems not covered in benchmarks), or only become apparent at a much larger model scale. It doesn't necessarily mean arXiv data is useless, but rather that web data provided a better return on investment for the specific benchmarks tested.
    -   **The Nature of RL's Improvement:** The insight that RL improves `Maj@K` (robustness) but not `Pass@K` (capability) is fascinating. However, it raises a deeper question: is this an inherent property of current RLHF techniques, or a limitation of the reward model and data used? If RL is primarily a tool for "confidence boosting," then achieving the next leap in fundamental reasoning might require entirely new paradigms beyond preference-based alignment.
    -   **Generalizability of GRPO:** GRPO works exceptionally well for mathematics, where solutions often have a verifiably correct answer, making reward calculation straightforward. Its applicability to more subjective domains (e.g., creative writing, summarization) where rewards are fuzzier is an open question. The "group average" baseline might be less meaningful when preferences are highly diverse and subjective.