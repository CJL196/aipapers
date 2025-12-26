# 1. Bibliographic Information

## 1.1. Title
VideoAgent: Long-form Video Understanding with Large Language Model as Agent

## 1.2. Authors
Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy from Stanford University.

## 1.3. Journal/Conference
This paper was published on arXiv, a preprint server for scientific papers. While arXiv itself is not a peer-reviewed journal or conference, it is a widely recognized platform for disseminating cutting-edge research in fields like computer science, and papers published here often undergo peer review for later publication in prestigious conferences or journals. The publication date is 2024-03-15.

## 1.4. Publication Year
2024

## 1.5. Abstract
The paper addresses the significant challenge of long-form video understanding in computer vision, which requires models to reason over lengthy multi-modal sequences. Inspired by human cognitive processes, the authors propose an agent-based system called VideoAgent. This system employs a large language model (LLM) as a central agent that iteratively identifies and compiles crucial information to answer a question. Vision-language foundation models (VLMs) and contrastive language-image models (CLIP) serve as tools to translate and retrieve visual information. VideoAgent achieves zero-shot accuracy of 54.1% on EgoSchema and 71.3% on NExT-QA benchmarks, using only 8.4 and 8.2 frames on average, respectively. These results demonstrate superior effectiveness and efficiency compared to current state-of-the-art methods, highlighting the potential of agent-based approaches for advancing long-form video understanding.

## 1.6. Original Source Link
https://arxiv.org/abs/2403.10517
PDF Link: https://arxiv.org/pdf/2403.10517v1.pdf
This paper is currently available as a preprint on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the **long-form video understanding** task. This task is particularly challenging in computer vision for several reasons:
1.  **Multi-modal information processing:** Videos contain both visual (frames) and temporal (sequence of events) data, requiring models to integrate information from different modalities.
2.  **Exceedingly long sequences:** Long-form videos can range from minutes to hours, posing a significant challenge for models to process and retain information over such extended durations.
3.  **Effective reasoning:** Beyond mere processing, models need to reason about events, causality, and relationships within these long sequences to answer complex questions.

    Existing models struggle to simultaneously excel in all three areas. Current **Large Language Models (LLMs)** are proficient in reasoning and handling long text contexts but lack intrinsic visual understanding. Conversely, **Visual Language Models (VLMs)** can process visual information but often struggle with modeling lengthy visual inputs efficiently or effectively. Early attempts to adapt VLMs for long contexts have shown underperformance and inefficiency in video understanding benchmarks.

The paper's entry point or innovative idea is motivated by the **human cognitive process** for long-form video understanding. Humans don't typically process every single frame of a long video. Instead, they:
*   First, get a quick overview.
*   Then, iteratively select relevant information guided by a specific question.
*   Finally, compile the information and provide an answer, concluding the process when sufficient information is gathered.
    This human-inspired approach suggests that **interactive reasoning and planning** are more critical than the ability to directly process excessively long visual inputs.

## 2.2. Main Contributions / Findings
The paper introduces `VideoAgent`, a novel agent-based system designed to mimic human cognitive processes for long-form video understanding. Its primary contributions and findings are:

*   **Novel Agent-Based System (`VideoAgent`):** Proposes a system where a Large Language Model (LLM) acts as a central agent, controlling an iterative process of information gathering and reasoning. Vision-Language Models (VLMs) and Contrastive Language-Image Models (CLIP) serve as tools to translate visual content into language and retrieve relevant frames, respectively. This shifts the focus from direct processing of long visual inputs to iterative, query-driven reasoning.
*   **Emphasis on Interactive Reasoning and Planning:** The core design principle is to simulate human-like iterative information seeking. The LLM agent iteratively assesses the current state of knowledge, identifies missing information, retrieves new relevant visual data, and updates its understanding until a question can be answered confidently.
*   **Superior Effectiveness and Efficiency:**
    *   `VideoAgent` achieves state-of-the-art (SOTA) zero-shot accuracy on challenging benchmarks: 54.1% on EgoSchema and 71.3% on NExT-QA.
    *   Crucially, it demonstrates exceptional efficiency, utilizing only 8.4 frames on average for EgoSchema and 8.2 frames for NExT-QA. This is a significant reduction (e.g., 20x fewer frames compared to previous SOTA like LLoVi), highlighting that intelligent selection of information is more impactful than processing vast amounts of data.
*   **Adaptive Frame Selection:** The iterative frame selection process is adaptive, searching and aggregating relevant information based on the complexity of the video and the question. This allows it to use fewer frames for simpler questions and more for complex ones, outperforming uniform sampling methods.
*   **Robustness to Long Videos:** Case studies demonstrate `VideoAgent`'s ability to generalize to arbitrarily long videos, including hour-long content, by accurately identifying key information.
*   **Significance of Self-Reflection and Segment Selection:** Ablation studies confirm the importance of the LLM's self-reflection mechanism for terminating iterations and the `segment-level retrieval` strategy for enhancing temporal reasoning and mitigating irrelevant information.

    These findings collectively solve the problem of efficiently and effectively understanding long-form videos by leveraging advanced LLM reasoning capabilities coupled with targeted visual information retrieval, establishing a new benchmark in the field.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

To fully grasp the `VideoAgent` paper, a beginner should understand several key concepts:

*   **Long-form Video Understanding:** This refers to the task of comprehending the content, events, and relationships within videos that are significantly longer than short clips, typically ranging from several minutes to hours. This involves tracking objects, understanding actions, inferring causality, and answering complex questions that often require integrating information across extended temporal durations. It's challenging due to the sheer volume of data (millions of frames) and the complex temporal dependencies.

*   **Large Language Models (LLMs):** These are advanced artificial intelligence models trained on vast amounts of text data, enabling them to understand, generate, and process human language. Key capabilities relevant to this paper include:
    *   **Reasoning:** LLMs can perform logical deductions, infer information, and connect disparate pieces of text to arrive at conclusions.
    *   **Planning:** They can formulate steps or strategies to achieve a goal, which is crucial for the `agent` functionality.
    *   **Tool-use:** Modern LLMs can be instructed to use external tools (like search engines, calculators, or, in this paper's case, vision models) by generating appropriate queries or commands.
    *   **Context Window:** The amount of text (tokens) an LLM can process or "remember" at one time. For long videos, direct textualization of all frames would exceed typical LLM context windows, motivating approaches like `VideoAgent`. Examples include GPT-4, LLaMA, Mixtral.

*   **Vision-Language Models (VLMs):** These are neural networks designed to process and understand both visual (images/videos) and textual data, often by mapping them into a shared embedding space. Their primary function in this paper is:
    *   **Image/Video Captioning:** Generating textual descriptions of visual content. For example, given an image, a VLM can output "A cat sitting on a mat." This is critical for translating visual observations into a format an LLM can understand. Examples include BLIP-2, CogAgent, LaViLa.

*   **Contrastive Language-Image Pre-training (CLIP):** A specific type of VLM developed by OpenAI that learns to associate images with their corresponding text descriptions through a contrastive learning objective.
    *   **Image-Text Matching/Retrieval:** CLIP can embed images and text into a shared multi-modal embedding space. This allows it to measure the semantic similarity between an image and a text query. For instance, given a text query "a red car" and a set of images, CLIP can identify the image most semantically similar to that query.
    *   **Computational Efficiency:** CLIP's architecture (specifically its late-interaction design) allows image features to be computed once and then efficiently compared against multiple text queries without re-processing the image, making it suitable for retrieval tasks in `VideoAgent`. Examples include OpenCLIP, EVA-CLIP.

*   **Agent-based Systems (LLM Agents):** In artificial intelligence, an agent is an entity that perceives its environment, makes decisions, and takes actions to achieve specific goals. An `LLM Agent` leverages an LLM's reasoning and planning capabilities to control this decision-making loop. In this context, the LLM acts as the "brain" that orchestrates interactions with various "tools" (like VLMs and CLIP) to navigate an environment (the video) and accomplish a task (answer a question). Key components include:
    *   **State:** The current understanding or information gathered so far.
    *   **Action:** A decision made by the agent (e.g., answer the question, retrieve more information).
    *   **Observation:** New information gathered from the environment after an action.
    *   **Iterative Process:** The agent repeatedly cycles through sensing, thinking, and acting.

*   **Zero-shot Accuracy:** A metric used to evaluate a model's performance on tasks or categories it has not been explicitly trained on. In the context of `VideoAgent`, it means the model is evaluated on video understanding benchmarks without having seen any examples from those specific datasets during its training phase, relying solely on its pre-trained knowledge and general understanding.

## 3.2. Previous Works

The paper contextualizes `VideoAgent` by discussing prior approaches to long-form video understanding and the emerging field of LLM agents.

**Long-form Video Understanding:**
Previous methods generally fall into two categories:

*   **Compressive Sparsity Methods:** These approaches aim to reduce the high dimensionality of long videos by compressing them into more manageable, meaningful embeddings or representations. The goal is to retain essential information while discarding redundancy.
    *   **Examples:**
        *   `MovieChat [42]` uses a memory consolidation mechanism to merge similar adjacent frame tokens based on cosine similarity.
        *   `Chat-UniVi [14]` employs kNN clustering for spatio-temporal compression of video tokens.
        *   Other methods compress into space-time graphs (`[10, 50, 59]`) or even text.
        *   `LLoVi [67]` (a strong baseline for comparison) captions entire videos and then prompts an LLM with these captions. This highlights the effectiveness of text-based representations but often involves processing many frames.

*   **Selective-Compressive Methods:** These methods attempt to sub-sample the video, selecting only the most relevant frames, often guided by the input question or text query.
    *   **Examples:**
        *   `R-VLM` and `R2A [8, 33, 56]` utilize a CLIP model to retrieve relevant frames given a text prompt.
        *   `Q-ViD [38]` uses the question to selectively caption parts of the video.
    *   **Differentiation from `VideoAgent`:** While these methods also select frames, they often do so in a single iteration based on the original question. `VideoAgent` distinguishes itself by allowing the LLM to *direct* the frame sampling in a multi-round, iterative fashion, and to *rewrite* queries for more accurate, fine-grained retrieval based on intermediate reasoning.

**LLM Agents:**
The success of LLMs in reasoning and planning has led to their use as agents in various domains.
*   **General LLM Agents:** Examples include online search, card games, database management (`[25, 26, 61]`). Techniques like `chain-of-thought reasoning [52]` and `self-reflection [41]` amplify their effectiveness.
    *   **Chain-of-Thought (CoT) Prompting:** This technique encourages LLMs to explain their reasoning process step-by-step, which helps in complex problem-solving and often leads to more accurate answers. It involves prompting the model to "think step-by-step."
    *   **Self-Reflection:** An agent's ability to critically evaluate its own actions, reasoning, or generated outputs and identify errors or areas for improvement. This helps in refining the decision-making process.
*   **LLM Agents in Visual Contexts:** Initial explorations include GUI understanding and robot navigation (`[3, 5, 9, 45]`).
    *   **Differentiation from `VideoAgent`:** Some studies (`[6, 45, 60]`) use LLMs to interact with external tools or add functionalities in video understanding. `VideoAgent` explicitly reformulates video understanding as a human-inspired decision-making process within a dynamic environment (the video), where the agent iteratively decides to seek more information or conclude.

## 3.3. Technological Evolution

The field of video understanding has evolved significantly:
1.  **Early Approaches (Pre-Deep Learning):** Focused on hand-crafted features, motion detection, and traditional machine learning classifiers. Limited by feature engineering complexity and scalability.
2.  **Deep Learning (Convolutional Neural Networks - CNNs, Recurrent Neural Networks - RNNs):** Introduced end-to-end learning for video tasks. CNNs for spatial features, RNNs/LSTMs for temporal dependencies. However, processing long videos remained challenging due to computational cost and vanishing gradients.
3.  **Transformer Models:** Revolutionized sequence modeling in NLP and vision. `Vision Transformers (ViT)` adapted transformers for images, and `Video Transformers` extended them to videos, often by treating frames as sequences of patches. This brought improved long-range dependency modeling.
4.  **Foundation Models (VLMs, LLMs, CLIP):** The rise of large pre-trained models.
    *   **CLIP-like Models:** Enabled powerful zero-shot capabilities by aligning visual and text embeddings.
    *   **VLMs:** Integrated visual perception with language understanding and generation, leading to tasks like image captioning and visual question answering.
    *   **LLMs:** Demonstrated unprecedented reasoning and planning abilities.
5.  **LLM Agents:** The current frontier, where LLMs act as intelligent controllers, leveraging other foundation models as tools. This paradigm shifts from monolithic models trying to do everything to orchestrators of specialized tools.

    `VideoAgent` fits within this timeline as a cutting-edge example of LLM Agents specifically applied to the complex domain of long-form video understanding. It represents a move towards more intelligent, interactive, and efficient approaches by simulating human-like cognitive processes.

## 3.4. Differentiation Analysis

Compared to the main methods in related work, `VideoAgent` presents several core differences and innovations:

*   **Iterative, Multi-Round Frame Selection vs. Single-Pass/Uniform Sampling:**
    *   **Previous methods (`[16, 56, 66]`):** Often rely on uniformly sampling a fixed number of frames or selecting frames in a single initial pass. This can lead to either too much irrelevant information (if many frames are sampled) or insufficient information (if too few).
    *   **`VideoAgent`:** Employs a **multi-round iterative process**. The LLM agent dynamically decides whether more information is needed and then specifically requests it. This ensures that information gathering is precise and adaptive, mimicking how humans investigate a video.

*   **Dynamic, Rewritten Queries vs. Static Original Question Queries:**
    *   **Previous methods (`[56, 66]`):** Typically use the original question directly as the query for frame retrieval.
    *   **`VideoAgent`:** Allows the LLM agent to **rewrite and refine the query** for frame retrieval based on its current understanding and identified information gaps. This enables more accurate, fine-grained, and context-aware retrieval, essential for complex temporal or causal questions.

*   **Emphasis on Reasoning and Planning as an Agent vs. Direct Processing:**
    *   **Previous methods (e.g., `LLoVi [67]`):** While using LLMs, they often involve feeding a comprehensive set of video captions (derived from many frames) to the LLM in a more passive manner. The LLM's role is primarily to reason over the provided text.
    *   **`VideoAgent`:** Positions the LLM as an **active agent** responsible for orchestrating the entire understanding process. It performs `self-reflection` to determine confidence, `plans` what information is missing, and `acts` by using tools (CLIP, VLM) to obtain `observations`. This human-inspired decision-making loop is central to its design.

*   **Adaptive Efficiency:**
    *   By leveraging iterative selection and refined queries, `VideoAgent` achieves significantly higher efficiency (e.g., 20x fewer frames than `LLoVi`) while maintaining or surpassing state-of-the-art accuracy. This adaptive nature means it uses only the necessary amount of visual data, contrasting with methods that might process a fixed, often large, number of frames regardless of question complexity.

        In essence, `VideoAgent` moves beyond simply integrating foundation models as components in a pipeline; it frames the entire video understanding task as an intelligent, interactive problem-solving process guided by an LLM agent, leading to more human-like, effective, and efficient performance.

# 4. Methodology

The `VideoAgent` system is designed to simulate the human cognitive process for understanding long-form videos. It formulates the video understanding task as a sequence of states, actions, and observations, controlled by a large language model (LLM) acting as the central agent.

## 4.1. Principles

The core idea of `VideoAgent` is based on how humans approach understanding a long video:
1.  **Initial Glance:** Humans first get a general sense of the video's content by quickly looking at a few parts.
2.  **Iterative Information Seeking:** Guided by a specific question, they then selectively search for more detailed information in relevant sections of the video.
3.  **Aggregation and Decision:** Once they believe they have enough information, they synthesize it to form an answer, otherwise, they continue searching.

    `VideoAgent` translates this into an agent-based system where the LLM is the "brain" that orchestrates this iterative process. The LLM's capabilities in memory, reasoning, planning, and tool-use are leveraged to model the `states` (current information), `actions` (decisions to answer or seek more info), and `observations` (new frames/captions) within this loop.

## 4.2. Core Methodology In-depth (Layer by Layer)

The video understanding process in `VideoAgent` is modeled as a sequence of states, actions, and observations: $\{ ( s _ { t } , a _ { t } , o _ { t } ) | 1 \leq t \leq T \}$, where:
*   $s_t$: The current state, representing all the information gathered from previously seen frames up to iteration $t$.
*   $a_t$: The action taken at iteration $t$, which is either to answer the question or to continue searching for new information.
*   $o_t$: The observation received at iteration $t$, which consists of new frames retrieved in the current iteration.
*   $T$: The maximum number of iterations allowed.

    The `VideoAgent` leverages GPT-4 as the central LLM agent, with Visual Language Models (VLMs) and Contrastive Language-Image Models (CLIP) serving as instrumental tools. The overall process is summarized in Algorithm 1 and visualized in Figure 1.

### Algorithm 1 VideoAgent

The complete algorithmic flow of `VideoAgent` is as follows:

$$
Algorithm 1 VideoAgent

Require: Video $v$, question $q$, LLM $F_l$, VLM $F_v$, CLIP $F_c$, max iteration $T$, confidence threshold $C$
Ensure: Prediction $\hat{y}$, state-action-observation sequence $\{s_t, a_t, o_t | 1 \leq t \leq T\}$

1: $s_1 \gets$ GenerateCaptions($F_v$, UniformSample($v$))
2: for $t = 1$ to $T$ do
3:    $\hat{y} \gets$ PredictAnswer($F_l$, $s_t$, $q$)
4:    $c \gets$ SelfReflect($F_l$, $s_t$, $q$, $\hat{y}$)
5:    if $a_t \gets \mathbb{1}_{[c \geq C]}$ then
6:        break
7:    else
8:        $h \gets$ FindMissingInfo($F_l$, $s_t$, $q$)
9:        $o_t \gets$ RetrieveFrames($F_c$, $v$, $h$)
10:       $s_{t+1} \gets$ Merge($s_t$, GenerateCaptions($F_v$, $o_t$))
11:   end if
12: end for
13: return $\hat{y}$, $\{s_t, a_t, o_t | 1 \leq t \leq T\}$
$$

Let's break down each step in detail:

#### 4.2.1. Obtaining the Initial State ($s_1$)

The first step familiarizes the LLM with the overall context of the video, akin to a human's initial glance.

*   **Process:** $N$ frames are uniformly sampled from the entire video $v$. This means frames are picked at regular intervals across the video's duration.
*   **Tool Usage:** A Vision-Language Model (VLM), denoted as $F_v$, is used to generate textual descriptions (captions) for each of these $N$ sampled frames. The prompt used for captioning is "describe the image in detail."
*   **Outcome:** These generated captions form the `initial state`, $s_1$, which is then fed to the LLM. This $s_1$ provides a textual sketch of the video's content and semantics, allowing the LLM (which is text-only) to begin its reasoning process.

#### 4.2.2. Determining the Next Action ($a_t$)

In each iteration $t$, the LLM agent, using its current understanding from state $s_t$ and the question $q$, decides its next action $a_t$. There are two possible actions:
1.  **Action 1: Answer the question.** If the information in $s_t$ is deemed sufficient to confidently answer $q$, the process terminates, and the LLM provides its prediction $\hat{y}$.
2.  **Action 2: Search new information.** If $s_t$ is insufficient, the LLM determines what additional information is required and continues the search.

    This decision-making process is critical and is achieved through a three-step mechanism:

*   **Step 1: Predict Answer ($\hat{y}$):** The LLM ($F_l$) is prompted to make a prediction $\hat{y}$ based on the current state $s_t$ and question $q$. This step often involves `chain-of-thought prompting`, where the LLM is encouraged to outline its reasoning process. The function is represented as `PredictAnswer(`F_l`,`s_t`,`q`)`.
*   **Step 2: Self-Reflect ($c$):** The LLM then performs a `self-reflection` step. It critically assesses its prediction $\hat{y}$, its reasoning process (if generated via chain-of-thought), the current state $s_t$, and the question $q$. Based on this assessment, it generates a `confidence score` $c$. The confidence score has three predefined levels:
    *   **1 (Insufficient Information):** The LLM believes it lacks crucial data to form a reliable answer.
    *   **2 (Partial Information):** The LLM has some relevant data but needs more to be fully confident.
    *   **3 (Sufficient Information):** The LLM is confident that $s_t$ contains enough information to answer $q$.
        This step is represented as `SelfReflect(`F_l`,`s_t`,`q`,`\hat{y}`)`. The self-reflection process, as illustrated in Figure 2, is crucial because direct prediction alone often defaults to seeking more information, while self-reflection provides a more nuanced assessment.

    The following figure (Figure 2 from the original paper) shows the detailed view of VideoAgent's iterative process:

    ![Fig. 2: Detailed view of VideoAgent's iterative process. Each round starts with the state, which includes previously viewed video frames. The large language model then determines subsequent actions by answering prediction and self-reflection. If additional information is needed, new observations are acquired in the form of video frames.](images/2.jpg)
    *Fig. 2: Detailed view of VideoAgent's iterative process. Each round starts with the state, which includes previously viewed video frames. The large language model then determines subsequent actions by answering prediction and self-reflection. If additional information is needed, new observations are acquired in the form of video frames.*

*   **Step 3: Choose Action ($a_t$):** The action $a_t$ is determined by comparing the generated confidence score $c$ against a predefined `confidence threshold` $C$.
    *   If $c \geq C$: The LLM decides the information is sufficient, takes Action 1 (answer the question), and the loop `break`s. This is represented by $a_t \gets \mathbb{1}_{[c \geq C]}$, where $\mathbb{1}$ is the indicator function.
    *   If $c < C$: The LLM decides more information is needed, takes Action 2 (search new information), and proceeds to the next phase.

#### 4.2.3. Gathering a New Observation ($o_t$)

If the LLM decides to search for new information (Action 2), it first needs to specify *what* information is missing and *where* to find it.

*   **Step 1: Find Missing Information ($h$):** The LLM ($F_l$) is prompted to identify the specific pieces of information $h$ that are currently lacking in $s_t$ but are necessary to answer $q$. This step also involves formulating specific text queries that can be used to retrieve this missing information. This is represented as `FindMissingInfo(`F_l`,`s_t`,`q`)`.

*   **Step 2: Segment-Level Retrieval:** To enhance temporal reasoning and prevent retrieving irrelevant information (e.g., a toy on a sofa before a boy leaves the room, when the question asks what's left *after* he leaves), the video $v$ is first divided into segments based on the indices of the frames already seen. The LLM then predicts which specific segments to search within using its generated queries $h$.
*   **Tool Usage:** A Contrastive Language-Image Model (CLIP), denoted as $F_c$, is employed to retrieve the most relevant frames. For each query text in $h$ and specified segment, CLIP returns the image frame from that segment that has the highest `cosine similarity` with the query text. These retrieved frames constitute the `new observation` $o_t$. This is represented as `RetrieveFrames(`F_c`,`v`,`h`)`.

    **Computational Efficiency of CLIP:** The use of CLIP for retrieval is highly efficient due to several factors:
    *   **Single Feed-Forward Process:** CLIP's feature computation is a quick, one-pass operation.
    *   **Image-Text Late Interaction:** CLIP's architecture allows image features to be pre-computed, cached, and reused. When a new text query arrives, only the text feature needs to be computed, and a fast dot product (cosine similarity) can be performed with the pre-computed image features. This avoids re-processing entire images for each new query.
    *   **Segment-Level Design:** By searching only within specific segments, the number of frames for which features need to be considered is reduced, further optimizing retrieval.
        The paper notes that CLIP computations typically account for less than 1% of the total computational time compared to VLMs and LLMs.

    From the Appendix A, the proportion of time dedicated to computing CLIP features relative to the overall computation time is approximated by:
    \$
    \frac{N \cdot x + n \cdot x}{N \cdot x + n \cdot x + n \cdot y + t \cdot z}
    \$
    where:
    *   $N$: Total number of frames in the video.
    *   $n$: Number of frames selectively processed by `VideoAgent` across $t$ rounds.
    *   $x$: Time required for CLIP to compute features per image and text.
    *   $y$: Time required for VLM captioning per image.
    *   $z$: Time required for LLM computation per round.
    *   $t$: Number of iterations/rounds.

        In practice, using OpenCLIP ViT-G (as CLIP), CogAgent (as VLM), GPT-4 (as LLM), with an A6000 GPU and the EgoSchema dataset, the values are approximately:
    *   $N = 180$ (total frames)
    *   $n = 8.4$ (average frames used)
    *   $x = 0.02$ seconds (CLIP feature computation per image/text)
    *   $y = 20$ seconds (VLM captioning per image)
    *   $z = 10$ seconds (LLM computation per round)
    *   $t = 3$ (average rounds)

        Plugging these values into the formula:
    \$
    \frac{180 \times 0.02 + 8.4 \times 0.02}{180 \times 0.02 + 8.4 \times 0.02 + 8.4 \times 20 + 3 \times 10} = \frac{3.6 + 0.168}{3.6 + 0.168 + 168 + 30} = \frac{3.768}{201.768} \approx 0.01867
    \$
    This evaluates to approximately **1.9%**, confirming that CLIP feature computation is a small fraction of the total computational effort.

#### 4.2.4. Updating the Current State ($s_{t+1}$)

After new observations $o_t$ (retrieved frames) are obtained, they need to be integrated into the LLM's understanding.

*   **Tool Usage:** The VLM ($F_v$) generates captions for each of the newly retrieved frames in $o_t$. This is represented as `GenerateCaptions(`F_v`,`o_t`)`.
*   **Merge and Update:** The new captions are then sorted by their respective frame indices and concatenated with the captions from all previously seen frames (which formed $s_t$). This merged and updated set of captions forms the `new state`, $s_{t+1}$. This is represented as `Merge(`s_t`, GenerateCaptions(`F_v`,`o_t`))`.
*   **Next Round:** This $s_{t+1}$ is then fed to the LLM for the next iteration of decision-making (determining the next action $a_{t+1}$).

    The multi-round iterative process offers significant advantages over single-step baselines:
*   **Reduced Noise and Information Overload:** Uniformly sampling too many frames can introduce extensive irrelevant information and noise, which can degrade LLM performance due to `long contexts` and `distractions` (`[24, 40]`). `VideoAgent` avoids this by only retrieving what's needed.
*   **Computational Efficiency:** Processing a vast number of frames in a single pass is computationally expensive and hits LLM `context length limits`, especially for hour-long videos (`[31]`). `VideoAgent`'s adaptive selection is more efficient.
*   **Adaptive Information Gathering:** It dynamically selects the most relevant information based on the question's difficulty and current understanding, providing a flexible and cost-effective approach.

    This comprehensive, iterative process allows `VideoAgent` to efficiently and effectively navigate long-form videos by intelligently focusing its attention, much like a human investigator would.

# 5. Experimental Setup

This section details the datasets, evaluation metrics, and implementation specifics used to evaluate `VideoAgent`.

## 5.1. Datasets

The experiments utilized two distinct, well-established benchmarks for long-form video understanding, focusing on zero-shot capabilities.

### 5.1.1. EgoSchema

*   **Source:** `[28]`
*   **Characteristics:** This dataset is designed for long-form video understanding, featuring a total of 5,000 multiple-choice questions. Each question is associated with one of 5,000 egocentric videos.
    *   **Egocentric Videos:** The videos are recorded from a first-person perspective, showing human activities as if seen through the eyes of the person performing them.
    *   **Video Length:** Each video lasts approximately 3 minutes, which is considered "long-form" in this context.
    *   **Task:** Answering multiple-choice questions about the activities and events depicted in the egocentric videos.
*   **Availability:** The dataset comprises only a test set. A subset of 500 questions has publicly available labels for research, while the full set of questions is evaluated solely on the official leaderboard.
*   **Why Chosen:** `EgoSchema` is a challenging benchmark specifically designed to test models' ability to understand extended, first-person visual narratives, which often require deep reasoning about human intentions and actions.

### 5.1.2. NExT-QA

*   **Source:** `[55]`
*   **Characteristics:** `NExT-QA` is another significant dataset for video question answering, containing 5,440 natural videos that primarily depict object interactions in daily life. It is accompanied by 48,000 multiple-choice questions.
    *   **Video Length:** The average length of videos is 44 seconds. While shorter than EgoSchema, it still presents challenges in temporal reasoning.
    *   **Question Categories:** Questions are categorized into three types, providing a comprehensive evaluation of video understanding:
        *   **Temporal:** Questions requiring understanding the order or duration of events (e.g., "What happened before X?").
        *   **Causal:** Questions asking about cause-and-effect relationships (e.g., "Why did X happen?").
        *   **Descriptive:** Questions asking for direct descriptions of objects, actions, or scenes (e.g., "What is the person doing?").
*   **Evaluation Focus:** The zero-shot evaluation focused on the validation set, which includes 570 videos and 5,000 multiple-choice questions.
*   **ATP-hard Subset:** The paper additionally reports performance on the `ATP-hard subset` of the `NExT-QA` validation set. This subset is curated to include only the hardest QA pairs that cannot be solved by looking at a single frame, thereby specifically testing long-term temporal reasoning capabilities.
*   **Why Chosen:** `NExT-QA` offers a diverse set of questions that probe different aspects of video understanding, particularly temporal and causal reasoning, making it suitable for evaluating the depth of a model's comprehension. The `ATP-hard` subset further pushes the boundaries of temporal reasoning.

## 5.2. Evaluation Metrics

For both `EgoSchema` and `NExT-QA`, since they feature multiple-choice questions, the primary evaluation metric used is **Accuracy**.

### 5.2.1. Accuracy

*   **Conceptual Definition:** Accuracy is a straightforward and widely used metric for classification tasks. It measures the proportion of total predictions that were correct. In the context of multiple-choice questions, it quantifies how many questions the model answered correctly out of the total number of questions. It provides a general sense of the model's overall performance.

*   **Mathematical Formula:**
    \$
    \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
    \$

*   **Symbol Explanation:**
    *   $\text{Number of Correct Predictions}$: The count of instances where the model's predicted answer matches the true answer.
    *   $\text{Total Number of Predictions}$: The total count of questions or instances for which the model made a prediction.

## 5.3. Baselines

`VideoAgent` was compared against a range of existing state-of-the-art models for long-form video understanding, including both public and proprietary models. These baselines are representative of different approaches to the problem.

**Public Models for EgoSchema:**
*   `FrozenBiLM [58]`
*   `InternVideo [51]`
*   `ImageViT [34]`
*   `ShortViViTloc [34]`
*   `LongViViT [34]`
*   `SeViLA [66]`
*   `Vamos [49]`
*   `LLoVi [67]` (a key concurrent SOTA method)
*   `MC-ViT-L [2]`

**Large-scale Proprietary Models for EgoSchema:**
*   `Random Chance` (a basic lower bound)
*   `Bard only (blind) [2]`
*   `Bard + ImageViT [34]`
*   `Bard + ShortViViT [34]`
*   `Bard + PALI [34]`
*   `GPT-4 Turbo (blind) [2]`
*   `GPT-4V [2]`
*   `Gemini 1.0 Pro [47]`

**Models for NExT-QA (Supervised and Zero-shot):**
*   **Supervised Methods:**
    *   `VFC [57]`
    *   `ATP [4]`
    *   `MIST GF [7]`
    *   `CoVGT [54]`
    *   `SeViT [15]`
    *   `HiTeA [64]`
*   **Zero-shot Methods:**
    *   `VFC [29]`
    *   `InternVideo [51]`
    *   `AssistGPT [6]`
    *   `ViperGPT [45]`
    *   `SeViLA [66]`
    *   `LLoVi [67]` (a key concurrent SOTA method)

        These baselines represent a spectrum of techniques, from video-first encoders to LLM-augmented systems, allowing for a comprehensive comparison of `VideoAgent`'s effectiveness and efficiency.

## 5.4. Implementation Details

The specific configurations and models used for the components of `VideoAgent` are crucial for reproducibility and understanding its performance.

*   **Video Decoding:** All videos in the experiments were decoded at **1 frame per second (fps)**. This downsampling helps manage the data volume of long-form videos while retaining sufficient temporal information for many tasks.

*   **CLIP for Frame Retrieval:**
    *   Model: `EVA-CLIP-8Bplus [43]` was utilized.
    *   Function: Retrieves the most relevant frames based on the `cosine similarity` between text queries (generated by the LLM) and the frame features.
    *   Details (from Appendix B): `EVA-CLIP-8Bplus` is a state-of-the-art CLIP model. Its vision encoder has 7.5 billion parameters, and its text encoder has 0.7 billion parameters. It processes images at a resolution of $448 \times 448$ and produces output features with a dimensionality of 1280.

*   **VLM for Captioning:**
    *   **For EgoSchema:** `LaViLa [68]` was used as the captioner.
        *   Type: This is a `clip-based captioning model`.
        *   Zero-shot consideration: Following `[67]`, a version of `LaViLa` retrained on `ego4D` data was used, with overlapped videos with `EgoSchema` filtered out, to ensure a zero-shot evaluation.
        *   Details (from Appendix B): `LaViLA` takes input video clips with a resolution of $4 \times 336 \times 336$ (meaning 4 frames, each $336 \times 336$ pixels).
    *   **For NExT-QA:** `CogAgent [9]` was used as the captioner.
        *   Type: This is a `frame-based captioning model`.
        *   Details (from Appendix B): `CogAgent` has 18 billion parameters and takes input images at a resolution of $1120 \times 1120$.

*   **LLM as Agent:**
    *   Model: `GPT-4 [31]` was used for all experiments.
    *   Version: Specifically, the `gpt-4-1106-preview` version was fixed to ensure reproducibility of results.
    *   Function: Controls the entire iterative process, performs reasoning, planning, self-reflection, query generation, and final answer prediction.

*   **Prompts:** The specific prompts used for `GPT-4` for `PredictAnswer`, `SelfReflect`, and `FindMissingInfo` are detailed in Appendix C of the paper. These prompts are crucial for guiding the LLM's behavior and ensuring its structured output (e.g., JSON format for confidence scores).

# 6. Results & Analysis

## 6.1. Core Results Analysis

`VideoAgent` demonstrates significant advancements in long-form video understanding, achieving state-of-the-art (SOTA) results on both EgoSchema and NExT-QA benchmarks. A key highlight is its exceptional efficiency, requiring substantially fewer frames than previous methods.

### 6.1.1. EgoSchema Performance

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<td colspan="2">Method</td>
<td>Frames</td>
<td>Subset</td>
<td>Full</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="2">FrozenBiLM [58]</td>
<td>90</td>
<td>-</td>
<td>26.9</td>
</tr>
<tr>
<td colspan="2">InternVideo [51]</td>
<td>90</td>
<td>-</td>
<td>32.1</td>
</tr>
<tr>
<td colspan="2">ImageViT [34]</td>
<td>16</td>
<td>40.8</td>
<td>30.9</td>
</tr>
<tr>
<td colspan="2">ShortViViTloc [34]</td>
<td>32</td>
<td>49.6</td>
<td>31.3</td>
</tr>
<tr>
<td colspan="2">LongViViT [34]</td>
<td>256</td>
<td>56.8</td>
<td>33.3</td>
</tr>
<tr>
<td colspan="2">SeViLA [66]</td>
<td>32</td>
<td>25.7</td>
<td>22.7</td>
</tr>
<tr>
<td colspan="2">Vamos [49]</td>
<td>-</td>
<td>.</td>
<td>48.3</td>
</tr>
<tr>
<td colspan="2">LLoVi [67]</td>
<td>180</td>
<td>57.6</td>
<td>50.3</td>
</tr>
<tr>
<td colspan="2">MC-ViT-L [2]</td>
<td>128+</td>
<td>62.6</td>
<td>44.4</td>
</tr>
<tr>
<td colspan="2">VideoAgent (ours)</td>
<td>8.4</td>
<td>60.2</td>
<td>54.1</td>
</tr>
</tbody>
</table>

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<td>Model</td>
<td>Subset</td>
<td>Full</td>
</tr>
</thead>
<tbody>
<tr>
<td>Random Chance</td>
<td>20.0</td>
<td>20.0</td>
</tr>
<tr>
<td>Bard only (blind) [2]</td>
<td>27.0</td>
<td>33.2</td>
</tr>
<tr>
<td>Bard + ImageViT [34]</td>
<td>35.0</td>
<td>35.0</td>
</tr>
<tr>
<td>Bard + ShortViViT [34]</td>
<td>42.0</td>
<td>36.2</td>
</tr>
<tr>
<td>Bard + PALI [34]</td>
<td>44.8</td>
<td>39.2</td>
</tr>
<tr>
<td>GPT-4 Turbo (blind) [2]</td>
<td>31.0</td>
<td>30.8</td>
</tr>
<tr>
<td>GPT-4V [2]</td>
<td>63.5</td>
<td>55.6</td>
</tr>
<tr>
<td>Gemini 1.0 Pro [47]</td>
<td>-</td>
<td>55.7</td>
</tr>
<tr>
<td>VideoAgent (ours)</td>
<td>60.2</td>
<td>54.1</td>
</tr>
</tbody>
</table>

*   **Superior Accuracy:** `VideoAgent` achieves an accuracy of **54.1%** on the full EgoSchema dataset, significantly outperforming the previous state-of-the-art method, `LLoVi [67]`, by **3.8%** (54.1% vs. 50.3%). On the 500-question subset, `VideoAgent` also achieves 60.2% accuracy.
*   **Efficiency:** Remarkably, `VideoAgent` achieves this performance using an average of **only 8.4 frames per video**. This is a stark contrast to `LLoVi`, which uses 180 frames, making `VideoAgent` approximately **20 times more frame-efficient**.
*   **Competitive with Proprietary Models:** While slightly trailing `GPT-4V` on the subset (60.2% vs. 63.5%), `VideoAgent`'s full dataset performance (54.1%) is competitive with advanced proprietary models like `Gemini 1.0 Pro` (55.7%) and even surpasses `GPT-4V` on the full dataset (54.1% vs. 55.6% implies a slight lag, but note the paper says "achieves comparable performance"). The abstract says "surpasses current state-of-the-art methods", and the table shows it outperforms `GPT-4V` in the full set. The table shows `GPT-4V` is 55.6% and `VideoAgent` is 54.1% so GPT-4V is slightly higher. The text states "achieves comparable performance to advanced proprietary models like Gemini-1.0 [47]". `Gemini-1.0` is 55.7%, and `VideoAgent` is 54.1%, so it's comparable. The abstract saying "superior effectiveness" and "highlighting the potential of agent-based approaches in advancing long-form video understanding" still stands.

### 6.1.2. NExT-QA Performance

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<td rowspan="2" colspan="2">Methods</td>
<td colspan="4">Val</td>
<td colspan="3">ATP-hard subset</td>
</tr>
<tr>
<td>Acc@C</td>
<td>Acc@T</td>
<td>Acc@D</td>
<td>Acc@All</td>
<td>Acc@C</td>
<td>Acc@T</td>
<td>Acc@All</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9">Supervised</td>
</tr>
<tr>
<td colspan="2">VFC [57]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>63.2</td>
<td>49.6</td>
<td>51.5</td>
<td>52.3</td>
</tr>
<tr>
<td colspan="2">ATP [4]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>66.8</td>
<td>53.1</td>
<td>50.2</td>
<td>54.3</td>
</tr>
<tr>
<td colspan="2">MIST GF [7]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>66.9</td>
<td>54.6</td>
<td>56.6</td>
<td>57.2</td>
</tr>
<tr>
<td colspan="2">CoVGT [54]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>70.5</td>
<td>56.9</td>
<td>57.1</td>
<td>58.8</td>
</tr>
<tr>
<td colspan="2">SeViT [15]</td>
<td>59.7</td>
<td>58.0</td>
<td>60.7</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">HiTeA [64]</td>
<td>54.0</td>
<td>54.1</td>
<td>56.7</td>
<td>63.1</td>
<td>43.3</td>
<td>46.5</td>
<td>-</td>
</tr>
<tr>
<td colspan="9">Zero-shot</td>
</tr>
<tr>
<td colspan="2">VFC [29]</td>
<td>62.4</td>
<td>58.3</td>
<td>71.3</td>
<td>75.6</td>
<td>47.8</td>
<td>48.6</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">InternVideo [51]</td>
<td>51.6</td>
<td>45.4</td>
<td>48.0</td>
<td>64.1</td>
<td>51.5</td>
<td>32.2</td>
<td>30.0</td>
</tr>
<tr>
<td colspan="2">AssistGPT [6]</td>
<td>43.4</td>
<td>65.1</td>
<td>49.1</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>31.4</td>
</tr>
<tr>
<td colspan="2">ViperGPT [45]</td>
<td>60.0</td>
<td>51.4</td>
<td>67.3</td>
<td>58.4</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">SeViLA [66]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>60.0</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">LLoVi [67]</td>
<td>61.3</td>
<td>61.5</td>
<td>75.6</td>
<td>63.6</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">VideoAgent (ours)</td>
<td>69.5</td>
<td>61.0</td>
<td>75.6</td>
<td>67.7</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="2">VideoAgent (ours)</td>
<td>72.7</td>
<td>64.5</td>
<td>81.1</td>
<td>71.3</td>
<td>57.8</td>
<td>58.8</td>
<td>58.4</td>
</tr>
</tbody>
</table>

*   **Overall SOTA:** `VideoAgent` achieves **71.3% accuracy** on the NExT-QA full validation set, surpassing the previous zero-shot SOTA, `LLoVi [67]`, by **3.6%** (71.3% vs. 67.7%). It also outperforms all supervised methods listed.
*   **Performance Across Question Types:** `VideoAgent` shows strong performance across all NExT-QA question categories: Causal (72.7%), Temporal (64.5%), and Descriptive (81.1%), consistently outperforming other methods.
*   **Challenging Subsets:** Crucially, `VideoAgent` demonstrates remarkable improvements on the more difficult `ATP-hard subset`, achieving 58.4% accuracy overall, with 57.8% on Causal and 58.8% on Temporal questions. This highlights its ability to tackle complex long-form video queries that require deep temporal reasoning.
*   **Efficiency:** Similar to EgoSchema, `VideoAgent` uses an average of **only 8.2 frames per video** on NExT-QA for zero-shot evaluation, underscoring its efficiency.

    These results unequivocally establish `VideoAgent` as a highly effective and efficient method for long-form video understanding, particularly for zero-shot tasks and complex reasoning scenarios.

## 6.2. Ablation Studies / Parameter Analysis

The paper conducts comprehensive analyses and ablation studies to investigate the effectiveness of `VideoAgent`'s key components and design choices.

### 6.2.1. Frame Efficiency and Number of Rounds

The following figure (Figure 3 from the original paper) shows frame efficiency compared to uniform sampling and previous methods, and the number of frames for different types of NExT-QA questions:

![Fig. 3: (Left) Frame efficiency compared to uniform sampling and previous methods. Xaxis is in log scale. Our method achieves exceptional frame efficiency for long-form video understanding. (Right) Number of frames for different types of NExT-QA questions. Min, mean, max, distribution are plotted. VideoAgent selects more frames on questions related to temporal reasoning than causal reasoning and descriptive questions.](images/3.jpg)
*Fig. 3: (Left) Frame efficiency compared to uniform sampling and previous methods. Xaxis is in log scale. Our method achieves exceptional frame efficiency for long-form video understanding. (Right) Number of frames for different types of NExT-QA questions. Min, mean, max, distribution are plotted. VideoAgent selects more frames on questions related to temporal reasoning than causal reasoning and descriptive questions.*

*   **Frame Efficiency (Figure 3, Left):** The left plot compares `VideoAgent`'s accuracy against uniform sampling baselines and other methods as a function of the number of frames used (x-axis in log scale). `VideoAgent` consistently and significantly outperforms uniform selection and other baselines at the same number of frames. For instance, `VideoAgent` achieves 60.2% accuracy with only 8.4 frames, surpassing a baseline that uniformly samples 180 frames to achieve 59.6% accuracy. This indicates `VideoAgent`'s superior ability to identify and retrieve the *most informative* frames, demonstrating that simply using more frames (especially if uninformative) does not guarantee better performance and can even degrade it due to LLM context overload and distractions.
*   **Number of Rounds (Figure 3, Left):** The plot also shows performance across 1-4 rounds of iteration on the EgoSchema 500-question subset.
    *   1 Round: 53.8% accuracy with 5 frames.
    *   2 Rounds: 58.6% accuracy with 7.5 frames.
    *   3 Rounds: 60.2% accuracy with 8.4 frames.
    *   4 Rounds: 59.8% accuracy with 9.9 frames.
        Performance improves with additional rounds but saturates at three rounds, after which more iterations (and frames) do not yield further gains and can even slightly decrease performance. This confirms that the iterative process efficiently finds the necessary information.

### 6.2.2. Different Question Types

*   **Adaptive Frame Selection for Question Difficulty (Figure 3, Right):** Analyzing the NExT-QA dataset, the distribution of frames used varies by question type:
    *   **Descriptive questions:** Average 5.9 frames.
    *   **Causal questions:** Average 7.1 frames.
    *   **Temporal questions:** Average 7.8 frames.
        This aligns with human intuition: descriptive tasks often need less information (initial uniform sampling might suffice), while reasoning tasks, particularly temporal ones, require more frames to accurately capture sequences and relationships. This highlights the adaptiveness of `VideoAgent`'s dynamic frame selection.

### 6.2.3. Initial Number of Frames

The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<td></td>
<td>Uni-7</td>
<td>Uni-9</td>
<td>Uni-11</td>
</tr>
</thead>
<tbody>
<tr>
<td>Uniform</td>
<td>54.6</td>
<td>54.8</td>
<td>55.8</td>
</tr>
<tr>
<td>Ours</td>
<td>3→6.4 (58.4)</td>
<td>5→8.4 (60.2)</td>
<td>8→11.0 (57.4)</td>
</tr>
</tbody>
</table>

*Note: The table format in the paper is slightly unusual. "Uni-7" likely refers to uniform sampling with 7 frames, with the accuracy below. "3→6.4" means starting with 3 initial frames, leading to 6.4 average frames, and 58.4% accuracy.*

This ablation studies the impact of the initial number of uniformly sampled frames ($N$) on overall performance and frame count (on the EgoSchema 500-question subset).
*   Starting with 3 frames: Achieves 58.4% accuracy with an average of 6.4 frames.
*   Starting with 5 frames: Achieves the highest performance at 60.2% accuracy with an average of 8.4 frames.
*   Starting with 8 frames: Achieves 57.4% accuracy with an average of 11.0 frames.
    This suggests an optimal initial "glance" size, with 5 frames yielding the best balance between initial context and subsequent efficiency. Compared to uniform sampling (e.g., 55.8% for 11 uniform frames vs. 57.4% for 8 initial frames leading to 11.0 average frames in `VideoAgent`), `VideoAgent`'s iterative selection remains superior even with comparable total frames.

### 6.2.4. Self-evaluation

The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<td>Method</td>
<td>Frames</td>
<td>Acc</td>
</tr>
</thead>
<tbody>
<tr>
<td>Ours w/o Seg. Selection</td>
<td>7.5</td>
<td>56.6</td>
</tr>
<tr>
<td>Ours w/o Self-Evaluation</td>
<td>11.8</td>
<td>59.6</td>
</tr>
<tr>
<td>Ours</td>
<td>8.4</td>
<td>60.2</td>
</tr>
</tbody>
</table>

This ablation examines the effectiveness of the LLM's `self-evaluation` mechanism (the `SelfReflect` step).
*   **`Ours w/o Self-Evaluation`:** If every question is processed through a fixed number of iterations (e.g., three rounds, as opposed to terminating early via self-evaluation), the average number of frames used increases from 8.4 to 11.8. Simultaneously, accuracy decreases from 60.2% to 59.6%.
*   This demonstrates that `self-evaluation` is crucial for:
    1.  **Efficiency:** It prevents unnecessary iterations and information gathering once enough context is acquired.
    2.  **Effectiveness:** It avoids `over-contextualization` or `noise` from superfluous information, leading to better accuracy.

### 6.2.5. Segment Selection

*   **`Ours w/o Seg. Selection` (Table 5):** When the `segment selection` strategy (where the LLM specifies which video segments to search within) is disabled, `VideoAgent` experiences a 3.6% accuracy degradation (from 60.2% to 56.6%), even though it uses fewer frames (7.5 vs 8.4).
*   This highlights the importance of `segment selection` for:
    1.  **Temporal Reasoning:** It allows the model to focus retrieval on temporally relevant portions of the video, crucial for questions like "what happens after...".
    2.  **Mitigating Irrelevance:** It prevents the retrieval of frames that, while visually matching a query, are temporally out of context for the question.

### 6.2.6. Ablation of Foundation Models

The paper also ablates the impact of different choices for the underlying LLM, VLM, and CLIP models on `VideoAgent`'s performance.

#### LLM Ablation

The following are the results from Table 6 of the original paper:

<table>
<thead>
<tr>
<td>LLM</td>
<td>Model Size</td>
<td>Acc. (%)</td>
</tr>
</thead>
<tbody>
<tr>
<td>Mistral-8x7B</td>
<td>70B</td>
<td>37.8</td>
</tr>
<tr>
<td>Llama2-70B</td>
<td>70B</td>
<td>45.4</td>
</tr>
<tr>
<td>GPT-3.5</td>
<td>N/A</td>
<td>48.8</td>
</tr>
<tr>
<td>GPT-4</td>
<td>N/A</td>
<td>60.2</td>
</tr>
</tbody>
</table>

*   `GPT-4` significantly outperforms other LLMs (LLaMA-2-70B, Mixtral-8x7B, GPT-3.5) by a large margin (60.2% vs. 48.8% for GPT-3.5, and even lower for others).
*   This superior performance is primarily attributed to `GPT-4`'s robust capability in `structured prediction` (specifically, generating correct `JSON` formats), which is crucial for orchestrating the iterative process (e.g., outputting confidence scores and missing information queries in a parseable format). Other models struggled more with consistent JSON generation.

#### VLM Ablation

The following are the results from Table 7 of the original paper:

<table>
<thead>
<tr>
<td>Captioner</td>
<td>Type</td>
<td># Words</td>
<td>Acc. (%)</td>
</tr>
</thead>
<tbody>
<tr>
<td>BLIP-2</td>
<td>Frame-based</td>
<td>8.5</td>
<td>52.4</td>
</tr>
<tr>
<td>LaViLa</td>
<td>Clip-based</td>
<td>7.2</td>
<td>60.2</td>
</tr>
<tr>
<td>CogAgent</td>
<td>Frame-based</td>
<td>74.2</td>
<td>60.8</td>
</tr>
</tbody>
</table>

*   `CogAgent` and `LaViLa` yield similar high performance (60.8% and 60.2% respectively) despite `CogAgent` generating significantly longer captions (74.2 words vs. 7.2 words for LaViLa).
*   `BLIP-2` performs considerably worse (52.4%), indicating that the quality and detail of VLM captions are important, but not necessarily their length. `LaViLa`'s concise, high-quality captions are effective.

#### CLIP Ablation

The following are the results from Table 8 of the original paper:

<table>
<thead>
<tr>
<td>CLIP</td>
<td>Model Size</td>
<td>Resolution</td>
<td>Acc. (%)</td>
</tr>
</thead>
<tbody>
<tr>
<td>OpenCLIP ViT-G</td>
<td>1B</td>
<td>224</td>
<td>59.2</td>
</tr>
<tr>
<td>EVA-CLIP-8B</td>
<td>8B</td>
<td>224</td>
<td>59.4</td>
</tr>
<tr>
<td>EVA-CLIP-8B-plus</td>
<td>8B</td>
<td>448</td>
<td>60.2</td>
</tr>
</tbody>
</table>

*   The performance across different CLIP models (`OpenCLIP ViT-G`, `EVA-CLIP-8B`, `EVA-CLIP-8B-plus`) is comparable, with `EVA-CLIP-8B-plus` achieving the highest (60.2%).
*   This suggests that the `retrieval step` itself is not a bottleneck for the `VideoAgent` methodology, as even slightly smaller or lower-resolution CLIP models perform quite well. The robustness indicates that the overall system design is effective regardless of minor variations in the CLIP component.

    Overall, the ablation studies validate the design choices of `VideoAgent`, confirming the effectiveness of its iterative reasoning, self-evaluation, segment selection, and the critical role of strong LLM reasoning capabilities.

## 6.3. Case Studies

The paper presents several case studies to qualitatively demonstrate `VideoAgent`'s capabilities in understanding long-form videos.

### 6.3.1. Questions from NExT-QA

The following figure (Figure 4 from the original paper) shows a case study on NExT-QA:

![Fig. 4: Case study on NExT-QA. VideoAgent accurately identifies missing information in the first round, bridges the information gap in the second round, and thereby makes the correct prediction.](images/9.jpg)
*Fig. 4: Case study on NExT-QA. VideoAgent accurately identifies missing information in the first round, bridges the information gap in the second round, and thereby makes the correct prediction.*

*   **Scenario:** A question asks, "Why did the man in black sweater hold up a cup of water while talking to friends?"
*   **Initial State (Round 1):** The initial uniform sampling provides captions for several frames (e.g., man sitting, young woman sitting, man gesturing). `VideoAgent`'s `Predict Answer` step notes that descriptions do not explicitly mention a man holding a cup of water. The `Self Reflect` step assigns a confidence of "1" (Insufficient Information).
*   **Finding Missing Info (Round 1 Action):** The LLM agent then identifies the need for information about "the man in the black sweater holding a cup of water." It formulates a query and specifies segments to search.
*   **Retrieval and Update (Round 2):** CLIP retrieves a new frame (Frame 69) where the man is clearly holding a glass and drinking. This new observation is captioned and merged into the state.
*   **Final Prediction (Round 2 Action):** With this crucial new information, `VideoAgent` can now confidently predict the answer, which indicates the man is taking a drink. This case study effectively illustrates the iterative process of identifying information gaps and strategically retrieving relevant visual details to bridge them.

### 6.3.2. Hour-long Videos

The following figure (Figure 5 from the original paper) shows a case study on hour-long videos:

![Fig. 5: Case study on hour-long videos. VideoAgent accurately identifies the key frame during the second iteration, subsequently making an accurate prediction. Conversely, GPT-4V, when relying on 48 uniformly sampled frames up to its maximum context length, does not get successful prediction. However, by integrating the frame pinpointed by VideoAgent, GPT-4V is able to correctly answer the question.](images/10.jpg)
*Fig. 5: Case study on hour-long videos. VideoAgent accurately identifies the key frame during the second iteration, subsequently making an accurate prediction. Conversely, GPT-4V, when relying on 48 uniformly sampled frames up to its maximum context length, does not get successful prediction. However, by integrating the frame pinpointed by VideoAgent, GPT-4V is able to correctly answer the question.*

*   **Scenario:** A question asks about the color of the stairs surrounded by green plants in an hour-long YouTube video. This information exists but occupies only a small portion of the very long video.
*   **`VideoAgent` Performance:** `VideoAgent` efficiently identifies the key frame containing the stairs (frame 1754, in the second iteration). With this single crucial frame, it accurately answers the question within only two iterations and a total of seven frames.
*   **Comparison with `GPT-4V`:**
    *   `GPT-4V` (a powerful multi-modal model) struggles when relying on its maximum context length of 48 uniformly sampled images from the hour-long video. It fails to make a successful prediction. This highlights the limitation of even advanced multi-modal models when faced with a needle-in-a-haystack problem in long videos.
    *   However, when `GPT-4V` is provided with the *specific frame (frame 1754)* pinpointed by `VideoAgent`, it *can* successfully answer the question.
*   **Implication:** This case study vividly demonstrates `VideoAgent`'s ability to handle extreme video lengths and its superior efficiency in finding critical information. It also suggests that `VideoAgent` could serve as an effective "scout" or "attention mechanism" to enhance the capabilities of other powerful multi-modal models like `GPT-4V` for long-form content.

    These case studies underscore `VideoAgent`'s practical utility and its ability to generalize to challenging real-world scenarios, going beyond what traditional sparse or dense sampling methods can achieve.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This work introduces `VideoAgent`, a novel system that effectively addresses the complex challenge of long-form video understanding. By mirroring the human cognitive process, `VideoAgent` leverages a large language model (LLM) as an intelligent agent to iteratively search for and aggregate crucial information from videos. Vision-language models (VLMs) translate visual content into text, while contrastive language-image models (CLIP) efficiently retrieve relevant frames guided by the LLM's dynamic queries. Through a multi-round, self-reflecting, and segment-aware iterative process, `VideoAgent` achieves state-of-the-art zero-shot accuracy on EgoSchema (54.1%) and NExT-QA (71.3%) benchmarks. Critically, it does so with exceptional efficiency, using an average of only 8.4 and 8.2 frames, respectively, far fewer than previous methods. This demonstrates a significant stride in prioritizing reasoning and intelligent planning over the direct processing of vast visual data.

## 7.2. Limitations & Future Work
The authors implicitly point to certain limitations and suggest future directions:
*   **Reliance on Foundation Models:** The performance of `VideoAgent` is inherently tied to the capabilities of its underlying foundation models (LLM, VLM, CLIP). While this is a strength in leveraging SOTA models, it also means any weaknesses or biases in these components can propagate through the system.
*   **Caption-based Modality Translation:** The current approach relies on VLMs to generate textual captions from frames, which are then fed to the text-only LLM. This translation step might lose subtle visual nuances that cannot be fully captured in text.
*   **Future Work Suggestions:**
    *   **Integration of Better Models:** As LLMs, VLMs, and CLIPs rapidly advance, integrating newer, more capable models into `VideoAgent` could further improve its performance.
    *   **Caption-free Methodology:** The authors suggest exploring "caption-free methodology by replacing GPT-4 with GPT-4V." This implies moving towards multi-modal LLMs that can directly process visual inputs alongside text, potentially eliminating the VLM captioning step and reducing information loss. This would allow the LLM agent to "see" directly rather than through textual descriptions.

## 7.3. Personal Insights & Critique

This paper presents a highly inspiring and practically valuable approach to long-form video understanding.

*   **Inspirations Drawn:**
    *   **Human-Cognition as a Blueprint:** The most compelling aspect is the direct inspiration from human cognition. This approach feels intuitive and resource-efficient, aligning with how we naturally process complex information. It's a strong argument for agent-based systems over monolithic, brute-force processing.
    *   **Orchestration over Integration:** Instead of trying to build a single, massive multi-modal model that can do everything, `VideoAgent` effectively orchestrates specialized tools. This modularity makes it flexible, robust to component failures (if one VLM isn't great, swap it out), and leverages the best-in-class for each sub-task.
    *   **Efficiency as a First-Class Citizen:** The dramatic reduction in frames needed (20x fewer than LLoVi) is a game-changer for practical applications, especially with long videos where computational resources and time are critical constraints. This makes real-time or near-real-time analysis of long content more feasible.
    *   **Potential for Enhancing Other Models:** The case study showing `VideoAgent` pinpointing a frame for `GPT-4V` is particularly insightful. It suggests that specialized agents like `VideoAgent` could act as intelligent pre-processors or attentional mechanisms for broader multi-modal LLMs, helping them overcome their context limitations on specific tasks.

*   **Potential Issues, Unverified Assumptions, or Areas for Improvement:**
    *   **Dependency on LLM Quality for Reasoning and Structured Output:** The ablation study clearly shows the strong dependency on `GPT-4`'s superior reasoning and JSON parsing capabilities. If open-source LLMs cannot reliably produce structured outputs or perform complex reasoning steps (self-reflection, query rewriting), then deploying `VideoAgent` with them would be challenging. This creates a reliance on proprietary models, which might be a barrier for some researchers or applications.
    *   **Prompt Engineering Fragility:** The performance of the LLM agent is heavily reliant on the quality of prompts used for `PredictAnswer`, `SelfReflect`, and `FindMissingInfo`. Small changes in prompt wording could potentially lead to significant performance shifts, requiring careful engineering and validation.
    *   **Semantic Fidelity of Captioning:** While `CogAgent` and `LaViLa` are strong VLMs, any errors or ambiguities in their captions will directly impact the LLM's understanding and subsequent reasoning. This "language bottleneck" is inherent when translating visual information to text for a text-only LLM.
    *   **"Human-like" but not truly Human:** While inspired by human cognition, the current model still operates on predefined categories of "confidence" and explicit JSON outputs. True human reasoning is more nuanced, involves implicit understanding, and can adapt to entirely novel situations without explicit instructions for tool use. The framework is a step towards, but not yet a replica of, human understanding.
    *   **Scalability for Extreme Long-form Videos:** While shown for hour-long videos, scaling to truly massive datasets (e.g., days or weeks of video) might still present challenges for the initial uniform sampling and the iterative process if the "needle" becomes extraordinarily small in an immense "haystack."
    *   **Lack of Temporal Context in CLIP Retrieval:** While `segment selection` helps, CLIP itself performs image-text similarity independently for each frame. It doesn't inherently understand complex temporal relationships between frames when retrieving. The LLM's guidance helps, but the underlying retrieval mechanism is still frame-centric.

        Overall, `VideoAgent` represents an excellent example of how to combine the strengths of different foundation models under an intelligent agent paradigm. Its efficiency and effectiveness are highly promising for the future of video understanding, particularly in practical, resource-constrained environments.