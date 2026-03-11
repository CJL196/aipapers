# 1. Bibliographic Information

## 1.1. Title
The central topic of this paper is introduced in its title: **PaLM-E: An Embodied Multimodal Language Model**. This title indicates the core contribution is a new type of artificial intelligence model named "PaLM-E," which combines the capabilities of large language models (PaLM) with embodied interaction (E), allowing it to perceive and act within the physical world through multiple sensory modalities.

## 1.2. Authors
The paper was authored by a large collaborative team primarily affiliated with robotics and AI research divisions. The lead authors include Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Their affiliations span **Robotics at Google**, **TU Berlin** (Technische Universität Berlin), and **Google Research**. This mix of academia and industry suggests a strong focus on both theoretical innovation and practical application in real-world robotic systems.

## 1.3. Journal/Conference
The paper was published as a preprint on **arXiv** under the identifier `2303.03378`. While not a peer-reviewed conference proceeding at the time of this specific release, arXiv is the primary repository for cutting-edge machine learning research. The work reflects state-of-the-art developments in multimodal learning and robotics as of early 2023. The official publication status listed is **preprint**, meaning it is publicly available for review before final journal acceptance.

## 1.4. Publication Year
The paper was published on **2023-03-06T18:58:06.000Z**. This places the research in the context of the rapid evolution of Large Language Models (LLMs) between late 2022 and 2023, a period characterized by the emergence of foundation models capable of few-shot learning and complex reasoning.

## 1.5. Abstract
The abstract outlines the research objective: enabling general inference in the real world for robotics problems, specifically addressing the challenge of "grounding" (linking words to percepts). The core methodology involves proposing **embodied language models** that incorporate real-world continuous sensor modalities directly into language models. The main results show that **PaLM-E**, a single large model trained end-to-end, can address various embodied reasoning tasks across different observation modalities and embodiments. Key conclusions highlight positive transfer benefits from joint training across diverse domains and demonstrate that the largest variant (**PaLM-E-562B**) achieves state-of-the-art performance on visual-language benchmarks while retaining general language capabilities.

## 1.6. Original Source Link
The official source for the paper is available at the following link:
https://arxiv.org/abs/2303.03378
The PDF version can be accessed via:
https://arxiv.org/pdf/2303.03378v1

---

# 2. Executive Summary

## 2.1. Background & Motivation
Large Language Models (LLMs) have demonstrated exceptional capabilities in tasks such as dialogue, mathematical reasoning, and code generation. However, a significant limitation exists when applying these models to inference in the real world, particularly for robotics. This problem is known as **Grounding**. While LLMs are trained on massive textual data which may contain representations related to the physical world, connecting these abstract linguistic representations to real-world visual and physical sensor modalities (like camera images or robot joint angles) is essential for solving grounded problems. Prior approaches often interface LLM outputs with learned policies but remain limited because the LLM itself receives only textual input, lacking the geometric configuration of the scene required for decision-making.

The motivation for this work is to bridge this gap by creating **embodied language models**. These models directly incorporate continuous inputs from sensors into the language embedding space, allowing the LLM to make more grounded inferences for sequential decision-making. The entry point is the architectural idea of injecting continuous observations (like images or state estimates) into the same latent embedding space used for language tokens, processed autoregressively within a Transformer-based LLM.

## 2.2. Main Contributions / Findings
The paper makes several primary contributions:
1.  **Proposed a Generalist Embodied Agent:** Demonstrated that a transfer-learned, multi-embodiment decision-making agent can be trained by mixing embodied data into the training of a multimodal large language model.
2.  **Solved Embodied Reasoning:** Showed that current state-of-the-art general-purpose visual-language models cannot well address embodied reasoning problems out-of-the-box, but a competent model (PaLM-E) can be trained to do so efficiently.
3.  **Novel Architectural Ideas:** Introduced techniques such as neural scene representations (OSRT) and entity-labeling multimodal tokens to handle object-centric information effectively.
4.  **Quantitative Competence:** Proved that PaLM-E is quantitatively competent not just as an embodied reasoner but also as a vision and language generalist.
5.  **Scaling Benefits:** Demonstrated that scaling the language model size enables multimodal fine-tuning with significantly less catastrophic forgetting of original language capabilities.

    A key finding is that despite robotics data being scarce compared to internet-scale datasets, the model exhibits **positive transfer**, benefiting from diverse joint training across language, vision, and visual-language domains to achieve high data efficiency in robotics tasks.

---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, a reader must grasp several foundational technologies and theories in Artificial Intelligence and Robotics.

*   **Large Language Models (LLMs):** These are generative models trained to predict the probability of a piece of text given previous text. They are typically based on the **Transformer** architecture, which uses self-attention mechanisms to process sequences of data. An LLM learns to represent words as vectors (embeddings) in a high-dimensional space, capturing semantic relationships. In this paper, the base model is **PaLM** (Pathways Language Model).
*   **Vision Transformers (ViT):** A transformer architecture adapted for image recognition. Unlike traditional Convolutional Neural Networks (CNNs) that process pixels locally, ViTs divide an image into patches (small squares) and treat each patch as a token, similar to words in a sentence. This allows the model to capture global context.
*   **Embedding Space:** A mathematical representation where data (words, images, states) are converted into vectors. In this paper, continuous sensor data is mapped into the same vector space as language tokens, allowing them to be processed together.
*   **Embodiment:** In robotics and AI, embodiment refers to an agent having a physical body or sensorimotor system that interacts with the environment. "Embodied AI" requires the AI to perceive actions and their consequences in the real world.
*   **Grounding:** The process of linking abstract symbols (like words) to real-world referents (like objects, scenes, or actions). Without grounding, an AI might know the word "cup" but not how to interact with one physically.
*   **Autoregressive Generation:** A method where a model generates output sequentially, predicting the next element (token) based on the previously generated elements. This is how LLMs write sentences.
*   **Transfer Learning:** A technique where a model trained on one task (e.g., visual question answering) is reused as the starting point for another task (e.g., robotic manipulation).
*   **Catastrophic Forgetting:** A phenomenon in machine learning where a neural network loses its ability to perform a previously learned task (e.g., general language understanding) when it is fine-tuned on a new task (e.g., robotics).

## 3.2. Previous Works
The authors situate their work within several lines of prior research:

*   **General Vision-Language Modeling:** Previous works like **Flamingo** and **PaLI** combined vision and language models. Unlike predecessors that might augment models with a mechanism to attend to a single context image, PaLM-E represents images and text as "multimodal sentences" of latent vectors, allowing flexible processing of multiple images within any part of a sentence.
*   **Actions-output Models:** Prior works (e.g., **VIMA**, **Gato**) combined vision and language for direct action prediction. In these works, language often serves as task specification. In contrast, PaLM-E generates high-level instructions as text, conditioning low-level commands.
*   **LLMs in Embodied Task Planning:** Many works leverage LLMs for planning but often rely on prompting or external grounding mechanisms. Few consider natural language as a representation for planning directly integrated into the model parameters without auxiliary models.
*   **Frozen Language Models:** Work by **Tsimpoukelli et al. (Frozen)** optimizes vision encoder parameters via backpropagation through a frozen LLM. Inspired by this, the authors investigate design choices where the LLM is either frozen or unfrozen during training.

## 3.3. Technological Evolution
The field has evolved from purely text-based LLMs to multimodal models that can see (Vision-Language Models), and now to Embodied Models that can act. Early robotics relied on hard-coded rules or separate perception and control modules. Modern approaches attempt to unify these using foundation models. This paper represents a shift towards **unified multimodal transformers** where perception (vision/state) and cognition (language/planning) share the same architecture and training objective, rather than being separate modules communicating via fixed interfaces.

## 3.4. Differentiation Analysis
Compared to related methods, PaLM-E differs in three critical ways:
1.  **Unified Architecture:** It injects continuous observations directly into the language embedding space as interleaved tokens, rather than treating vision and language as separate branches that are fused only at the decision layer.
2.  **End-to-End Training:** Unlike some baselines that freeze the LLM or use heuristic planners, PaLM-E trains encoders and the LLM jointly to generate plans in textual form.
3.  **Scale and Generalization:** By scaling to 562B parameters and training on diverse internet-scale data alongside robotics data, it demonstrates emergent capabilities like multimodal Chain-of-Thought (CoT) reasoning, which smaller or task-specific models fail to exhibit.

    ---

# 4. Methodology

## 4.1. Principles
The core principle of **PaLM-E** is to enable a decoder-only Large Language Model (LLM) to process continuous, embodied observations (such as images or robot state vectors) by treating them as if they were discrete language tokens. Instead of converting images into text captions or using separate classifiers, the model maps these observations into the language embedding space $\mathcal{X}$. This allows the model's existing self-attention layers to process sensory data and text in a unified manner, facilitating seamless grounding and reasoning.

## 4.2. Core Methodology In-depth (Layer by Layer)

### Step 1: Decoder-only LLM Structure
The foundation is a standard decoder-only LLM trained to predict the probability $p ( w _ { 1 : L } )$ of a piece of text $w _ { 1 : L } = ( w _ { 1 } , \dots , w _ { L } )$. This is represented as a sequence of tokens $w _ { i } \in \mathcal W$. Typical neural architectures realize this by factorizing the probability into:
$$
p ( w _ { 1 : L } ) = \prod _ { l = 1 } ^ { L } p _ { \mathrm { LM } } ( w _ { l } | w _ { 1 : l - 1 } )
$$
In this formula, $p _ { \mathrm { LM } }$ represents the large transformer network, and the product symbol $\prod$ indicates that the total probability is calculated by multiplying the probabilities of each token given its predecessors. This autoregressive nature means the model predicts the next token step-by-step.

### Step 2: Prefix Conditioning
Since the LLM is autoregressive, a pre-trained model can be conditioned on a prefix `w _ { 1 : n }` without changing the architecture. This allows the model to continue predicting subsequent tokens $w _ { n + 1 : L }$ based on a provided context or prompt. The conditional probability is defined as:
$$
p ( w _ { n + 1 : L } | w _ { 1 : n } ) = \prod _ { l = n + 1 } ^ { L } p _ { \mathrm { LM } } ( w _ { l } | w _ { 1 : l - 1 } )
$$
This formula shows that the prediction for token $l$ depends on all previous tokens from index `1` to `l-1`, including those in the prefix $w_{1:n}$.

### Step 3: Token Embedding Space and Continuous Injection
Normally, tokens `w _ { i }` belong to a fixed vocabulary $\mathcal W$ and are embedded into a word token embedding space $\mathcal X \subset \mathbb R ^ { k }$ via $\gamma : \mathcal W \to \mathcal X$. To support embodied inputs, PaLM-E skips the discrete token level for observations. Continuous observations $O$ are mapped into the language embedding space $\mathcal X$ using an encoder $\phi : \mathcal O \to \mathcal X ^ { q }$. This encoder maps a continuous observation space into a sequence of $q$-many vectors in $\mathcal X$. These vectors are interleaved with normal embedded text tokens to form the prefix. Each vector `x _ { i }` in the prefix is formed from either the word token embedder $\gamma$ or an encoder $\phi _ { j }$:
$$
x _ { i } = \left\{ \begin{array} { l l } { \gamma ( w _ { i } ) } & { \mathrm { i f ~ } i \mathrm { ~ a ~ i s ~ t e x t ~ t o k e n , ~ o r ~ } } \\ { \phi _ { j } ( O _ { j } ) _ { i } } & { \mathrm { i f ~ } i \mathrm { ~ c o r r e s p o n d s ~ t o ~ o b s e r v a t i o n ~ } O _ { j } . } \end{array} \right.
$$
Here, $x_i$ represents the embedding of the $i$-th token in the sequence. If $i$ corresponds to text, it uses the standard embedder $\gamma$. If $i$ corresponds to an observation $O_j$ (like an image), it uses the projection of the observation encoder $\phi_j$. This allows multiple observations (e.g., multiple images) to be interleaved dynamically within the text, reusing the LLM's positional encodings.

### Step 4: Input Encoders and Representations
The paper investigates different types of encoders $\phi$ for different sensor modalities:
*   **State Estimation Vectors:** Simple vectors $\boldsymbol { s } \in \mathbb R ^ { S }$ describing object poses, sizes, colors, etc., are mapped into the embedding space via an MLP $\phi _ { \mathrm { s t a t e } }$.
*   **Vision Transformer (ViT):** Images $I$ are mapped into token embeddings $\tilde { x } _ { 1 : m } \ = \ \tilde { \phi } _ { \mathrm { v i T } } ( \bar { I } ) \ \in \ \mathbb R ^ { m \times \tilde { k } } $. Since the dimensionality $\tilde { k }$ of ViT embeddings may differ from the LLM's $k$, each embedding is projected using an affine transformation $\psi$:
    $$
    x _ { i } = \overset { \vartriangle } { \phi _ { \mathrm { V i T } } } ( I ) _ { i } = \psi ( \widetilde { \phi } _ { \mathrm { V i T } } ( I ) _ { i } )
    $$
*   **Object Scene Representation Transformer (OSRT):** This learns 3D-centric neural scene representations unsupervisedly. Scene representations consist of object slots $o _ { j } = \bar { \phi } _ { 0 \mathrm { S R T } } ( I _ { 1 : v } ) _ { j } \in \bar { \mathbb R } ^ { \bar { k } }$. These are projected similarly using $\psi$ to map into $m$-many embeddings.
*   **Entity Referrals:** To reference objects in plans, object-centric tokens are labeled (e.g., ${<}\mathrm { obj-1 >}$). This allows the model to reference objects via special tokens in the output sentence, assuming low-level policies operate on these tokens as well.

### Step 5: Embodying the Output
When PaLM-E is tasked with producing decisions, it generates text autoregressively. If the task requires a plan, the output is a sequence of skills chosen from a vocabulary available to the low-level policy. These predictions are executed by a low-level policy or planner, leading to new observations based on which PaLM-E can replan. This creates a closed-loop control system where PaLM-E acts as a high-level policy sequencing low-level skills.

### Step 6: Training Recipes
The model is trained on a dataset $D$ consisting of examples $\left( I _ { 1 : u _ { i } } ^ { i } , w _ { 1 : L _ { i } } ^ { i } , n _ { i } \right)$. Each example contains continuous observations $I$, text $w$, and an index $n_i$ splitting the text into a prefix (multi-modal sentence) and a prediction target (text only). The loss function is cross-entropy averaged over non-prefix tokens:
$$
\mathcal { L } = \frac { 1 } { \sum (L_i - n_i) } \sum_i \sum_{j=n_i+1}^{L_i} \text{CrossEntropy}(y_{ij}, \hat{y}_{ij})
$$
*(Note: While the paper describes the cross-entropy loss, the specific summation notation above is derived from the standard definition described in the text: "loss function is therefore a cross-entropy loss averaged over the individual non-prefix tokens ni+1:Li".)*

Training involves variations such as freezing the LLM and only training encoders, or co-training across diverse tasks (robotics, VQA, captioning) using a "full mixture" dataset where embodied data constitutes only about $8.9\%$.

---

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize a diverse range of datasets to validate both embodied reasoning and general vision-language capabilities.

*   **Robotics Domains:**
    *   **TAMP (Task and Motion Planning):** Involves manipulating (grasping and stacking) objects. Used to test combinatorial planning and spatial reasoning.
    *   **Language-Table:** Taken from the publicly available Language-Table dataset. Involves tabletop pushing environments with multiple objects and complex language commands.
    *   **Mobile Manipulation:** Similar to SayCan, involving robots navigating kitchens to find objects, pick them up, and deliver them to humans.
*   **General Vision-Language Tasks:**
    *   **Webli:** Internet-scale image-text pairs.
    *   **VQAv2, OK-VQA:** Visual Question Answering benchmarks requiring visual understanding.
    *   **COCO:** Image captioning dataset.
    *   **Wikipedia:** Text data for language modeling.
*   **Data Mixture Composition:** The "full mixture" consists primarily of internet-scale vision-and-language data. Sampling frequencies ensure embodied data is a minority (e.g., Mobile Manipulator is 3.1%, Language Table is 4.2%, TAMP is 1.6%).

**Example Data Sample:**
An example of a multi-modal sentence input format from the paper is:
$Q: What happened between <img1> and <img9.2>?$
Where $<img9.2>$ represents an embedding of an image injected into the sequence.

**Why Chosen:** These datasets allow the authors to evaluate whether the model can generalize from internet-scale pretraining to specific, low-data robotics tasks (data efficiency) and whether it retains general capabilities (language/VQA benchmarks).

## 5.2. Evaluation Metrics
The paper uses several metrics to quantify performance across different tasks.

### Success Rate
*   **Conceptual Definition:** Measures the percentage of times a task is completed successfully out of the total number of attempts. It focuses on the binary outcome of whether a robot achieved its goal.
*   **Mathematical Formula:**
    $$
    \text{Success Rate} = \frac{\text{Number of Successful Trials}}{\text{Total Number of Trials}} \times 100\%
    $$
*   **Symbol Explanation:** "Number of Successful Trials" counts instances where the final state matches the goal; "Total Number of Trials" is the count of all executed attempts.

### F1-Score
*   **Conceptual Definition:** A metric used to balance Precision (how many selected items are relevant) and Recall (how many relevant items are selected). It is crucial for tasks like failure detection where both false positives and false negatives matter.
*   **Mathematical Formula:**
    $$
    F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
    $$
*   **Symbol Explanation:** "Precision" is $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$; "Recall" is $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$.

### Accuracy
*   **Conceptual Definition:** The ratio of correctly predicted observations to the total observations. Commonly used for classification tasks like VQA.
*   **Mathematical Formula:**
    $$
    \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
    $$

## 5.3. Baselines
The proposed PaLM-E is compared against several representative baseline models:
1.  **SayCan (Ahn et al., 2022):** A method that interfaces LLMs with affordance functions. It relies on external oracle affordances and struggles with long-horizon planning in complex geometries.
2.  **PaLI (Chen et al., 2022):** A state-of-the-art vision-language model trained on general VQA and captioning but not trained on embodiment robot data. It serves as a zero-shot baseline for visual tasks.
3.  **CLIP variants (e.g., CLIP-FT, QT-OPT):** Used as baselines for failure detection and affordance prediction in mobile manipulation tasks.
4.  **Flamingo:** Another multimodal model used for comparison on general vision-language tasks.

    These baselines are representative because they cover the spectrum from specialized robotic controllers (SayCan) to general vision-language models (PaLI), highlighting PaLM-E's unique position as a unified embodied model.

---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results demonstrate that PaLM-E significantly outperforms baselines in embodied reasoning tasks while maintaining competence in general vision-language tasks.

### Transfer Learning Effectiveness
The most significant finding is the benefit of **transfer learning**. As shown in Figure 11 (referenced in text), training on a "full mixture" of robotics and general visual-language data provides a performance increase compared to training only on in-domain robotics data. Specifically, in the TAMP environment with only 1% of training data (320 examples), using the full mixture more than doubled the planning performance compared to single-task training.

![Figure 3: Overview of transfer learning demonstrated by PaLME: across three different robotics domains, using PaLM and ViT pretraining together with the full mixture of robotics and general visual-language data provides a significant performance increase compared to only training on the respective in-domain data. See Tab. 1, Fig. 4, Tab. 2, Tab. 4 for additional data in each domain.](images/11.jpg)
*Figure 3: Overview of transfer learning demonstrated by PaLM-E: across three different robotics domains, using PaLM and ViT pretraining together with the full mixture of robotics and general visual-language data provides a significant performance increase compared to only training on the respective in-domain data.*

### Planning Success in TAMP Environment
Table 1 compares different input representations. Using the OSRT (Object Scene Representation Transformer) input leads to the best performance in the TAMP environment, even without large-scale data. The full mixture approach improves the performance of the ViT-4B variant significantly compared to training on a single robot domain.

The following table presents the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2"></th>
<th rowspan="2">Object-cent</th>
<th rowspan="2">LLM pre-train</th>
<th colspan="4">Embodied VQA</th>
<th colspan="2">Planning</th>
</tr>
<tr>
<th>q1</th>
<th>q2</th>
<th>q3</th>
<th>q4</th>
<th>p1</th>
<th>P2</th>
</tr>
</thead>
<tbody>
<tr>
<td>SayCan (oracle afford.) (Ahn et al., 2022)</td>
<td></td>
<td>✓</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>38.7</td>
<td>33.3</td>
</tr>
<tr>
<td>PaLI (zero-shot) (Chen et al., 2022)</td>
<td></td>
<td>✓</td>
<td>-</td>
<td>0.0</td>
<td>0.0</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="9"><b>PaLM-E (ours) w/ input enc:</b></td>
</tr>
<tr>
<td>State</td>
<td>(GT)</td>
<td>X</td>
<td>-</td>
<td>99.4</td>
<td>89.8</td>
<td>90.3</td>
<td>-</td>
<td>88.3</td>
<td>45.0</td>
<td>46.1</td>
</tr>
<tr>
<td>State</td>
<td>(GT)</td>
<td>✓</td>
<td>-</td>
<td>100.0</td>
<td>96.3</td>
<td>95.1</td>
<td>-</td>
<td>93.1</td>
<td>55.9</td>
<td>49.7</td>
</tr>
<tr>
<td>ViT + TL</td>
<td>(GT)</td>
<td>✓</td>
<td>34.7</td>
<td>54.6</td>
<td>74.6</td>
<td>-</td>
<td>-</td>
<td>91.6</td>
<td>24.0</td>
<td>14.7</td>
</tr>
<tr>
<td>ViT-4B single robot</td>
<td>X</td>
<td>✓</td>
<td>-</td>
<td>45.9</td>
<td>78.4</td>
<td>-</td>
<td>-</td>
<td>92.2</td>
<td>30.6</td>
<td>32.9</td>
</tr>
<tr>
<td>ViT-4B full mixture</td>
<td>X</td>
<td>✓</td>
<td>-</td>
<td>70.7</td>
<td>93.4</td>
<td>-</td>
<td>-</td>
<td>92.1</td>
<td>74.1</td>
<td>74.6</td>
</tr>
<tr>
<td>OSRT (no VQA)</td>
<td>✓</td>
<td>✓</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>71.9</td>
<td>75.1</td>
</tr>
<tr>
<td>OSRT</td>
<td>✓</td>
<td>✓</td>
<td>99.7</td>
<td>98.2</td>
<td>100.0</td>
<td>93.7</td>
<td>82.5</td>
<td>76.2</td>
<td>-</td>
</tr>
</tbody>
</table>

*Analysis:* Table 1 highlights that the OSRT model performs best in planning (Tasks P1, P2) and VQA, surpassing SayCan and PaLI. Notably, the "ViT-4B full mixture" shows a significant jump in planning success (74.6% vs 32.9% for single robot), proving the transfer effect.

### Impact of Training Strategy
Figure 12 illustrates the impact of different training strategies on planning success rates using the PaLM-E-12B model in the TAMP environment. The chart shows that models using the "full training mixture" vastly outperform those trained on single robot data, especially when the LLM is frozen or finetuned appropriately.

![Figure 4: Planning success results in the TAMP environment $1 \\%$ data) for PaLM-E-12B, comparing of the effects of PaLM-E models (i) using the full training mixture, (ii) pre-training (ViT and PaLM), and (iii) freezing or finetuning the language model. Transfer from full mixture is particularly effective. Note that full mixture contains only $1 \\%$ of the training data (320 examples each) for the tasks evaluated here. Shown is the mean of tasks $\\mathsf { p } _ { 1 } , \\mathsf { p } _ { 2 }$ .](images/12.jpg)
*Figure 4: Planning success results in the TAMP environment 1% data) for PaLM-E-12B, comparing of the effects of PaLM-E models (i) using the full training mixture, (ii) pre-training (ViT and PaLM), and (iii) freezing or finetuning the language model. Transfer from full mixture is particularly effective.*

### Real Robot Generalization
The model demonstrates robustness in real-world scenarios. Figure 15 shows the robot guiding through long horizon tasks and achieving goals under adversarial disturbances. PaLM-E is capable of one-shot and zero-shot generalization to novel object pairs and unseen objects (e.g., a toy turtle).

![该图像是示意图，展示了在不同学习模式下，机器人如何根据语言指令完成复杂的物体操作任务，包括多次演示学习、一次学习和零次学习。通过对比可见，机器人能够在有干扰的情况下，成功实现操作目标。](images/15.jpg)
*Figure 7: PaLM-E guiding a real robot through long horizon tasks to adversarial disturbances. We find evidence that PaLM-E is capable of one-shot and zero shot generalization.*

## 6.2. Data Presentation (Tables)

### Table 2: Planning Tasks in Simulated Environment
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th colspan="7">Zero-shot Baselines</th>
<th colspan="3">Task 1</th>
<th colspan="3">Task 2</th>
<th colspan="3">Task 3</th>
</tr>
<tr>
<th colspan="5">SayCan (oracle afford.) (Ahn et al., 2022) PaLI (Chen et al., 2022) trained</th>
<th colspan="3">0.0 0.0</th>
<th colspan="5">-</th>
<th colspan="2">- -</th>
</tr>
<tr>
<th colspan="5">from scratch</th>
<th colspan="3">LLM</th>
<th colspan="5"># Demos</th>
</tr>
<tr>
<th>PL-E-</th>
<th>on</th>
<th>scratch</th>
<th>pretrain</th>
<th>frozen</th>
<th>finetune</th>
<th>10</th>
<th>20</th>
<th>40</th>
<th>10</th>
<th>20</th>
<th>40</th>
<th>10</th>
<th>20</th>
<th>80</th>
</tr>
<tr>
<td>12B</td>
<td>Single robot</td>
<td>✓</td>
<td>X</td>
<td>n/a</td>
<td>✓</td>
<td>20.0</td>
<td>30.0</td>
<td>50.0</td>
<td>2.5</td>
<td>6.3</td>
<td>2.5</td>
<td>11.3</td>
<td>16.9</td>
<td>28.3</td>
</tr>
<tr>
<td>12B</td>
<td>Full mixture</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
<td>X</td>
<td>-</td>
<td>-</td>
<td>20.0</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>36.3</td>
<td>-</td>
<td>29.4</td>
</tr>
<tr>
<td>12B</td>
<td>Full mixture</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>-</td>
<td>-</td>
<td>80.0</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>57.5</td>
<td>-</td>
<td>50.0</td>
</tr>
<tr>
<td>12B</td>
<td>Full mixture</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>✓</td>
<td>70.0</td>
<td>80.0</td>
<td>80.0</td>
<td>31.3</td>
<td>58.8</td>
<td>-</td>
<td>58.8</td>
<td>57.5</td>
<td>56.3</td>
</tr>
<tr>
<td>84B</td>
<td>Full mixture</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>-</td>
<td>-</td>
<td>90.0</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>53.8</td>
<td>-</td>
<td>64.4</td>
</tr>
</tbody>
</table>

*Task Definitions:*
*   **Task 1:** Push block closest to top right corner to other block of same color.
*   **Task 2:** Sort blocks by colors into corners.
*   **Task 3:** Push all blocks from left/right side together without bringing over blocks from the opposite side.

### Table 5: Performance on General Visual-Language Tasks
The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th colspan="2">VQAv2 test-dev test-std</th>
<th>OK-VQA val</th>
<th>COCO Karpathy test</th>
</tr>
<tr>
<td colspan="5"><b>Generalist (one model)</b></td>
</tr>
<tr>
<td>PaLM-E-12B</td>
<td>76.2</td>
<td>-</td>
<td>55.5</td>
<td>135.0</td>
</tr>
<tr>
<td>PaLM-E-562B</td>
<td>80.0</td>
<td>-</td>
<td>66.1</td>
<td>138.7</td>
</tr>
<tr>
<td colspan="5"><b>Task-specific finetuned models</b></td>
</tr>
<tr>
<td>Flamingo (Alayrac et al., 2022)</td>
<td>82.0</td>
<td>82.1</td>
<td>57.8†</td>
<td>138.1</td>
</tr>
<tr>
<td>PaLI (Chen et al., 2022)</td>
<td>84.3</td>
<td>84.3</td>
<td>64.5</td>
<td>149.1</td>
</tr>
<tr>
<td>PaLM-E-12B</td>
<td>77.7</td>
<td>77.9</td>
<td>60.1</td>
<td>136.0</td>
</tr>
<tr>
<td>PaLM-E-66B</td>
<td>-</td>
<td>-</td>
<td>62.9</td>
<td>-</td>
</tr>
<tr>
<td>PaLM-E-84B</td>
<td>80.5</td>
<td>-</td>
<td>63.3</td>
<td>138.0</td>
</tr>
<tr>
<td colspan="5"><b>Generalist (one model), with frozen LLM</b></td>
</tr>
<tr>
<td>(Tsimpoukelli et al., 2021)</td>
<td>48.4</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PaLM-E-12B frozen</td>
<td>70.3</td>
<td>-</td>
<td>51.5</td>
<td>128.0</td>
</tr>
</tbody>
</table>

*Note: † is 32-shot on OK-VQA (not finetuned).*

### Table 8: Natural Language Generation and Understanding Results
The following are the results from Table 8 of the original paper regarding NLU/NLG performance across scales.

<table>
<thead>
<tr>
<th>1-shot evals</th>
<th>PaLM-8B</th>
<th>PaLM-E-12B (unfrozen)</th>
<th>PaLM-62B</th>
<th>PaLM-E-84B (unfrozen)</th>
<th>PaLM-540B</th>
<th>PaLM-E-562B (unfrozen)</th>
<th>Category</th>
</tr>
<tr>
<td>TriviaQA (wiki) (EM)</td>
<td>48.5</td>
<td>10.1</td>
<td>72.7</td>
<td>31.8</td>
<td>81.4</td>
<td>74.6</td>
<td>NLG</td>
</tr>
<tr>
<td>Natural Questions (EM)</td>
<td>10.6</td>
<td>1.6</td>
<td>23.1</td>
<td>7.6</td>
<td>29.3</td>
<td>27.2</td>
<td>NLG</td>
</tr>
<tr>
<td>HellaSwag</td>
<td>68.2</td>
<td>48.4</td>
<td>79.7</td>
<td>75.3</td>
<td>83.6</td>
<td>83.5</td>
<td>NLU</td>
</tr>
<tr>
<td>Avg NLU</td>
<td>64.7</td>
<td>55.0</td>
<td>72.3</td>
<td>69.2</td>
<td>78.2</td>
<td>78.5</td>
<td></td>
</tr>
<tr>
<td>Avg NLG</td>
<td>32.4</td>
<td>4.1</td>
<td>47.8</td>
<td>18.4</td>
<td>53.8</td>
<td>51.7</td>
<td></td>
</tr>
<tr>
<td>NLU delta (%, relative)</td>
<td></td>
<td>-15.0%</td>
<td></td>
<td>-61.6%</td>
<td></td>
<td>+0.4%</td>
<td></td>
</tr>
<tr>
<td>NLG delta (%, relative)</td>
<td></td>
<td>-87.3%</td>
<td></td>
<td>-4.3%</td>
<td></td>
<td>-3.8%</td>
<td></td>
</tr>
</tbody>
</table>

*Analysis:* This table reveals the phenomenon of **Catastrophic Forgetting**. For the smallest model (PaLM-E-12B), NLG performance degraded by 87.3%. However, for the largest model (PaLM-E-562B), the degradation was only 3.8%, and NLU actually improved slightly (+0.4%). This confirms the finding that scaling reduces catastrophic forgetting.

![Figure 6: Results on general language tasks ( $\\mathbf { N L G } =$ natural language generation): increasing scale leads to less catastrophic forgetting between a corresponding PaLM-E model and its inherited PaLM model. See full suite of tasks and results in Tab. 8.](images/14.jpg)
*Figure 6: Results on general language tasks (NLG = natural language generation): increasing scale leads to less catastrophic forgetting between a corresponding PaLM-E model and its inherited PaLM model.*

## 6.3. Ablation Studies / Parameter Analysis
The paper includes extensive ablation studies regarding input representations and training strategies.
*   **Freezing vs. Finetuning:** Experiments showed that freezing the LLM and only training encoders is viable but sometimes struggled for robotics tasks compared to full training (see Table 2, Row 4 vs Row 2).
*   **Pre-training:** Pre-training the LLM and ViT before joint fine-tuning consistently improves performance over training from scratch (see Table 1).
*   **Representation Choice:** OSRT generally outperformed ViT variants in planning tasks due to its 3D-aware object representations, though ViT performed better when large-scale pre-training data was available (Full Mixture).
*   **Scale:** Increasing model parameters from 12B to 562B significantly mitigates performance degradation on language tasks while improving embodied reasoning capabilities.

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper proposes **PaLM-E**, an embodied multimodal language model that integrates continuous sensor modalities (images, state estimates) directly into the embedding space of a pre-trained LLM. Key findings include:
1.  **Embodied Capability:** PaLM-E successfully solves robotic planning and manipulation tasks where traditional VLMs fail due to lack of grounding.
2.  **Positive Transfer:** Training on a mixture of internet-scale data and robotics data significantly boosts performance on robotics tasks, enabling high data efficiency.
3.  **Scalability:** Scaling the model to 562B parameters allows it to retain general language and vision capabilities (state-of-the-art on OK-VQA) while becoming a competent embodied agent, largely avoiding catastrophic forgetting.
4.  **Emergent Skills:** The model exhibits zero-shot multimodal chain-of-thought reasoning and multi-image reasoning despite being trained on single-image prompts.

## 7.2. Limitations & Future Work
Despite the successes, several limitations exist:
*   **Low-Level Policy Dependency:** The effectiveness of PaLM-E is contingent on the availability of low-level policies that can execute the high-level plans generated. The model does not learn the low-level control itself.
*   **Sim-to-Real Gap:** While tested on real robots, the core planning data often comes from simulation (TAMP) or specific real-world setups. Generalization to completely unstructured environments remains challenging.
*   **Data Scarcity:** Although transfer helps, robotics data is still scarce compared to text/vision data. The model relies heavily on the 90% internet-scale data to function well on the 10% robotics data.
*   **Future Directions:** Combining neural scene representations (like OSRT) with large-scale visual data could further improve data efficiency. Additionally, exploring larger model scales may reveal further emergent behaviors in embodied reasoning.

## 7.3. Personal Insights & Critique
**Inspirations:** The concept of injecting continuous observations into the language embedding space as "tokens" is elegant. It avoids the complexity of building separate fusion modules and leverages the LLM's inherent attention mechanisms for cross-modal reasoning. The finding that **scaling reduces catastrophic forgetting** is profound; it suggests that future embodied agents should prioritize model size over task-specific fine-tuning to preserve general knowledge.

**Potential Issues:** The reliance on "entity referrals" (labeling objects as `obj-1`, `obj-2`) simplifies the reference problem but requires ground-truth masks or segmentation in many cases (except OSRT). In dynamic real-world settings, automatically identifying stable entities for referencing could be difficult. Furthermore, generating plans in text requires a downstream system to parse and translate text commands into actions reliably. If the low-level policy fails to interpret a valid text command, the entire planning loop breaks.

**Applicability:** The methodology is highly transferable. Any domain requiring reasoning about continuous physical states guided by language (e.g., autonomous driving, smart home automation, industrial assembly) could benefit from this architecture. The "full mixture" training strategy suggests a path forward for developing generalist robots that don't require millions of hours of domain-specific demonstration data for every new skill.