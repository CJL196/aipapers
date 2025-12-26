# 1. Bibliographic Information

## 1.1. Title
VL-JEPA: Joint Embedding Predictive Architecture for Vision-language

## 1.2. Authors
The authors are Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Allen Bolourchi, Yann LeCun, and Pascale Fung. They represent Meta FAIR (Fundamental AI Research), the Hong Kong University of Science and Technology (HKUST), Sorbonne Université, and New York University (NYU). Yann LeCun is a pioneer in deep learning and the Chief AI Scientist at Meta.

## 1.3. Journal/Conference
The paper was published on arXiv (a prominent preprint server for rapid dissemination of research) on December 11, 2025. Given the authorship and the novelty of applying the `Joint Embedding Predictive Architecture (JEPA)` to vision-language tasks, it is highly likely to be submitted to top-tier computer vision or machine learning conferences like CVPR, ICCV, or NeurIPS.

## 1.4. Publication Year
2025

## 1.5. Abstract
The paper introduces `VL-JEPA`, a vision-language model that moves away from traditional autoregressive token generation. Instead of predicting tokens one by one, it predicts continuous embeddings of target texts in an abstract representation space. This allows the model to focus on semantics rather than surface-level linguistic variability (like word choice or style). In controlled experiments, `VL-JEPA` achieved stronger performance than standard `Vision-Language Models (VLMs)` while using 50% fewer trainable parameters. It also features `selective decoding`, reducing decoding operations by 2.85x. The model excels at video classification, retrieval, and visual question answering (VQA) with 1.6B parameters.

## 1.6. Original Source Link
*   **Original Source:** [https://arxiv.org/abs/2512.10942](https://arxiv.org/abs/2512.10942)
*   **PDF Link:** [https://arxiv.org/pdf/2512.10942v1.pdf](https://arxiv.org/pdf/2512.10942v1.pdf)
*   **Status:** Preprint (under review or recently released).

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The current dominant paradigm for `Vision-Language Models (VLMs)` is **generative**. These models take an image or video and a question, then generate a text response token by token (autoregressively). While effective, this approach has two major drawbacks:
1.  **Computational Waste:** Generative models spend significant energy learning "surface" details—such as specific phrasing, grammar, and synonyms—that don't necessarily change the underlying meaning or correctness of an answer.
2.  **High Latency:** For real-time applications (like a robot describing a live video stream), waiting for a model to generate text word-by-word is too slow.

    The core problem is that these models work in the "data space" (pixels and words) rather than the "latent space" (abstract concepts). The paper aims to solve this by applying the `Joint Embedding Predictive Architecture (JEPA)`, which predicts high-level semantic embeddings instead of raw data.

## 2.2. Main Contributions / Findings
*   **VL-JEPA Architecture:** The first non-generative model designed for general-domain vision-language tasks.
*   **Latent Space Prediction:** Demonstrates that predicting abstract embeddings is more efficient and accurate than predicting discrete tokens, especially in data-constrained environments.
*   **Efficiency:** Achieves better performance than traditional `VLMs` with 50% fewer trainable parameters.
*   **Selective Decoding:** Introduced a method to monitor the "semantic stream" of embeddings and only invoke the expensive text decoder when a significant change in meaning occurs.
*   **Generalist Capability:** A single model that outperforms specialized models in video classification and retrieval while matching large generative models in `VQA`.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Joint Embedding Predictive Architecture (JEPA)
`JEPA` is a self-supervised learning paradigm proposed by Yann LeCun. Unlike traditional models that try to reconstruct missing pixels (like `Masked Autoencoders`) or predict the next token (like `GPT`), `JEPA` tries to predict the **representation** of a target part of the data from the **representation** of a context part. 
*   **Context:** What the model sees (e.g., an image or the start of a video).
*   **Target:** What the model needs to predict (e.g., what happens next or a textual description).
*   **Benefit:** By working in "embedding space," the model can ignore noise. For example, if predicting a video of a tree in the wind, it doesn't need to predict the exact position of every leaf (noise); it only needs to predict "the tree is swaying" (semantics).

### 3.1.2. Autoregressive vs. Non-Autoregressive
*   **Autoregressive:** The model predicts token $t$ based on tokens `1` to `t-1`. It is a serial process ($A \to B \to C$).
*   **Non-Autoregressive:** The model predicts the entire target at once or in a single forward pass. `VL-JEPA` predicts a single vector representing the whole answer, which can then be decoded in one go.

### 3.1.3. Embedding Space
An `embedding` is a numerical vector (a list of numbers) that represents a piece of data (like a word or image) such that similar items are mathematically "close" to each other in a multi-dimensional space.

## 3.2. Previous Works
The authors build upon several key precursors:
1.  **V-JEPA:** A previous model that used `JEPA` to learn video representations by predicting the latent features of masked video blocks.
2.  **CLIP (Contrastive Language-Image Pre-training):** A model that uses two separate encoders (one for images, one for text) and trains them to produce similar embeddings for matching pairs using the `InfoNCE` loss.
3.  **Generative VLMs (e.g., LLaVA, Flamingo):** These connect a vision encoder to a `Large Language Model (LLM)` and use "next-token prediction" to generate text.

## 3.3. Technological Evolution
The field has moved from simple classifiers to `CLIP`-style alignment (matching images to text) to generative `VLMs` (chatting about images). `VL-JEPA` represents a new branch: **Predictive Latent VLMs**, combining the semantic alignment of `CLIP` with the task flexibility of generative models.

---

# 4. Methodology

## 4.1. Principles
The core intuition of `VL-JEPA` is that the "answer" to a visual query exists as a semantic concept before it is turned into words. By training the model to reach that semantic point directly, we skip the hard work of learning "how to speak" during the main visual reasoning phase.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Model Architecture
The following figure (Figure 1 from the original paper) shows the system architecture:

![Figure 1. VL-JEPA model architecture](images/1.jpg)
*该图像是VL-JEPA模型架构示意图。它展示了视觉输入通过X-Encoder进行编码，生成的表示和文本查询一起输入到预测器，最终输出文本目标的表示。模型还包含Y-Encoder和Y-Decoder，用于生成相应的文本输出。其中损失函数用L表示。*

The model consists of four distinct modules:

1.  **X-Encoder ($X_V \mapsto S_V$):** This maps the visual input $X_V$ (images or video frames) into a sequence of visual embeddings $S_V$. The authors use a frozen `V-JEPA 2 ViT-L` (304M parameters).
2.  **Predictor ($\langle S_V, X_Q \rangle \mapsto \hat{S}_Y$):** This is the "brain." It takes the visual embeddings $S_V$ and a textual query $X_Q$ (the prompt/question). It outputs a predicted target embedding $\hat{S}_Y$. It uses 8 Transformer layers from `Llama-3.2-1B`.
3.  **Y-Encoder ($Y \mapsto S_Y$):** This maps the ground-truth text $Y$ into the "target" embedding $S_Y$. This is used during training to provide the signal for the Predictor. They use `EmbeddingGemma-300M`.
4.  **Y-Decoder ($\hat{S}_Y \mapsto \hat{Y}$):** A lightweight module used only at inference time to turn the predicted embedding back into human-readable text.

### 4.2.2. Training Objective and Formulas
The training is performed in two stages. Unlike generative models that use `Cross-Entropy` loss on tokens, `VL-JEPA` uses an embedding-based loss.

**Stage 1: Pretraining.** The model is trained on massive captioning data to align vision and text. The authors use a bi-directional `InfoNCE` (Information Noise Contrastive Estimation) loss.

The `InfoNCE` loss is used to maximize the similarity between the predicted embedding $\hat{S}_Y$ and the actual target embedding $S_Y$, while minimizing similarity with "negative" samples (other texts in the batch). The simplified alignment term is:

$$
\mathcal{L}_{align} = D(\hat{S}_Y, S_Y)
$$

Where:
*   $\hat{S}_Y$ is the predicted embedding from the **Predictor**.
*   $S_Y$ is the actual target embedding from the **Y-Encoder**.
*   $D$ is a distance metric (usually based on cosine similarity).

    To avoid "representation collapse" (where the model predicts the same constant vector for everything), the `InfoNCE` loss includes a uniformity regularization term that pushes different embeddings in a batch apart. The total loss $\mathcal{L}_{VL-JEPA}$ is calculated as:

$$
\mathcal{L}_{VL-JEPA} = -\log \frac{\exp(\mathrm{sim}(\hat{S}_{Y,i}, S_{Y,i}) / \tau)}{\sum_{j=1}^{N} \exp(\mathrm{sim}(\hat{S}_{Y,i}, S_{Y,j}) / \tau)}
$$

Where:
*   $\mathrm{sim}(A, B)$ is the cosine similarity between vectors $A$ and $B$.
*   $\tau$ is a temperature hyper-parameter that controls the sharpness of the distribution.
*   $N$ is the batch size.
*   $i$ and $j$ are indices for samples in the batch.

### 4.2.3. Multi-tasking Logic
The following figure (Figure 2 from the paper) illustrates how the single architecture handles multiple tasks:

![Figure 2. Left: VL-JEPA Architecture. It learns to predict the target embedding `S _ { Y }` , instead of reconstructing the raw target Yin token space as in retrieval tasks using a single unified model architecture.](images/2.jpg)
*该图像是 VL-JEPA 的架构示意图，展示了模型如何通过预测目标嵌入 $S_{Y}$ 来实现任务，而不是在检索任务中重构原始目标。图中还说明了视觉到文本生成、区分性 VQA 和分类及文本到视觉检索的选择性解码过程。*

*   **Generation (VQA/Captioning):** Predict $\hat{S}_Y \to$ Decode to text.
*   **Classification:** Predict $\hat{S}_Y \to$ Compare to candidate label embeddings $\to$ Pick the closest.
*   **Retrieval:** Use a query to predict an embedding $\to$ Match against a database of video embeddings.

### 4.2.4. Selective Decoding
For long videos, `VL-JEPA` monitors the "stream" of predicted embeddings $\hat{S}_Y$ over time. Because it's non-autoregressive, this stream is computationally cheap to produce.
The model calculates the variance of the embeddings in a sliding window. If the variance is low, the scene is semantically stable, and no new decoding is needed. If a "semantic shift" (high variance) is detected, the `Y-Decoder` is triggered.

---

# 5. Experimental Setup

## 5.1. Datasets
The authors used a massive collection of data across two stages:
1.  **Pretraining:**
    *   **Images:** `Datacomp`, `YFCC-100M`, `PLM-Image-Auto` (approx. 2B samples seen).
    *   **Videos:** `Ego4D`, `HowTo100M` (Action100M captions).
2.  **SFT (Supervised Finetuning):**
    *   A mixture of 25M `VQA` samples, 2.8M caption samples, and 1.8M classification samples.
3.  **Evaluation Benchmarks:**
    *   **Classification:** `Kinetics-400`, `Something-Something v2 (SSv2)`, `Epic-Kitchens 100 (EK-100)`.
    *   **VQA:** `GQA` (reasoning), `TallyQA` (counting), $POPE/POPEv2$ (hallucination).

## 5.2. Evaluation Metrics

### 5.2.1. Recall@1 (R@1)
*   **Conceptual Definition:** Used in retrieval tasks. It measures if the correct item is the very first result returned by the model.
*   **Mathematical Formula:**
    $$
    R@1 = \frac{1}{Q} \sum_{q=1}^{Q} \mathbb{I}(\mathrm{rank}_q = 1)
    $$
*   **Symbol Explanation:** $Q$ is the total number of queries; $\mathbb{I}(\cdot)$ is the indicator function (1 if true, 0 if false); $\mathrm{rank}_q$ is the rank of the ground-truth item for query $q$.

### 5.2.2. Accuracy (Top-1)
*   **Conceptual Definition:** The percentage of times the model's highest-probability prediction matches the ground-truth label.
*   **Mathematical Formula:**
    $$
    \mathrm{Acc} = \frac{\text{Number of correct predictions}}{\text{Total number of samples}}
    $$

### 5.2.3. CIDEr (Consensus-based Image Description Evaluation)
*   **Conceptual Definition:** Measures how similar a generated caption is to a set of human reference captions using `TF-IDF` weighted n-gram overlap.
*   **Mathematical Formula (Standardized):**
    $$
    \mathrm{CIDEr}(c_i, S_i) = \frac{1}{M} \sum_{j=1}^{M} \frac{\mathbf{g}^n(c_i) \cdot \mathbf{g}^n(s_{ij})}{\|\mathbf{g}^n(c_i)\| \|\mathbf{g}^n(s_{ij})\|}
    $$
*   **Symbol Explanation:** $c_i$ is the generated caption; $S_i$ is the set of reference captions; $\mathbf{g}^n$ is a vector of `TF-IDF` weights for n-grams of length $n$; $M$ is the number of references.

## 5.3. Baselines
*   **CLIP / SigLIP2:** Standard joint-embedding models (encoders only).
*   **Perception Encoder (PE-Core):** A recent state-of-the-art visual embedding model.
*   **Generative VLMs:** `InstructBLIP`, `Qwen-VL`, `LLaVA-1.5`, `SmolVLM`.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
*   **Efficiency:** `VL-JEPA` achieved higher accuracy than `VLM` baselines while having 50% fewer parameters and seeing significantly less data (2B samples vs. 86B for some baselines).
*   **Video Mastery:** The model performed exceptionally well on motion-centric datasets like `SSv2` and `EgoExo4D`, likely due to the strengths of the `V-JEPA` vision backbone.
*   **Selective Decoding:** The model reduced decoding operations by 2.85x while maintaining the same `CIDEr` score.

## 6.2. Data Presentation (Tables)

The following are the results from Table 1 of the original paper, showing classification and retrieval performance:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2"># Params</th>
<th rowspan="2"># Samples Seen</th>
<th colspan="4">Classification (Zero-Shot)</th>
<th colspan="4">Retrieval (Zero-Shot)</th>
</tr>
<tr>
<th>K400</th>
<th>SSv2</th>
<th>EK-100</th>
<th>Avg (8)</th>
<th>MSR-VTT</th>
<th>ActivityNet</th>
<th>EK-100</th>
<th>Avg (8)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP ViT-L</td>
<td>389M</td>
<td>12.8B</td>
<td>58.3</td>
<td>2.4</td>
<td>14.7</td>
<td>35.3</td>
<td>30.7</td>
<td>23.4</td>
<td>7.9</td>
<td>35.9</td>
</tr>
<tr>
<td>SigLIP2 ViT-g</td>
<td>1.9B</td>
<td>40B</td>
<td>68.0</td>
<td>6.1</td>
<td>26.0</td>
<td>47.5</td>
<td>38.9</td>
<td>33.9</td>
<td>22.2</td>
<td>43.4</td>
</tr>
<tr>
<td>PE-Core ViT-G</td>
<td>2.3B</td>
<td>86B</td>
<td>76.4</td>
<td>13.0</td>
<td>29.0</td>
<td>58.1</td>
<td>44.5</td>
<td>49.1</td>
<td>26.0</td>
<td>51.6</td>
</tr>
<tr>
<td><b>VL-JEPA Base</b></td>
<td>1.6B</td>
<td>2.0B</td>
<td>57.8</td>
<td>21.1</td>
<td>39.8</td>
<td>46.4</td>
<td>49.2</td>
<td>55.4</td>
<td>78.2</td>
<td>58.4</td>
</tr>
<tr>
<td><b>VL-JEPA SFT</b></td>
<td>1.6B</td>
<td>2.5B</td>
<td>81.4</td>
<td>59.5</td>
<td>60.3</td>
<td>70.7</td>
<td>49.1</td>
<td>53.8</td>
<td>81.1</td>
<td>59.5</td>
</tr>
</tbody>
</table>

*Note: $Avg (8)$ refers to the average over 8 datasets in each category.*

The following are VQA results from Section 4.2, comparing `VL-JEPA SFT` to standard VLMs:

<table>
<thead>
<tr>
<th>Model</th>
<th>GQA (Acc)</th>
<th>TallyQA (Acc)</th>
<th>POPE (Acc)</th>
</tr>
</thead>
<tbody>
<tr>
<td>InstructBLIP (Vicuna-13B)</td>
<td>49.5</td>
<td>68.0</td>
<td>79.0</td>
</tr>
<tr>
<td>Qwen-VL-7B</td>
<td>59.3</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>LLaVA-1.5-7B</td>
<td>62.0</td>
<td>71.9</td>
<td>85.9</td>
</tr>
<tr>
<td><b>VL-JEPA SFT (1.6B)</b></td>
<td>60.8</td>
<td>67.4</td>
<td>84.2</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The authors tested several configurations (Table 5 in the paper):
*   **Pretraining:** Removing the caption-based pretraining stage dropped classification accuracy from 49.0 to 27.3.
*   **Loss Function:** `InfoNCE` was superior to simple `Cosine` or `L2` loss for classification and retrieval.
*   **Predictor Size:** Increasing the number of layers in the Predictor significantly improved `VQA` performance.

    The following figure (Figure 3) shows the sample efficiency comparison:

    ![该图像是一个图表，展示了VL-JEPA与传统视觉语言模型（VLM）在零样本视频标题生成和视频分类任务中的性能比较。左侧图表中，VL-JEPA在CIDEr得分上优于VLM，而右侧图表则显示VL-JEPA的可训练参数和平均推理时间成本更低。在文本解码上，VL-JEPA的平均时间为203毫秒。](images/3.jpg)
    *该图像是一个图表，展示了VL-JEPA与传统视觉语言模型（VLM）在零样本视频标题生成和视频分类任务中的性能比较。左侧图表中，VL-JEPA在CIDEr得分上优于VLM，而右侧图表则显示VL-JEPA的可训练参数和平均推理时间成本更低。在文本解码上，VL-JEPA的平均时间为203毫秒。*

As seen in Figure 3, `VL-JEPA` reaches higher performance much faster (with fewer samples) than the `VLM` baseline.

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
`VL-JEPA` successfully demonstrates that the `Joint Embedding Predictive Architecture` is a viable and highly efficient alternative to autoregressive generative models for vision-language tasks. By predicting semantic embeddings rather than tokens, the model achieves better performance with fewer parameters, excels at video-based tasks, and supports real-time streaming via selective decoding. It bridges the gap between discriminative models (like `CLIP`) and generative ones.

## 7.2. Limitations & Future Work
*   **Reasoning and Agents:** The authors acknowledge they haven't yet tested `VL-JEPA` on complex multi-step reasoning, tool use, or agentic behaviors where large `VLMs` currently excel.
*   **Scaling:** While 1.6B parameters is efficient, they haven't yet explored the limits of scaling this architecture to 70B+ parameters.
*   **Latent Reasoning:** Future work could involve "reasoning in latent space"—performing logical steps on embeddings before ever decoding them into text.

## 7.3. Personal Insights & Critique
This paper is a significant shift in thinking. The move from "generating text" to "predicting meaning" aligns with theories of how biological brains might work (not predicting every word, but the concept of what to say).
*   **Inspiration:** The `selective decoding` mechanism is brilliant for robotics and wearable tech (like smart glasses), where battery life and latency are the biggest constraints.
*   **Potential Issue:** The model relies heavily on the quality of the `Y-Encoder`. If the text embedding model (Gemma) has biases or lacks specific domain knowledge, `VL-JEPA` will inherit those weaknesses without the ability to "re-learn" the language space easily.
*   **Assumption:** The paper assumes that `InfoNCE` is the best loss; however, other `JEPA`-specific losses like `VICReg` (which don't require negative samples) might offer even more stability in future iterations.