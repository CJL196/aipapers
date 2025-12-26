# 1. Bibliographic Information

## 1.1. Title
OpenVLA: An Open-Source Vision-Language-Action Model

The title clearly states the paper's central topic: the introduction of OpenVLA, a model designed for robotics that is open-source and operates by integrating vision, language, and action capabilities.

## 1.2. Authors
Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn.

The authors are affiliated with several leading institutions in AI and robotics, including Stanford University, Google DeepMind, and the Massachusetts Institute ofto Technology (MIT). This roster includes prominent researchers who have made significant contributions to the fields of robot learning, foundation models, and machine learning, indicating a high-profile and impactful research effort.

## 1.3. Journal/Conference
The paper was submitted to arXiv, an open-access repository of electronic preprints. As of its publication date, it has not been formally published in a peer-reviewed journal or conference. However, arXiv is the standard platform in the machine learning and robotics communities for rapidly disseminating cutting-edge research.

## 1.4. Publication Year
2024 (Published on arXiv on June 13, 2024).

## 1.5. Abstract
The abstract introduces the concept of Vision-Language-Action (VLA) models, which are large policies pretrained on both internet-scale data and robot demonstrations. These models hold the promise of being fine-tuned for new robotic skills rather than training them from scratch. The authors identify two main challenges hindering their adoption: 1) existing VLAs are predominantly closed-source, and 2) there is a lack of research on how to efficiently fine-tune them.

To address these challenges, the paper introduces `OpenVLA`, a 7-billion-parameter open-source VLA. It is trained on 970,000 real-world robot demonstrations. The model architecture is based on a `Llama 2` language model and a visual encoder that fuses features from `DINOv2` and `SigLIP`.

The key results are:
1.  `OpenVLA` demonstrates superior performance in generalist manipulation, outperforming the much larger 55B-parameter `RT-2-X` model by 16.5% in absolute task success rate, despite being 7 times smaller.
2.  `OpenVLA` can be effectively fine-tuned for new tasks, outperforming from-scratch methods like `Diffusion Policy` by 20.4%.
3.  The paper shows that `OpenVLA` can be fine-tuned on consumer-grade GPUs using `LoRA` (low-rank adaptation) and served efficiently using quantization without performance degradation.

    Finally, the authors announce the release of the model checkpoints, fine-tuning notebooks, and their PyTorch codebase to foster further research.

## 1.6. Original Source Link
*   **Original Source Link:** [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
*   **PDF Link:** [https://arxiv.org/pdf/2406.09246v3.pdf](https://arxiv.org/pdf/2406.09246v3.pdf)
*   **Publication Status:** This is a preprint available on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
A fundamental challenge in robotics is **generalization**: training a robot to perform a task and having it successfully execute that task under new conditions (e.g., with different objects, lighting, or backgrounds) or follow entirely new instructions. Traditional robot policies, trained from scratch on specific tasks, often fail to generalize robustly.

In parallel, the fields of computer vision and natural language processing have seen the rise of **foundation models** (e.g., `CLIP`, `Llama 2`). These models are pretrained on vast amounts of internet data and exhibit remarkable generalization capabilities. A promising direction in robotics is to leverage these powerful models to create more general-purpose robot policies.

This has led to the development of **Vision-Language-Action (VLA)** models, which fine-tune a pretrained Vision-Language Model (VLM) to output robot actions. By doing so, they inherit the rich understanding of language and visual concepts from the foundation model. However, the widespread adoption of VLAs has been blocked by two key issues:

1.  **Accessibility:** The most powerful VLAs, like Google's `RT-2-X`, have been closed-source and proprietary. This prevents the broader research community from building upon them, analyzing their properties, or deploying them in new applications.
2.  **Adaptability:** Prior work has focused on the "zero-shot" capabilities of these large models but has not provided a clear methodology for efficiently **fine-tuning** them for new robots, tasks, or environments. Fine-tuning massive models typically requires significant computational resources (e.g., large GPU clusters), which is a major barrier for many academic labs and smaller companies.

    This paper's entry point is to directly tackle these two problems by creating a powerful, fully **open-source** VLA and demonstrating a practical, computationally efficient workflow for adapting it to new settings.

## 2.2. Main Contributions / Findings
The paper makes several key contributions that significantly advance the field of robot learning:

1.  **Introduction of OpenVLA:** The authors present `OpenVLA`, a 7-billion-parameter VLA that is fully open-source. It combines a strong `Llama 2` language model with a novel dual-stream vision encoder (`DINOv2` + `SigLIP`) and is pretrained on a massive, diverse dataset of 970k real-world robot demonstrations.

2.  **State-of-the-Art Performance:** `OpenVLA` establishes a new state-of-the-art for generalist robot manipulation. Despite being 7x smaller, it outperforms the 55B-parameter closed-source `RT-2-X` model by a significant margin (16.5% absolute success rate) on a comprehensive suite of 29 tasks across multiple robot types.

3.  **Pioneering Efficient Fine-Tuning for VLAs:** This is the first work to systematically investigate and validate methods for efficiently adapting VLAs. The authors show that **Low-Rank Adaptation (`LoRA`)** can match the performance of full model fine-tuning while only training 1.4% of the parameters. This makes it possible to adapt `OpenVLA` on a single consumer-grade GPU, drastically lowering the barrier to entry.

4.  **Demonstrating Efficient Inference:** The paper shows that **4-bit quantization** can be applied to `OpenVLA` at inference time, reducing its memory footprint by more than half (to just 7.0 GB) without any loss in task performance. This enables the deployment of `OpenVLA` on more accessible, lower-cost hardware.

5.  **Comprehensive Open-Source Release:** Beyond just the model weights, the authors release their entire ecosystem: the PyTorch codebase for training VLAs at scale, ready-to-use fine-tuning notebooks, and deployment tools. This comprehensive release is designed to empower the community to conduct further research on VLAs.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Vision-Language Models (VLMs)
A Vision-Language Model (VLM) is a type of AI model designed to understand and process information from both images and text simultaneously. It can perform tasks that require multimodal understanding, such as:
*   **Visual Question Answering (VQA):** Answering a text-based question about an image (e.g., "What color is the car in the image?").
*   **Image Captioning:** Generating a descriptive text sentence for an image.

    Most modern VLMs consist of three core components:
1.  **Vision Encoder:** A pretrained model (often a Vision Transformer, or ViT) that processes an image and converts it into a set of numerical representations (embeddings). It typically divides the image into patches and generates an embedding for each patch.
2.  **Language Model:** A pretrained Large Language Model (LLM), like `GPT` or `Llama`, that processes text.
3.  **Projector:** A small neural network (like an MLP) that translates the vision embeddings into the same "language" as the text embeddings, allowing the LLM to process them.

    The common approach, used in `OpenVLA`, is the **"patch-as-token"** method. The vision encoder's output patch embeddings are projected and then treated as if they were text tokens, prepended to the actual text prompt. The LLM then processes this combined sequence of "visual tokens" and "text tokens."

### 3.1.2. Key Pretrained Components of OpenVLA
*   **Llama 2:** A family of open-source Large Language Models (LLMs) developed by Meta AI. `OpenVLA` uses the 7-billion-parameter version. An LLM is a massive neural network trained on a huge corpus of text data to predict the next word (or token) in a sequence. This ability allows it to generate human-like text, answer questions, and follow instructions.
*   **DINOv2:** A vision model trained using **self-supervision**, meaning it learns about the visual world without requiring human-labeled text descriptions. It is particularly good at learning **fine-grained spatial and geometric features**, such as object boundaries, textures, and parts. This is highly valuable for robotics, which requires precise spatial reasoning.
*   **SigLIP (Sigmoid Loss for Language-Image Pre-training):** A vision model trained on image-text pairs from the web, similar to `CLIP`. It excels at learning **high-level semantic concepts** (e.g., recognizing an "apple" regardless of its color or position). By using a sigmoid loss instead of the standard softmax, it learns to associate an image with multiple concepts more effectively.

    By fusing `DINOv2` and `SigLIP`, `OpenVLA`'s vision system gets the best of both worlds: the precise spatial understanding of `DINOv2` and the rich semantic understanding of `SigLIP`.

### 3.1.3. Parameter-Efficient Fine-Tuning (PEFT) and LoRA
Fine-tuning a foundation model with billions of parameters on a new task requires updating all of its weights, which is computationally expensive and memory-intensive. **Parameter-Efficient Fine-Tuning (PEFT)** refers to a set of techniques that freeze the vast majority of the pretrained model's parameters and only train a small number of new or adapted parameters.

**Low-Rank Adaptation (`LoRA`)** is a popular PEFT method. The core idea is based on the hypothesis that the *change* in the weights during fine-tuning has a "low intrinsic rank." In simpler terms, the adaptation can be represented efficiently. Instead of updating a large weight matrix $W$, `LoRA` freezes $W$ and introduces two small, low-rank matrices, $A$ and $B$, which are trained. The original model's output `Wx` is modified to become $Wx + BAx$. Since $A$ and $B$ are much smaller than $W$, the number of trainable parameters is drastically reduced.

### 3.1.4. Quantization
Quantization in deep learning is the process of reducing the precision of the numbers used to represent a model's weights. For example, instead of storing a weight as a 32-bit floating-point number, it can be converted to an 8-bit or 4-bit integer. This has two main benefits:
1.  **Reduced Memory:** A 4-bit model requires 8x less memory to store than a 32-bit model.
2.  **Faster Inference:** Integer arithmetic can be much faster than floating-point arithmetic on many modern processors (like GPUs and CPUs).
    The trade-off is a potential loss of accuracy, but modern quantization techniques (like those used in this paper) can often achieve significant memory savings with little to no performance degradation.

## 3.2. Previous Works
The paper positions itself relative to three main lines of research:

*   **Generalist Robot Policies (e.g., Octo):** These models aim to control multiple robots across various tasks by training on large, diverse datasets. `Octo` is a notable open-source example. However, its architecture typically involves "stitching" together pretrained components (like a vision encoder and language embedder) with other randomly initialized parts. This is different from `OpenVLA`'s end-to-end approach of fine-tuning a complete, pretrained VLM.
*   **Vision-Language-Action Models (VLAs; e.g., RT-2-X):** These models, like `OpenVLA`, directly fine-tune a large VLM to predict robot actions. Google's `RT-1` and `RT-2` pioneered this approach, showing impressive generalization by transferring knowledge from web-scale data to robotics. `RT-2-X` was the state-of-the-art VLA, trained on the Open X-Embodiment dataset. However, it is a massive 55B-parameter model and is closed-source, which are the primary limitations `OpenVLA` aims to overcome.
*   **From-Scratch Imitation Learning (e.g., Diffusion Policy):** These methods learn visuomotor policies by mimicking expert demonstrations without leveraging large pretrained models. `Diffusion Policy`, which models the action distribution using a diffusion model, is a state-of-the-art example known for its data efficiency and ability to produce smooth, precise motions for specific tasks. However, it lacks the broad semantic and visual priors of a VLA and may struggle with generalization to new instructions or highly cluttered scenes.

## 3.3. Technological Evolution
The field of robot learning has evolved from training specialized policies for single tasks to creating more general-purpose agents. This evolution can be seen as a progression:
1.  **Single-Task Policies:** Models trained from scratch for one specific skill (e.g., picking up a block).
2.  **Multi-Task Policies:** Models trained on a collection of related tasks, showing some generalization within a domain.
3.  **Generalist Policies (`Octo`):** The first wave of models trained on large, multi-embodiment datasets, often using pretrained components in a modular fashion.
4.  **Vision-Language-Action Models (`RT-2`, `OpenVLA`):** The current frontier, which moves beyond modularity to an end-to-end fine-tuning paradigm. These models treat robotics as another downstream task for massive web-trained foundation models, aiming to directly transfer their powerful generalization capabilities.

    `OpenVLA` represents a critical step in this timeline by making the VLA paradigm open, accessible, and efficient to adapt.

## 3.4. Differentiation Analysis
`OpenVLA` distinguishes itself from prior work in several key ways:
*   **vs. `RT-2-X`:** `OpenVLA` is **open-source**, significantly **smaller** (7B vs. 55B parameters), and **outperforms** `RT-2-X` on generalist benchmarks. It also uses a more advanced fused vision encoder (`DINOv2` + `SigLIP`) and is trained on a larger, more curated dataset.
*   **vs. `Octo`:** `OpenVLA` adopts a true **end-to-end VLA architecture**, fine-tuning a complete pretrained VLM. In contrast, `Octo` "stitches" pretrained parts together with from-scratch components. The experiments show that `OpenVLA`'s approach leads to substantially better performance and language grounding.
*   **vs. `Diffusion Policy`:** `OpenVLA` is a **pretrained model** that excels at generalization, especially in tasks requiring language understanding and robustness to visual distractors. `Diffusion Policy` is trained **from scratch** and, while excellent for specific dexterous skills, does not have the same level of semantic prior knowledge. `OpenVLA` aims for breadth of capability, while `Diffusion Policy` targets depth in specific skills.

# 4. Methodology

## 4.1. Principles
The core principle behind `OpenVLA` is to reframe the complex problem of robotic control as a **sequence prediction task**, making it solvable by a powerful Vision-Language Model (VLM). The intuition is that an AI model pretrained on vast amounts of internet data has already learned rich, generalizable representations of objects, language concepts, and their relationships. By fine-tuning this model on robot data, we can "teach" it to map its existing understanding of vision and language to physical actions.

The model processes an image and a language command and predicts a sequence of "action tokens" that correspond to the robot's movements. This end-to-end approach allows the model to leverage its pretrained knowledge for robust and generalizable control.

## 4.2. Core Methodology In-depth (Layer by Layer)
The `OpenVLA` architecture, shown in Figure 2 from the paper, consists of three main components that work in sequence.

![Figure 2: OpenVLA model architecture. Given an image observation and a language instruction, the model predicts 7-dimensional robot control actions. The architecture consists of three key components:(1 a vision ener that concatenates Dino V2 \[25\] and SigLIP \[79\] features, () a projector that maps visual featuresto the language embeding space, and (3) the LLM backbone, a Llama 2 7B-parameter large language model \[10\].](images/2.jpg)
*该图像是OpenVLA模型架构示意图。该架构根据输入的图像和语言指令，预测7维机器人控制动作。主要由三个关键组件组成：视觉编码器DinoV2与SigLIP特征的连接、映射视觉特征至语言嵌入空间的多层感知器项目器，以及7B参数的Llama 2大语言模型。*

### 4.2.1. Step 1: Fused Vision Encoding
The process begins with the model observing the environment through a single image. This image is fed into a dual-stream vision encoder to extract a rich set of features.

*   **Input:** A single RGB image observation from the robot's camera (e.g., a third-person view of the workspace). The paper notes they use a resolution of 224x224 pixels.
*   **Parallel Encoding:** The image is passed through two separate, pretrained vision models simultaneously:
    1.  **DINOv2:** This model processes the image to extract **low-level spatial and geometric features**. It is excellent at understanding object shapes, boundaries, and part segmentations.
    2.  **SigLIP:** This model processes the same image to extract **high-level semantic features**. It is excellent at recognizing what objects are and understanding concepts described in natural language.
*   **Feature Fusion:** The output features from `DINOv2` and `SigLIP` are generated for patches of the image. For each patch, the feature vectors from the two encoders are **concatenated channel-wise**. This creates a single, richer feature vector for each patch that contains both spatial and semantic information. The sequence of these fused feature vectors forms the "image patch embeddings."

### 4.2.2. Step 2: Projection to Language Space
The LLM backbone (`Llama 2`) cannot directly process the feature embeddings from the vision encoders. They must first be translated into a format the LLM can understand.

*   **Input:** The sequence of fused image patch embeddings from the vision encoder.
*   **Projector:** A small, 2-layer Multi-Layer Perceptron (MLP) acts as a "projector." Its job is to map the high-dimensional visual feature vectors into the lower-dimensional word embedding space of the `Llama 2` model.
*   **Output:** A new sequence of "visual tokens." These are now in the same vector space as the LLM's text embeddings, making them comprehensible to the language model.

### 4.2.3. Step 3: LLM Processing and Action Prediction
This is where the vision, language, and action components come together.

*   **Input Assembly:** The model's final input sequence is constructed by concatenating the visual tokens with the tokens from the natural language command. For a command like "pick up the apple," the sequence would be:
    $[visual_token_1, ..., visual_token_k, token_("pick"), token_("up"), token_("the"), token_("apple")]$
*   **LLM Backbone:** This entire sequence is fed into the 7B-parameter `Llama 2` model. The LLM processes this multimodal sequence using its transformer architecture, attending to both the visual information and the language instruction to understand the context and the required task.
*   **Action Prediction:** The LLM's task is to predict the next tokens in the sequence. In the VLA framework, these next tokens are defined to be the robot's actions. To make this possible, continuous robot actions must be converted into discrete tokens that the LLM can predict.

### 4.2.4. Action Discretization and Representation
`OpenVLA` follows the approach from `RT-2` to represent continuous robot actions as discrete tokens.
*   **Continuous Actions:** A robot's action is typically represented as a continuous vector, for example, a 7-dimensional vector for a 7-DoF arm (6 dimensions for end-effector pose and 1 for gripper state).
*   **Discretization:** Each dimension of this continuous action vector is discretized into one of 256 bins. The paper makes a key improvement over prior work: instead of using the absolute minimum and maximum values in the training data to define the range for these bins, they use the **1st and 99th quantiles**. This makes the discretization process much more robust to outlier actions (e.g., rare, large movements) which could otherwise reduce the effective precision for common actions.
*   **Tokenization:** After discretization, an $N$-dimensional action is represented by $N$ integers, each between 0 and 255. These integers are then mapped to special tokens in the LLM's vocabulary. The authors note that the `Llama` tokenizer has a limited number of "special tokens" available for fine-tuning. To handle this, they simply **overwrite the 256 least-used tokens** in the original vocabulary with their 256 new action bin tokens.
*   **Final Output:** The LLM predicts a sequence of $N$ action tokens, which are then de-tokenized and converted back into a continuous action vector to be sent to the robot controller.

### 4.2.5. Training Procedure
The model is trained using a standard language modeling objective.
*   **Objective:** The model is trained with a **next-token prediction** loss (cross-entropy).
*   **Loss Calculation:** Crucially, the loss is only computed on the predicted **action tokens**. The model is not penalized for its predictions on the input image or language tokens. It is only trained to predict the correct action sequence given the visual and linguistic context.
*   **Key Training Decisions:**
    *   **Fine-tuning the Vision Encoder:** Unlike in many VLM training recipes, the authors found it was essential to fine-tune the vision encoder weights during VLA training. They hypothesize this is necessary to adapt the visual features to the fine-grained spatial details required for precise robot control.
    *   **Training Epochs:** The model was trained for 27 epochs, far more than is typical for LLM/VLM pretraining. This was necessary to achieve a high action token prediction accuracy (over 95%), which correlated with better real-robot performance.
    *   **Learning Rate:** A constant learning rate of 2e-5 was used, without a warmup schedule.

# 5. Experimental Setup

## 5.1. Datasets
The experiments used a combination of a large-scale pretraining dataset and several smaller datasets for evaluation and fine-tuning.

*   **Pretraining Dataset:**
    *   **Open X-Embodiment (OpenX):** This is the primary dataset used for pretraining `OpenVLA`. It is a massive, standardized collection of robotic learning datasets from many different institutions. The authors curated a subset of **970,000 real-world robot trajectories** from OpenX.
    *   **Curation:** To ensure consistency, they filtered the data to include only manipulation tasks with a third-person camera view and single-arm end-effector control. They balanced the mixture of different sub-datasets using weights from the `Octo` model, which prioritizes datasets with high task and scene diversity. They also included the new `DROID` dataset.

*   **Evaluation & Fine-tuning Datasets:**
    *   **BridgeData V2 (WidowX robot):** This dataset features a WidowX arm in a kitchen sink environment. It is used for out-of-the-box evaluation of generalist policies. An example task is "Put Eggplant into Pot."
    *   **Google Robot Dataset:** This dataset features a mobile manipulation robot used in the `RT-1` and `RT-2` papers. It is also used for out-of-the-box evaluation. An example task is "Pick Coke Can."
    *   **Franka-Tabletop:** A new dataset collected by the authors for fine-tuning experiments, featuring a Franka Emika Panda robot on a tabletop. It includes both single-instruction tasks (e.g., "Pour Corn into Pot") and multi-instruction tasks requiring language grounding (e.g., "Move <object> onto Plate").
    *   **Franka-DROID:** A setup using a Franka arm from the `DROID` dataset, used for fine-tuning experiments on tasks like "Wipe Table."
    *   **LIBERO:** A benchmark of simulated robot tasks used to test `OpenVLA`'s ability to adapt to simulation environments.

## 5.2. Evaluation Metrics
The primary metric used throughout the paper is the **Task Success Rate**.

*   **Success Rate**
    1.  **Conceptual Definition:** The Success Rate measures the fraction of trials in which the robot successfully completes the assigned task. The criteria for success are task-specific (e.g., the object is in the correct final location). In some more complex tasks, a partial success (e.g., correctly grasping the target object but failing to place it) is awarded a score of 0.5.
    2.  **Mathematical Formula:**
        \$
        \text{Success Rate} = \frac{\sum_{i=1}^{N} \text{Result}_i}{N}
        \$
    3.  **Symbol Explanation:**
        *   $N$: The total number of evaluation trials for a given task or set of tasks.
        *   $\text{Result}_i$: The outcome of the $i$-th trial. It is typically 1 for a full success and 0 for a failure. For some tasks, it can be 0.5 for a partial success.

            The paper also reports the **Standard Error (StdErr)** to quantify the uncertainty of the success rate estimate.

*   **Standard Error**
    1.  **Conceptual Definition:** The Standard Error of the mean measures how much the calculated sample success rate is likely to vary from the true, unknown success rate of the policy. A smaller standard error indicates a more precise estimate.
    2.  **Mathematical Formula:**
        \$
        \text{StdErr} = \frac{s}{\sqrt{n}}
        \$
    3.  **Symbol Explanation:**
        *   $n$: The number of trials.
        *   $s$: The sample standard deviation of the trial outcomes. For a success rate $p$, the standard deviation of the Bernoulli-distributed outcomes is `s = \sqrt{p(1-p)}`.

## 5.3. Baselines
`OpenVLA` was compared against a strong set of representative models:

*   **RT-1-X (35M parameters):** A transformer-based policy trained from scratch on a subset of the OpenX dataset. Represents a non-VLM generalist policy.
*   **Octo (93M parameters):** The previous open-source state-of-the-art generalist policy. It uses pretrained components but is not an end-to-end VLA.
*   **RT-2-X (55B parameters):** The previous state-of-the-art VLA, which is closed-source. This is the main high-performance VLA baseline.
*   **Diffusion Policy:** A state-of-the-art imitation learning method trained from scratch on the target task data. This baseline represents the best performance achievable without large-scale pretraining.
*   **OpenVLA (scratch):** An ablation of `OpenVLA` where the Prismatic VLM backbone is fine-tuned directly on the target task without the large-scale OpenX robot pretraining. This is used to isolate the benefit of the robot-specific pretraining.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Out-of-the-Box Generalist Performance
The first set of experiments evaluates `OpenVLA`'s "out-of-the-box" performance on two different robot platforms without any further fine-tuning.

**BridgeData V2 (WidowX Robot) Evaluations:**
The results from Figure 3 and Table 4 show a clear performance hierarchy.
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Category</th>
<th rowspan="2">Task</th>
<th rowspan="2"># Trials</th>
<th rowspan="2">RT-1-X # Successes</th>
<th rowspan="2">Octo # Successes</th>
<th rowspan="2">RT-2-X # Successes</th>
<th rowspan="2">OpenVLA (ours) # Successes</th>
</tr>
<tr></tr>
</thead>
<tbody>
<tr>
<td>Visual gen</td>
<td>Put Eggplant into Pot (Easy Version)</td>
<td>10</td>
<td>1</td>
<td>5</td>
<td>7</td>
<td>10</td>
</tr>
<tr>
<td>Visual gen</td>
<td>Put Eggplant into Pot</td>
<td>10</td>
<td>0</td>
<td>1</td>
<td>5</td>
<td>10</td>
</tr>
<tr>
<td>Visual gen</td>
<td>Put Cup from Counter into Sink</td>
<td>10</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>7</td>
</tr>
<tr>
<td>Visual gen</td>
<td>Put Eggplant into Pot (w/ Clutter)</td>
<td>10</td>
<td>1</td>
<td>3.5</td>
<td>6</td>
<td>7.5</td>
</tr>
<tr>
<td>Visual gen</td>
<td>Put Yellow Corn on Pink Plate</td>
<td>10</td>
<td>1</td>
<td>4</td>
<td>8</td>
<td>9</td>
</tr>
<tr>
<td>Motion gen</td>
<td>Lift Eggplant</td>
<td>10</td>
<td>3</td>
<td>0.5</td>
<td>6.5</td>
<td>7.5</td>
</tr>
<tr>
<td>Motion gen</td>
<td>Put Carrot on Plate (w/ Height Change)</td>
<td>10</td>
<td>2</td>
<td>1</td>
<td>4.5</td>
<td>4.5</td>
</tr>
<tr>
<td>Physical gen</td>
<td>Put Carrot on Plate</td>
<td>10</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>8</td>
</tr>
<tr>
<td>Physical gen</td>
<td>Flip Pot Upright</td>
<td>10</td>
<td>2</td>
<td>6</td>
<td>5</td>
<td>8</td>
</tr>
<tr>
<td>Physical gen</td>
<td>Lift AAA Battery</td>
<td>10</td>
<td>0</td>
<td>0</td>
<td>2</td>
<td>7</td>
</tr>
<tr>
<td>Semantic gen</td>
<td>Move Skull into Drying Rack</td>
<td>10</td>
<td>1</td>
<td>0</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>Semantic gen</td>
<td>Lift White Tape</td>
<td>10</td>
<td>3</td>
<td>0</td>
<td>0</td>
<td>1</td>
</tr>
<tr>
<td>Semantic gen</td>
<td>Take Purple Grapes out of Pot</td>
<td>10</td>
<td>0.5</td>
<td>0</td>
<td>5</td>
<td>4</td>
</tr>
<tr>
<td>Semantic gen</td>
<td>Stack Blue Cup on Pink Cup</td>
<td>10</td>
<td>2.5</td>
<td>4</td>
<td>5.5</td>
<td>4.5</td>
</tr>
<tr>
<td>Language grounding</td>
<td>Put {Eggplant, Red Bottle} into Pot</td>
<td>10</td>
<td>1.5</td>
<td>2.5</td>
<td>8.5</td>
<td>8.5</td>
</tr>
<tr>
<td>Language grounding</td>
<td>Lift {Cheese, Red Chili Pepper}</td>
<td>10</td>
<td>5</td>
<td>5.5</td>
<td>8.5</td>
<td>7.5</td>
</tr>
<tr>
<td>Language grounding</td>
<td>Put {Blue Cup, Pink Cup} on Plate</td>
<td>10</td>
<td>1.5</td>
<td>2.5</td>
<td>5.5</td>
<td>9.5</td>
</tr>
<tr>
<td colspan="2">Mean Success Rate</td>
<td></td>
<td>18.5±2.7%</td>
<td>20.0±2.6%</td>
<td>50.6±3.5%</td>
<td>70.6±3.2%</td>
</tr>
</tbody>
</table>

*   **`OpenVLA` vs. `RT-2-X`:** `OpenVLA` achieves a mean success rate of **70.6%**, dramatically outperforming the much larger `RT-2-X` model's **50.6%**. This is a remarkable result, demonstrating that a smaller, open-source model with better architecture and data can surpass a giant, closed-source one. `OpenVLA` particularly excels in tasks requiring physical and visual generalization.
*   **`OpenVLA` vs. Open-Source Baselines:** `OpenVLA` massively outperforms `RT-1-X` (18.5%) and `Octo` (20.0%), confirming that the end-to-end VLA approach is superior to policies trained from scratch or with modular pretrained components.
*   **Semantic Generalization:** `RT-2-X` performs better on some semantic generalization tasks (e.g., "Take Purple Grapes out of Pot"). The authors suggest this is because `RT-2-X` was co-trained on robot data and internet data simultaneously, which helps preserve more of the base VLM's knowledge.

**Google Robot Evaluations:**
The results on the Google Robot platform (Figure 4, Table 6) show a similar trend.
The following are the results from Table 6 of the original paper:

<table>
<thead>
<tr>
<th>Category</th>
<th>Task</th>
<th># Trials</th>
<th>RT-1-X # Successes</th>
<th>Octo # Successes</th>
<th>RT-2-X # Successes</th>
<th>OpenVLA (ours) # Successes</th>
</tr>
</thead>
<tbody>
<tr>
<td>In-distribution</td>
<td>Pick Coke Can</td>
<td>5</td>
<td>5</td>
<td>1</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>In-distribution</td>
<td>Move Apple near Green Can</td>
<td>5</td>
<td>3</td>
<td>3</td>
<td>3</td>
<td>5</td>
</tr>
<tr>
<td>In-distribution</td>
<td>Move Blue Chip Bag near Apple</td>
<td>5</td>
<td>0</td>
<td>3</td>
<td>4</td>
<td>5</td>
</tr>
<tr>
<td>In-distribution</td>
<td>Place Coke Can Upright</td>
<td>5</td>
<td>0</td>
<td>0</td>
<td>4</td>
<td>4</td>
</tr>
<tr>
<td>In-distribution</td>
<td>Open Middle Drawer</td>
<td>5</td>
<td>0</td>
<td>4</td>
<td>2</td>
<td>3</td>
</tr>
<tr>
<td>OOD</td>
<td>Move Orange near Brown Chip Bag</td>
<td>5</td>
<td>1</td>
<td>2</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>OOD</td>
<td>Pick Pepsi Can</td>
<td>5</td>
<td>3</td>
<td>0</td>
<td>5</td>
<td>4</td>
</tr>
<tr>
<td>OOD</td>
<td>Pick Banana</td>
<td>5</td>
<td>5</td>
<td>3</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>OOD</td>
<td>Pick Green Cup</td>
<td>5</td>
<td>1</td>
<td>0</td>
<td>5</td>
<td>5</td>
</tr>
<tr>
<td>OOD</td>
<td>Place Apple on Plate</td>
<td>5</td>
<td>0</td>
<td>0</td>
<td>4</td>
<td>4</td>
</tr>
<tr>
<td>OOD</td>
<td>Place Banana in Pan</td>
<td>5</td>
<td>0</td>
<td>0</td>
<td>2</td>
<td>4</td>
</tr>
<tr>
<td>OOD</td>
<td>Move Coke Can near Taylor Swift</td>
<td>5</td>
<td>2</td>
<td>0</td>
<td>3</td>
<td>2</td>
</tr>
<tr>
<td colspan="2">Mean Success Rate</td>
<td></td>
<td>33.3±6.1%</td>
<td>26.7±5.8%</td>
<td>78.3±5.4%</td>
<td>85.0±4.6%</td>
</tr>
</tbody>
</table>

*   On this evaluation suite, `OpenVLA` (**85.0%**) and `RT-2-X` (**78.3%**) achieve comparable, strong performance, and both significantly outperform `RT-1-X` (33.3%) and `Octo` (26.7%). This confirms `OpenVLA`'s status as a top-tier generalist policy.

### 6.1.2. Data-Efficient Adaptation to New Setups
This section tests how well `OpenVLA` can be adapted to new tasks with only a small number of demonstrations (10-150). The results are shown in Figure 5 and Table 7.

*   **`OpenVLA` vs. `Diffusion Policy`:** On narrow, single-instruction tasks ("Put Carrot in Bowl", "Pour Corn into Pot"), `Diffusion Policy` (trained from scratch) performs exceptionally well. However, on more diverse, multi-instruction tasks that require strong language grounding ("Move <object> onto Plate", "Knock <object> Over"), fine-tuned **`OpenVLA` is substantially better**. Overall, `OpenVLA` achieves the highest average success rate (**67.2%** on Franka-Tabletop), making it a strong default choice for a wide range of imitation learning problems.
*   **The Value of Pretraining:** The `OpenVLA (scratch)` ablation, which fine-tunes the VLM backbone directly on the task data without OpenX pretraining, performs much worse (**43.4%**). This result powerfully demonstrates that the large-scale robot pretraining on OpenX is critical for enabling effective and data-efficient fine-tuning on new tasks.

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Parameter-Efficient Fine-Tuning
Table 1 evaluates different strategies for fine-tuning `OpenVLA` on a consumer GPU budget.
The following are the results from Table 1 of the original paper:

<table>
<tr>
<td>Strategy</td>
<td>Success Rate</td>
<td>Train Params (× 10<sup>6</sup>)</td>
<td>VRAM (batch 16)</td>
</tr>
<tr>
<td>Full FT</td>
<td>69.7 ± 7.2 %</td>
<td>7,188.1</td>
<td>163.3 GB*</td>
</tr>
<tr>
<td>Last layer only</td>
<td>30.3 ± 6.1 %</td>
<td>465.1</td>
<td>51.4 GB</td>
</tr>
<tr>
<td>Frozen vision</td>
<td>47.0 ± 6.9 %</td>
<td>6,760.4</td>
<td>156.2 GB*</td>
</tr>
<tr>
<td>Sandwich</td>
<td>62.1 ± 7.9 %</td>
<td>914.2</td>
<td>64.0 GB</td>
</tr>
<tr>
<td>LoRA, rank=32</td>
<td>68.2 ± 7.5%</td>
<td>97.6</td>
<td>59.7 GB</td>
</tr>
<tr>
<td>rank=64</td>
<td>68.2 ± 7.8%</td>
<td>195.2</td>
<td>60.5 GB</td>
</tr>
</table>

*\*: Sharded across 2 GPUs with FSDP.*

*   **`LoRA` is a clear winner:** `LoRA` fine-tuning achieves a success rate (**68.2%**) that is statistically indistinguishable from full fine-tuning (**69.7%**). However, it does so by training only **1.4%** of the parameters (97.6M vs. 7.2B) and requires far less VRAM (59.7 GB vs. 163.3 GB). This is a crucial finding, as it makes adapting VLAs practical for a much wider audience.
*   **Importance of Vision Encoder:** Freezing the vision encoder (`Frozen vision`) or only training the last layer (`Last layer only`) leads to poor performance. This corroborates the author's finding that adapting the visual features to the new domain is critical.

### 6.2.2. Memory-Efficient Inference via Quantization
Table 2 investigates the trade-off between model precision, memory usage, and performance at inference time.
The following are the results from Table 2 of the original paper:

<table>
<tr>
<td>Precision</td>
<td>Bridge Success</td>
<td>VRAM</td>
</tr>
<tr>
<td>bfloat16</td>
<td>71.3 ± 4.8%</td>
<td>16.8 GB</td>
</tr>
<tr>
<td>int8</td>
<td>58.1 ± 5.1%</td>
<td>10.2 GB</td>
</tr>
<tr>
<td>int4</td>
<td>71.9 ± 4.7%</td>
<td>7.0 GB</td>
</tr>
</table>

*   **4-bit Quantization is Highly Effective:** Using 4-bit (`int4`) quantization results in a success rate (**71.9%**) that is identical to the default half-precision (`bfloat16`) inference (**71.3%**). However, it reduces the VRAM requirement from 16.8 GB to just **7.0 GB**. This enables `OpenVLA` to be run on a wide range of consumer and professional GPUs.
*   **8-bit Quantization Anomaly:** 8-bit (`int8`) quantization performs worse. The authors hypothesize this is not due to a loss of model quality, but because the specific `int8` operations are slower on their hardware, leading to a lower control frequency which degrades performance. Further experiments in Appendix D.4 with blocking control confirm this hypothesis.

### 6.2.3. Additional Ablations (from Appendices)
The appendices provide further strong evidence for the design choices made in `OpenVLA`:
*   **OpenX Training is Essential:** Ablating the OpenX pretraining and training only on BridgeData V2 causes the success rate to plummet from **76.3%** to **45.6%** (Table 9). This confirms the huge benefit of pretraining on a large, diverse robotic dataset.
*   **Fused Vision Encoder Helps:** Removing the `DINOv2` component from the vision backbone (leaving only `SigLIP`) causes a further drop in performance from 45.6% to 40.6% (Table 9), demonstrating the value of the fused spatial-semantic features.
*   **Fine-tuning the Vision Encoder is Crucial:** Experiments in Table 10 show that fine-tuning the vision encoder leads to dramatically better performance (e.g., 80% success) compared to freezing it (46.7% success), a key finding that contrasts with common practices in VLM training.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper introduces `OpenVLA`, a 7-billion-parameter open-source Vision-Language-Action model that sets a new state-of-the-art for generalist robot manipulation. By leveraging a strong VLM backbone (`Llama 2`), a novel fused vision encoder (`DINOv2` + `SigLIP`), and pretraining on a massive dataset of 970k robot demonstrations, `OpenVLA` significantly outperforms prior closed-source models that are much larger.

Beyond raw performance, the paper's most significant contribution is a practical roadmap for making powerful VLA models accessible and adaptable. The authors are the first to demonstrate that parameter-efficient fine-tuning techniques like `LoRA` and inference optimizations like 4-bit quantization can be used to adapt and deploy VLAs on consumer-grade hardware without compromising performance. By open-sourcing their model, codebase, and methodologies, the authors have provided an invaluable resource that is likely to accelerate research and development in robot learning.

## 7.2. Limitations & Future Work
The authors candidly acknowledge several limitations and avenues for future work:

*   **Limited Sensory Inputs:** The current `OpenVLA` model only accepts a single, static image as input. Future work should explore incorporating observation history (video), proprioceptive feedback (robot joint states), and multiple camera views to provide a richer context for decision-making.
*   **Inference Speed:** While efficient, the model's inference speed (around 6 Hz on an RTX 4090) is insufficient for high-frequency control tasks (e.g., 50 Hz) or highly dexterous, dynamic manipulation. Techniques like action chunking or speculative decoding could help improve throughput.
*   **Reliability:** While `OpenVLA` is state-of-the-art, its success rates are typically below 90%, which is not yet sufficient for reliable real-world deployment in safety-critical or commercial applications. Further improvements in data, architecture, and training are needed.
*   **Unexplored Design Space:** Many fundamental questions about VLAs remain unanswered due to computational constraints. These include the impact of VLM scale (e.g., would a 70B `OpenVLA` be better?), the benefits of co-training on robot and internet data, and identifying the optimal visual features for robotics. The release of `OpenVLA` is intended to help the community explore these questions.

## 7.3. Personal Insights & Critique
*   **Significance and Inspiration:** This paper feels like a landmark moment for open-source robotics. It successfully democratizes a cutting-edge paradigm (VLAs) that was previously locked within large industrial research labs. The dual focus on state-of-the-art performance and computational accessibility is a powerful combination. The methodology of demonstrating a practical, efficient adaptation workflow (`LoRA` + quantization) provides a clear and valuable template for researchers in many other fields looking to apply foundation models.

*   **Potential Issues & Areas for Improvement:**
    *   **Action Representation:** The simple binning of continuous actions into 256 discrete tokens per dimension is effective but may be a bottleneck. It is a coarse representation that might struggle with tasks requiring very high precision or smoothness. Future research could explore more sophisticated, continuous, or hybrid action spaces that are still compatible with transformer architectures.
    *   **Knowledge Transfer vs. Catastrophic Forgetting:** The paper notes that `RT-2-X`, which co-trains on internet data, retains more semantic knowledge. `OpenVLA`, being fine-tuned solely on robot data, might be experiencing a mild form of "catastrophic forgetting" of the VLM's original general knowledge. Investigating training schemes that balance specialization on robot tasks with the preservation of broad world knowledge is a critical next step.
    *   **Data-Driven Nature:** Like all such models, `OpenVLA`'s capabilities are fundamentally tied to its training data. The 970k OpenX trajectories are impressive, but still orders of magnitude smaller than the data used to train the base VLM. The model's ability to generalize to truly novel physical interactions not represented in the data (e.g., manipulating deformable objects, using tools in novel ways) remains an open question.

*   **Future Directions:** This work opens up exciting possibilities. One could imagine using `OpenVLA` as a base model for a "robotics app store," where users could download the base model and then easily fine-tune it on a few demonstrations of a specific household chore using `LoRA` on their home computer. The open-source nature of the project will undoubtedly spawn a vibrant ecosystem of new datasets, fine-tuned models, and improved architectures, much like what has been seen in the open-source LLM community.