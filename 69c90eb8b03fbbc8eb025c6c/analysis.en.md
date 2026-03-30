# 1. Bibliographic Information
## 1.1. Title
The central topic of the paper is **EagleVision**, a custom multimodal large language model (MLLM) tailored specifically for remote sensing (RS), which enables joint precise object detection and fine-grained object-level attribute understanding of objects in high-resolution RS images.

## 1.2. Authors
The authors are Hongxiang Jiang, Jihao Yin, Qixiong Wang, Jiaqi Feng, and Guo Chen, all affiliated with Beihang University (Beijing University of Aeronautics and Astronautics), a top Chinese research institution focused on aerospace and remote sensing science. Jihao Yin is the corresponding author.

## 1.3. Journal/Conference
This work is currently released as a preprint on arXiv, the most widely trusted open preprint server for computer science research. It has not yet been peer-reviewed and formally published in a conference or journal as of the current date.

## 1.4. Publication Year
The preprint was uploaded to arXiv in March 2025, so the publication year (for the preprint) is 2025.

## 1.5. Abstract
This paper addresses the key limitations of existing MLLMs for remote sensing: existing methods struggle with precise localization and fine-grained attribute description of small objects in high-resolution RS images, and have not yet outperformed classical visual perception models in real-world RS tasks. To solve this, the authors propose EagleVision, an RS-tailored MLLM that integrates object detection and fine-grained attribute comprehension, equipped with a custom Attribute Disentangle module that learns disentangled vision tokens to represent distinct object attributes. The authors also construct two new public resources: EVAttrs-95K, the first large-scale RS object attribute understanding dataset for instruction tuning, and EVBench, a new standardized evaluation benchmark for this task. Experimental results show EagleVision achieves state-of-the-art performance on both fine-grained object detection and object attribute understanding, demonstrating that detection and understanding capabilities mutually reinforce each other in RS MLLMs.

## 1.6. Original Source Link
Original preprint link: https://arxiv.org/abs/2503.23330  
PDF link: https://arxiv.org/pdf/2503.23330  
Publication status: Unpublished preprint. All code, data, and models will be released publicly at https://github.com/XiangTodayEatsWhat/EagleVision.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Existing MLLMs for remote sensing suffer from two critical limitations: (1) RS images are high-resolution, and most objects of interest (e.g., ships, aircraft) are small, so existing MLLMs only produce coarse, sparse understanding, missing many small objects and failing to provide fine-grained attribute descriptions for detected objects. (2) Traditional visual perception models (classical object detectors) only output coarse category labels for objects, with no fine-grained attribute understanding, which is insufficient for practical RS applications that require detailed analysis. Prior RS MLLMs have not outperformed traditional detectors, leading to limited real-world utility.

### Importance of the Problem
Remote sensing has critical real-world applications including maritime surveillance, disaster response, infrastructure monitoring, and national security. Object-level fine-grained understanding is far more useful than coarse image-level classification or category-only detection for these use cases. General MLLMs are designed for natural images, and do not adapt to the unique characteristics of RS data, creating a significant gap for practical deployment.

### Gaps in Prior Research
1. Traditional RS object detectors only predict category labels, no attribute understanding, and cannot handle open-ended queries or novel object types beyond predefined categories.
2. General and existing RS MLLMs struggle with small objects in high-resolution RS images, have low object recall (miss most objects), produce only coarse descriptions, and do not support dense object-level attribute comprehension.
3. There was no existing large-scale dataset of object attributes for RS instruction tuning, and no standardized benchmark for evaluating object attribute understanding in RS.

### Innovative Entry Point
The paper proposes that explicitly disentangling attribute features at the object level, paired with a new large-scale attribute dataset for instruction tuning, will allow an MLLM to jointly improve both object detection and fine-grained attribute understanding, creating an RS-specific MLLM that outperforms both traditional detectors and prior MLLMs.

## 2.2. Main Contributions / Findings
1. **Model Innovation**: Proposes EagleVision, the first object-level attribute MLLM for remote sensing, which jointly performs precise oriented object detection and dense fine-grained attribute understanding for each detected object. Introduces the *Attribute Disentangle module*, which uses orthogonal subspace learning to learn disentangled vision tokens that each represent a distinct object attribute, solving the problem of mixed attribute features that hurt fine-grained understanding.
2. **New Data Resources**: Constructs **EVAttrs-95K**, the first large-scale object attribute understanding dataset for RS, containing 95.1k annotated objects with over 60 distinct fine-grained attributes for ships and aircraft. Also constructs **EVBench**, the first standardized benchmark for evaluating object-level attribute understanding in RS.
3. **Key Experimental Findings**:
   - EagleVision achieves state-of-the-art performance on both tasks, outperforming 15 SOTA traditional detectors and 6 advanced general/RS MLLMs across three benchmark datasets.
   - Attribute understanding training mutually improves object detection: EagleVision improves mean average precision (mAP) by 11.2%, 2.7%, and 0.3% on the three test datasets respectively compared to the baseline detector.
   - Larger LLM backbones consistently improve both detection and attribute understanding performance, confirming that scaling works for this domain.
   - Ablation studies verify that every component of the proposed design (patch embedding size, attribute disentanglement, orthogonal constraint) contributes to performance gains.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All core concepts are explained below for novice readers:
- **Remote Sensing (RS)**: The process of acquiring information about the Earth's surface from airborne or satellite sensors, producing high-resolution aerial/satellite images. RS images have unique characteristics compared to natural images: much higher resolution, small objects of interest, arbitrary object orientations, and a requirement for detailed individual object analysis.
- **Large Language Model (LLM)**: A transformer-based neural language model trained on massive text corpora, capable of understanding and generating human-like text and following natural language instructions.
- **Multimodal Large Language Model (MLLM)**: An extension of LLMs that processes both text and image input, aligning visual features with LLM text embeddings to enable tasks like visual question answering, image captioning, and object grounding.
- **Object Detection**: A computer vision task that locates all objects of interest in an image and classifies their category. For remote sensing, *oriented object detection* (predicting rotated bounding boxes instead of axis-aligned boxes) is standard to handle arbitrary object orientations.
- **Mean Average Precision (mAP)**: The standard evaluation metric for object detection, measuring both detection precision (how many detected objects are correct) and recall (how many total objects are detected), with higher values indicating better performance.
- **Instruction Tuning**: A training process for (M)LLMs where the model is fine-tuned on a dataset of natural language instruction-input/output pairs, to teach the model to follow human instructions and align outputs with human expectations.
- **Disentangled Representation Learning**: A representation learning approach that aims to learn separate latent features, each corresponding to a distinct independent attribute of the input data, improving interpretability and fine-grained understanding.
- **Orthogonal Subspace Learning**: A disentanglement method that enforces that the basis vectors representing different attributes are orthogonal (their dot product is zero), ensuring they encode linearly independent information for each attribute.

## 3.2. Previous Works
Previous works are grouped into two categories as presented by the authors:
### 3.2.1. Traditional Visual Perception Models for Remote Sensing
Early RS object detectors adapted general detection architectures (e.g., Faster R-CNN, CenterNet) for oriented objects, with representative methods including Oriented R-CNN, R3Det, and Gliding Vertex. These methods only focus on improving localization and category classification, and output only coarse category labels with no fine-grained attribute understanding. They cannot handle open-ended queries or novel object types beyond predefined categories.

More recent visual grounding methods (e.g., Grounding-DINO, PolyFormer) can localize objects based on text reference, but require a reference text prompt for each object, so they cannot automatically detect and describe all objects in an image without prior input, making them unsuitable for automated RS analysis.

Prior attribute recognition methods for natural images (e.g., OvarNet) rely on multi-stage training and CLIP-based contrastive retrieval, do not generate open-ended freeform descriptions, and do not generalize well to the RS domain.

### 3.2.2. Multimodal Large Language Models
General-domain MLLMs (e.g., LLaVA, LLaVA-Grounding) focus on global image-level understanding or reference-based grounding, and have low object recall especially for small objects in high-resolution RS images, leading to sparse, coarse understanding. Even top general MLLMs (GPT-4o, Qwen2-VL, Gemini) struggle with this problem for RS data.

Existing RS MLLMs (e.g., RSGPT, GeoChat, RSUniVLM) adapt general MLLM architectures to RS, but focus on image-level question answering and general conversational tasks, and do not address dense object-level fine-grained attribute understanding. All prior RS MLLMs suffer from low object recall and coarse descriptions, and have not outperformed traditional object detectors in detection performance.

## 3.3. Technological Evolution
The field of RS computer vision has evolved in four main phases:
1. **Phase 1 (1990s – 2020): Classical RS Object Detection**: Focused on improving oriented object localization and category classification, only output category labels, no attribute understanding.
2. **Phase 2 (2021 – 2023): General MLLMs for Visual Tasks**: Developed general vision-language alignment for natural images, demonstrating strong capabilities but not adapted to the unique characteristics of RS data.
3. **Phase 3 (2023 – 2024): Early RS MLLMs**: Adapted general MLLM architectures to RS, focused on image-level tasks, but did not address object-level fine-grained understanding.
4. **This Work (2025): Object-Level Attribute MLLMs for RS**: This work is the first to focus on dense object-level attribute understanding for RS MLLMs, integrates detection and attribute understanding, demonstrates mutual improvement of both tasks, and creates the first large-scale attribute dataset and benchmark for this new task. It sits at the current frontier of RS MLLM research, moving the field from image-level to object-level dense understanding.

## 3.4. Differentiation Analysis
- *vs Traditional RS object detectors*: EagleVision not only detects object categories and locations, but also provides dense fine-grained attribute descriptions for each object, supports open-ended queries, and actually improves detection performance compared to the baseline detector, outperforming traditional methods on both detection and understanding.
- *vs Visual grounding methods*: EagleVision does not require reference text prompts to detect objects, it can automatically detect all objects and describe their attributes without prior input, which is suitable for automated RS analysis.
- *vs General and existing RS MLLMs*: EagleVision is the first MLLM for RS that focuses on dense object-level attribute understanding, achieves much higher object recall and attribute accuracy than prior MLLMs, and is the first RS MLLM that outperforms traditional detectors on detection performance. It also introduces the first large-scale attribute dataset and benchmark for this new task, which did not exist before.

# 4. Methodology
## 4.1. Principles
The core idea of EagleVision is to jointly optimize object detection and object attribute understanding by explicitly disentangling attribute-specific visual features, leveraging a large-scale annotated attribute dataset for instruction tuning. The key intuitions are:
1. Mixed visual features (where multiple attributes are encoded into a single token) make it hard for the LLM to generate accurate fine-grained attribute descriptions.
2. Disentangling each attribute into its own independent vision token, enforced by orthogonal constraints, allows the LLM to clearly separate different attributes, improving attribute understanding.
3. Aligning visual features to attribute descriptions via language modeling loss indirectly improves the quality of visual features, which in turn improves object detection performance, leading to mutual reinforcement between the two tasks.

## 4.2. Core Methodology In-depth (Layer by Layer)
EagleVision has three core components: 1) Baseline Detector for object proposal and feature extraction, 2) Attribute Disentangle Module for learning disentangled attribute tokens, 3) Object-level Description Generation for generating open-ended attribute descriptions. We break down the full pipeline step-by-step:

### 4.2.1. Step 1: Baseline Detection and Object Feature Extraction
The pipeline starts with an input remote sensing image, and extracts object proposals and their visual features using a baseline detector. For an input image:
$$X_v \in \mathbb{R}^{H \times W \times 3}$$
where $H$ = image height, $W$ = image width, 3 = number of RGB color channels. The baseline detector (can be any single-stage or two-stage oriented object detector) extracts Region of Interest (ROI) features:
$$F_v = f(X_v; \theta)$$
- $f(\cdot)$: the baseline detector function
- $\theta$: learnable parameters of the detector
- $F_v \in \mathbb{R}^{N \times H' \times W' \times C}$: output ROI features, where $N$ = number of object proposals, `H', W'` = height and width of each ROI feature map, $C$ = number of feature channels.

  The baseline detector retains its original classification head $f_{cls}$ and bounding box regression head $f_{reg}$ to produce detection results. We feed $F_v$ into these heads to get detection outputs, then select the ROI features of all $N_{pos}$ foreground (actual object) proposals to get object features:
$$F_v^{pos} \in \mathbb{R}^{N_{pos} \times H' \times W' \times C}$$
These object features are used for all subsequent processing.

For optimization, we use the standard detection loss $\mathcal{L}_d$, which includes cross-entropy loss for classification and L1 or Rotated IoU loss for bounding box regression. All parameters of the detector are trained end-to-end.

### 4.2.2. Step 2: Patch Embedding Sampling
Before the Attribute Disentangle module, we sample neighborhood features around each object to get the initial patch embedding $E_v$, because original ROI features are often too coarse for fine-grained attribute learning.
- For two-stage detectors: We directly adjust the ROI feature size to $(2s+1) \times (2s+1)$ (where $s$ is a hyperparameter defining the neighborhood size) to get:
  $$E_v \in \mathbb{R}^{N_{pos} \times (2s+1) \times (2s+1) \times C}$$
- For single-stage detectors: Single-stage detectors usually output a single center feature per object, with $H' = W' = 1$. We find the center coordinate $r_i = (x_i, y_i)$ of each object $i$, then sample all features in a square neighborhood around the center:
  $$
  \begin{array}{r l}
  & R = \{ r_i \}_{i = 1, 2, \dots, N_{pos}} , r_i = (x_i, y_i) \\
  & S_i = \{ (x_i + s_x, y_i + s_y) | s_x, s_y \in [-s, s] \}
  \end{array}
  $$
  Where $S_i$ is the set of coordinates in the neighborhood of object $i$, and we extract the features at these coordinates to form $E_v$.

Regardless of detector type, we get an initial patch embedding $E_v$ of size $N_{pos} \times (2s+1) \times (2s+1) \times C$.

### 4.2.3. Step 3: Attribute Disentangle Module
The initial patch embedding $E_v$ mixes all attribute features together, so the LLM cannot easily distinguish between different attributes. To solve this, we use orthogonal subspace learning to disentangle the features into separate attribute tokens.

First, we learn a set of orthogonal basis vectors $p_1, p_2, ..., p_n$, where each basis $p_k \in \mathbb{R}^{1 \times C}$ spans an orthogonal attribute subspace $\mathcal{P} = span\{p_1, p_2, ..., p_n\}$. Each basis corresponds to a distinct attribute space. We then project the patch embedding $E_v$ onto each basis to get disentangled attribute tokens:
$$
\begin{array}{l}
T_v = cat(T_v^1, T_v^2, ..., T_v^n) \\
T_v^k = c_k p_k, \ c_k = \sum_{i}^{2s+1} \sum_{j}^{2s+1} E_v^{i,j} p_k^T
\end{array}
$$
- $E_v^{i,j} \in \mathbb{R}^{N_{pos} \times C}$: the feature at position `(i,j)` of the patch embedding for all objects
- $c_k$: the projection coefficient of the input patch onto the $k$-th attribute basis
- $cat(\cdot)$: concatenation along the attribute dimension
- $T_v \in \mathbb{R}^{N_{pos} \times n \times C}$: final output disentangled vision tokens, with $n$ independent tokens per object, each corresponding to a distinct attribute.

  To enforce that the basis vectors are orthogonal (so each token represents independent information), we add an orthogonality loss $\mathcal{L}_o$. Orthogonality requires that for any two different basis vectors $p_i$ and $p_j$, $p_i p_j^T = 0$. The orthogonality loss is:
$$
\mathcal{L}_o = \frac{2}{n \times (n - 1)} \sum_{i = 1}^{n} \sum_{j > i}^{n} |p_i p_j^T|
$$
This loss averages the absolute value of the dot product of all pairs of different basis vectors. Minimizing this loss pushes all off-diagonal dot products towards zero, enforcing orthogonality.

Next, we add an attribute matching loss $\mathcal{L}_a$ to ensure that each disentangled token actually corresponds to its correct attribute. We want to maximize the mutual information $I(c_k, T_a^k)$ between the projection coefficient $c_k$ and the ground-truth attribute token $T_a^k$ (encoded from the annotated attribute label). The loss is defined as:
$$
\mathcal{L}_a = - \frac{1}{n} \sum_{k}^{n} I(c_k, T_a^k)
$$
Maximizing mutual information directly is intractable, so we optimize its variational lower bound, which simplifies to the following mean squared error (MSE) loss:
$$
\mathcal{L}_a = \frac{1}{n} \sum_{k}^{n} \left( q(T_a^k; \varphi) - c_k \right)^2
$$
Where $q(T_a^k; \varphi)$ is a learnable projection of the ground-truth attribute token $T_a^k$, parameterized by $\varphi$. This loss pushes the projected coefficient $c_k$ from the visual feature to match the projection of the ground-truth attribute, enforcing that each token corresponds to the correct attribute.

*Note: The ground-truth attribute token $T_a^k$ is only used during training; during inference, no ground-truth attributes are needed as input.*

### 4.2.4. Step 4: Object-level Description Generation and Final Loss
After getting the disentangled vision tokens $T_v$, we concatenate them with the text tokens encoded from the input instruction prompt, and feed the combined sequence into a frozen pre-trained LLM to generate open-ended object-level attribute descriptions. The generation process is:
$$Y = g(T_v, T_q; \phi)$$
- $T_q$: encoded instruction prompt text tokens
- $g(\cdot)$: frozen pre-trained LLM
- $\phi$: frozen parameters of the LLM (only visual components are trained, the LLM is kept frozen)
- $Y$: generated text output of attribute descriptions.

  We calculate the language modeling loss $\mathcal{L}_q$, which is the standard next-token prediction loss, comparing the generated output $Y$ to the ground-truth attribute description $\hat{Y}$ from the EVAttrs-95K dataset. This loss aligns the disentangled visual features with the LLM's text embedding space, enabling the LLM to correctly interpret the visual tokens and generate accurate descriptions. This language loss also improves the quality of visual features extracted by the detector, which indirectly improves detection performance.

The final overall loss function for end-to-end training is:
$$
\mathcal{L}_{overall} = \lambda_d \mathcal{L}_d + \lambda_o \mathcal{L}_o + \lambda_a \mathcal{L}_a + \lambda_q \mathcal{L}_q
$$
Where $\lambda_d, \lambda_o, \lambda_a, \lambda_q$ are hyperparameter weights for each loss term, all set to 1.0 by default except $\lambda_q$ on FAIR1M which is set to 0.1.

### 4.2.5. EVAttrs-95K Dataset Construction
To train EagleVision, the authors constructed the first large-scale object attribute dataset for RS with a three-stage pipeline:
1. **Dataset Preprocessing**: Images are selected from three existing public RS detection datasets (FAIR1M, MAR20, ShipRSImageNet). Object patches are cropped for airplanes and ships, with 24 predefined attributes for airplanes and 38 predefined attributes for ships. The total dataset contains 95.1k annotated objects.
2. **Two-stage Automated Annotation**: A two-stage LLM data engine is used: Qwen2-VL-72B first automatically annotates all samples, then GPT-4o re-annotates low-confidence samples (confidence < 0.5). All outputs are restricted to JSON format with an explicit confidence score from 0 to 1.
3. **Human Refinement**: All annotations with confidence < 0.7 are manually reviewed, inconsistent annotations are corrected, and low-quality uncertain annotations are removed.

### 4.2.6. EVBench Evaluation Protocol
EVBench is a standardized benchmark for evaluating object attribute understanding in RS:
1. **Data Split**: Follows the split of EVAttrs-95K: FAIR1M split 3:1 train/test, MAR20 uses its original train/test split, ShipRSImageNet uses the original train/validation split for train/test.
2. **Response Preprocessing**: Model outputs and ground-truth annotations are converted to JSON format, with attribute names as keys and descriptions as values. Undetected objects get empty responses.
3. **Evaluation Metrics**:
   - **Recall**: Proportion of ground-truth objects that have non-empty valid responses, measuring how many objects the model does not miss.
   - **Attribute Score**: GPT-3.5-turbo is used as an evaluator to compare generated descriptions to ground-truth, scoring each attribute 1-5 based on correctness and expressiveness. The average score is scaled to 0-100, with higher values indicating better performance.

# 5. Experimental Setup
## 5.1. Datasets
All experiments are conducted on three standard public RS object detection datasets, with EVAttrs-95K derived from these datasets:

The following are the results from Table 2 of the original paper, showing the distribution of EVAttrs-95K:

<table>
<thead>
<tr>
<th>Data</th>
<th>FAIR1M</th>
<th>MAR20</th>
<th>ShipRSImageNet</th>
</tr>
</thead>
<tbody>
<tr>
<td>Total Size</td>
<td>59.8k</td>
<td>22.3k</td>
<td>13.0k</td>
</tr>
<tr>
<td>Train Split</td>
<td>44.2k</td>
<td>7.8k</td>
<td>10.1k</td>
</tr>
<tr>
<td>Test Split</td>
<td>15.6k</td>
<td>14.5k</td>
<td>2.9k</td>
</tr>
<tr>
<td>Average Number of Attributes per Object</td>
<td>~25</td>
<td>~24</td>
<td>~28</td>
</tr>
</tbody>
</table>

1. **FAIR1M-v1.0**: A standard benchmark for fine-grained object recognition in high-resolution RS, containing 37 subcategories across 5 major categories. 59.8k objects are included in EVAttrs-95K.
2. **MAR20**: A benchmark for military aircraft recognition in RS, containing 20 types of airplanes. 22.3k objects are included.
3. **ShipRSImageNet**: A large-scale fine-grained dataset for ship detection, containing 50 types of ships. 13.0k objects are included.

   These datasets are chosen because they are the most widely used public benchmarks for fine-grained oriented object detection in RS, covering the two most common object types (airplanes, ships) that require fine-grained attribute analysis for practical applications, making them ideal for validating the method's performance.

## 5.2. Evaluation Metrics
Two sets of metrics are used, for object detection and attribute understanding respectively:

### 5.2.1. Object Detection: Mean Average Precision (mAP)
1. **Conceptual Definition**: mAP is the standard metric for object detection that measures both the ability to correctly detect objects (recall) and correctly classify them (precision). It averages the average precision for each category to produce a single overall score, with higher values indicating better detection performance.
2. **Mathematical Formula**:
   First, for a single category, precision and recall are calculated as:
$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
$$
Average Precision (AP) for a single category is the area under the precision-recall curve:
$$
AP = \int_0^1 p(r) dr
$$
Mean Average Precision (mAP) is the average AP over all categories:
$$
mAP = \frac{1}{C} \sum_{c=1}^C AP_c
$$
3. **Symbol Explanation**:
- `TP`: Number of true positive detections (detections that match a ground-truth object with correct category and sufficient bounding box overlap)
- `FP`: Number of false positive detections (detections that do not match any ground-truth object, or match with wrong category)
- `FN`: Number of false negatives (ground-truth objects not detected by the model)
- `p(r)`: Precision at a given recall level $r$
- $AP_c$: Average precision for category $c$
- $C$: Total number of object categories

### 5.2.2. Object Attribute Understanding: Recall and Attribute Score
1. **Object Recall**
   - **Conceptual Definition**: Measures the proportion of all ground-truth objects that the model successfully detects and generates a non-empty valid attribute description for. It quantifies how many objects the model does not miss, which is critical for practical RS analysis.
   - **Mathematical Formula**:
     $$
   \text{Recall} = \frac{N_{\text{detected}}}{N_{\text{total}}} \times 100\%
   $$
   - **Symbol Explanation**: $N_{\text{detected}}$ = number of ground-truth objects with non-empty valid descriptions, $N_{\text{total}}$ = total number of ground-truth objects in the test set. Higher recall = better performance.

2. **Attribute Score**
   - **Conceptual Definition**: Measures the accuracy and expressiveness of generated attribute descriptions, using a GPT-assisted evaluation that correlates strongly with human judgment. The score is scaled from 0 to 100, with higher values indicating more accurate and complete descriptions.
   - **Mathematical Formula**:
     $$
   \text{Attribute Score} = \left( \frac{1}{N_{\text{detected}}} \sum_{i=1}^{N_{\text{detected}}} \frac{1}{A_i} \sum_{k=1}^{A_i} s_{i,k} \right) \times 20
   $$
   - **Symbol Explanation**: $N_{\text{detected}}$ = total number of detected objects, $A_i$ = number of attributes for object $i$, $s_{i,k}$ = score (1-5) given by the GPT evaluator for attribute $k$ of object $i$. Multiplying by 20 scales the average score from 1-5 to 0-100.

## 5.3. Baselines
Two sets of baselines are used for rigorous comparison:
1. **Object Detection Baselines**: 15 state-of-the-art traditional oriented object detectors, including one-stage methods (RetinaNet, R3Det, GGD, KLD, FCOS, S2ANet, TIOE-Det, RTMDet) and two-stage methods (Faster R-CNN, Gliding Vertex, ReDet, KFIoU, ROI Transformer, Oriented R-CNN, LSKNet). These are the most prominent and widely used SOTA detectors for RS oriented object detection, so comparison against them provides rigorous validation.
2. **Object Attribute Understanding Baselines**: 6 advanced MLLMs, including general-domain MLLMs (LLaVA-Grounding, Qwen2-VL, InternVL2.5, GPT-4o-mini) and RS-specific MLLMs (GeoChat, HRS-Bot). These are the latest top MLLMs in both domains, so comparison against them demonstrates the improvement of EagleVision over prior work.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### Object Detection Results
EagleVision outperforms all baseline detectors across all three datasets. Even the smallest 1B version improves mAP by 3.7% on ShipRSImageNet, 0.9% on MAR20, and 0.1% on FAIR1M compared to the baseline Oriented R-CNN. The largest 7B version under multi-scale testing on FAIR1M outperforms the previous SOTA LSKNet by 0.3% mAP. This confirms that adding attribute understanding training actually improves detection performance, demonstrating the mutual reinforcement between the two tasks.

### Attribute Understanding Results
All prior MLLMs have very low object recall (most below 10% on ShipRSImageNet and FAIR1M; even the best general MLLM Qwen2-VL only has 52.5% recall on MAR20, and GPT-4o-mini only gets an attribute score of 38.0 on ShipRSImageNet). In contrast, EagleVision-7B achieves 79.0% recall and 69.9 attribute score on ShipRSImageNet, 92.8% recall and 91.1 score on MAR20, and 86.6% recall and 75.7 score on FAIR1M, far outperforming all prior MLLMs. This is a massive improvement, solving the core problem of sparse detection and coarse description for RS MLLMs.

## 6.2. Data Presentation (Tables)
The following are the results from Table 4 of the original paper (object detection mAP (%)):

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">ShipRSImageNet</th>
<th rowspan="2">MAR20</th>
<th colspan="6">FAIR1M</th>
</tr>
<tr>
<th>Airplane</th>
<th>Ship</th>
<th>Vehicle</th>
<th>Court</th>
<th>Road</th>
<th>Mean</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9">One-stage Detector</td>
</tr>
<tr>
<td>RetinaNet</td>
<td>20.1</td>
<td>68.6</td>
<td>37.7</td>
<td>11.9</td>
<td>10.8</td>
<td>62.5</td>
<td>21.0</td>
<td>26.6</td>
</tr>
<tr>
<td>R3Det</td>
<td>23.8</td>
<td>65.6</td>
<td>39.0</td>
<td>18.8</td>
<td>18.2</td>
<td>64.8</td>
<td>30.8</td>
<td>31.1</td>
</tr>
<tr>
<td>GGD</td>
<td>26.7</td>
<td>74.3</td>
<td>40.2</td>
<td>13.3</td>
<td>13.2</td>
<td>62.8</td>
<td>26.1</td>
<td>28.1</td>
</tr>
<tr>
<td>KLD</td>
<td>49.2</td>
<td>80.8</td>
<td>39.6</td>
<td>13.2</td>
<td>13.7</td>
<td>63.8</td>
<td>26.4</td>
<td>28.3</td>
</tr>
<tr>
<td>FCOS</td>
<td>56.0</td>
<td>80.2</td>
<td>42.4</td>
<td>23.8</td>
<td>18.9</td>
<td>66.9</td>
<td>35.5</td>
<td>34.1</td>
</tr>
<tr>
<td>S2ANet</td>
<td>49.4</td>
<td>42.6</td>
<td>43.8</td>
<td>23.0</td>
<td>23.4</td>
<td>65.7</td>
<td>28.2</td>
<td>34.7</td>
</tr>
<tr>
<td>TIOE-Det</td>
<td>-</td>
<td>-</td>
<td>45.8</td>
<td>16.9</td>
<td>25.0</td>
<td>69.9</td>
<td>32.7</td>
<td>35.2</td>
</tr>
<tr>
<td>RTMDet</td>
<td>59.2</td>
<td>77.2</td>
<td>44.5</td>
<td>27.2</td>
<td>28.3</td>
<td>70.9</td>
<td>34.3</td>
<td>38.4</td>
</tr>
<tr>
<td colspan="9">Two-stage Detector</td>
</tr>
<tr>
<td>Faster R-CNN</td>
<td>54.8</td>
<td>75.0</td>
<td>48.9</td>
<td>21.4</td>
<td>25.7</td>
<td>65.5</td>
<td>33.0</td>
<td>36.8</td>
</tr>
<tr>
<td>Gliding Vertex</td>
<td>58.6</td>
<td>80.3</td>
<td>46.1</td>
<td>21.4</td>
<td>26.4</td>
<td>67.3</td>
<td>33.5</td>
<td>36.5</td>
</tr>
<tr>
<td>ReDet</td>
<td>53.9</td>
<td>65.5</td>
<td>47.2</td>
<td>21.9</td>
<td>25.3</td>
<td>68.7</td>
<td>30.4</td>
<td>36.5</td>
</tr>
<tr>
<td>KFIoU</td>
<td>37.5</td>
<td>77.0</td>
<td>44.4</td>
<td>25.4</td>
<td>19.2</td>
<td>61.3</td>
<td>26.8</td>
<td>33.7</td>
</tr>
<tr>
<td>ROI Transformer</td>
<td>61.0</td>
<td>82.5</td>
<td>50.8</td>
<td>24.1</td>
<td>28.2</td>
<td>68.3</td>
<td>34.7</td>
<td>39.2</td>
</tr>
<tr>
<td>Oriented R-CNN</td>
<td>63.4</td>
<td>81.8</td>
<td>46.0</td>
<td>28.5</td>
<td>26.0</td>
<td>69.6</td>
<td>35.8</td>
<td>38.5</td>
</tr>
<tr>
<td>Oriented R-CNN*</td>
<td>-</td>
<td>-</td>
<td>53.6</td>
<td>32.2</td>
<td>38.9</td>
<td>73.3</td>
<td>38.2</td>
<td>45.6</td>
</tr>
<tr>
<td>LSKNet*</td>
<td>-</td>
<td>-</td>
<td>53.6</td>
<td>32.8</td>
<td>40.9</td>
<td>76.6</td>
<td>40.8</td>
<td>46.9</td>
</tr>
<tr>
<td colspan="9">Ours (EagleVision)</td>
</tr>
<tr>
<td>EagleVision-1B</td>
<td>67.1</td>
<td>82.7</td>
<td>46.4</td>
<td>28.6</td>
<td>26.1</td>
<td>69.7</td>
<td>35.4</td>
<td>38.6</td>
</tr>
<tr>
<td>EagleVision-2B</td>
<td>71.6</td>
<td>84.0</td>
<td>50.3</td>
<td>27.1</td>
<td>26.6</td>
<td>69.7</td>
<td>31.7</td>
<td>39.2</td>
</tr>
<tr>
<td>EagleVision-4B</td>
<td>73.3</td>
<td>84.3</td>
<td>49.3</td>
<td>29.0</td>
<td>26.3</td>
<td>68.0</td>
<td>30.9</td>
<td>39.0</td>
</tr>
<tr>
<td>EagleVision-7B</td>
<td>74.6</td>
<td>84.5</td>
<td>48.1</td>
<td>29.4</td>
<td>27.6</td>
<td>70.6</td>
<td>36.6</td>
<td>39.9</td>
</tr>
<tr>
<td>EagleVision-7B*</td>
<td>-</td>
<td>-</td>
<td>54.4</td>
<td>33.3</td>
<td>40.6</td>
<td>76.5</td>
<td>41.2</td>
<td>47.2</td>
</tr>
</tbody>
</table>

* * indicates multi-scale testing setting.

  The following are the results from Table 5 of the original paper (object attribute understanding):

  <table>
  <thead>
  <tr>
  <th rowspan="2">Method</th>
  <th colspan="2">ShipRSImageNet</th>
  <th colspan="2">MAR20</th>
  <th colspan="2">FAIR1M</th>
  </tr>
  <tr>
  <td>Recall (%)</td>
  <td>Attribute Score</td>
  <td>Recall (%)</td>
  <td>Attribute Score</td>
  <td>Recall (%)</td>
  <td>Attribute Score</td>
  </tr>
  </thead>
  <tbody>
  <tr>
  <td colspan="7">General MLLMs</td>
  </tr>
  <tr>
  <td>LLaVA-Grounding</td>
  <td>0.5</td>
  <td>3.4</td>
  <td>1.8</td>
  <td>1.5</td>
  <td>1.2</td>
  <td>3.7</td>
  </tr>
  <tr>
  <td>Qwen2-VL</td>
  <td>8.2</td>
  <td>36.2</td>
  <td>52.5</td>
  <td>42.2</td>
  <td>16.9</td>
  <td>40.3</td>
  </tr>
  <tr>
  <td>InternVL2.5</td>
  <td>9.7</td>
  <td>28.9</td>
  <td>21.8</td>
  <td>44.3</td>
  <td>3.2</td>
  <td>44.7</td>
  </tr>
  <tr>
  <td>GPT-4o-mini</td>
  <td>0.7</td>
  <td>38.0</td>
  <td>4.8</td>
  <td>45.7</td>
  <td>3.5</td>
  <td>39.9</td>
  </tr>
  <tr>
  <td colspan="7">Remote Sensing MLLMs</td>
  </tr>
  <tr>
  <td>GeoChat</td>
  <td>1.6</td>
  <td>22.1</td>
  <td>5.9</td>
  <td>19.8</td>
  <td>3.7</td>
  <td>23.5</td>
  </tr>
  <tr>
  <td>HRS-Bot</td>
  <td>7.3</td>
  <td>37.8</td>
  <td>2.0</td>
  <td>27.7</td>
  <td>2.5</td>
  <td>33.4</td>
  </tr>
  <tr>
  <td colspan="7">Ours (EagleVision)</td>
  </tr>
  <tr>
  <td>EagleVision-1B</td>
  <td>77.3</td>
  <td>69.3</td>
  <td>91.6</td>
  <td>86.2</td>
  <td>90.2</td>
  <td>75.0</td>
  </tr>
  <tr>
  <td>EagleVision-2B</td>
  <td>77.1</td>
  <td>68.8</td>
  <td>93.5</td>
  <td>88.8</td>
  <td>89.5</td>
  <td>76.2</td>
  </tr>
  <tr>
  <td>EagleVision-4B</td>
  <td>76.8</td>
  <td>69.5</td>
  <td>94.3</td>
  <td>88.4</td>
  <td>89.5</td>
  <td>76.3</td>
  </tr>
  <tr>
  <td>EagleVision-7B</td>
  <td>79.0</td>
  <td>69.9</td>
  <td>92.8</td>
  <td>91.1</td>
  <td>86.6</td>
  <td>75.7</td>
  </tr>
  </tbody>
  </table>

## 6.3. Ablation Studies / Parameter Analysis
All ablation studies are conducted on ShipRSImageNet, with results shown below:

The following are the results from Table 3 of the original paper (ablation study):

<table>
<thead>
<tr>
<th>Method</th>
<th>Patch Embedding</th>
<th>Vision Token Type</th>
<th>LLM</th>
<th>mAP (%)</th>
<th>Attribute Score</th>
</tr>
</thead>
<tbody>
<tr>
<td>EagleVision-1B†</td>
<td>1×1</td>
<td>Entangled</td>
<td>Qwen2-0.5B-Instruct</td>
<td>56.8</td>
<td>56.8</td>
</tr>
<tr>
<td>EagleVision-1B†</td>
<td>3×3</td>
<td>Entangled</td>
<td>Qwen2-0.5B-Instruct</td>
<td>59.5</td>
<td>63.9</td>
</tr>
<tr>
<td>EagleVision-1B†</td>
<td>5×5</td>
<td>Entangled</td>
<td>Qwen2-0.5B-Instruct</td>
<td>64.4</td>
<td>65.1</td>
</tr>
<tr>
<td>EagleVision-1B†</td>
<td>7×7</td>
<td>Entangled</td>
<td>Qwen2-0.5B-Instruct</td>
<td>62.2</td>
<td>64.3</td>
</tr>
<tr>
<td>EagleVision-1B†</td>
<td>5×5</td>
<td>Disentangled (no orthogonal constraint)</td>
<td>Qwen2-0.5B-Instruct</td>
<td>67.0</td>
<td>66.2</td>
</tr>
<tr>
<td>EagleVision-1B†</td>
<td>5×5</td>
<td>Orthogonal Disentangled</td>
<td>Qwen2-0.5B-Instruct</td>
<td>66.4</td>
<td>67.4</td>
</tr>
<tr>
<td>EagleVision-1B</td>
<td>5×5</td>
<td>Orthogonal Disentangled</td>
<td>Qwen2-0.5B-Instruct</td>
<td>67.1</td>
<td>69.3</td>
</tr>
<tr>
<td>EagleVision-2B</td>
<td>5×5</td>
<td>Orthogonal Disentangled</td>
<td>InternLM2-1.8B</td>
<td>71.6</td>
<td>68.6</td>
</tr>
<tr>
<td>EagleVision-4B</td>
<td>5×5</td>
<td>Orthogonal Disentangled</td>
<td>Phi-3-Mini-128K-Instruct</td>
<td>73.3</td>
<td>69.5</td>
</tr>
<tr>
<td>EagleVision-7B</td>
<td>5×5</td>
<td>Orthogonal Disentangled</td>
<td>InternLM2.5-7B-Chat</td>
<td>74.6</td>
<td>69.9</td>
</tr>
</tbody>
</table>

† † indicates RTMDet is used as baseline detector, otherwise Oriented R-CNN.

Key conclusions from ablation:
1. **Patch Embedding Size**: Increasing patch size from 1×1 to 5×5 improves both mAP (+7.6%) and attribute score (+8.3) because larger patches provide more visual context for attribute learning. However, increasing to 7×7 degrades performance because it includes too much irrelevant background noise around small objects. The optimal size is 5×5, which is used for all other experiments.
2. **Vision Token Type**: Compared to entangled (mixed) tokens, disentangled tokens improve mAP by 2.6% and attribute score by 1.1. Adding the orthogonal constraint further improves attribute score by 1.2, with only a tiny 0.6% drop in mAP. Visualization of token-attribute correlation confirms that orthogonal disentangled tokens have much lower cross-attribute confusion than other token types, verifying that the disentanglement works as intended.
3. **Baseline Detector Compatibility**: Replacing single-stage RTMDet with two-stage Oriented R-CNN improves mAP by 0.7% and attribute score by 1.9, showing that EagleVision is compatible with any detector and consistently improves performance regardless of the baseline.
4. **LLM Scaling**: Increasing LLM size from 1B to 7B consistently improves both mAP (from 67.1% to 74.6%) and attribute score (from 69.3 to 69.9), confirming that larger LLM backbones bring consistent performance gains for this task.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper makes three landmark contributions to remote sensing multimodal large language models:
1. Proposes EagleVision, the first object-level attribute MLLM tailored for remote sensing, which jointly performs precise oriented object detection and dense fine-grained attribute understanding. The novel Attribute Disentangle module with orthogonal subspace learning effectively disentangles attribute-specific visual tokens, solving the problem of mixed attribute features that hurts fine-grained understanding.
2. Creates the first large-scale object attribute dataset EVAttrs-95K (95.1k annotated objects) and the first standardized evaluation benchmark EVBench for object attribute understanding in RS, filling a critical data gap in the field.
3. Extensive experiments confirm that EagleVision achieves state-of-the-art performance on both object detection and attribute understanding, outperforming 15 SOTA traditional detectors and 6 top MLLMs. The core finding that object detection and attribute understanding mutually reinforce each other (attribute training improves detection performance) opens a new direction for MLLM design in vertical domains.

## 7.2. Limitations & Future Work
The authors do not explicitly list limitations, but natural future directions implied by this work are:
1. Extend the EVAttrs-95K dataset to more object categories beyond airplanes and ships, covering other common RS objects like vehicles, buildings, and infrastructure, to demonstrate generalizability across the full domain.
2. Explore end-to-end training of the entire model including the LLM backbone, instead of only fine-tuning visual components with a frozen LLM, to further improve vision-language alignment.
3. Optimize the model for edge deployment, to enable on-orbit processing of RS images directly on satellites, which is a key practical requirement for many applications.
4. Extend the object-level attribute representations to other downstream RS tasks like instance segmentation, change detection, and visual question answering, to leverage the improved feature quality for more use cases.

## 7.3. Personal Insights & Critique
- **Key Inspirations**: This work identifies a critical unaddressed gap in current RS MLLM research: most prior work focused on image-level tasks, but practical RS applications require dense object-level understanding. The finding that attribute understanding training improves detection performance challenges the common assumption that MLLMs are only useful for language tasks and cannot outperform specialist detectors. This idea of mutual reinforcement between detection and understanding is a promising direction for future MLLM design for all vertical domains.
- **Transferability**: The core approach (object-level attribute disentanglement, large-scale attribute instruction tuning, joint optimization of detection and understanding) can be easily transferred to other domains that require fine-grained object-level analysis, such as medical image analysis, autonomous driving, and industrial defect detection.
- **Potential Limitations**: (1) The current dataset only covers airplanes and ships, so generalizability to other object categories remains to be validated. (2) Attribute evaluation relies on GPT-assisted scoring, which can have small biases compared to human evaluation, though prior work has shown it correlates well with human judgment. (3) The method adds a small amount of computational overhead compared to a traditional detector, but the performance gain far outweighs the overhead for almost all practical applications.
- **Overall Impact**: This work establishes a new task direction for RS MLLMs (object-level attribute understanding), provides the first dataset and benchmark for this task, and demonstrates that MLLMs can outperform traditional specialist detectors when properly designed for the domain. This is a significant contribution that will likely inspire substantial follow-up research in remote sensing computer vision.