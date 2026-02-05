# 1. Bibliographic Information

## 1.1. Title
Learning Generative Interactive Environments By Trained Agent Exploration

## 1.2. Authors
Naser Kazemi, Nedko Savov, and Danda Pani Paudel. The authors are affiliated with INSAIT (Institute for Computer Science, Artificial Intelligence and Technology) at Sofia University "St. Kliment Ohridski."

## 1.3. Journal/Conference
The paper was published on the arXiv preprint server (arXiv:2409.06445) and was updated in September 2024. While not yet appearing in a traditional journal, it is a significant contribution to the field of `world models` and controllable video generation, building directly upon Google DeepMind's `Genie` framework.

## 1.4. Publication Year
2024 (First published September 10, 2024).

## 1.5. Abstract
`World models` are increasingly pivotal in interpreting and simulating the rules and actions of complex environments. `Genie`, a recent model, excels at learning from visually diverse environments but relies on costly human-collected data. We observe that their alternative method of using random agents is too limited to explore the environment. We propose to improve the model by employing `reinforcement learning` (RL) based agents for data generation. This approach produces diverse datasets that enhance the model's ability to adapt and perform well across various scenarios and realistic actions within the environment. In this paper, we first release the model `GenieRedux` — an implementation based on `Genie`. Additionally, we introduce `GenieRedux-G`, a variant that uses the agent's readily available actions to factor out action prediction uncertainty during validation. Our evaluation, including a replication of the `Coinrun` case study, shows that `GenieRedux-G` achieves superior visual fidelity and controllability using the trained agent exploration. The proposed approach is reproducible, scalable, and adaptable to new types of environments.

## 1.6. Original Source Link
*   **ArXiv Link:** [https://arxiv.org/abs/2409.06445](https://arxiv.org/abs/2409.06445)
*   **PDF Link:** [https://arxiv.org/pdf/2409.06445v2](https://arxiv.org/pdf/2409.06445v2)
*   **Codebase:** [https://github.com/insait-institute/GenieRedux](https://github.com/insait-institute/GenieRedux)

# 2. Executive Summary

## 2.1. Background & Motivation
The paper addresses the challenge of creating `world models`—artificial intelligence systems that can simulate the "physics" and logic of an environment (like a video game) based on actions. A primary bottleneck in training these models is data. The original `Genie` model relied on hundreds of thousands of hours of human gameplay videos, which are expensive and difficult to collect for new tasks. 

While `Genie` suggested using `random exploration` (letting a bot move randomly) as an alternative, the authors of this paper argue that random bots stay stuck near the beginning of levels and never see diverse scenarios. This lack of variety leads to `overfitting`, where the model performs well on starting screens but fails to understand the rest of the environment.

## 2.2. Main Contributions / Findings
*   **GenieRedux:** The first complete open-source PyTorch implementation of the `Genie` architecture, providing a baseline for the research community.
*   **Trained Agent Exploration:** Instead of human data or random bots, the authors use a `Reinforcement Learning (RL)` agent to explore the environment. This agent learns to play the game, thus exploring far more of the environment and generating a much richer training dataset at a fraction of the cost of human data.
*   **GenieRedux-G:** A "Guided" variant of the model that accepts explicit actions as input. This allows the model to bypass the errors inherent in predicting "what the action was" and focus on high-quality video generation.
*   **Performance Gains:** The models trained on `trained agent` data (`-TA` versions) significantly outperformed those trained on `random agent` data (`-Base` versions) in terms of visual quality and how well the character followed the user's commands.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
*   **World Models:** These are models that learn a "mental map" of an environment. They predict what the next state (or frame) will look like given a current state and a specific action.
*   **Video Tokenization (VQ-VAE):** Just as Large Language Models (LLMs) turn words into "tokens," a video tokenizer turns images into a grid of discrete numbers (tokens) from a "codebook." This simplifies the problem from predicting millions of pixel colors to predicting a sequence of integers.
*   **Latent Action Model (LAM):** In scenarios where we have video but no record of what buttons were pressed, a `LAM` looks at two consecutive frames and "guesses" the action that took place between them.
*   **Dynamics Model:** This is the "brain" of the world model. It takes the current frame tokens and an action, then predicts the tokens for the next frame.
*   **Reinforcement Learning (RL):** A type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize a reward. In this paper, `Proximal Policy Optimization (PPO)` is used to train the exploration agent.

## 3.2. Previous Works
*   **Genie (Bruce et al., 2024):** The direct predecessor. It introduced the idea of learning an interactive environment from video alone using a `spatio-temporal transformer`. It used a `Latent Action Model` to infer actions from unlabeled internet videos.
*   **MaskGIT (Chang et al., 2022):** A generative model that predicts image tokens by starting with a fully "masked" (hidden) image and gradually filling in the tokens over several iterations. `Genie` and `GenieRedux` use this for frame prediction.
*   **ST-ViViT (Xu et al., 2020):** A `Spatio-temporal Vision Transformer`. Traditional transformers look at sequences (like sentences). `ST-ViViT` looks at both space (pixels in a frame) and time (how pixels change across frames) using specialized attention blocks.

## 3.3. Technological Evolution
The field has moved from simple `Recurrent Neural Networks (RNNs)` for low-resolution simulation to `Transformer-based` architectures that can generate high-fidelity, controllable video. `Genie` represented a leap toward "zero-shot" controllability (controlling a new image based on patterns learned from others). This paper, `GenieRedux`, refines the data collection pipeline, moving away from expensive human data toward automated, intelligent exploration.

# 4. Methodology

## 4.1. Principles
The core idea is to build a generative environment that is **controllable**. The model must understand that if the "Jump" action is given, the character in the video tokens must move upward. The theoretical basis is the `Discrete Latent Variable Model`, where complex video is compressed into a discrete latent space (tokens) to make the generative task manageable for a `Transformer`.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. The Video Tokenizer (ST-ViViT)
To process video, the model first needs to compress $64 \times 64$ pixel images into a smaller set of tokens. The `ST-ViViT` architecture is used for this purpose. It consists of an **Encoder** that turns a sequence of frames into latent codes and a **Decoder** that reconstructs the image from those codes.

The architecture uses `Spatio-temporal (ST) Blocks`. Instead of one massive attention calculation, it alternates between:
1.  **Spatial Attention:** Looking at relationships between pixels within a single frame.
2.  **Temporal Attention:** Looking at how a specific pixel location changes over time.

    The objective is a `Vector Quantized-Variational Autoencoder (VQ-VAE)` loss, which forces the model to map continuous image features to the nearest entry in a discrete `codebook` of 1024 possible values.

### 4.2.2. The Latent Action Model (LAM)
Because the model often trains on video where the actual joystick inputs are unknown, the `LAM` learns to infer them. It takes two frames, $x_t$ and $x_{t+1}$, and predicts a discrete latent action $a_t$. 

In the `GenieRedux` architecture (seen in Figure 1), the `LAM` is an `ST-ViViT` encoder-decoder that predicts an action token. This token is then used by the dynamics model to understand the transition.

The following figure (Figure 1 from the original paper) shows the architecture of the models:

![Figure 1: Architecture of our models. GenieRedux shares the architecture of Genie; GenieRedux-G takes agent actions as input instead of predicting them.](images/1.jpg)
*该图像是一个示意图，展示了模型GenieRedux和其变体GenieRedux-G的架构。GenieRedux的输入为视频序列，通过视频分词器处理，并包含潜在动作模型和动态模型，而GenieRedux-G则使用代理动作作为输入，以优化动作预测的不确定性。*

### 4.2.3. The Dynamics Model (MaskGIT)
The dynamics model is the core predictive engine. It takes the tokens of the current frame and the action (either from the `LAM` or ground truth) and predicts the tokens for the next frame.

The model uses the `MaskGIT` approach. During training, some percentage of the target frame's tokens are "masked" (hidden). The model learns to predict these missing tokens based on the surrounding context. At inference (generation) time, it uses an iterative decoding process:
1.  Start with all tokens masked.
2.  Predict the most certain tokens.
3.  Use those to predict the next most certain tokens.
4.  Repeat for 25 steps (as specified in Table 6).

### 4.2.4. GenieRedux vs. GenieRedux-G
The paper introduces two variants:
*   **GenieRedux:** Follows the original `Genie`. It predicts actions using the `LAM` and adds those action embeddings to the frame tokens.
*   **GenieRedux-G (Guided):** Takes the actual actions from the `RL agent` (one-hot encoded) and concatenates them with the frame tokens. This version is used to evaluate the exploration strategy without the "noise" of the `LAM` making mistakes about what the action was.

### 4.2.5. Trained Agent Exploration Pipeline
This is the paper's primary innovation. The process is as follows:
1.  **Agent Training:** Train a CNN-based agent using `Proximal Policy Optimization (PPO)` on the environment (e.g., `Coinrun`).
2.  **Data Collection:** Use this trained agent to play the game. Unlike a random agent, this agent reaches the end of levels and interacts with various obstacles.
3.  **Model Training:** Train the `Tokenizer`, `LAM`, and `Dynamics` models on this high-quality, diverse data.

# 5. Experimental Setup

## 5.1. Datasets
The authors use the `Coinrun` environment, a 2D platformer.
*   **Basic Dataset:** Generated by a `random agent`. 88,000 episodes on hard levels. The agent mostly stays near the start.
*   **Diverse Dataset:** Generated by the `trained RL agent`. 10,000 episodes on easy levels. Although the episode count is lower, the content diversity is significantly higher because the agent actually traverses the levels.

## 5.2. Evaluation Metrics
The authors use four primary metrics to evaluate the quality of generated environments:

1.  **FID (Fréchet Inception Distance):**
    *   **Definition:** Measures the similarity between the distribution of generated images and real images. Lower is better (indicating more realistic images).
    *   **Formula:** $d^2 = \|\mu_1 - \mu_2\|^2 + \mathrm{Tr}(C_1 + C_2 - 2\sqrt{C_1 C_2})$
    *   **Symbols:** $\mu_1, \mu_2$ are the mean feature vectors of real and generated images; $C_1, C_2$ are the covariance matrices.

2.  **PSNR (Peak Signal-to-Noise Ratio):**
    *   **Definition:** Measures the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher is better (indicating clearer images).
    *   **Formula:** $\mathrm{PSNR} = 10 \cdot \log_{10}\left(\frac{\mathit{MAX}^2}{\mathit{MSE}}\right)$
    *   **Symbols:** $\mathit{MAX}$ is the maximum pixel value; $\mathit{MSE}$ is the Mean Squared Error between the predicted and ground truth frames.

3.  **SSIM (Structural Similarity Index):**
    *   **Definition:** Measures the perceived quality of digital images by comparing luminance, contrast, and structure. Higher is better (1.0 is perfect).

4.  **$\Delta_{t} \mathrm{PSNR}$:**
    *   **Definition:** Quantifies controllability. It measures the improvement in prediction accuracy when the model is given the correct action versus when it is not. A higher value indicates the model is successfully using the action to change the output.

## 5.3. Baselines
*   **GenieRedux-Base:** Trained using random exploration (the original `Genie` approach for small case studies).
*   **Jafar:** A concurrent JAX-based implementation of `Genie`. The authors compare their results against it to prove the superiority of their PyTorch implementation and exploration strategy.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The results clearly show that `Trained Agent (TA)` data leads to better models.
*   **Visual Fidelity:** Models trained on `TA` data achieved lower `FID` and higher `PSNR` (Table 2 vs. Table 1).
*   **Controllability:** `GenieRedux-G-TA` showed the best performance, with a $\Delta_{t} \mathrm{PSNR}$ of 1.89 compared to 0.70 for the random exploration version (Table 3).
*   **Comparison with Jafar:** `GenieRedux` achieved significantly higher PSNR (17.91 vs 12.66) and lower FID, and did not suffer from "hole digging" artifacts (where the character appears to disappear into the ground).

## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper, showing the baseline model performance on the `Basic Test Set`:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">Basic Test Set</th>
</tr>
<tr>
<th>FID↓</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>Tokenizer-Base</td>
<td>18.14</td>
<td>38.25</td>
<td>0.96</td>
</tr>
<tr>
<td>LAM-Base</td>
<td>37.01</td>
<td>33.97</td>
<td>0.92</td>
</tr>
<tr>
<td>GenieRedux-Base</td>
<td>21.88</td>
<td>25.51</td>
<td>0.77</td>
</tr>
<tr>
<td>GenieRedux-G-Base</td>
<td>18.88</td>
<td>33.41</td>
<td>0.92</td>
</tr>
</tbody>
</table>

The following are the results from Table 2, showing the models trained with the `Trained Agent (TA)`:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">Basic Test Set</th>
</tr>
<tr>
<th>FID↓</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>Tokenizer-TA</td>
<td>12.10</td>
<td>39.53</td>
<td>0.97</td>
</tr>
<tr>
<td>LAM-TA</td>
<td>47.73</td>
<td>28.24</td>
<td>0.85</td>
</tr>
<tr>
<td>GenieRedux-TA</td>
<td>13.26</td>
<td>25.47</td>
<td>0.82</td>
</tr>
<tr>
<td>GenieRedux-G-TA</td>
<td>13.01</td>
<td>32.09</td>
<td>0.94</td>
</tr>
</tbody>
</table>

The following are the results from Table 3, comparing the models on the `Diverse Test Set`:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4">Diverse Test Set</th>
</tr>
<tr>
<th>FID↓</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>∆tPSNR↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>Tokenizer-Base</td>
<td>19.13</td>
<td>35.85</td>
<td>0.94</td>
<td>-</td>
</tr>
<tr>
<td>Tokenizer-TA</td>
<td>11.63</td>
<td>40.62</td>
<td>0.97</td>
<td>-</td>
</tr>
<tr>
<td>GenieRedux-Base</td>
<td>23.97</td>
<td>23.82</td>
<td>0.73</td>
<td>-</td>
</tr>
<tr>
<td>GenieRedux-G-Base</td>
<td>19.51</td>
<td>31.66</td>
<td>0.90</td>
<td>0.70</td>
</tr>
<tr>
<td>GenieRedux-TA</td>
<td>12.57</td>
<td>31.97</td>
<td>0.90</td>
<td>-</td>
</tr>
<tr>
<td>GenieRedux-G-TA</td>
<td>12.40</td>
<td>34.44</td>
<td>0.92</td>
<td>1.89</td>
</tr>
</tbody>
</table>

## 6.3. Qualitative Results
The model demonstrates high-quality controllable generation. As shown in the following figure (Figure 4), given a single starting frame and a sequence of actions, the model successfully predicts the character falling and then jumping:

![Figure 4: GenieRedux-G-TA Qualitative Result. We give a single frame and actions from the test set and we generate 10 frames. In this example our model first successfully progresses the motion of falling. Then, it performs a jump. Ground truth frames are at the top; generated - at the bottom.](images/4.jpg)
*该图像是一个示意图，展示了Ground Truth（上方）与模型预测帧（下方）的对比。我们给出了一帧图像及其对应的动作，生成了10帧，其中模型成功表现出下落的动作，然后进行了跳跃。*

The model's ability to handle all possible actions in the environment is demonstrated here (Figure 2):

![Figure 2: GenieRedux-G-TA Control Demonstration. GenieRedux-G-TA is able to consistently perform all environment actions. Here we demonstrate all of them as generated by the model.](images/2.jpg)
*该图像是一个示意图，展示了GenieRedux-G-TA在环境中的控制演示。展示了不同输入（如向下、跳跃、左移、右移等）对应的环境动作。每个动作的效果通过左侧的输入和右侧的结果进行展示。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully replicates and extends the `Genie` world model. By introducing `trained agent exploration`, the authors provide a pathway to training high-quality, controllable world models without the massive overhead of human data collection. The resulting `GenieRedux-G` model shows superior visual fidelity and responsiveness to user input.

## 7.2. Limitations & Future Work
The authors identify two main failure cases (Figure 11):
1.  **Unknown Environments:** When the character moves into a completely new area of a level not visible in the first frame, the model must "hallucinate" the background. It may generate something that doesn't match the actual game level.
2.  **Motion Ambiguity:** If the model is only given one frame where a character is mid-air, it doesn't know if the character was jumping up or falling down. This can lead to visual artifacts.
3.  **Future Work:** This could be addressed by providing more than one initial frame to the model to give it context on existing momentum and direction.

## 7.3. Personal Insights & Critique
This paper provides a very practical "middle ground" in the world model debate. While Google’s `Genie` proved that scale works, `GenieRedux` proves that **smart data collection** works even better for specific environments. 

The introduction of the `GenieRedux-G` variant is particularly clever for research; by using ground-truth actions, researchers can isolate whether their model's failure is due to poor video generation or poor action inference. 

One critique is that the study is limited to `Coinrun`. While `Coinrun` is a standard benchmark for RL generalization, it is a visually simple 2D game. Future research should investigate if `trained agent exploration` can scale to 3D photorealistic environments (like `GTA` or `CARLA`), where training an RL agent to explore comprehensively is significantly more difficult.