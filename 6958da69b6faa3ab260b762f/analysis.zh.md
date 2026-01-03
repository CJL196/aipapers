# 1. 论文基本信息

## 1.1. 标题
**VBench: Comprehensive Benchmark Suite for Video Generative Models**
（VBench：视频生成模型的全面评估基准套件）

## 1.2. 作者
**Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, et al.**
来自南洋理工大学 S-Lab (S-Lab, Nanyang Technological University)、上海人工智能实验室 (Shanghai Artificial Intelligence Laboratory)、香港中文大学 (The Chinese University of Hong Kong) 以及南京大学 (Nanjing University) 的研究团队。

## 1.3. 发表期刊/会议
**CVPR 2024** (Conference on Computer Vision and Pattern Recognition)。该会议是计算机视觉领域的顶级国际会议，具有极高的影响力和学术声誉。

## 1.4. 发表年份
**2023年11月29日**（首次发布于 arXiv），后被 CVPR 2024 接收。

## 1.5. 摘要
随着视频生成技术的迅猛发展，如何有效评估这些模型成为了一个巨大的挑战。现有的评估指标（如 FVD、FID）往往与人类的主观感知不一致，且无法提供深入的洞察来指导未来的模型开发。为此，本文提出了 `VBench`。这是一个全面的基准套件，它将“视频生成质量”分解为 **16 个细粒度的、分层的、解耦的评估维度**（如主体身份一致性、运动平滑度、时间闪烁、空间关系等）。每个维度都配有专门的提示词（Prompts）和评估方法。此外，研究者还提供了一个人类偏好标注数据集，证明了 `VBench` 与人类感知的高度一致性。通过 `VBench`，研究者揭示了当前主流视频生成模型的优劣势，并探讨了视频模型与图像模型之间的差距。

## 1.6. 原文链接
*   **arXiv:** [https://arxiv.org/abs/2311.17982](https://arxiv.org/abs/2311.17982)
*   **PDF:** [https://arxiv.org/pdf/2311.17982v1.pdf](https://arxiv.org/pdf/2311.17982v1.pdf)
*   **项目主页:** [https://vchitect.github.io/VBench-project/](https://vchitect.github.io/VBench-project/)

    ---

# 2. 整体概括

## 2.1. 研究背景与动机
近年来，图像生成模型（如 `Diffusion Models`）取得了突破，带动了文本生成视频（Text-to-Video, T2V）研究的热潮。然而，评估生成的视频质量却异常困难：
1.  **现有指标失效:** 传统的 `Fréchet 视频距离 (Fréchet Video Distance, FVD)` 和 `IS (Inception Score)` 往往只给出一个单一的分数，无法反映人类对视频流畅度、真实感或文字对齐度的真实感受。
2.  **缺乏细粒度反馈:** 一个单一的数值无法告诉开发者模型是“画面模糊”还是“动作不连贯”，导致无法针对性地改进训练策略。
3.  **人类感知对齐难题:** 一个理想的评估系统必须与人类的直觉高度对齐，即“人觉得好的视频，分数也应该高”。

## 2.2. 核心贡献/主要发现
*   **提出了 VBench:** 这是一个由 16 个解耦维度组成的全面评估套件，涵盖了视频质量和视频-条件一致性。
*   <strong>解耦评估 (Disentangled Evaluation):</strong> 将复杂的视频质量拆解为具体的维度，例如将“时间连贯性”细化为主体一致性、背景一致性和时间闪烁等。
*   **人类偏好数据集:** 收集了大规模的人类标注数据，验证了 `VBench` 的每个维度都与人类主观判断高度相关。
*   **深度洞察:** 揭示了当前模型在处理复杂动作（如人类动作）时的瓶颈，以及文本生成视频模型在物体组合能力上远逊于文本生成图像（T2I）模型的现状。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
*   <strong>生成式模型 (Generative Models):</strong> 一类能够学习数据分布并生成新样本的模型。本文主要关注基于 <strong>扩散模型 (Diffusion Models)</strong> 的视频生成。
*   <strong>解耦 (Disentanglement):</strong> 在评估中，指将不同的属性（如颜色、动作、清晰度）分开评估，互不干扰。
*   <strong>提示词 (Prompt):</strong> 用户输入给模型的文本描述，指导模型生成特定的内容。

## 3.2. 前人工作与局限
以往的研究主要依赖以下指标：
*   **FVD (Fréchet Video Distance):** 计算生成的视频特征分布与真实视频特征分布之间的距离。
    *   **局限:** 严重依赖预训练的特征提取器，且对视频的局部瑕疵（如轻微闪烁）不敏感。
*   **CLIPSIM:** 使用 `CLIP` 模型计算视频帧与文本描述之间的余弦相似度。
    *   **局限:** 只能评估语义一致性，无法评估运动质量。
*   **VQA (Video Quality Assessment):** 针对真实视频的评估。
    *   **局限:** 无法有效检测生成模型特有的伪影（Artifacts）。

## 3.3. 技术演进与差异化
`VBench` 改变了以往“用一个数字概括一切”的做法。它参考了人类评价视频的逻辑，建立了一套多维度的评价体系。这使得它不仅能作为性能榜单，更能作为一份“诊断报告”。

---

# 4. 方法论

`VBench` 将视频生成质量分为两个顶层维度：<strong>视频质量 (Video Quality)</strong> 和 <strong>视频-条件一致性 (Video-Condition Consistency)</strong>。

下图（原文 Figure 1）展示了 VBench 的评估维度套件、提示套件和评估方法套件：

![该图像是示意图，展示了 VBench 的评估维度套件、提示套件和评估方法套件。图中详细分解了视频生成质量的多个维度，并提供了相应的评估方法和生成视频的示例，为视频生成模型的评估提供了全面框架。](images/1.jpg)
*该图像是示意图，展示了 VBench 的评估维度套件、提示套件和评估方法套件。图中详细分解了视频生成质量的多个维度，并提供了相应的评估方法和生成视频的示例，为视频生成模型的评估提供了全面框架。*

## 4.1. 视频质量 (Video Quality)
视频质量评估不考虑文本，仅关注视频本身是否“看起来好”。它被进一步拆分为 <strong>时间质量 (Temporal Quality)</strong> 和 <strong>帧质量 (Frame-Wise Quality)</strong>。

### 4.1.1. 时间质量维度
为了量化视频在时间维度上的连贯性，`VBench` 提出了以下计算方法：

1.  <strong>主体一致性 (Subject Consistency):</strong>
    评估视频中的主体（如一只猫）在移动过程中外貌是否保持不变。
    使用 `DINO` 特征提取器提取每一帧的特征，并计算首帧与后续帧、相邻帧之间的相似度。
    **核心公式:**
    $$
    S_{subject} = \frac{1}{T-1} \sum_{t=2}^T \frac{1}{2} (\langle d_1 \cdot d_t \rangle + \langle d_{t-1} \cdot d_t \rangle)
    $$
    *   $d_i$: 第 $i$ 帧的 `DINO` 图像特征（已归一化）。
    *   $T$: 视频的总帧数。
    *   $\langle \cdot \rangle$: 点积操作（计算余弦相似度）。

        下图（原文 Figure A8）展示了不同程度的主体一致性：

        ![Figure A8. Visualization of Subject Consistency. We demonstrate different degrees of subject consistency, as indicated by our Subject Consistency score (the larger the better) (a) The cow has a relatively consistent look throughout across different frames. (b) The cow shows inconsistency in its appearance over time. The red boxes indicate areas of subject inconsistency.](images/8.jpg)
        *该图像是插图，展示了不同程度的主体一致性，分为两部分：(a) 一致的主体外观（得分 96.81%），展现牛在不同画面间保持较一致的外观；(b) 不一致的主体外观（得分 83.25%），显示牛在外观上存在较大变化，红框标出不一致区域。*

2.  <strong>背景一致性 (Background Consistency):</strong>
    评估背景场景在相机移动或主体移动时是否稳定。
    方法与主体一致性类似，但使用 `CLIP` 图像编码器，因为它对整体场景特征更敏感。
    **核心公式:**
    $$
    S_{background} = \frac{1}{T-1} \sum_{t=2}^T \frac{1}{2} (\langle c_1 \cdot c_t \rangle + \langle c_{t-1} \cdot c_t \rangle)
    $$
    *   $c_i$: 第 $i$ 帧的 `CLIP` 图像特征。

        下图（原文 Figure A9）展示了背景一致性的可视化：

        ![Figure A9. Visualization of Background Consistency. We showcase varying levels of background consistency, as indicated by our Background Consistency metrics (larger values denote better consistency) (a) The background scene maintains a high degree of consistency (i.e., still the same scene) across different frames. (b) The background exhibits noticeable distortion and abrupt changes over time.](images/9.jpg)
        *该图像是示意图A9，展示了背景一致性的可视化结果。第一部分（a）表现出高背景一致性，得分93.39%，背景场景在不同帧间保持高度一致；第二部分（b）则显示背景不一致，得分88.08%，背景随时间发生显著扭曲和突变。*

3.  <strong>时间闪烁 (Temporal Flickering):</strong>
    生成视频常在局部细节出现快速的高频亮度或像素变动。
    通过计算相邻帧之间的 <strong>平均绝对误差 (Mean Absolute Error, MAE)</strong> 来衡量。
    **核心公式:**
    $$
    S_{flicker} = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{T-1} \sum_{t=1}^{T-1} \mathrm{MAE}(f_i^t, f_i^{t+1}) \right)
    $$
    最终得分会映射到 `[0, 1]`：$S_{flicker-norm} = \frac{255 - S_{flicker}}{255}$。
    *   $f_i^t$: 第 $i$ 个视频的第 $t$ 帧。

        下图（原文 Figure A10）展示了像素级的时间闪烁现象：

        ![Figure A10. Visualization of Temporal Flickering. We demonstrate different degrees of temporal flickering, with a mild occurrence in (a), and a severe occurrence in (b), both reflected by our flicker score metrics (the larger the better). To visualize temporal flickering, given a generated video (top row), we extract a small segment of pixels (marked as the red segment) from each frame at the same location and stack them in frame order (bottom row). (a) Pixel values do not vary abruptly, and the video suffers less from flickering. (b) Pixel values vary abruptly and frequently across different frames, showing strong temporal flickering. Our evaluation metrics also give a lower score.](images/10.jpg)
        *该图像是示意图，展示了不同程度的时间闪烁，左侧为轻微闪烁（分数99.68%），右侧为严重闪烁（分数96.01%）。上方为生成的视频片段，下方为不同帧提取的像素段，红色框标记了相应位置。轻微闪烁的像素值变化平稳，而严重闪烁则明显急剧变化。*

4.  <strong>运动平滑度 (Motion Smoothness):</strong>
    评估动作是否符合物理规律，而非瞬间跳变。
    利用 <strong>视频插帧模型 (Video Frame Interpolation)</strong> 的先验知识。将视频抽掉奇数帧，用插帧模型补回，然后计算补回帧与原始帧的 MAE。差异越小，说明原始运动越平滑。

    下图（原文 Figure A12）展示了运动平滑度的对比：

    ![Figure A12. Visualization of Motion Smoothness. We investigate various levels of motion smoothness, ranging from being smooth as depicted in (a) to highly erratic as depicted in (b), as indicated by our motion score metrics (larger values denote better smoothness). The red boxes indicate areas of discontinuous motion.](images/12.jpg)
    *该图像是插图，展示了运动平滑度的不同级别。图示中（a）代表平滑运动，得分为96.04%；（b）代表不自然运动，得分为88.47%。红框标识了运动不连续的区域。*

5.  <strong>动态程度 (Dynamic Degree):</strong>
    有些模型为了获得高一致性分数，倾向于生成几乎静止的视频。`VBench` 使用 `RAFT` 光流估计来检测视频的运动量，确保模型确实生成了有意义的动作。

    下图（原文 Figure A13）展示了动态与静态视频的区别：

    ![Figure A13. Visualization of Dynamic Degree. We present generated examples of different degrees of motion. (a) In the video, there is obvious motion of the camera and the object, which is identified as dynamic. (b) The video remains almost unchanged from the start to the end and is identified as static.](images/13.jpg)
    *该图像是插图，展示了按动态程度分类的摩托车转弯的生成示例。图中(a)显示了明显运动，评分为1；而(b)则表现为静止状态，评分为0。*

### 4.1.2. 帧质量维度
1.  <strong>美学质量 (Aesthetic Quality):</strong> 使用 `LAION` 美学预测器对每一帧打分（0-10分），评估构图、色彩和艺术感。
2.  <strong>成像质量 (Imaging Quality):</strong> 侧重于低级失真，如噪声、模糊或过度曝光。使用 `MUSIQ` 模型评估。

## 4.2. 视频-条件一致性 (Video-Condition Consistency)
评估生成的视频是否听从了用户的指令。

1.  <strong>物体类别 (Object Class):</strong> 使用 `GRiT` 目标检测器检测提示词中的物体是否出现。
2.  <strong>多物体组合 (Multiple Objects):</strong> 检测多个物体是否在同一帧中同时出现。
3.  <strong>人类动作 (Human Action):</strong> 使用 `UMT` 动作识别模型检测人类主体是否执行了特定的动作（如“跳舞”）。
4.  <strong>空间关系 (Spatial Relationship):</strong> 检测物体之间的方位（如“左边”、“上面”）是否正确。
5.  **风格评估:** 分别使用 `CLIP` 和 `ViCLIP` 特征相似度来评估 <strong>外观风格 (Appearance Style)</strong> 和 <strong>时间风格 (Temporal Style)</strong>（如“延时摄影”、“镜头推进”）。

    ---

# 5. 实验设置

## 5.1. 数据集 (Prompt Suite)
`VBench` 并不直接提供固定视频，而是提供 <strong>提示词套件 (Prompt Suite)</strong>。
*   **维度提示词:** 为 16 个维度各设计了约 100 个专用提示词。
*   **类别提示词:** 将内容分为 8 大类（动物、建筑、食物、人类、生活方式、植物、风景、车辆），每类 100 个提示词。

    下图（原文 Figure 3）展示了提示词的词云分布和各维度的数量统计：

    ![Figure 3. Prompt Suite Statistics. The two graphs provide an overview of our prompt suites. Left: the word cloud to visualize word distribution of our prompt suites. Right: the number of prompts across different evaluation dimensions and different content categories.](images/3.jpg)
    *该图像是图表，展示了我们提示套件的统计信息。左侧为单词云，直观显示提示词的分布情况；右侧为不同评估维度和内容类别下的提示数量统计图，展示了在各维度下的提示数量差异。*

## 5.2. 评估指标
对所有指标，`VBench` 均将其归一化至 `[0, 1]` 或百分比形式，**分数越高代表表现越好**（对于闪烁和噪声指标，也是分数越高表示闪烁/噪声越少）。

## 5.3. 对比基线 (Baselines)
实验对比了四个当时最先进的开源 T2V 模型：
1.  **LaVie**
2.  **ModelScope**
3.  **VideoCrafter**
4.  **CogVideo**

    此外还设置了：
*   **Empirical Max/Min:** 使用真实视频数据（WebVid-10M）或高斯噪声计算出的经验最大值/最小值。
*   **WebVid-Avg:** 真实数据集的平均水平。

    ---

# 6. 实验结果分析

## 6.1. 核心结果分析
以下是原文 Table 1 的完整转录，展示了四种模型在各维度的得分：

<table>
<thead>
<tr>
<th>模型 (Models)</th>
<th>主体一致性 (Subj Cons)</th>
<th>背景一致性 (Back Cons)</th>
<th>时间闪烁 (Temp Flick)</th>
<th>运动平滑度 (Mot Smooth)</th>
<th>动态程度 (Dyn Deg)</th>
<th>美学质量 (Aesthet Q)</th>
<th>成像质量 (Imag Q)</th>
<th>物体类别 (Obj Class)</th>
</tr>
</thead>
<tbody>
<tr>
<td>LaVie</td>
<td>91.41%</td>
<td>97.47%</td>
<td>98.30%</td>
<td>96.38%</td>
<td>49.72%</td>
<td>54.94%</td>
<td>61.90%</td>
<td>91.82%</td>
</tr>
<tr>
<td>ModelScope</td>
<td>89.87%</td>
<td>95.29%</td>
<td>98.28%</td>
<td>95.79%</td>
<td>66.39%</td>
<td>52.06%</td>
<td>58.57%</td>
<td>82.25%</td>
</tr>
<tr>
<td>VideoCrafter</td>
<td>86.24%</td>
<td>92.88%</td>
<td>97.60%</td>
<td>91.79%</td>
<td>89.72%</td>
<td>44.41%</td>
<td>57.22%</td>
<td>87.34%</td>
</tr>
<tr>
<td>CogVideo</td>
<td>92.19%</td>
<td>95.42%</td>
<td>97.64%</td>
<td>96.47%</td>
<td>42.22%</td>
<td>38.18%</td>
<td>41.03%</td>
<td>73.40%</td>
</tr>
</tbody>
</table>

*(注：表中展示的是部分关键维度。完整实验还包括多物体、人类动作、空间关系等维度。)*

**分析发现：**
*   <strong>权衡 (Trade-off):</strong> 模型在 **时间一致性** 和 **动态程度** 之间存在明显的权衡。例如 `LaVie` 一致性极高，但动态程度较低（生成的视频较死板）；而 `VideoCrafter` 动作幅度很大（动态程度 89.72%），但一致性较差。
*   **人类感知对齐:** 下图（原文 Figure 5）展示了 VBench 自动评分与人类手动标注的相关性，拟合直线表明两者高度一致：

    ![该图像是一个示意图，展示了 VBench 中不同视频生成模型与人类评估之间的相关性。每个维度的胜率和相关系数 `ho` 被标注在图中，以评估各模型在主观一致性、运动平滑度等方面的表现。](images/5.jpg)
    *该图像是一个示意图，展示了 VBench 中不同视频生成模型与人类评估之间的相关性。每个维度的胜率和相关系数 `ho` 被标注在图中，以评估各模型在主观一致性、运动平滑度等方面的表现。*

## 6.2. 视频模型 vs 图像模型
研究者将 T2V 模型与 `Stable Diffusion (SDXL)` 等 T2I 模型进行了对比（见 Figure 6a）。
*   **发现:** 在 **多物体组合** 和 **空间关系** 维度上，视频模型表现极其糟糕。这说明目前的视频模型主要在学习如何让画面动起来，而在理解复杂语义组合上还远落后于图像模型。

    ![Figure 6. More Comparisons of Video Generation Models with Other Models and Baselines. We use VBench to evaluate other models and baselines for further comparative analysis of T2V models. (a) Comparison with text-to-image (T2I) generation models. (b) Comparison with WebVid-Avg and Empirical Max baselines. See the Supplementary File for comprehensive numerical results and details on normalization methods.](images/6.jpg)
    *该图像是图表，展示了T2V模型与T2I生成模型（a）以及与WebVid-Avg和Max基线（b）之间的评估比较。通过雷达图形式，各个模型在多个维度上的表现被清晰呈现，包括美学质量、整体一致性和运动平滑度等。*

## 6.3. 内容类别分析
下图（原文 Figure 7）按类别展示了模型表现。可以发现，所有模型在 <strong>人类 (Human)</strong> 类别下的美学评分都较低，即便 `WebVid` 数据集中有 26% 是人类视频。这说明单纯增加数据量无法解决复杂结构（如人体关节）的生成难题。

![该图像是多个视频生成模型在不同评估维度上的性能对比图，包括 LaVie、ModelScope、VideoCrafter 和 CogVideo 四个模型的各项得分，如主体一致性、运动流畅度和总体一致性等，帮助分析模型的优缺点。](images/7.jpg)
*该图像是多个视频生成模型在不同评估维度上的性能对比图，包括 LaVie、ModelScope、VideoCrafter 和 CogVideo 四个模型的各项得分，如主体一致性、运动流畅度和总体一致性等，帮助分析模型的优缺点。*

---

# 7. 总结与思考

## 7.1. 结论总结
`VBench` 为视频生成领域建立了一套科学、全面、且与人类感知对齐的评估标准。它通过 16 个解耦维度的细粒度分析，能够精准地指出一个模型的强项（如：画面清晰度高）与短板（如：动作不平滑）。这对于推动视频生成技术从“能生成视频”向“生成高质量、可控视频”跨越具有重要意义。

## 7.2. 局限性与未来工作
*   **模型覆盖:** 目前仅评估了 4 个模型，未来需要纳入更多商业模型（如 Gen-2, Pika 等）。
*   **安全性评估:** 目前尚未包含对生成内容安全性、偏见和伦理的评估。
*   **任务扩展:** 未来可扩展至图像生成视频 (I2V) 或视频编辑任务的评估。

## 7.3. 个人启发与批判
*   **启发:** `VBench` 的解耦思想非常值得借鉴。在人工智能领域，当我们面对一个复杂的黑盒系统时，最有效的方法就是将其拆解为一个个可观测、可量化的子系统。
*   **批判:** 尽管 `VBench` 宣称是自动化的，但其底层依然依赖了大量的预训练模型（DINO, CLIP, GRiT, UMT 等）。如果这些基础模型本身存在偏差（Bias），那么 `VBench` 的评分也会受到影响。此外，针对“运动平滑度”的插帧模型评估法，可能对那些采用特定帧率训练的模型不够公平。
*   **思考:** 随着 `Sora` 等更强模型的出现，单纯依靠这些基于规则或小模型的评估指标是否还足够？未来或许需要更高阶的“大模型裁判 (LLM-as-a-Judge)”来辅助视频质量评估。