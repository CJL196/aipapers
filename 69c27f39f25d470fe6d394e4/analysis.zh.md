# 1. 论文基本信息
## 1.1. 标题
论文标题为《Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification》，中文译名为《基于形式验证的文本生成视频模型神经符号评估方法》，核心主题是提出一种新的评估框架，用于严谨衡量文本生成视频模型的输出与输入提示的时序对齐程度。
## 1.2. 作者
所有作者均隶属于美国德克萨斯大学奥斯汀分校（The University of Texas at Austin, UT Austin），作者列表为：S P Sharan、Minkyu Choi、Sahil Shah、Harsh Goel、Mohammad Omama、Sandeep Chinchali，该团队主要研究方向为神经符号系统、机器人与生成模型评估。
## 1.3. 发表状态
当前为arXiv预印本，尚未正式发表在学术期刊或会议上。
## 1.4. 发表年份
2024年，预印本发布时间为2024年11月22日（UTC）。
## 1.5. 摘要
现有文本生成视频（T2V）模型（如Sora、Gen-3等）已广泛应用于自动驾驶、娱乐等领域，但现有评估指标仅关注视觉质量与流畅度，忽略了对安全关键场景至关重要的时序保真度与文本-视频对齐能力。为填补该空白，论文提出了名为**NeuS-V**的新型评估指标，通过神经符号形式验证技术严谨评估文本-视频对齐度：首先将输入提示转换为形式化定义的<strong>时序逻辑（Temporal Logic, TL）</strong>规范，再将生成视频转换为自动机表示，最后通过形式化检查视频自动机是否满足时序逻辑规范得到对齐分数。同时论文提出了一个包含时序复杂提示的数据集，用于评估现有SOTA T2V模型。实验表明，NeuS-V与人类评价的相关性是现有指标的5倍以上，且现有T2V模型在时序复杂提示上表现较差，凸显了提升文本生成视频时序对齐能力的必要性。
## 1.6. 原文链接
- 预印本主页：https://arxiv.org/abs/2411.16718
- PDF链接：https://arxiv.org/pdf/2411.16718v5
- 项目开源主页：utaustin-swarmlab.github.io/NeuS-V

# 2. 整体概括
## 2.1. 研究背景与动机
### 2.1.1. 核心问题
随着文本生成视频（T2V）模型的快速发展，其应用逐渐延伸到自动驾驶仿真、机器人训练等安全关键领域，这类场景对生成视频的**时序一致性**和**文本对齐度**要求极高：例如用于自动驾驶训练的视频需要严格满足“卡车在第10帧出现，2秒内并入当前车道”这类时序要求，如果生成视频不符合该时序逻辑，会导致训练出的自动驾驶模型做出错误决策。但现有T2V评估指标普遍存在两大缺陷：
1.  仅关注视觉质量、播放流畅度等表面属性，忽略事件的时序逻辑对齐；
2.  大多基于神经网络黑盒评估（用一个VLM/LLM评估另一个生成模型的输出），结果不严谨、不可解释，无法给出对齐度的严格保证。
### 2.1.2. 研究空白（Gap）
现有评估方法没有将形式化验证方法引入T2V评估领域，缺乏对时序逻辑关系的严格建模能力，无法满足安全关键场景的评估需求。
### 2.1.3. 创新思路
论文结合神经符号方法的优势：用VLM的感知能力提取视频的语义信息，用时序逻辑与形式验证的严谨性建模时序关系，实现可解释、高可靠性的T2V对齐评估。
## 2.2. 核心贡献/主要发现
论文的核心贡献包括四点：
1.  提出了首个基于形式验证的T2V评估框架NeuS-V，实现了严谨、可解释的文本-视频时序对齐评估；
2.  提出了PULS（Prompt Understanding via Temporal Logic Specification）模块，可将自然语言提示自动转换为时序逻辑规范，覆盖对象存在、空间关系、动作对齐、整体一致性四个评估维度；
3.  构建了包含360个时序复杂提示的评估数据集，覆盖自然、人类动物活动、物体交互、驾驶四大主题，按时序复杂度分为基础、中级、高级三个等级；
4.  实验验证：NeuS-V与人类评价的皮尔逊相关系数比主流基准VBench高5倍以上，且现有SOTA T2V模型在时序复杂度高的提示上得分普遍下降30%以上，证明现有模型的时序生成能力存在明显短板。

# 3. 预备知识与相关工作
## 3.1. 基础概念
为方便初学者理解，本部分对论文涉及的核心专业概念进行逐一解释：
### 3.1.1. 文本生成视频（Text-to-Video, T2V）模型
一类生成式人工智能模型，输入为自然语言文本提示，输出为符合文本描述的视频内容，代表模型包括闭源的Gen-3、Pika、Sora，以及开源的CogVideoX、T2V-Turbo等。
### 3.1.2. 神经符号（Neuro-Symbolic）方法
结合了神经网络的感知能力与符号系统的推理能力的一类方法：用神经网络处理非结构化的感知数据（如图像、视频、文本），用符号系统（如逻辑规则、形式验证）实现严谨、可解释的推理，避免纯神经网络黑盒的不可控性。
### 3.1.3. 时序逻辑（Temporal Logic, TL）
一种用于描述事件时序关系的形式化语言，是形式验证的核心工具之一，由三部分组成：
1.  **原子命题**：不可拆分的真假判断语句，如“存在卡车”、“卡车左转”；
2.  **一阶逻辑算子**：包括与（$\wedge$）、或（$\vee$）、非（$\neg$）、蕴含（$\Rightarrow$）等；
3.  **时序算子**：用于描述时间维度的关系，常用的包括：
    - $\square$（Always，始终）：表示命题在所有时间步都为真；
    - $\diamondsuit$（Eventually，最终）：表示命题在未来某个时间步会为真；
    - $\mathbf{U}$（Until，直到）：表示前一个命题始终为真，直到后一个命题为真，之后后一个命题保持为真。
### 3.1.4. 离散时间马尔可夫链（Discrete-Time Markov Chain, DTMC）
一种用于建模离散时间步随机状态转移的数学模型，定义为四元组$\mathcal{A} = (Q, q_0, \delta, \lambda)$：
- $Q$：有限状态集合；
- $q_0$：初始状态；
- $\delta: Q \times Q \to [0,1]$：状态转移函数，表示从一个状态转移到另一个状态的概率，每个状态的所有出边概率之和为1；
- $\lambda: Q \to 2^{|\mathcal{P}|}$：标签函数，将每个状态映射到该状态下为真的原子命题集合。
  DTMC非常适合表示视频序列：视频的每一帧对应一个离散时间步，每个状态对应一帧的语义标签。
### 3.1.5. 形式验证（Formal Verification）
一种基于数学方法严格证明系统满足给定规范的技术，与传统的测试方法相比，形式验证可以给出100%的可靠性保证。本论文中，“系统”是视频对应的DTMC自动机，“规范”是提示转换得到的时序逻辑公式。
### 3.1.6. 概率模型检查（Probabilistic Model Checking）
形式验证的一个分支，用于计算概率系统（如DTMC）满足给定时序逻辑规范的概率，本论文中使用开源工具STORM实现概率模型检查。
### 3.1.7. 视觉语言模型（Vision-Language Model, VLM）
一类可以同时处理图像/视频和文本的多模态大模型，能够回答与视觉内容相关的文本问题，本论文中用VLM检测视频帧中是否存在指定的原子命题（如“是否存在卡车”）。
## 3.2. 前人工作
### 3.2.1. T2V模型评估相关工作
现有T2V评估方法可分为三类：
1.  **视觉质量指标**：如FID、FVD、CLIPSIM等，仅衡量生成视频的视觉真实度，无法衡量语义与时序对齐度；
2.  **VLM-based评估方法**：如EvalCrafter、T2V-Bench等，将提示拆解为VQA问题，用VLM逐帧回答得到对齐分数，这类方法是黑盒评估，结果不严谨，且没有对时序关系进行显式建模；
3.  **综合基准**：如VBench是当前主流的T2V评估基准，覆盖多个视觉维度，但仅有的时序相关评估仅关注播放速度、慢动作等表面属性，不涉及事件的时序逻辑对齐。
### 3.2.2. 视频事件理解相关工作
现有视频事件理解方法大多基于VLM或感知模型检测物体与动作，但无法保证时序关系理解的可靠性。与本论文最相关的工作是NSVS-TL，提出用神经符号方法做视频场景检索，但尚未被用于T2V模型评估。
## 3.3. 技术演进
T2V评估技术的发展脉络可分为三个阶段：
1.  第一阶段：仅用图像质量指标扩展得到的视频指标，完全忽略语义与时序对齐；
2.  第二阶段：引入VLM做语义对齐评估，但为黑盒方法，时序建模能力弱；
3.  第三阶段：本论文首次将形式化时序逻辑与验证方法引入T2V评估，实现严谨的时序对齐评估。
## 3.4. 差异化分析
与现有方法相比，NeuS-V的核心创新点在于：
1.  首次显式用时序逻辑建模提示中的时序关系，而非隐式依赖VLM的黑盒推理；
2.  用形式验证给出对齐度的严谨量化结果，而非近似分数，结果可解释；
3.  是首个重点关注时序逻辑对齐的T2V评估指标，更适合安全关键场景的评估需求。

# 4. 方法论
NeuS-V的核心思路是将文本提示与生成视频都转换为形式化表示，通过概率模型检查计算两者的对齐程度，整体流程如下图（原文Figure 3）所示：

![Figure 3. Spatio-temporal and semantic measurements between a text prompt and a video by NeuS-V. We first decompose the text prompt to TL specification $\\Phi$ , then transform the synthetic video into an automaton representation $\\mathbf { \\mathcal { A } } _ { \\nu }$ . Finally, we calculate the satisfaction probability by probabilistically checking the extent to which $\\mathbf { \\mathcal { A } } _ { \\nu }$ satisfies $\\Phi$ .](images/3.jpg)
*该图像是示意图，展示了通过NeuS-V进行文本提示与生成视频之间的时空和语义测量。图中包括文本提示、时间逻辑规范 $oldsymbol{ ext{Φ}}$、生成的视频以及模型检查过程，通过计算满意度概率来评估视频与文本规范的对齐程度。*

## 4.1. 方法原理
NeuS-V的底层直觉是：自然语言提示中的语义、空间、时序要求可以被严谨地编码为时序逻辑规范，而生成视频的语义与时序演化可以被表示为DTMC自动机，通过概率模型检查计算自动机满足规范的概率，即可得到视频与提示的对齐分数，该方法兼具VLM的感知灵活性与形式验证的严谨性。
## 4.2. 核心方法详解
NeuS-V的完整流程分为5个步骤，对应算法1的实现逻辑：
### 4.2.1. 步骤1：提示转时序逻辑规范（PULS模块）
PULS（Prompt Understanding via Temporal Logic）是论文提出的自然语言提示转时序逻辑规范的模块，支持四个评估维度的规范生成：

| 评估模式 | 评估目标 | 示例时序逻辑规范 |
| --- | --- | --- |
| 对象存在 | 检查提示中要求的对象是否存在 | $\diamondsuit (\text{car} \wedge \text{cyclist} \wedge \text{obstacle})$（最终会同时出现汽车、骑行者、障碍物） |
| 空间关系 | 检查对象之间的空间关系是否符合要求 | $\square (\text{cyclist is in front of car}) \wedge \diamondsuit (\text{obstacle is next to cyclist})$（骑行者始终在汽车前方，且最终障碍物会出现在骑行者旁边） |
| 对象动作对齐 | 检查对象的动作是否符合要求 | $\diamondsuit (\text{cyclist signals turn} \wedge \text{cyclist turns} \mathbf{U} \text{cyclist avoids obstacle})$（最终骑行者会打转向灯，然后转弯直到避开障碍物） |
| 整体一致性 | 检查整个视频的语义与时序是否完全符合提示 | $\square \left( (\text{car driving} \wedge \text{clear day}) \Rightarrow \diamondsuit (\text{cyclist turns} \wedge \text{cyclist avoids obstacle}) \right)$（如果始终有汽车在晴天行驶，那么最终骑行者会转弯并避开障碍物） |

PULS分为两个子模块，基于LLM与少样本优化实现：
#### 4.2.1.1. 文本转原子命题（T2P）模块
该模块从输入提示中提取所有需要检测的原子命题，定义为：
$$
LM_{\mathrm{T2P}}: \mathcal{T} \times M \xrightarrow{\theta_{\mathrm{T2P}}^\star} \mathcal{P}
$$
其中：
- $\mathcal{T}$为输入文本提示，$M$为评估模式；
- $\theta_{\mathrm{T2P}}^\star$为优化后的少样本提示，从训练数据集$\mathfrak{D}_{\mathrm{T2P|train}}$中选择最优的少样本示例得到；
- $\mathcal{P}$为输出的原子命题集合。
#### 4.2.1.2. 文本转时序逻辑（T2TL）模块
该模块根据输入提示、原子命题集合、评估模式生成对应的时序逻辑规范，定义为：
$$
LM_{\mathrm{T2TL}}: \mathcal{T} \times \mathcal{P} \times M \xrightarrow{\theta_{\mathrm{T2TL}}^\star} \Phi
$$
其中：
- $\theta_{\mathrm{T2TL}}^\star$为优化后的少样本提示，从训练数据集$\mathfrak{D}_{\mathrm{T2TL|train}}$中选择最优的少样本示例得到；
- $\Phi$为输出的时序逻辑规范。
  论文使用DSPy框架的MIPROv2算法自动选择最优的少样本示例，最大化PULS的转换准确率。
### 4.2.2. 步骤2：VLM语义置信度计算
对于每个原子命题$p_i \in \mathcal{P}$，用VLM在每段帧序列上检测该命题是否成立，并计算置信度分数：
$$
c_i = \mathcal{M}_{\mathrm{VLM}}(p_i, \mathcal{F}) = \prod_{j=1}^k P\left(t_j \mid p_i, \mathcal{F}, t_1, \dots, t_{j-1}\right) \quad \forall p_i \in \mathcal{P}
$$
其中：
- $\mathcal{M}_{\mathrm{VLM}}$为视觉语言模型，$\mathcal{F}$为输入的帧序列；
- $t_j$为VLM输出响应（是/否）的第$j$个词元；
- $P(t_j | \cdot)$为第$j$个词元的生成概率，通过VLM输出的logit经softmax计算得到：
  $$
  P(t_j | \cdot) = \frac{e^{l_{j,t_j}}}{\sum_z e^{l_{j,z}}}
  $$
  其中$l_{j,t_j}$为第$j$个位置对应词元$t_j$的logit值。
得到原始置信度后，通过校准函数$f_{\mathrm{VLM}}$将其映射为校准后的置信度$c_i^\star$，降低VLM的过自信问题：
$$
c_i^\star \gets f_{\mathrm{VLM}}(c_i; \gamma_{fp})
$$
其中$\gamma_{fp}$为预设的假阳性阈值，基于COCO数据集校准得到。
### 4.2.3. 步骤3：构建视频自动机
基于所有帧的校准后置信度，将生成视频构建为DTMC形式的自动机$\mathcal{A}_\mathcal{V}$：
$$
\mathcal{A}_\mathcal{V} = \xi(\mathcal{P}, C^\star) = \mathcal{A} = (Q, q_0, \delta, \lambda)
$$
其中$C^\star$为所有帧所有原子命题的校准后置信度集合，状态转移概率$\delta(q, q')$的计算公式为：
$$
\delta(q, q') = \prod_{i=1}^{|\mathcal{P}|} (C_i^\star)^{\mathbf{1}_{\{q_i'=1\}}} (1-C_i^\star)^{\mathbf{1}_{\{q_i'=0\}}}
$$
其中：
- $\mathbf{1}_{\{\cdot\}}$为指示函数，条件成立时取值为1，否则为0；
- $q_i'=1$表示原子命题$p_i$在状态$q'$中为真，$q_i'=0$表示为假；
- 该公式表示状态转移概率为所有原子命题在目标状态下的置信度乘积。
  以骑行者的场景为例，视频自动机的结构如下图（原文Figure 2）所示：

  ![Figure 2. Video automaton from the running example. Above is an automaton of a video generated by the Gen-3 model constructed with a TL specification (See Eq. (1)). Every state `q _ { t }` from a frame $\\mathcal { F }$ is labeled and has incoming and outgoing transition probabilities $\\delta ( q , q ^ { \\prime } ) \\in \[ 0 , 1 \]$ . For example, in frame $\\mathcal { F } _ { 8 }$ , we have probabilities $P ( p _ { 4 } ) = 0 . 8$ and $P ( p _ { 5 } ) = 0 . 9$ ,where `p _ { 4 }` represents the atomic proposition "cyclist turns", and `p 5` represents "cyclist avoids obstacle". These probabilities are assigned because the cyclist (red-dotted circle) has turned left to avoid obstacles on the road. In the state `q _ { 2 5 6 }` , we have an incoming probability $0 . 7 2 = P ( p _ { 1 } ) \\times P ( p _ { 2 } ) \\times P ( p _ { 4 } ) \\times P ( p _ { 5 } ) \\times ( 1 - P ( p _ { 3 } ) )$ from the previous state `q _ { 2 2 4 }` , where that label is true for `p _ { 1 } , p _ { 2 } , p _ { 4 }` , and `p _ { 5 }` , and false for `p _ { 3 }` denoted as $\\neg \\{ p _ { 3 } \\}$ .](images/2.jpg)
  *该图像是一个示意图，展示了由Gen-3模型生成的视频自动机。上方展示了三个帧（$F_1$, $F_8$, $F_{11}$）及其对应的概率信息，标记了骑行者的动作。下方为该视频的状态转移图，其中每个状态$q_t$均标记，并显示了状态之间的转移概率，如$P(p_4)=0.8$和$P(p_5)=0.9$。图中还包括初始状态和终止状态的表示。*

### 4.2.4. 步骤4：形式验证计算满足概率
使用概率模型检查工具STORM，计算视频自动机$\mathcal{A}_\mathcal{V}$满足时序逻辑规范$\Phi$的概率：
$$
\mathbb{P}[\mathcal{A}_\mathcal{V} \models \Phi] = \Psi(\mathcal{A}_\mathcal{V}, \Phi)
$$
其中$\Psi(\cdot)$为概率模型检查函数，通过分析自动机的状态转移与标签得到满足概率。
### 4.2.5. 步骤5：校准得到最终NeuS-V分数
将满足概率通过经验累积分布函数（ECDF）映射为最终的NeuS-V分数，每个评估模式得到一个分数$s_m$：
$$
s_m = f_{\mathrm{ECDF}}(\mathbb{P}[\mathcal{A}_\mathcal{V} \models \Phi], D_m) \quad \forall m \in M
$$
其中$D_m$为对应评估模式的满足概率分布，来自大量生成视频的统计结果。最终NeuS-V分数为四个模式得分的平均值：
$$
S_{NeuS-V} = \frac{\sum S}{|S|}
$$
其中$S = \{s_1, s_2, s_3, s_4\}$为四个评估模式的分数集合。
NeuS-V的完整算法流程如下（原文Algorithm 1）：
```
Algorithm 1: NeuS-V
Require: Frame window size w, evaluation mode M , PULS
semantic score mapping function fVLM(·), video automaton generation function ξ(·) probabilistic model checking function Ψ(·), probability mapping function fECDF(·), probability distribution D
Input : Text prompt T, text-to-video model MT2V
Output : NeuS-V score SNeuS-V
1 begin
2 S ← {} // Initialize an empty set for NeuS-V scores of each evaluation mode
3 V ← MT2V(T) // Generate a video
4 for m ∈ M do
5   for n = 0 to length(V) − w step w do
6     C*←{} // Initialize an empty set for semantic confidence
7     F ← {V[n], V[n + 1], . . . , V[n + w − 1]} // Select a sequence of frames
8     P, Φ ← LMPULS(T, m) // Translate the T to P and Φ for m
9     for pi ∈ P do
10      ci ← MvLM(pi, F) // Obtain semantic confidence scores
11      c ← fVLM(ci; γf p) // Calibrate Ci with false positive threshold γfp
12      C*[pi,n] ← c // Append c to the semantic confidence set
13    end for
14  end for
15  Aν ← ξ(P, C*) // Construct Aν
16  P[Aν |= Φ] = Ψ(Aν, Φ) // Obtain satisfaction probability
17  sm = fECDF(P[Aν |= Φ], Dm) // Calculate the calibrated score
18  S← S∪{sm} // Append the score Sm to the set S
19 end for
20 SNeus-V ← ∑ S / |S|
```
# 5. 实验设置
## 5.1. 数据集
实验使用两个数据集验证NeuS-V的性能：
### 5.1.1. NeuS-V时序提示数据集
论文自行构建的时序复杂提示数据集，共包含160个有效提示（总规模360个），覆盖四大主题，按时序复杂度分为三个等级，统计信息如下表（原文Table 5）：

| 主题 | 基础（1个TL算子） | 中级（2个TL算子） | 高级（3个TL算子） | 总提示数 |
| --- | --- | --- | --- | --- |
| 自然 | 20 | 15 | 5 | 40 |
| 人类与动物活动 | 20 | 15 | 5 | 40 |
| 物体交互 | 20 | 15 | 5 | 40 |
| 驾驶数据 | 20 | 15 | 5 | 40 |
| 总计 | 80 | 60 | 20 | 160 |

典型提示示例：
- 基础难度：“雪一直下直到覆盖地面”
- 中级难度：“太阳一直照耀直到云聚集，然后开始下雨”
- 高级难度：“河流始终安静地流过山谷，直到天空被乌云覆盖，然后开始下大雨”
### 5.1.2. MSR-VTT数据集
开源的大规模视频-描述配对数据集，包含10000个视频和对应的人工标注描述，用于验证NeuS-V的跨数据集鲁棒性：将视频与正确描述配对为正样本，与随机描述配对为负样本，测试NeuS-V区分正负样本的能力。
## 5.2. 评估指标
论文使用两个核心评估指标：
### 5.2.1. 皮尔逊相关系数（Pearson Correlation Coefficient）
#### 概念定义
衡量两个连续变量之间线性相关程度的指标，取值范围为$[-1, 1]$，值越接近1表示正相关性越强，越接近-1表示负相关性越强，0表示无相关性。本实验中用于衡量模型给出的分数与人类评价分数的相关性，相关性越高说明评估指标越符合人类判断。
#### 数学公式
$$
r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}
$$
#### 符号解释
- $n$：样本数量；
- $x_i$：第$i$个样本的模型评估分数；
- $y_i$：第$i$个样本的人类评价分数；
- $\bar{x}$：所有模型评估分数的平均值；
- $\bar{y}$：所有人类评价分数的平均值。
### 5.2.2. 正负样本分差
用于MSR-VTT数据集的评估，计算正样本（视频与描述对齐）的平均得分与负样本（视频与描述不对齐）的平均得分的差值，差值越大说明指标区分对齐与不对齐样本的能力越强。
## 5.3. 对比基线
### 5.3.1. 主流评估基准
VBench：当前最主流的T2V评估基准，覆盖多个视觉质量维度，作为对比的核心基线。
### 5.3.2. 纯VLM基线
替换NeuS-V的形式化验证模块，直接用VLM回答拆解的VQA问题得到对齐分数，包括两种配置：
1.  LLaMa-3.2-11B-Vision-Instruct：单帧VLM，逐帧检测命题；
2.  LLaVA-Video-7B-Qwen2：视频专用VLM，直接输入完整视频检测。
### 5.3.3. 待评估T2V模型
选择4个当前SOTA T2V模型：
1.  闭源模型：Gen-3、Pika；
2.  开源模型：T2V-Turbo-v2、CogVideoX-5B。
# 6. 实验结果与分析
## 6.1. 核心结果分析
### 6.1.1. 与人类评价的相关性
NeuS-V与VBench和人类评价的相关性对比如下图（原文Figure 4）所示：

![Figure 4. Correlation with Human Annotations. NeuS-V consistently shows a stronger alignment with human text-to-video annotations (Pearson coefficients displayed at the top of each plot).](images/4.jpg)
*该图像是图表，展示了不同视频生成模型（如Gen-3、Pika、T2V-Turbo-v2和CogVideoX-5B）在文本与视频对齐评估中的相关性。左侧为VBench，右侧为NeuS-V，显示其与人类标注的Pearson相关系数。*

实验结果显示，NeuS-V与人类评价的平均皮尔逊相关系数达到0.72，是VBench（平均0.13）的5倍以上，证明NeuS-V的评估结果更符合人类对文本-视频对齐的判断。
### 6.1.2. 不同T2V模型的性能
4个SOTA T2V模型在NeuS-V提示集上的得分如下表（原文Table 1）：

<table>
<thead>
<tr>
<th colspan="2">提示分类</th>
<th>Gen-3</th>
<th>Pika</th>
<th>T2V-Turbo-v2</th>
<th>CogVideoX-5B</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">按主题</td>
<td>自然</td>
<td>0.716 (0.47)</td>
<td>0.479 (0.70)</td>
<td>0.564 (0.46)</td>
<td>0.580 (0.53)</td>
</tr>
<tr>
<td>人类与动物活动</td>
<td>0.752 (0.80)</td>
<td>0.531 (0.67)</td>
<td>0.564 (0.66)</td>
<td>0.623 (0.43)</td>
</tr>
<tr>
<td>物体交互</td>
<td>0.710 (0.16)</td>
<td>0.500 (0.40)</td>
<td>0.553 (0.66)</td>
<td>0.573 (0.65)</td>
</tr>
<tr>
<td>驾驶数据</td>
<td>0.716 (0.48)</td>
<td>0.525 (0.66)</td>
<td>0.525 (0.30)</td>
<td>0.580 (0.52)</td>
</tr>
<tr>
<td rowspan="3">按复杂度</td>
<td>基础（1个TL算子）</td>
<td>0.774 (0.60)</td>
<td>0.589 (0.70)</td>
<td>0.610 (0.58)</td>
<td>0.641 (0.65)</td>
</tr>
<tr>
<td>中级（2个TL算子）</td>
<td>0.680 (0.27)</td>
<td>0.464 (0.44)</td>
<td>0.508 (0.38)</td>
<td>0.549 (0.28)</td>
</tr>
<tr>
<td>高级（3个TL算子）</td>
<td>0.692 (-0.01)</td>
<td>0.400 (0.33)</td>
<td>0.494 (0.42)</td>
<td>0.550 (0.78)</td>
</tr>
<tr>
<td colspan="2">总得分</td>
<td>0.723 (0.48)</td>
<td>0.508 (0.62)</td>
<td>0.552 (0.55)</td>
<td>0.589 (0.54)</td>
</tr>
</tbody>
</table>

结果显示：
1.  模型排名：Gen-3 > CogVideoX-5B > T2V-Turbo-v2 > Pika，与人类评价的排名一致；
2.  时序复杂度越高，所有模型的得分越低：高级复杂度的平均得分比基础复杂度低15%~20%，证明现有T2V模型的时序生成能力存在明显短板。
### 6.1.3. 跨数据集鲁棒性
NeuS-V与VBench在MSR-VTT数据集上的表现如下表（原文Table 2）：

| 指标 | 对齐描述得分 | 未对齐描述得分 | 分差（越高越好） |
| --- | --- | --- | --- |
| VBench | 0.78 (±0.12) | 0.40 (±0.24) | 0.38 |
| NeuS-V | 0.82 (±0.10) | 0.30 (±0.18) | 0.52 |

NeuS-V的分差比VBench高36.8%，证明NeuS-V区分对齐与不对齐样本的能力更强，鲁棒性更好。
## 6.2. 消融实验分析
### 6.2.1. 形式化方法的重要性
将NeuS-V的时序逻辑与形式验证模块替换为纯VLM基线，得到的相关性结果如下图（原文Figure 6）所示：

![Figure 6. Is Formal Language Important? VLMs without grounding in formal temporal logic lead to lower pearson coefficients (in brackets) as compared to NeuS-V from Figure 4.](images/6.jpg)
*该图像是一个散点图，展示了不同文本到视频生成模型的对齐程度与准确率的关系。图中包含四组模型的评估结果，标出皮尔逊相关系数(r)值，以比较它们在文本到视频对齐方面的表现。*

纯VLM基线的平均皮尔逊相关系数仅为0.2~0.3，远低于NeuS-V的0.72，证明形式化时序逻辑与验证模块是NeuS-V高性能的核心来源。
### 6.2.2. VLM上下文帧数的影响
对比VLM输入1帧与3帧的性能，结果如下表（原文Table 3）：

| VLM上下文 | Gen-3 | Pika | T2V-Turbo-v2 | CogVideoX-5B |
| --- | --- | --- | --- | --- |
| 1帧 | 0.859 (0.29) | 0.613 (0.51) | 0.590 (0.34) | 0.715 (0.32) |
| 3帧 | 0.614 (0.48) | 0.409 (0.62) | 0.417 (0.55) | 0.461 (0.54) |

注：表格中括号内为皮尔逊相关系数，3帧输入的相关性更高，证明引入时序上下文可以提升VLM的检测准确率，进而提升NeuS-V的性能。
### 6.2.3. VLM选型的影响
对比InternVL2-8B与LLaMa3.2-11B的性能，结果如下表（原文Table 4）：

| VLM选型 | Gen-3 | Pika | T2V-Turbo-v2 | CogVideoX-5B |
| --- | --- | --- | --- | --- |
| LLaMa3.2-11B | 0.921 (0.15) | 0.660 (0.16) | 0.730 (0.08) | 0.785 (-0.02) |
| InternVL2-8B | 0.859 (0.29) | 0.613 (0.51) | 0.590 (0.34) | 0.715 (0.3) |

InternVL2-8B的平均相关系数更高，因为LLaMa3.2-11B存在严重的过自信问题，校准后仍然存在偏斜，影响最终性能。
# 7. 总结与思考
## 7.1. 结论总结
论文首次将形式验证方法引入T2V评估领域，提出的NeuS-V框架解决了现有评估指标忽略时序对齐、黑盒不严谨的问题，实验证明其与人类评价的相关性是现有基准的5倍以上。同时论文构建的时序复杂提示数据集与评估结果，揭示了现有SOTA T2V模型在时序生成能力上的明显短板，为未来T2V模型的优化指明了方向。
## 7.2. 局限性与未来工作
### 7.2.1. 局限性
论文指出NeuS-V当前存在两大局限性：
1.  仅检查提示要求的内容是否存在，无法惩罚生成视频中出现的无关内容（如提示要求生成猫，视频中额外出现狗不会被扣分）；
2.  PULS模块与VLM的准确率依赖大模型的性能，对于非常复杂的时序提示可能存在转换或识别误差。
### 7.2.2. 未来工作
作者提出的未来研究方向包括：
1.  扩展NeuS-V的功能，加入对无关内容的惩罚机制；
2.  用NeuS-V的分数作为奖励信号，微调T2V模型或优化提示词，提升生成视频的对齐度；
3.  扩展提示数据集，覆盖更多主题与时序复杂度。
## 7.3. 个人启发与批判
### 7.3.1. 启发
NeuS-V的框架具有很强的可迁移性：
1.  可迁移到其他生成任务的评估，如文本生成3D动画、文本生成机器人控制序列的仿真验证；
2.  可用于安全关键场景的合规性检查，如自动驾驶仿真场景是否符合交通规则的时序要求。
### 7.3.2. 潜在改进方向
1.  可引入时序平滑机制，减少VLM单帧识别误差对自动机构建的影响；
2.  可优化PULS模块，降低对闭源大模型的依赖，支持开源LLM实现提示到时序逻辑的转换；
3.  可扩展支持更丰富的时序算子，覆盖更复杂的时序关系（如事件的持续时间、时间窗口约束等）。