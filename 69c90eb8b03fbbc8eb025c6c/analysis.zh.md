# 1. 论文基本信息
## 1.1. 标题
论文标题为 **EagleVision: Object-level Attribute Multimodal LLM for Remote Sensing**，核心主题是针对遥感（Remote Sensing, RS）领域，提出了一个专门面向对象级属性理解的多模态大语言模型，解决现有方法无法对遥感图像中小目标做精确检测和细粒度属性描述的问题。

## 1.2. 作者
作者分别是Hongxiang Jiang、Jihao Yin、Qixiong Wang、Jiaqi Feng、Guo Chen，其中除Qixiong Wang就职于小红书外，其余作者均隶属**北京航空航天大学**，属于国内顶尖高校的计算机视觉与遥感交叉领域研究团队。

## 1.3. 发表来源
本文目前发布在arXiv预印本平台，arXiv是计算机科学领域认可度最高的预印本平台，本文尚未正式发表在会议或期刊。

## 1.4. 发表时间
2025年3月30日（UTC时间）。

## 1.5. 摘要
现有多模态大语言模型（Multimodal Large Language Model, MLLM）在通用视觉任务已经取得不错效果，但在遥感领域，由于遥感图像高分辨率、目标占比小的特点，现有MLLM难以处理以对象为中心的任务，尤其在精确对象定位和每个对象的细粒度属性描述上存在严重缺陷，性能甚至不如传统经典视觉感知模型，难以落地实际场景。为解决这一问题，本文提出EagleVision，一个专门为遥感设计的MLLM，同时具备出色的对象检测和属性理解能力；模型配备属性解耦模块，能够学习解耦的视觉词元来表达不同对象的独立属性；为支持对象级视觉语言对齐，本文构建了首个大规模遥感对象属性理解指令微调数据集EVAttrs-95K，以及对应的评测基准EVBench。实验证明EagleVision在细粒度对象检测和对象属性理解两个任务都达到了最先进性能，验证了MLLM中检测能力和理解能力存在互相促进的关系。

## 1.6. 原文链接
- 原文链接：https://arxiv.org/abs/2503.23330
- PDF链接：https://arxiv.org/pdf/2503.23330
- 发布状态：预印本，尚未正式发表。

# 2. 整体概括
## 2.1. 研究背景与动机
遥感图像分析在国土资源监测、军事侦察、港口管理、灾害评估等领域有重大应用价值，现有研究的核心痛点如下：
1. **传统视觉感知模型的缺陷**：传统遥感对象检测模型只能输出预定义类别的标签，无法给出每个对象的细粒度属性描述，遇到未知类别只能标注为"其他"，缺乏可解释性，无法满足实际应用的深度分析需求。
2. **现有开放世界检测的缺陷**：基于文本参考的开放世界检测模型（如Grounding-DINO）依赖用户提供的参考文本才能定位对象，无法自动对图像中所有对象做属性理解，不适合遥感场景的全图分析需求。
3. **现有MLLM的缺陷**：通用MLLM和现有遥感MLLM都只能做粗粒度的图像级理解，由于遥感图像分辨率高、目标占比小，现有MLLM普遍存在严重的漏检问题，只能给出稀疏、粗糙的描述，无法对每个对象做细粒度属性描述，性能远达不到实用要求。

   本文的切入点是：将对象检测和细粒度属性理解结合到同一个MLLM框架中，通过属性解耦学习实现每个对象的独立属性表达，同时构建大规模数据集和评测基准，验证双向促进的效果，填补领域空白。

## 2.2. 核心贡献/主要发现
本文的核心贡献包括三点：
1. **模型架构创新**：提出了首个面向遥感的对象级属性多模态大语言模型EagleVision，同时实现精确对象检测和细粒度属性理解；提出了属性解耦模块，通过正交子空间学习得到解耦的属性视觉词元，解决了原始特征混合多个属性的问题，提升细粒度理解能力。
2. **数据和基准创新**：首次构建了大规模遥感对象属性理解数据集EVAttrs-95K，包含95.1k个对象的细粒度属性标注，同时构建了对应的评测基准EVBench，为该领域后续研究提供了数据和评测基础。
3. **实验验证了核心结论**：通过大量实验证明，EagleVision在两个任务都达到了最先进性能，对象属性理解的训练能够反过来提升对象检测的性能，验证了MLLM中检测和理解能力互相促进的结论。

# 3. 预备知识与相关工作
## 3.1. 基础概念
对初学者需要掌握的核心基础概念解释如下：
1. <strong>多模态大语言模型（MLLM）</strong>：结合了大语言模型（LLM）的语言理解与生成能力，以及视觉编码器的视觉特征提取能力，能够处理图文多模态输入，完成视觉问答、图像描述、目标定位等任务的人工智能模型。
2. <strong>遥感（RS）</strong>：通过卫星、航空飞行器等远距离传感器获取地球表面图像的技术，是国土资源、军事、环境等领域重要的信息获取方式。
3. **对象检测**：计算机视觉核心任务之一，目标是识别图像中所有感兴趣目标，输出每个目标的位置坐标和类别。
4. **属性解耦**：表示学习中的技术，目标是让神经网络学习到的不同特征分别对应对象的不同属性，每个特征只编码一种属性的信息，不同属性之间的特征互不干扰，从而提升模型对细粒度信息的表达能力。
5. **指令微调**：大语言模型领域的微调技术，使用大量格式化为「指令-输出」对的数据微调模型，让模型能够更好地理解人类指令，适配不同下游任务。

## 3.2. 前人工作
前人工作可以分为两大类：
1. <strong>传统视觉感知模型（对象检测）</strong>
   - 传统遥感对象检测：主流方法包括单阶段的R3Det、RTMDet，双阶段的Faster R-CNN、Oriented R-CNN等，这些方法都只能输出预定义类别的粗粒度标签，无法给出细粒度属性描述，对未知类别泛化能力差。
   - 开放世界视觉 grounding 模型：如PolyFormer、Grounding-DINO，这类方法支持开放类别检测，但严重依赖用户提供的参考文本，无法自动对所有对象做属性描述，不适合遥感全图分析场景。
   - 自然图像属性识别模型：如OvarNet、TAP，这类方法依赖CLIP的对比检索，无法生成自由格式的属性描述，很难泛化到遥感领域。

2. **多模态大语言模型**
   - 通用MLLM：如LLaVA、LLaVA-Grounding、Qwen2-VL等，通用MLLM要么侧重全局图像级理解，要么只能做稀疏的对象级理解，在遥感场景下由于目标小、分辨率高，漏检严重，属性描述粗糙，无法满足要求。
   - 现有遥感MLLM：如RSGPT、GeoChat、RSUniVLM，这些方法都侧重图像级问答和通用多任务对话，没有解决对象级细粒度属性理解和检测提升的核心问题。

## 3.3. 技术演进
遥感图像理解领域的技术演进路径：
`粗粒度图像分类 → 粗粒度对象检测（仅输出类别）→ 开放类别检测（依赖参考文本）→ 图像级多模态对话（MLLM）→ 对象级细粒度属性理解+检测一体化（本文工作）`
本文工作是遥感MLLM从粗粒度到细粒度、从图像级到对象级的重要演进，填补了领域空白。

## 3.4. 差异化分析
本文方法和现有工作的核心差异可以用原文的对比表格说明：
以下是原文Table 1的对比结果：

<table>
<thead>
<tr>
<th>方法类别</th>
<th>方法</th>
<th>需要参考文本</th>
<th>理解粒度</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">传统视觉感知模型</td>
<td>DETR</td>
<td>✗</td>
<td>无属性理解</td>
</tr>
<tr>
<td>KFIoU</td>
<td>✗</td>
<td>无属性理解</td>
</tr>
<tr>
<td>PolyFormer</td>
<td>✓</td>
<td>无属性理解</td>
</tr>
<tr>
<td>Grounding-DINO</td>
<td>✓</td>
<td>无属性理解</td>
</tr>
<tr>
<td rowspan="5">多模态大语言模型</td>
<td>LLaVA</td>
<td>✓</td>
<td>图像级</td>
</tr>
<tr>
<td>LLaVA-Grounding</td>
<td>✗</td>
<td>稀疏对象级</td>
</tr>
<tr>
<td>GeoChat</td>
<td>✓</td>
<td>稀疏对象级</td>
</tr>
<tr>
<td>GPT-4o</td>
<td>✓</td>
<td>稀疏对象级</td>
</tr>
<tr>
<td>EagleVision (本文)</td>
<td>✗</td>
<td>密集对象级</td>
</tr>
</tbody>
</table>

可以看到，本文是首个不需要参考文本、能够对所有对象做密集细粒度属性理解的方法，同时还能提升对象检测性能，这是和现有工作最核心的差异。

# 4. 方法论
## 4.1. 方法原理
EagleVision的整体架构分为三个核心部分：基线检测器提取候选对象特征、属性解耦模块将混合特征分解为独立属性的视觉词元、冻结大语言模型生成对象级属性描述。训练时同时优化检测损失、正交约束损失、属性匹配损失和语言生成损失，实现检测和属性理解的双向促进。核心直觉是：解耦的属性特征能够帮助大语言模型更好地做细粒度理解，而属性理解的训练又能让视觉编码器学习到更具区分度的对象特征，反过来提升检测性能。

## 4.2. 核心方法详解
### 4.2.1 基线检测器（Baseline Detector）
对于输入的遥感图像 $X_v \in \mathbb{R}^{H \times W \times 3}$，其中$H$是图像高度、$W$是图像宽度、3是RGB颜色通道，使用任意单阶段或双阶段检测器提取候选区域（ROI）特征：
$$F_v = f(X_v; \theta)$$
符号解释：
- $F_v \in \mathbb{R}^{N \times H' \times W' \times C}$：输出的候选区域特征，$N$是候选对象数量，`H'、W'`是候选特征图的高和宽，$C$是特征通道数
- $f(\cdot)$：检测器网络，$\theta$是检测器的可训练参数

  检测器保留分类头$f_{cls}$和回归头$f_{reg}$，分别预测对象类别和旋转 bounding box 的位置，之后筛选出所有前景对象（即确实是目标的对象）的ROI特征，记为 $F_v^{pos} \in \mathbb{R}^{N_{pos} \times H' \times W' \times C}$，$N_{pos}$是前景对象的数量，供后续处理。

训练时，检测损失$\mathcal{L}_d$和传统检测器一致，包含交叉熵分类损失、L1位置损失或旋转IoU损失，检测器的所有参数都可训练。

### 4.2.2 属性解耦模块（Attribute Disentangle）
原始的$F_v^{pos}$混合了所有属性的信息，无法为大语言模型提供细粒度的属性特征，因此需要做解耦处理，步骤如下：
#### 步骤1：采样邻域得到补丁嵌入
对于单阶段检测器，通常每个对象只输出中心位置的特征，大小为$H'=W'=1$，因此需要以对象中心$r_i=(x_i,y_i)$为中心，采样大小为$(2s+1) \times (2s+1)$的邻域特征，邻域集合定义为：
$$
\begin{aligned}
& R = \{ r_i \}_{i=1,2,\dots,N_{pos}}, r_i = (x_i, y_i) \\
& S_i = \{ (x_i + s_x, y_i + s_y) | s_x, s_y \in [-s, s] \},
\end{aligned}
$$
对于双阶段检测器，直接将ROI特征调整为$(2s+1) \times (2s+1)$大小即可。最终得到所有对象的补丁嵌入 $E_v \in \mathbb{R}^{N_{pos} \times (2s+1) \times (2s+1) \times C}$。

#### 步骤2：正交子空间投影得到解耦视觉词元
本文学习一组正交基$p_1, p_2, ..., p_n$，每个基对应一个独立的属性子空间，$n$是属性基的数量，对应需要学习的属性数量。将补丁嵌入$E_v$投影到每个基上，得到解耦后的视觉词元，公式如下：
$$
\begin{aligned}
& \boldsymbol{T_v} = cat(\boldsymbol{T_v^1}, \boldsymbol{T_v^2}, ..., \boldsymbol{T_v^n}) \\
& \boldsymbol{T_v^k} = c_k p_k, \quad c_k = \sum_{i}^{2s+1} \sum_{j}^{2s+1} E_v^{i,j} p_k^T,
\end{aligned}
$$
符号解释：
- $p_k \in \mathbb{R}^{1 \times C}$：第k个属性基的可学习参数
- $E_v^{i,j} \in \mathbb{R}^{N_{pos} \times C}$：补丁嵌入中第i行第j列的特征
- $cat(\cdot)$：张量拼接操作
- $T_v \in \mathbb{R}^{N_{pos} \times n \times C}$：最终解耦后的视觉词元，每个对象有n个独立的词元，每个词元对应一个属性。

#### 步骤3：解耦约束损失
为了保证解耦效果，添加两个约束损失：
1. **正交损失$\mathcal{L}_o$**：约束不同属性基之间正交，让不同属性的子空间互不相关，公式：
   $$
\mathcal{L}_o = \frac{2}{n \times (n - 1)} \sum_{i=1}^{n} \sum_{j > i}^{n} |p_i p_j^T|
$$
我们期望当$i \neq j$时，$p_i p_j^T = 0$（完全正交），因此最小化所有两两基之间内积的绝对值，式子前的系数是归一化因子，对所有两两组合的结果做平均。

2. **属性匹配损失$\mathcal{L}_a$**：最大化每个投影特征$c_k$和对应真实属性编码之间的互信息，保证每个基确实学到了对应的属性，公式：
   $$
\mathcal{L}_a = - \frac{1}{n} \sum_{k}^{n} I(c_k, T_a^k)
$$
其中$I(\cdot, \cdot)$是互信息，$T_a^k$是第k个真实属性的编码。由于互信息无法直接计算，本文通过优化变分下界得到可计算的损失：
$$
\begin{aligned}
\mathcal{L}_a &= \frac{1}{n} \sum_{k}^{n} (q(T_a^k; \varphi) - c_k)^2 \\
&= - \frac{1}{n} \sum_{k}^{n} \mathbb{E}_{T_a^k} [ \mathbb{E}_{c_k \sim P(c_k | T_a^k)} [ log(Q(c_k | T_a^k)) ] ] \\
&\geq - \frac{1}{n} \sum_{k}^{n} I(c_k, T_a^k) + H(c),
\end{aligned}
$$
符号解释：$Q(c_k | T_a^k) \sim \mathcal{N}(q(T_a^k; \varphi), I)$是变分分布，$q(\cdot; \varphi)$是预测网络，参数为$\varphi$，`H(c)`是$c$的熵，为常数，因此优化这个均方误差等价于最大化互信息，实现属性和特征的一对一匹配。注意$T_a^k$仅在训练时使用，推理时不需要。

### 4.2.3 对象级描述生成
将用户指令提示编码得到文本词元$T_q$，和已经解耦好的视觉词元$T_v$拼接后，输入到冻结的大语言模型中，生成每个对象的属性描述，过程表示为：
$$Y = g(T_v, T_q; \phi)$$
符号解释：$g(\cdot)$是大语言模型，$\phi$是大语言模型的参数，训练时保持冻结，只优化视觉部分的参数；$Y$是模型生成的属性描述。

根据生成结果$Y$和真实标注$\hat{Y}$，计算下一个词元预测的语言损失$\mathcal{L}_q$，这个过程实现了对象级视觉特征和大语言模型文本空间的对齐，类似LLaVA的对齐过程。

### 4.2.4 总损失函数
整个模型的总损失是所有损失的加权和：
$$
\mathcal{L}_{overall} = \lambda_d \mathcal{L}_d + \lambda_o \mathcal{L}_o + \lambda_a \mathcal{L}_a + \lambda_q \mathcal{L}_q
$$
其中每个$\lambda$是权重系数，用来平衡不同损失的贡献。

### 4.2.5 EVAttrs-95K数据集构建
为了支持指令微调，本文构建了首个大规模遥感对象属性数据集EVAttrs-95K，总共包含95.1k个对象的属性标注，数据来自三个公开遥感数据集，分布如下：
以下是原文Table 2的EVAttrs-95K数据分布：

<table>
<thead>
<tr>
<th>数据来源</th>
<th>FAIR1M</th>
<th>MAR20</th>
<th>ShipRSImageNet</th>
</tr>
</thead>
<tbody>
<tr>
<td>总对象数</td>
<td>59.8k</td>
<td>22.3k</td>
<td>13.0k</td>
</tr>
<tr>
<td>训练集对象数</td>
<td>44.2k</td>
<td>7.8k</td>
<td>10.1k</td>
</tr>
<tr>
<td>测试集对象数</td>
<td>15.6k</td>
<td>14.5k</td>
<td>2.9k</td>
</tr>
<tr>
<td>平均每个对象属性数</td>
<td>~25</td>
<td>~24</td>
<td>~28</td>
</tr>
</tbody>
</table>

构建流程为**两阶段自动标注+人工精修**：
1. 数据预处理：从三个原始数据集裁剪飞机和船舶的对象块，预定义飞机有24个细粒度属性，船舶有38个细粒度属性，包含颜色、大小、类型、结构、状态等。
2. 两阶段自动标注：第一阶段用Qwen2-VL-72B对所有样本做标注，要求输出JSON格式并给出置信度；第二阶段用GPT-4o重新标注置信度低于0.5的低质量样本。
3. 人工精修：人工检查所有置信度低于0.7的标注，修正错误描述，删除不确定的标注，最终得到高质量的数据集。

### 4.2.6 EVBench评测基准
为了评测对象属性理解能力，本文构建了EVBench，评测流程如下：
1. 数据划分：遵循表2的划分得到测试集。
2. 响应预处理：将模型生成的描述和真实标注都转换为JSON格式，key为属性名，value为属性描述。
3. 评测指标：
   - 对象召回率：被检测到（输出非空描述）的对象占总真实对象的比例，衡量模型的对象检测能力。
   - 属性得分：使用GPT-3.5-turbo作为评判员，对每个属性的生成结果从正确性和表达性两个维度打分（1-5分），然后平均后缩放到0-100分，总得分是所有属性得分的平均值。

# 5. 实验设置
## 5.1. 数据集
实验在三个公开遥感细粒度对象检测数据集上进行：
1. **FAIR1M-v1.0**：高分辨率遥感细粒度对象识别基准，包含飞机、船舶、车辆等五大类37个子类，实验中将原始图像裁剪为$1024 \times 1024$的子图，块之间重叠200像素。
2. **MAR20**：遥感图像军用飞机识别基准，包含20类不同的军用飞机，实验中将图像直接缩放为$1024 \times 1024$。
3. **ShipRSImageNet**：高分辨率遥感船舶检测数据集，包含50类不同的船舶，实验中将图像直接缩放为$1024 \times 1024$。

   本文的EVAttrs-95K就是基于这三个数据集做的属性标注，EVBench基于上述测试集构建，三个数据集都是遥感领域权威的细粒度对象检测数据集，覆盖了飞机、船舶这两个最需要属性分析的常见遥感目标，能够有效验证模型性能。

## 5.2. 评估指标
实验包含两个任务，对应不同的评估指标：
### 5.2.1 对象检测任务：平均精度均值（mAP）
1. **概念定义**：mAP是对象检测任务的标准评测指标，同时考虑了检测的精确率和召回率，衡量模型对所有类别对象的检测精度，值越高说明检测性能越好。
2. **数学公式**：
   首先对每个类别$c$计算平均精度AP：
$$
AP_c = \int_{0}^{1} P(R) dR
$$
然后对所有类别的AP取平均得到mAP：
$$
mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c
$$
3. **符号解释**：
- $C$：类别总数
- $P$：精确率，即检测为正的样本中真实正样本的比例
- $R$：召回率，即所有真实正样本中被检测出来的比例
- `P(R)`：精确率随召回率变化的曲线
- mAP范围为0-100%，越高越好。

### 5.2.2 对象属性理解任务：两个指标
#### （1）对象召回率
1. **概念定义**：衡量模型成功检测并输出属性描述的对象占所有真实对象的比例，反映模型发现对象的能力，越高越好。
2. **数学公式**：
   $$
Recall = \frac{N_{detected}}{N_{total}}
$$
3. **符号解释**：
- $N_{detected}$：成功输出非空属性描述的对象数量
- $N_{total}$：真实对象的总数量
- 范围0-100%，越高越好。

#### （2）属性得分
1. **概念定义**：基于GPT辅助评估的得分，衡量模型生成的属性描述的正确性和完整性，范围0-100，越高越好。
2. **数学公式**：
   $$
Score = \frac{1}{n_{all}} \sum_{i=1}^{n_{all}} s_i \times 20
$$
3. **符号解释**：
- $n_{all}$：所有对象所有属性的总数量
- $s_i$：GPT对第i个属性的打分，范围1-5分
- 乘以20将得分缩放为0-100分。

## 5.3. 对比基线
- **对象检测任务**：对比了15个遥感对象检测领域最先进的模型，包括单阶段的RetinaNet、R3Det、RTMDet等，双阶段的Faster R-CNN、Oriented R-CNN、LSKNet等，覆盖了传统检测的所有代表性SOTA模型。
- **对象属性理解任务**：对比了6个先进MLLM，包括通用MLLM：LLaVA-Grounding、Qwen2-VL、InternVL2.5、GPT-4o-mini，遥感MLLM：GeoChat、HRS-Bot，覆盖了通用和遥感领域最先进的MLLM。

# 6. 实验结果与分析
## 6.1. 核心结果分析
### 6.1.1 对象检测结果
以下是原文Table 4的对象检测mAP结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">ShipRSImageNet</th>
<th rowspan="2">MAR20</th>
<th colspan="6">FAIR1M</th>
</tr>
<tr>
<td>Airplane</td>
<td>Ship</td>
<td>Vehicle</td>
<td>Court</td>
<td>Road</td>
<td>Mean</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8">One-stage Detector</td>
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
<td colspan="8">Two-stage Detector</td>
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
<td colspan="8">Ours</td>
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

注：*表示多尺度训练测试设置。

从结果可以得到：
- EagleVision在三个数据集都超过了基线检测器，单尺度设置下，最佳的EagleVision-7B在ShipRSImageNet的mAP是74.6%，比之前最好的Oriented R-CNN提升了11.2%；在MAR20是84.5%，比之前最好的ROI Transformer提升了2.0%；
- 多尺度设置下，EagleVision-7B*在FAIR1M的平均mAP是47.2%，比之前最好的LSKNet*提升了0.3%，达到了最先进的检测性能；
- 哪怕本文只给飞机和船舶做了属性标注，FAIR1M中其他类别的检测性能也有提升，证明属性理解确实能够促进检测性能，验证了本文的核心结论。

### 6.1.2 对象属性理解结果
以下是原文Table 5的对象属性理解结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">ShipRSImageNet</th>
<th colspan="2">MAR20</th>
<th colspan="2">FAIR1M</th>
</tr>
<tr>
<td>Recall</td>
<td>Score</td>
<td>Recall</td>
<td>Score</td>
<td>Recall</td>
<td>Score</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7">General MLLMs</td>
</tr>
<tr>
<td>LLaVA-Grounding</td>
<td>0.5%</td>
<td>3.4</td>
<td>1.8%</td>
<td>1.5</td>
<td>1.2%</td>
<td>3.7</td>
</tr>
<tr>
<td>Qwen2-VL</td>
<td>8.2%</td>
<td>36.2</td>
<td>52.5%</td>
<td>42.2</td>
<td>16.9%</td>
<td>40.3</td>
</tr>
<tr>
<td>InternVL2.5</td>
<td>9.7%</td>
<td>28.9</td>
<td>21.8%</td>
<td>44.3</td>
<td>3.2%</td>
<td>44.7</td>
</tr>
<tr>
<td>GPT-4o-mini</td>
<td>0.7%</td>
<td>38.0</td>
<td>4.8%</td>
<td>45.7</td>
<td>3.5%</td>
<td>39.9</td>
</tr>
<tr>
<td colspan="7">Remote Sensing MLLMs</td>
</tr>
<tr>
<td>GeoChat</td>
<td>1.6%</td>
<td>22.1</td>
<td>5.9%</td>
<td>19.8</td>
<td>3.7%</td>
<td>23.5</td>
</tr>
<tr>
<td>HRS-Bot</td>
<td>7.3%</td>
<td>37.8</td>
<td>2.0%</td>
<td>27.7</td>
<td>2.5%</td>
<td>33.4</td>
</tr>
<tr>
<td colspan="7">Ours</td>
</tr>
<tr>
<td>EagleVision-1B</td>
<td>77.3%</td>
<td>69.3</td>
<td>91.6%</td>
<td>86.2</td>
<td>90.2%</td>
<td>75.0</td>
</tr>
<tr>
<td>EagleVision-2B</td>
<td>77.1%</td>
<td>68.8</td>
<td>93.5%</td>
<td>88.8</td>
<td>89.5%</td>
<td>76.2</td>
</tr>
<tr>
<td>EagleVision-4B</td>
<td>76.8%</td>
<td>69.5</td>
<td>94.3%</td>
<td>88.4</td>
<td>89.5%</td>
<td>76.3</td>
</tr>
<tr>
<td>EagleVision-7B</td>
<td>79.0%</td>
<td>69.9</td>
<td>92.8%</td>
<td>91.1</td>
<td>86.6%</td>
<td>75.7</td>
</tr>
</tbody>
</table>

从结果可以看出：
- 现有通用和遥感MLLM在该任务上性能极差，召回率普遍低于20%，属性得分最高的GPT-4o-mini在ShipRSImageNet也只有38分；
- EagleVision的召回率和得分都远超所有对比模型，EagleVision-7B在ShipRSImageNet达到79%召回和69.9分，在MAR20达到92.8%召回和91.1分，在FAIR1M达到86.6%召回和75.7分，优势极其显著。

  可视化结果也证明，EagleVision不仅能准确检测所有对象，还能给出每个对象完整的细粒度属性描述，比如能识别出"带有直升机停机坪的登陆舰"，而传统检测只能标注"其他船舶"，现有MLLM会漏检大部分对象，效果远不如EagleVision。

## 6.2. 消融实验/参数分析
消融实验在ShipRSImageNet上进行，验证各个组件的有效性，结果如下：
以下是原文Table 3的消融实验结果：

<table>
<thead>
<tr>
<th>Method</th>
<th>Patch Embedding Size</th>
<th>Vision Token Type</th>
<th>LLM</th>
<th>mAP</th>
<th>Score</th>
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
<td>Disentangled (only $\mathcal{L}_a$)</td>
<td>Qwen2-0.5B-Instruct</td>
<td>67.0</td>
<td>66.2</td>
</tr>
<tr>
<td>EagleVision-1B†</td>
<td>5×5</td>
<td>Orthogonal Disentangled ($\mathcal{L}_a + \mathcal{L}_o$)</td>
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

注：†表示用RTMDet作为基线检测器，否则用Oriented R-CNN。

消融实验得到的结论：
1. **补丁嵌入大小的影响**：从1×1到5×5，随着补丁大小增加，mAP和得分都持续提升，因为更大的补丁能提供更多对象信息；但到7×7的时候性能下降，因为引入了太多周围无关的噪声信息，所以最优补丁大小是5×5。
2. **属性解耦的有效性**：原始混合词元性能最差，只加属性匹配损失的解耦词元，mAP提升2.6%，得分提升1.1；再加正交约束后，得分再提升1.2，证明属性解耦和正交约束都对性能有正向贡献。原文的相关性可视化（Figure 3）也验证了这一点：

   ![Figure 3. Visualization of the correlation between vision tokens and attributes. The horizontal axis represents different dimensions of vision tokens, and the vertical axis represents their attributes, where sls, hc, hs, ds, da denote ship-load-status, hullcolor, hull-size, deck-structure, deck-accessories, respectively.](images/3.jpg)
   *该图像是图表，展示了视觉标记与属性之间的相关性。横轴表示不同维度的视觉标记，纵轴表示其属性，其中sls表示船载状态，hc表示船体颜色，hs表示船体大小，ds表示甲板结构，da表示甲板配件。*

正交约束后，每个词元只和对应属性高度相关，不同属性之间的混淆大大降低，因此属性理解性能更好。
3. **基线检测器的影响**：换成更强的Oriented R-CNN作为基线后，mAP提升0.7%，得分提升1.9，证明EagleVision兼容不同的检测器，基线越强最终性能越好。
4. **LLM缩放规律**：随着LLM规模从1B增加到7B，检测mAP和属性得分都持续提升，符合大语言模型的缩放规律，更大的LLM能带来更好的性能。

# 7. 总结与思考
## 7.1. 结论总结
本文针对遥感领域现有MLLM无法做细粒度对象级属性理解的核心痛点，提出了首个面向遥感的对象级属性多模态大语言模型EagleVision：通过属性解耦模块学习正交解耦的属性视觉词元，解决了原始特征混合多个属性的问题；构建了首个大规模遥感对象属性理解数据集EVAttrs-95K和评测基准EVBench，为该领域后续研究提供了基础；大量实验证明，EagleVision在对象检测和对象属性理解两个任务都达到了最先进性能，并且验证了检测能力和属性理解能力在MLLM中是互相促进的，属性理解的训练能够显著提升对象检测的性能。

## 7.2. 局限性与未来工作
本文没有明确指出自身局限性，但从工作内容可以总结出潜在的改进方向：
1. **覆盖类别有限**：目前EVAttrs-95K只对飞机和船舶做了属性标注，覆盖的对象类别较少，未来可以扩展到更多遥感对象类别比如车辆、建筑、道路、基础设施等，构建更大规模的数据集，让EagleVision成为遥感领域的通用基础模型。
2. **正交约束的微小代价**：从消融实验可以看到，正交约束提升了属性理解得分，但带来了微小的检测mAP下降，未来可以改进约束方式，同时提升两个任务的性能。
3. **评估方式的偏差**：目前EVBench使用GPT辅助评估属性得分，虽然和人类判断一致性较高，但仍然存在评估偏差，未来可以构建全人工标注的评测集，实现更准确的评测。
4. **架构可以进一步优化**：目前EagleVision是两阶段架构，需要基线检测器做前处理，未来可以探索端到端的一体化架构，进一步提升推理效率。

## 7.3. 个人启发与批判
这篇文章的思路非常有启发性，打破了之前遥感MLLM只做图像级对话的研究思路，创新性地将对象检测和细粒度属性理解结合，证明了MLLM不仅能做语言理解，还能通过属性理解的训练反过来提升传统视觉任务（对象检测）的性能，这个双向促进的思路可以推广到几乎所有垂直领域，比如医疗影像（病灶检测+病灶属性描述）、自动驾驶（目标检测+目标属性理解）等，都可以用类似的架构实现两个任务的互相提升。

另外，本文使用大模型两阶段自动标注+人工精修构建大规模垂直领域指令数据集的方式，成本低、效率高，为垂直领域构建MLLM微调数据集提供了一个非常好的可复用范例，降低了数据构建的门槛。

整体来说，这篇工作打开了遥感MLLM研究的新方向，将遥感理解从粗粒度图像级推向了细粒度对象级，有非常大的实际应用价值，在军事侦察、港口管理、国土资源监测等场景都能发挥重要作用。