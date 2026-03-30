# 1. 论文基本信息
## 1.1. 标题
论文标题为《DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception》，核心主题是**面向交互式遥感影像变化分析的多模态大模型**，通过指令引导的差异感知技术，实现双时相遥感影像的多轮、多任务交互解译。
## 1.2. 作者
作者团队包括：
- Pei Deng
- Wenqian Zhou（IEEE学生会员）
- Hanlin Wu（IEEE会员，代码仓库维护者，主要通讯作者）
  三位作者均从事遥感影像解译、多模态大模型相关方向研究。
## 1.3. 发表期刊/会议
当前论文为arXiv预印本，未公开正式发表的期刊/会议信息。
## 1.4. 发表年份
预印本发布时间为2025年7月30日（UTC时间）。
## 1.5. 摘要
本文针对现有多时相遥感影像解译方法仅能输出静态变化掩码或固定描述、无法支持交互式查询的缺陷，提出了<strong>遥感影像变化分析（RSICA）</strong> 这一新范式，结合变化检测和视觉问答的能力，支持双时相遥感影像的多轮、指令引导的变化探索。为支撑该任务，论文通过规则+GPT辅助的混合方式构建了包含10.5万条指令响应对的大规模数据集ChangeChat-105k，覆盖6类交互场景。基于该数据集，论文提出端到端的DeltaVLM模型，包含三个核心创新：微调的双时相视觉编码器、带交叉语义关系测量（CSRM）机制的视觉差异感知模块、指令引导的Q-former对齐模块。实验证明DeltaVLM在单轮变化描述和多轮交互式变化分析任务上均达到最先进水平，性能超过现有通用多模态大模型和遥感专用视觉语言模型。
## 1.6. 原文链接
- 预印本链接：images/2.jpg)
*该图像是示意图，展示了 DeltaVLM 模型的结构，包括三个组件：图像编码器、指令引导的差异感知模块和语言解码器。图像编码器使用双时间视觉编码器处理多时相卫星影像，指令引导模块通过交叉语义关系测量提取差异信息，最终将结果传递给语言解码器以生成文本响应。*

## 4.2. 核心方法详解
### 4.2.1. 双时相视觉编码（Bi-VE）
为充分利用预训练视觉模型的通用特征能力，本文采用EVA-ViT-g/14作为主干网络，采用选择性微调策略：冻结前37层Transformer层的参数，仅微调最后2层，避免灾难性遗忘（微调时丢失预训练学到的通用特征）。
输入为同一区域两个时刻的影像对 $I_{t_1} \in \mathbb{R}^{H\times W\times3}$ 和 $I_{t_2} \in \mathbb{R}^{H\times W\times3}$，其中 $H$ 为影像高度、$W$ 为影像宽度、3为RGB通道数。两个影像分别独立输入ViT编码器，避免过早融合时序特征带来的偏差：每个影像被切分为16×16的图像块（Patch），转换为Patch嵌入后输入Transformer层，提取倒数第二层的特征（跳过分类头），得到双时相特征：
$$
\begin{array} { r l } & { F _ { t _ { 1 } } = \Phi _ { \mathrm { V i T } } ( I _ { t _ { 1 } } ; \Theta _ { \mathrm { f i n e - t u n e d } } ) \in \mathbb { R } ^ { \frac { H } { 1 6 } \times \frac { W } { 1 6 } \times D } } \\ & { F _ { t _ { 2 } } = \Phi _ { \mathrm { V i T } } ( I _ { t _ { 2 } } ; \Theta _ { \mathrm { f i n e - t u n e d } } ) \in \mathbb { R } ^ { \frac { H } { 1 6 } \times \frac { W } { 1 6 } \times D } , } \end{array}
$$
符号解释：
- $\Phi_{ViT}$：EVA-ViT-g/14编码器函数
- $\Theta_{fine-tuned}$：编码器最后2层的微调参数
- $D$：特征的隐藏层维度
- 空间分辨率降低16倍是因为每个Patch大小为16×16，输出特征的尺寸为原影像的1/16。
### 4.2.2. 指令引导的差异感知模块（IDPM）
该模块负责过滤非语义变化噪声，同时提取和用户指令相关的变化特征，分为交叉语义关系测量（CSRM）和指令引导Q-former两个子模块。
#### 4.2.2.1. 交叉语义关系测量（CSRM）
首先计算原始的像素级差异特征：
$$F_{diff} = F_{t_2} - F_{t_1} \in \mathbb{R}^{N\times D}$$
其中$N$为Patch的总数量，该特征包含真实地物变化和大量噪声（光照、大气、季节变化、传感器偏移等），CSRM通过三步过滤噪声：
1.  **上下文建模**：将差异特征和对应时刻的原始特征拼接，通过线性投影和tanh激活得到上下文向量，建模变化和原始地物的语义关联：
    $$
    \begin{array} { r l } & { C _ { t _ { 1 } } = \operatorname { t a n h } ( W _ { c } [ F _ { \mathrm { d i f f } } ; F _ { t _ { 1 } } ] + b _ { c } ) } \\ & { C _ { t _ { 2 } } = \operatorname { t a n h } ( W _ { c } ^ { \prime } [ F _ { \mathrm { d i f f } } ; F _ { t _ { 2 } } ] + b _ { c } ^ { \prime } ) , } \end{array}
    $$
    符号解释：
    - $[\cdot;\cdot]$：通道维度的特征拼接操作
    - $W_c, W_c' \in \mathbb{R}^{D\times2D}$：可学习的线性投影权重矩阵
    - $b_c, b_c' \in \mathbb{R}^D$：可学习的偏置向量
    - $\tanh$：双曲正切激活函数，将输出值压缩到`[-1,1]`区间
2.  **门控生成**：通过sigmoid激活生成门控向量，衡量每个位置的变化的语义相关性，值越高代表该位置是真实地物变化的概率越高：
    $$
    \begin{array} { r l } & { G _ { t _ { 1 } } = \sigma ( W _ { \mathrm { g } } [ F _ { \mathrm { d i f f } } ; F _ { t _ { 1 } } ] + b _ { \mathrm { g } } ) } \\ & { G _ { t _ { 2 } } = \sigma ( W _ { \mathrm { g } } ^ { \prime } [ F _ { \mathrm { d i f f } } ; F _ { t _ { 2 } } ] + b _ { \mathrm { g } } ^ { \prime } ) , } \end{array}
    $$
    符号解释：
    - $\sigma$：sigmoid激活函数，输出值范围为$(0,1)$，作为相关性得分
    - $W_g, W_g' \in \mathbb{R}^{D\times2D}$：可学习的线性投影权重矩阵
    - $b_g, b_g' \in \mathbb{R}^D$：可学习的偏置向量
3.  **特征过滤**：将门控向量和上下文向量逐元素相乘，压制噪声对应的特征，保留真实变化的特征：
    $$
    \begin{array} { r l } & { F _ { t _ { 1 } } ^ { \prime } = G _ { t _ { 1 } } \odot C _ { t _ { 1 } } } \\ & { F _ { t _ { 2 } } ^ { \prime } = G _ { t _ { 2 } } \odot C _ { t _ { 2 } } . } \end{array}
    $$
    符号解释：
    - $\odot$：逐元素相乘操作，门控值低的位置特征会被压缩到接近0，实现噪声过滤。
#### 4.2.2.2. 指令引导Q-former
参考InstructBLIP的Q-former设计，实现过滤后的视觉特征和用户指令的对齐，提取和查询相关的变化特征：
1.  首先初始化32个可学习的查询嵌入 $Q \in \mathbb{R}^{L\times d}$，其中$L=32$为查询数量，$d$为和LLM输入匹配的特征维度。查询首先经过自注意力层交互，得到更新后的查询：
    $$Q_{SA} = \mathrm{SelfAttention}(Q)$$
    自注意力是Transformer的核心机制，让查询向量之间互相计算相关性，捕捉内部依赖。
2.  然后更新后的查询通过交叉注意力层，同时关注过滤后的双时相特征和用户指令$P$，提取和查询相关的变化特征：
    $$Q_{CA} = \mathrm{CrossAttention}(Q_{SA}, [F_{t_1}'; F_{t_2}'], P)$$
    交叉注意力让查询向量和视觉特征、文本指令计算相关性，动态对齐用户需求。
3.  最后经过前馈网络（FFN）得到紧凑的指令对齐差异特征：
    $$\hat{F}_{diff} = \mathrm{FFN}(Q_{CA}) \in \mathbb{R}^{32\times d}$$
    输出的32个特征向量仅包含和用户查询相关的变化信息，减少冗余，适配LLM的输入要求。
### 4.2.3. LLM语言解码器
本文采用Vicuna-7B作为语言解码器，该模型是基于LLaMA微调的开源大语言模型，具备强大的语言理解和生成能力，训练时冻结整个LLM的所有参数，仅训练视觉和对齐模块，大幅降低训练成本。
首先将用户指令$P$通过LLM的分词器和嵌入层转换为文本嵌入：
$E = \Phi_{embedding}(P)$
其中$\Phi_{embedding}$为LLM的分词和嵌入函数，将自然语言文本转换为模型可处理的向量序列。
然后将指令对齐的差异特征$\hat{F}_{diff}$和文本嵌入$E$拼接后输入LLM，生成自然语言响应$T$：
$$T = \Phi_{LLM}(\hat{F}_{diff}, E) \in \mathcal{C}^N$$
符号解释：
- $\Phi_{LLM}$：Vicuna-7B解码器函数
- $\mathcal{C}$：LLM的词表
- $N$：生成响应的词元数量
### 4.2.4. 训练目标
模型采用交叉熵损失函数进行训练，训练集为指令-响应对集合$D_{train} = \{(I_1,P_1,T_1),\dots,(I_M,P_M,T_M)\}$，其中$I$为双时相影像对，$P$为用户指令，$T$为真实响应。损失公式为：
$$\mathcal{L}_{train} = -\frac{1}{K}\sum_{i=1}^K w_i \log(\hat{w}_i)$$
符号解释：
- $K$：真实响应的总词元数量
- $w_i$：第$i$个位置真实词元的one-hot编码
- $\hat{w}_i$：模型预测的第$i$个位置对应词元的概率
  交叉熵损失衡量预测词元分布和真实分布的差异，最小化该损失可让模型生成更准确的响应。

# 5. 实验设置
## 5.1. 数据集
实验采用本文构建的ChangeChat-105k数据集，基于公开的LEVIR-CC和LEVIR-MCI数据集构建，通过规则+ChatGPT辅助的混合方式生成105107条指令响应对，对应256×256分辨率的双时相影像，空间分辨率为0.5m/像素，覆盖城市区域的建筑、道路两类地物变化。数据集分为训练集（87935条）和测试集（17172条），覆盖6类交互任务，详细统计如下（原文Table I）：

<table>
<thead>
<tr>
<th>指令类型</th>
<th>源数据</th>
<th>生成方式</th>
<th>响应格式</th>
<th>训练集数量</th>
<th>测试集数量</th>
</tr>
</thead>
<tbody>
<tr>
<td>变化描述</td>
<td>LEVIR-CC</td>
<td>规则生成</td>
<td>描述性文本</td>
<td>34075</td>
<td>1929</td>
</tr>
<tr>
<td>二分类变化检测</td>
<td>LEVIR-MCI</td>
<td>规则生成</td>
<td>Yes/No</td>
<td>6815</td>
<td>1929</td>
</tr>
<tr>
<td>类别特定变化量化</td>
<td>LEVIR-MCI</td>
<td>规则生成</td>
<td>对象数量</td>
<td>6815</td>
<td>1929</td>
</tr>
<tr>
<td>变化定位</td>
<td>LEVIR-MCI</td>
<td>规则生成</td>
<td>3×3网格位置</td>
<td>6815</td>
<td>1929</td>
</tr>
<tr>
<td>开放式QA</td>
<td>LEVIR-CC/MCI衍生</td>
<td>GPT辅助生成</td>
<td>问答对</td>
<td>26600</td>
<td>7527</td>
</tr>
<tr>
<td>多轮对话</td>
<td>LEVIR-MCI衍生</td>
<td>规则生成</td>
<td>多轮对话</td>
<td>6815</td>
<td>1929</td>
</tr>
<tr>
<td>总计</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>87935</td>
<td>17172</td>
</tr>
</tbody>
</table>

该数据集是目前首个覆盖多类交互式变化分析场景的大规模双时相遥感数据集，可全面验证模型的多任务、多轮交互能力。
## 5.2. 评估指标
本文针对不同任务采用对应的专用评估指标，各指标的详细说明如下：
### 5.2.1. 变化描述/开放式QA指标
1.  **BLEU-N**
    - 概念定义：衡量生成文本和真实文本的n元语法（n-gram）重合度，n取1~4，数值越高代表生成文本和真实文本的匹配度越高，BLEU-1衡量单词匹配度，BLEU-4衡量短语匹配度。
    - 公式：
      $$BLEU-N = BP \times \exp\left(\sum_{n=1}^N w_n \log p_n\right)$$
    - 符号解释：`BP`为短句惩罚因子，避免生成过短的文本；$p_n$为n-gram的准确率；$w_n$为权重，通常取$1/N$。
2.  **METEOR**
    - 概念定义：考虑同义词、词干、词序的匹配度，比BLEU更贴合人类的语义判断，数值越高越好。
    - 公式：
      $$METEOR = F_{mean} \times (1 - Penalty)$$
    - 符号解释：$F_{mean}$为准确率和召回率的调和平均；`Penalty`为词序惩罚项。
3.  **ROUGE-L**
    - 概念定义：衡量生成文本和真实文本的最长公共子序列重合度，关注文本的整体流畅性和内容完整性，数值越高越好。
    - 公式：
      $$ROUGE-L = \frac{2 \times R_{lcs} \times P_{lcs}}{R_{lcs} + P_{lcs}}$$
    - 符号解释：$R_{lcs}$为最长公共子序列的召回率；$P_{lcs}$为最长公共子序列的准确率。
4.  **CIDEr-D**
    - 概念定义：专门为图像描述任务设计，衡量生成文本和多个人工标注的共识程度，数值越高代表生成文本越符合人类的通用描述习惯。
    - 公式：
      $$CIDEr = \frac{1}{M}\sum_{i=1}^M \frac{g(c_i) \cdot g(r_i)}{||g(c_i)|| \cdot ||g(r_i)||}$$
    - 符号解释：$g(c_i)$为生成文本的TF-IDF向量；$g(r_i)$为真实文本的TF-IDF向量；$M$为人工标注的数量。
### 5.2.2. 二分类变化检测指标
1.  <strong>准确率（Accuracy）</strong>
    - 概念定义：预测正确的样本占总样本的比例，越高越好。
    - 公式：
      $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
    - 符号解释：`TP`为真阳性（真实变化预测为变化）；`TN`为真阴性（真实无变化预测为无变化）；`FP`为假阳性（真实无变化预测为变化）；`FN`为假阴性（真实变化预测为无变化）。
2.  <strong>精确率（Precision）</strong>
    - 概念定义：预测为变化的样本中真实为变化的比例，衡量模型减少误报的能力，越高越好。
    - 公式：
      $Precision = \frac{TP}{TP + FP}$
3.  <strong>召回率（Recall）</strong>
    - 概念定义：真实变化的样本中被预测为变化的比例，衡量模型减少漏检的能力，越高越好。
    - 公式：
      $Recall = \frac{TP}{TP + FN}$
4.  **F1-score**
    - 概念定义：精确率和召回率的调和平均，综合衡量分类性能，越高越好。
    - 公式：
      $$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$
### 5.2.3. 变化量化指标
1.  <strong>MAE（平均绝对误差）</strong>
    - 概念定义：预测数量和真实数量的绝对差的平均值，对所有误差平等惩罚，越低越好。
    - 公式：
      $$MAE = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
    - 符号解释：$y_i$为真实数量；$\hat{y}_i$为预测数量；$N$为样本总数。
2.  <strong>RMSE（均方根误差）</strong>
    - 概念定义：预测误差平方的均值的平方根，对大误差惩罚更大，越低越好。
    - 公式：
      $$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
### 5.2.4. 变化定位指标
除精确率、召回率、F1-score外，还包含：
1.  <strong>Jaccard相似度（交并比）</strong>：预测的变化区域和真实变化区域的交集除以并集，越高代表定位越准确。
2.  **子集准确率**：所有3×3网格的预测都完全正确的样本比例，越高越好。
## 5.3. 对比基线
本文选择两类代表性基线进行对比：
1.  **遥感专用变化描述模型**：包括RSICCFormer、PromptCC、PSNet、SFT，均为针对遥感变化描述任务优化的最先进模型，用于验证本文方法和专用单任务模型的性能差异。
2.  **通用多模态大模型**：包括GPT-4o、Qwen-VL-Plus、GLM-4V-Plus、Gemini-1.5-Pro，均为目前通用域性能最好的VLM，用于验证本文方法相对通用模型的域适配增益。

# 6. 实验结果与分析
## 6.1. 核心结果分析
### 6.1.1. 变化描述任务
以下是变化描述任务的实验结果（原文Table II）：

<table>
<thead>
<tr>
<th>类别</th>
<th>方法</th>
<th>BLEU-1</th>
<th>BLEU-2</th>
<th>BLEU-3</th>
<th>BLEU-4</th>
<th>METEOR</th>
<th>ROUGE-L</th>
<th>CIDEr-D</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">通用VLM</td>
<td>GPT-4o</td>
<td>46.03</td>
<td>33.09</td>
<td>24.66</td>
<td>18.05</td>
<td>22.50</td>
<td>56.49</td>
<td>90.92</td>
</tr>
<tr>
<td>Qwen-VL-Plus</td>
<td>41.31</td>
<td>33.19</td>
<td>27.96</td>
<td>22.95</td>
<td>18.04</td>
<td>51.24</td>
<td>92.99</td>
</tr>
<tr>
<td>GLM-4V-Plus</td>
<td>35.59</td>
<td>24.26</td>
<td>18.54</td>
<td>13.85</td>
<td>20.13</td>
<td>54.39</td>
<td>93.16</td>
</tr>
<tr>
<td>Gemini-1.5-Pro</td>
<td>45.68</td>
<td>33.59</td>
<td>25.53</td>
<td>19.01</td>
<td>22.64</td>
<td>56.25</td>
<td>91.37</td>
</tr>
<tr>
<td rowspan="4">遥感专用变化描述模型</td>
<td>RSICCFormer</td>
<td>84.72</td>
<td>76.27</td>
<td>68.87</td>
<td>62.77</td>
<td>39.61</td>
<td>74.12</td>
<td>134.12</td>
</tr>
<tr>
<td>PromptCC</td>
<td>83.66</td>
<td>75.73</td>
<td>69.10</td>
<td>63.54</td>
<td>38.82</td>
<td>73.72</td>
<td>136.44</td>
</tr>
<tr>
<td>PSNet</td>
<td>83.86</td>
<td>75.13</td>
<td>67.89</td>
<td>62.11</td>
<td>38.80</td>
<td>73.60</td>
<td>132.62</td>
</tr>
<tr>
<td>SFT</td>
<td>84.56</td>
<td>75.87</td>
<td>68.64</td>
<td>62.87</td>
<td>39.93</td>
<td>74.69</td>
<td>137.05</td>
</tr>
<tr>
<td>本文方法</td>
<td>Ours</td>
<td>85.78</td>
<td>77.15</td>
<td>69.24</td>
<td>62.51</td>
<td>39.47</td>
<td>75.01</td>
<td>136.72</td>
</tr>
</tbody>
</table>

结果分析：通用VLM的性能远低于遥感专用模型，BLEU-4仅为13.85~22.95，说明通用模型缺乏遥感域适配，难以捕捉遥感影像的细粒度变化。本文模型在BLEU-1/2/3、ROUGE-L指标上均达到最高，BLEU-4接近最优的SFT模型，说明生成的文本的单词、短语匹配度更高，更贴合用户指令的需求。
### 6.1.2. 二分类变化检测任务
以下是二分类变化检测任务的实验结果（原文Table III）：

<table>
<thead>
<tr>
<th>方法</th>
<th>准确率(%)</th>
<th>精确率(%)</th>
<th>召回率(%)</th>
<th>F1(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPT-4o</td>
<td>84.81</td>
<td>83.58</td>
<td>86.62</td>
<td>85.07</td>
</tr>
<tr>
<td>Qwen-VL-Plus</td>
<td>58.22</td>
<td>73.65</td>
<td>25.52</td>
<td>37.90</td>
</tr>
<tr>
<td>GLM-4V-Plus</td>
<td>79.83</td>
<td>88.38</td>
<td>68.67</td>
<td>77.29</td>
</tr>
<tr>
<td>Gemini-1.5-Pro</td>
<td>83.83</td>
<td>84.03</td>
<td>83.51</td>
<td>83.77</td>
</tr>
<tr>
<td>本文方法</td>
<td>93.99</td>
<td>96.29</td>
<td>91.49</td>
<td>93.83</td>
</tr>
</tbody>
</table>

结果分析：本文方法的F1-score比最优基线GPT-4o高8.76%，精确率和召回率均超过91%，说明模型的误报和漏检率都很低，平衡的精确率-召回率表现证明了CSRM模块对噪声的过滤能力。
### 6.1.3. 变化量化任务
以下是变化量化任务的实验结果（原文Table IV）：

<table>
<thead>
<tr>
<th rowspan="2">方法</th>
<th colspan="2">道路</th>
<th colspan="2">建筑</th>
</tr>
<tr>
<th>MAE</th>
<th>RMSE</th>
<th>MAE</th>
<th>RMSE</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPT-4o</td>
<td>0.49</td>
<td>1.00</td>
<td>1.86</td>
<td>4.57</td>
</tr>
<tr>
<td>Qwen-VL-Plus</td>
<td>0.90</td>
<td>1.50</td>
<td>4.41</td>
<td>9.03</td>
</tr>
<tr>
<td>GLM-4V-Plus</td>
<td>0.82</td>
<td>1.62</td>
<td>2.05</td>
<td>4.61</td>
</tr>
<tr>
<td>Gemini-1.5-Pro</td>
<td>0.58</td>
<td>1.25</td>
<td>2.56</td>
<td>8.71</td>
</tr>
<tr>
<td>本文方法</td>
<td>0.24</td>
<td>0.70</td>
<td>1.32</td>
<td>2.89</td>
</tr>
</tbody>
</table>

结果分析：本文方法的MAE和RMSE均为最低，道路MAE比最优基线GPT-4o降低51%，建筑MAE降低29%，所有指标平均提升35%，说明模型能精准聚焦于目标类别的变化，过滤无关噪声。
### 6.1.4. 变化定位任务
以下是变化定位任务的实验结果（原文Table V）：

<table>
<thead>
<tr>
<th>类别</th>
<th>方法</th>
<th>精确率(%)</th>
<th>召回率(%)</th>
<th>F1(%)</th>
<th>Jaccard相似度(%)</th>
<th>子集准确率(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="5">道路</td>
<td>GPT-4o</td>
<td>30.44</td>
<td>27.01</td>
<td>28.62</td>
<td>7.80</td>
<td>33.85</td>
</tr>
<tr>
<td>Qwen-VL-Plus</td>
<td>15.42</td>
<td>1.40</td>
<td>2.56</td>
<td>0.25</td>
<td>67.19</td>
</tr>
<tr>
<td>GLM-4V-Plus</td>
<td>21.99</td>
<td>33.32</td>
<td>26.49</td>
<td>7.93</td>
<td>6.79</td>
</tr>
<tr>
<td>Gemini-1.5-Pro</td>
<td>43.01</td>
<td>40.55</td>
<td>41.74</td>
<td>9.62</td>
<td>48.63</td>
</tr>
<tr>
<td>本文方法</td>
<td>69.63</td>
<td>66.32</td>
<td>67.94</td>
<td>14.00</td>
<td>70.92</td>
</tr>
<tr>
<td rowspan="5">建筑</td>
<td>GPT-4o</td>
<td>55.63</td>
<td>33.70</td>
<td>41.98</td>
<td>14.09</td>
<td>41.47</td>
</tr>
<tr>
<td>Qwen-VL-Plus</td>
<td>22.23</td>
<td>20.78</td>
<td>21.48</td>
<td>6.52</td>
<td>7.26</td>
</tr>
<tr>
<td>GLM-4V-Plus</td>
<td>38.98</td>
<td>57.83</td>
<td>46.57</td>
<td>17.93</td>
<td>17.11</td>
</tr>
<tr>
<td>Gemini-1.5-Pro</td>
<td>65.71</td>
<td>51.75</td>
<td>57.90</td>
<td>18.62</td>
<td>45.62</td>
</tr>
<tr>
<td>本文方法</td>
<td>77.79</td>
<td>80.22</td>
<td>78.99</td>
<td>23.15</td>
<td>65.53</td>
</tr>
</tbody>
</table>

结果分析：本文方法的F1-score相对最优基线Gemini-1.5-Pro，道路提升26.2%，建筑提升21.09%，其他指标也均为最优，证明模型的空间定位能力远超过通用VLM。
### 6.1.5. 开放式QA任务
以下是开放式QA任务的实验结果（原文Table VI）：

<table>
<thead>
<tr>
<th>方法</th>
<th>BLEU-1</th>
<th>BLEU-2</th>
<th>BLEU-3</th>
<th>BLEU-4</th>
<th>METEOR</th>
<th>ROUGE-L</th>
<th>CIDEr-D</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPT-4o</td>
<td>33.08</td>
<td>21.08</td>
<td>14.06</td>
<td>9.68</td>
<td>22.24</td>
<td>35.53</td>
<td>72.58</td>
</tr>
<tr>
<td>Qwen-VL-Plus</td>
<td>24.75</td>
<td>12.55</td>
<td>6.70</td>
<td>3.88</td>
<td>16.69</td>
<td>27.74</td>
<td>27.22</td>
</tr>
<tr>
<td>GLM-4V-Plus</td>
<td>34.27</td>
<td>22.38</td>
<td>15.66</td>
<td>11.43</td>
<td>22.48</td>
<td>37.11</td>
<td>100.66</td>
</tr>
<tr>
<td>Gemini-1.5-Pro</td>
<td>32.90</td>
<td>20.44</td>
<td>13.38</td>
<td>9.06</td>
<td>21.85</td>
<td>35.19</td>
<td>68.64</td>
</tr>
<tr>
<td>本文方法</td>
<td>36.67</td>
<td>27.09</td>
<td>20.62</td>
<td>16.21</td>
<td>17.85</td>
<td>32.60</td>
<td>127.38</td>
</tr>
</tbody>
</table>

结果分析：本文方法在BLEU-1~4、CIDEr-D指标上均为最高，说明模型能更好地理解多样化的开放式查询，生成的答案和人类标注的契合度更高。
各任务的综合性能对比如下图（原文Figure 1）所示，DeltaVLM在所有任务上均明显超过通用VLM：

![Fig. 1. The performance of DeltaVLM against state-of-the-art VLMs on five RS change analysis tasks. Each axis corresponds to a task-specific metric: captioning (BLEU-1), classification (precision), quantification (inverted Road's-MAE), localization (F1-score), and open-ended QA (BLEU-1).](images/1.jpg)
*该图像是一个雷达图，展示了 DeltaVLM 与其他前沿 VLM 在五个遥感变化分析任务上的性能对比。图中每个轴对应一个特定任务的指标，包括标题生成（BLEU-1）、分类（精确度）、量化（Road's MAE）、定位（F1-score）和开放式问答（BLEU-1）。*

### 6.1.6. 多轮交互定性结果
DeltaVLM的多轮对话能力如下图（原文Figure 5）所示，模型能保持上下文信息，依次完成变化检测、量化、描述等多轮查询：

![Fig. 5. Demonstration of multi-round dialogue capability of DeltaVLM.](images/3.jpg)
*该图像是示意图，展示了DeltaVLM在多轮对话中的能力。图中显示了三轮关于卫星图像变化的对话，用户询问了道路和建筑物的变化，系统根据给定的图像提供了详细的描述和判断。*

## 6.2. 消融实验
本文通过消融实验验证了Bi-VE微调、CSRM模块的有效性，结果如下：
### 6.2.1. 变化描述任务消融
以下是变化描述任务的消融结果（原文Table VII）：

<table>
<thead>
<tr>
<th>方法</th>
<th>BLEU-1</th>
<th>BLEU-2</th>
<th>BLEU-3</th>
<th>BLEU-4</th>
<th>METEOR</th>
<th>ROUGE-L</th>
<th>CIDEr-D</th>
</tr>
</thead>
<tbody>
<tr>
<td>去掉CSRM模块</td>
<td>64.42</td>
<td>56.52</td>
<td>53.08</td>
<td>51.40</td>
<td>29.31</td>
<td>60.54</td>
<td>101.92</td>
</tr>
<tr>
<td>去掉Bi-VE微调</td>
<td>84.24</td>
<td>75.62</td>
<td>67.91</td>
<td>61.40</td>
<td>39.29</td>
<td>74.73</td>
<td>134.76</td>
</tr>
<tr>
<td>完整DeltaVLM</td>
<td>85.78</td>
<td>77.15</td>
<td>69.24</td>
<td>62.51</td>
<td>39.47</td>
<td>75.01</td>
<td>136.72</td>
</tr>
</tbody>
</table>

结果分析：去掉CSRM模块后所有指标暴跌，说明CSRM是模型的核心组件，没有该模块无法区分真实变化和噪声；去掉Bi-VE微调后指标略有下降，说明微调视觉编码器的最后两层能有效提升模型对遥感域特征的适配能力。
### 6.2.2. 二分类变化检测任务消融
以下是二分类变化检测任务的消融结果（原文Table VIII）：

<table>
<thead>
<tr>
<th>方法</th>
<th>准确率(%)</th>
<th>精确率(%)</th>
<th>召回率(%)</th>
<th>F1(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>去掉CSRM模块</td>
<td>50.13</td>
<td>75.00</td>
<td>0.31</td>
<td>0.62</td>
</tr>
<tr>
<td>去掉Bi-VE微调</td>
<td>90.57</td>
<td>99.49</td>
<td>81.54</td>
<td>89.62</td>
</tr>
<tr>
<td>完整DeltaVLM</td>
<td>93.99</td>
<td>96.29</td>
<td>91.49</td>
<td>93.83</td>
</tr>
</tbody>
</table>

结果分析：去掉CSRM模块后模型几乎完全偏向预测"无变化"，召回率仅0.31%，进一步验证了CSRM对噪声过滤的关键作用；去掉Bi-VE微调后F1下降4.21%，验证了微调的有效性。

# 7. 总结与思考
## 7.1. 结论总结
本文首次定义了交互式遥感影像变化分析（RSICA）任务，填补了遥感解译从静态输出到动态交互的空白；构建了目前规模最大的双时相遥感指令数据集ChangeChat-105k，覆盖6类交互场景；提出了DeltaVLM专用架构，通过CSRM模块过滤遥感特有噪声，通过指令引导的Q-former实现查询和变化特征的精准对齐，在所有RSICA子任务上均达到最先进水平，相对通用VLM有显著的性能提升，为交互式遥感解译提供了完整的数据集和模型方案，可直接应用于灾害应急、城市规划、国土监测等实际场景。
## 7.2. 局限性与未来工作
作者指出的局限性和未来研究方向：
1.  目前模型仅支持文本输出，未来将研究统一多模态输出架构，可同时输出文本响应和变化掩码、定位坐标等可视化结果。
2.  进一步增强模型的复杂推理能力，支持更复杂的逻辑查询（如"变化的建筑是否占用了耕地？"）。
3.  优化模型推理效率，适配边缘设备的实时交互需求。
## 7.3. 个人启发与批判
### 7.3.1. 启发
本文是遥感大模型从单时相到多时相、从单任务到多任务交互的典型范式创新，其"领域特性模块+通用大模型"的架构设计思路可迁移到其他垂直领域的大模型适配中：针对垂直领域的特有问题设计专用的预处理/特征过滤模块，冻结通用大模型参数仅训练领域专用模块，可兼顾训练效率和领域适配性能。此外，本文构建数据集的"规则+大模型辅助生成"的方式，可大幅降低垂直领域大规模指令数据集的构建成本。
### 7.3.2. 潜在改进点
1.  数据集的覆盖范围有限：目前仅覆盖城市区域的建筑、道路两类地物，未来可扩展植被、水体、农田等更多地物类型，同时增加不同地形（山区、农村、森林）、不同传感器的数据，提升模型的泛化能力。
2.  定位精度较粗：目前的变化定位仅支持3×3网格输出，未来可支持像素级变化掩码或更精细的坐标输出，满足更高精度的定位需求。
3.  模型推理成本较高：目前采用7B参数的大模型，推理速度较慢，未来可通过量化、蒸馏、轻量化架构设计等方式，适配移动端和边缘设备的部署需求。
4.  可解释性不足：大模型的决策过程黑盒问题仍存在，未来可加入中间结果可视化（如高亮变化区域），提升结果的可解释性和可信度。