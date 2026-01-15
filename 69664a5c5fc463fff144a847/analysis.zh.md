# 1. 论文基本信息

## 1.1. 标题
Matrix-Game: 交互式世界基础模型 (Matrix-Game: Interactive World Foundation Model)

论文标题明确指出了研究的核心：一个名为 `Matrix-Game` 的基础模型，其主要功能是生成**可交互的**、**可控制的**游戏世界。

## 1.2. 作者
Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, Yahui Zhou.

所有作者均来自 **Skywork AI** (昆仑万维)。这是一个专注于大模型研发的公司，这表明该研究具有很强的工程实践和产品化导向背景。

## 1.3. 发表期刊/会议
这篇论文目前以预印本 (Preprint) 的形式发布在 **arXiv** 上。

**arXiv** 是一个收集物理学、数学、计算机科学、生物学等领域预印本的网站。在人工智能，特别是深度学习领域，由于技术迭代速度极快，研究者通常会先将论文发布在 arXiv 上，以抢占首发权并与学术界快速交流，之后再投稿到正式的顶级会议（如 NeurIPS, ICML, CVPR 等）。因此，虽然是预印本，但其内容代表了该领域的最新进展。

## 1.4. 发表年份
2024年6月23日 (提交至 arXiv)。

## 1.5. 摘要
我们引入了 **Matrix-Game**，一个用于**可控游戏世界生成**的交互式世界基础模型。Matrix-Game 的训练采用**两阶段流水线**：首先进行大规模无标签预训练以理解环境，然后进行动作标签训练以实现交互式视频生成。为了支持这一点，我们构建了 **Matrix-Game-MC** 数据集，这是一个全面的《我的世界》(Minecraft) 数据集，包含超过 2700 小时的无标签游戏视频片段和超过 1000 小时的带有精细键盘和鼠标动作标注的高质量有标签片段。我们的模型采用一种<strong>可控的图像到世界 (image-to-world) 生成范式</strong>，以参考图像、运动上下文和用户动作为条件。Matrix-Game 拥有超过 170 亿参数，能够精确控制角色动作和相机移动，同时保持高视觉质量和时间连贯性。为了评估性能，我们开发了 **GameWorld Score**，这是一个统一的基准测试，用于衡量《我的世界》世界生成的视觉质量、时间质量、动作可控性和物理规则理解。大量实验表明，Matrix-Game 在所有指标上都持续优于先前的开源《我的世界》世界模型（包括 Oasis 和 MineWorld），在可控性和物理一致性方面取得了特别显著的进步。双盲人类评估进一步证实了 Matrix-Game 的优越性，突出了其在各种游戏场景中生成感知上真实且可精确控制的视频的能力。为了促进交互式图像到世界生成的未来研究，我们将在 `https://github.com/SkyworkAI/Matrix-Game` 开源 Matrix-Game 模型权重和 GameWorld Score 基准测试。

## 1.6. 原文链接
- **arXiv 链接:** https://arxiv.org/abs/2506.18701
- **PDF 链接:** https://arxiv.org/pdf/2506.18701v1
- **项目主页:** https://matrix-game-homepage.github.io
- **发布状态:** 预印本 (Preprint)。

  ---

# 2. 整体概括

## 2.1. 研究背景与动机
近年来，将视频生成模型用作<strong>世界模型 (World Models)</strong> 成为人工智能领域的一个热门方向。世界模型能够模拟环境的动态变化，让智能体 (agent) 理解“如果我采取某个动作，世界会发生什么变化”，这对于自动驾驶、具身智能和游戏等领域至关重要。

然而，构建一个理想的视频世界模型面临三大核心挑战：
1.  **数据稀缺与质量问题：** 训练一个能够理解交互的世界模型，需要大量带有精细动作标注（如键盘按键、鼠标移动）的视频数据。这类数据的采集成本极高，导致现有数据集规模有限。
2.  **物理动态与精细控制的建模难题：** 模型不仅要生成视觉上连贯的视频，还必须精确响应用户的控制信号，并遵循环境的物理规则（如物体不会随意消失，场景保持一致）。这对于现有的视频生成模型来说非常困难。
3.  **缺乏标准化的评估体系：** 如何客观、全面地评估一个生成式世界模型的性能？特别是如何衡量其“可控性”和“物理一致性”？现有视频评估基准（如 `VBench`）主要关注视觉质量和文本对齐，无法满足交互式世界模型的需求。

    为了解决上述挑战，本文以流行的沙盒游戏<strong>《我的世界》(Minecraft)</strong> 为试验场，提出了一个完整的解决方案，其核心思路是：**通过构建大规模、高质量的数据集，训练一个超大规模的参数模型，并设计一个全新的评估基准，来系统性地推进可交互游戏世界的生成。**

## 2.2. 核心贡献/主要发现
本文的贡献是系统性的，可以概括为三个方面，分别对应了上述的三个挑战：

1.  <strong>大规模高质量数据集 (Matrix-Game-MC):</strong> 针对数据稀缺问题，作者们构建了一个包含两个部分的大型《我的世界》视频数据集：
    *   **2700+ 小时的无标签视频：** 通过精细的过滤流水线从公开数据中筛选而来，用于让模型学习游戏世界的基本视觉和动态规律。
    *   **1000+ 小时的高质量有标签视频：** 通过自动化智能体和程序化模拟生成，包含精细的键盘和鼠标动作标注，用于训练模型的可控性。

2.  <strong>交互式世界基础模型 (Matrix-Game):</strong> 针对建模难题，作者提出了一个拥有 **170 亿参数**的扩散模型 `Matrix-Game`。其特点包括：
    *   <strong>图像到世界 (Image-to-World) 范式：</strong> 仅使用一张参考图像作为起点来生成世界，专注于从视觉和物理线索中学习，避免了文本可能带来的语义偏差。
    *   **两阶段训练策略：** 先用无标签数据进行大规模预训练，再用有标签数据进行微调，有效提升了模型的学习效率和最终性能。
    *   **精细的动作控制：** 模型架构专门设计用于接收并响应逐帧的键盘和鼠标动作输入，实现了高精度的交互控制。

3.  <strong>统一评估基准 (GameWorld Score):</strong> 针对评估体系缺失问题，作者设计了一个专门用于评估《我的世界》这类交互式世界模型的基准 `GameWorld Score`。它从四个维度、八个子项全面衡量模型性能：
    *   <strong>视觉质量 (Visual Quality)</strong>
    *   <strong>时间质量 (Temporal Quality)</strong>
    *   <strong>动作可控性 (Action Controllability)</strong>
    *   <strong>物理规则理解 (Physical Rule Understanding)</strong>

        通过这套组合拳，论文的主要发现是：<strong>`Matrix-Game` 在 `GameWorld Score` 基准上的表现全面超越了现有的开源模型（如 `Oasis` 和 `MineWorld`），尤其在动作可控性和物理一致性上优势巨大，并且在双盲人类评估中获得了压倒性的好评。</strong> 这证明了通过<strong>“大数据 + 大模型 + 好评估”</strong>的思路，可以显著提升交互式世界模型的生成质量和实用性。

---

# 3. 预备知识与相关工作

## 3.1. 基础概念

### 3.1.1. 世界模型 (World Models)
世界模型是一种能够学习环境动态并对其进行模拟的内部表征。简单来说，它是一个在“脑中”模拟世界的模型。智能体可以利用这个内部模型来预测其动作可能产生的后果，从而进行规划和决策，而无需在真实世界中反复试错。例如，一个自动驾驶汽车的世界模型可以预测“如果我向左打方向盘，前方的行人和车辆会如何移动”。本文的 `Matrix-Game` 就是一个针对游戏环境的视频生成式世界模型。

### 3.1.2. 扩散模型 (Diffusion Models)
扩散模型是一类强大的生成模型，近年来在图像、音频和视频生成领域取得了巨大成功。其核心思想分为两个过程：
1.  <strong>前向过程（加噪）：</strong> 从一个真实的图像或视频开始，逐步地、多次地向其中添加少量高斯噪声，直到它完全变成纯粹的随机噪声。
2.  <strong>反向过程（去噪）：</strong> 训练一个神经网络（通常是 `U-Net` 或 `Transformer` 架构），让它学习如何从一个充满噪声的输入中，一步步地“恢复”出原始的、清晰的图像或视频。

    生成新内容时，模型从一个随机噪声开始，通过多次迭代的反向去噪过程，最终“创造”出一个全新的、高质量的样本。

### 3.1.3. 潜在扩散模型 (Latent Diffusion Models, LDM)
直接在像素空间（即原始图像或视频）上运行扩散模型计算成本非常高。潜在扩散模型通过引入一个<strong>变分自编码器 (Variational Autoencoder, VAE)</strong> 来解决这个问题。
*   <strong>编码器 (Encoder):</strong> 首先使用 VAE 的编码器将高分辨率的图像/视频压缩到一个低维度的<strong>潜在空间 (Latent Space)</strong>。这个潜在表示保留了原始数据的主要信息，但尺寸小得多。
*   **扩散过程:** 然后，在低维的潜在空间中执行扩散模型的加噪和去噪过程。
*   <strong>解码器 (Decoder):</strong> 最后，使用 VAE 的解码器将去噪后的潜在表示恢复成高分辨率的图像/视频。

    由于扩散过程在计算量更小的潜在空间中进行，LDM 极大地提高了训练和推理的效率，使得生成高分辨率内容成为可能。`Matrix-Game` 正是基于 LDM 的思想构建的。

### 3.1.4. 扩散型 Transformer (Diffusion Transformer, DiT)
传统的扩散模型通常使用 `U-Net` 作为去噪网络。`DiT` 是一种将 `Transformer` 架构应用于扩散模型去噪过程的创新。它将经过 VAE 编码的潜在表示视为一系列<strong>词元 (tokens)</strong>（类似于自然语言处理中的单词），然后利用 `Transformer` 强大的序列建模能力来学习这些词元之间的关系，从而进行去噪。`Transformer` 架构具有很好的可扩展性，模型参数和数据量越大，性能通常越好，这使得 `DiT` 成为构建超大规模生成模型（如 `Sora` 和本文的 `Matrix-Game`）的理想选择。

## 3.2. 前人工作
论文将相关工作分为三类：

1.  <strong>视频扩散模型 (Video Diffusion Models):</strong>
    *   随着扩散模型在图像生成领域的成功，研究者们将其扩展到视频生成。早期的模型多基于 `U-Net`，而近期如 `Sora` 等工作开始转向更具扩展性的 `Transformer` 架构，能够生成更长、更连贯的视频。
    *   这些模型展示了从原始视频数据中学习物理规律、物体动态和因果关系的潜力，因此被视为构建世界模型的有力工具。

2.  <strong>可控视频生成 (Controllable Video Generation):</strong>
    *   仅靠文本提示很难精确控制视频的细节。因此，研究者们引入了额外的控制信号，如参考图像、相机运动轨迹等。
    *   `Direct-a-Video`、`CameraCtrl`、`MotionCtrl` 等工作探索了使用相机参数来控制视频视角。
    *   另一些世界模型方法则强调通过<strong>动作 (action)</strong> 来控制视频的演化，但这通常受限于较低的视觉质量或简化的动作空间。
    *   `Matrix-Game` 专注于<strong>交互式图像到世界 (interactive image-to-world)</strong> 的生成，通过键盘和鼠标动作进行精确和直观的控制。

3.  <strong>游戏视频生成 (Game Video Generation):</strong>
    *   利用视频生成模型来模拟游戏世界是一个新兴的研究方向。
    *   `Genie` 提出了一个可玩环境的基础模型。`DIAMOND`、`GameNGen`、`OASIS` 和 `PlayGen` 等工作也利用扩散模型来模拟游戏世界。
    *   `GameGenX` 和 `WorldMem` 等同期工作也做出了贡献，但作者认为它们往往对特定的游戏数据集过拟合，泛化能力有限。
    *   `Matrix`、`Genie 2`、`GameFactory` 和 `MineWorld` 等工作尝试提高控制的泛化性，但受限于模型容量和数据规模，难以有效捕捉物理规则和实现精确交互。

## 3.3. 差异化分析
`Matrix-Game` 与先前工作的核心区别和创新点在于：

*   **规模上的巨大飞跃：** 无论是 **170 亿**的模型参数，还是总计近 **4000 小时**的训练数据，`Matrix-Game` 在规模上都远超此前的开源游戏世界模型。这种规模化是其实现高质量生成和精确控制的基础。
*   **纯视觉的“图像到世界”范式：** 与许多依赖文本提示的模型不同，`Matrix-Game` 刻意排除了文本输入，强迫模型直接从视觉和动作信号中学习世界的几何、物理和动态规律。作者认为这能减少语义偏见，让模型更忠实于物理世界。
*   **系统性的解决方案：** 本文不只是提出了一个新模型，而是提供了一个从<strong>数据构建 (`Matrix-Game-MC`)</strong> 到 <strong>模型训练 (`Matrix-Game`)</strong> 再到 <strong>性能评估 (`GameWorld Score`)</strong> 的完整闭环。这种系统性的方法论是其重要贡献，为后续研究铺平了道路。
*   **高质量的动作控制：** 相较于之前模型有限或简化的动作空间，`Matrix-Game` 能够处理精细的键盘和鼠标输入，实现了前所未有的高精度交互控制，这在定量（动作准确率）和定性（人类评估）上都得到了验证。

    ---

# 4. 方法论

`Matrix-Game` 的方法论可以分为三个主要部分：大规模数据集的构建、交互式世界模型的设计与训练。

## 4.1. Matrix-Game-MC: 大规模游戏世界数据集

高质量、大规模的数据是训练强大世界模型的基石。作者以《我的世界》为目标环境，通过两种互补的方式构建了 `Matrix-Game-MC` 数据集。

### 4.1.1. 无标签数据收集与过滤
这部分数据用于让模型学习游戏世界的通用视觉特征、环境动态和物理规则，而不需要动作标注。

*   **数据来源：** 从公开的 `MineDojo` 数据集中获取了约 6000 小时的原始游戏视频。
*   **处理流程：**
    1.  **场景分割：** 使用 `TransNet V2` 模型检测视频中的场景切换点，并将长视频分割成独立的镜头片段。
    2.  **分层过滤：** 设计了一个三阶段的过滤流水线来筛选出高质量的视频片段。如下图（原文 Figure 3）所示：

        ![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171358.png)

        *   <strong>第一阶段 (质量过滤):</strong>
            *   `Video quality filtering`: 使用 `DOVER` 模型评估视频的清晰度、分辨率等技术质量。
            *   `Aesthetic filtering`: 使用 `LAION` 预测器评估画面的美学分数，保留视觉上吸引人的内容。
        *   <strong>第二阶段 (内容过滤):</strong>
            *   `Menu-State filtering`: 使用<strong>逆动力学模型 (Inverse Dynamics Model, IDM)</strong> 检测无玩家输入的画面（如菜单、暂停界面）并剔除。
            *   `Subtitle filtering`: 使用 `CRAFT` 文本检测器移除带有多余字幕或水印的视频。
            *   `Human face filtering`: 使用 `DeepFace` 检测并移除包含主播头像等非游戏内容的视频。
        *   <strong>第三阶段 (动态过滤):</strong>
            *   `Motion filtering`: 使用 `GMFlow` 计算光流，剔除运动过少（静态）或过多（剧烈晃动）的片段。
            *   `Camera movement filtering`: 使用 IDM 估计相机旋转，剔除视角变化过于剧烈的片段。

*   **最终产出：** 经过这套严格的流程，最终获得了 **2700 小时**的高质量无标签视频数据用于第一阶段的预训练。

### 4.1.2. 有标签数据创建
这部分数据包含精确的动作标注，是训练模型可控性的关键。

*   **数据来源：**
    1.  <strong>探索智能体 (Exploration agent):</strong> 在 `MineRL` 平台中部署经过课程学习引导的 `VPT` 智能体。这些智能体能自主探索游戏世界并完成任务，同时记录下它们每帧的键盘和鼠标操作。
    2.  <strong>虚幻引擎程序化模拟 (Unreal Procedural Simulation):</strong> 在虚幻引擎 (Unreal Engine) 中构建了多种自定义环境（城市、沙漠等），通过脚本精确控制智能体行为，从而生成带有完美、无噪声标注的数据。这部分数据作为对 `MineRL` 数据的补充。

*   **高质量数据策展策略：**
    *   **相机运动限制：** 在数据生成时，限制相机每帧的偏航 (yaw) 和俯仰 (pitch) 角度变化在 15° 以内，避免剧烈晃动，保证视频的视觉稳定性。
    *   **MineRL 引擎修改：** 修改了游戏引擎，禁用了会导致地形突然出现的“区块加载”机制，并自动停止记录濒死或暂停等无效片段，以保证视频画面的连贯性和内容质量。
    *   **场景多样化：** 精心挑选了 14 种不同的《我的世界》生物群系（如森林、沙漠、海洋、蘑菇岛等），并平衡了各种动作（移动、跳跃、攻击）的采样，以提高模型的泛化能力。下表（原文 Table 1）展示了部分场景的分布。

        <table><tr><td>Biome</td><td>Percentage</td><td>Environmental Features</td></tr><tr><td>Forest</td><td>4.0%</td><td>Dense trees, wolves, flowers, mushrooms</td></tr><tr><td>Taiga</td><td>4.5%</td><td>Spruce trees, foxes, berry bushes, snowfall</td></tr><tr><td>Swamp</td><td>4.6%</td><td>Mangrove trees, slime blocks, lily pads, vines</td></tr><tr><td>Ocean</td><td>7.2%</td><td>Coral reefs, kelp forests, drowned ruins, prismarine</td></tr><tr><td>Mesa</td><td>6.7%</td><td>Red sandstone, hardened clay strata, dead bushes</td></tr><tr><td>Extreme hills</td><td>7.2%</td><td>Mountain peaks, emerald ore, snowcaps, waterfalls</td></tr><tr><td>Savanna</td><td>6.3%</td><td>Baobab trees, acacia wood, herds of llamas</td></tr><tr><td>Plains</td><td>6.0%</td><td>Rolling grasslands, villages, sunflowers, horses</td></tr><tr><td>Beach</td><td>6.5%</td><td>Sandy shores, turtle nests, sugarcane, shallow waters</td></tr><tr><td>Jungle</td><td>5.9%</td><td>Giant trees, cocoa plants, ocelots, temple ruins</td></tr><tr><td>River</td><td>5.8%</td><td>Flowing water, clay deposits, salmon, gravel banks</td></tr><tr><td>Desert</td><td>7.9%</td><td>Sand dunes, cacti, desert temples, husks</td></tr><tr><td>Mushroom</td><td>6.3%</td><td>Mycelium terrain, giant mushrooms, mooshrooms</td></tr><tr><td>Icy</td><td>6.8%</td><td>Icebergs, polar bears, packed ice, strays</td></tr><tr><td>Random</td><td>14%</td><td>Random spawn for scenarios like Nether/End</td></tr></table>

*   **最终产出：** 最终获得了超过 **1200 小时**的高质量、多样化、动作均衡的有标签视频数据，用于第二阶段的交互式生成训练。

## 4.2. Matrix-Game: 交互式世界基础模型

`Matrix-Game` 是一个基于潜在扩散模型的 Transformer 架构，专为交互式图像到世界生成而设计。

### 4.2.1. 模型架构与设计
下图（原文 Figure 4）展示了模型的核心范式。

![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171400.png)

1.  <strong>图像到世界 (Image-to-World) 建模：</strong>
    *   模型以一张<strong>参考图像 (reference image)</strong> 作为生成视频的起点和主要条件。这种设计旨在让模型纯粹从视觉信息中学习世界的空间结构、物体动态和物理交互，而不是依赖可能引入偏见的文本描述。
    *   参考图像和待生成的视频首先通过一个 <strong>3D 因果 VAE (3D Causal VAE)</strong> 被压缩到低维度的潜在空间。这个 VAE 将视频在空间维度上压缩 8 倍，在时间维度上压缩 4 倍。

2.  <strong>自回归生成与扩散型 Transformer (MMDiT)：</strong>
    *   为了生成长视频，模型采用<strong>自回归 (autoregressive)</strong> 策略，即逐段生成视频。如下图（原文 Figure 5）所示，当前段落生成的最后几帧（论文中为 5 帧）会被用作下一段生成的<strong>运动条件 (motion context)</strong>，以保证段落之间的衔接流畅。
    *   模型的主体是一个<strong>多模态扩散型 Transformer (Multi-Modal Diffusion Transformer, MMDiT)</strong>，它接收多种输入并进行融合处理。

        ![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171402.png)

    *   **输入融合：** 在生成第 N+1 段视频时，MMDiT 的输入包括：
        *   <strong>带噪的潜在表示 (Noisy Latent):</strong> 目标视频段的随机噪声潜在表示。
        *   <strong>运动条件 (Motion Frames):</strong> 第 N 段视频最后几帧的潜在表示，与带噪潜在表示在通道维度上拼接。一个二进制掩码会指明哪些部分是有效的运动信息。
        *   <strong>参考图像 (Reference Image):</strong> 经过视觉编码器处理后的图像词元，与前两者的潜在词元在词元维度上拼接。
        *   <strong>动作信号 (Action Signals):</strong> 用户的键盘和鼠标动作，作为额外的条件注入到 Transformer 模块中。
    *   **训练技巧：** 为了提高自回归生成的稳定性，训练时会以一定概率向运动条件和参考图像中加入噪声，并使用<strong>无分类器指导 (Classifier-Free Guidance, CFG)</strong> 策略，即以一定概率将运动条件置零，迫使模型更好地学习利用运动上下文。

3.  **动作注入实现可控生成：**
    *   为了让模型响应用户操作，动作信号被精确地注入到 MMDiT 的每个 Transformer 模块中。如下图（原文 Figure 6c）所示，键盘和鼠标动作被分别处理。

        ![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171403.png)

    *   <strong>键盘动作 (Keyboard Actions):</strong> 包括“前、后、左、右、跳、攻击”等离散动作。它们被编码后，通过<strong>交叉注意力 (cross-attention)</strong> 机制注入到 Transformer 模块中，指导生成内容。
    *   <strong>鼠标动作 (Mouse Movements):</strong> 鼠标移动代表连续的相机视角变化（俯仰角）。它被视为连续值，与输入的潜在表示拼接，经过一个 MLP 和时间自注意力层处理后融入模型。
    *   同样，训练时也对动作信号使用 CFG 策略（以 0.1 的概率置零），以增强模型对动作信号的依赖和响应精度。

## 4.3. 模型训练
训练过程分为两个阶段，以实现从理解世界到与之交互的渐进式学习。

*   **训练范式：** 模型采用<strong>流匹配 (Flow Matching)</strong> 范式进行训练，并使用<strong>矫正流损失 (Rectified Flow Loss)</strong>。相比传统的 DDPM，这种方法收敛更快，采样效率更高。

*   <strong>第一阶段：无标签训练以理解游戏世界 (Stage 1: Unlabeled Training)</strong>
    *   **目标：** 让模型学习游戏世界的空间布局、物体动态和基本物理规则。
    *   **初始化：** 模型权重由预训练的 `HunyuanVideo` 图生视频模型初始化，并将原来的文本条件分支替换为图像条件分支。
    *   **训练数据：** 使用 `Matrix-Game-MC` 数据集中的 **2700 小时无标签视频**。训练时混合了多种帧数和宽高比，以增强模型的鲁棒性。
    *   **微调：** 在大规模预训练后，使用一个包含 **870 小时**的更高质量（运镜平稳、界面干净）的子集进行微调，以提升模型的视觉和物理理解能力。

*   <strong>第二阶段：动作标签训练以实现交互生成 (Stage 2: Action-Labeled Training)</strong>
    *   **目标：** 在理解世界的基础上，学习响应用户的动作指令，实现可控的视频生成。
    *   **模型调整：** 在 MMDiT 中集成<strong>动作控制模块 (Action Control Module)</strong>。最终模型 `Matrix-Game` 包含 **170 亿参数**。
    *   **训练数据：** 使用 `Matrix-Game-MC` 数据集中的 **1200 小时有标签视频**（包含 Minecraft 和 Unreal Engine 数据）。
    *   **数据均衡与长时序训练：** 为了解决场景类别不平衡问题，作者进一步整理了数据，使其在 8 个主要生物群系中分布更均匀。并在此均衡数据集上，将训练的视频长度从 33 帧增加到 65 帧，以加强模型对长程时间依赖的捕捉能力。

        通过这两阶段的训练，`Matrix-Game` 最终学会了从单张图像出发，根据用户的精确控制，生成连贯、真实且可交互的虚拟世界视频。

---

# 5. 实验设置

## 5.1. 数据集
实验的核心数据集是本文提出的 **Matrix-Game-MC**，其详细构建过程已在方法论部分阐述。关键信息回顾：
*   **来源:** 结合了公开数据集 (`MineDojo`) 的筛选和自动化智能体 (`MineRL`, `Unreal Engine`) 的生成。
*   **规模:**
    *   无标签数据: 约 2,700 小时。
    *   有标签数据: 约 1,200 小时，覆盖 14 种《我的世界》生物群系。
*   **特点:** 规模大、质量高、标注精细（键盘和鼠标）、场景多样且均衡。下图展示了模型在不同场景（包括 Minecraft 和 Unreal Engine）下的生成效果，直观反映了数据集的多样性。

    ![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171356.png)

    ![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171409.png)

## 5.2. 评估指标
本文的一大核心贡献是提出了一个统一的评估基准 **GameWorld Score**，专门用于评估像《我的世界》这样的交互式世界模型。它包含四大支柱和八个细分维度。

![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171404.png)

### 5.2.1. 视觉质量 (Visual Quality)
评估单帧画面的好坏。

1.  <strong>美学质量 (Aesthetic Quality):</strong>
    *   **概念定义:** 衡量生成画面的主观美感，包括构图、色彩、光影等是否符合人类审美偏好。
    *   **计算方式:** 使用在 `LAION` 数据集上训练的<strong>美学预测器 (Aesthetic Predictor)</strong> 对每帧图像打分，分数越高表示越美观。
2.  <strong>图像质量 (Image Quality):</strong>
    *   **概念定义:** 衡量画面的客观保真度，即是否存在模糊、噪声、过曝、压缩失真等低级伪影。
    *   **计算方式:** 使用<strong>无参考图像质量评估模型 <code>MUSIQ</code></strong> 对每帧图像打分，分数越高表示图像越清晰、伪影越少。

### 5.2.2. 时间质量 (Temporal Quality)
评估视频在时间维度上的连贯性和平滑性。

1.  <strong>时间一致性 (Temporal Consistency):</strong>
    *   **概念定义:** 衡量视频中静态背景和场景元素是否随时间保持稳定，避免出现闪烁、纹理漂移等问题。
    *   **计算方式:** 使用 `CLIP` 模型提取视频中每帧的特征向量，然后计算相邻帧特征向量之间的<strong>余弦相似度 (Cosine Similarity)</strong>，并取平均值。
    *   **数学公式:**
        $$
        \text{Temporal Cons.} = \frac{1}{N-1} \sum_{i=1}^{N-1} \frac{f_i \cdot f_{i+1}}{\|f_i\| \|f_{i+1}\|}
        $$
    *   **符号解释:**
        *   $N$: 视频的总帧数。
        *   $f_i$: 第 $i$ 帧图像经过 `CLIP` 编码器得到的特征向量。
2.  <strong>运动平滑性 (Motion Smoothness):</strong>
    *   **概念定义:** 衡量视频中的物体和相机运动是否流畅、自然，没有卡顿或不合逻辑的跳跃。
    *   **计算方式:** 使用一个预训练的<strong>视频插帧网络 (Video Frame Interpolation Network)</strong>。具体来说，从视频中取第 `i-1` 帧和第 $i+1$ 帧，让插帧网络预测出中间的第 $i$ 帧。然后计算预测帧与视频中真实第 $i$ 帧之间的重建误差。误差越小，说明原始视频的运动轨迹越平滑、越可预测。

### 5.2.3. 动作可控性 (Action Controllability)
评估生成的视频是否准确响应了输入的动作指令。

*   **评估工具:** 使用一个在大量《我的世界》游戏视频上预训练的<strong>逆动力学模型 (Inverse Dynamics Model, IDM)</strong>。IDM 的作用是“看”一段视频，然后反推出玩家当时可能进行了什么操作。通过比较 IDM 推断的动作和我们输入的真实动作，就可以衡量可控性。

1.  <strong>键盘控制准确率 (Keyboard Control Accuracy):</strong>
    *   **概念定义:** 衡量模型对前进、后退、跳跃、攻击等离散键盘指令的响应准确度。
    *   **计算方式:** 将动作分为四个独立的类别（如`前进/后退/无`，`左/右/无`等），计算 IDM 预测的动作与真实指令的<strong>精确率 (Precision)</strong>。
2.  <strong>鼠标控制准确率 (Mouse Control Accuracy):</strong>
    *   **概念定义:** 衡量模型对鼠标控制的相机视角转动（上、下、左、右及其组合的八个方向）的响应准确度。
    *   **计算方式:** IDM 预测视频中的相机运动方向，如果与输入的鼠标指令方向一致，则视为正确。计算所有正向预测的<strong>精确率 (Precision)</strong>。

### 5.2.4. 物理规则理解 (Physical Rule Understanding)
评估视频是否遵循基本的物理规律。

1.  <strong>物体一致性 (Object Consistency):</strong>
    *   **概念定义:** 衡量视频中的物体在三维空间中的几何结构是否随时间保持稳定，即使外观（如光照）发生变化。
    *   **计算方式:** 使用 `DROID-SLAM` 算法从生成的视频中估计相机位姿和场景深度图。然后计算跨帧共视像素点的<strong>重投影误差 (Reprojection Error)</strong>。误差越低，说明物体的 3D 结构越稳定。
2.  <strong>场景一致性 (Scenario Consistency):</strong>
    *   **概念定义:** 衡量模型对整个场景的记忆和重建能力。
    *   **计算方式:** 设计一种特殊的对称运动测试：让相机先朝一个方向移动（例如，向左移动一段距离，使初始场景移出视野），然后再沿原路返回。理想情况下，返回路径上看到的画面应该与去程路径上对应的画面完全一致。该指标通过计算这两组对应帧之间的<strong>均方误差 (Mean Squared Error, MSE)</strong> 来衡量场景的一致性，误差越低越好。

## 5.3. 对比基线
论文将 `Matrix-Game` 与两个当时最先进的、开源的《我的世界》世界模型进行了比较：
*   **OASIS [9]:** 一个基于 Transformer 的游戏世界生成模型。
*   **MineWorld [18]:** 一个专注于《我的世界》的实时、开源交互式世界模型。

    选择这两个模型作为基线，是因为它们是公开可用的，并且在《我的世界》生成任务上展示了有竞争力的结果，具有很好的代表性。

---

# 6. 实验结果与分析

## 6.1. 核心结果分析

### 6.1.1. GameWorld Score 基准测试结果
实验的核心定量结果体现在 GameWorld Score 基准测试上。

以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Visual Quality</th>
<th colspan="2">Temporal Quality</th>
<th colspan="2">Action Controllability</th>
<th colspan="2">Physical Understanding</th>
</tr>
<tr>
<th>Image Quality ↑</th>
<th>Aesthetic ↑</th>
<th>Temporal Cons. ↑</th>
<th>Motion smooth. ↑</th>
<th>Keyboard Acc. ↑</th>
<th>Mouse Acc. ↑</th>
<th>Obj. Cons. ↑</th>
<th>Scenario Cons. ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>Oasis [9]</td>
<td>0.65</td>
<td>0.48</td>
<td>0.94</td>
<td>0.98</td>
<td>0.77</td>
<td>0.56</td>
<td>0.56</td>
<td>0.86</td>
</tr>
<tr>
<td>MineWorld [18]</td>
<td>0.69</td>
<td>0.47</td>
<td>0.95</td>
<td>0.98</td>
<td>0.86</td>
<td>0.64</td>
<td>0.51</td>
<td>0.92</td>
</tr>
<tr>
<td>Ours</td>
<td><strong>0.72</strong></td>
<td><strong>0.49</strong></td>
<td><strong>0.97</strong></td>
<td>0.98</td>
<td><strong>0.95</strong></td>
<td><strong>0.95</strong></td>
<td><strong>0.76</strong></td>
<td><strong>0.93</strong></td>
</tr>
</tbody>
</table>

下图（原文 Figure 2）以雷达图形式直观展示了这一对比：

![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171409.png)

**分析：**
*   **全面领先：** `Matrix-Game` (Ours) 在所有八个维度上均取得了最佳性能，展示了其综合能力的强大。
*   **可控性巨大优势：** 最显著的提升来自于**动作可控性**。键盘准确率 (`Keyboard Acc.`) 从基线的 0.86 提升到 0.95，而鼠标准确率 (`Mouse Acc.`)更是从 0.64 **飙升至 0.95**。这证明 `Matrix-Game` 能够极其精确地响应用户的交互指令，这是其作为“交互式”世界模型的关键。
*   **物理理解显著增强：** 在**物理规则理解**方面，`Matrix-Game` 的物体一致性 (`Obj. Cons.`) 得分为 0.76，远高于基线的 0.51 和 0.56，表明其生成的物体在 3D 空间中更加稳定。场景一致性 (`Scenario Cons.`) 也达到了最优水平。
*   **视觉与时间质量保持顶尖：** 在保证可控性和物理性的同时，`Matrix-Game` 在传统的视觉质量（图像质量、美学）和时间质量（时间一致性、运动平滑度）方面也略胜一筹。

### 6.1.2. 人类评估结果
为了补充客观指标，作者进行了双盲人类评估，结果如下图（原文 Figure 8）所示。

![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171405.png)

**分析：**
*   **压倒性偏好：** 评估者在不知道模型来源的情况下，对三个模型的生成结果进行比较。`Matrix-Game` 在所有四个维度上都获得了压倒性的“胜率 (Win Rate)”。
*   <strong>总体质量 (Overall Quality):</strong> 96.3% 的情况下，人类评估者认为 `Matrix-Game` 的生成结果是最好的。
*   <strong>可控性 (Controllability):</strong> 93.76% 的胜率再次印证了其在交互控制上的巨大优势。
*   **主观与客观一致：** 人类评估的结果与 `GameWorld Score` 的客观分数高度一致，这不仅证明了 `Matrix-Game` 的卓越性能，也反向验证了 `GameWorld Score` 作为一个评估基准的有效性和可靠性。

## 6.2. 消融实验/参数分析

### 6.2.1. 动作可控性细分分析
为了深入探究模型的可控性，作者对不同类型的键盘和鼠标动作的准确率进行了详细分析。

以下是原文 Table 3 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="6">Keyboard Action</th>
<th colspan="8">Mouse Movement Action</th>
</tr>
<tr>
<th>forward</th>
<th>backward</th>
<th>left</th>
<th>right</th>
<th>jump</th>
<th>attack</th>
<th>camera↑</th>
<th>camera↓</th>
<th>camera←</th>
<th>camera→</th>
<th>camera↖</th>
<th>camera↗</th>
<th>camera↙</th>
<th>camera↘</th>
</tr>
</thead>
<tbody>
<tr>
<td>Oasis [9]</td>
<td>0.85</td>
<td>0.78</td>
<td>0.80</td>
<td>0.79</td>
<td>0.77</td>
<td>0.89</td>
<td>0.66</td>
<td>0.55</td>
<td>0.33</td>
<td>0.35</td>
<td>0.56</td>
<td>0.53</td>
<td>0.45</td>
<td>0.51</td>
</tr>
<tr>
<td>MineWorld [18]</td>
<td>0.86</td>
<td>0.80</td>
<td>0.87</td>
<td>0.88</td>
<td>0.82</td>
<td>0.87</td>
<td>0.46</td>
<td>0.45</td>
<td>0.53</td>
<td>0.54</td>
<td>0.66</td>
<td>0.77</td>
<td>0.87</td>
<td>0.96</td>
</tr>
<tr>
<td>Ours</td>
<td><strong>0.99</strong></td>
<td><strong>0.91</strong></td>
<td><strong>0.92</strong></td>
<td><strong>0.96</strong></td>
<td><strong>0.88</strong></td>
<td><strong>0.95</strong></td>
<td><strong>0.91</strong></td>
<td><strong>0.98</strong></td>
<td><strong>0.89</strong></td>
<td><strong>0.90</strong></td>
<td><strong>0.92</strong></td>
<td><strong>0.97</strong></td>
<td><strong>0.98</strong></td>
<td><strong>0.98</strong></td>
</tr>
</tbody>
</table>

**分析：**
*   `Matrix-Game` 在**所有**细分的动作类别上都取得了最高的准确率。
*   对于键盘动作，如“前进 (forward)”准确率高达 99%。
*   对于更具挑战性的鼠标控制，基线模型在某些方向（如 `Oasis` 的 `camera←` 只有 0.33）上表现很差，而 `Matrix-Game` 在所有八个方向上均保持了约 90% 或更高的准确率，展现了其对精细、连续控制信号的强大理解能力。
*   下图（原文 Figure 12, 13, 14）直观展示了模型对各种简单及复杂组合动作的精确响应。

    ![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171407.png)

    ![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171408.png)

    ![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171408.png)

### 6.2.2. 场景泛化能力分析
为了检验模型是否只在某些特定场景下表现良好，作者在 8 种不同的《我的世界》生物群系上分别进行了测试。

结果如下图（原文 Figure 9）所示：

![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171406.png)

**分析：**
*   在**所有八个场景**中，`Matrix-Game` 的雷达图面积都最大，表明其性能在不同环境中都保持了全面领先。
*   特别是在 `icy`（冰原）和 `mushroom`（蘑菇岛）这类视觉风格独特、出现频率可能较低的场景中，`Matrix-Game` 依然保持了极高的可控性和物理一致性，证明了其强大的泛化能力。这得益于其大规模、多样化的训练数据。

### 6.2.3. 自回归长视频生成分析
作者还定性展示了模型在自回归模式下生成长视频的能力。

![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171359.png)

**分析：**
*   上图（原文 Figure 10）展示了连续生成三个视频片段的结果。尽管每个片段是独立生成的，但片段之间的衔接非常平滑，没有出现明显的几何错位或运动跳变。
*   模型能够跨越多个片段，持续响应用户的动作指令（如 `forward` -> `left` -> `forward`），保持了行为的连贯性。这证明了其自回归生成策略和运动条件设计的有效性。

### 6.2.4. 失败案例分析
作者坦诚地展示了模型的局限性。

![](https://raw.githubusercontent.com/Strivin-007/Image-Hosting/main/20240713171406.png)

*   <strong>边缘场景泛化问题 (a):</strong> 在一些非常罕见或训练数据覆盖不足的场景中，模型可能难以保持时间一致性，出现画面崩坏的情况。
*   <strong>物理理解不足 (b):</strong> 尽管物理一致性得分很高，但模型对某些精细的物理交互（如碰撞）的理解仍有不足。例如，图中角色直接“穿过”了树叶，而不是与之发生碰撞。

    ---

# 7. 总结与思考

## 7.1. 结论总结
本文成功地推出了 `Matrix-Game`，一个用于生成可控游戏世界的交互式基础模型。其核心贡献是系统性的，涵盖了从数据、模型到评估的全链路：
1.  <strong>构建了 <code>Matrix-Game-MC</code>：</strong> 一个迄今为止规模最大、标注最精细的《我的世界》视频数据集，为训练强大的世界模型奠定了数据基础。
2.  **提出了 `Matrix-Game` 模型：** 一个 170 亿参数的潜在扩散模型，通过“图像到世界”范式和两阶段训练策略，实现了前所未有的高精度动作控制和高质量视频生成。
3.  **开发了 `GameWorld Score` 基准：** 一套专为交互式世界模型设计的全面评估体系，填补了该领域的评估空白。

    实验结果有力地证明，`Matrix-Game` 在视觉质量、时间连贯性，尤其是在**动作可控性**和**物理一致性**方面，全面超越了现有开源模型，为构建更真实、更具交互性的虚拟世界迈出了重要一步。

## 7.2. 局限性与未来工作
作者指出了当前工作的局限性，并展望了未来的研究方向：

*   **局限性：**
    *   **泛化能力：** 在数据稀疏的边缘场景中，模型的性能会下降。
    *   **物理理解：** 对碰撞、支撑等复杂物理交互的建模仍有待提高。

*   **未来工作：**
    1.  **提升长时序一致性：** 通过引入更长的运动上下文或记忆机制，进一步改善超长视频的连贯性。
    2.  **丰富动作空间：** 扩展支持的键盘动作类型，并实现更连续、更精细的鼠标控制。
    3.  **超越《我的世界》：** 将该框架扩展到更复杂、视觉更写实的游戏环境中，如《黑神话：悟空》、赛车模拟器或 `CS:GO` 等多智能体对抗游戏，迎接新的挑战。

## 7.3. 个人启发与批判
这篇论文给我带来了深刻的启发，也引发了一些思考：

*   **启发：大力出奇迹，但要有章法。** `Matrix-Game` 的成功再次印证了“大数据+大模型”在 AI 领域的威力。但它并非盲目堆砌资源，而是通过精心设计的数据过滤流水线、高效的两阶段训练策略以及创新的评估体系，将庞大的资源用在了刀刃上。这种**系统工程**的思维方式对于解决复杂的 AI 问题至关重要。

*   **方法上的亮点：**
    *   <strong>“图像到世界”</strong>范式的纯粹性： 摒弃文本输入是一个大胆而明智的选择。它强迫模型从最本质的视觉和物理信息中学习，可能有助于模型构建一个更“通用”的物理世界表征，而不是被语言的抽象概念所束缚。
    *   **`GameWorld Score` 的价值：** 这个评估基准的提出，其意义甚至不亚于模型本身。它为“可交互世界模型”这个新兴领域提供了一把标尺，使得后续研究能够有据可依、量化比较，极大地推动了领域的规范化发展。

*   **潜在问题与批判性思考：**
    *   **对《我的世界》的依赖：** 尽管《我的世界》是一个极佳的试验场，但其方块化的世界和简化的物理规则与现实世界相去甚远。从 `Matrix-Game` 到一个能够模拟真实世界动态的通用世界模型，还有很长的路要走。模型学到的“物理规则”在多大程度上是《我的世界》的特有规则，在多大程度上是可泛化的物理规律，这是一个值得探究的问题。
    *   **评估的局限性：** `GameWorld Score` 虽已非常全面，但仍有可改进之处。例如，它对物理规则的评估主要集中在“一致性”上，对于更复杂的因果关系（如“攻击”动作是否真的能“破坏”方块）的评估还比较间接。
    *   **可解释性缺失：** 作为一个 170 亿参数的庞然大物，`Matrix-Game` 像一个黑箱。我们知道它能行，但不知道它“脑中”的世界是什么样的。如何可视化和理解其内部的世界表征，将是未来一个有趣且重要的方向。

*   **迁移价值：** `Matrix-Game` 的整套方法论可以被广泛应用于任何具有清晰动作定义和可模拟环境的领域，例如机器人模拟、虚拟现实内容生成、甚至是电影预演等。它为我们描绘了一个未来：用户不再是被动的观众，而是可以通过简单的交互，成为虚拟世界的导演和创造者。