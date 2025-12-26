# 可推广的类人操控与 3D 扩散策略

严杰 $\mathrm { Z e ^ { 1 } }$ 陈子轩2 王文浩3 陈天怡3 何霞林4 袁莹5 彭雪彬2 吴嘉峻1 1斯坦福大学 2西蒙弗雷泽大学 3宾夕法尼亚大学 4伊利诺伊大学厄本那-香槟分校 5卡内基梅隆大学 HUMANOID-MANIPULATION.GITHUB.IO

![](images/1.jpg)  
Fi.  Humanoid manipulation in divereunseen scenarios. With ur system we are abletcollct huma-ike using data only from a single scene, The scenes are not cherry-picked. Videos are available on our website.

摘要—能够在多种环境中自主操作的人形机器人一直是机器人研究者的目标。然而，人形机器人的自主操作在很大程度上受到局限，主要是由于难以获取具有普适性的技能以及在真实环境中获取人形机器人数据的高成本。在本研究中，我们构建了一个真实世界的机器人系统来解决这一挑战性问题。我们的系统主要集成了1) 一个完整的上半身机器人遥操作系统，用于获取类人机器人数据；2) 一个具备25个自由度的人形机器人平台，配备高度可调的推车和3D激光雷达传感器；3) 一个改进的3D扩散策略学习算法，帮助人形机器人从噪声人类数据中学习。我们在真实机器人上运行了超过2000个策略推演以进行严格的策略评估。通过这个系统，我们展示了仅使用在单一场景中收集的数据，并依靠机载计算，完整的人形机器人能够在多样的真实场景中自主执行技能。视频可在humanoid-manipulation.github.io上观看。

# I. 引言

能够在非结构化环境中执行多样化任务的机器人长期以来一直是机器人领域的重要目标，而智能类人机器人代表了一条有前景的途径。最近，在类人机器人硬件的开发以及这些机器人的远程操作和学习系统方面取得了显著进展。然而，由于所采用学习方法的有限泛化能力以及从不同场景获取类人机器人数据的高成本，这些自主类人操作技能都局限于其训练场景，难以推广到新场景。我们的工作旨在开发一种能够通过三维视觉运动策略学习可泛化类人操作技能的现实世界类人机器人学习系统。我们的系统概述见图2。首先，我们设计了一个类人机器人学习平台，其中一个29自由度的全尺寸类人机器人固定在一个可移动且高度可调的推车上。该平台能够稳定类人机器人，即使在腰部前倾时，也能安全利用类人机器人的腰部自由度。此外，机器人头部配备了用于可泛化策略学习的3D激光雷达传感器。其次，为了进行类人数据采集，我们设计了一个全上半身远程操作系统，将人类关节映射到全尺寸类人机器人。与常见的双手操作系统不同，我们的远程操作系统结合了腰部自由度和主动视觉，极大地扩展了机器人的操作工作空间，尤其是在处理不同高度的任务时。我们还将来自激光雷达传感器的实时视觉流传输给人类进行以自我为中心的远程操作。第三，为了利用以自我为中心的人类数据学习可泛化的操作技能，我们重新构建了第三人称3D学习算法3D扩散策略（DP3）为以自我为中心的版本，消除了对相机标定和点云分割的需求。通过超过2000个现实世界评估试验，我们在现实世界类人操作方面相较于原始DP3取得了显著的改进。生成的策略被称为改进的3D扩散策略（iDP3）。尽管这项工作仅在傅里叶GR1类人机器人上应用了iDP3，但我们强调iDP3是一种通用的3D学习算法，可应用于包括移动机器人和类人机器人在内的不同机器人平台。最后，我们将系统部署到未见的现实世界场景中。令人惊讶的是，由于我们3D表示的鲁棒性和平台的灵活性，我们的策略能够零样本泛化到许多随机选择的未见场景，如厨房、会议室和办公室。总之，我们构建了一个现实世界类人机器人系统，只需从一个单一场景中学习可泛化的操作技能，利用3D视觉运动策略。据我们所知，我们是首个成功使全尺寸类人机器人能够在多样化的未见场景中自主执行技能，并且仅通过来自单一场景的数据使用3D模仿学习的方法。

# II. 相关工作

在复杂的现实环境中，人形机器人自主执行多种技能长期以来一直是机器人技术的核心目标。最近，基于学习的方法在实现这一目标方面显示出良好的进展，特别是在行走 [23][27]、操控 [4][10][28] 和行走操控 [6][7][16][29] 领域。尽管已有多项研究成功展示了在人类生活环境中，机器人行走的能力 [23][24][26]，但在未见环境中的操控技能仍然基本未被探讨 [6][7][10]。

在表I中，我们列出了为人形机器人/灵巧操作构建现实世界机器人系统的近期研究工作。我们发现，现有人形机器人研究[3]、[4]、[6]、[7]、[10]、[22]在学习人形操作的一般化能力方面存在不足，主要是由于其算法的一般化能力有限和系统的灵活性不足。例如，OpenTeleVision [10] 和 HATO [22] 的平台不支持可移动的底座和腰部，限制了机器人的工作空间。HumanPlus [7] 和 OmniH2O [6] 可以对人形机器人进行全身遥控，但从其系统中学习到的操作技能仅限于训练场景，无法推广到其他场景，因为收集多样化数据的难度较大。Maniwhere [9] 在简单的桌面推送任务上实现了现实场景的一般化，但由于人形机器人的系统复杂性，难以将其仿真到真实的管道应用于人形机器人。同样，3D扩散策略（DP3，[2]）仅展示了桌面机器人手臂的物体/视角一般化。机器人实用模型 [30] 也通过模仿学习将技能进行场景的一般化，但他们必须使用从20个场景收集的数据进行场景一般化，而我们仅使用1个场景。在本文中，我们通过构建一个现实世界的人形机器人学习系统迈出了重要一步，使全尺寸人形机器人能够在未见过的真实场景中执行操作任务，利用3D视觉运动策略。

# III. 通用类人手臂操作与 3D 扩散策略

在这一部分，我们介绍了部署在全尺寸人形机器人上的真实世界模仿学习系统。该系统的概述如图2所示。表中展示了在先前人形机器人研究中缺失的多个方面的可推广的策略。

<table><tr><td rowspan="2">Method</td><td colspan="4">Teleoperation</td><td colspan="3">Generalization Abilities</td><td>Rigorous Policy Evaluation</td></tr><tr><td>Arm&amp;Hand</td><td>Head</td><td>Wais </td><td></td><td></td><td>Object Camera View</td><td>Scene</td><td>Real-World Episodes</td></tr><tr><td>AnyTeleop [1]</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>0</td></tr><tr><td>DP3 [2]</td><td></td><td>X</td><td>X</td><td>X</td><td></td><td>√</td><td>X</td><td>186</td></tr><tr><td>BunnyVisionPro [3]</td><td></td><td>X</td><td>X</td><td>X</td><td></td><td>X</td><td>X</td><td>540</td></tr><tr><td>ACE [4]</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>60</td></tr><tr><td>Bi-Dex [5]</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>50</td></tr><tr><td>OmniH20 [6]</td><td></td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>90</td></tr><tr><td>HumanPlus [7]</td><td></td><td>X</td><td>X</td><td></td><td>X</td><td>X</td><td>X</td><td>160</td></tr><tr><td>Hato [8]</td><td></td><td>X</td><td>X</td><td>X</td><td></td><td>X</td><td>X</td><td>300</td></tr><tr><td>ManiWhere [9]</td><td></td><td>X</td><td>X</td><td>X</td><td></td><td></td><td>√</td><td>20</td></tr><tr><td>OpenTeleVision [10]</td><td></td><td></td><td>X</td><td>X</td><td></td><td></td><td>X</td><td>75</td></tr><tr><td>This Work</td><td></td><td></td><td></td><td>X</td><td></td><td></td><td></td><td>2253</td></tr></table>

# A. 人形机器人平台

类人机器人。我们使用了 Fourier GR1，一款全尺寸类人机器人，配备了两个 Inspire 手。我们使整个上半身（包括头部、腰部、手臂和手）具有 25 个自由度（DoF）。我们禁用了下半身以保证稳定，并使用小车进行移动。尽管之前的系统如 HumanPlus 和 OmniH2O 显示了类人的腿部使用，但由于硬件限制，这些系统的动作操控技能仍然受限。我们强调，我们的系统结合了 3D 学习算法，具有通用性，能够推广到其他有腿和无腿的类人机器人。 激光雷达相机。为了捕获高质量的 3D 点云，我们使用了 RealSense L515，一款固态激光雷达相机。相机安装在机器人头部，提供自我中心视角。之前的研究表明，深度感知精度较低的相机，例如 RealSense D435，可能会导致 DP3 的性能不尽如人意。然而，值得注意的是，RealSense L515 也无法生成完全准确的点云。我们还尝试了其他激光雷达相机，如 Livox Mid-360，但发现这些激光雷达的分辨率和频率不支持富接触和实时的机器人操作。 高度可调小车。将操作技能推广到现实环境中的主要挑战是场景条件的广泛变化，特别是桌面的高度差异。为了解决这个问题，我们采用了高度可调的小车，消除了复杂的全身控制需求。虽然这简化了操作过程，但我们相信一旦全身控制技术趋于成熟，我们的方法将表现同样出色。

# B. 类人机器人数据

全身上半部远程操作。为了获取类人仿人机器人的数据，我们设计了一个能够远程操控机器人整个上半身的系统，包括头部、腰部、手部和手臂。我们使用 Apple Vision Pro (AVP, [35]) 获取准确的实时人类数据，例如头部/手部/手腕的三维位置和方向 [36]。利用这些人类数据，我们分别计算相应的机器人关节角度。更具体地说，1) 机器人手臂关节使用松弛逆运动学（Relaxed IK） [37] 计算，以跟踪人类手腕位置；2) 机器人腰部和头部关节通过人类头部的旋转进行计算。我们还将实时的机器人视觉反馈流回给人类，以实现沉浸式远程操作体验 [10]。 远程操作的延迟。使用激光雷达传感器显著占用机载计算机的带宽和CPU，导致远程操作的延迟约为0.5秒。我们还尝试了两台激光雷达传感器（其中一台额外安装在手腕上），这引入了极高的延迟，从而使数据收集不可行。 学习的数据。我们在远程操作期间收集观察-动作对的轨迹，其中观察由两个部分组成：1) 视觉数据，如点云和图像，以及 2) 自我感知数据，如机器人关节位置。动作由目标关节位置表示。我们还尝试使用末端执行器姿态作为自我感知/动作，发现直接将关节位置应用于动作空间更加准确，主要是由于现实世界中计算末端执行器姿态时的噪声。

# C. 改进的三维扩散策略

3D扩散策略（DP3，[2]）是一种有效的3D视觉运动策略，结合了稀疏点云表示和扩散策略。尽管DP3在广泛的操控任务中表现出色，但由于其对精确相机标定和细粒度点云分割的固有依赖，这使得它无法直接在通用机器人（如类人机器人或移动操控器）上部署。此外，DP3的准确性需要进一步提高，以在更复杂的任务中实现有效的性能。接下来，我们详细介绍几项修改，以实现针对性的改进。改进后的算法称为改进3D扩散策略（iDP3）。自我中心3D视觉表示。DP3利用了一个

![](images/2.jpg)  
s, hesor pl ear method,and he eal-wor deplymt. Wit th ystm, urun roo performs autonomous skills in diverse real-world scenes.

![](images/3.jpg)  
Fig. 3: iDP3 utilizes 3D representations in the camera frame, while the 3D representations of other recent 3D policies including DP3 [2] are in the world frame, which relies on accurate camera calibration and can not be extended to mobile robots.

在世界坐标系中的三维视觉表示，使目标对象的分割变得容易。然而，对于像类人机器人这样的通用机器人来说，相机支架并不固定，使得相机标定和点云分割变得不切实际。为了解决这个问题，我们建议直接使用来自相机框架的三维表示，如图3所示。我们将这一类三维表示称为自我中心三维视觉表示。扩展视觉输入。利用自我中心三维视觉表示在消除多余点云（例如背景或桌面）时面临挑战，特别是在不依赖基础模型的情况下。为此，我们提出了一个简单但有效的解决方案：扩展视觉输入。我们不再使用先前系统中的标准稀疏点采样，而是显著增加样本点的数量，以捕捉整个场景。尽管这个方法看似简单，但在我们的真实世界实验中证明是有效的。改进的视觉编码器。我们用金字塔卷积编码器替换DP3中的MLP视觉编码器。我们发现，在从人类数据中学习时，卷积层比全连接层产生更平滑的行为，结合来自不同层的金字塔特征进一步提高了准确性。更长的预测范围。人类专家的抖动和噪声传感器在学习人类示范时表现出很大的困难，导致DP3在短期预测方面遇到困难。通过扩展预测范围，我们有效地缓解了这个问题。实施细节。在优化方面，我们使用AdamW训练iDP3和所有其他方法，训练300个epoch。对于扩散过程，我们使用50个训练步骤和10个推理步骤，采用DDIM。对于点云采样，我们用体素采样和均匀采样的级联替代了DP3中使用的最远点采样（FPS），确保采样点覆盖三维空间，并提供更快的推理速度。

# D. 现实世界部署

我们在收集的人类演示数据上训练 iDP3。值得注意的是，我们并不依赖于前文提到的相机标定或手动点云分割。因此，我们的 iDP3 策略可以无缝迁移到新场景，而无需额外的标定/分割等工作。此外，iDP3 在仅用机器人 onboard 计算的情况下实现实时推理（15hz），使其在开放环境中的部署变得可行。

# 四. 实验与分析

为了评估我们系统的有效性，我们进行了广泛的真实世界消融实验。我们选择 Pick&Place 任务作为分析的主要基准，并进一步展示 Pick&Place、Pour 和 Wipe 任务在各种未见场景中的表现。

# A. 实验设置

任务描述。在此任务中，机器人抓取一个轻量级的杯子并将其移开。对于具有灵巧手的类人机器人来说，挑战在于杯子的大小与手相似；因此，即使是小的误差也会导致碰撞或抓取失败。与可以张开更宽以避免碰撞的平行夹爪相比，此任务需要更高的精度。

![](images/4.jpg)  
lzatn svatshgughtexi ve scenes. Videos are available on our website.

表 II：iDP3 与基线方法的效率比较。为了提高基线方法的鲁棒性，我们在训练期间对所有基于图像的方法添加了随机裁剪和颜色抖动增强。所有方法都经过超过 100 次试验评估，确保在实际评估中减少随机性。未经修改，原始 DP [17] 和 DP3 [2] 在我们的人形机器人上表现不佳。

<table><tr><td>Baselines</td><td>DP</td><td>DP3</td><td>DP (*R3M)</td><td>DP (*R3M)</td><td>iDP3 (DP3 Encoder)</td><td>iDP3</td></tr><tr><td>1st-1</td><td>0/0</td><td>0/0</td><td>11/33</td><td>24/39</td><td>15/34</td><td>21/38</td></tr><tr><td>1st-2</td><td>7/34</td><td>0/0</td><td>10/28</td><td>27/36</td><td>12/27</td><td>19/30</td></tr><tr><td>3rd-1</td><td>7/36</td><td>0/0</td><td>18/38</td><td>26/38</td><td>15/32</td><td>19/34</td></tr><tr><td>3rd-2</td><td>10/36</td><td>0/0</td><td>23/39</td><td>22/34</td><td>16/34</td><td>16/37</td></tr><tr><td>Total</td><td>24/106</td><td>0/0</td><td>62/138</td><td>99/147</td><td>58/127</td><td>75/139</td></tr></table>

任务设置。我们在四种设置下训练Pick&Place任务：$\{$ 1st-1, 1st-2, 3rd-1, 3rd-2 $\}$。 "1st" 使用自我中心视角，而 "3rd" 使用第三人称视角。后面的数字表示用于训练的演示数量，每个演示由20轮成功执行组成。训练数据集保持较小，以突出不同方法之间的差异。物体位置在 $10 \mathrm{cm} \times 20 \mathrm{cm}$ 区域内随机采样。评估指标。我们对每种方法运行三次实验，每次实验包括1,000个动作步骤。总的来说，每种方法大约经过130次实验进行评估，以确保对每种方法的全面评估。我们记录成功抓取的数量和总的抓取尝试次数。成功抓取数量反映了策略的准确性。总尝试次数则作为策略平滑性的度量，因为我们观察到抖动策略往往会停留在某个位置，尝试次数较少。

# B. 效能

我们将 iDP3 与几个强有力的基线进行比较，包括：a) DP：使用 ResNet18 编码器的扩散策略 [17]；b) DP (*R3M)：使用冻结的 R3M [41] 编码器的扩散策略；c) DP $( { \bf \star R 3 M } )$：使用微调 R3M 编码器的扩散策略；d) 原始 DP3 没有任何修改；e) iDP3 (DP3 编码器)：使用 DP3 编码器 [17] 的 iDP3。所有基于图像的方法均使用与 iDP3 相同的策略主干，并采用随机裁剪和颜色抖动增广以提高鲁棒性和泛化能力。RGB 图像的分辨率为 $224 \times 224$，由 RealSense 摄像头的原始图像调整大小而来。

![](images/5.jpg)  
Fig. 5: Trajectories of our three tasks in the training scene, including Pick&Place, Pour, and Wipe. We carefully select daily tasks so that the objects are common in daily scenes and the skills are useful across scenes.

结果如表 II 所示，iDP3 显著优于传统的 DP 和 DP3、使用冻结 R3M 编码器的 DP，以及使用 DP3 编码器的 iDP3。然而，我们发现使用微调 R3M 的 DP 是一个特别强的基线，在这些设置中优于 iDP3。我们假设这是因为微调预训练模型通常比从头开始训练更有效 [42]，而目前机器人领域内没有类似的预训练 3D 视觉模型。当在新场景中抓取训练物体时，DP 会产生抖动行为。

![](images/6.jpg)  
Fig. 6: Failure cases of image-based methods in new scenes. Here DP corresponds to DP $( { \bf { \star R 3 M } } )$ in Table II, which is methods still struggle in the new scene/object.

![](images/7.jpg)  
Fig. 7: Training time. Due to using 3D representations, iDP3 saves training time compared to Diffusion Policy (DP), even after we scale up the 3D vision input. This advantage becomes more evident when the number of demonstrations gets large.

尽管经过微调的 $\mathrm{DP^+}$ R3M 在这些设定中更为有效，但我们发现基于图像的方法在特定场景和物体上过拟合，无法推广到野外场景，如第四节D所示。此外，我们认为 iDP3 仍有改进的余地。由于传感硬件的限制，我们当前的三维视觉观测相当嘈杂。我们预计，更准确的三维观测能够在三维视觉运动策略中实现最佳性能，如模拟中所示 [2]。

# C. 消融实验

我们对DP3的几个改进进行了消融研究，包括改进的视觉编码器、缩放视觉输入和更长的预测范围。我们在表III中提供的结果表明，若没有这些改进，DP3要么无法有效地从人类数据中学习，要么表现出显著降低的准确性。具体而言，我们观察到1) 改进的视觉编码器能够提高策略的平滑性和准确性；2) 缩放的视觉输入是有帮助的，但在我们任务中随着点的数量增加，性能出现饱和；3) 适当的预测范围至关重要，缺少这一点，DP3无法从人类示范中学习。此外，图7展示了iDP3的训练时间，与扩散策略相比显著减少。这一效率在点云数量增加到DP3的几倍时保持不变。表III：iDP3的消融分析。结果表明，从iDP3中移除某些关键改进会显著影响DP3的性能，导致无法从人类数据中学习或准确性降低。所有方法均评估于100次以上的试验，确保了实际评估中的随机性较低。

<table><tr><td>Visual Encoder</td><td>1st-1</td><td>1st-2</td><td>3rd-1</td><td>3rd-2</td><td>Total</td></tr><tr><td>Linear (DP3)</td><td>15/34</td><td>12/27</td><td>15/32</td><td>16/34</td><td>58/127</td></tr><tr><td>Conv</td><td>9/33</td><td>14/32</td><td>14/33</td><td>12/33</td><td>49/131</td></tr><tr><td>Linear+Pyramid</td><td>15/34</td><td>20/31</td><td>13/33</td><td>18/36</td><td>66/134</td></tr><tr><td>Conv+Pyramid (iDP3)</td><td>21/38</td><td>19/30</td><td>19/34</td><td>16/37</td><td>75/139</td></tr><tr><td colspan="6">Number of Points</td></tr><tr><td></td><td>1st-1</td><td>1st-2</td><td>3rd-1</td><td>3rd-2</td><td>Total</td></tr><tr><td>1024 (DP3) 2048</td><td>11/28 17/35</td><td>10/30</td><td>18/35 17/32</td><td>17/36 18/33</td><td>56/129 65/128</td></tr><tr><td>4096 (iDP3)</td><td>21/38</td><td>13/28 19/30</td><td>19/34</td><td>16/37</td><td>75/139</td></tr><tr><td>8192</td><td>24/35</td><td>16/28</td><td>14/33</td><td>18/36</td><td>72/132</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">Prediction Horizon</td></tr><tr><td></td><td>1st-1</td><td>1st-2</td><td>3rd-1</td><td>3rd-2</td><td>Total</td></tr><tr><td>4 (DP3) 8</td><td>0/0 0/0</td><td>0/0</td><td>0/0</td><td>0/0 12/34</td><td>0/0 33/88</td></tr><tr><td>16 (iDP3)</td><td>21/38</td><td>3/18</td><td>18/36 19/34</td><td>16/37</td><td>75/139</td></tr><tr><td>32</td><td>9/34</td><td>19/30 20/30</td><td>14/33</td><td>12/33</td><td>55/130</td></tr></table>

# D. 能力

在这一节中，我们展示了我们的系统在类人机器人上的更强泛化能力。我们还对 iDP3 和 DP $( { \bf \star R 3 M } )$（在本节中简称为 DP）进行了更多比较，指出 iDP3 在复杂和具有挑战性的现实世界中更加适用。结果见表 IV。任务。我们选择了三个任务，即拿取与放置、倒水和擦拭，来演示我们的系统能力。我们确保这些任务在日常生活中是常见的并且对人类有用。例如，倒水在餐馆中经常进行，而擦拭则是在家庭中清洁桌子时常见的操作。数据。对于每个任务，我们收集了 10 次演示，$\times 10$ 表明 iDP3 对大视角变化具有鲁棒性。表 IV：iDP3 的能力。虽然 iDP3 在效率上与 DP $( { \bf \star R 3 M } )$（简称 DP）相似，但在复杂的厨房场景中表现突出。

<table><tr><td>Training</td><td>DP</td><td>iDP3</td><td>New Object</td><td>DP</td><td>iDP3</td><td>New View</td><td>DP</td><td>iDP3</td><td>New Scene</td><td>DP</td><td>iDP3</td></tr><tr><td>Pick&amp;Place</td><td>9/10</td><td>9/10</td><td>Pick&amp;Place</td><td>3/10</td><td>9/10</td><td>Pick&amp;Place</td><td>2/10</td><td>9/10</td><td>Pick&amp;Place</td><td>2/10</td><td>9/10</td></tr><tr><td>Pour</td><td>9/10</td><td>9/10</td><td>Pour</td><td>1/10</td><td>9/10</td><td>Pour</td><td>0/10</td><td>9/10</td><td>Pour</td><td>1/10</td><td>9/10</td></tr><tr><td>Wipe</td><td>10/10</td><td>10/10</td><td>Wipe</td><td></td><td></td><td>Wipe</td><td></td><td></td><td>Wipe</td><td></td><td></td></tr></table>

![](images/8.jpg)

![](images/9.jpg)  
DP fails to grasp training objects under large view changes.   
Fig. 8: View invariance of iDP3. We find that egocentric 3D representations are surprisingly view-invariant. Here DP corresponds to DP $( { \bf \# } { \bf R } { \bf 3 } { \bf M } )$ in Table II, which is the strongest image-based baseline we have.

推演，共计300个回合用于所有任务。对于拿起与放置和倒水，物体姿态在$10 \mathrm { cm } \times 10 \mathrm { cm }$的区域内随机化。有效性。如表IV所示，iDP3和DP在训练环境中对训练物体都实现了高成功率。特性1：视图不变性。我们的人本中心3D表示展示了令人印象深刻的视图不变性。如图8所示，iDP3能够在大视角变化下始终成功抓取物体，而DP在抓取训练物体时则表现不佳。DP仅在视角变化较小的情况下偶尔取得成功。值得注意的是，与最近的一些研究不同，我们没有为等变性或不变性采用特定设计。特性2：物体泛化。我们评估了除训练杯子外的新款杯子/瓶子，如图9所示。尽管由于使用了颜色抖动增强，DP偶尔能够处理未见过的物体，但成功率较低。相比之下，由于采用了3D表示法，iDP3能够自然地处理广泛的物体。特性3：场景泛化。我们进一步在各种真实世界场景中部署我们的策略，如图1所示。这些场景靠近实验室，并且没有选择特定场景。真实世界比实验室使用的受控桌面环境更嘈杂更复杂，这导致基于图像的方法准确率下降（图6）。与DP不同，iDP3在所有场景中展现了意想不到的鲁棒性。此外，我们在图4中提供了2D和3D观测的可视化。

![](images/10.jpg)  
Fig. 9: Objects used in Pick&Place and Pour. We only use the cups as the training objects, while our method naturally handles other unseen bottles/cups.

# V. 结论与局限性

结论。本研究提出了一种现实世界模仿学习系统，使得全尺寸的人形机器人能够将实用的操作技能推广到多样化的现实环境中，该系统只在单一场景中收集数据进行训练。通过2000多个严格的评估实验，我们展示了一种改进的3D扩散策略，能够从人类数据中稳健学习，并在我们的人形机器人上有效执行。我们的结果表明，人形机器人能够在多样化的现实场景中执行自主操作技能，显示了在现实操作任务中使用3D视觉运动策略的潜力及数据效率。局限性。1）虽然Apple Vision Pro的遥操作易于设置，但对于人类遥操作员来说是耗人的，这使得在研究实验室中增加模仿数据变得困难。2）深度传感器仍然产生嘈杂和不准确的点云，限制了iDP3的性能。3）由于使用AVP进行遥操作，收集细粒度操作技能（例如拧螺丝）耗时较长；目前阶段，像Aloha [18]这样的系统更易收集灵巧的操作任务。4）我们避免使用机器人的下半身，因为由于当前人形机器人所带来的硬件限制，保持平衡仍然具有挑战性。总体而言，扩展高质量操作数据是主要瓶颈。未来，我们希望探索如何利用更多高质量数据来扩展3D视觉运动策略的训练，以及如何将我们的3D视觉运动策略学习流程应用于具有全身控制的人形机器人。

# 致谢

我们感谢傅里叶智能的顾杰、周滨和蔡宇生提供的硬件支持，感谢高宇翔在远程操控方面的帮助，感谢胡硕在相机支架3D打印方面的帮助，感谢包泽前、陶仁志和谷佳妍的有益讨论。此外，我们还感谢斯坦福大学的王晨、张云志、李自章和熊浩宇的深刻讨论。这项工作部分得到了海军研究办公室MURI N00014-22-1-2740、海军研究办公室MURI N00014-24-1-2748和大川财团的资助。

# REFERENCES

[1] Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y.-W. Chao, and D. Fox, "Anyteleop: A general vision-based dexterous robot armhand teleoperation system," in Robotics: Science and Systems, 2023.   
[2] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, "3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations," arXiv preprint arXiv:2403.03954, 2024. [3] R. Ding, Y. Qin, J. Zhu, C. Jia, S. Yang, R. Yang, X. Qi, and X. Wang, "Bunny-visionpro: Real-time bimanual dexterous teleoperation for imitation learning," arXiv preprint arXiv:2407.03162, 2024. [4] S. Yang, M. Liu, Y. Qin, R. Ding, J. Li, X. Cheng, R. Yang, S. Yi, and X. Wang, "Ace: A cross-platform visual-exoskeletons system for low-cost dexterous teleoperation," arXiv preprint arXiv:2408.11805, 2024.   
[5] K. Shaw, Y. Li, J. Yang, M. K. Srirama, R. Liu, H. Xiong, R. Mendonca, and D. Pathak, "Bimanual dexterity for complex tasks," in 8th Annual Conference on Robot Learning, 2024. [6] T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. Kitani, C. Liu, and G. Shi, "Omnih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning," in arXiv, 2024. [7] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn, "Humanplus: Humanoid shadowing and imitation from humans," in arXiv, 2024.   
[8] T. Lin, Z.-H. Yin, H. Qi, P. Abbeel, and J. Malik, "Twisting lids off with two hands," arXiv:2403.02338, 2024. [9] Z. Yuan, T. Wei, S. Cheng, G. Zhang, Y. Chen, and H. Xu, "Learning to manipulate anywhere: A visual generalizable framework for reinforcement learning," arXiv preprint arXiv:2407.15815, 2024.   
[10] X. Cheng, J. Li, S. Yang, G. Yang, and X. Wang, "Open-television: Teleoperation with immersive active visual feedback," arXiv preprint arXiv:2407.01512, 2024.   
[11] Boston Dynamics, "Atlas," 2024, online. [Online]. Available: https://bostondynamics.com/atlas/   
[12 Tesl, "Optimus," 2024, online. [Online.Available: https://w. tesla.com/en_eu/AI   
[13] Figure, "01," 2024, online. [Online]. Available: https://www.figure.ai/   
[4 Unitree, "H1," 2024, online. [Online].Available: https:/www.unitree. com/h1   
[15] Fourier Intelligence, "Gr1," 2024, online. [Online]. Available: https://www.fourierintelligence.com/gr1   
[16] T. He, Z. Luo, W. Xiao, C. Zhang, K. Kitani, C. Liu, and G. Shi, Learnng human-to-humanoid real-im whole-body teleoperation," in arXiv, 2024.   
. S.Y.  . X  , Diffusion policy: Visuootor policy learning via action diffusion." arXiv preprint arXiv:2303.04137, 2023.   
[18 T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, "Learning fine-grained bimanual manipulation with low-cost hardware," arXiv preprint arXiv:2304.13705, 2023.   
[9] Z. Fu, T. Z. Zhao, and C. Finn, "Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation," in arXiv, 2024.   
[20] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, .Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu et al. Rt-: Robotics transformer for real-world control at scale," arXiv preprint arXiv:2212.06817, 2022.   
[21] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn et al., "Rt-2: Visionlanguage-action models transfer web knowledge to robotic control," arXiv preprint arXiv:2307.15818, 2023.   
[22] T. Lin, Y. Zhang, Q. Li, H. Qi, B. Yi, S. Levine, and J. Malik, "Learning visuotactile skills with two multifingered hands," arXiv preprint arXiv:2404.16823, 2024.   
[23] X. Gu, Y.-J. Wang, X. Zhu, C. Shi, Y. Guo, Y. Liu, and J. Chen, "Advancing humanoid locomotion: Mastering challenging terrains with denoising world model learning," arXiv preprint arXiv:2408.14472, 2024.   
[24] I. Radosavovic, B. Zhang, B. Shi, J. Rajasegaran, S. Kamat, T. Darrell K. Sreenath, and J. Malik, "Humanoid locomotion as next token prediction," arXiv preprint arXiv:2402.19469, 2024.   
[25] I. Radosavovic, T. Xiao, B. Zhang, T. Darrell, J. Malik, and K. Sreenath, "Real-world humanoid locomotion with reinforcement learning," Science Robotics, vol. 9, no. 89, p. eadi9579, 2024.   
[26] Z. Zhuang, S. Yao, and H. Zhao, "Humanoid parkour learning," arXiv preprint arXiv:2406.10759, 2024.   
[27] X. Cheng, Y. Ji, J. Chen, R. Yang, G. Yang, and X. Wang, "Expressive whole-body control for humanoid robots," arXiv preprint arXiv:2402.16796, 2024.   
[28] C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu, Dexcap: Scalable and portable mocap data collection system for dexterous manipulation," arXiv preprint arXiv:2403.07788, 2024.   
[29] M. Seo, S. Han, K. Sim, S. H. Bang, C. Gonzalez, L. Sentis, and Y. Zhu, "Deep imitation learning for humanoid loco-manipulation through human teleoperation," in IEEE-RAS International Conference on Humanoid Robots (Humanoids), 2023.   
[30] H. Etukuru, N. Naka, Z. Hu, J. Mehu, A. Edsinger, C. Paxton, S. Chintala, L. Pinto, and N. M. M. Shafiullah, "General policies for zero-shot deployment in new environments," arXiv, 2024.   
[31] Inspire Robots, "Dexterous hands," 2024, online. [Online]. Available: http://www.inspire-robots.store/collections/the-dexterous-hands   
[32] Intel RealSense, "Lidar camera 1515," 2024, online. [Online]. Available: https://www.intelrealsense.com/lidar-camera-1515/   
[33] "Depth camera d435," 2024, online. [Online]. Available: https://www.intelrealsense.com/depth-camera-d435/   
[34] C. Wang, H. Fang, H.-S. Fang, and C. Lu, "Rise: 3d perception maks real-world robot imitation simple and effective," arXiv preprint arXiv:2404.12281, 2024.   
[35] Apple, "Apple vision pro," 2024, online. [Online]. Available: https://www.apple.com/apple-vision-pro/   
[36] Y. Park and P. Agrawal, "Using apple vision pro to train and control robots," 2024, online. [Online]. Available: https: //github.com/Improbable-AI/VisionProTeleop   
[37] D. Rakita, B. Mutlu, and M. Gleicher, "Relaxedik: Real-time syntheis of accurate and feasible robot arm motion." in Robotics: Science and Systems, vol. 14. Pittsburgh, PA, 2018, pp. 2630.   
[38] J. Yang, Z. ang Cao, C. Deng, R. Antonova, S. Song, and J. Bohg, "Equibot: Sim(3)-equivariant diffusion policy for generalizable and data efficient learning," arXiv preprint arXiv:2407.01479, 2024.   
[39] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7   
[40] J. Song, C. Meng, and S. Ermon, "Denoising diffusion implicit models," arXiv preprint arXiv:2010.02502, 2020.   
[41] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, "R3m: A universal visual representation for robot manipulation," arXiv preprint arXiv:2203.12601, 2022.   
[42] N. Hansen, Z. Yuan, Y. Ze, T. Mu, A. Rajeswaran, H. Su, H. Xu, and X. Wang, "On pre-training for visuo-motor control: Revisiting a learning-from-scratch baseline," arXiv preprint arXiv:2212.05749, 2022.   
[43] S. Tian, B. Wulfe, K. Sargent, K. Liu, S. Zakharov, V. Guizilini, and J. Wu, "View-invariant policy learning via zero-shot novel view synthesis," arXiv preprint arXiv:2409.03685, 2024.   
[44] D. Wang, S. Hart, D. Surovik, T. Kelestemur, H. Huang, H. Zhao, M. Yeatman, J. Wang, R. Walters, and R. Platt, "Equivariant diffusion policy," arXiv preprint arXiv:2407.01812, 2024.