# Yume: An Interactive World Generation Model

Xiaofeng Mao1,2, Shaoheng $\mathbf { L i n ^ { 1 } }$ , Zhen $\mathbf { L i } ^ { 1 }$ , Chuanhao $\mathbf { L i } ^ { 1 }$ , Wenshuo $\mathbf { P e n g ^ { 1 } }$ , Tong $\mathbf { H e } ^ { 1 }$ , Jiangmiao Pang1, Mingmin $\mathbf { C h i ^ { 2 \dagger } }$ , Yu Qiao1, Kaipeng Zhang1,3†‡ 1Shanghai AI Laboratory 2Fudan University 3Shanghai Innovation Institute

8 Github:https://github.com/stdstu12/YUME Huggingface:https://huggingface.co/stdstu123/Yume-I2V-540P ProjectPage: https://stdstu12.github.io/YUME-Project Data: https://github.com/Lixspl1/sekai-codebase

# ABSTRACT

Yume aims to use images, text, or videos to create an interactive, realistic, and dynamic world, which allows exploration and control using peripheral devices or neural signals. In this report, we present a preview version of Yume, which creates a dynamic world from an input image and allows exploration of the world using keyboard actions. To achieve this high-fidelity and interactive video world generation, we introduce a well-designed framework, which consists of four main components, including camera motion quantization, video generation architecture, advanced sampler, and model acceleration. First, we quantize camera motions for stable training and user-friendly interaction using keyboard inputs. Then, we introduce the Masked Video Diffusion Transformer (MVDT) with a memory module for infinite video generation in an autoregressive manner. After that, training-free Anti-Artifact Mechanism (AAM) and Time Travel Sampling based on Stochastic Differential Equations (TTS-SDE) are introduced to the sampler for better visual quality and more precise control. Moreover, we investigate model acceleration by synergistic optimization of adversarial distillation and caching mechanisms. We use the high-quality world exploration dataset Seka i to train Yume, and it achieves remarkable results in diverse scenes and applications. All data, codebase, and model weights are available on https://github.com/stdstu12/YUME. Yume will update monthly to achieve its original goal.

We are looking for collaboration and self-motivated interns interested in interactive world generation. Contact: zhangkaipeng@pjlab.org.cn

![](images/1.jpg)  
Figure 1: We introduce Yume, a streaming interactive world generation model, which allows using continuous keyboard inputs to explore a dynamic world created by an input image.

# Contents

# 1 Introduction 3 2 Related Works 4

2.1 Video Diffusion Models . . . 4   
2.2 Camera Control in Video Generation 4   
2.3 Navigatable World Generation 5   
2.4 Mitigating Generation Artifacts 5   
2.5 Video Diffusion Acceleration . 6

3 Preliminaries 6

3.1 Rectified Flow. 6   
3.2 Wan Architecture . 7

# Data Processing 7

4.1 Dataset 7   
4.2 Camera Motion Quantization . 7

# 5 Method

# 5.1 Overview 8

5.2 Model Architecture 9   
5.2.1 Masked Video Diffusion Transformers 9   
5.2.2 Image-to-Video (I2V) and Video-to-Video (V2V) Generation 10   
5.2.3 Long Video Generatetion .. 10   
5.3 Sampler Design . . 12   
55.3.1 Training-Free Anti-Artifact Mechanism 12   
5.3.2 Time Travel Sampling based on SDE (TTS-SDE) for Enhanced Video Generation . . 14

# 5.4 Camera Motion Control 14

5.5 Application 15   
55.5.1 World Generalization 15   
5.5.2 orl  Editing 15   
i.6 Acceleration 15   
5.6.1 Adversarial Distillation for Accelerated Diffusion Sampling 16   
5.6.2 Cache-Accelerating 16

# 6 Experiment 17

# 1 Experimental Settings

6.1.1 Training Details 17   
6.1.2 Evaluation Dataset 17   
6.1.3 Evaluation Details. 17

Qualitative Results

6.2.1 Image-to-Video Generation 18   
6.2.2 Validation of Long-video Generation Performance 18   
.3 Ablation study . . 19   
6.3.1 Verification of TTS-SDE Effectiveness 19   
6.3.2 Validating the effect of model distillation 19

6.4 Visualization Results . . . 19

# 7 Conclusion 20

A Ablation Study on Image to Video 26   
B Quantized Camera Motion of Text 26   
C Action Distribution Statistics 26   
D Core Algorithm Implementation of Gaussian Blur Kernel 26   
E The Joint Optimization of Adversarial Distillation and Caching 29

# Past or future, just let them be — Link Click

Yume aims to create an interactive, realistic, and dynamic world through the input of text, images, or videos. And we can explore and control the world using peripheral devices or neural signals Zhou et al. (2025). A practical application is to enter a world through a photo, like time travel, regardless of its location, scenes, and time, while you can interact with everything and change everything following your mind. We can compensate for our real-world regrets or realize our wishes through the virtual world created by Yume. In this paper, we present a preview version of Yume, which is an interactive world generation model that allows the use of keyboard inputs to explore a dynamic world created by an input image. We have released all data, code, and model weights on ht tps : / /github . com/stdstu12 /YuME. Yume will update monthly to achieve its original goal.

Video diffusion models Singer et al. (2023); Bar-Tal et al. (2024); Blattmann et al. (2023a); Ma et al. (2024); Nagrath et al. (2024); Singer et al. (2023); The Step-Video-T2V Team (2025); Chen et al. (2025), which have shown remarkable capabilities in synthesizing high-fidelity and temporally coherent visual content Ho et al. (2022); Blattmann et al. (2023a), present a promising avenue to realize such a sophisticated interactive world generation task. Recently, the automated generation of vast, interactive, and persistent virtual worlds Zhang et al. (2025); Agarwal et al. (2025) has advanced rapidly, driven by progress in generative models and the growing demand for immersive experiences in domains such as world simulation, interactive entertainment Li et al. (2025a), and virtual embodiment Xiao et al. (2025); Zhang et al. (2025).

The motion of the camera is an important control signal for interactive video generation of world exploration. However, existing video diffusion methods face significant challenges when applied to producing interactive (continuous camera motion controlled) and realistic videos, especially in urban scenarios. First, existing approaches Zhang et al. (2025); Xiao et al. (2025) focus mainly on synthetic or static scenarios, in terms of methods and data. The domain gap between them and the real world limits their generalizability. In addition, these methods are mostly based on absolute camera motions, which require precise annotation and extra-learner modules that increase the training and architecture design difficulty. In addition, urban environments present unique complexities characterized by diverse architectural styles, dynamic objects, and intricate details. Existing approaches demonstrate limited adaptability to such complexity and struggle to maintain consistent realism across varied scenes. Common visual artifacts, including flickering, unnatural textures, and geometric distortions, significantly degrade the perceptual quality and disrupt immersive experiences.

To address these limitations, we introduce Yume for the generation of autoregressive interactive video through an input image with discrete keyboard control. Yume is designed to offer more intuitive and stable camera control via keyboard input while improving the visual quality and realism of complex scene generation. Implements systematic optimizations on four key dimensions: camera motion control, model architecture, sampler, and model acceleration.

First, we proposed Quantized Camera Motion (QCM) control module. Yume quantizes camera trajectories into translational movements (forward, backward, left, right) and rotational motions (turn right, turn left, tilt up, tilt down), which can be flexibly combined and transferred by keyboard input. The QCM is produced by changes in relative camera poses during training and naturally embeds temporal context and spatial relationships into the control signal. QCM are parsed into textual conditions without introducing new learnable modules for pretrained I2V foundation models.

Second, as noted in Zhang et al. (2025), text-based control frequently leads to unnatural outputs. To address this limitation, we investigated an alternative approach using Masked Video Diffusion Transformers (MVDT), inspired by prior work Gao et al. (2023). Moreover, we achieved interactive video generation with theoretically infinite duration via a chunk-based autoregressive generation framework with a modified FramePack memory module Zhang & Agrawala (2025).

Third, to improve the visual quality of sophisticated real-world scenes, we introduce a training-free Anti-Artifact Mechanism (AAM) module. Specifically, it refines the high-frequency components of the latent representation at each diffusion step. This targeted refinement improves fine-grained details, smooths out inconsistencies, and substantially reduces visual artifacts, leading to an overall improvement in visual quality without requiring any additional model training or specialized datasets.

However, it does not perform well on autogressive long video generation due to a lack of V2V foundation models, and we will add it to our long video generation in the next version of Yume.

Furthermore, we propose a novel sampling method based on the Time-Travel Stochastic Differential Equation (TTS-SDE) framework. Inspired by DDNM Wang et al. (2022) and OSV Mao et al. (2025), our approach leverages information from later denoising stages to guide the earlier denoising process, while incorporating stochastic differential equations to enhance sampling randomness, thereby improving textual controllability.

Finally, we investigate multiple acceleration techniques for diffusion-based video generation and jointly optimize step distillation and cache mechanisms. It significantly enhances sampling efficiency without compromising visual fidelity or temporal coherence.

Yume offers a significant step towards generating high-quality, dynamic, and interactive infinite video generation, particularly for complex real-world scenes exploration.

# 2 RELATED WORKS

# 2.1 Video Diffusion Models

Diffusion models Sohl-Dickstein et al. (2015); Ho et al. (2020), initially transformative for image synthesis, have rapidly become the cornerstone of video generation. The application of these models in a compressed latent space, pioneered by Latent Diffusion Models (LDMs), was a key enabler for efficient video synthesis, leading to works like Video LDM (Align your Latents) Blattmann et al. (2023b) which extended this paradigm to achieve high-resolution video generation by integrating temporal awareness.

Early breakthroughs in text-to-video generation, such as Imagen Video Ho et al. (2022) and Make-AVideo Singer et al. (2023), quickly demonstrated the potential for creating dynamic scenes from textual descriptions. The field has since witnessed significant advancements in model scale, architectural design, and training strategies. Large-scale models like Google's Lumiere Bar-Tal et al. (2024), featuring a Space-Time U-Net, and OpenAI's Sora Brooks et al. (2024), which employs a diffusion transformer architecture, have significantly pushed the boundaries of generating long, coherent, and high-fidelity video content. Concurrently, the open-source ecosystem has flourished, with contributions such as Stable Video Diffusion Blattmann et al. (2023a) offering robust and accessible baselines. Recent notable open-source efforts include HunyuanVideo Ma et al. (2024), which provides a systematic framework for very large video models, MoChi-Diffusion-XL Nagrath et al. (2024), focusing on efficient high-resolution video synthesis, Step-Video-T2V The Step-Video-T2V Team (2025), a large-parameter foundation model, and SkyReels-V2 Chen et al. (2025), which aims for generating extended film-like content. Based on these advanced foundation models, we built Yume for realistic dynamic world exploration.

# 2.2 CameRa Control In Video GenEratioN

Precise and flexible camera control is crucial for creating compelling and customizable video content, allowing for the emulation of cinematic effects and a more interactive user experience. Early video generation models often lacked explicit mechanisms for detailed camera path guidance, with camera movements emerging implicitly from textual prompts or initial frames. Subsequent research has focused on more direct methods for specifying and integrating camera motion into the generation process.

A significant body of work has emerged to explicitly condition video diffusion models on camera parameters. For instance, MotionCtrl Wang et al. (2023b) introduced a unified controller to manage both camera and object motion, where camera motion is determined by a sequence of camera poses that are temporally fused into the video generation model. Following this, Direct-a-Video Yang et al. (2024) enabled decoupled control of camera pan/zoom and object motion, utilizing temporal cross-attention layers for interpreting quantitative camera movement parameters, trained via a selfsupervised approach. CameraCtrl He et al. (2024) proposed a plug-and-play module to integrate accurate camera pose control into existing video diffusion models, exploring effective trajectory parameterizations. Further advancing fine-grained control, CameraCtrl II Zhang et al. (2024) focused on dynamic scene exploration by allowing iterative specification of camera trajectories for generating coherent, extended video sequences. More recently, training-free approaches have also been investigated; for example, CamTrol (Training-free Camera Control) Geng et al. (2024) leverages 3D point cloud modeling from a single image and manipulates latent noise priors to guide camera perspective without requiring model fine-tuning or camera-annotated datasets. These methods typically rely on explicit sequences of absolute camera pose matrices or derived parameters to define the camera trajectory, aiming to improve the stability and precision of the generated viewpoint transformations. In contrast to existing approaches that rely on explicit camera trajectories (requiring fine-grained motion parameter adjustments), we innovatively propose a Quantized Camera Motion (QCM) mechanism, which achieves intuitive keyboard-based control by discretizing the camera pose space.

# 2.3 NAvigataBle WorLD GENEration

The generation of expansive, interactive, and temporally coherent virtual worlds is a long-standing ambition in artificial intelligence, with significant implications for gaming, simulation, and robotics. Early efforts in this domain often involved learning world models Ha & Schmidhuber (2018) that capture environment dynamics to enable agents to plan or learn behaviors within a learned latent space, as demonstrated by subsequent lines of work like the Dreamer series Hafner et al. (2023). These models emphasized understanding and predicting future states based on actions, laying the groundwork for more explicit world generation.

More recent approaches have focused directly on generating interactive environments and controllable long-duration video sequences. For instance, Genie Bruce et al. (2024) introduced a foundation model capable of generating an endless variety of action-controllable 2D worlds from image prompts, trained on unlabeled internet videos. Efforts to generate navigable driving scenes, such as GAIÅ-1 Wayve Technologies (2023) by Wayve, have demonstrated the potential for creating realistic, controllable driving experiences from text and action inputs. Similarly, projects like SIMA Google DeepMind (2024) aim to develop generalist AI agents that can understand and interact within diverse 3D virtual settings based on natural language instructions. The challenge of maintaining long-term consistency in generated worlds, especially for extended exploration, is actively being addressed. For example, StreamingT2V Henschel et al. (2024) proposed methods for consistent and extendable long video generation. Very recent works are pushing the envelope on creating interactive foundation models for worlds and ensuring their coherence. Matrix-Game Zhang et al. (2025) presents an interactive world foundation model aimed at controllable game world generation, emphasizing fine-grained action control. Concurrently, WORLDMEM Xiao et al. (2025) introduces a framework to enhance long-term consistency in world simulation by employing a memory bank and attention mechanisms to accurately reconstruct previously observed scenes, even with significant viewpoint or temporal changes, and to model dynamic world evolution. These advancements are critical for generating the kind of scene-level, interactive, and navigable experiences that are the focus of current research. However, these methods primarily rely on game-based scenarios and actions (e.g., Matrix-Game Zhang et al. (2025) utilizes the Minecraft dataset), which remain relatively simplistic compared to real-world environments. Furthermore, collecting motion data from real-world video datasets is significantly more challenging than collecting it in game settings.

# 2.4 MITIGATING GENERATIoN ARTifACTS

Despite the remarkable progress of diffusion models in generating realistic images and videos, the synthesized results can still suffer from various artifacts, such as unnatural textures, flickering, or semantic inconsistencies, particularly in complex scenes or long video sequences. Efforts to mitigate these issues can be broadly categorized into training-based and training-free approaches.

Training-based methods often involve architectural modifications or specialized fine-tuning strategies. For instance, some works focus on improving the autoencoder stage in latent diffusion models to better capture high-frequency details and reduce reconstruction errors that can propagate into the generation process, as explored in models like LTX-Video HaCohen et al. (2025), which tasks the VAE decoder with a final denoising step. Others employ parameter-efficient fine-tuning (PEFT) techniques with objectives specifically designed to enhance visual quality and temporal consistency, as seen in frameworks like DAPE Xia et al. (2025) for video editing. Diffusion models themselves have also been adapted as post-processing tools to enhance the quality of already compressed or generated videos that may contain artifacts Liu & Wang (2023).

While training-free methods typically operate during the inference or sampling stage, which often involves manipulations of the latent space, guidance of the denoising process, or modifications to sampling strategies. For video generation, FreqPrior Yuan et al. (2025) introduces a novel noise initialization strategy by refining noise in the frequency domain to improve detail and motion dynamics. Enhance-A-Video Luo et al. (2025) proposes a training-free module to enhance the coherence and visual quality of videos from DiT-based models by adjusting temporal attention distributions during sampling. These methods aim to improve perceptual quality and reduce artifacts without altering the underlying model weights, offering flexible solutions for enhancing generated content. The exploration of latent space refinement, especially concerning frequency components and detail enhancement during the denoising process, remains an active area of research for improving the visual fidelity of diffusion-based generation.

# 2.5 VIdeo Diffusion Acceleration

Numerous distillation methodologies Wang et al. (2023a); Song et al. (2023); Wang et al. (2024b;a; 2025; 2024c); Mao et al. (2024a); Kim et al. (2023) have been proposed to reduce computational costs in video generation. For instance, some approaches implement joint consistency distillation and adversarial training Wang et al. (2025), accelerating diffusion models through phased consistency models integrated with GANs. Concurrently, Mao et al. Mao et al. (2024a) enhanced the discriminator architecture for adversarial distillation. Given the feature similarity across timesteps in DiT architectures, cache mechanisms have been developed for diffusion frameworks, such as ToCa Zou et al. (2024)'s dynamic feature storage guided by token sensitivity and error propagation analysis, which accelerates diffusion transformers through adaptive caching strategies and layer-specific retention techniques. Similarly, AdaCache Kahatapitiya et al. (2024) improves diffusion transformer inference without retraining via dynamically adjusted caching policies and motion-aware resource allocation during denoising steps, and TeaCache Liu et al. (2024) accelerate sampling via estimating fluctuating differences among model outputs across timesteps. We introduced a co-optimization strategy that integrates step distillation with cache acceleration to further boost sampling efficiency of interactive video generation.

# 3 PRELIMINARIES

# 3.1 REctifiED FLOW.

Rectified Flow Liu et al. (2022) is a technique that facilitates ordinary differential equation (ODE)- based training by minimizing the transport cost between marginal distributions $\pi _ { 0 }$ and $\pi _ { 1 }$ ,i.e., ${ \mathbb E } [ c ( x _ { 1 } - x _ { 0 } ) ]$ . Here, $x _ { 1 } = \mathrm { L a w } ( \pi _ { 1 } )$ , $x _ { 0 } = \mathrm { L a w } ( \pi _ { 0 } )$ , and $c : \mathbf { \bar { \mathbb { R } } ^ { d } } \longrightarrow \mathbb { R }$ denote a cost function. Given the computational complexity of Optimal Transport (OT) Villani et al. (2009), Rectified Flow provides a simple yet effective approach to generate a new coupling from a preexisting one. This new coupling can be optimized using Stochastic Gradient Descent (SGD), an optimization method extensively employed in deep learning:

$$
\theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \mathbb { E } _ { t \sim U [ 0 , 1 ] } \mathbb { E } _ { x _ { 0 } , x _ { 1 } \sim \pi _ { 0 } , \pi _ { 1 } } \left[ \mathbf { M S E } \left( x _ { 1 } - x _ { 0 } , v _ { \theta } ( x _ { t } , t ) \right) \right] ,
$$

where the term MSE denotes Mean Squared Error. The parameter $t$ is in the range of [0, 1], we select $x _ { t } = t x _ { 1 } + ( 1 - t ) x _ { 0 }$ to ensure that $\forall v _ { \theta } ( x _ { t } , t )$ matches the identical target velocity $x _ { 1 } - x _ { 0 }$ Upon completion of the training phase, sampling can be performed through a definite integral, specifically $\begin{array} { r } { x _ { 1 } = x _ { 0 } + \int _ { 0 } ^ { 1 } v _ { \theta } ( x _ { \tau } , \tau ) \mathrm { d } \tau } \end{array}$ . In practical scenarios, the aforementioned continuous system is typically approximated in discrete time using the Euler method (or its variants): $x _ { t _ { n - 1 } } =$ $x _ { t _ { n } } + ( t _ { n - 1 } - t _ { n } ) v _ { \theta } ( x _ { t _ { n } } , t _ { n } )$ , where $t _ { 0 } < t _ { 1 } < . . . < t _ { N - 1 }$ is a set of predefined time steps.

In the video domain, $x _ { 0 }$ represents the pixel space of the videos, while $c$ denotes the conditioning inputs (such as text and image conditions for controlled generation). Following recent advancements in Rectified Flow that employ VAEs Rombach et al. (2022) for latent space compression during both training and inference to reduce computational costs, we formulate the training objective as:

$$
\theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \mathbb { E } _ { t \sim U [ 0 , 1 ] } \mathbb { E } _ { z _ { 0 } , z _ { 1 } \sim \pi _ { 0 } , \pi _ { 1 } } \left[ | | z _ { 1 } - z _ { 0 } - v _ { \theta } ( z _ { t } , c , t ) | | _ { 2 } ^ { 2 } \right] ,
$$

where $z = { \mathrm { V A E . E n c o d e r } } ( x )$ and $x = \operatorname { V A E } \operatorname { D e c o d e r } ( z )$ perform latent space projection and reconstruction respectively. During inference, the sampling process follows: $z _ { t _ { n - 1 } } = z _ { t _ { n } } + ( t _ { n - 1 } -$ $t _ { n } ) v _ { \theta } ( z _ { t _ { n } } , c , \mathbf { \bar { \it t } } _ { n } )$ .

# 3.2 WAN ARCHITECTURE

The foundational architecture employs an identical design to Wan Wan et al. (2025), utilizing its spatio-temporal VAE encoder and denoising DiT model. The VAE encoder compresses input video sequences into latent representations of dimensionality $[ 1 + \mathrm { T } / 4$ , H/8, W/8] with expanded channel depth ${ \mathrm { C } } { = } 1 6$ T denoising DiT backbone incorporates: 1) Patchify module utilizing 3D convolution (kernel=(1,2,2) to downsample spatial resolution while expanding channels into transformer tokens; 2) CLIP Radford et al. (2021) for image encoder and umT5 Chung et al. (2023) for text encoder; 3) Transformer blocks that concurrently process modality-specific features, where cross-ttention mechanisms fuse the video tokens (queries) with image/text embeddings (keys/values). This configuration maintains spatio-temporal coherence while ensuring computational efficiency throughout the encoding-to-denosing pipeline.

# 4 DatA PRoCESsing

# 4.1 DATASET

We use the Sekai-Real-HQ, a subset of Sekai Li et al. (2025b), as the training dataset. It consists of large-scale walking video clips with corresponding high-quality annotations of camera trajectory and semantic labels. In this section, we briely introduce the dataset and more details could be found in the paper of Sekai.

Video Collection We manually collect high-quality video URLs from popular YouTubers and extend them by searching additional videos using related keywords (e.g., walk, drone, HDR, and 4K). In total, we collect 10471 hours of walking videos (with stereo audio) and 628 hours of drone (FPV or UAV) videos. All videos were released over the past three years, with a 30-minute to 12-hour duration. They are at least 1080P with 30 to 60 FPS. We download the 1080P version with the highest Mbps for further video processing and annotation. Due to network issues and some videos are broken, there are 8409 hours of walking videos and 214 hours of drone videos after downloading.

Video Preprocessing For YouTube videos, we trim two minutes from the start and end of each original video to remove the opening and ending. Then we do the five preprocessing steps, including shot boundary detection, clip extraction and transcoding, luminance filtering, quality filtering, camera trajectory filtering, to obtain 6620 hours of video clips as Sekai-Real.

Video Annotation We annotate video clips using multiple tools, large vision-language models, and meta inforation from YouTube. The annotations including location, multiple categories, caption, and camera trajectories.

Video Sampling We sample the best-of-the-best video clips considering content diversity, location diversity, category diversity, and camera trajectory diversity. Finally, we obtain 400 hours of video clips as Sekai-Real-HQ.

# 4.2 CAMERA MOTIoN QUanTiZatION

Though low-quality camera trajectories from Sekai-Real-HQ are already fltered, the trajectories estimated by MegaSaM Li et al. (2025c) are inevitably imprecise enough. In addition, raw camera trajectories with large variances are difficult to follow and require extra learnable modules. Consequently, we developed a trajectory quantization method to transfer continuous camera poses to discrete actions. This approach inherently facilitates filtering excessively jittery trajectories, balances trajectory distributions, and mitigates model training difficulty.

Specifically, existing approaches for camera control in video generation often rely on providing a dense sequence of per-frame camera-to-world (c2w) transformation matrices Wang et al. (2023b); He et al. (2024). While offering explicit control, this representation can be overly granular, potentially leading to less stable or unintuitive camera motion, and may not effectively capture the inherent temporal coherence of continuous camera movements.

To address these limitations and foster more robust and intuitive navigation, Yume introduces a quantized camera motion representation. Our core idea is to define a discrete set of predefined, holistic camera motion, $\mathbb { A } _ { \mathrm { s e t } }$ Each motion $A _ { j } \in \mathbb { A } _ { \mathrm { s e t } }$ (e.g., "move-forward", "tilts up", "tilts down". See Supplementary Materials for these motion.) corresponds to a canonical relative transformation matrix, $T _ { \mathrm { c a n o n i c a l } , j }$   
matrices, we process the input sequence of ${ \mathfrak { c } } 2 { \mathfrak { w } }$ transformation matrices. For each segment of camera movement (defined by a pair of consecutive transformation matrices, potentially after downsampling the trajectory), we construct the reference coordinate system using the transformation matrix at the current segment, and calculate the relative transformation matrix $T _ { \mathrm { r e l , a c t u a l } }$ of the next segment. We then select the motion $A _ { j } ^ { * }$ from our predefined set $\mathbb { A } _ { \mathrm { s e t } }$ whose canonical transformation matrix $T _ { \mathrm { c a n o n i c a l } , j }$ is closest to $T _ { \mathrm { r e l , a c t u a l } }$ , as outlined in Algorithm 1. This matching process effectively quantizes the continuous camera trajectory into a sequence of semantically meaningful motion, inherently integrating temporal context from the relative pose changes.

In Algorithm 1, the Distance $( T _ { 1 } , T _ { 2 } )$ function measures the dissimilarity between two transformation matrices, which can be a weighted combination of differences in their translational and rotational components.

As shown in Figure 2, the selected discrete motion $A ^ { * }$ are assigned to textual descriptions through a predefined dictionary (detailed in the Supplementary Materials). By injecting the action descriptions into the text condition, this approach achieves camera pose-controlled video generation without introducing additional learnable parameters to the pre-trained I2V model, leading to fast model converged and stable and precise camera control.

By quantifying camera motion for videos from Sekai-Real-HQ and extracting segments with consis. tent camera movements, we selected clips longer than 33 frames, ultimately obtaining 139019 video clips. The distribution of camera motions is provided in the supplementary materials.

# Algorithm 1 Camera Motion Quantization

Require: Sequence of camera-to-world matrices $\mathcal { C } = \{ C _ { 0 } , C _ { 1 } , \ldots , C _ { N - 1 } \}$ , Predefined set of $K$ camera motion $\mathbb { A } _ { \mathrm { s e t } } = \{ A ^ { ( 1 ) } , \ldots , A ^ { ( K ) } \}$ , Corresponding canonical relative $S E ( 3 )$ transformations $\{ T _ { \mathrm { c a n o n i c a l } } ^ { ( 1 ) } , \ldots , \overline { { T } } _ { \mathrm { c a n o n i c a l } } ^ { ( K ) } \}$ cannical.   
Ensure: Sequence of selected camera motion $\mathcal { A } ^ { * } = \{ A _ { 0 } ^ { * } , \ldots , A _ { M - 1 } ^ { * } \}$ (where $M$ depends on processing stride).   
1: Initialize $A ^ { * }  \emptyset$   
2: for each relevant pair $( C _ { \mathrm { c u r r } } , C _ { \mathrm { n e x t } } )$ from $\mathcal { C }$ do   
3: $\begin{array} { r l } & { T _ { \mathrm { r e l , a c t u a l } }  C _ { \mathrm { c u r r } } ^ { - 1 } \cdot C _ { \mathrm { n e x t } } } \\ & { A _ { \mathrm { c u r r e n t } } ^ { * }  \arg \underset { A ^ { ( j ) } \in \mathbb { A } _ { \mathrm { s e t } } } { \mathrm { m i n } } \mathrm { D i s t a n c e } ( T _ { \mathrm { r e l , a c t u a l } } , T _ { \mathrm { c a n o n i c a l } } ^ { ( j ) } ) } \end{array}$   
4:   
5: Append $A _ { \mathrm { c u r r e n t } } ^ { * }$ to $\ b { A } ^ { * }$   
6: end for   
7: return $\ b { A } ^ { * }$

# 5 METHOD

# 5.1 OVERVIEW

In this section, we provide a comprehensive overview of the Yume. The architecture builds upon Wan Wan et al. (2025)'s network design but introduces masked video diffusion transformers and employs a ReFlow-based training methodology. To enable long-video generation, we concatenate downsampled historical video clips (using Patchify) with generated segments during training and feed them into the DiT denoising model—a strategy replicated during inference by similarly downsampling and stitching multiple video segments. Moreover, we detail various sampler designs, including AAM and TTS-SDE, which yield distinct sampling effects. Then we introduced our camera motion control and the motion speed control. Finally, we discuss practical applications of Yume, including world generalization and world editing.

![](images/2.jpg)  
Figure 2: Four core components of Yume: camera motion quantization, model architecture, long video training, and generation. We also make advanced sampler, please see Section 5.3.

# 5.2 MODEL ARCHITECTURE

# 5.2.1 Masked Video Diffusion Transformers

We employ the video diffusion model described in Section 3.2 and introduce Masked Video Diffusion Transformers (MVDT) for better visual quality.

Existing video diffusion models often overlook the critical role of masked representation learning, leading to artifacts and structural inconsistencies in cross-frame generation. Drawing inspiration from proven masking strategies in Gao et al. (2023); Mao et al. (2024b), we introduce MVDT to significantly enhance video generation quality. As shown in Figure 2, the architecture employs an asymmetric network to process selectively masked input features through three core components: encoder, side-interpolator, and decoder.

Masking Operation. The feature transformation begins with stochastic masking of the input feature $z _ { t _ { n } } \in \bar { \mathbb { R } ^ { N \times \bar { d } } }$ (where $d$ denotes the channel dimension and $_ \mathrm { N }$ represents the token count), derived from noisy video token embeddings. Applying random masking ratio $\rho$ yields a reduced set of active features $\bar { z } _ { t _ { n } , u } \in \mathbb { R } ^ { d \times \hat { N } } ( \hat { N } = ( 1 - \rho ) \bar { N } )$ accopanied by a positional binary sk $M A S K \in \mathbb { R } ^ { N }$ The masking ratio $\rho$ serves as a hyperparameter. We set $\rho = 0 . 3$ in this paper.

This selective processing concentrates computational resources on visible tokens $z _ { t _ { n } , u }$ while main.   
taining representational accuracy and substantially reducing memory/computational overhead.

Encoder. This stage utilizes a streamlined architecture that exclusively processes preserved tokens $z _ { t _ { n } , u }$ , mapping them to compact latent representations. By bypassing masked regions, it achieves a computational load reduction compared to conventional full-feature encoders.

Side-Interpolator. Inspired by prior work Gao et al. (2023), this component adopts an innovative dual-path architecture. During training, it combines learnable latent tokens $z _ { l }$ with encoded features through $z _ { I } = S A ( \mathrm { C o n c a t } [ z _ { l } , E n c o d e r ( z _ { t _ { n } , u } ) ] )$ , dynamically predicting masked content via selfattention mechanisms. The gated fusion operation $\hat { z } _ { t _ { n } , u } = ( 1 - M A S K ) \odot z _ { t _ { n } , u } + M A S K \odot z _ { I }$ seamlessly integrates the original and synthesized features while preserving temporal coherence between video sequences.

Decoder. Finally, we process the interpolated features $\hat { z } _ { t _ { n } , u }$ using the remaining DiT-blocks.

During training, it optimizes MVDT through end-to-end gradient propagation, while inference directly processes complete feature sets without interpolation. During model training, the diffusion module receives both $z _ { t _ { n } }$ and $z _ { t _ { n } , u }$ .This strategy prevents the system from concentrating exclusively on masked-area regeneration while compromising essential diffusion learning.

![](images/3.jpg)  
Figure 3: Long-form video generation method.

5.2.2 IMAGE-TO-VIDEO (I2V) AND VIDEO-TO-VIDEO (V2V) GENErATION

For image-to-video, we adopt the methodology established in Wan. In image-to-video synthesis, a conditioning image initializes video generation. Temporally zero-padded frames $x _ { c }$ undergo compression via the Wan-VAE encoder to yield latent representations $\boldsymbol { z } _ { c } ^ { \textbf { i } } \in \mathbb { R } ^ { C \times t \times h \times w }$ .

A binary mask $M _ { c } \in \{ 0 , 1 \} ^ { 1 \times T \times h \times w }$ (with 1 denoting preserved regions and 0indicating generated regions) is reshaped to dimensions $s \times t \times h \times w$ (s denoting the Wan-VAE temporal stride). Channelwise concatenation of $[ z _ { t } , z _ { c } , M ]$ feeds into Wan's DiT backbone. An additional zero-initialized projection layer accommodates the increased channel dimensionality inherent to image-to-video tasks $2 c + s$ versus $c$ for text-to-video). For video-to-video generation, our approach diverges from the image-to-video methodology by extracting exclusively the final frame of the video condition as the image input for CLIP encoding. We concurrently adjust the masking tensor $M _ { c }$ to ensure precise alignment between the conditional video sequence and generated output. This video-to-video synthesis lacks inherent memory retention capabilities.

# 5.2.3 Long Video GenEratetION

Based on meticulous experimentation, we observe that autoregressive methods for long video generation frequently exhibit inter-frame discontinuity and lack temporal coherence. To address these limitations, we introduce a long-form video generation methodology, adopting a compression design analogous to FramePack Zhang & Agrawala (2025) (architectural details illustrated in Figure 3). FramePack employs contextual compression of historically generated frames via downsampling of pretrained Patchify parameters. The baseline Patchify configuration, specified by hyperparameters (2, 4, 4), denotes sequential compression ratios of $2 \times$ temporally, $4 \times$ spatially in height, and $4 \times$ spatially in width. Higher compression ratios are applied as generated video length increases.

For conditional input to the diffusion model, our established compression scheme operates as follows:

$$
\begin{array} { r l r } & { \operatorname { F r a m e } t - 1 : } & { ( 1 , 2 , 2 ) } \\ & { \operatorname { F r a m e s } t - 2 \tan t - 5 : } & { ( 1 , 4 , 4 ) } \\ & { \operatorname { F r a m e s } t - 6 \tan t - 2 2 : } & { ( 1 , 8 , 8 ) } \\ & { \vdots } & \\ & { \vdots } & \end{array}
$$

.. and so forth, including scenarios with spatiotemporal compression.

More high-resolution tokens generally lead to better performance. We finalized:

$$
\begin{array} { r l r } & { { \mathrm { F r a m e s ~ } } t - 1 { \mathrm { ~ t o ~ } } t - 2 : } & { ( 1 , 2 , 2 ) } \\ & { { \mathrm { F r a m e s ~ } } t - 3 { \mathrm { ~ t o ~ } } t - 6 : } & { ( 1 , 4 , 4 ) } \\ & { { \mathrm { F r a m e s ~ } } t - 7 { \mathrm { ~ t o ~ } } t - 2 3 : } & { ( 1 , 8 , 8 ) } \\ & { \vdots } & { \vdots } \\ & { { \mathrm { I n i t i a l ~ f r a m e } } : } & { ( 1 , 2 , 2 ) } \end{array}
$$

This configuration optimally balances temporal and spatial compression while preserving highresolution representation for recent frames and retaining critical features from initial input images.

We implement a stochastic sampling strategy for the training where, with 0.3 probability, we select historical frames from 33-400 frame videos to predict subsequent 33-frame sequences, and with 0.7 probability, we utilize 400-800 frame videos as historical context for 33-frame predictions. This sampling distribution optimizes the frame count allocation during training.

In ourplngehogy letting $\boldsymbol { z } _ { t _ { n } } \in \mathbb { R } ^ { C \times F \times H \times W }$ step $t _ { n }$ , where $C$ indexes channels, $F$ frames, and $H \times W$ spatial dimensions. Given conditioning inputs $c$ (text embeddings), historical frames $\mathbf { I } _ { \mathrm { i n p u t } }$ , noise $z _ { \mathrm { n o i s e } }$ , and diffusion model $f _ { \theta }$ , our sampling process implements:

$$
\begin{array} { r l } & { z _ { t _ { n - 1 } } = z _ { \mathrm { i n } } + ( t _ { i - 1 } - t _ { i } ) \cdot v _ { \theta } \big ( \hat { z } _ { \mathrm { i n } } , c , t _ { n } \big ) } \\ & { \quad \hat { z } _ { \mathrm { i n } } = ( ( 1 - t _ { n } ) { \bf I } _ { \mathrm { i n } } + t _ { n } z _ { \mathrm { n o i s e } } \big ) \oplus z _ { i n } } \end{array}
$$

where $\oplus$ is frame-wise concatenation.

Our method supports both video-to-video (V2V) and image-to-video (I2V) generation. However, during V2V synthesis, we observe that when the conditioning video contains prolonged homogeneous motion, the model becomes overly reliant on the motion characteristics of the conditioning video (resulting in identical motion patterns between source and generated sequences), weakening the motion following capability. To address this conditional video dependency issue, we implement a training strategy where $30 \%$ of iterations use static image conditions, specifically, by temporally tiling individual images 16 times to construct static video sequences, thereby improving the model's ability to generate diverse motion patterns.

# Algorithm 2 Anti-Artifact by High Frequency Refinement

Require: Initial latent estimate $z _ { \mathrm { o r i g } }$ (from a full standard denoising pass), Diffusion model $v _ { \theta }$ , condition $c$ b number of inference steps $N$ , refinement steps $K _ { r e f i n e } < N$ , Timesteps $\{ t _ { i } \} _ { i = 0 } ^ { N - 1 }$ from $T$ down to , Low-pass filter operator $B$ (e.g., Gaussian blur). 1: $z _ { t _ { N - 1 } }$ is initialized during the first denoising step.   
2: for $i = N - 1$ down to 0 do   
3: $t _ { i } \gets$ current timestep from schedule   
4: $z _ { \mathrm { i n } } ^ { \prime }  z _ { t _ { i } } ^ { \prime }$ b   
5: if $\dot { \iota } \geq N ^ { \bullet } - K _ { r e f i n e }$ then   
6: $\begin{array} { r l } & { z _ { \mathrm { o r i g } , t _ { \mathrm { i } } } \gets ( 1 - t _ { i } ) * z _ { \mathrm { o r i g } } + t _ { \mathrm { i } } * z _ { t _ { N - 1 } } } \\ & { z _ { \mathrm { l o w } . \mathrm { f r o m . o r i g } } \gets B ( z _ { \mathrm { o r i g } , t _ { i } } ) } \\ & { z _ { \mathrm { h i g h . c u r r e n t } } ^ { \prime } \gets z _ { \mathrm { i n } } ^ { \prime } - B ( z _ { \mathrm { i n } } ^ { \prime } ) } \\ & { z _ { \mathrm { i n } } ^ { \prime } \gets z _ { \mathrm { l o w . f r o m . o r i g } } + z _ { \mathrm { h i g h . c u r r e n t } } ^ { \prime } } \end{array}$   
7:   
8:   
9:   
10: end if   
11: $z _ { t _ { i - 1 } } ^ { \prime } \gets z _ { \mathrm { i n } } ^ { \prime } + ( t _ { i - 1 } - t _ { i } ) * v _ { \theta } ( z _ { \mathrm { i n } } ^ { \prime } , c , t _ { i } )$   
12: end for   
13: return $z _ { t _ { 0 } } ^ { \prime }$

This section details our advanced sampler, which incorporates two key innovations to enhance the quality of the generated videos. We first introduce a Training-Free Anti-Artifact Mechanism (AAM) to eliminate visual artifacts and then present Time Travel Sampling based on SDE (TTS-SDE) to improve video sharpness and textual controllability.

# 5.3.1 TrainIng-FrEE AntI-ARtifAcT MEcHaNiSm

While diffusion models excel at generating diverse content, complex scenes such as urban cityscapes can often exhibit visual artifacts, such as blurred details, unnatural textures, or fickering which detract from realism. To address this without requiring additional training or model modification, Yume incorporates a novel Training-Free Anti-Artifact Mechanism, AAM. The core idea is to enhance high-frequency details and overall visual quality by strategically refining the latent representation during a second stage of denoising pass, leveraging information from an initial standard generation. This approach draws inspiration from NVIDIA DLSS, which primarily employs neural networks to super-resolve low-resolution rendered frames in video games.It is also inspired by DDNM Wang et al. (2022), a training-free method achieving image deblurring, super-resolution, etc..

Our mechanism involves a two-stage process. Let $\mathbf { z } _ { t _ { N } } \in \mathbb { R } ^ { C \times F \times H \times W }$ denote the latent representation at denoising step $t _ { n }$ , where $C$ indexes channels, $F$ frames, and $H \times W$ spatial dimensions. First, a standard multi-step denoising process is performed, starting from random noise $z _ { t _ { N - 1 } }$ and conditioned on text and any other relevant input, to obtain an initial latent estimate, which we denote as $z _ { \mathrm { o r i g } }$ . In the second stage, we employ the pre-trained diffusion model and remove motion control from the text conditioning, since $z _ { \mathrm { o r i g } }$ already incorporates trajectory characteristics. This $z _ { \mathrm { o r i g } }$ typically captures the overall structure and semantics of the scene well, but may contain the aforementioned artifacts or lack fine details.

Seond, a refinement denoising pass is initiated, also starting from the sample o $z _ { t _ { N - 1 } }$ .During the initial $K$ steps of this refinement pass (where $K$ is a small number, e.g., 5), we intervene in the denoising process before each model prediction.

For the current latent $z _ { t _ { i } } ^ { \prime }$ at timestep $t$ $( t _ { i } > = t _ { N - K } )$ in the refinement pass, we perform a detailed $z _ { \mathrm { o r i g } }$ back to the current noise level corresponding to timestep $t _ { i }$ , resulting in $z _ { \mathrm { o r i g } , t _ { i } } = ( 1 - t _ { i } ) * z _ { \mathrm { o r i g } } \dot { + }$ $t _ { i } * z _ { t _ { N - 1 } }$ . From this noised estimate, we extract the low-frequency component using a low-pass filter (e.g., a blur operator $A$ , $B ( z ) = A ^ { \mathrm { P i n v } } A z$ , the supplementary materials contain additional information about matrix $A$ N ote that $A$ is not full-rank and $A ^ { \mathrm { P i n v } }$ represents the pseudo-inverse of $A$ ), yielding $z _ { \mathrm { l o w \mathrm { - } f r o m \mathrm { - } o r i g } } = B ( z _ { \mathrm { o r i g } , t _ { i } } )$ T of the initial generation. Simultaneously, we extract the high-frequency component of the current refinement latent $z _ { t _ { i } } ^ { \prime }$ by taking the difference between $z _ { t _ { i } } ^ { \prime }$ and its low-frequency version, giving us

$$
z _ { \mathrm { h i g h \mathrm { . c u r r e n t } } } ^ { \prime } = z _ { t _ { i } } ^ { \prime } - B ( z _ { t _ { i } } ^ { \prime } )
$$

which can also be expressed as $( I - A ^ { \mathrm { P i n v } } A ) z _ { t } ^ { \prime }$ , where $I$ is te identity matix. The latent that will be fed into the diffusion model for the current denoising step is then recomposed by combining these complementary frequency components:

$$
z _ { t _ { i } } ^ { \prime } \gets z _ { \mathrm { l o w . f r o m . o r i g } } + z _ { \mathrm { h i g h . c u r r e n t } } ^ { \prime }
$$

This approach effectively merges the low-frequency information from the initial generation with the high-frequency information from the current refinement step. After this recomposition for the first $K$ steps, the diffusion model $v _ { \theta }$ pc he es o $z _ { t _ { i - 1 } } ^ { \prime }$ as usual. For the remaining steps of the refinement pass (i.e., after the initial $K$ steps), the standard denoising procedure continues without this frequency-domain intervention.

This strategy allows the refinement pass to preserve the robust low-frequency structure established by the initial generation while focusing its generative capacity on producing higher-fidelity highfrequency details. By guiding the initial stages of the refinement in this manner, we effectively reduce common visual artifacts, enhance sharpness, and improve the overall perceptual quality of the generated urban scenes without incurring any additional training costs.

This method increases sampling time but allows step adjustment. The standard denoising step is 50 with CFG at 100 Number of Function Evaluations (NFEs). We use 30 steps without CFG for the first

# Algorithm 3 Time Travel Sampling based on SDE (TTS-SDE)

Input: Initial conditioning S dule $\{ t _ { i } \} _ { i = 0 } ^ { N - 1 }$ nd $t _ { - 1 } = 0$ $c$ , model ddeph $V _ { \theta }$ , $s = 5$ $l = 5$

Output: Generated latent $z _ { \mathrm { 0 } }$

1: Initialize noise ztN −1   
2: for $i = N - 1$ down to 1 do   
3: $t _ { i } \gets$ current timestep from schedule   
4: $\begin{array} { r l } & { \frac { 1 } { \Delta t } \int _ { 0 } ^ { \infty } \epsilon _ { - } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } } \\ & { = \epsilon _ { - } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } } \\ & { \frac { 1 } { \Delta t } \epsilon _ { - } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } - 2 \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } } \\ & { \frac { 1 } { \Delta t } + \epsilon _ { - } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde { \epsilon } \tilde \tilde { \epsilon } } \\ & { \frac { 1 } { \Delta t } + \epsilon _ { - } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } } \\ & { \frac { 1 } { \Delta t } + \epsilon _ { - } \tilde { \epsilon } \tilde { \epsilon } \tilde { \epsilon \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } } \\ &  \frac { 1 } { \Delta t } \epsilon _ { + } \tilde { \epsilon } \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon } \tilde \tilde { \epsilon }  \end{array}$   
5:   
6:   
7:   
8:   
9:   
10:   
11:   
12:   
13:   
14:   
15:   
16:   
17:   
18:   
19:   
20:   
21: end for   
22: v ← Ukmax   
23: end if   
24: $z _ { t _ { i - 1 } } ^ { \mathrm { o u t } } \gets z _ { \mathrm { i n } } + ( t _ { i - 1 } - t _ { i } ) * \hat { v }$   
25: end for   
26: return z0u

denoising pass and 30 steps with CFG for the second pass at 90 NFE, achieving $10 \%$ NFE reduction while improving sampling quality.

While AAM demonstrates excellent performance in generating high-quality individual frames, making it particularly suitable for image-to-video (I2V) conversion and high-quality synthetic data production, it exhibits significant limitations in autoregressive long-video generation scenarios, often resulting in discontinuity between generated frames and historical frames. This limitation may stem from the fact that AAM's pretrained diffusion model is fundamentally based on an I2V architecture rather than a video-to-video (V2V) framework. Fine-tuning AAM's pretrained model on V2V tasks could potentially alleviate this issue.

# 5.3.2 TiMe TraveL Sampling BaseD oN SDE (TTS-sDE) foR EnHanceD Video GENERATION

Inspired by DDNM Wang et al. (2022), Repaint Lugmayr et al. (2022), and OSV Mao et al. (2025) approaches, we introduced a novel high-quality sampling methodology Time Travel Sampling based on SDE (TTS-SDE), as shown in Algorithm 3. For a selected timestep $t _ { n }$ , we first sample forward $l$ steps to obtain $\boldsymbol { x } _ { t _ { \mathrm { m a x } ( n - l , 0 ) } }$ to reconstruct $x _ { t _ { n - 1 } }$ , effectively utilizing prospective information to refine past states.

However, given the deterministic nature of ODE-based sampling with fixed noise inputs, this method's capability to enhance textual controllability remains limited, primarily improving only the sharpness of generated videos. We find that replacing ODE with SDE sampling significantly boosts textual controllability by introducing controlled stochasticity into the generation process.

# 5.4 CAmera MOtion COnTROL

Existing approaches for camera control in video generation often rely on providing a dense sequence of per-frame absolute camera-to-world (c2w) pose matrices Wang et al. (2023b); He et al. (2024). While offering explicit control, this representation can be overly granular, potentially leading to less stable or unintuitive camera trajectories, and may not effectively capture the inherent temporal coherence of continuous camera movements.

Instead of directly using continuous pose matrices, we introduced a discrete camera motion representation, as introduced in Section 4.2. We quantize the camera motion and parse the motion into text as a textual condition.

In addition, beyond the motion direction, we propose to control the motion speed to achieve stable movement in the generated videos. Specifically, as outlined in Algorithm 4, for each camera trajectory we compute three quantitative indicators of the motion speed:

Translational Motion $( \mathcal { V } )$ is captured by the displacement vector between the positions of two consecutive frames, which quantifies the local translational changes along the camera trajectory. By explicitly modeling this motion, the model learns to sustain consistent forward movements or lateral shifts, thereby preserving spatial continuity and fostering a natural flow in the generated videos.

Directional Change $( \mathcal { D } )$ is measured by the angle formed between displacement vectors across every three consecutive frames, effectively measuring local turning or bending of the trajectory. Incorporating this measure encourages the model to produce smooth directional transitions and avoid abrupt changes, resulting in more realistic and stable motion paths.

Rotational Dynamics $( { \mathcal { R } } )$ are characterized by the change in camera orientation angle between two consecutive frames, reflecting the camera's rotational behavior. By explicitly incorporating this rotational cue, the model can better synchronize viewpoint adjustments with the underlying motion dynamics, leading to natural camera panning, tilting, and turning movements.

We also parse these speeds into the text, and they are explicitly provided to the model during training and fixed during inference to prevent the generation of videos with irregular or fluctuating speeds. We then combine it with the discrete camera motion description as the final camera motion condition. It effectively quantizes the continuous camera trajectory into a sequence of semantically meaningful actions, inherently integrating temporal context from the relative pose changes.

The camera motion condition is then combined with a prefix text description of the video ("This video depicts a city walk scene with a first-person view") as the final textual condition.

# Algorithm 4 Camera Motion Speed Calculation

Require: Sequence of camera-to-world matrices $\mathcal { C } = \{ C _ { 0 } , C _ { 1 } , \ldots , C _ { N - 1 } \}$ .   
1: Initialize $\nu  \emptyset$ , $\mathcal { D }  \emptyset$ , $\mathcal { R }  \emptyset$   
2: for $i = 0$ to $N - 3$ do   
3: $p _ { i } \gets C _ { i } [ : , : 3 , 3 ]$   
4: $\begin{array} { r l } & { p _ { i } \gets \mathbf { { \chi } } _ { i \lfloor \cdot \rangle } , \mathbf { { \phi } } _ { \mathrm { { - } } } \mathbf { { { \phi } } } \mathrm { { , } } \mathbf { { \phi } } _ { \mathrm { { - } } } \mathbf { { { \phi } } } \mathrm { { , } } } \\ & { p _ { i + 1 } \gets C _ { i + 1 } [ : , : 3 , 3 ] } \\ & { p _ { i + 2 } \gets C _ { i + 2 } [ : , : 3 , 3 ] } \\ & { v _ { 1 } \gets p _ { i + 1 } - p _ { i } } \\ & { v _ { 2 } \gets p _ { i + 2 } - p _ { i + 1 } } \\ & { \mathrm { { C o m p u t e } } \phi = \operatorname { a r c c o s } \left( \frac { v _ { 1 } \cdot v _ { 2 } } { \| v _ { 1 } \| \| v _ { 2 } \| } \right) } \end{array}$   
5:   
6:   
7:   
8:   
9: Append $v _ { 1 }$ to $\nu$ // Translational vector change   
10: Append $\phi$ to $\mathcal { D }$ // Directional angle change   
11: end for   
12: 13: $\begin{array} { r } { p _ { N - 2 }  C _ { N - 2 } [ : , : 3 , 3 ] } \\ { p _ { N - 1 }  C _ { N - 1 } [ : , : 3 , 3 ] } \end{array}$   
14: v ← pN −1 − pN −2   
15: Append $v$ to $\nu$   
16: for $i = 0$ to $N - 2$ do   
17: zi ← Ci[:, 2]   
18: zi+1 ← Ci+1[:, 2]   
19: Compute $\begin{array} { r } { \theta = \operatorname { a r c c o s } \left( \frac { z _ { i } \cdot z _ { i + 1 } } { | | z _ { i } | | | | z _ { i + 1 } | | } \right) } \end{array}$   
20: Append $\theta$ to $\mathcal { R }$ // Rotation angle change   
21: end for   
22: return $\nu , \mathcal { D } , \mathcal { R }$

# 5.5 APPLICATION

We show some applications in the demo video on the project page.

# 5.5.1 WORLD GENERALIZATION

Though Yume is trained by real-world videos, it shows impressive generalizability to diverse unreal scenes, such as animation, video games, and AI-generated images. Thus, Yume allows not only real-world exploration, but also facilitates unreal-world exploration. Moreover, Yume supports V2V and thus also can adapt to live images taken by iPhone.

# 5.5.2 World Editing

Since Yume shows strong generalizability and thus we can achieve world editing by simply combining Yume with image editing methods such as GPT-4o. We show some examples that change weather, time, and style during video generation using GPT-4o in the demo video.

# 5.6 ACCELERATION

We present an acceleration framework for the Yume model. Departing from conventional approaches, we devise a co-optimization strategy integrating step distillation with cache acceleration. This design stems from an insight: aggressively reducing sampling steps to just one step severely compromises the model's adaptability to diverse complex scenarios, while video quality progressively deteriorates due to error accumulation as generation length increases. To address this, we reduce sampling steps to 14 while incorporating a cache acceleration mechanism.

Weintroduce n adverarial distillationfamework o reduce he amplng es while prein visual quality. The core innovation leverages Generative Adversarial Networks (GANs) to distill the iterative denoising process into fewer steps, following the formulation:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d i f f u s i o n } } + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } }
$$

where $\mathcal { L } _ { \mathrm { d i f f u s i o n } }$ is the standard diffusion loss and ${ \mathcal { L } } _ { \mathrm { a d v } }$ is the adversarial loss term. The training alternates between updating the denoiser DiT Model and the discriminator $\mathcal { D }$ .

The discriminator $\mathcal { D }$ is trained to distinguish between real samples $x _ { \mathrm { r e a l } }$ from the training distribution and denoised samples ${ \hat { x } } _ { 0 }$ generated by the diffusion model:

$$
\hat { x } _ { 0 } = \mathbf { z } _ { t } - t \cdot v _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } )
$$

The discriminator loss combines feature-level and image-level discrimination:

$$
\mathcal { L } _ { \mathcal { D } } = \mathbb { E } [ \mathrm { R e L U } ( 1 - \mathcal { D } ( x _ { \mathrm { r e a l } } ) ) ] + \mathbb { E } [ \mathrm { R e L U } ( 1 + \mathcal { D } ( \hat { x } _ { 0 } ) ) ]
$$

where $\mathcal { D }$ returns both image-level predictions and intermediate feature maps.

The denoiser DiT Model Yume aims to fool the discriminator while maintaining denoising accuracy:

$$
\mathcal { L } _ { \mathrm { a d v } } = - \mathbb { E } [ \mathcal { D } ( \hat { x } _ { 0 } ) ]
$$

We adopt the discriminator design from OSV Mao et al. (2025), as this approach significantly reduces memory consumption while achieving excellent discriminative performance.

# 5.6.2 CACHE-ACCELERATING

We use the caching mechanism that reduces computational redundancy by reusing intermediate residual features across denoising steps. The system employs layer-specific caching policies to achieve computation reduction while preserving output quality.

For layer $l$ at timestep $t _ { n }$ , the residual feature $\Delta x _ { t _ { n } } ^ { l }$ is cached when:

$$
\mathcal { C } _ { t _ { n } } ^ { l } = \left\{ \begin{array} { l l } { ( \mathrm { B l o c k } ^ { l } ( x _ { t _ { n } } ^ { l - 1 } ) - x _ { t _ { n } } ^ { l - 1 } ) _ { \mathrm { b f l o a t l 6 } } } & { \mathrm { i f ~ } l \in \mathcal { L } _ { \mathrm { c a c h e } } } \\ { \emptyset } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

where $\mathcal { L } _ { \mathrm { c a c h e } }$ denotes predefined cacheable layers. These cached residuals are stored in bf loat 1 6 precision.

At subsequent timestep $t _ { n - 1 }$ , layer $l$ computes:

$$
\begin{array} { r } { x _ { t _ { n - 1 } } ^ { l } = \left\{ \begin{array} { l l } { x _ { t _ { n - 1 } } ^ { l - 1 } + \mathcal { C } _ { t _ { n } } ^ { l } } & { \mathrm { w h e n } l \in \mathcal { L } _ { \mathrm { c a c h e } } } \\ { { \mathrm { B l o c k } } ^ { l } ( x _ { t _ { n - 1 } } ^ { l - 1 } ) } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}
$$

We consider an acceleration ratio of $1 : l _ { c }$ , whereby full computation is performed at time step $t _ { n }$ to obtain cache $\mathcal { C } _ { t _ { n } } ^ { l }$ $t _ { n - 1 }$ through $t _ { n - l _ { c } }$ utilize $\mathcal { C } _ { t _ { n } } ^ { l }$ for skip-step processing. To evaluate the impact of individual blocks within the DiT denoising model, we introduce an MSE-based importance metric. Specifically, across $N / ( l _ { c } + 1 )$ cached computation segments from $t _ { n }$ to $t _ { n - l _ { c } }$ in t ef $\mathcal { C } _ { t _ { n } } ^ { l }$ at $t _ { n }$ we systematically ablate each of the 40 DiT blocks at step $t _ { n - l _ { c } }$ . The influence score for the $i$ -th block is quantified as:

$$
\begin{array} { r } { \mathrm { M S E ~ S c o r e } _ { i } = \mathrm { M e a n } \left( \left| x _ { t _ { n - l _ { c } } } ^ { \mathrm { R e m o v e } i } - x _ { t _ { n - l _ { c } } } \right| ^ { 2 } \right) , i \in [ 1 , 2 , 3 , . . , 4 0 ] } \end{array}
$$

where Mean denotes the averaging operation. Each video yields $N / ( l _ { c } + 1 )$ measurements per block, generating 40 temporal MSE profiles per segment. These profiles are averaged across time steps and aggregated over 32 videos to produce 40 composite MSE scores. Figure 4 illustrates that central DiT blocks exhibit minimal impact while initial and terminal blocks demonstrate maximal influence, leading to the selection of the 10 lowest-scoring blocks as predefined cacheable layers.

![](images/4.jpg)  
MSE Score Distribution by Layer Index   
Figure 4: Significance of Individual DiT Blocks.

# 6 EXPERIMENT

# 6.1 EXPERIMENTAL SETTINgS

# 6.1.1 TRaining Details

We utilized the SkyReels-V2-14B-540P as the pre-trained model. The training process involved video resolutions of $5 4 4 \times 9 6 0$ , with a frame rate of 16 FPS, a batch size of 40, and the Adam optimizer with a learning rate of 1e-5. Training was conducted across NVIDIA A800 GPUs over 7,000 iterations.

# 6.1.2 Evaluation Dataset

Existing video generation evaluation methods are not well-suited for complex scenes and interactive generation with keyboard inputs. To address this issue, we developed the Yume-Bench evaluation framework. Specifically, we exclude training videos from the Sekai-Real-HQ dataset and instead sample test videos with quantized camera motions. For rare actions, such as walking backward or tilting the camera up and down, we randomly sample images rather than video clips. In total, we collected 70 videos or images, covering a wide range of complex action combinations, as detailed in Table 1.

# 6.1.3 Evaluation Details

Yume-Bench evaluates two core capabilities of models: visual quality and instruction following (camera motion tracking), using six fine-grained metrics. In the instruction following evaluation, we assess whether the generated videos correctly follow the intended walking direction and camera movements. While camera poses estimation methods, such as MegaSaM Li et al. (2025c), can automate this evaluation, the camera motion prediction in the generated videos is not sufficiently accurate, and quantization errors may occur. Therefore, we conduct a human evaluation to identify the accuracy of the generated motion. For the remaining metrics, we adopt those from VBench Huang et al. (2024), including subject consistency, background consistency, motion smoothness, aesthetic quality, and imaging quality. The test data resolution is $5 4 4 \times 9 6 0$ with a frame rate of 16 FPS, comprising a total of 96 frames. We applied 50 inference steps for all models tested.

Table 1: Keyboard-Mouse Action Combinations   

<table><tr><td>Keyboard-Mouse Action</td><td>Count</td></tr><tr><td>No Keys + Mouse Down</td><td>2</td></tr><tr><td>No Keys + Mouse Up</td><td>2</td></tr><tr><td>S Key + No Mouse Movement</td><td>2</td></tr><tr><td>W+A Keys + No Mouse Movement</td><td>29</td></tr><tr><td>W+A Keys + Mouse Left</td><td>6</td></tr><tr><td>W+A Keys + Mouse Right</td><td>17</td></tr><tr><td>W+D Keys + No Mouse Movement</td><td>5</td></tr><tr><td>W+D Keys + Mouse Left</td><td>5</td></tr><tr><td>W+D Keys + Mouse Right</td><td>2</td></tr></table>

# 6.2 QUALITATIVE RESULTS

# 6.2.1 IMAGE-TO-VIDEO GENERATION

We compared several state-of-the-art (SOTA) image-to-video generation models, including Wan-2.1 and MatrixGame, as shown in Table 2. Our experimental results revealed that: (1) Wan-2.1 shows limited instruction-following capabilities while using textual instructions to control camera motion. (2) Although MatrixGame demonstrates some degree of controllability, it struggles to generalize to real-world scenarios and lacks sufficient scene replication control. In contrast, our Yume excels in controllability, with its instruction-following capability scoring 0.657, significantly outperforming other models. Additionally, Yume achieves optimal or near-optimal performance across other metrics, demonstrating our superior visual quality.

![](images/5.jpg)  
Figure 5: Metric Dynamics in Long-video Generation. We use TTS-SDE. We performed a total of 9 extrapolations. "4 infs" represents using videos obtained from 4 extrapolations (totaling 8 seconds) for metric calculation, while $^ { \circ \circ } 2$ infs" represents using videos obtained from 2 extrapolations (also totaling 4 seconds) for metric calculation.

# 6.2.2 Validation OF Long-video Generation Performance

To assess the long-video generation capability, we created an 18-second video sequence where Yume generates 2-second segments incrementally. During the first 8 seconds, the motion patterns remained consistent with the test set, followed by a transition to continuous forward movement (W) in the subsequent 10 seconds. As shown in Figure 5, mild content decay was observed: subject consistency decreased by $0 . 5 \%$ $\textcircled { 0 . 9 3 4 } \xrightarrow { } 0 . 9 3 0$ ), and background consistency dropped by $0 . 6 \%$ $( 0 . 9 4 7 \substack {  } 0 . 9 4 1 \ '$ between the 0-8s and 12-18s segments, indicating that Yume maintains reasonable stability over time. It is worth noting that during the motion transition phase (8-12s), instruction-following performance dropped by $8 . 6 \%$ $0 . 9 4 7 {  } 0 . 9 4 1 _ { . }$ ). This decline can be attributed to the inertia from the motion in the input video, which hindered an immediate reversal of direction. However, the inertial effect diminished after 12 seconds, leading to a significant recovery in instruction-following performance by $2 2 . 3 \%$ $0 . 6 3 6 {  } 0 . 8 1 9 _ { . }$ .

Table 2: Quality comparison of different models. Wan-2.1 utilize text-based control. MatrixGame employs its own native keyboard/mouse control scheme. All models use the same random seed.   

<table><tr><td>Model</td><td>Instruction Following ↑</td><td>Subject Consistency ↑</td><td>Background Consistency ↑</td><td>Motion Smoothness ↑</td><td>Aesthetic Quality ↑</td><td>Imaging Quality ↑</td></tr><tr><td>Wan-2.1 Wan et al. (2025)</td><td>0.057</td><td>0.859</td><td>0.899</td><td>0.961</td><td>0.494</td><td>0.695</td></tr><tr><td>MatrixGame Zhang et al. (2025)</td><td>0.271</td><td>0.911</td><td>0.932</td><td>0.983</td><td>0.435</td><td>0.750</td></tr><tr><td>Yume (Ours)</td><td>0.657</td><td>0.932</td><td>0.941</td><td>0.986</td><td>0.518</td><td>0.739</td></tr></table>

Table 3: Ablation study on different samplers.   

<table><tr><td>Model</td><td>Instruction Following ↑</td><td>Subject Consistency ↑</td><td>Background Consistency ↑</td><td>Motion Smoothness ↑</td><td>Aesthetic Quality ↑</td><td>Imaging Quality ↑</td></tr><tr><td>Yume-ODE</td><td>0.657</td><td>0.932</td><td>0.941</td><td>0.986</td><td>0.518</td><td>0.739</td></tr><tr><td>Yume-SDE</td><td>0.629</td><td>0.927</td><td>0.938</td><td>0.985</td><td>0.516</td><td>0.737</td></tr><tr><td>Yume-TTS-ODE</td><td>0.671</td><td>0.923</td><td>0.936</td><td>0.985</td><td>0.521</td><td>0.737</td></tr><tr><td>Yume-TTS-SDE</td><td>0.743</td><td>0.921</td><td>0.933</td><td>0.985</td><td>0.507</td><td>0.732</td></tr></table>

# 6.3 ABLATION STUDY

# 6.3.1 VERiFication OF TTS-SDE EFfectiveNESS

To assess the effectiveness of TTS-SDE, we replaced the ODE sampling with SDE and TTS-SDE for comparison. As shown in Table 4, while SDE sampling resulted in a decline across all metrics, TTSSDE achieved a significant improvement in instruction following, despite a slight reduction in other indicators. This indicates that TTS-SDE strategically introduces noise perturbations, enhancing the refinement of motion trajectories in the generated videos. Furthermore, the integration of TTS-SDE has resulted in improved aesthetic scores, with our observations revealing clearer and more detailed generated videos.

Table 4: Validation of distillation method effectiveness.   

<table><tr><td>Model</td><td>Time (s)↓</td><td>Instruction FoFllowing </td><td>Subject Consistency ↑</td><td>Background Consistency ↑</td><td>Motion Smoothness ↑</td><td>Aesthetic Quality ↑</td><td>Imaging Quality ↑</td></tr><tr><td>Baseline</td><td>583.1</td><td>0.657</td><td>0.932</td><td>0.941</td><td>0.986</td><td>0.518</td><td>0.739</td></tr><tr><td>Distil</td><td>158.8</td><td>0.557</td><td>0.927</td><td>0.940</td><td>0.984</td><td>0.519</td><td>0.739</td></tr></table>

# 6.3.2 VALIDATING THE EFFECT OF MODEL DISTILLATION

After distilling the model to reduce the number of steps from 50 to 14, we compared it with the original model. We found that, except for instruction following, the other metrics showed minimal differences from the original model. This may be because fewer steps weaken the model's text-control capability.

# 6.4 VIsualization REsults

As illustrated in Figure 6, we generated multiple video sequences using the initial frame image and quantized camera trajectories, demonstrating that Yume accurately follows the predefined motion paths during video generation. Figure $^ { 7 }$ demonstrates the effectiveness of AAM by generating clearer videos while avoiding illogical scenes such as aberrant snowman artifacts.

![](images/6.jpg)  
Figure 6: Yume demonstrates superior visual quality and precise adherence to keyboard control in real-world and unreal scenarios.

![](images/7.jpg)  
Figure 7: AAM Improves Structural Details in Urban and Architectural Scenes.

# 7 CONCLuSION

In this paper, we introduce a preview version of Yume, which is an interactive world generation model that allows the use of keyboard inputs to explore a dynamic world created by an input image. Moreover, it can do infinite video generation in an autoregressive manner. Yume consists of four

main components, including camera motion quantization, video generation architecture, advanced sampler, and model acceleration.

Yume is a long-term project that has established a solid foundation, yet stillfaces numerous challenges to address, such as the visual quality, runtime efficiency, and control accuracy. Moreover, many functions need to be achieved, such as interaction with objects.

# REFERENCES

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023a.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2256322575, 2023b.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. Technical report, OpenAI, February 2024. URL ht tps : / /opena i . com/research/video-generation-models-as-world-simulators.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Stefan Bechtle, Sumit Bileschi, Sebastian Borgeaud, Stephanie Borja, Arun Byravan, Ken Caluwaerts, Marion Caron, Tiago Carvalho, Andrew Cassirer, Yiding Chen, Michele Covell, Silvia de Abreu, Andrew Fant, AMA Glaese, Tom Henighan, Lennon Hughes, Egle Kasinskaite, Cosmin Kema, Misha Kumar, Matt Kyle, Hubert Laur, Tom Lovitt, Elliot Rutherford, Maria Rutherford, Martin Salz, Laurent Sifre, John Simon, Oleksandr Smytnis, Tom Valdevit, Oriol Vinyals, Greg Wayne, Jonathan Zack, and Zhaosen Zhu. Genie: Generative interactive environments. In Sanjoy Dasgupta, Analia Gomez, Andrew Lan, and Yee Whye Teh (eds.), Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235, pp. 46174640. PMLR, 0915 Jul 2024.

Sheng Chen, Zheng Chen, Chengchen Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151, 2023.

Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2316423173, 2023.

Shuai Geng, Zexu Liu, Zuanbang Shang, Jinxiang Chai, Chen Change Loy, and Ziwei Liu. Training. free camera control for video generation with text and image prompts, 2024.

Google DeepMind. SIMA: A Scalable, Instructable Multiworld Agent. https://deepmind.google/discover/blog/ sima-generalist-ai-agent-for-3d-virtual-environments/, March 2024. Accessed: May 20, 2025.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Samy Bengio, Hanna Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 24512463. Curran Associates, Inc., 2018. URL https://papers.nips.cc/paper/2018/hash/ 7512c32074c2314308c0951959dee873-Abstract.html.

Yoav HaCohen, Mendi Shen, Daniel Dror, Daniel Glickman, Rotem Mulayoff, Oran Shimony, Oran Langman, David Elyada, Eden Kristal, Evgeny Bistrov, Eyal Molad, Hila Chefer, Michal Geyer, Shai Bagon, Yedid Hoshen, and Tomer Michaeli. LTX-Video: Realtime video latent diffusion, 2025.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for text-to-video generation, 2024.

Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Danil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. StreamingT2V: Consistent, dynamic, and extendable long video generation from text, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840 6851, 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models, 2022.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2180721818, 2024.

Kumara Kahatapitiya, Haozhe Liu, Sen He, Ding Liu, Menglin Jia, Chenyang Zhang, Michael S Ryoo, and Tian Xie. Adaptive caching for faster video generation with diffusion transformers. arXiv preprint arXiv:2411.02397, 2024.

Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023.

Ruihuang Li, Caijin Zhou, Shoujian Zheng, Jianxiang Lu, Jiabin Huang, Comi Chen, Junshu Tang, Guangzheng Xu, Jiale Tao, Hongmei Wang, et al. Hunyuan-game: Industrial-grade intelligent game creation model. arXiv preprint arXiv:2505.14135, 2025a.

Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, et al. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025b.

Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast and robust structure and motion from casual dynamic videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1048610496, 2025c.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It's time to cache for video diffusion model. arXiv preprint arXiv:2411.19108, 2024.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Zheng Liu and Lisong Wang. A diffusion model based quality enhancement method for HEVC compressed video, 2023.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1146111471, 2022.

Yang Luo, Xuanlei Zhao, Mengzhao Chen, Kaipeng Zhang, Wenqi Shao, Kai Wang, Zhangyang Wang, and Yang You. Enhance-A-Video: Better generated video for free, 2025.

Zhaoitem Ma, Yufei Liu, Lin Geng, Jiaxin Liu, Yaohua Tang, Ming Lu, Xinyuan Chen, Jingwen He, Zilong Huang, Fan Wen, Ping Li, Deliang Fan, Sitong Su, Kai Li, Can Wang, ShiFeng Zhang, Min Dou, Xiaoyi Dong, JiaLUnLiu, Boqin He, Yong He, Yang Song, Haibo E, Gang Yue, Yaokun Liu, u  Xu  hih i boa -an Yong Dou, Yujun Shen, Yu Qiao, and Tat-Seng Chua. Hunyuanvideo: A systematic framework for large video generative models, 2024.

Xiaofeng Mao, Zhengkai Jiang, Fu-Yun Wang, Wenbing Zhu, Jiangning Zhang, Hao Chen, Mingmin Chi, and Yabiao Wang. Osv: One step is enough for high-quality image to video generation. arXiv preprint arXiv:2409.11367, 2024a.

Xiaofeng Mao, Zhengkai Jiang, Qilin Wang, Chencan Fu, Jiangning Zhang, Jiafu Wu, Yabiao Wang, Chengjie Wang, Wei Li, and Mingmin Chi. Mdt-a2g: Exploring masked diffusion transformers for co-speech gesture generation. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 32663274, 2024b.

Xiaofeng Mao, Zhengkai Jiang, Fu-Yun Wang, Jiangning Zhang, Hao Chen, Mingmin Chi, Yabiao Wang, and Wenhan Luo. Osv: One step is enough for high-quality image to video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1258512594, 2025.

Prem Nagrath, Nareddy Reddy, Jian Ren, Robin keyValue, Saad Nadeem, Meng Li, Ser-Nam Lim, Chao Liu, Guttu TG, Rama Chellappa, and Ajinkya Kale. Mochi-diffusion-xl: An efficient model for high-resolution video generation, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 87488763. PmLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1068410695, 2022.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-tovideo generation without text-video data. In International Conference on Learning Representations (ICLR), 2023.

Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics, 2015.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023.

The Step-Video-T2V Team. Step-Video-T2V technical report: The practice, challenges, and future of video foundation model, 2025.

Cédric Villani et al. Optimal transport: old and new, volume 338. Springer, 2009.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Fu-Yun Wang, Zhaoyang Huang, Weikang Bian, Xiaoyu Shi, Keqiang Sun, Guanglu Song, Yu Liu, and Hongsheng Li. Animatelcm: Computation-efficient personalized style video generation without personalized video data. In SIGGRAPH Asia 2024 Technical Communications, pp. 15. 2024a.

Fu-Yun Wang, Zhaoyang Huang, Xiaoyu Shi, Weikang Bian, Guanglu Song, Yu Liu, and Hongsheng Li. Animatelcm: Accelerating the animation of personalized diffusion models and adapters with decoupled consistency learning. arXiv preprint arXiv:2402.00769, 2024b.

Fu-Yun Wang, Ling Yang, Zhaoyang Huang, Mengdi Wang, and Hongsheng Li. Rectified diffusion: Straightness is not your need in rectified flow. arXiv preprint arXiv:2410.07303, 2024c.

Fu-Yun Wang, Zhaoyang Huang, Alexander Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. Advances in Neural Information Processing Systems, 37:8395184009, 2025.

Xiang Wang, Shiwei Zhang, Han Zhang, Yu Liu, Yingya Zhang, Changxin Gao, and Nong Sang. Videolcm: Video latent consistency model. arXiv preprint arXiv:2312.09109, 2023a.

Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A unified and flexible motion controller for video generation, 2023b.

Wayve Technologies. GAIA-1: A Generative AI Model for Autonomous Driving. https : /wayve.com/blog/introducing-gaial-generative-ai-model-driving/, 2023. Accessed: May 20, 2025.

Junhao Xia, Shiyuan Yang, Hongbo Zhao, Xinyuan Chen, Di Zhang, Yake Wei, Ceyuan Yang, Yujun Shen, and Di Liu. DAPE: Dual-stage parameter-efficient fine-tuning for consistent video editing with diffusion models, 2025.

Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. WORLDMEM: Long-term consistent world simulation with memory, 2025.

Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-Video: Customized video generation with user-directed camera movement and object motion. arXiv preprint arXiv:2402.03162, 2024.

Yunlong Yuan, Yuanfan Guo, Chunwei Wang, Wei Zhang, Hang Xu, and Li Zhang. FreqPrior: Improving video diffusion models with frequency filtering gaussian noise. In International Conference on Learning Representations (ICLR), 2025.

Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.

Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Zedong Gao, Eric Li, Yang Liu, and Yahui Zhou. Matrix-game: Interactive world foundation model. arXiv preprint, 2025. Technical report, https://github.com/SkyworkAI/Matrix-Game/ raw/main/assets/report . pdf. Accessed: May 20, 2025. (Year listed as 2025 in some project BibTeX, but content available 2024 or earlier.).

Yucan Zhang, Zixu Zhang, Ceyuan Yang, Yuming Liu, Zhaowei Chen, Shiliang Pu, Yaxiong Wang, Yujun Shen, Yu Qiao, and Yuliang Liu. CameraCtrl II: Dynamic scene exploration via cameracontrolled video diffusion models, 2024.

Pengfei Zhou, Jie Xia, Xiaopeng Peng, Wangbo Zhao, Zilong Ye, Zekai Li, Suorong Yang, Jiadong Pan, Yuanxiang Chen, Ziqiao Wang, et al. Neural-driven image editing. arXiv preprint arXiv:2507.05397, 2025.

Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. arXiv preprint arXiv:2410.05317, 2024.

# A Ablation Study on Image to Video

Table 5: Effect of Controlled Condition Injection Methods.   

<table><tr><td>Name</td><td>instruction following ↑</td><td>aesthetic quality ↑</td><td>imaging quality ↑</td><td></td><td>motion smoothness ↑ background consistency ↑</td></tr><tr><td>AdaLN-Zero</td><td>0.35</td><td>0.517</td><td>0.694</td><td>0.988</td><td>0.936</td></tr><tr><td>Cross-Attention</td><td>0.45</td><td>0.507</td><td>0.701</td><td>0.987</td><td>0.935</td></tr><tr><td>Text injection</td><td>0.45</td><td>0.492</td><td>0.694</td><td>0.991</td><td>0.902</td></tr></table>

We validated the effectiveness of the MVDT and AAM modules within our image-to-video (I2V) pipeline through comparative analysis of 20 randomly selected video sequences. For this experiment, all models were trained exclusively for 1,000 iterations.

Effect of Controlled Condition Injection Methods. We replace Yume's MVDT architecture with DiT and removed the text injection approach as Baseline-1. To evaluate different controlled condition injection methods, we incorporated adaLN-zero, cross-attention, and text injection into Baseline-1 while maintaining identical training parameters. As shown in Table 5, these methods demonstrated complementary advantages in V-bench metrics. We adopt the text injection approach due to its superior controllability, seamless integration with pretrained models (requiring no architectural modifications), and parameter-efficient design that introduces no additional learnable parameters.

Table 6: Effect of MVDT.   

<table><tr><td>MVDT ↓</td><td>aesthetic quality ↑</td><td>imaging quality ↑</td><td>motion smoothness ↑</td><td>background consistency ↑</td></tr><tr><td>+</td><td>0.517</td><td>0.702</td><td>0.985</td><td>0.929</td></tr><tr><td>-</td><td>0.492</td><td>0.694</td><td>0.991</td><td>0.902</td></tr></table>

Effect of MVDT Architecture. We validated the efficacy of MVDT by comparing Yume performance with and without this architecture in identical training configurations, we deliberately excluded the AAM in all experimental configurations. Table 6 confirms that the MVDT structure consistently enhances Yume's Generate capabilities. The MVDT architecture enhances Yume's ability to capture structural relationships between frames, significantly reducing artifacts in generated videos.

Table 7: Effect of Anti-Artifact Mechanism (AAM).   

<table><tr><td>AAM ↑</td><td>aesthetic quality ↑</td><td>imaging quality ↑</td><td>motion smoothness ↑</td><td>dynamic degree ↑</td></tr><tr><td>+</td><td>0.529</td><td>0.737</td><td>0.986</td><td>0.937</td></tr><tr><td>-</td><td>0.517</td><td>0.702</td><td>0.985</td><td>0.929</td></tr></table>

# B QuantizEd CaMERA MotioN OF Text

We generate natural language descriptions for human movement and camera motion like the example in vocab_human and vocab_camera of Figure 8, 9.

# C ACTIoN DISTRiBUTION StATIsTICS

Figure 10 presents the distribution of 21,526 video clips across different action combinations.

# D Core Algorithm Implementation Of Gaussian Blur Kernel

We implement a separable 2D linear operator using height/width blur kernels with SVD decomposition. The projection $\bar { B ( z ) } = A ^ { \mathrm { P i n v } } A z$ extracts low-frequency components from first-stage results, while the null-space projection $z - B ( z ) = ( I - A ^ { \mathrm { P i n v } } A ) z$ preserves hig-requency details i secondste outputs. We provide PyTorch-style implementation code.

W : Person moves forward (W). A : Person moves left (A). S : Person moves backward (S). D : Person moves right (D). vocab_human $=$ $\mathrm { W } { + } \mathrm { A }$ : Person moves forward and left $( \mathsf { W } { + } \mathbf { A } )$ . $\mathrm { W } { + } \mathrm { D }$ : Person moves forward and right $( \mathrm { W } \mathrm { + D } )$ . $_ { \mathrm { S + D } }$ : Person moves backward and right $( \mathbf { S } { + } \mathbf { D } )$ . $\mathsf { S } { + } \mathsf { A }$ : Person moves backward and left $( \mathsf { S } { + } \mathsf { A } )$ . None $:$ Person stands still (·).

→ : Camera turns right $(  )$ . ← : Camera turns left (←). ↑ : Camera tilts up (↑). ↓ : Camera tilts down (↓). vocab_camera $=$ ↑→ : Camera tilts up and turns right $( \uparrow \to )$ . $\uparrow $ : Camera tilts up and turns left $( \uparrow  )$ . $\downarrow $ : Camera tilts down and turns right $( \downarrow  )$ . $\downarrow $ : Camera tilts down and turns left $( \downarrow  )$ . : Camera remains still (·).

1 def project_null_space(x: torch.Tensor,   
2 A_operator: 'LinearOperator2D') $- >$ torch.Tensor:   
3   
4 original_shape $=$ x.shape   
5 x_flat $=$ x.view(-1, \*original_shape[-2:])   
6   
7   
8 Ax = A_operator.A(x_flat)   
9 A_Pinv_Ax $=$ A_operator.A_inv(Ax)   
10   
11   
12 I_A_Pinv_A_x $=$ x_flat - A_Pinv_Ax   
13 return I_A_Pinv_A_x.view(original_shape)

# LiNEAROPERATOR2D CLASS

1 class LinearOperator2D: 2 def _init_(self, 3 kernel_H: torch.Tensor, 4 kernel_W: torch.Tensor, 5 H: int, 6 W: int, 7 device: str $=$ None): 8 """Initialize separable 2D operator with SVD decomposition"""   
9 self.device $=$ device or ('cuda' if torch.cuda.is_available() else 'cpu')   
10 self.H, self.W $\qquad = \ \mathrm { ~ H ~ }$ , W   
11   
12 # Height-direction operator A_H (H x H)   
13 A_H $=$ torch.zeros (H, H, device $=$ self.device)   
14 for i in range(H):   
15 for j in range(i - len(kernel_H)//2, i $^ +$ len(kernel_H)//2 1):   
16 if $0 ~ < = ~ \mathrm { ~ j ~ } ~ < ~ \mathrm { ~ H ~ }$ :   
17 A_H[i,j] $=$ kernel_H[j - i + len(kernel_H)//2]   
18   
19 U_H, S_H, Vt_H $=$ torch.linalg.svd(A_H, full_matrices $=$ False)   
20 self.U_H, self.S_H, self.Vt_H $=$ U_H, S_H, Vt_H   
21 self.S_pinv_H $=$ torch.where(S_H $>$ 1e-6, 1/S_H, torch.zeros_like (S_H))   
22   
2\$ # Width-direction operator A_W $W ~ \times ~ W )$ A_W $=$ torch.zeros(W, W, device $=$ self.device) for i in range(W):   
26 for j in range(i - len(kernel_W)//2, i $^ +$ len(kernel_W)//2 + 1):   
27 if $0 < = \dot { \mathrm { ~ \ j ~ } } < \mathtt { W }$ :   
28 A_W[i,j] $=$ kernel_W[j - i + len(kernel_W)//2]   
29   
30 U_W, S_W, Vt_W $=$ torch.linalg.svd(A_W, full_matrices $=$ False)   
31 self.U_W, self.s_W, self.Vt_W $=$ U_W, S_W, Vt_W   
32 self.s_pinv_W $=$ torch.where $\mathrm { ~ \bar { ~ } S ~ } \_ { \mathsf { W } } \ > \ \mathrm { ~ 1 e - } 6$ , 1/s_W, torch.zeros_like (S_W))

# OPERAToR APPLiCATIon METHODS

1 def A(self, x: torch.Tensor) -> torch.Tensor:   
2 """Forward operation: A $=$ A_W @ A_H""   
3 # Height processing   
4 x_h $=$ x.movedim(-2, -1) # [..., W, H]   
5 x_h $=$ torch.matmul(x_h, self.Vt_H.T)   
6 x_h $=$ torch.matmul(x_h, torch.diag(self.S_H))   
7 x_h $=$ torch.matmul(x_h, self.U_H.T)   
8 x_h $=$ x_h.movedim(-1, -2) # [..., H, W]   
9   
10 # Width processing   
11 x_hw $=$ torch.matmul(x_h, self.Vt_W.T)   
12 x_hw $=$ torch.matmul(x_hw, torch.diag(self.s_W))   
13 x_hw $=$ torch.matmul(x_hw, self.U_W.T)   
14 return x_hw   
15   
16 def A_inv(self, y: torch.Tensor) -> torch.Tensor:   
17 """Pseudoinverse operation"""   
18 # Width pseudoinverse   
19 $\begin{array} { r l } { \mathrm { Y } \_ { \mathsf { W } } } & { { } = } \end{array}$ torch.matmul(y, self.U_W)   
20 Y_w $=$ torch.matmul(y_w, torch.diag(self.S_pinv_W))   
21 $\begin{array} { r l } { \mathrm { Y } \_ { \mathsf { W } } } & { { } = } \end{array}$ torch.matmul(y_w, self.Vt_W)   
22   
23 # Height pseudoinverse   
24 y_hw $=$ y_w.movedim(-2, -1) # [..., W, H]   
25 y_hw $=$ torch.matmul(y_hw, self.U_H)   
26 y_hw $=$ torch.matmul(y_hw, torch.diag(self.S_pinv_H))   
27 y_hw $=$ torch.matmul(y_hw, self.Vt_H)   
28 return y_hw.movedim(-1, -2) # [..., H, W]

# OPEratoR InItiaLIZatioN EXaMPLE

1 # Blur kernels   
2 kernel_H $=$ torch.tensor([0.1, 0.8, 0.1], device $u = \prime$ cpu') # Height kernel   
3 kernel_w $=$ torch.tensor([0.2, 0.6, 0.2], device $u = \bar { }$ cpu' # Width kernel   
4   
5 # Operator instantiation   
6 A_op $=$ LinearOperator2D(   
7 kernel_H $=$ kernel_H,

![](images/8.jpg)  
Figure 10: Overview of action distribution.

# E The Joint Optimization Of Adversarial Distillation and Caching

We consider the joint optimization of adversarial distillation and caching. As illustrated in Figure 11, the DiT module performs intermediate feature caching for specific layers $l \in \mathcal { L } _ { \mathrm { c a c h e } }$ during the state transition from $\boldsymbol { x } _ { t _ { n + k } }$ to $x _ { t _ { n - 1 } }$ (where $t _ { n + k } < 1 ,$ . The "Stop Grad" operation denotes gradient truncation, simulating scenarios where the DiT model encounters cached features during inference.

![](images/9.jpg)  
Figure 11: Acceleration Method Design.

This learning approach reduces errors when utilizing cached features. We optimize denoising fidelity using the Diffusion Loss while additionally incorporating the adversarial loss.