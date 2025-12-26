# RELIC: Interactive Video World Model with Long-Horizon Memory

Yicong Hong\*†, Yiqun Mei\*, Chongjian ${ \pmb { \mathrm { G e } } } ^ { * }$ , Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, Hao Tan \*First Authors in Random Order, \*Project Lead.

A truly interactive world model requires three key ingredients real-time long-horizon streaming, consistent spatial memory, and precise user control. However, most existing approaches address only one of these aspects in isolation, as achieving all three simultaneously is highly challenging—for example, long-term memory mechanisms often degrade real-time performance. In this work, we present RELIC, a unified framework that tackles these three challenges altogether. Given a single image and a text description, RELIC enables memory-aware, long-duration exploration of arbitrary scenes in real time. Built upon recent autoregressive video-diffusion distillation techniques, our model represents long-horizon memory using highly compressed historical latent tokens encoded with both relative actions and absolute camera poses within the KV cache. This compact, camera-aware memory structure supports implicit 3D-consistent content retrieval and enforces long-term coherence with minimal computational overhead. In parallel, we fine-tune a bidirectional teacher vidmode togneratesequences beyondt riinal-second traininghorizon, andtransformntocual student generator using a new memory-efficient self-forcing paradigm that enables full-context distilation over long-duration teacher as well as long student self-rollouts. Implemented as a 14B-parameter model and trained on a curated Unreal Enginerendered dataset, RELIC achieves real-time generation at 16 FPS while demonstrating more accurate action following, more stable long-horizon streaming, and more robust spatial-memory retrieval compared with prior work. These capabilities establish RELI as a strong foundation for the next generation of interactive world modeling. Date: December 1st, 2025 Project Page: https://relic-worldmodel.github.io/

# 1 Introduction three-dimensional physical world (Ha and Schmidhuber, 2018; World Labs, 20c; Brooks et al, 2024). By enablig ibetw e n  womo crewva iH  h PrHaBr R 22 Gu, 205). Recent advances in videogenerationthorouh diffusion models (Wan e al., 2025;Gao al 02; oae relz umersivean iteractiveeperiences by learng cntrollablutoregressiveAR)vido mode frm n i u  Fh; H l0e PH Bal 0T Yu 0 et al., 2025; Huang et al., 2025b; Chen et al., 2025b).

Til paRaliooi ha i a resnsecntuus sreamuser controls, u  amemovements r keyboarinus.Consisten sal mya a   o  pT t ouregressive (AR)video diffusion models, real-timelatency and throuput emand ew-step or even one-ste dde (Son el 0; Yine al 04b,a, 2025; Lin etl 20 Wan etl 025b.Lo-oz ai H Achiei consistent memoyurhe requi stori onghistori  pas neateormatn—eithe the Vchdeia ehani u l 0; Xi 0Ma l0R Yu  l20 Huan l 02Howeve satisyitheerequreets imutaneously challe: maai-erpa steyiy sanhal cpeiy bandwidth bottleneck, which directly conflicts with the need for real-time responsiveness.

![](images/1.jpg)  
first-frame image in real time. Built as a 14B-parameter autoregressive model, RELIC generates videos at $4 8 0 \times 8 3 2$ resolution, 16 FPS, for up to 20 seconds, exhibiting consistent long-horizon spatial memory.

Ta  o dr parallelis R nraatirotat n htecin-elate ti rspevey.

<table><tr><td></td><td>The Matrix</td><td>Genie-2</td><td>GameCraft</td><td>Yume</td><td>Yan</td><td>Matrix-Game 2.0</td><td>Genie-3</td><td>RELIC (ours)</td></tr><tr><td>Data Source</td><td>AAA Games</td><td>Unknown</td><td>AAA Games</td><td>Sekai</td><td>3D game</td><td>Minecraft+UE+Sekai</td><td>Unknown</td><td>UE</td></tr><tr><td>Action Space</td><td>4T4R</td><td>5T4R2E</td><td>4T4R</td><td>4T4R</td><td>7T2R</td><td>4T</td><td>5T4R1E</td><td>6T6R</td></tr><tr><td>Resolution</td><td>720×1280</td><td>720×1280</td><td>720×1280</td><td>544×960</td><td>1080×1920</td><td>352×640</td><td>704×1280</td><td>480×832</td></tr><tr><td>Speed</td><td>8-16 FPS</td><td>Unknown</td><td>24 FPS</td><td>16 FPS</td><td>60 FPS</td><td>25 FPS</td><td>24 FPS</td><td>16 FPS</td></tr><tr><td>Duration</td><td>Infinite</td><td>10-20 sec</td><td>1 min</td><td>20 sec</td><td>Infinite</td><td>1 min</td><td>1 min</td><td>20 sec</td></tr><tr><td>Generalization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Memory</td><td>None</td><td></td><td></td><td>None</td><td>None</td><td>None</td><td></td><td></td></tr><tr><td>Model Size</td><td>2.7B</td><td>Unknown</td><td>13B</td><td>14B</td><td>Unknown</td><td>1.3B</td><td>Unknown</td><td>14B</td></tr></table>

Taealhallen lisal reLI-ol m tha orts h o n -eeo uor uiu the SelForc paral 0 Hun 0 wh distil esiv AR) set moal eesieisMatistii M)  ). Specifically,

Tounlcein -ori trea nd robus spatmemy erialwe repreenthetoresive model's memory as a set of highly compressed historical latents, in a similar spirit to (Zhang and Agrawala, 2025), which are encoded with both relative and absolute camera-pose and stored within the KV cache. This desig enables implicit 3D scene-content retrieval through viewpoint-aware context alignment, while the high compression ratio allows RELIC to retain the entire memory history with high computational eficiency. Our method stands in contrast to approaches that maintain spatial memory through recurrent model updates (Zhang e al 202; Po e al 202; Dalal et al, 025), which arefundamentally constrained by the capacity  the internal model state and often tailored to specificvisual domains. It also difrs from methods that introuce ele bnk w ariva uris    Xi ha eplcit 3D sce epreentais (Ma etl 2025 Ren etl 2025 Yu al202a Huang et al 2025c) which canuronucivnd eqotteneckebyrcrant. To overcome the lmitationsof the -second short-context trainng winow use in most prior wor (Lil, n  horel povianras changes, both of which are essential for world modeling, we fine-tune the teacher model to generate 20-second sequences. This extended temporal horizon enables supervision that enforces spatial and temporal consistency over significantly longer trajectories. However, perorming DMD over the full 20-second student rollout during distillation coputatinallyintractableToaddress this wtrduce replaye back-propagation te that enables memory-efficient diferentiation of student parameters by accumulating cached gradients of the DMD loss in a temporal block-wise fashion over the entire self-rollout. We curated 350 licensed Unreal Enginerendered scenes, encompassing approximately 1600 minutes of training video hi-qualiex atnacrajectasIrantl wecolle racorhat i  e deculecntrols, exibl ct cpositin, nd ong-ranmemory eivlOveralurrchitectura d enables both efcient training and inerence: RELIC achieves 16 FPS generation throughput on 4 H100 GPUs whil m nx w   s  aW ehaELC fo eneivoe pe future capabilities across numerous domains.

# 2 Related Works

Video World Models. Building interactive video world models requires the integration of diverse techniques, tressiv id ran   Huan l 02Ten Xi 0; He Che an e l 2024b,a;Son al 3; Genl 02, alongi breakhrous i seveal undamental chal, Zh Xi H Zh W u 0 n Lu   Xil 0re, 0 Lu 0ur  0R intria syste such as Mari-Game 2.0 (He al 025) Magica2 (Lab, 2025), RTF (World Labs, 202), nd G3(Ball havtrakabl rretowarhi al e stanalvan required to make world models truly practical.

LoViThei heeo s typil ssooiuesiv , mqualy  au aeuc FFs K uc rat ha a  i lat wi oelA   e p  ho uh . FrieaPX RoRu 24 SkyReels-v2 (Chee al., 025a), AR-Difuin (Sun al 025), StreamD (Kodara e al, 2025), Rolg tihar cs  Diffusion-Forcig (he  al, 2024a), whee latents within the modelin window may carry uneven noise luro irc So l , efecivey ervi  for meory,  senal conent r cnten wr Mhi Te exopeicpe thee  reicdte eius aan acheupu s J Valeski  al20Zhou al02a;Hu l 202Gaoel 02 Hoee uc e te ufr vH vatnh arhevoarco 8Honl 02FurAPT- i l0n LongLivYa l02extse training to minute lengths, enabling long-horizon generation. Data for World Exploration.Building a world exploration model fundamentally requires hig-quality data that a -or  n p wis a i Wa 0Che Fe 0He Lca ucSk ntolkoviu th ahiak e Althogh AAA game datasts provide clean and reliable actionvideo pairs (Feng et al., 2024; Li t al 2025; He yv et Zh e att Zhana,0; Heel ,  ptableolt u neralize  roader andmoreynami erments.In this ork, weserve hat when startio improve controllability while preserving generalization capability.

# 3 Data Curation for Interactive Video World Model

# 3.1 Data Overview

Tosuort aireal-ime on-horizoteracivio wor mode weonstruc arecalc videdataset rendered entirely n Ureal Engine UE). Thedataset is designed to provide i)diverse and cope

![](images/2.jpg)  
Figure 2 Dataset statistics visualization. Left:video duration distribution; Right: action distribution.

3Denvoments nd ) prececontroovercamertrajectoryOur fal crate dataset contains ve00- controllmajetorcollecro30hih-qualyens 3 scene sanigbothndor  ome, o seco.webtai more han 00miutes hi-deliy0p videequencesThe ip uratin sriu shown in figure 2a, with an average of ${ \sim } 7 5$ seconds and a maximum of up to 9 minutes. We derive action labels from ton  ehau ea training modelWe now summarize the data curation pipeline fltering strategy, and annotation proces below.

# 3.2 Data Processing and Filtering

We begin by curating 350 photorealistic static UE scenes covering a wide rangeof indoor and outdoor layouts. Human perators avigatc seeusg colsi-constrai camr cntroller tensure ysically plle movment. Durignavigation, werecor cntnuous -DoF cmrtrajecori—includig positions,intations, and the corresponding tmestamps, whic are thenrendered into high-quality720p vide sequences using theUErende. This syntheiapture pipeli s designed tdress hendamentalmitations existin eal-worlnava datasetsPrir wor has priariytraiedmodels nrl-world videocorpor, sch as V walkiataet n bas naviationvide Wanl 0;Li 0 However, the sur suffm tr limitations.1) Imbalanced action distributions: real-world videos are dominated by forward motion, wih very limied lateal rrotaionalmovement, whic webserve prevents models from learni divere ocentricmovement bavorsOveycupleactn beavirrl-worl videosotcntahly coupl actins, su u hiohakeiul hee  n L eviitatin pawintsor vidos are etu  previusysoatns ov consistent world generation. BycasrUE- allya o wencp ne v noete afrementionemitations. Moreover, we observe that raw UE-rendere videos may contai various types arac, su a ver-xposuremissi textures.Theservatins motivate werigpipeline decribe eow •Camera Motio. Trajectories exhibiting unnatural camera-motion patterns, such as excessive panning speed, abrupt rotations, or inconsistent velocity profiles introduced by human operators, are manually remove This ensures that the camera dynamics remai smooth and physically plausible, preventing the model from leaning unrealistic motion behavior.

![](images/3.jpg)

ViewpoinStabilitySegents exhibitig micro-itters scilatorydri r near-colsion paths reexce Filtering out these unstable videos prevents high-frequency noise from contaminating the training signal. Exposure nd Lighti robles.Each ciis revieweor xposurerelatenomales, includigoverpo . Rende Quality.Clips with missing textures, geometry popping, incomplete meshes, or other rendering defects ardiscardedEnsuringhigh-fidelity rendering is esential or hig-qualiy worl generationandvideo synthei.

# 3.3 Data Annotations

Camera Pose Annotations. For every rendered video clip $( f _ { 1 } , f _ { 2 } , . . . , f _ { T } )$ , we record precise, frame-aligned camera p rc ivom o rac provide  lgEThe include full 6-DoF camera poses, which consist of absolute camera positions $( \mathbf { P } _ { 1 } , \mathbf { P } _ { 2 } , . . . \mathbf { P } _ { T } )$ and world-to-camera camera orientations $( R _ { 1 } , R _ { 2 } , . . . , R _ { T } )$ . These complete camera annotations are essential ingredients for training interactive video wr model becaue hey llo ccramappin betweeinu acn controls nd the coepondi reson omhe gentn emet.urhermore ful-DoFm pose l uport efecivon-nxt spatial content retrieval, as shown in the experiment section. AcoaslhogUrlEng o ous -Docmtra hectolinte forwar pai, lokigup, rathr than cnti ame posTobrie his gap we cnvertcnt 6-DoF camera-pose sequences into per-frame action labels $\mathcal { A } \in \mathbb { R } ^ { 1 3 }$ , i.e., 13-DOF input-control format described in section 4.1.2. Our action-annotation pipeline derives per-frame actions $\boldsymbol { A } _ { t }$ by comparing the relative camera poses between adjacent frames $( f _ { t } , f _ { t + 1 } )$ in the video (see algorithm 1 for details). Specifically, we first compute the relative camera-translation vector $\Delta \mathbf { P } _ { t } ^ { c }$ in the camera's egocentric coordinate system using the camera positions in world coordinates $\mathbf { P } _ { t }$ and the world-to-camera rotation matrix $R _ { t }$ at time $t$ :

$$
\Delta \mathbf { P } _ { t } ^ { c } = R _ { t } ( \mathbf { P } _ { t + 1 } - \mathbf { P } _ { t } ) .
$$

The $x , \ y$ ,and $z$ components of $\Delta \mathbf { P } _ { t } ^ { c }$ constitute the translational 3-DOF component of our action labels. This vector represents the per-frame camera displacement, $i . e .$ , the instantaneous motion between two consecutive frames. Int  if usse in heStructure-rom-Motio ) liteatue.T ensur consistent behavir cross evomets we Require: Annotation JSON $\mathcal { I }$   

Ensure: Action sequence $\mathcal { A } _ { 1 : T }$ (including, movement action $\mathbf { A } ^ { \mathrm { m } }$ , and rotate action ${ \bf A } ^ { \mathrm { r } }$ )   
1:function ReADUEActIONS $( \mathcal { T } )$   
2: Load frames $\{ f _ { t } \} _ { t = 1 } ^ { T }$ , extract $\mathbf { P } _ { t }$ and $R _ { t }$   
3: $\Delta \mathbf { P } _ { t } ^ { \mathrm { c } } \gets R _ { t } ( \mathbf { P } _ { t + 1 } - \mathbf { P } _ { t } )$   
4: $\bar { d } \gets$ mean movement magnitude over non-zero $\Delta \mathbf { P } _ { t } ^ { \mathrm { c } }$   
5: for $t = 1$ to $T - 1$ do   
6: $\mathbf { A } _ { t } \gets \mathrm { I N F E R P A I R A C T I O N } ( \Delta \mathbf { P } _ { t } ^ { \mathrm { c } } , \bar { d } , t )$   
7: end for   
8: Prepend static action; return $\mathbf { A } _ { 1 : T }$   
9: end function   
10: function InferPAiRActIoN $( \Delta \mathbf { P } _ { t } ^ { \mathrm { c } } , \bar { d } , t )$   
11: // Translational motion in camera coordinates   
12: $( d _ { f } , d _ { s } , d _ { z } ) \gets \Delta \mathbf { P } _ { t } ^ { \mathrm { c } } / \bar { d }$   
13: Assign $\mathbf { A } ^ { \mathrm { m } } : \{ \mathbf { w } / \mathbf { s } , \mathbf { a } / \mathsf { d } , 1 \mathsf { i f t i n g \_ u p } / \mathsf { d o w n } \} \mathrm { f r o m } d _ { f } , d _ { s } , d _ { z }$   
14: // Camera rotation   
15: $\Delta R _ { t } ^ { c } \gets R _ { t + 1 } ( R _ { t } ) ^ { \top }$   
16: ( $\Delta \mathrm { y a w } , \Delta \mathrm { p i t c h } , \Delta \mathrm { r o l l } ) \gets \mathrm { F }$ Euler decomposition of $R _ { \mathrm { r e l } }$   
17: Assign Ar : {camera_up/down, left/right, roll_ccw/cw} from the angles   
18: Mark static if no action activated   
19: return action vector $( \mathbf { A } ^ { m } , \mathbf { A } ^ { r } )$   
20: end function noralzeislacemet vcr y hever spaceent manite the nt put  al ohaTza each $\Delta \mathbf { P } _ { t } ^ { c }$ as a relative motion ratio, indicating how the current displacement compares to the typical displacement within the clip. During inference, the average displacement magnitude (a coeficient $\gamma$ can be adjusted to control the overall motion scale of the generated videos. Similarly, we annotate 6-DoF rotational actions by computing the relative camera rotation $\Delta R _ { t } ^ { c }$ between adjacent frames:

$$
\Delta R _ { t } ^ { c } = R _ { t + 1 } ( R _ { t } ) ^ { T } ,
$$

We then convert $\Delta R _ { t } ^ { c }$ into yaw, pitch, and roll Euler angles following UE's intrinsic rotation convention. These three relative pose are near zero, we assign the action label at the current time step to static. Eme-ahaoeoee ure erentati abl emode ansistnappio pe-esteiseonrol coondiiensehe nvtTroveacollowpabilo e et al., 2025; Wang et al., 2025a) that are dominated by forward or dolly-in motions. Sa o eosurUa ca spa ulteie,maki   lobal ext t n a hort, -eel pt    us  OA,  D within the sampled video as the caption prompt for that training sample. Because text descriptions can sometimes contradict userntended action puts—for example, a usermay wish tove forwardn the environment while the text inadvertenty describes hecameramovi backwar r panning—we thus attpt t avoi ay cnraicton between text cnditining an usenput actions a erence tiTov ticteplc aevteal to supress desriptions o camera motion and ne-gaine objec motion, as we empirically und that common oel (Bail0ual Che tenvha suy w ptWsa raa o pre j  nexn generation.

# 3.4 Data Augmentation

Althoug urcurate camera trajectori delberately ecourage revisitatio past viewpoints, rando ent samplidur ai es ot arntee hat os sm e cnti suffnt al r he m t l Teo he-ea training clips. Specifically, as shown in figure 3 given a sampled training video segment of length $T$ , we uniformly sample a pivot index $t ^ { * }$ from the second half of the clip, $t ^ { * } \sim \mathcal { U } ( T / 2 , , T )$ , and construct a palindrome-style training sequence by concatenating the forward segment $f _ { 1 : t ^ { * } }$ with its time-reversed counterpart $f _ { t ^ { * } : ( 2 t ^ { * } - T ) }$ This augmentation produces a c o and to leverage long-horizon memory.

# 4 RELIC World Model

Our goal is to generate a video stream $( \hat { f } _ { 1 } , \hat { f } _ { 2 } , \dots , \hat { f } _ { T } )$ from an input RGB image $f _ { 0 }$ and a text description $\mathbf { C } _ { \mathrm { t e x t } }$ , given a stream of action-control inputs (e.g., keyboard or mouse commands) $( \mathcal { A } _ { 1 } , \mathcal { A } _ { 2 } , \ldots , \mathcal { A } _ { T } )$ . The model must preserve seveoi l taive inputs. F H0 hi  ge distill esteressiAR)idifinodeoinaliefiaeode h aloH y eal-me pe-ter ee nl; L, achieving both capabilities simultaneously—along with long-horizon memory—is substantially more challenging, as theetcnI prti oripa alaGPU memory  store,transer, and reason ver past tokens, which itrodue gificantFLOPs andmemory-banwi bottlenecks for real-time applications. Our RELmodeisdesie drss heehallengesbytruci everalkeyovatis itoheror. First, weedehe idinal idefuiteacermodeprt n-urati (20-sendrai w an AR vido-difusionmodeland devispatil-awamemory meanimhat efficiently ompresses historilvid toke ithWurbie  witarata du eteoy .e memory-eficint distillation ofa few-steAR mode even under theteachers 20-second lon context (sect4.). Lastly, we incorporateditionaluntime optimizations that bring the model inference torel-time (secti4.).

# 4.1 Action-Conditioned Teacher for Long Video Generation

# 4.1.1 Base Architecture

Our framework is built upon Wan-2.1 (Want al, 2025), a bidirectional video diffusion model pretrained for for xToa autcoder (ST-VAE) and diffusion transormers (DTs).The ST-VAE eploys a 3D causal architecture that map between high-dimensional video space and a lower-dimensional latent token space, achieving an $8 \times$ spatial compression ratio and a $4 \times$ temporal compression ratio. Its temporally causal design, together with an internal feature-cache mechanism, plays a crucial role in enabling our real-time video streaming capabilities. Th ido T ncde patii nptiiyer long i  eranormerblocksx deriptions are encoded with umT5 (Chung  al, 03) and integrated into the model vi cross-attention, whil denoising timestep embeddings are injected through a shared MLP module. Our RELIC adopts the 14B-parameter the model and always set its noise level to zero when it is concatenated with other noisy video latents.

![](images/4.jpg)  
Q-No cou

To enable preciekeyboard control, ng-duration videstreamig,and long-horizon spatiamemory—the three key caplr—unsuoh-qualy norm videoneraion fromstrea actin iputs. In the followi ctin, wedescur deshoic oteratiac-contro l tohebshtecurnd etendinraiono beyond the original 5-second limit.

# 4.1.2 Action Control

Action Space. We design a 13-degree action space $\mathcal { A } \in \mathbb { R } ^ { 1 3 }$ for RELIC, enabling full 6-DoF camera viewpoint control. Specifically, $\mathcal { A }$ specifies the magnitudes of six translational motions—Dolly In ↑, Dolly Out ↓, Truck Left Truck Right , Pedestal Up, and Pedestal Down—and six rotational motions—Tilt Up $\wedge$ , Tilt Down $\vee$ Pan Left $<$ , Pan Right $>$ , Roll Clockwise, and Roll Counter-Clockwise—between consecutive frames in tevideo, along with astaticacton [Stati] representingno camera movement Each action represente s a induced by user inputs. Action Condtioning .To enhane action following and spatial memory rerival, ur RELIC model incorporats -ao signals. Because the action vector $\mathcal { A } \in \mathbb { R } ^ { 1 3 }$ represents relative camera motion—i.e., the translational velocity $\Delta \mathbf { P } _ { t } ^ { c }$ and rotational velocity $\Delta R _ { t } ^ { c }$ between frames at times $t$ and $t + 1$ —we obtain the absolute camera poses $\mathbf { P } _ { t } \in \mathbb { R } ^ { 3 }$ and $R _ { t } \in S O ( 3 )$ by integrating the relative motions:

$$
\mathbf { P } _ { t } = \sum _ { i = 1 } ^ { t } ( R _ { i } ) ^ { T } \Delta \mathbf { P } _ { i } ^ { c } , \quad R _ { t } = \prod _ { i = 1 } ^ { t } \Delta R _ { i } ^ { c }
$$

Usitu-luccodiloheodeere ot reh nst servm taniatniu c qutsdeobruts, reensulotvca y predefined coefficient $\gamma$ , which modulates the magnitude of camera motion in the generated video. We embed both the relative actions $\mathcal { A }$ and the absolute camera poses $( \mathbf { P } _ { t } , R _ { t } )$ using two dedicated encoders and inject t eso piodoweMLulayehaoae control signal by $4 \times$ .Relative action embeddings are added directly to the latents after the self-attention layer, while absolute camera pose embeddings are added to the query $( Q )$ and key $( K )$ projections before the scaled dot-product attention (SDPA), with the value $( V )$ projections left unchanged. This design reflects the distinct computational roles of tla iheraamraensns control, whereas absolutecamera poses act as proxies for retrieving spatil content across viewpoints and time.

# 4.1.3 Long-Horizon Training

Therial Wan-.1de is preraie erate5-evideos 81frame a 16 FP.Alh re wori 0an las  that usite moel wi horn i sl asudent mode capable  streaming lngvideosequences, weargue that 5-second videotraini context nh is ypiyo b-te  eu T ct ps ro teoyTheerehe qen  abl c ode theparaeinar-u o raction-videdatase ctiotextend t eratioduration 0-seconds (3rames).In partcu 1- vids r,0eains, annally by - videsorothe,0itnscla pta n  plyR n xten heRota Embdings (RoPE) (Su e al., 2024) for the query and key tokens in each sel-attentionlayerof the DiT bloc.

# 4.2 Autoregressive Student for Real-Time Streaming a e  n i inference, tnable low-atency, real-timinteractiveexploration; and (3Long-horizon generation t rovide sufficiently extended temporal context for user navigation and discovery.

Tde ealeng al edetwew student videodiffusion model.Our student model is also built upon Wan2.1-4B (Wan  al., 025), but relac iv H a  e k full mathematical formulation of autoregressive video generation.

# 4.3 Memory

Most recent utoregressive AR) videomodel adopt causal atention over hort slidingwindow enablefent c in the KV cache duriAR iference.However, both the KV-cachememory footprint and the per-tokenatent cost scalelnearly with sequenceength.Consequently, as videolengthncrease durinAR generation, both computa and communication overhead grow proportionally, making real-time streaming inference particularly challenging.

To address this dilemma, we introduce a memory-compression mechanism, illustrated infigure .Given a newly denoised latent at index $i$ , our KV cache is composed of two branches: a rolling-window cache and a compressed log-horizon spatial memory cche.The olng-window cache stores uncompressed K tokens o recent videlatent between indices $i - w$ and $i$ , where $w$ denotes the sliding-window size. We maintain a small rolling window to prevent o memory The compressed long-horizon spatial memory cache, in contrast, stores spatially downsampled KV tokens for video latent from the beginning of the sequence up to index $i - w$ , following a predefined compression schedule. In practice, we adopt an empirically balanced configuration that interleaves spatial downsampling factors of $1 \times$ (no compression), $2 \times$ , and $4 \times$ across latents. This design is motivated by the observation that VAE-encoded latent spaces s eseul el  tuchdeynte e context. On average, our strategy reduces the total token count by approximately $4 \times$ (e.g., from 120K to 30K), which in turn yields a proportional $4 \times$ reduction in both KV-cache memory and attention FLOPs.

![](images/5.jpg)  
and diffusion forcing (mask shown on the right).

# 4.4 Distillation Framework

Wu de ucepoebs.elorci la e-im rollt r ai y pre new chunks conditioned on the model's own previously gnerated tokens (ie., usig generated history rather han ground-truth context). along-horizon student model usinteacher that enerates only hort video segments (e. 5second windows from the Wan-.1 teacher.However, thisdes eposes onidos intlosely cupl short sments, ihe taiuon -esabireeude-idu the teacher's long-video distribution.

# 4.4.1 ODE Initialization with Hybrid Forcing

Folowingrecent ARvideo distilatonframeworks (Yin  al, 2025; Huang al, 2025, weadopt the sameODEilzt ruraptheialaeualmoeanableheden loo paoTeiziae we nai ODE trajectories at the four denoising time steps used during distillation and inference. I itWheeuczti rata opa fast convergence. Specifically, given a training sequence of $B$ latent blocks, we divide them into two chunks: the first $B - K$ blocks contain clean, spatially compressed latents, while the remaining $K$ blocks contain uncompressed memory retrieval compared to either forcing method alone.

![](images/6.jpg)  
i . maps to accumulate parameter gradients. Parameters are updated once after the full replay.

# 4.4.2 Long-Video Distillation with Replayed Back-propagation

Seori Hun l 02poys he isruMatchiDstn MD) loss l 0, whiczheevediveeeteeatis ndheuis sampled timesteps $u$ .The gradient of the objective can be approximated by the difference between the real-data and generated-data score functions, $s ^ { \mathrm { d a t a } }$ and $s ^ { \mathrm { g e n } }$ :

$$
\nabla _ { \theta } \mathcal { L } _ { K L } \approx - \mathbb { E } _ { u } [ \int ( s ^ { \mathrm { d a t a } } ( \Psi ( G _ { \theta } ( \epsilon , c _ { \mathrm { t e x t } } ) , u ) - s ^ { \mathrm { g e n } } ( \Psi ( G _ { \theta } ( \epsilon , c _ { \mathrm { t e x t } } ) , u ) ) \frac { d G _ { \theta } ( \epsilon , c _ { \mathrm { t e x t } } ) } { d \theta } d \epsilon ] ,
$$

where $\Psi$ denotes the forward diffusion process, $\epsilon$ is Gaussian noise, and $G _ { \theta }$ is the student generator. Teet m   abnus c-u ca.ha ory idh,  iceab scenarios. Trian wureplaye bac ni at oeal h   esaatr in figure 6, we first generate the entire predicted sequence of $L$ video latents using $G _ { \theta }$ with autograd disabled, and compute the score-difference maps using frozen real and fine-tuned fake score models:

$$
\hat { \mathbf { x } } _ { 0 : L } = \mathrm { s t o p } \mathrm { - g r a d } ( G _ { \theta } ( \epsilon _ { 0 : L } ) ) ,
$$

$$
\Delta \hat { s } _ { 0 : L } = s ^ { \mathrm { d a t a } } ( \hat { \mathbf { x } } _ { 0 : L } ) - s ^ { \mathrm { g e n } } ( \hat { \mathbf { x } } _ { 0 : L } )
$$

Next, we replay the AR rollout block by block. For each block index $l$ , we re-run the student AR forward pass with aurnablcnditnenhe previousyneratcontext anthen back-ropagatee orrepondi score-difference map to update the gradients:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { K L } \approx \sum _ { l = 1 } ^ { L } - \Delta \hat { s } _ { l } \frac { \partial G _ { \boldsymbol { \theta } } } { \partial \boldsymbol { \theta } } .
$$

After processing block $i$ , its computation graph is immediately freed before moving to the next block. Parameters are pauht b still capturing gradients that reflect the full-length video distribution of the teacher.

# 4.5 Runtime Efficiency Optimization troput arlynd by GUmey bnwih nd U speweirtat ordiyW frs appl dukereu veheaanemy costs, e RMNorm RoP b, anmulainrlocel veuislntet n the cacen P8 EM3 ormat t halve emory usage and reduce memory anser ime durin erence.We also empoy FlashAttention v3 (Shah t al., 2024) with FP8 kernels to improve performance on NVIDIA Hopper GPUs. Fial  ba u

After minimizing memory and CU costs, we utilize parallization to shard computation and memory load across mulWeo  pralllz rat mhat usuroneic.; slat izve/ paralelis) whi -attentio perators ae parallelizoverttetion ea tesor parallelism)Weus NCCL Allto-All operations to switch tensor layouts between these two parallelization schemes. For example, when tanioq parallelensoparalettentin eai scat dimesion hr loeuces uneusnsor paraleli for computing.

# 5 Experiments

I against existing methods.

# 5.1 Implementation details.

Training a 20-second 1B base model can be challenging as the model needs to process more than 120K tokens for a li (FSDP) (Zhal., 023)t hardtraini bath, mode parameer, gradients nd ptiizatio states over GPUs; ()eeeu parallelism Li l 03 t ttehe quee;and usten pareliSi e T  qt  e ovailable computes.For the student model, our memory spatial compression confguration is mpirically set to $S = [ 1 , 4 , 2 , 4 , 4 , 4 , 2 , 4 , 4 , 2 , 4 , 4 , 4 , 2 , 4 , 4 , 2 , 4 ]$ to fit a 20-second context into the original pre-trained token context l nnW  th e frame $i$ is selected as $s _ { i } = S [ i$ mod $\mathsf { l e n } ( S ) ]$ .During distillation, we progressively increase the rollout length to keep utn o optimizedecoding-time eiciency by replacing theoriginal VAE with the same Tiny VAE used in MotionSream (Shin et al., 2025). Our final model is trained on 32 H100 GPUs, each with 80GB of memory. i . u bj onBac s Mot  egAethe ual, Quality, and then calculate the average score over them.   

<table><tr><td rowspan="2">Model</td><td colspan="3">Visual quality ↑</td><td colspan="2">Action accuracy (RPE ↓)</td></tr><tr><td>Average Score†</td><td>Image Quality</td><td>Aesthetic</td><td>Trans</td><td>Rot</td></tr><tr><td>Matrix-Game-2.0 (He et al., 2025)</td><td>0.7447</td><td>0.6551</td><td>0.4931</td><td>0.1122</td><td>1.48</td></tr><tr><td>Hunyuan-GameCraft (Li et al., 2025a)</td><td>0.7885</td><td>0.6737</td><td>0.5874</td><td>0.1149</td><td>1.23</td></tr><tr><td>RELIC (ours)</td><td>0.8015</td><td>0.6665</td><td>0.5967</td><td>0.0906</td><td>1.00</td></tr></table>

# 5.2 Capabilities Showcase

RELIC achieves high-qaliy, diverseand precisely cntrollable video gneraion whilemaintaini stronspia and styles, and we refer readers to the videos on our project page for full results. Diversity. RELIC generalizes far beyond real indoor or outdoor environments. Starting from a single initial frame, it domains (gure(a). Notably, the model naturally exhibits correct distance awareness—arawa eements move more slowly than nearby objects—and demonstrates strong 3D shape understanding as the camera moves around sbjects. Th rzapab abp  nil Long-horizon Memory. Our RELIC model enables robust spatial memory retrieval even under large camera moets,  o hs Thede  te ve pes on al   wehant exhprv ancnttosoum po truc pl eretanhna heuristics, or auxiliary hyper-networks. AdustablVelociy Becaue camra actions are rereent as contuus relativvelocis, users caneey control the exploration speed by adjusting the displacement coefficient $\lambda$ . As shown in figure 7(b), RELIC supports a eutaalvc hisy -alyo outputs. uoR e elb e l ha  o acabndtcn wha woThishdeoe users to explore the scenes in real time with precision and flexibility.

# 5.3 Quantitative Comparison

We construct a benchmark tet et of 220 images sourced from Adobe Stock.The set spans both realistic scenes inl r0 i eandoy parte ino gousor group, e valuate  baseemodels us pree aco scrit, resultingn 0 neratevidos per baselneThe utput duration is xe at  secons.r la baseom tspeis qualyd tcuye cpai t statehe baselines: Matrix-Game-2.0 (He et al., 2025) and Hunyuan-GameCraft (Li et al., 2025a).

![](images/7.jpg)  
support for complex multi-key control.

Vi qualiyWevaluate isal qualiy usi selec dimensns romVBen (Huan l., ), wi ruls sumarized inabl RL achieves the trongest overal performance among all compare baselines. Alu trai  8 olut   rablyHGameCra   hiai o 2.0 (He et al., 2025) across metrics.

![](images/8.jpg)  
translating left. Matrix-Game-2.0 drifts left instead of executing the commanded leftward rotation.

A T aligned to the canonical ground-truth camera trajectory using a $\mathrm { S i m } ( 3 )$ Umeyama alignment (Umeyama, 2002), which remove cal n cordinaterame differencesW then evaluate ranslationalanotational Relativ os (RPoo t hoe sl  a e os u e   n resulting in the lowest overall RPE.

# 5.4 Qualitative Comparison

Ac accraWe cpare heacuracy  acn cntrol gu.RLheres  the can ns most fithuly, produci motion that stays closes t the ntende rajectory For example, when applyiTi $\tt { U p } \wedge ,$ Mae    ul hil Hunyuan-GameCraft (Ll, 0) exhibits egligible vertial amea movement Similary, when applyi tuck Left $\gets$ , Hunyuan-GameCraft behaves more like Pan Left $<$ , and Matrix-Game-2.0 incorrectly remains static shifting the viewing angle correctly without artifacts. Mu  heiul  gll ba he  s e ho  qulivc meor urre ase uc  Hu-Gameraf i nMat-Game.0He oaHGC viotmove way nres, nd Mat-Gam0 quick oss he ontex heu a wher ur model consistently regenerates the previously observed content (also see figure 1). Comparison with Marble.We also qualitatively compare urmethod with Marble (WorldLabs, 2025b), a comercial h lsuls ntasvoheurtc n live n ablu.

# 6 Discussion and Conclusion

LimtationsOuryste exhibits everal imitaions.Firs, he eerate vides nstrate mite iy anrestric cenaiprimariyuerainatase ompomost cee enderrm the Unreal EngAddiinaly our ppra srugges t nera extremey n s—onhe scalee. Morver, e atamodeze eets orionmeorynmultiiv denoising steps sgnifcantlypacts ierence latenc undereourcconsraine settings. Nonetheless,we belve

![](images/9.jpg)  
l Game-2.0 (He et al., 2025), forget the bench on the right-hand side in this case quickly.

![](images/10.jpg)  
Fiure 10Qualitative cmparison with Marblefrom World Labs (World Labs, 20b). We cpare wit a coc introduces artifacts such as Gaussian floaters. Our method RELIC , instead, generates clean results.

Conlusion. In this work, we presented RL, an interactivevido world model that nables real-time ice a on-horion spatl mey ovirtal seploatoBynteratigtwh, spatl-away mechanim with the scalable Self-Forcingdistillation paradigm, RELI enables consistent world generation rom a singl mage, withoutelyng on explici geometri representationsOur method shows that integrating comressd hisle uihale dg nmeoye whi aveoindeiratTeiteulovatns alabladaptaleodationorgenera-purpose wor sulator wihpotentialaplications bAI and immersive virtual content creation.

# References oyioFu modeling:Visual details maerin atari.Advances in Neural Information Processing Systems, 37:5875758791, 2024.

, uy JkBruMarJ iü e Gouln HevI Proceedings of the IEEE conference on computer vision and pattern recognition, pages 36743683, 2018. Bh hZ   H technical report. arXiv preprint arXiv:2309.16609, 2023. Aeolynski roniKplsMar attcGiYkiv Jacrk-ol avi Juvak BuollB Dasagi Maxie Gazeu Char Gbadmosi, Woohyun Han, E Hirst,AshyaKachra, LuciKerey, Krista Kes, Ev n o HT   y J anHami T, DmViar, Luyu Wan r Welat WonKeya Xur Yew, NickYounVadiZubov, DougasEck, DumEran, Kora Kavulu, Demis Hassais ZouGha Ria Ha r IbosBoShnTce worldmodels,2025.https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/. Latent video prediction for visual representation learning, 2023. TBrook, Bil Pebs Hol Wil Dee ui uo LiJi avi S JeTaorTro L Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh.Video generation models as world simulators, 2024.https: //openai.com/research/video-generation-models-as-world-simulators. e H arXiv preprint arXiv:2411.00769, 2024. he ao uaRuee prediction meets full-sequence diffusion.Advances inNeural Information Processing Systems, 37:2408124125, 024a. Ghe i n Ju J huMinH ZhaSZee, Chencheng Ma, e al.Skyreel-v:Infinite-ength fl generative odelarXiv preprint arXiv:2504.130742025a. J u XieW ooiz o THsi XiXiv:6 h Xioa i u conference on computer vision and pattern recognition, pages 2418524198, 2024b. HyuWon hu oa onstat, Xav Gar Robets,  Ty, Shararaand OranFani mocu /30.9 semanticscholar.org/CorpusID:258187051. Jus uu,Mi Tao Xi RuiWadBi Bannho-JuiHiel $^ { + + }$ Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. pages 1770217711, 2025. Thati:Infi-orizon wor ean with eal-emovicontrorXi prerint arXiv:212.56 02 o Ji  hau Xindhe diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375, 2024. Y oTH  Xi Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025. Zhe Ge MaDeg Xing ai ZicoKoler nd HeMe oweiv. arXiv preprint arXiv:2505.13447, 2025. o yal Generative AI, pages 3939, 2025. David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3), 2018. XHheBaaui , Robert Henschel, Levon Khachatryan, Hayk Poghosyan, Danil Hayrapetyan, Vahram adevosyan, Zhangyang Wang Shan Navaardyan, nd HurehiConstentnaic,and extendable oideraoxtI Proceedings of the Computer Vision and Pattern Recognition Conference, pages 25682577, 2025. Ou arXiv preprint arXiv:2011.13922, 2020. u uo auo or A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. J Hu eHu u o uHu MiW HoZhou Z LuWi-Ma, ndMun. AIaeiiv17 JHua, un Zhou HsRabe leksoroko, Huan Lng XuaRen Tnhen Jun ao y ni Juai for 3d geometric perception. In NVIDIA Research Whitepapers, 2025a. HuJal, ixihou hMidMieoVirai interactive world models. arXiv preprint arXiv:2505.14357, 2025b. He T Wuu he Wa u,  Hu, u arXiv preprint arXiv:2506.04225, 2025c. e video diffusion. In Advances in neural information processing systems, 2025d. HHeu,T YXieLW dZ i unuXuHa uzu Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024. in.ACTraniGraph,42),July2023https://re-samri.r/ungraph/3gaussiansplati. Advances in Neural Information Processing Systems, 37:8983489868, 2024. HHokuhalex arXiv preprint arXiv:2507.03745, 2025. Wo Zj haRox Mnai hou J XioXiBoW J Hunyuanvideo:A systematicframework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. Je U pial 0 THH preprint arXiv:2504.21332, 2025. Dynamics Lab. Magica 2, 2025. https://blog. dynamicslab. ai/. Tu isi: 2025a. u  o Papers), pages 23912404, 2023. ZXioin ihi ha h Xu, iyi uk i et al. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025b. He uxXXiY u X adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025. preprint arXiv:2509.25161, 2025a. preprint arXiv:2509.25161, 2025b. R Generating interactable articulated objects from a single image. arXiv preprint arXiv:2507.05763, 2025. uoH e anWa L Gua al (ICRA), pages 1238612393. IEEE, 2025. a   H oT uT X Wo L at pages 20162029, 2025. e Toe, Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025. OpenAI. Gpt-5.1, 2024. https : //www. openai . com. Large language model. JacPrke-Holer,hilBal JakeBrueVavDsagiis HolheerrisKplaisexao , Gy Sculy, Jry ar Ji hi S e Jessi u Mic e Sul nyban Long Vad Mni, Har han Maxi Gaz Boni iFabio rdo, Luyu Wang, Lei Zhang FiBesse, Tm Har Aa Mikova Jne Wang, Jef Clune, Demis Hassab, Rai HadseAdri Bolon, StierSi Tim Rocktächel.Genie :A large-scale foundation world model, 204.https://deepmind.oogle/discover/blg/ genie-2-a-large-scale-foundation-world-model/. models. arXiv preprint arXiv:2309.00071, 2023. hBer u state-space video world models. arXiv preprint arXiv:2505.20171, 2025. Aolyakm Zohadronnd nda Sianeos, Bw hi hiM Chin-Yao Chuang, et al. Movie gen: A cast of media foundation models.arXiv preprint arXiv:2410.13720, 2024. Te ü Computer Vision and Pattern Recognition Conference, pages 61216132, 2025. robotic navigation. arXiv preprint arXiv:2504.19322, 2025. DavRuheJo HeekTS EmHoRoliferXierXiv:0.9 2024. J a a aib/2.C: 271098045. J video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025. s multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019. SnB hen ax oYiluRu Tedakndnc Hisyuii arXiv preprint arXiv:2502.06764, 2025.   
Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models, 2023.   
M Se Bo u o position embedding. Neurocomputing, 568:127063, 2024.   
nWn Ze A Recognition Conference, pages 73647373, 2025.   
HWorT eYuu u Ziu, H WXu uoH, preprint arXiv:2507.21809, 2025.   
n Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. Xio  i autonomous driving: A comprehensive survey. arXiv preprint arXiv:2502.10498, 2025.   
analysis and machine intelligence, 13(4):376380, 2002.   
a   i arXiv:2408.14837, 2024.   

Taa ol he i ei Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.   
JRhe, e aSpatialvid:A large-scale video dataset with spatial annotations.arXiv preprint arXiv:2509.09676, 202a.   
auto-regressive video diffusion models: A unified framework. arXiv preprint arXiv:2503.10704, 2025b.   
Tha Widee, uxn Li, Paul VicolShix ShaneG, Nicaee, Kevi wersky, Bee Ki iykJa Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.   
World Labs. Rtfm: A real-time frame model, 2025a. https: //www . worldlabs.ai/blog/rtfm.   
World Labs. Marble, 2025b. https://marble. worldlabs . ai/. Product site.   
World Labs. Generating worlds, 2025c. https://www.worldlabs.ai/blog/generating-worlds. Product site.   
H  uT e uo diffusion and 3d representation for consistent world modeling. arXiv preprint arXiv:2507.07982, 2025a.   
TWuoXu Z spatial memory. arXiv preprint arXiv:2506.05284, 2025b.   
l Computer Vision and Pattern Recognition Conference, pages 2177121782, 2025.   
Z world simulation with memory. arXiv preprint arXiv:2504.12369, 2025.   
X o diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 63226332, 2025.   
u zXieL, SHnkeLogReaivviierXiv:509.22

Yan: Foundational interactive video generation. arXiv preprint arXiv:2508.08601, 2025.   
mai  8   
recognition, pages 66136623, 2024b.   
T n g ZhaRichrZhanWire, reDuran hech and uHuaro w baesiiat Conference, pages 2296322974, 2025.   
X fromae.n rontheuerVisnd atteRecitnCnerenepag916926   
u    Xi W ZXi: Scene-consistent interactive long video generation with memory retrieval.arXiv preprint arXiv:2506.03141, 2025b.   
Ju  XiaanZi interactive videos. arXiv preprint arXiv:2501.08325, 2025c.   
detailed video description to comprehensive video understanding. arXiv preprint arXiv:2501.07888, 2025.   
co ZuvI European Conference on Computer Vision, pages 717733. Springer, 2022.   
Z preprint arXiv:2504.12626, 2025.   
J Wocetif iep Iieatt Conference, pages 2168521695, 2025a.   
Test-time training done right. arXiv preprint arXiv:2505.23884, 2025b.   
u Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025c.   
YhaondG RoVara ianLuohihiHaMi Xu, sWri, Ha o MylOttS Sy  a Xi:30   
D Zho u unYa un R o  WZee, a anXi ZhaT Conference, pages 73747384, 2025a.   
Xo Xi  n Her eriomodeanudanti IEEE/CVF International Conference on Computer Vision, 2025b.