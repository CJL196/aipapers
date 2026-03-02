# SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model

SkyReels Team Skywork AI

# ABSTRACT

SkyReels-V4 is a unified multi-modal video foundation model for joint videoaudio generation, inpainting, and editing. The model adopts a dual-stream Multimodal Diffusion Transformer (MMDiT) architecture, where one branch synthesizes video and the other generates temporally aligned audio, while sharing a powerful text encoder based on the Multimodal Large Language Models (MLLM). SkyReels-V4 accepts rich multi-modal instructions, including text, images, video clips, masks, and audio references. By combining the MLLM's multi-modal instruction-following capability with incontext learning in the video-branch MMDiT, the model can inject fine-grained visual guidance under complex conditioning, while the audio-branch MMDiT simultaneously leverages audio references to guide sound generation. On the video side, we adopt a channel-concatenation formulation that unifies a wide range of inpainting-style tasks—such as image-to-video, video extension, and video editing—under a single interface, and naturally extends to vision-referenced inpainting and editing via multi-modal prompts. SkyReels-V4 supports up to 1080p resolution, 32 FPS, and 15-second duration, enabling high-fidelity, multi-shot, cinema-level video generation with synchronized audio. To make such high-resolution, long-duration generation computationally feasible, we introduce an efficiency strategy: Joint generation of low-resolution full sequences and high-resolution keyframes, followed by dedicated super-resolution and frame interpolation models. To our knowledge, SkyReels-V4 is the first video foundation model that simultaneously supports multi-modal input, joint videoaudio generation, and a unified treatment of generation, inpainting, and editing, while maintaining strong efficiency and quality at cinematic resolutions and durations.

# 1 Introduction

From the earliest days o cinema, flmmakers haveunderstood that compelling storytellingdemands the seamless oerat  undhehror s proriai r in 9     a  p ve;ye  was ot t The iz Jolson vic with  os2 ta euymelivTh histoi lu coosiialontext, whildit conve poralytotial texture, nd arrivcntuyNee modality alone suffices—their synergy creates the immersive experiences that define modern media. Over the past year, the fieldof video generation has witnessed a decisive paradigm shift from unimodal synhesis toward joint audi-vidogeneration roprietary comercial ystems sch as eo-3.1 [], Sora-2 [2], Kling-2. [3], G4.[4]Sn [.[6av  a tolhatnativey roduc yncroizudioalongsidvisucntentThe stes marksian dar fromearlier text-to-video(T2V) orvideo-to-audio (V2A) pipelines, whichhandledonemodality a atime ande suffered from audio-visual asynchrony, lip-speech mismatches, and degraded unimodal quality.

In parallel, substantial progress has been made in multimodal-referenced video generation, where models accpt diverse conditoning inputs beyond text. Foristance, Vidu [7] pioneeredReference-to-Videogeneration, eabli coerent syntheis ro multiple referncmages. RunwayAleph [8] introduce state--he-art in-context vide eitig, performing a wide rangeof operatons—adding, removing, and transformig objects,generating arbitray scene angles, and modifying style and lighting—directyon input vidos. In theaudio-to-videodomain, systems suc as Omniuman-1/1.5 [9, 10], SkyReels-A3 [11, 12], KlinAvatar [13, 14], and Multitalk [15] have demostrated compelling talki-head syntheis andaudio-drivenimaton. Recently, Kli-Omi [16] emerge as thefrst oel to support bot mageandvideo references or ideogneration, y it remais imitedtoisualynthes wihut audiooutput.Alongside these developments, concurent works including Kling-3.0 [17], Seedance-2.0 [18], and Vidu-Q3 [19] have takenmeaningul steps toward bridging this gap, each integratng multimodal inputs wih joint vidud eat whi unode Neverhees, the stes ill horfullyev solution. Despittheseavance, no existing system smultaneouslyunifes multimodal inputs (text, images, videos, masks, and audio references), joint videoaudio generation, comprehensive inpainting, and editing capabilities within aigeframeworkCurrent stateo-the-art models remainfundamentallyfragmenteaudio-riven systems such Oiuman-1. n Mutital adopt hallousimehanims ( crs-attentionrghtwet pr that fail to fuly alig audio-visual representations, while multimodal-referenced models such as Klin-Omi fos excusivey nvisual conditiiwthou nativeaudi syntheisAlthourecent eforts—Kling3.0 Seedanc2.0, and Vidu-Q3—have taken meaningful steps toward joint videoaudio generation under multimodal inputs, none of sativaenT conditioned on arbitrary combinations of text, images, videos, masks and audio references. To adress thesemitations, we present SkyReel-V4, amult-modal videofoundationmodel that jointly gnerates R l- MulMoal fusTranoredeerancdediateienthihe oudeeratioThetworances harcotexncostaniate yronLL hat proie mul-modalunderstanding and instruction-followig across ext, images,vidos an audios.This shared MLLM bacoellows SkyReels-V4tocnditio iverseputs—including ext, ages, videos,and udiefern—n a unified, semantically coherent manner. To support a broad set of video manipulation tasks, we design the video branch around a channel concatenation folaat eativeets elasaipealytasks eratnoextensnnediepreaskutnditaldita th eeaihaterThinii perpeiyRee heergeneous orkfows withinsigle model, while the nderyi MLablevisally eerenc ipant: from a provided frame, or editing specific regions indicated by a mask. SkyRels-sdeeot oexibiliybu lsoore qualiyTheode sort vide u  3 n -   haul techiiu olns   ifnbsuaially m of direct $1 0 8 0 \mathrm { p }$ generation, we present a joint low-resolution / high-resolution keyframe generation, where the model f o ntl el anramnterolatioduls that onstrucmporalycnsistenth-reolutiviThis desable SkyReel-V4 toachieve surprisingly high generation spee even or long, high-resolution videos with synchronized audio, making it practical for real-world creative and production environments. T te bes u owegeSkyReels-s the  t orwe hatnesy ports (riumt skio ocpabilit positions SkyReels-V4 s veratifoundation modelor ext-eerationvido ceatin an i Extensivperetmnstrateuper en SkyReels-Vcmparecue athearhs.Ourdelhie ate-hert esultheArlAalysea [20hensvuva throug SkyReels-VABench reveals that SkyReels-V4 significantly outperforms proprietary commercial systems, with demonstrates obust peormancacros verultmoalconditionntasksicludineference-tovidomo-tovieotlvene toe osialu creation. In summary, our contributions are: •We introduce SkyReels-V4, a dual-stream MMDiT-based foundation model that jointly generates video and audio under multi-modal instruction and reference inputs. We propose a unifiedchannel-concatenation inpaintingframework for video, enabling image-to-video, video extension, video editing, and vision-referenced inpainting within a single architecture.

![](images/1.jpg)  

Figure 1: Overview of the proposed method.

•We design an efficiency scheme — Joint low-res / high-res keyframe generation with super-resolution and interpolation—that makes 1080p, 32 FPS, 15-second, multi-shot video generation with synchronized audio computationally feasible. •We demonstrate that SkyReels-V4 is, to our knowledge, the first model to unify multi-modal input, joint videoaudio generation, and generation/inpainting/editing tasks at cinematic quality and speed, setting a new baseline for multi-modal video foundation models.

# 2 Related Work

# 2.1 Video Generative Models

Diffusion models have transformed video generation, evolving from early $2 \mathrm { D } { + } \mathrm { 1 D }$ architectures like Video Diffusion Models [1] and AnimateDiff [22] to DiT-basedframeworks []. Sora [24] demonstrated the effeciveness f largecal training with spatiotemporal attention. While closed-source systems (Veo-3.1 [1], Kling-O1 [16], Sora- [2], Hailuo-2.3 [25], Gen-4.5 [4]) lead commercially, open-source models—CogVideoX [26], HunyuanVideo [27, 28], WAN-2.1/2.2 [29], SkyReels series [30, 31, 32, 3, 34], LTX [35, 36], MAGI-1 [37]—are rapidy narrowing the gap through data scaling and quality improvements.

# 2.2 Video-Audio Generative Models

Joint text-to-audio+video (T2AV) generation aims to synthesize synchronizedaudiovisual content from text.Comls e031 [], Sor- [2],K-3.[7 or pal t .Oe ape evolvefo couple U-Nets [38] t DiT-asmethodsapter-based AV-DT [39], expert-rhetra (MMDisCo [40], Universe-1 [41], and dual-stream architectures (Ovi [42], BridgeDiT [43], JavisDiT [44]) using cross-attention or flow matching—though these incur high computational costs. LTX-2 [36] proposes asymmetric streams or efciency. Unified single-towermodels like Apoll [5] pro audio-video tokens jointlyvia Omi-Full Attention, enabling multitask training (T2AV/TI2AV/TI2V) with tighter couplingDespite progress, ynchroized spee-video synthesis an completesoundscapes remainunderexplore, with precispati-temporal lignmet a open challenge.

# 3 Model Design

We preset SkyReel-4, aunifmul-odal videoundation modelor joint video-audigeneration, ipat, seamless integration of text, image,video, mask, and audio conditioning signals whilemaintaining computational eo olutns n uratsThvervemo hitecu how.

# 3.1 Dual-Stream MMDiT Architecture for Joint Video-Audio Generation

Ohc raan uausT Theralzo text-to-video model while the audiobranc is trainedfrom scratch with matching architectural specifications. Hybrid Dual-Stream and Single-Stream MMDiT Blocks. Following the MMDiT design, each transformer block prsesvio udio and extmodalit throug ahybriarchitecturehat blancesmodalitylgmet wh parameter efficiency. The initial $M$ layers employ a Dual-Stream design where video/audio and text tokens maintain separate parameters for adaptive layernormalization, QKV projections, and MLPs, but interact during joint selattention:

$$
\begin{array} { r l } & { \mathbf { Q } _ { v } , \mathbf { K } _ { v } , \mathbf { V } _ { v } = \mathbf { Q } \mathbf { K } \mathbf { V } _ { v } ( \mathrm { L a y e r N o r m } _ { v } ( \mathbf { x } _ { v } ) ) , } \\ & { \mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } = \mathbf { Q } \mathbf { K } \mathbf { V } _ { t } ( \mathrm { L a y e r N o r m } _ { t } ( \mathbf { x } _ { t } ) ) , } \\ & { \qquad \mathbf { x } _ { v } ^ { \prime } , \mathbf { x } _ { t } ^ { \prime } = \mathbf { A } \mathrm { t t e n t i o n } ( [ \mathbf { Q } _ { v } ; \mathbf { Q } _ { t } ] , [ \mathbf { K } _ { v } ; \mathbf { K } _ { t } ] , [ \mathbf { V } _ { v } ; \mathbf { V } _ { t } ] ) , } \end{array}
$$

where $\mathbf { x } _ { v }$ and $\mathbf { x } _ { t }$ denote video/audio and text token embeddings, respectively, and $[ \cdot ; \cdot ]$ represents concatenation. This design facilitates strong cross-modal alignment during early layers. The subsequent $N$ layers transition to a Singleh  anex   ha computational efficiency. This hybrid strategy achieves faster convergence than either pure approach. Reinorced Text Conditioning vi ross-Attention.To adress potential emanticdilution f text feature the sigl a hbloc i  alxt ossatte yeiaty self-attention:

$$
\mathbf { x } _ { v } ^ { \prime \prime } = \mathbf { x } _ { v } ^ { \prime } + \mathrm { A t t e n t i o n } ( \mathbf { Q } = \mathbf { x } _ { v } ^ { \prime } , \mathbf { K } = \mathbf { x } _ { t } , \mathbf { V } = \mathbf { x } _ { t } ) ,
$$

hid querhexbedin eorcextualnchrohout enerat s.   
This cross-attention mechanism is crucial for maintaining fine-grained semantic control in later model stages. BidirectionalAudo-VideoCross-Attention Tenablemporal syncronizationbetweemodalities,eachtransorme blicoporates paircss-attention lyersheudtreattend vidatures, and thevi reirocallytten tudiotursThis idrecioalmehanim exchange yncronizationcuesrouhouhe entire network depth:

$$
\begin{array} { r } { { \bf a } _ { i } ^ { \prime } = { \bf a } _ { i } + { \bf C } \mathrm { r o s s A t t n } ( { \bf Q } = { \bf a } _ { i } , { \bf K } = { \bf v } _ { i } , { \bf V } = { \bf v } _ { i } ) , } \\ { { \bf v } _ { i } ^ { \prime \prime } = { \bf v } _ { i } ^ { \prime } + { \bf C } \mathrm { r o s s A t t n } ( { \bf Q } = { \bf v } _ { i } ^ { \prime } , { \bf K } = { \bf a } _ { i } ^ { \prime } , { \bf V } = { \bf a } _ { i } ^ { \prime } ) , } \end{array}
$$

where ${ \bf a } _ { i }$ and $\mathbf { v } _ { i }$ are audio and video features at layer $i$ . The architectural symmetry ensures both modalities share the samlatonahe rociyen prtte from unimodal pretraining. TemoralAlget RoScalDespirhiteurl yheoalsolutis vi span 21 frames while audio latents contain 218 tokens $( 4 4 . 1 \mathrm { k H z } \times 5 \mathrm { s } )$ . To align these temporal scales, we apply Rotary Positional Embeddings (RoPE) to both modalities and scale the audio RoPE frequencies by $2 1 / 2 1 8 \approx 0 . 0 9 6 3 3$ tma heidaroralolui.This sues atudionvitkesattend temporally consistent correspondence. Shared Multi-Modal Text Encoder. We simplify prompt conditioning by employing a single frozen MLLM text eer applicombin propt thatcncatenates visual anacoustidesciptionsTheresultmulodal embeddings areindependentyconsumed by both audio and videobranches vel-attention and cross-attenion T  hy e uex

T $\mathbf { z } _ { v } ^ { 0 }$ and audio latent ${ \mathbf z } _ { a } ^ { 0 }$ we sample timestep $t \sim \mathcal { U } ( 0 , 1 )$ and construct noisy latents $\mathbf { z } _ { v } ^ { t } = t \mathbf { z } _ { v } ^ { 0 } \dot { + } ( 1 - t ) \mathbf { \epsilon } _ { v }$ and $\mathbf { z } _ { a } ^ { t } = t \mathbf { z } _ { a } ^ { \tilde { 0 } } + ( 1 - t ) \epsilon _ { a }$ , where $\epsilon _ { v } , \epsilon _ { a } \sim \mathcal { N } ( 0 , \mathbf { I } )$ .The model predicts the velocity field $\mathbf { v } _ { \theta }$ that pushes noise toward data:

$$
\mathcal { L } _ { \mathrm { f l o w } } = \mathbb { E } _ { t , z _ { v } ^ { 0 } , z _ { a } ^ { 0 } , \epsilon _ { v } , \epsilon _ { a } } \left[ \left\| \mathbf { v } _ { \theta } ^ { v } ( t , \mathbf { z } _ { v } ^ { t } , \mathbf { z } _ { a } ^ { t } , \mathbf { c } ) - ( \mathbf { z } _ { v } ^ { 0 } - \epsilon _ { v } ) \right\| ^ { 2 } + \left\| \mathbf { v } _ { \theta } ^ { a } ( t , \mathbf { z } _ { a } ^ { t } , \mathbf { z } _ { v } ^ { t } , \mathbf { c } ) - ( \mathbf { z } _ { a } ^ { 0 } - \epsilon _ { a } ) \right\| ^ { 2 } \right] ,
$$

wher denotes the conditioning information (multi-modal embeddings and optional spatial-temporal masks). The jiai jevcurge bothranelea nroni eat whiepecheepv modality-specific characteristics.

# 3.2 Unified Video Inpainting via Channel Concatenation

Tv la.T the channel dimension:

$$
{ \bf Z } _ { \mathrm { i n p u t } } = \mathrm { C o n c a t } ( { \bf V } , { \bf I } , { \bf M } ) ,
$$

where $\mathbf { V } \in \mathbb { R } ^ { T \times H \times W \times C }$ is the noisy video latent, $\mathbf { I } \in \mathbb { R } ^ { T \times H \times W \times C }$ contains VAE-encoded conditional frames (with $\mathbf { M } \in \mathbb { R } ^ { T \times H \times W \times 1 }$ spatiotemporal regions are conditions (value 1) versus regions to be generated (value 0). This formulation unifies multiple generation tasks through different mask configurations: Text-to-Video (T2V): $\mathbf M = \mathbf 0$ (all frames generated) Image-to-Video (I2V): $M _ { t = 0 } = 1 , M _ { t > 0 } = 0$ (first frame conditioned) Video Extension: $M _ { t < k } = 1 , M _ { t \geq k } = 0$ (first $k$ frames conditioned) •Start-End Frame Interpolation: $M _ { t = 0 } = M _ { t = T - 1 } = 1$ , others 0 Video Editing: $M _ { t , h , w } = 1$ for preserved regions, 0 for edited regions (arbitrary spatiotemporal masks) This unified formulation naturally accommodates both fixed foreground/background masks and dynamic per-fram editing masks, enabling precise control over spatial and temporal modifications. iintTh video modifcations whil maintaining temporal synchronization through the bidirectional cross-attention mechanism.

# 3.3 Multi-Modal In-Context Learning for Vision-Referenced Generation and Editing

Beyond tex anipaini masks, ourameworkuport i mulmol cnditning throg referenae and video clips, enabling complex vision-referenced generation tasks such as multi-identity video generation and identity-preserving video editing under multi-modal prompts. MuliMoal IstrFollwiiMLLM.Reeeiuut miaee jnt r with the text prompt through the MLLM text encoder to extract semantically enriched multi-modal embeddings. The MLLM'sollowpabilabhestacealetha l  extualti  aid  @ <dialogue>hello, how are you $\scriptscriptstyle \cdot < /$ dialogue> in the style of person $\mathbf { B }$ s $@$ video_1"). These multi-modal embeddings are consumed by both the video and audio branches. In-Context Visual Conditioning via Self-Attention. To provide explicit visual reference signals beyond semanic m AE n o . These condition latents $\mathbf { Z } _ { \mathrm { c o n d } }$ are prepended to the noisy video latents $\mathbf { Z } _ { \mathrm { v i d e o } }$ before self-attention:

$$
\mathbf { Z } _ { \mathrm { a t t n } } = [ \mathbf { Z } _ { \mathrm { c o n d } } ; \mathbf { Z } _ { \mathrm { v i d e o } } ] ,
$$

jhay when generating or editing video content. Temporal Positional Disambiguation via Offset 3D RoPE. To distinguish condition latents from noisy vide latents and organize multiple reference visuals, we employ 3D Rotary Positonal Embeddings with temporal index offsets. Coilaten recvnegativporalndi qtallyncieenisl bohea video frames:

$$
\mathrm { R o P E } _ { \mathrm { t e m p o r a l } } ( \mathbf { Z } _ { \mathrm { c o n d } , i } ) = \mathrm { R o P E } ( t = - N _ { \mathrm { c o n d } } + i ) , \quad \mathrm { R o P E } _ { \mathrm { t e m p o r a l } } ( \mathbf { Z } _ { \mathrm { v i d e o } , j } ) = \mathrm { R o P E } ( t = j ) ,
$$

where $N _ { \mathrm { c o n d } }$ is the total number of condition tokens and $i , j$ index the condition and video tokens respectively. Spatial Thi ofe-bas positnal ecoding provide efeciveductiv as o istuishin cndtining xt fo duskeuaatuayte u reference visuals of varying types (images, short clips, etc.). Audio Reference Conditioning. Similarly, audio references (e.g., speech samples, musical themes, ambient soundapepr-neits eranyul-o guidance from the MLLM with in-context visual patterns from the vido branch and audio patterns from audio references, the model achieves fine-grained control over both visual and acoustic generation.

# 3.4 Data Pipeline

Oa penst atcollec   tTh e handles three modalities—images, videos, and audio—to support multimodal model training.

# 3.4.1 Data Collection urrainda cri bot eal-worlanntheataacross reeodalagidos ndi

ReaorWcolleorppublbt h ataPublita cludeage LAION [6],Flickr [7], etc, videos (WebVi-10M[48], Kala-36M [49], OpeHnVid [50], tc., andaudio Emilia [51], AudioSet [52], VGGSound [53], SundNet [54], tc.. Our -ue licensed data encompasses authorized movies, TV series, short videos, and web series. Synthetic DataWe enerate synthetidat toaddress sparse scenarios and generation tasks inadequately ove y r-wor daWn tree y rmultigual ex netin, multingal spee nthe multimodal inpainting/editing tasks. Fo  n stu nthcveult  inlihi,gh J Korean Gern French, ec.Our nthetiage-ext dataincludes simple text renderin nd contet-awa xt . Toan spe rain nmultngual cvege weplymulte TTS moel  var ng. Wecurate iversetext corpor tensurehemode lear proncations beyon con haracter includire and uncommon scripts. FuliskpaaiatvlabrtaW thereorecnstru heatthrouhisticate pipeievolvigsual entatnmodels, mag/vide editing models, and controllable generation techniques.

# 3.4.2 Data Processing

Oa p iereatpe ddesi. Deplatp imea cba qualn, IQA , a   u bla us oourp-e ntt them against captions for fine-grained balancing. AuDatarosiThei ipeieclcato siain, qualyer coc  t asurat—e se, and singi—usig Qwen3-Omni [55]. ext, we perorm qualiy fteg basd on SNR, MOS score, cippig ra andudibanwihWusvoicactiviy detetin AD) tselcudio wi enc raios below0.. Fo nt dsu  ho y te 1nsFor e nsi ater we plWhisperran soke nd sconenaly w uniformly caption all audio using Qwen3-Omni. Video Data Processing. Video processing consists of four stages: preprocessing (segmentation and deduplication), iltering, balancing, and audio-video synchronization for videos with audio tracks. Prercessing Traditnal methods using PyDetect and TransNet-V2 [56] produce sene-cut clips that often lack o VLM yletcen et i inteplica e using VideoCLIP embeddings [57]. aley, and motion quality (camera stability, motion magnitude/speed, frame drops). BangToroveraini efncy balancedat alongtwodiensinscncetual iversiy ndo diveriyWetb vent  pee vitet  larlt ye ydy a or scene category. AVio  t s  ee ethe e udeeru bs o herst ame, an pl ud-visal coizt terdglyWedopt he widey SyncNet [58] mode which uses  ConvNet architecure o learn joint embedings betwee sound and mout ages, ti eidckdiciziaptedenils samples and produce scalar confidence and offset values. We retain only clips satisfying |offset| $\leq 3 \wedge$ confidence $> 1 . 5$ w  eoludecelFial teratudnviocptnstonides.

# 3.4.3 Captioning

W ort pt  uto pt concise descriptions of video content and audio information. Long captions offer comprehensive descriptions of emet, subjects, ght aosphere another uandetailsStructue captisfollow andriz decriptivrder with specil tokens todenote in-video text(<text></text>, sundeffects(<sfx></sfx>, speh content(<dialogue></dialogue>), singing content(<singing></singing>), and background music $\mathrm { < b g m > < / b g m > }$ . In fal tai a xluiyu u psTol u pots wih th rat, e prompt enhancer that reformats free-form input into the structured representation.

# 4 Training Strategy

We adopt a progressivemuli-stagetrainig paradig that systematicaly develops the mode's capabilities aros l e ae Viu JoaiiThaee spaalcncepts, tmporalynami, udiratonandmu-oallment stablean a epochs for each stage.

# 4.1 Video Pretrain

T anskcplexiyWebe wiex-m)raitabli soadtndinil concept learning, which we find significantly accelerates subsequent video training convergence. Sta 1:Text-to-Image Foundation.We rst train the T2I task at 256px resoluton usin 3 billion mages or 3 T x foundation for spatial composition and concept formation. Stage 2: Initial Video LearningW introduce text-to-video (T2V) generation while aintaining T2I training. At 26 resolution and 16 s, we train  1 billionages and 400 million videos or 3 epohs, wih vid durations raning rom 2 to 10 seconds.Training at lower resolution llows the model to more rapidly converge on otion dynamics and temporal coherence. length, and task complexity.   

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Resolution</td><td rowspan=1 colspan=1>Data Volume</td><td rowspan=1 colspan=1>Epochs</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Video Pretrain</td></tr><tr><td rowspan=1 colspan=1>T2I</td><td rowspan=1 colspan=1>Stage 1</td><td rowspan=1 colspan=1>256px</td><td rowspan=1 colspan=1>3B images</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2I + T2V</td><td rowspan=1 colspan=1>Stage 2</td><td rowspan=1 colspan=1>256px, 16fps, 2-10s</td><td rowspan=1 colspan=1>1B images / 400M videos</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2I + T2V + Inpaint(Image Inpaint, I2V, V2V, Edit)</td><td rowspan=1 colspan=1>Stage 3</td><td rowspan=1 colspan=1>256px, 16fps, 2-15s(Inpaint: 5% each)</td><td rowspan=1 colspan=1>1B images / 400M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Mixed Tasks(T2I, T2V, Inpaint)</td><td rowspan=1 colspan=1>Stage 4</td><td rowspan=1 colspan=1>256/480px, 16fps, 2-15s(Inpaint ratio unchanged)</td><td rowspan=1 colspan=1>100M images / 100M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Mixed Tasks(T2I, T2V, Inpaint)</td><td rowspan=1 colspan=1>Stage 5</td><td rowspan=1 colspan=1>480/720/1080px,16fps, 3-15s</td><td rowspan=1 colspan=1>50M images / 50M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Multi-modal Condition(Image/Video Ref: 20% each)(T2V: 60%)</td><td rowspan=1 colspan=1>Stage 6</td><td rowspan=1 colspan=1>480/720/1080px,16fps, 3-15s</td><td rowspan=1 colspan=1>20M images / 50M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=4>Audio Pretrain</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Audio Backbone</td><td rowspan=1 colspan=2>Pretrain        Variable length, up to 15s</td><td rowspan=1 colspan=1>Hundreds of thousands of hours</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Video-Audio Joint Training</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T2V + T2AV + T2A</td><td rowspan=1 colspan=1>Joint Pretrain</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>50% video data + T2A data</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Video-Audio Supervised Fine-tuning</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T2AV + Multi-modal</td><td rowspan=1 colspan=1>SFT Stage 1</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>5M videos (Multi-modal: 20%)</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2AV + Multi-modal</td><td rowspan=1 colspan=1>SFT Stage 2</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>1M curated videos</td><td rowspan=1 colspan=1>3</td></tr></table>

Ssc -- video (V2V), and video editing tasks, each comprising $5 \%$ of the training mix. This stage trains for 2 epochs with video durations extended to 215 seconds, enabling the model to learn spatial and temporal inpainting capabilities. Stage 4: Mixed Resolution Scaling. We employ mixed-resolution training at $2 5 6 \mathrm { p x }$ and $4 8 0 \mathrm { p x }$ , maintaining 16 fps and21 second durations. Training on 100 millon mages and 100millon videos, we keep thenpainting taskra unchanged, allowing the model to gradually adapt to higher resolution generation. Stage 5: High Resolution Training. We further scale to mixed resolutions of $4 8 0 \mathrm { p x }$ $7 2 0 \mathrm { p x }$ , and $1 0 8 0 \mathrm { p x }$ at 16 fps, w viuratins  osThis a u50millags nd 0mll ideubstantially the model's high-resolution generation quality. Stage 6:Multi-modal Condition Pretrain. We introduce image reference and video reference conditioning for both generation and inpainting tasks, comprising $20 \%$ of the training data each, with the remaining $60 \%$ dedicated to T2V. This stage trains on 20 million mages and 50 million videos, equipping the model with exible multiodal conditioning capabilities.

# 4.2 Audio Pretrain

Theaudio backbone is pretrainefromscratcon hundredo thousands o hours primariy speech dat wth prnbcerosuaualvbra nteThe audio enables the model to generate consistent audio that respects speaker traits such as pitch and emotion.

# 4.3 Video-Audio Joint Training

Foe e eai ja ane skexiTV-TAVex-T ha o t ideo pretra data TAV jin traiig whilincorporati TA dat onable nchronizaudivisual generation.

# 4.4 Video-Audio Supervised Fine-tuning

In te a T auexcsve  ja atai miles w mu condition support (image, video, and audio), which comprises $20 \%$ of the data. We conclude with a final fine-uning ste onmilmanualcurated h-qualiyvids, rthereineraton qualty, motn chere and audio-visual alignment.

![](images/2.jpg)  
base model. KF demotes the key frames latent of our base model.

# 4.5 Video Super-Resolution and Frame Interpolation (Refiner)

Tourthe ehance visual quality nd temporal smoothness  enerate videos, we introducea dedicated Ref hajo pesupeeoateolat  oiThi aritecuperate heutputtheasulmodavieatmode everagoh loreo u increasing temporal resolution. Architecture and Design. We initialize the Refiner weights from the pre-trained video generation model toensure shunualx b heil task durig pos-raii whe he basemodel larns  ltneusy redic alrame t owreolut and kefam latents omhe base model Finaly, the combe ltents reconcatenate wih high-resolutio latents along the channel dimension as input to the DiT model. Tu k h o  l  hee reinement and those that should remain unchangedThis desig enables the Refiner to handle both uncondiioal super-resolution and conditional inpainting across multiple modalities. Computational Efficiency. To address the computational overhead imposed by long temporal contexts and highreolution inputs, weadopt Video Sparse Atention (SA) [59], a trainable sparse attention mechanism deed for video diffusion transformers.VSA employs a hierarchical two-stage approach: a coarse stage that aggregates sl  n attentn whilmantainihardwarefcin troghblock-parsyouts cmpatible wih moderG krels. By exploitigspatio-temporal redundancy in alearnabl manner, VSA enablesus t reduceattention computainal cost by approximately $3 \times$ while preserving generation quality, making it practical to process high-resolution video sequences during both training and inference. Trai Datan Coguratio.ordatacnstrutioncramillon -qualiyiocis v the oe's pablyoreraesaldetsTheasor hebsmoemu trainable throughout the training process, following the flow matching paradigm.

# 5 Model Performance

We evaluate model performance on a public arena leaderboard o assess overall user preference i an open-ended setting. Beyond this, we conduct comprehensive human assessments spanning five key dimensions: Instruction FollowigAudi-Visual ynchrozaton Visual QualityMotio QualityandAudio Qualiy, providingeaid alFreerualti r  ouevab reference-guided video synthesis. We showcase representative examples of these applications in Appendix A.

# 5.1 Artificial Analysis Arena

Aral Analyss [20] is  widely recnizebencmarking platform o evaluatignerative modes acros e and vido generation domains.The platform operates an open arena where models are scored by the public, with Elo score alculatefrom pairwicparisons eec user preerences.e evaluate urmoden heArta AnalysisVideArenaspecicallyn th text--ideo withaudioeneatn trackwhicsdesigeasses he qlijo iyntheOure earkaiotabexteal baseliclud., Kling 3.0, grok-imagine-video, Sora-2, Vidu-Q3, Wan 2.6, etc. Results: Our model ranks second on the leaderboard (as of 2026-02-5) among all participating systems (Figur 3).   
demonstrating strong and competitive audiovisual generation quality as evaluated by public user preferences.

![](images/3.jpg)  
Figure 3:Artificial Analysis Text-to-Video with Audio Arena Leaderboard. Our model ranks second among all competing baselines including Veo 3.1, grok-imagine-vide, Sora-2, Vidu-Q3, Wan 2.6 and etc.

# 5.2 Human Assessments

To comprehensively assess the joint video-audiogeneration capabilities, we introduce SkyReels-VABench, a novel human evaluation benchmark designed to evaluate state-of-the-art text-to-video+audio models in the market.

# 5.2.1 Benchmark Design

SkyRees-VABench extends our previous SkyReels-Bench [32] by incorporating comprehensive audio dimensions and multi-shot video scenarios. The benchmark comprises $^ { 2 0 0 0 + }$ carefully curated prompts spanning diverse content ccvetiynt The prompts are designed to test models across varying complexity levels, from single-hot scenarios to complex nulti-shot sequences with sophisticated audio requirements. Language Coverage: The benchmark includes prompts in multiple languages, with particular emphasis on Chinese and English to assess cross-lingual generation capabilities. Dvrt oajalt (indoor, outdoor, natural, urban), and temporal dynamics (static, slow-motion, fast-action sequences). AuoComplexity:The bencmark test varius udiomodlitncluding spe monologuedialogue, naaion, e sy,e eal mean   b (various genres and emotional tones).

# 5.2.2 Evaluation Metrics

Our evaluation framework encompasses five primary dimensions:

Table 2: Comprehensive Evaluation Dimensions for Audio-Visual Generation   

<table><tr><td>Dimension</td><td>Sub-dimension</td><td>Evaluation Criteria</td></tr><tr><td rowspan="2">Instruction Follow- ing</td><td>Video Instruction Following Subject description Subject interaction Camera movement</td><td>Accurate representation of subjects, attributes, and appearances Correct execution of actions, interactions, and motion dynamics Proper execution of camera operations (pan, tilt, zoom, dolly)</td></tr><tr><td>Style and aesthetics Multi-shot consistency Audio Instruction Following Semantic adherence</td><td>Adherence to visual styles, color palettes, and artistic directions Correct shot transitions, cross-shot coherence, and reference accuracy Fidelity to audio content and characteristics</td></tr><tr><td>Audio-Visual Syn-</td><td>Lip-sync accuracy Sound effect alignment Atmospheric matching</td><td>accuracy Precise speech-mouth synchronization and correct speaker identification Temporal correspondence between visual events and sound effects Coherence between BGM, scene atmosphere, and emotional tone</td></tr><tr><td>Visual Quality</td><td>Visual clarity Color accuracy Compositional quality Structural integrity</td><td>Sharpness, definition, and resolution Natural color balance and saturation without distortion Aesthetic composition, framing, and visual balance Absence of visual artifacts and corruptions</td></tr><tr><td>Motion Quality</td><td>Physical plausibility Motion fluidity Motion stability Temporal consistency Motion vividness</td><td>Adherence to physical laws (gravity, inertia, momentum) Smooth transitions without abrupt discontinuities Absence of jittering, deformation, and flickering Consistency of dynamic elements across frames Action, camera, atmospheric, and emotional expressiveness</td></tr><tr><td>Audio Quality</td><td>Absence of artifacts Spatial soundstage Timbre realism Signal clarity Dynamic range</td><td>No clipping, truncation, distortion, or glitches Appropriate stereo imaging and spatial rendering Natural and realistic tonal qualities Clean audio with appropriate signal-to-noise ratio Appropriate audio level variation without compression artifacts</td></tr></table>

# 5.2.3 Evaluation Methodology

Wp ul-ealua protdc y pan 0 posialvalat wi bai video production, audio engineering, and content creation: Absolute Scoring: Evaluators rate each dimension using a 5-point Likert scale ( $1 =$ Extremely Dissatisfied, $2 =$ Dissatisfied, $3 =$ Neutral, $4 = \mathbb { S }$ Satisfied, $5 =$ Extremely Satisfied), enabling standardized performance comparison across models.

![](images/4.jpg)  
Figure 4:Absolute scoring results (5-point Likert scale comparing SkyReels V4 against baselines. Higher score indicate better performance.

Good-Same-Bad (GSB) Comparison: Pairwise comparisons between model outputs enable more granular quality iaor po alto tutset de  ass t "Good" (clearly better), "Same" (comparable quality), or "Bad" (clearly worse).

# 5.2.4 Baselines

We compareour model against state-of-the-art video-audio generation systems, including • Veo 3.1 (Google) •Kling 2.6 (Kuaishou)   
Seedance 1.5 Pro (ByteDance) •Wan 2.6 (Alibaba)

# 5.2.5 Results

AWae e   a dimension on  5-point Likert scale. s hown i Figure , kyRees V achieves the highestoverall averag re a al cpetinmodels.The perdimensinbreakdow eveals uanc pictur SkyReels V rens i demonsrates particulary rongperoranc Prompt Followin and Motion ualiy.For Visual Qualiy, SkyReels V4 peorms comparably  the strongest competing models. While SkyReels V4 shows relatively modest advantages inAudVisualyncronizationanAudio Qualiy it nonetheless aintais stateo-the-ar perormancehee dimensions as well, underscoring its overall competitiveness across the full evaluation spectrum. Good-Same-Bad (GSB) Comparison. To further validate our model's superiority, we conduct pairwise GSB cprios betwe kyReels n eac baseli.s illustat nFgurSkyReel Vcnstent ve higr rortio  Go"t gainst pegodel  ter vel qualiyThe per-imesi GSB rul o pairwis cparion e preente dmstrain that SkyReels 4utpeors Kling ., Snc 1.5 Pro, Veo 3.1, and Wan 2.6 across the majority of evaluation dimensions.

# 6 Conclusion

In this work, we present SkyReels-V4, a unifed multi-modal video foundation model that jointly generates video hu MMDiT design with  shared MLLM-based text encoder, SkyReels-V4 accepts rich multi-modal conditioning inputs— inext ages ideops,asks, ndudie—an rod hig-deliy ncoizidi ouu acea qaliy (p o100p, 3 FP, 15 ecnds). To sport dive vide ceain tasks, wep caai  ep semaskcogurations whieveragpora-cnaenatio exiblnorporali-modale such a age, videocips,andaudiAdditnalyour joint ow-resoltin/high-resolutikeyframegan strategy enables efficient generation at scale.

![](images/5.jpg)  
GSoverll qualitycparisonSkyRee Vsl baselines.Eac bar hows the proportion  Good and "Bad" ratings.

![](images/6.jpg)  
(a) SkyReels V4 vs. Kling 2.6

![](images/7.jpg)  
(b) SkyReels V4 vs. Seedance 1.5 Pro

![](images/8.jpg)  

Fiure : GSB comparisn results. Top: Overall quality comparison between SkyReels V4 and al baselines. Bottom: Per-dimension GSB comparison across five evaluation dimensions: Prompt Following, Audio-Visual Synchronization, Visual Quality, Motion Quality, and Audio Quality.

![](images/9.jpg)

E vas vala kyReels-'efeiO rtialyiAr ue t o systems in the text-to-video-with-audiotrack.On ur propose SkyRees-VABench, SkyReels-V4 chievs the h veralaver score wi partiulary rog peoranc Propt Folowi n Moti Qualy whil mtia-her paros  ltsirriso rea SkyReels-V4 consistently outperforms competing baseline systems. Tohe be uow SkyReelheo nulputs, j ogneration, n eeratinpaintiedi apabilit at ea qualy n alWeoe his w serves as a foundation for future research in multi-modal video generation systems.

# 7 Contributors

T their primary contribution roles: Project Sponsor: Yahui Zhou •Project Leader: Guibin Chen (guibin.chen@kunlun-inc.com)   

Contributors: Infrastructure: Hao Zhang, Zhiheng Xu, Weiming Xiong, Yuzhe Jin, Zhuangzhuang Liu, Wenyan Liu Data & Video Understanding: Mingyuan Fan, Yiming Wang, Mingshan Chang, Jiahua Wang, Yuqiang Xie, Peng Zhao, Xuanyue Zhong, Fuxiang Zhang, Peiyu Wang Video Model Training: Dixuan Lin, Jiangping Yang, Sheng Chen, Chaofeng Ao, Yunjie Yu, Jujie He, Yuhao Feng, Shiwen Tu, Chaojie Wang, Rui Yan, Wei Shen, Jingchen Wu, Weikai Xu Audio Model Training: Zhengcong Fei, Zheng Chen, Tuanhui Li, Baoxuan Gu, Kaifei Wang, Xuchen Song, Max W. Y. Lam, Chien-Hung Liu Multi-modal Training: Youqiang Zhang, Debang Li, Nuo Pang, Yikun Dou, Xiaopeng Sun, Jingtao Xu, Binjie Mao, Liang Zeng, Haoxiang Guo Model Evaluation: Binglu Zhang, Yu Shen, Tianhui Xiong, Bin Peng

# References

[1] DeepMind. Veo-3.1. Oct. 15, 2025. URL: https://aistudio.google. com/models/veo-3.   
[2] OpenAI. Sora-2. Oct. 15, 2025. URL: https://openai. com/index/sora-2/.   
[3] KlingAI. kling-2.6. Dec. 3, 2025. URL: https://app. klingai. com/global/.   
[4] Runwayml. Gen-4.5.Dec. 1, 2025. URL: https://runwayml.com/research/introducing-runway-gen4.5.   
[5] Team Seedance, Heyi Chen, Siyan Chen, et al. Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model. 2025. arXiv: 2512.13507 [cs.CV]. URL: https://arxiv.org/abs/2512.13507.   
[6] Wan. Wan-2.6. Dec. 12, 2025. URL: https://wan.video/introduction/wan2.6.   
[7] Vidu. Vidu-Q2. Sept. 25, 2025. URL: https: //www.vidu. com/.   
[8] runwayml. runway-aleph. July 25, 2025. URL: https://runwayml.com/research/introducing-runwayaleph.   
[9] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. OmniHuman-1: Rethinking the ScalingUp of One-Stage Conditioned Human Animation Models. 2025. arXiv: 2502. 01061 [cs. CV]. URL: https: //arxiv.org/abs/2502.01061.   
[10] Jianwen Jiang, Weihong Zeng, Zerong Zheng, Jiaqi Yang, Chao Liang, Wang Liao, Han Liang, Yuan Zhang, and Mingyuan Gao. OmniHuman-1.5: Instilling an Active Mind in Avatars via Cognitive Simulation. 2025. arXiv: 2508.19209 [cs.CV].URL: https://arxiv.org/abs/2508.19209.   
[11] SkyReels. SkyReelsA3. Aug. 12, 2025. URL: https://skyworkai.github.io/skyreels-a3.github.io/.   
[12] Zhengcong Fei, Hao Jiang, Di Qiu, Baoxuan Gu, Youqiang Zhang, Jiahua Wang, Jialin Bai, Debang Li, Mingyuan Fan Guibin Chen, e al.Skyreels-audiOmniaudio-conditinetalkig portraits invideodiffusiontransormers". In: arXiv preprint arXiv:2506.00830 (2025).   
[13] Yikang Ding, Jiwen Liu, Wenyuan Zhang, et al. Kling-Avatar: Grounding Multimodal Instructions for Cascaded Long-Duration Avatar Animation Synthesis. 2025. arXiv: 2509.09595 [cs. CV]. URL: https: //arxiv.org/ abs/2509.09595.   
[14] Ki 1eam, Jau Cnen, riang Dng, et al. KungAvaar 2.U 1ecnnal Repor. 202. arXv: 2512. 13313 [cs.CV].URL: https://arxiv.org/abs/2512.13313.   
[15] Zhe Kong, Feng Gao, Yong Zhang, Zhuoliang Kang, Xiaoming Wei, Xunliang Cai, Guanying Chen, and Wenan Luo."Let Them Talk:Audio-Driven Multi-Person Conversational Video Generation". In: arXiv preprint arXiv:2505.22647 (2025).   
[16] Kling Team, Jialu Chen, Yuanzheng Ci, et al. Kling-Omni Technical Report. 2025. arXiv: 2512.16776 [cs.CV]. URL: https://arxiv.org/abs/2512.16776.   
[17] KlingAI. kling-3.0. Feb. 6, 2026. URL: https : //app. klingai . com/global/.   
[18] ByteDance. Seedance-2.0. Feb. 12, 2026. URL: https://seed. bytedance. com/en/seedance2_0.   
[19] Vidu. Vidu-Q3. Jan. 30, 2026. URL: https: //www. vidu. com/.   
[20] Artificial Analysis. AI Model and API Providers Analysis. https: / /artificialanalysis . ai/.   
[21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video Diffusion Models. 2022. arXiv: 2204. 03458 [cs .CV]. URL: https: //arxiv. org/abs/2204. 03458.   
[22] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning 2024. arXiv: 2307.04725 [cs.CV]. URL: https://arxiv.org/abs/2307.04725.   
[23] William Peebles and Saining Xie. Scalable Diffusion Models with Transformers. 2023. arXiv: 2212 . 09748 [cs.CV].URL: https://arxiv.org/abs/2212.09748.   
[24] Tim Brooks, Bill Peebles, Connor Holmes, et al. "Video generation models as world simulators". In: (2024). URL: https://openai.com/research/video-generation-models-as-world-simulators.   
[25] Hailuo. Hailuo-2.3. Oct. 28, 2025. URL: https://www.minimax.io/news/minimax-hailuo-23.   
[26] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, et al. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. 2025. arXiv: 2408.06072 [cs.CV]. URL: https://arxiv.org/abs/2408.06072.   
[27] Weijie Kong, Qi Tian, Zijian Zhang, et al. HunyuanVideo: A Systematic Framework For Large Video Generative Models. 2025. arXiv: 2412.03603 [cs.CV]. URL: https://arxiv.org/abs/2412.03603.   
[28] Tencent Hunyuan Foundation Model Team. HunyuanVideo 1.5 Technical Report. 2025. arXiv: 2511. 18870 [cs.CV].URL: https://arxiv.org/abs/2511.18870.   
[29] Team Wan, Ang Wang, Baole Ai, et al. Wan: Open and Advanced Large-Scale Video Generative Models. 2025. arXiv: 2503.20314 [cs.CV].URL: https://arxiv.org/abs/2503.20314.   
[30] Di Qiu, Zhengcong Fei, Rui Wang, Jialin Bai, Changqian Yu, Mingyuan Fan, Guibin Chen, and Xiang Wen. SkyReels-A1: Expressive Portrait Animation in Video Diffusion Transformers. 2025. arXiv: 2502 . 10841 [cs.CV].URL: https://arxiv.org/abs/2502.10841.   
[31] SkyReels-AI. Skyreels V1: Human-Centric Video Foundation Model. https : / / github. com/SkyworkAI/ SkyReels-V1. 2025.   
[32] Guibin Chen, Dixuan Lin, Jiangping Yang, et al. SkyReels-V2: Infinite-length Film Generative Model. 2025. arXiv: 2504.13074 [cs.CV]. URL: https://arxiv.org/abs/2504.13074.   
[33] Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Ch, Yang Li, t al. "SkyReels-A: Compose Anything in Video Diffusion Transormers". In: arXiv preprint arXiv:2504.02436 (2025).   
[34] Debang Li, Zhengcong Fei, Tuanhui Li, et al. SkyReels-V3 TechniqueReport. 2026. arXiv: 2601.17323 [cs.CV]. URL:https://arxiv.org/abs/2601.17323.   
[35] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, et al. LTX-Video: Realtime Video Latent Diffusion. 2024. arXiv: 2501.00103 [cs.CV].URL: https://arxiv.org/abs/2501.00103.   
[36] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, et al. LTX-2: Efficient Joint Audio-Visual Foundation Model. 2026.arXiv: 2601.03233 [cs.CV].URL: https://arxiv.org/abs/2601.03233.   
[37] Sand. ai, Hansi Teng, Hongyu Jia, et al. MAGI-1: Autoregressive Video Generation at Scale. 2025. arXiv: 2505.13211 [cs.CV].URL: https://arxiv.org/abs/2505.13211.   
[38] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation. 2023. arXiv: 2212.09478 [cs.CV]. URL: https://arxiv.org/abs/2212.09478.   
[39] Kai Wang, Shijian Deng, Jing Shi, Dimitrios Hatzinakos, and Yapeng Tian. AV-DiT: Efficient Audio-Visual Diffusion Transformer for Joint Audio and Video Generation. 2024. arXiv: 2406.07686 [cs.CV]. URL: https: //arxiv.org/abs/2406.07686.   
[40] Akio Hayakawa, Masato Ishi, Takashi Shibuya, and Yuki Mitsufuji. MMDisCo: Multi-Modal DiscriminatorGuided Cooperative Diffusion for Joint Audio and Video Generation. 2025. arXiv: 2405.17842 [cs. CV]. URL: https://arxiv.org/abs/2405.17842.   
[41] Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. UniVerse-1: Unified Audio-Video Generation via Stitching of Experts. 2025. arXiv: 2509.06155 [cs.CV].URL: https://arxiv.org/abs/2509.06155.   
[42] Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation. 2025. arXiv: 2510.01284 [cs.MM]. URL: https://arxiv.org/abs/2510.01284.   
[43] Kaisi Guan, Xihua Wang, Zhengfeng Lai, Xin Cheng, Peng Zhang, XiaoJiang Liu, Ruihua Song, and Meng Cao. Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction. 2025. arXiv: 2510.03117 [cs.CV]. URL: https://arxiv.org/abs/2510.03117.   
[44] Kai Liu, Wei Li, Lai Chen, et al. JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical SpatioTemporal Prior Synchronization. 2025. arXiv: 2503.23377 [cs.CV]. URL: https://arxiv.org/abs/2503. 23377.   
[45] Jun Wang, Chunyu Qiang, Yuxin Guo, Yiran Wang, Xijuan Zeng, and Feng Deng. Apollo: Unified Multi-Task Audio-Video Joint Generation. 2026. arXiv: 2601.04151 [cs.CV]. URL: https : //arxiv. org/abs/2601. 04151.   
[46] LAION. Large-scale Artificial Intelligence Open Network. 2021. URL: https: //1aion. ai/.   
[47] hlky.Flickr. [https://huggingface.co/datasets/bigdata-pw/Flickr](https://huggingface. co/datasets/bigdata-pw/Flickr).2024.   
[48] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. "Frozen in Time Joint Video and Image Encoder for End-to-End Retrieval". In: IEEE International Conference on Computer Vision. 2021.   
[49] Queng Wang, Yukai Shi, Jiarong Ou, et al. Koala-36M:A Large-scale Video Dataset Improving Consistency between Fine-grained Conditions and Video Content. 2025. arXiv: 2410 . 08260 [cs.CV]. URL: https : //arxiv.org/abs/2410.08260.   
[50] Hui Li, Mingwang Xu, Yun Zhan, et al. OpenHumanVid: A Large-Scale High-Quality Dataset for Enhancing Human-Centric Video Generation. 2025. arXiv: 2412. 00115 [cs.CV]. URL: https: / /arxiv. org/abs/ 2412.00115.   
[51] Haorui He, Zengqiang Shang, Chaoren Wang, et al. Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation. 2024. arXiv: 2407 . 05361 [eess .AS]. URL: https : / /arxiv. org/abs/2407.05361.   
[52] Jort F. Gemmeke, Daniel P. W. Ells, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio Set:An ontology and human-labeled dataset for audio events". In: Proc. IEEE ICASSP 2017. New Orleans, LA, 2017.   
[53] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. VGGSound: A Large-scale Audio-Visual Dataset. 2020. arXiv: 2004.14368 [cs.CV]. URL: https://arxiv.org/abs/2004.14368.   
[54] YusAytar, Carl Vondrick, and Antonio Torralba. Soundnet: Learning sound representations from unlabeled video". In: Advances in Neural Information Processing Systems. 2016.   
[55] Jin Xu, Zhifang Guo, Hangrui Hu, et al. Qwe3-Omni Technical Report". In: arXiv preprint arXiv:2509.17765 (2025).   
[56] Tomá Souek and Jakub Loko. "TransNet V: An effective deep network architecture for fast shot transition detection". In: arXiv preprint arXiv:2008.04838 (2020).   
[57] Jiapeng Wang, Chengyu Wang, Kunzhe Huang, Jun Huang, and Lianwen Jin. VideoCLIP-XL: Advancing Long Description Understanding for Video CLIP Models. 2024. arXiv: 2410. 00741 [cs. CL]. URL: https: //arxiv.org/abs/2410.00741.   
[58] Akshay Raina and Vipul Arora. SyncNet: correlating objective for time delay estimation in audio signals. 2025. arXiv: 2203.14639 [eess.AS].URL: https://arxiv.org/abs/2203.14639.   
[59] Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. VSA: Faster Video Diffusion with Trainable Sparse Attention. 2025. arXiv: 2505.13389 [cs. CV]. URL: https://arxiv.org/abs/2505.13389.

# A Application Examples

Table 3: Summary of video generation, inpainting, and editing tasks   

<table><tr><td>Main Task</td><td>Subtask</td><td>Description</td></tr><tr><td>Generation</td><td>Image + Audio Ref Image + Motion Ref</td><td>Generate videos from multiple reference images and audio inputs Generate videos from image and video/motion reference (poses, trajec-</td></tr><tr><td>Inpainting</td><td>Region Inpainting Reference-Guided</td><td>tories) Inpaint subjects, attributes, or backgrounds in video regions Inpaint using reference image guidance for</td></tr><tr><td rowspan="6">Editing</td><td>Element Removal</td><td>style consistency Remove watermarks, subtitles, and logos intelligently</td></tr><tr><td>Subject Manipulation Attribute Editing</td><td>Add, delete, or modify subjects in videos Edit local attributes (color, texture, shape,</td></tr><tr><td>Background Editing</td><td>etc.) Modify backgrounds while preserving fore- ground</td></tr><tr><td>Style Transfer</td><td>Transform videos into different artistic</td></tr><tr><td>Camera Control</td><td>styles Modify shot angle, shot type, and camera</td></tr><tr><td>Scene Attributes</td><td>position Edit weather, lighting, tone, and time of day Combine subject and motion from different</td></tr><tr><td rowspan="2">First-Frame + Effect Ref</td><td>Subject + Motion Ref</td><td>references</td></tr><tr><td>Subject + Expression Ref Background + Video Ref</td><td>Transfer facial expressions from reference video Combine background and video references</td></tr></table>

Thisapen emtateypiplicaticasurmodi-audneratain n Ooextalu pa lc, and Editing.

# A.1 Reference-based Generation

# A.1.1 Multiple Image and Audio Reference Generation

Oy consistent with the references and audio-matched. The result is shown in Fig. 6.

I daronabeThe frst ce nActor, look weary sys ftly, <igIittle tired, I'm going back to my room to rest.</dialogue>@Audio-0. $@$ Actor-1 sits across from @Actor-0, hands clasped on the table, and says with determination, <ialogue>I wil a a visit to your parents tmorow.</dialogue>@Aud-. Then @Actor- in another room, by the window, holds her phone to her ear and speaks, <dialogue>Mom, Li Zeting said he's coming to our house tomorrow.</dialogue>@Audio-0. The scene shifts to another location—a warm-toned homeinterior, with a red abric sofa visible in the background.@Actor-2, sittng tensely with phone to her r worries, <dialogue>But with ourfamily's situation, do you think he might lok downon us?</dialogue>@Audio-. < ueo eeu two shots.</bgm>

![](images/10.jpg)  

Figure 6: Example of multiple images and audios reference.

# A.1.2 Image Reference and Motion Reference Generation e.g., pose sequences, trajectories) to control the dynamic characteristics of the generated video.

Instruction: Animate the person in @image_1 using the movements from @video_1.

![](images/11.jpg)

Instruction: The medical professional in@image1 and the curly-hairedwomanfrom@image_2 execute the dance movements demonstrated in @video_1, al set within the same stage environment as $@$ video_1.

![](images/12.jpg)  

Figure 7: Examples of motion transfer in video reference.

# A.2 Video Inpainting

# A.2.1 Subject/Attribute/Background Inpainting

The t background replacement. Instruction: Replace the subject in the mask area in@video1 with a majestic elk standing in the same feld.

![](images/13.jpg)

![](images/14.jpg)  

Instruction: Change the color of the tie in the masked area of @video_1 to blue. Instruction:Replace the backgroundin the masked areaof @video1 with a stunning cinematic view o theAmal Coast in Italy during a warm golden hour sunset.

![](images/15.jpg)  

Figure 8: Examples of subject/attribute/background inpainting.

# A.2.2 Image Reference Inpainting

Tmo ot seag gui ai pr sha p  i consistent with the reference style. Instruction: Add the man from @image_1 to the left mask area of @video_1.

![](images/16.jpg)

Instruction: Replace the right mask area in @video_1 with the cat from @image_1 and the left mask area in @video_1 with the woman from @image_2, ensuring a harmonious and natural scene.

![](images/17.jpg)  

Figure 9: Examples of image reference inpainting.

# A.3 Video Editing

# A.3.1 Local Editing

The model enables fine-grained local video editing: subject, attribute, and element edits. Watermark/Subtitle/Logo Removal The model can intelligenty identiy and rmove watermarks, subtite, lgos and other elements from videos while maintaining content coherence and naturalness. Instruction: Remove watermarks in @video_1.

![](images/18.jpg)

![](images/19.jpg)  

Instruction: Remove the text overlay at the bottom of @video_1.

![](images/20.jpg)  

Instruction: Remove the logo in the upper right corner in @video_1.

Figure 10: Examples of watermark/subtitle/logo removal.

Subject Manipulation The model supports adding, deleting and modifying subjects in videos while maintaiing temporal consistency. I a n exhr side of the path in @video_1.

![](images/21.jpg)  

Figure 11: Examples of subject manipulation.

Le pe videos, such as color, texture, shape, etc.

![](images/22.jpg)  

Instruction: Change the chair's color to black and replace its edges with wooden material in @video_1.

![](images/23.jpg)  

Instruction: Change the man's sleeveless shirt in @video_1 to a blue Polo shirt style. Bacground Edig The mode supports modiyig background ements whil preringhe oreoun bjs. Instruction: Replace the background of @video1 with a post-rain European cobblestone street scene at dusk.

![](images/24.jpg)  

Figure 12: Examples of local attribute editing.

# A.3.2 Global Editing

Them ot oals h e cylm pro attributes. Syl TTel om s f r y hii consistency of the video content. Instruction: Transform @video_1 into Paper-Cutting style.

![](images/25.jpg)

Instruction: Transform @video_1 into LEGO style.

![](images/26.jpg)  

Figure 13: Examples of style transfer.

Caera Control The model supports modifying camera properties including shot angle, shot type, and camera position. Instruction: Re-render @video_1 with a Pan Right camera movement.

![](images/27.jpg)  

Figure 14: Examples of camera control.

G ttTheo t al u u s ee hc tone, and time of day. Instruction: Make @video_1 nighttime.

![](images/28.jpg)  

Figure 15: Examples of global scene attributes.

# A.3.3 Reference-Based Editing

Themodel uportsvidedit basenag eferens icludisubje referencebackroneren n o expression, or visual effects guidance. Suet Reece with Motion Re Themode cn nee des b cbi  sb ro a ece image with motion pattes from  referece vido, matchingactin hythm and tajery. Instruction: Woman from @image_1 mimics gestures from @video_1 in its golden field background.

![](images/29.jpg)  

Figure 16: Example of subject reference with motion reference.

Subect Refece wh Exrss Re The oel can tranr atal cl exprss om ec video to a subject from a reference image. Instruction: Transfer the facial expressions from @video_1 to the man in @image_1.

![](images/30.jpg)  

Figure 17: Example of subject reference with expression reference

Background Reference with Video Reference The model can combine a background from a reference image with content or motion from any reference video. Instruction: Replace the background of @video_1 with @image_1.

![](images/31.jpg)  

Figure 18: Example of background reference with video reference.

FR  e   l a video starting from a reference image, enabling effect transfer from the reference video. Instruction: Transfer the diamond morphing effect from @video_1 onto the subject in @image_1.

![](images/32.jpg)  

Figure 19: Example of first-frame reference with effect reference.