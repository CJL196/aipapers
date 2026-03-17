# DIffusionNfT: ONLine DIffusion Reinforce-Ment With Forward Process

Kaiwen Zheng1,2,\* Huayu Chen $^ { 1 , 2 , * }$ Haotian $\mathbf { Y e } ^ { 2 , 3 }$ Haoxiang Wang2 Qinsheng Zhang2 Kai Jiang1 Hang $\mathbf { S u } ^ { 1 }$ Stefano Ermon³ Jun Zhu1,† Ming-Yu Liu2 \*Equal Contribution  Corresponding Author 1Tsinghua University 2NVIDIA 3Stanford University https://research.nvidia.com/labs/dir/DiffusionNFT

# ABSTRACT

Online reinforcement learning (RL) has been central to post-training language models, but its extension to diffusion models remains challenging due to intractable likelihoods. Recent works discretize the reverse sampling process to enable GRPO-style training, yet they inherit fundamental drawbacks, including solver restrictions, forwardreverse inconsistency, and complicated integration with classifier-free guidance (CFG). We introduce Diffusion Negative-aware Fine-Tuning (DiffusionNFT), a new online RL paradigm that optimizes diffusion models directly on the forward process via flow matching. DiffusionNFT contrasts positive and negative generations to define an implicit policy improvement direction, naturally incorporating reinforcement signals into the supervised learning objective. This formulation enables training with arbitrary black-box solvers, eliminates the need for likelihood estimation, and requires only clean images rather than sampling trajectories for policy optimization. DiffusionNFT is up to $2 5 \times$ more efficient than FlowGRPO in head-to-head comparisons, while being CFGfree. For instance, DiffusionNFT improves the GenEval score from 0.24 to 0.98 within 1k steps, while FlowGRPO achieves 0.95 with over $5 \mathrm { k }$ steps and additional CFG employment. By leveraging multiple reward models, DiffusionNFT significantly boosts the performance of SD3.5-Medium in every benchmark tested.

![](images/1.jpg)  

Figure 1: Performance of DiffusionNFT. (a) Head-to-head comparison with FlowGRPO on the GenEval task. (b) By employing multiple reward models, DiffusionNFT significantly boosts the performance of SD3.5-Medium in every benchmark tested, while being fully CFG-free.

# 1 INTRODUCTION

Online Reinforcement Learning (RL) has been pivotal in the post-training of LLMs, driving recent advances in LLMs' alignment and reasoning abilities (Achiam et al., 2023; Guo et al., 2025). However, replicating similar success for diffusion models in visual generation is not straightforward.

![](images/2.jpg)  

Figure 2: Comparison between Forward-Process RL (NFT) and Reverse-Process RL (GRPO). NFT allows using any solvers and does not require storing the whole sampling trajectory for optimization.

Policy Gradient algorithms assume that model likelihoods are exactly computable. This assumption holds for autoregressive models, but is inherently violated by diffusion models, where likelihoods can only be approximated via costly probabilistic ODE or variational bounds of SDE (Song et al., 2021). Recent works circumvent this barrier by discretizing the reverse sampling process, reframing diffusion generation as a multi-step decision-making problem (Black et al., 2023). This makes transitions between adjacent steps tractable Gaussians, enabling direct application of existing RL algorithms like GRPO to the diffusion domain (Xue et al., 2025; Liu et al., 2025). Despite promising efforts made, we argue that GRPO-style diffusion reinforcement still faces fundamental limitations: (1) Forward inconsistency. Focusing solely on the reverse sampling process breaks adherence to the forward diffusion process, risking the model degenerating into cascaded Gaussians. (2) Solver restriction. The data collection process relies on first-order SDE samplers, precluding the full utilization of ODE or high-order solvers that are default to flow models and advantageous for generation efficiency. (3) Complicated CFG integration. Diffusion models heavily rely on Classifier-Free Guidance (CFG) (Ho & Salimans, 2022), which requires training both conditional and unconditional models. Current RL practices typically incorporate CFG in post-training, leading to a complicated and inefficient two-model optimization scheme. We aim to disentangle data collection, remove solver restriction, and maintain consistency with standard supervised pretraining in diffusion RL. As a diffusion policy admits a single forward (noising) process but multiple reverse (denoising) processes (e.g., different samplers), a natural question is: Can diffusion reinforcement be performed on the forward process instead of the reverse? This paper proposes a novel online RL paradigm named Diffusion Negative-aware FineTuning (DiffusionNFT). Instead of building upon the conventional policy gradient framework, DiffusionNFT directly performs policy optimization on the forward diffusion process through the flow matching objective. Intuitively, it defines a contrastive improvement direction between two implicit policies learned on "positive" and "negative" generated samples split by reward signals, and optimizes toward the positive policy without modifying the sampling process. The forward-process RL formulation provides several practical benefits (Figure 2). First, DiffusionNFT allows data collection with arbitrary black-box solvers, rather than relying on first-order SDE samplers. Second, it eliminates the need to store entire sampling trajectories, requiring only clean images for policy optimization. Third, it is fully compatible with standard diffusion training, requiring minimal modifications to existing codebases. Finally, it is a native off-policy algorithm, naturally allowing decoupled training and sampling policies without importance sampling. We evaluate DiffusionNFT by post-training SD3.5-Medium (Esser et al., 2024) on multiple reward models. The entire training process deliberately operates in a CFG-free setting. Although this results in a significantly lower initialization performance, we find DiffusionNFT substantially improves performance across both in-domain and out-of-domain rewards, rapidly outperforming CFG and the GRPO baseline. We also conduct head-to-head comparisons against FlowGRPO in single-reward settings. Across four tasks tested, DiffusionNFT consistently exhibits $3 \times$ to $2 5 \times$ efficiency and achieves better final scores. For instance, it improves the GenEval score from 0.24 to 0.98 within 1k steps, while FlowGRPO achieves only 0.95 with over $5 \mathrm { k }$ steps and additional CFG employment. DiffusionNFT is a direct RL alternative to conventional Policy Gradient methods, introducing the Negative-aware FineTuning (NFT) paradigm (Chen et al., 2025c) into the diffusion domain. Grounded in a supervised learning foundation, we believe this paradigm offers a valid path toward a general, unified, and native off-policy RL recipe across various modalities.

# 2 BACKGROUND

# 2.1 Diffusion ANd FlOW ModELS

Diffusion models (Ho et al., 2020; Song et al., 2020b) learn continuous data distributions by gradually perturbing clean data $\pmb { x } _ { 0 } \sim \pi _ { 0 } = p _ { \mathrm { d a t a } }$ with Gaussian noise according to a forward process. Then, data can be generated by learning to reverse this process. The forward noising process admits a closed-form transition kernel $\pi _ { t | 0 } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) = \mathcal { N } ( \alpha _ { t } \pmb { x } _ { 0 } , \sigma _ { t } ^ { 2 } \mathbf { I } )$ with a specific noise schedule $\alpha _ { t } , \sigma _ { t }$ , enabling reparameterization as

$$
\begin{array} { r } { \pmb { x } _ { t } = \alpha _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { \epsilon } , \pmb { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) . } \end{array}
$$

One way to learn diffusion models is to adopt the velocity parameterization ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t )$ (Zheng et al., 2023b), which predicts the tangent of the trajectory, trained by minimizing where the target velocity $\textbf {  { v } }$ is defined by the schedule's time derivatives as ${ \pmb v } = \dot { \alpha } _ { t } { \pmb x } _ { 0 } + \dot { \sigma } _ { t } { \pmb \epsilon }$ under the notation $\bar { \dot { f } } _ { t } : = \mathrm { d } f _ { t } / \dot { \mathrm { d } { t } }$ , and $w ( t )$ is some weighting function. Reverse sampling typically follows h $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) } \end{array}$ using ${ \pmb v } _ { \theta }$ . This formulation is known as flow matching (Lipman et al., 2022), where simple Euler discretization serves as an effective ODE solver, equivalent to DDIM (Song et al., 2020a).

$$
\begin{array} { r } { \mathbb { E } _ { t , { \boldsymbol { x } } _ { 0 } \sim \pi _ { 0 } , { \boldsymbol { \epsilon } } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) } [ w ( t ) \| { \boldsymbol { v } } _ { \theta } ( { \boldsymbol { x } } _ { t } , t ) - { \boldsymbol { v } } \| _ { 2 } ^ { 2 } ] , } \end{array}
$$

Rectified flow (Liu et al., 2022) can be considered as a special case of the above-discussed diffusion models, where $\alpha _ { t } = 1 - t , \sigma _ { t } = t$ , which simplifies the velocity target to ${ \pmb v } = { \pmb \epsilon } - { \pmb x } _ { 0 }$ .

# 2.2 Policy Gradient Algorithms for Diffusion Models

In order to apply Policy Gradient algorithms such as PPO (Schulman et al., 2017) or GRPO (Shao et al., 2024) to diffusion models, recent works (Black et al., 2023; Fan et al., 2023; Liu et al., 2025; Xue et al., 2025) formulate the diffusion sampling as a multi-step Markov Decision Process (MDP). This can be achieved by discretizing the reverse sampling process of diffusion models. While flow models naturally admit simple and efficient sampling through ODE, the lack of stochasticity hinders the application of GRPO. FlowGRPO (Liu et al., 2025) addresses this by using the SDE form (Song et al., 2020b) under the velocity parameterization ${ \pmb v } _ { \theta }$ (see Appendix B.1):

$$
\mathrm { d } x _ { t } = \Big [ v _ { \theta } ( x _ { t } , t ) + \frac { g _ { t } ^ { 2 } } { 2 t } \big ( x _ { t } + ( 1 - t ) v _ { \theta } ( x _ { t } , t ) \big ) \Big ] \mathrm { d } t + g _ { t } \mathrm { d } w _ { t }
$$

where $\begin{array} { r } { g _ { t } = a \sqrt { \frac { t } { 1 - t } } } \end{array}$

$$
\pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t - \Delta t } \mid \mathbf { x } _ { t } ) = \mathcal { N } \Big ( \mathbf { x } _ { t } + \Big [ v _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t ) + \frac { g _ { t } ^ { 2 } } { 2 t } ( \mathbf { x } _ { t } + ( 1 - t ) v _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t ) ) \Big ] \Delta t , \ g _ { t } ^ { 2 } \Delta t \mathbf { I } \Big ) .
$$

This makes transition kernels between adjacent steps likelihood tractable Gaussians, enabling the direct application of existing policy gradient algorithms, such as GRPO.

# 3 Diffusion Reinforcement via Negative-aware Finetuning

# 3.1 PROBLEm SETup

Online RL. Consider a pretrained diffusion policy $\pi ^ { \mathrm { o l d } }$ and prompt datasets $\{ c \}$ At each iteration, we sample $K$ images $\pmb { x } _ { 0 } ^ { 1 : \hat { K } }$ for prompt $^ c$ $r \in [ 0 , 1 ]$ , representing its optimality probability $r ( \pmb { x } _ { 0 } , \pmb { c } ) : = p ( \mathbf { o } = 1 | \pmb { x } _ { 0 } , \pmb { c } )$ (Levine, 2018). This optimality serves as a bridge from continuous-valued rewards to a binary partition. Collected data can be randomly split into two imaginary subsets. An image $\scriptstyle { \pmb x } _ { 0 }$ will have a probability $r$ of falling into the positive dataset $\mathcal { D } ^ { + }$ and otherwise the negative dataset $\mathcal { D } ^ { - }$ .Given infinite samples, the underlying distributions of these two subsets are respectively

$$
\pi ^ { + } ( x _ { 0 } | c ) : = \pi ^ { \smash { \mathrm { o l d } } } ( x _ { 0 } | \mathbf { o } = 1 , c ) = \frac { p ( \mathbf { o } = 1 | x _ { 0 } , c ) \pi ^ { \mathrm { o l d } } ( x _ { 0 } | c ) } { p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) } = \frac { r ( x _ { 0 } , c ) } { p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) } \pi ^ { \mathrm { o l d } } ( x _ { 0 } | c ) 
$$

$$
\pi ^ { - } ( \boldsymbol { x } _ { 0 } | \boldsymbol { c } ) : = \pi ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { 0 } | \mathbf { o } = 0 , \boldsymbol { c } ) = \frac { p ( \mathbf { o } = 0 | \boldsymbol { x } _ { 0 } , \boldsymbol { c } ) \pi ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { 0 } | \boldsymbol { c } ) } { p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 0 | \boldsymbol { c } ) } = \frac { 1 - r ( \boldsymbol { x } _ { 0 } , \boldsymbol { c } ) } { 1 - p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | \boldsymbol { c } ) } \pi ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { 0 } | \boldsymbol { c } )
$$

RL requires performing policy improvement at each iteration. The optimized policy $\pi ^ { * }$ satisfies

$$
\mathbb { E } _ { \pi ^ { * } ( \cdot | c ) } r ( { \boldsymbol x } _ { 0 } , c ) > \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \cdot | c ) } r ( { \boldsymbol x } _ { 0 } , c ) \qquad ( \mathrm { d e n o t e d ~ a s } \quad \pi ^ { * } \succ \pi ^ { \mathrm { o l d } } )
$$

Policy Improvment on Posiive Data. It is easy to prove that $\pi ^ { + } \succ \pi ^ { \mathrm { o l d } } \succ \pi ^ { - }$ constantly holds, thus a straightforward improvement of $\pi ^ { \mathrm { o l d } }$ can be $\pi ^ { * } = \pi ^ { + }$ To ahive this, previous work (Lee et al., 2023) performs diffusion training solely on $\mathcal { D } ^ { + }$ , known as Rejection FineTuning (RFT). Despite the simplicity, RFT cannot effectively leverage negative data in $\mathcal { D } ^ { - }$ (Chen et al., 2025c). Reinforcement Guidance. We posit that negative feedback is crucial to policy improvement, especially for diffusion1. Rather than treating $\pi ^ { + }$ as an optimization point, we leverage both negative and positive data to derive an improvement direction $\Delta \in \mathbb { R } ^ { n }$ . The training target is defined as where $\textbf {  { v } }$ is the velocity predictor of the diffusion model, $\beta$ is a hyperparameter. This definition formally resembles diffusion guidance such as Classifier-Free Guidance (CFG) (Ho & Salimans, 2022). We term $\Delta ( \pmb { x } _ { t } , \pmb { c } , t ) \in \mathbb { R } ^ { n }$ reinforcement guidance, and $\textstyle { \frac { 1 } { \beta } } \in \mathbb { R }$ guidance strength.

$$
{ \boldsymbol v } ^ { * } ( { \boldsymbol x } _ { t } , c , t ) : = { \boldsymbol v } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } , c , t ) + \frac { 1 } { \beta } \Delta ( { \boldsymbol x } _ { t } , c , t ) .
$$

In Section 3.2, we address two challenges: 1. What is an appropriate form of $\Delta$ that enables policy improvement? 2. How to directly optimize ${ \pmb v } _ { \theta }  { \pmb v } ^ { * }$ leveraging collected dataset $\mathcal { D } ^ { + }$ and $\mathcal { D } ^ { - }$ ?

# 3.2 NEGativE-AWare Diffusion Reinforcement WitH ForWard PRocess

In Eq. (3), $\Delta$ corresponds to the distributional shift between an improved policy and the original policy. To formalize this, we first study the distributional difference between $\pi ^ { + } \stackrel { \cdot } { \succ } \pi ^ { \mathrm { o l d } } \succ \pi ^ { - }$ . Theorem 3.1 (Improvement Direction). Consider diffusion models $v ^ { + }$ , $v ^ { - }$ , and $v ^ { o l d }$ for the policy triplet $\pi ^ { + } , \pi ^ { - }$ , and $\pi ^ { o l d }$ The directional differences between these models are proportional:

$$
\begin{array} { r l r } & { } & { \Delta : = [ 1 - \alpha ( { \pmb x } _ { t } ) ] [ { \pmb v } ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) ] } \\ & { } & { = \alpha ( { \pmb x } _ { t } ) [ { \pmb v } ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) ] . } \end{array}
$$

where $0 \leq \alpha ( { \pmb x } _ { t } ) \leq 1$ is a scalar coefficient:

$$
\alpha ( { \pmb x } _ { t } ) : = \frac { \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { o l d } ( { \pmb x } _ { t } | { \pmb c } ) } \mathbb { E } _ { \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \pmb c } ) } r ( { \pmb x } _ { 0 } , { \pmb c } )
$$

Eq. (4) indicates an ideal guidance direction $\Delta$ for improving over $v ^ { \mathrm { o l d } }$ With appropriate guidance strength, policy improvement can be guaranteed. For instance, let $\beta \ = \ \alpha ( \pmb { x } _ { t } )$ in Eq. (3), we have ${ \pmb v } ^ { * } ( { \pmb x } _ { t } , { \pmb c } , t ) = { \pmb v } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } , { \pmb c } , t ) +$ α(xt)∆(xt, c, t) = v+(xt, c, t), such th+t $\pi ^ { * } = \pi ^ { + } \succ \pi ^ { \mathrm { o l d } }$ holds. Figure 3 contains an illustration for the improvement direction $\Delta$ .

![](images/3.jpg)  

Figure 3: Improvement Direction.

with Eq. (3) and (4), we now introduce a training objective that directly optimizes ${ \pmb v } _ { \theta }$ towards $v ^ { * }$ :

![](images/4.jpg)  

Figure 4: DiffusionNFT jointly optimizes two dual diffusion objectives, on both positive $( r = 1 )$ and negative $( r = 0$ branches. Rather than training two independent models ${ \boldsymbol { v } } _ { \theta } ^ { + }$ and ${ \boldsymbol v } _ { \boldsymbol { \theta } _ { } } ^ { - }$ , it adopts a implicit parameerization technique that directlyoptimizes single target poliy ${ \pmb v } _ { \theta }$ .

Theorem 3.2 (Policy Optimization). Consider the training objective:

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { c , \pi ^ { o l d } ( { \pmb x } _ { 0 } \mid c ) , t } r \| { \pmb v } _ { \theta } ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } + ( 1 - r ) \| { \pmb v } _ { \theta } ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } ,
$$

where $\begin{array} { r } { \pmb { v } _ { \theta } ^ { + } ( \pmb { x } _ { t } , \pmb { c } , t ) : = ( 1 - \beta ) \pmb { v } ^ { o l d } ( \pmb { x } _ { t } , \pmb { c } , t ) + \beta \pmb { v } _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t ) , \quad ( } \end{array}$ Implicit positive policy) and $\begin{array} { r } { \pmb { v } _ { \theta } ^ { - } ( \pmb { x } _ { t } , \pmb { c } , t ) : = ( 1 + \beta ) \pmb { v } ^ { o l d } ( \pmb { x } _ { t } , \pmb { c } , t ) - \beta \pmb { v } _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t ) . } \end{array}$ (Implicit negative policy) Given unlimited data and model capacity, the optimal solution of Eq. (5) satisfies

$$
{ v } _ { \theta ^ { * } } ( { x } _ { t } , c , t ) = { v } ^ { o l d } ( { x } _ { t } , c , t ) + \frac { 2 } { \beta } \Delta ( { x } _ { t } , c , t ) .
$$

Theorem 3.2 presents a new off-policy RL paradigm (Figure 4). Instead of applying Policy Gradient, it adopts supervised learning (SL) objectives, but additionally trains on online negative data $\mathcal { D } ^ { - }$ . This renders the algorithm highly versatile, compatible with existing SL methods. We term our method Diffusion Negative-aware FineTuning (DiffusionNFT), highlighting its negative-aware SL nature and conceptual similarity to parallel algorithm NFT in language models (Chen et al., 2025c). Below, we discuss several distinctive advantages of DiffusionNFT.

1. Forward Consistency. In contrast to policy gradient methods (e.g., FlowGRPO), which formulated RL on the reverse diffusion process, DiffusionNFT defines a typical diffusion loss on the forward process. This preserves what we term forward consistency—the adherence of the diffusion model's underlying probability density to the Fokker-Planck equation (Øksendal, 2003; Song et al., 2020b), ensuring that the learned model corresponds to a valid forward process (i.e., $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ are correctly coupled with $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ through a joint distribution $\pi _ { \boldsymbol { \theta } } ( \pmb { x } _ { t } , \pmb { x } _ { 0 } ) = \pi _ { \boldsymbol { \theta } } ( \pmb { x } _ { 0 } ) \pi _ { t | 0 } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) )$ . 2. Solver Flexibility. DiffusionNFT fully decouples policy training and data sampling. This enables the full utilization of any black-box solvers throughout sampling, rather than relying on first-order SDE samplers. It also eliminates the need to store the entire sampling trajectory during data collection, requiring only clean images with their associated rewards for training. 3. Implicit Guidance Integration. Intuitively, DiffusionNFT defines a guidance direction $\Delta$ and apply such guidance to the old policy $v ^ { \mathrm { o l d } }$ (Eq. (6)). However, instead of learning a separate guidance model $\Delta _ { \theta }$ and employing guided sampling, it adopts an implicit parameterization technique that enables direct integration of reinforcement guidance into the learned policy. This technique, inspired by recent advances in guidance-free training (Chen et al., 2025a), allows us to perform RL continuously on a single policy model, which is crucial to online reinforcement. 4. Likelihood-Free Formulation. Previous diffusion RL methods are fundamentally constrained by their reliance on likelihood approximation. Whether approximating the marginal data likelihood with variational bounds and applying Jensen's inequality to reduce loss computation cost (Wallace et al., 2024), or discretizing the reverse process to estimate sequence likelihood (Black et al., 2023), they inevitably introduce systematic estimation bias into diffusion post-training. In contrast, DiffusionNFT is inherently likelihood-free, bypassing such compromises.

# 3.3 PRACTICAL IMPLEMENTATION

We provide DiffusionNFT pseudo code in Algorithm 1. Below, we elaborate on key design choices.

# Algorithm 1 Diffusion Negative-aware FineTuning (DiffusionNFT)

Reqire: reiffion y ${ \pmb v } ^ { \mathrm { r e f } }$ , raw reward function $r ^ { \mathrm { r a w } } ( \cdot ) \in \mathbb { R }$ pompt dataset $\{ c \}$ . Initialize: Data collection policy ${ \pmb v } ^ { \mathrm { o l d } }  { \pmb v } ^ { \mathrm { r e f } }$ , training policy ${ \pmb v } _ { \theta }  { \pmb v } ^ { \mathrm { r e f } }$ , data buffer $\mathcal { D }  \emptyset$ 1: for each iteration $i$ do   
2: for each sampled prompt $^ c$ do //Rollout Step, Data Collection   
3: $K$ $\pmb { x } _ { 0 } ^ { 1 : K }$ $\{ r ^ { \mathrm { r a w } } \} ^ { 1 : K }$   
4: Normalize raw rewards in group: $r ^ { \mathrm { n o m } } : = r ^ { \mathrm { r a w } } - \mathsf { m e a n } ( \{ r ^ { \mathrm { r a w } } \} ^ { 1 : K } )$ .   
5: Define optimality probability $r = 0 . 5 + 0 . 5 * \mathrm { c } \mathrm { 1 } \mathrm { i } \mathrm { p } \{ r ^ { \mathrm { n o r m } } / Z _ { c } , - 1 , 1 \}$ .   
6: $\mathcal { D }  \{ c , \ x _ { 0 } ^ { 1 : K } , r ^ { \bar { 1 } : K } \in [ 0 , 1 ] \}$   
7: end for   
8: for each mini batch $\{ c , \pmb { x } _ { 0 } , r \} \in \mathcal { D }$ do // Gradient Step, Policy Optimization   
9: Forward diffusion process: $\begin{array} { r } { \pmb { x } _ { t } = \alpha _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { \epsilon } ; \pmb { v } = \dot { \alpha } _ { t } \pmb { x } _ { 0 } + \dot { \sigma } _ { t } \pmb { \epsilon } , } \end{array}$ .   
10: Implicit positive velocity: $\boldsymbol { v } _ { \theta } ^ { + } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) : = ( 1 - \beta ) \boldsymbol { v } ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) + \beta \boldsymbol { v } _ { \theta } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t )$ .   
11: Implicit negative velocity: $\boldsymbol { v } _ { \theta } ^ { - } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) : = ( 1 + \beta ) \boldsymbol { v } ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) - \beta \boldsymbol { v } _ { \theta } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t )$ .   
12: . $\theta  \theta - \lambda \nabla _ { \theta } [ r \| v _ { \theta } ^ { + } ( x _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } + ( 1 - r ) \| v _ { \theta } ^ { - } ( x _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } ]$ .(Eq. (5))   
13: end for   
14: Update data collection policy $\theta ^ { \mathrm { o l d } }  \eta _ { i } \theta ^ { \mathrm { o l d } } + ( 1 - \eta _ { i } ) \theta$ , and clear buffer $\mathcal { D }  \emptyset$ // Online Update   
15:end for Output: $v _ { \theta }$ Optimality Reward. In most visual reinforcement settings, rewards manifest as unconstrained continuous scalars rather than binary optimality signals. Motivated by existing GRPO practices (Shao et al., 2024; Liu et al., 2025; Xue et al., 2025), we first transform the raw reward $r ^ { \mathrm { { r a w } } }$ into $r \in [ 0 , 1 ]$ which represents the optimality probability:

$$
r ( \pmb { x } _ { 0 } , \pmb { c } ) : = \frac { 1 } { 2 } + \frac { 1 } { 2 } \mathrm { c } \mathrm { 1 } \mathrm { i } \mathrm { p } \left[ \frac { r ^ { \mathrm { r a w } } ( \pmb { x } _ { 0 } , \pmb { c } ) - \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \cdot \vert \pmb { c } ) } r ^ { \mathrm { r a w } } ( \pmb { x } _ { 0 } , \pmb { c } ) } { Z _ { c } } , - 1 , 1 \right] .
$$

$Z _ { c } > 0$ is some normalizing factor, which could take the form of a global reward st d. We sample $K$ images for each prompt $^ c$ durin ve $\mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \cdot | \boldsymbol { c } ) } r ^ { \mathrm { r a w } } ( \pmb { x } _ { 0 } , \pmb { c } )$ for each prompt can be estimated. Soft Update of Sampling Policy. The off-policy nature of DiffusionNFT decouples the sampling policy $\bar { \pi } ^ { \mathrm { o l d } }$ from the training policy $\pi _ { \theta }$ .This obviates the need for a "hard" update $\pi ^ { \mathrm { o l d } }  \pi ^ { \theta }$ after each iteration. Instead, we leverage this property to employ a "soft" EMA update:

$$
\theta ^ { \mathrm { o l d } }  \eta _ { i } \theta ^ { \mathrm { o l d } } + ( 1 - \eta _ { i } ) \theta
$$

where $i$ is the iteration number. The parameter $\eta$ governs a trade-off between learning speed and stability. A strictly on-policy scheme $( \eta = 0$ yields rapid initial progress but is prone to severe instability, leading to catastrophic collapse. Conversely, a nearly offline approach $\langle \eta \to 1 \rangle$ is robustly stable but suffers from impractically slow convergence (Figure 8).

Adaptive Loss Weighting. Typical diffusion loss includes a time-dependent weighting $w ( t )$ (Eq. (1)). Instead of manual tuning, we adopt an adaptive weighting scheme. The velocity predictor ${ \pmb v } _ { \theta }$ can be equivalently transformed into $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ predictor, denoted as $\scriptstyle { \mathbf {  { x } } } \theta$ (e.g., ${ \mathbf { } } x _ { \theta } = x _ { t } - t { \mathbf { } } v _ { \theta }$ under rectified flow schedule). We replace the weighting with a form of self-normalized $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ regression, motivated by the diffusion distillation method DMD (Yin et al., 2024):

$$
w ( t ) \| v _ { \theta } ( x _ { t } , c , t ) - v \| _ { 2 } ^ { 2 }  \frac { \| x _ { \theta } ( x _ { t } , c , t ) - x _ { 0 } \| _ { 2 } ^ { 2 } } { \mathrm { s g } ( \mathrm { m e a n } ( \mathrm { a b s } ( x _ { \theta } ( x _ { t } , c , t ) - x _ { 0 } ) ) ) }
$$

where $S \mathbb { { g } }$ is the stop-gradient operator. We find it typically leads to faster training (Figure 9). CFG-Free Optimization. Classifier-Free Guidance (CFG) (Ho & Salimans, 2022) is a default technique to enhance generation quality at inference time, yet it complicates post-training and reduces efficiency. Conceptually, we interpret CFG as an offine form of reinforcement guidance (Eq. (4)), where conditional and unconditional models correspond to positive and negative signals. With this understanding, we discard CFG in our algorithm design, and the policy is initialized solely by the conditional model. Despite this seemingly poor initialization, we observe that performance surges and quickly surpasses the CFG baseline (Figure 1). This suggests that the functionality of CFG can be effectively learned or substituted through RL post-training, echoing recent studies that achieve strong performance without CFG through post-training (Chen et al., 2025b;a; Zheng et al., 2025).

Table 1: Evaluation Results. Gray-colored: In-domain reward. † Evaluated on official checkpoints. ‡Evaluated under $1 0 2 4 \times 1 0 2 4$ resolution. Bold: best; Underline: second best.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Iter</td><td colspan="2">Rule-Based</td><td colspan="6">Model-Based</td></tr><tr><td>GenEval</td><td>OCR</td><td>PickScore</td><td>ClipScore</td><td>HPSv2.1</td><td>Aesthetic</td><td>ImgRwd</td><td>UniRwd</td></tr><tr><td>SD-XL‡</td><td></td><td>0.55</td><td>0.14</td><td>22.42</td><td>0.287</td><td>0.280</td><td>5.60</td><td>0.76</td><td>2.93</td></tr><tr><td>SD3.5-L‡</td><td></td><td>0.71</td><td>0.68</td><td>22.91</td><td>0.289</td><td>0.288</td><td>5.50</td><td>0.96</td><td>3.25</td></tr><tr><td>FLUX.1-Dev</td><td></td><td>0.66</td><td>0.59</td><td>22.84</td><td>0.295</td><td>0.274</td><td>5.71</td><td>0.96</td><td>3.27</td></tr><tr><td>SD3.5-M (w/o CFG) + CFG</td><td></td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td></td><td>—</td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td>+ FlowGRPO†</td><td>&gt;5k</td><td>0.95</td><td>0.66</td><td>22.51</td><td>0.293</td><td>0.274</td><td>5.32</td><td>1.06</td><td>3.18</td></tr><tr><td></td><td>2k</td><td>0.66</td><td>0.92</td><td>22.41</td><td>0.290</td><td>0.280</td><td>5.32</td><td>0.95</td><td>3.15</td></tr><tr><td></td><td>4k</td><td>0.54</td><td>0.68</td><td>23.50</td><td>0.280</td><td>0.316</td><td>5.90</td><td>1.29</td><td>3.37</td></tr><tr><td>+ Ours</td><td>1.7k</td><td>0.94</td><td>0.91</td><td>23.80</td><td>0.293</td><td>0.331</td><td>6.01</td><td>1.49</td><td>3.49</td></tr></table>

# 4 EXPERIMENTS

We demonstrate the potential of DiffusionNFT through three perspectives: (1) multi-reward joint training for strong CFG-free performance, (2) head-to-head comparison with FlowGRPO on single rewards, and (3) ablation studies on key design choices.

# 4.1 EXPERIMENTAL SETup

Our experiments are based on SD3 . 5-Medium (Esser et al., 2024) at $5 1 2 \times 5 1 2$ resolution, with most settings aligned with FlowGRPO (Liu et al., 2025). Reward Models. (1) Rule-based rewards, including GenEva1 (Ghosh et al., 2023) for compositional image generation and OCR for visual text rendering, where the partial reward assignment strategies follow FlowGRPO. (2) Model-based rewards, including PickScore (Kirstain et al., 2023), ClipScore (Hessel et al., 2021), HPSv2.1 (Wu et al., 2023), Aesthetics (Schuhmann, 2022), ImageReward (Xu et al., 2023) and Uni fiedReward (Wang et al., 2025), which measure image quality, image-text alignment and human preference. Prompt Datasets. For GenEval and OCR, we use the corresponding training and test sets from FlowGRPO. For other rewards, we train on Pick-a-Pic (Kirstain et al., 2023) and evaluate on DrawBench (Saharia et al., 2022). Training and Evaluation. We finetune with LoRA $\alpha = 6 4$ $r = 3 2$ ). Each epoch consists of 48 groups with group size $G = 2 4$ . We use 10 rollout sampling steps for head-to-head comparison and ablation studies, and 40 steps for best visual quality in multi-reward training. Evaluation is performed with 40-step first-order ODE sampler. Additional details are provided in Appendix C.

# 4.2 MUlti-ReWArd JoinT Training

We first assess DiffusionNFT's effectiveness in comprehensively enhancing the base model. Starting from the CFG-free SD3.5-M (2.5B parameters), we jointly optimize five rewards: GenEva1, OCR, PickScore, ClipScore, and HP Sv2 . 1. Since the rewards are based on different prompts, we first train on Pick-a-Pic with model-based rewards to strengthen alignment and human preference, followed by rule-based rewards (GenEval, OCR). Out-of-domain evaluation is conducted on Aesthetics, ImageReward, and UnifiedReward. As shown in Table 1, our final CFG-free model not only surpasses CFG and matches FlowGRPO (fitted only single rewards) on both in-domain and out-of-domain metrics, but also outperforms CFGbased larger models such as $\mathrm { S D } 3 . 5 \mathrm { - L }$ (8B parameters) and FLUX.1-Dev (12B parameters) (Labs, 2024). Qualitative comparison in Figure 5 demonstrates the superior visual quality of our method.

# 4.3 HEAD-TO-HEAd COMPARiSON

We conduct head-to-head comparisons with FlowGRPO on single training rewards. As shown in Figure 1(a) and Figure 6, our method is $3 \times$ to $2 5 \times$ more efficient in terms of wall-clock time,

![](images/5.jpg)  
FlowGRPO

DiffusionNFT achieving GenEval score of 0.98 within only ${ \sim } 1 \mathrm { k }$ iterations. This demonstrates that CFG-free models can rapidly adapt to specific reward environments under our framework.

![](images/6.jpg)  

Figure 5: Qualitative Comparison. The prompts are taken from GenEva1, OCR and DrawBench respectively, where we compare the corresponding FlowGRPO model with our model.   

Figure 6: Head-to-head comparison between DiffusionNFT with FlowGRPO on single rewards.

# 4.4 ABLATION STUDIES

![](images/7.jpg)

![](images/8.jpg)  

Figure 7: Different diffusion samplers for data collection.   

Figure 8: Soft-update strategies.

We analyze the impact of our core design choices: Negative Loss. The negative-aware component is crucial in DiffusionNFT. Without the negative policy loss on ${ \boldsymbol { v } } _ { \boldsymbol { \theta } } ^ { - }$ , we find rewards collapse almost instantly during online training, highlighting the essential role of negative signals in diffusion RL. This phenomenon is divergent from observations in LLMs, where RFT remains a strong baseline (Xiong et al., 2025; Chen et al., 2025c).

![](images/9.jpg)  

Figure 9: Different time-dependent weighting strategies.

![](images/10.jpg)  

Figure 10: Choices of strength $\beta$ .

Diffusion Sampler. Online samples in DiffusionNFT are used both for reward evaluation and as training data, making quality critical. Figure 7 shows that ODE samplers outperform SDE ones, especially on P ickScore, which is noise-sensitive. Second-order ODE slightly outperforms firstorder on GenEval, while being comparable on PickScore. Adaptive Weighting. We find stability improves when the flow-matching loss is given higher weight at larger $t$ , whereas inverse strategies (e.g., $w ( t ) = 1 - t )$ lead to collapse (Figure 9). Our adaptive schedule consistently matches or exceeds heuristic choices. Soft Update. We compare different $\eta _ { i }$ schedules for the soft update in Figure 8. Fully on-policy b $\gamma _ { i } = 0$ ) accelerates early progress but destabilizes training, while overly off-policy $( \eta = 0 . 9$ slows convergence. We find that starting with a small $\eta$ and gradually increasingit toalargervalue i latr stages strikes an effective balance between convergence speed and training stability. Guidance Strength. As shown in Figure 10, the guidance parameter $\beta$ also governs a trade-off between stability and convergence speed. We find that $\beta$ near 1 performs stably and select $\beta$ as 1 or 0.1 (for faster reward increase) in practice.

# 5 RELATED WORK

The transition of RL algorithms from discrete autoregressive (AR) to continuous diffusion models poses a central challenge: the inherent difficulty of diffusion models for computing exact model likelihoods (Song et al., 2021), which are nonetheless crucial for RL (Chen et al., 2023; Liu et al., 2025). To address this challenge, existing efforts include:

Likelihood-free methods: (1) Reward Backpropagation (Xu et al., 2023; Prabhudesai et al., 2023; Clark et al., 2023; Prabhudesai et al., 2024) proves highly effective, yet is limited to differentiable rewards and can only tune low-noise timesteps due to memory costs and gradient explosion when unrolling long denoising chains. (2) Reward-Weighted Regression (RWR) (Lee et al., 2023) is an offine finetuning method but lacks a negative policy objective to penalize low-reward generations. (3) Policy Guidance. This includes energy guidance (Janner et al., 2022; Lu et al., 2023) and CFGstyle guidance (Frans et al., 2025; Jin et al., 2025). These methods all require combining multiple models for guided sampling, thus complicating online optimization. (4) Score-based RL. These methods try to perform RL directly on the score rather than the likelihood field (Zhu et al., 2025).

Likelihood-based methods: (1) Diffusion-DPO (Wallace et al., 2024; Yang et al., 2024; Liang et al., 2024; Yuan et al., 2024; Li et al., 2025a) adapts DPO to diffusion for paired human preference data but requires additional likelihood and loss approximations compared to AR; DDO (Zheng et al., 2025) uses high-quality dataset as positive signals and self-generated samples as negative signals to avoid the requirement of paired data, achieving state-of-the-art CFG-free FIDs in visual generation, while still relying on likelihood approximation for the diffusion case. (2) Policy gradient methods, starting from PPO style (Black et al., 2023; Fan et al., 2023), decompose trajectory likelihoods step by step without considering forward consistency. Recent GRPO extensions (Liu et al., 2025; Xue et al., 2025) prove effective and scalable for diffusion RL, but they couple the training loss with SDE samplers and face efficiency bottlenecks. MixGRPO (Li et al., 2025b) improves efficiency by mixing SDE and ODE, while issues of coupling and forward inconsistency remain.

# 6 CONCLUSION

We introduce Diffusion Negative-aware FineTuning (DiffusionNFT), a new paradigm for online reinforcement learning of diffusion models that directly operates on the forward process. By formulating policy improvement as a contrast between positive and negative generations, DiffusionNFT integrates reinforcement signals seamlessly into the standard diffusion objective, eliminating the reliance on likelihood estimation and SDE-based reverse process. Empirically, DiffusionNFT demonstrates strong and efficient reward optimization, achieving up to $2 5 \times$ higher efficiency than Flow-GRPO while producing a single model that outperforms CFG baselines across diverse in-domain and out-of-domain rewards. We believe this work represents a step toward unifying supervised and reinforcement learning in diffusion, and highlights the forward process as a promising foundation for scalable, efficient, and theoretically principled diffusion RL. The Use of Large Language Models (Llms) We used large language models (LLMs) solely as a writing assistant for language polishing and improving clarity of presentation. The LLMs were not involved in research ideation, methodological design, experimental execution, or result analysis. All scientific contributions and substantive writing were carried out by the authors.

# ACKNOWLEDGMENTS

We thank Cheng Lu, Hanzi Mao, Zekun Hao, Tao Yang, Zhanhao Liang, Shuhuai Ren, Tenglong Ao, Xintao Wang, Haoqi Fan, Jiajun Liang, Yuji Wang, and Hongzhou Zhu for the valuable discussion.

# REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. Huayu Chen, Cheng Lu, Chengyang Ying, Hang Su, and Jun Zhu. Offline reinforcement learning via high-fidelity generative behavior modeling. In The Eleventh International Conference on Learning Representations, 2023. Huayu Chen, Kai Jiang, Kaiwen Zheng, Jianfei Chen, Hang Su, and Jun Zhu. Visual generation without guidance. Forty-second international conference on machine learning, 2025a. Huayu Chen, Hang Su, Peize Sun, and Jun Zhu. Toward guidance-free ar visual generation via condition contrastive alignment. In ICLR, 2025b. Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu, and Haoxiang Wang. Bridging supervised learning and reinforcement learning in math reasoning. arXiv preprint arXiv:2505.18116, 2025c. Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:7985879885, 2023. Kevin Frans, Seohong Park, Pieter Abbeel, and Sergey Levine. Diffusion guidance is a controllable policy improvement operator. arXiv preprint arXiv:2505.23458, 2025. Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36: 5213252152, 2023. Martin Gonzalez, Nelson Fernandez Pinto, Thuy Tran, Hatem Hajri, Nader Masmoudi, et al. Seeds: Exponential sde solvers for fast high-quality sampling from diffusion models. Advances in Neural Information Processing Systems, 36:6806168120, 2023. Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:68406851, 2020. Marlis Hochbruck and Alexander Ostermann. Exponential integrators. Acta Numerica, 19:209286, 2010. Chin-Wei Huang, Jae Hyun Lim, and Aaron C Courville. A variational perspective on diffusionbased generative models and score matching. Advances in Neural Information Processing Systems, 34:2286322876, 2021. Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, 2022. Luozhijie Jin, Zijie Qiu, Jie Liu, Zijie Diao, Lifeng Qiao, Ning Ding, Alex Lamb, and Xipeng Qiu. Inference-time alignment control for diffusion models with reinforcement learning guidance. arXiv preprint arXiv:2508.21016, 2025. Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:2169621707, 2021. Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:3665236663, 2023. Black Forest Labs. Flux. https://github.com/black-forest-labs/flux,2024. Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018. Binxu Li, Minkai Xu, Meihua Dang, and Stefano Ermon. Divergence minimization preference optimization for diffusion model alignment. arXiv preprint arXiv:2507.07510, 2025a. Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025b. Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Ji Li, and Liang Zheng. Step-aware preference optimization: Aligning preference with denoising performance at each step. arXiv preprint arXiv:2406.04314, 2(5):7, 2024. Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:57755787, 2022a. Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver $^ { + + }$ : Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022b. Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, and Jun Zhu. Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning. arXiv preprint arXiv:2304.12824, 2023. Bernt Øksendal. Stochastic differential equations. In Stochastic differential equations: an introduction with applications, pp. 3850. Springer, 2003. Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-toimage diffusion models with reward backpropagation. 2023. Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024. Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:3647936494, 2022. Christoph Schuhmann. Laion-aesthetics. https://laion.ai/blog/ laion-aesthetics/,2022. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a. Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b. Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of scorebased diffusion models. In Advances in Neural Information Processing Systems, volume 34, pp. 14151428, 2021. Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 82288238, 2024. Feng Wang and Zihao Yu. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025. Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025. Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20962105, 2023. Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025. Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:1590315935, 2023. Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 89418951, 2024. Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 66136623, 2024. Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning of diffusion models for text-to-image generation. Advances in Neural Information Processing Systems, 37: 7336673398, 2024. Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022. Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. In Thirty-seventh Conference on Neural Information Processing Systems, 2023a. Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Improved techniques for maximum likelihood estimation for diffusion odes. In International Conference on Machine Learning, pp. 42363 42389. PMLR, 2023b. Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Diffusion bridge implicit models. arXiv preprint arXiv:2405.15885, 2024. Kaiwen Zheng, Yongxin Chen, Huayu Chen, Guande He, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Direct discriminative optimization: Your likelihood-based visual generative model is secretly a gan discriminator. In ICML, 2025. Huaisheng Zhu, Teng Xiao, and Vasant G Honavar. Dspo: Direct score preference optimization for diffusion model alignment. In The Thirteenth International Conference on Learning Representations, 2025.

# A Proof Of Theorems

Lemma A.1 (Distribution Split). Consider the distribution triplet $\pi ^ { + }$ , $\pi ^ { - }$ , and $\pi ^ { o l d }$ , as defined in Section 3.1:

$$
\begin{array} { l } { { \pi ^ { + } ( { \pmb x } _ { 0 } | c ) : = \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \bf o } = 1 , c ) = \displaystyle \frac { p ( { \bf o } = 1 | { \pmb x } _ { 0 } , c ) \pi ^ { o l d } ( { \pmb x } _ { 0 } | c ) } { p _ { \pi ^ { o l d } } ( { \bf o } = 1 | c ) } = \displaystyle \frac { r ( { \pmb x } _ { 0 } , c ) } { p _ { \pi ^ { o l d } } ( { \bf o } = 1 | c ) } \pi ^ { o l d } ( { \pmb x } _ { 0 } | c ) \quad \mathrm { ~ o ~ r ~ } \quad ( 1 \leq \alpha \leq \alpha ) } } \\ { { \pi ^ { - } ( { \pmb x } _ { 0 } | c ) : = \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \bf o } = 0 , c ) = \displaystyle \frac { p ( { \bf o } = 0 | { \pmb x } _ { 0 } , c ) \pi ^ { o l d } ( { \pmb x } _ { 0 } | c ) } { p _ { \pi ^ { o l d } } ( { \bf o } = 0 | c ) } = \displaystyle \frac { 1 - r ( { \pmb x } _ { 0 } , c ) } { 1 - p _ { \pi ^ { o l d } } ( { \bf o } = 1 | c ) } \pi ^ { o l d } ( { \pmb x } _ { 0 } | c ) \quad \mathrm { ~ o ~ r ~ } \quad ( 1 \leq \alpha \leq \alpha ) } } \end{array}
$$

$\pi ^ { o l d } ( \pmb { x } _ { 0 } | \pmb { c } )$ is as a linear combination between its positive slt $\pi ^ { + } ( { \pmb x } _ { 0 } | { \pmb c } )$ and negative split $\pi ^ { - } \left( \pmb { x } _ { 0 } | \pmb { c } \right)$ .

$$
\pi ^ { o l d } ( x _ { 0 } | c ) = p _ { \pi ^ { o l d } } ( \mathbf { o } = 1 | c ) \pi ^ { + } ( x _ { 0 } | c ) + [ 1 - p _ { \pi ^ { o l d } } ( \mathbf { o } = 1 | c ) ] \pi ^ { - } ( x _ { 0 } | c )
$$

Proof. The result follows directly from Eq.(7) and Eq.(8). Lemma A.2 (Posterior Split). The diffusion posteriors for distribution triplet $\pi ^ { + } , \pi ^ { - }$ , and $\pi ^ { o l d }$ satisfy:

$$
\begin{array} { r l } & { { \pi ^ { o l d } } ( x _ { 0 } | x _ { t } , c ) = \alpha ( { x _ { t } } ) \pi ^ { + } ( { x _ { 0 } } | x _ { t } , c ) + [ 1 - \alpha ( { x _ { t } } ) ] \pi ^ { - } ( { x _ { 0 } } | { x _ { t } } , c ) } \\ & { \qquad w h e r e \qquad \alpha ( { x _ { t } } ) : = \frac { \pi _ { t } ^ { + } ( { x _ { t } } | c ) } { \pi _ { t } ^ { o l d } ( { x _ { t } } | c ) } \mathbb { E } _ { \pi ^ { o l d } ( { x _ { 0 } } | c ) } r ( { x _ { 0 } } , c ) } \end{array}
$$

Proof. Leveraging Bayes' Rule:

$$
\pi ^ { \mathrm { o l d } } ( { \pmb x } _ { 0 } | { \pmb c } ) = \frac { \pi _ { t } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } | { \pmb c } ) \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \pmb x } _ { 0 } | { \pmb x } _ { t } , { \pmb c } ) } { \pi ( { \pmb x } _ { t } | { \pmb x } _ { 0 } ) }
$$

Replacing all distributions in Eq. (9) (Lemma A.1) we get

$$
\begin{array} { r l } & { \frac { \pi _ { t } ^ { \mathrm { o d d } } ( x _ { t } | c ) \pi _ { 0 | t } ^ { \mathrm { o d d } } ( x _ { 0 } | x _ { t } , c ) } { \pi ( x _ { t } | x _ { 0 } ) } = p _ { \pi ^ { \mathrm { s i o } } } ( \mathbf { o } = 1 | c ) \frac { \pi _ { t } ^ { + } ( x _ { t } | c ) \pi _ { 0 | t } ^ { - } ( x _ { 0 } | x _ { t } , c ) } { \pi ( x _ { t } | x _ { 0 } ) } } \\ & { \qquad + \left[ 1 - p _ { \pi ^ { \mathrm { s i d } } } ( \mathbf { o } = 1 | c ) \right] \frac { \pi _ { t } ^ { - } ( x _ { t } | c ) \pi _ { 0 | t } ^ { - } ( x _ { 0 } | x _ { t } , c ) } { \pi ( x _ { t } | x _ { 0 } ) } } \\ & { \qquad \Rightarrow \pi _ { 0 | t } ^ { \mathrm { o d d } } ( x _ { 0 } | x _ { t } , c ) = p _ { \pi ^ { \mathrm { s i d } } } ( \mathbf { o } = 1 | c ) \frac { \pi _ { t } ^ { + } ( x _ { t } | c ) } { \pi _ { t } ^ { \mathrm { o d d } } ( x _ { t } | c ) } \pi _ { 0 | t } ^ { + } ( x _ { 0 } | x _ { t } , c ) } \\ & { \qquad + \left[ 1 - p _ { \pi ^ { \mathrm { s i d } } } ( \mathbf { o } = 1 | c ) \right] \frac { \pi _ { t } ^ { - } ( x _ { t } | c ) } { \pi _ { t } ^ { \mathrm { o d d } } ( x _ { t } | c ) } \pi _ { 0 | t } ^ { - } ( x _ { 0 } | x _ { t } , c ) } \end{array}
$$

Diffuse both sides of Eq. (9), we have

$$
\begin{array} { r l } & { \pi _ { t } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } | { \pmb c } ) = p _ { \pi ^ { \mathrm { o l d } } } ( { \bf o } = 1 | { \pmb c } ) \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) + [ 1 - p _ { \pi ^ { \mathrm { o l d } } } ( { \bf o } = 1 | { \pmb c } ) ] \pi _ { t } ^ { - } ( { \pmb x } _ { t } | { \pmb c } ) } \\ & { \qquad p _ { \pi ^ { \mathrm { o l d } } } ( { \bf o } = 1 | { \pmb c } ) \frac { \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } | { \pmb c } ) } + [ 1 - p _ { \pi ^ { \mathrm { o l d } } } ( { \bf o } = 1 | { \pmb c } ) ] \frac { \pi _ { t } ^ { - } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \pmb x } _ { t } | { \pmb c } ) } = 1 } \end{array}
$$

Note that

$$
p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | \pmb { c } ) = \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \pmb { x } _ { 0 } | \pmb { c } ) } r ( \pmb { x } _ { 0 } , \pmb { c } )
$$

We have

$$
\pi _ { 0 | t } ^ { \mathrm { o l d } } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) = \alpha ( \pmb { x } _ { t } ) \pi _ { 0 | t } ^ { + } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) + [ 1 - \alpha ( \pmb { x } _ { t } ) ] \pi _ { 0 | t } ^ { - } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } )
$$

Theorem A.3 (Improvement Direction). Consider diffusion models ${ \mathbf { } } v ^ { + } , v ^ { - }$ , and $v ^ { o l d }$ for the distribution triplet $\pi ^ { + } , \pi ^ { - }$ , and $\pi ^ { o l d }$ The directional differences between these models are parallel:

$$
\begin{array} { r l } & { \Delta : = [ 1 - \alpha ( { \pmb x } _ { t } ) ] \left[ v ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) - v ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) \right] \quad ( R e i n f o r c e m e n t G u i d a n c e ) } \\ & { \quad = \quad \alpha ( { \pmb x } _ { t } ) \qquad [ v ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - v ^ { o l d } ( { \pmb x } _ { t } , { \pmb c } , t ) ] . } \end{array}
$$

where $0 \leq \alpha ( { \pmb x } _ { t } ) \leq 1$ is a scalar coefficient:

$$
\alpha ( { \pmb x } _ { t } ) : = \frac { \pi _ { t } ^ { + } ( { \pmb x } _ { t } | { \pmb c } ) } { \pi _ { t } ^ { o l d } ( { \pmb x } _ { t } | { \pmb c } ) } \mathbb { E } _ { \pi ^ { o l d } ( { \pmb x } _ { 0 } | { \pmb c } ) } r ( { \pmb x } _ { 0 } , { \pmb c } )
$$

Proof. According to the relationship between the optimal velocity predictor and the posterior mean of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ (i.e., the optimal $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ predictor) (Zheng et al., 2023b):

$$
\begin{array} { r } { \pmb { v } ^ { \mathrm { o l d } } ( \pmb { x } _ { t } , \pmb { c } , t ) = a _ { t } \pmb { x } _ { t } + b _ { t } \mathbb { E } _ { \pi ^ { \mathrm { o l d } } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) } [ \pmb { x } _ { 0 } ] } \\ { \pmb { v } ^ { + } ( \pmb { x } _ { t } , \pmb { c } , t ) = a _ { t } \pmb { x } _ { t } + b _ { t } \mathbb { E } _ { \pi ^ { + } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) } [ \pmb { x } _ { 0 } ] } \\ { \pmb { v } ^ { - } ( \pmb { x } _ { t } , \pmb { c } , t ) = a _ { t } \pmb { x } _ { t } + b _ { t } \mathbb { E } _ { \pi ^ { - } ( \pmb { x } _ { 0 } | \pmb { x } _ { t } , \pmb { c } ) } [ \pmb { x } _ { 0 } ] } \end{array}
$$

where $\begin{array} { r } { a _ { t } = \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } , b _ { t } = \dot { \alpha } _ { t } - \frac { \dot { \sigma } _ { t } \alpha _ { t } } { \sigma _ { t } } } \end{array}$ Based on Lemma A.2 we have

$$
v ^ { \mathrm { o l d } } ( x _ { t } , c , t ) = \alpha ( x _ { t } ) v ^ { + } ( x _ { t } , c , t ) + [ 1 - \alpha ( x _ { t } ) ] v ^ { - } ( x _ { t } , c , t )
$$

Rearranging the equation, we complete the proof. Theorem A.4 (Reinforcement Guidance Optimization). Consider the training objective:

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { c , \pi ^ { o l d } ( { \pmb x } _ { 0 } \mid c ) , t } r \| { \pmb v } _ { \theta } ^ { + } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } + ( 1 - r ) \| { \pmb v } _ { \theta } ^ { - } ( { \pmb x } _ { t } , { \pmb c } , t ) - { \pmb v } \| _ { 2 } ^ { 2 } ,
$$

where $\begin{array} { r } { \pmb { v } _ { \theta } ^ { + } ( \pmb { x } _ { t } , \pmb { c } , t ) : = ( 1 - \beta ) \pmb { v } ^ { o l d } ( \pmb { x } _ { t } , \pmb { c } , t ) + \beta \pmb { v } _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t ) , } \end{array}$ (Implicit positive policy) and $v _ { \theta } ^ { - } ( x _ { t } , c , t ) : = ( 1 + \beta ) v ^ { o l d } ( x _ { t } , c , t ) - \beta v _ { \theta } ( x _ { t } , c , t )$ . (Implicit negative policy) Given unlimited data and model capacity, the optimal solution of Eq. (10) satisfies

$$
{ \pmb v } _ { \theta ^ { * } } ( { \pmb x } _ { t } , c , t ) = { \pmb v } ^ { o l d } ( { \pmb x } _ { t } , c , t ) + \frac { 2 } { \beta } \Delta ( { \pmb x } _ { t } , c , t ) .
$$

Proof.

$$
\begin{array} { r l } & { \mathrel { \mathop : } ( \theta ) = \mathbb { E } _ { c , t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } , c ) } r ( { \boldsymbol x } _ { 0 } , c ) \| v _ { \theta } ^ { + } ( { \boldsymbol x } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } + [ 1 - r ( { \boldsymbol x } _ { 0 } , c ) ] \| v _ { \theta } ^ { - } ( { \boldsymbol x } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } } \\ & { \qquad = \mathbb { E } _ { c . t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) } \{ \mathbb { E } _ { \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } , c ) } r ( { \boldsymbol x } _ { 0 } , c ) \| v _ { \theta } ^ { + } ( { \boldsymbol x } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } } \\ & { \qquad + \mathbb { E } _ { \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } _ { t } , c ) } [ 1 - r ( { \boldsymbol x } _ { 0 } , c ) ] \| v _ { \theta } ^ { - } ( { \boldsymbol x } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } \} } \end{array}
$$

From Lemma A.1 we have $r ( \pmb { x } _ { 0 } , \pmb { c } ) \pi ^ { \mathrm { o l d } } ( \pmb { x } _ { 0 } | \pmb { c } ) = p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | \pmb { c } ) \pi ^ { + } ( \pmb { x } _ { 0 } | \pmb { c } )$ , therefore:

$$
\begin{array} { r l } & { r ( { \boldsymbol x } _ { 0 } , c ) \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } _ { t } , c ) = r ( { \boldsymbol x } _ { 0 } , c ) \frac { \pi ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { 0 } | c ) \pi ( { \boldsymbol x } _ { t } | { \boldsymbol x } _ { 0 } ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) } } \\ & { \quad \quad \quad \quad \quad = p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) \frac { \pi _ { t } ^ { + } ( { \boldsymbol x } _ { t } | c ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) } \frac { \pi ^ { + } ( { \boldsymbol x } _ { 0 } | c ) \pi ( { \boldsymbol x } _ { t } | { \boldsymbol x } _ { 0 } ) } { \pi _ { t } ^ { + } ( { \boldsymbol x } _ { t } | c ) } } \\ & { \quad \quad \quad \quad = p _ { \pi ^ { \mathrm { o l d } } } ( \mathbf { o } = 1 | c ) \frac { \pi _ { t } ^ { + } ( { \boldsymbol x } _ { t } | c ) } { \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol x } _ { t } | c ) } \pi _ { 0 | t } ^ { + } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } _ { t } , c ) } \\ & { \quad \quad \quad \quad = \alpha ( { \boldsymbol x } _ { t } ) \pi _ { 0 | t } ^ { + } ( { \boldsymbol x } _ { 0 } | { \boldsymbol x } _ { t } , c ) } \end{array}
$$

Similarly,

$$
[ 1 - r ( { \pmb x } _ { 0 } , { \pmb c } ) ] \pi _ { 0 | t } ^ { \mathrm { o l d } } ( { \pmb x } _ { 0 } | { \pmb x } _ { t } , { \pmb c } ) = [ 1 - \alpha ( { \pmb x } _ { t } ) ] \pi _ { 0 | t } ^ { - } ( { \pmb x } _ { 0 } | { \pmb x } _ { t } , { \pmb c } )
$$

Then,

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = \mathbb { E } _ { c , t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol { \alpha } } _ { t } \mid c ) } \{ \alpha ( { \boldsymbol { x } } _ { t } ) \mathbb { E } _ { \pi _ { 0 | t } ^ { + } ( { \boldsymbol { \alpha } } _ { 0 } \mid { \boldsymbol { x } } _ { t } , c ) } \} \| v _ { \theta } ^ { + } ( { \boldsymbol { x } } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } } \\ & { \qquad + \left[ 1 - \alpha ( { \boldsymbol { x } } _ { t } ) \right] \mathbb { E } _ { \pi _ { 0 | t } ^ { - } ( { \boldsymbol { \alpha } } _ { 0 } \mid { \boldsymbol { x } } _ { t } , c ) } \| v _ { \theta } ^ { - } ( { \boldsymbol { x } } _ { t } , c , t ) - v \| _ { 2 } ^ { 2 } \} } \\ & { \qquad = \mathbb { E } _ { c , t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol { \alpha } } _ { t } \mid c ) } \{ \alpha ( { \boldsymbol { x } } _ { t } ) \| v _ { \theta } ^ { + } ( { \boldsymbol { x } } _ { t } , c , t ) - \mathbb { E } _ { \pi _ { 0 | t } ^ { + } ( { \boldsymbol { \alpha } } _ { 0 } \mid { \boldsymbol { x } } _ { t } , c ) } [ v ] \| _ { 2 } ^ { 2 } } \\ & { \qquad + \left[ 1 - \alpha ( { \boldsymbol { x } } _ { t } ) \right] \| v _ { \theta } ^ { - } ( { \boldsymbol { x } } _ { t } , c , t ) - \mathbb { E } _ { \pi _ { 0 | t } ^ { - } ( { \boldsymbol { x } } _ { 0 } \mid { \boldsymbol { x } } _ { t } , c ) } [ v ] \| _ { 2 } ^ { 2 } \} + C _ { 1 } } \\ &  \qquad = \mathbb { E } _ { c , t , \pi _ { t } ^ { \mathrm { o l d } } ( { \boldsymbol { \alpha } } _ { t } \mid c ) } \{ \alpha ( { \boldsymbol { x } } _ { t } ) \| v _ { \theta } ^ { + } \end{array}
$$

Combining Theorem A.3, we observe that

$$
\begin{array} { l } { v _ { \theta } ^ { + } ( x _ { t } , c , t ) - v ^ { + } ( x _ { t } , c , t ) = ( 1 - \beta ) v ^ { \mathrm { o l d } } ( x _ { t } , c , t ) + \beta v _ { \theta } ( x _ { t } , c , t ) - v ^ { + } ( x _ { t } , c , t ) } \\ { \displaystyle \qquad = \beta [ v _ { \theta } - v ^ { \mathrm { o l d } } - \frac { 1 } { \beta } \frac { \Delta } { \alpha ( x _ { t } ) } ] } \\ { v _ { \theta } ^ { - } ( x _ { t } , c , t ) - v ^ { - } ( x _ { t } , c , t ) = ( 1 + \beta ) v ^ { \mathrm { o l d } } ( x _ { t } , c , t ) - \beta v _ { \theta } ( x _ { t } , c , t ) - v ^ { - } ( x _ { t } , c , t ) } \\ { \displaystyle \qquad = - \beta [ v _ { \theta } - v ^ { \mathrm { o l d } } - \frac { 1 } { \beta } \frac { \Delta } { 1 - \alpha ( x _ { t } ) } ] } \end{array}
$$

Substituting these results into $\mathcal { L } ( \boldsymbol { \theta } )$ :

$$
\begin{array} { r l } { \mathcal { L } ( \theta ) = \mathbb { E } _ { \epsilon , 1 , r _ { i } ^ { \mathrm { s a l } } ( x _ { 1 } , \epsilon ) } \{ \alpha ( x _ { 1 } ) \beta ^ { 2 } \| v _ { \theta } - v ^ { \mathrm { o d d } } - \frac { 1 } { \beta } \frac { \Delta } { \alpha ( x _ { k } ) } \| _ { 2 } ^ { 2 } } & { } \\  + \left[ 1 - \alpha ( x _ { k } ) \right] \beta ^ { 2 } \| v _ { \theta } - v ^ { \mathrm { o d d } } - \frac { 1 } { \beta } \frac { \Delta } { 1 - \alpha ( x _ { k } ) } \| _ { 2 } ^ { 2 } \} & { } \\ { = \beta ^ { 2 } \mathbb { E } _ { \epsilon , t , r _ { i } ^ { \mathrm { s a l } } ( x _ { 1 } , \epsilon ) } \{ \alpha ( x _ { 1 } ) \| v _ { \theta } - ( v ^ { \mathrm { o d d } } + \frac { 1 } { \beta } \frac { \Delta } { \alpha ( x _ { k } ) } ) \| _ { 2 } ^ { 2 } } & { } \\ { + \left[ 1 - \alpha ( x _ { k } ) \right] \| v _ { \theta } - ( v ^ { \mathrm { o d d } } + \frac { 1 } { \beta } \frac { \Delta } { 1 - \alpha ( x _ { k } ) } ) \| _ { 2 } ^ { 2 } \} + C _ { 1 } } & { } \\ { - \beta ^ { 2 } \mathbb { E } _ { \epsilon , t , r ^ { \mathrm { s a l } } ( x _ { 1 } , \epsilon ) } \| v _ { \theta } - \alpha ( x _ { k } ) ( v ^ { \mathrm { o d d } } + \frac { 1 } { \beta } \frac { \Delta } { \alpha ( x _ { k } ) } ) - \left[ 1 - \alpha ( x _ { k } ) \right] ( v ^ { \mathrm { a d d } } + \frac { 1 } { \beta } \frac { \Delta } { 1 - \alpha ( x _ { k } ) } ) \| _ { 2 } ^ { 2 } + C _ { 1 } } & { } \\  = \beta ^ { 2 } \mathbb { E } _ { \epsilon , t , r ^ { \mathrm { s a l } } ( x _ { 1 } , \epsilon ) } \| v _   \end{array}
$$

from which it isvious that the tal $\theta ^ { * }$ satisfies $\begin{array} { r } { { v _ { \theta ^ { * } } } ( { x _ { t } } , c , t ) = { v } ^ { \mathrm { o l d } } ( { x _ { t } } , c , t ) + { \frac { 2 } { \beta } } \Delta ( { x _ { t } } , c , t ) . } \end{array}$

# B Theoretical Discussions

# B.1 FLOW SDE

As flow models are a special case of diffusion models under the rectified schedule $\alpha _ { t } = 1 - t , \sigma _ { t } = t$ the earliest results on diffusion SDEs (Song et al., 2020b) can be directly applied without difficulty. FlowGRPO (Liu et al., 2025) and DanceGRPO (Xue et al., 2025) derive the flow SDE with unexplained hyperparameters $\begin{array} { r } { g _ { t } = a \sqrt { \frac { t } { 1 - t } } } \end{array}$ or additional complexity. We provide a simpler and more principled perspective based solely on the diffusion model framework.

To leverage the diffusion SDE formulation in Song et al. (2020b), we need to match its forward SDE $\mathrm { d } \pmb { x } _ { t } = f ( t ) \pmb { x } _ { t } \mathrm { d } t + g ( t ) \mathrm { d } \pmb { w } _ { t }$ with the forward transition kernel ${ \pmb x } _ { t } = \alpha _ { t } { \pmb x } _ { 0 } + \sigma _ { t } { \pmb \epsilon }$ As noted in the first two arXiv versions of the VDM paper (Kingma et al., 2021), $f ( t ) , g ( t )$ are related to $\alpha _ { t } , \sigma _ { t }$ by $\begin{array} { r } { f ( t ) = \frac { \mathrm { d } \log \alpha _ { t } } { \mathrm { d } t } } \end{array}$ d logtα , g2(t) = $\begin{array} { r } { g ^ { 2 } ( t ) = \frac { \mathrm { d } \sigma _ { t } ^ { 2 } } { \mathrm { d } t } - 2 \frac { \mathrm { d } \log { \alpha _ { t } } } { \mathrm { d } t } \sigma _ { t } ^ { 2 } } \end{array}$ Setting $\alpha _ { t } = 1 - t , \sigma _ { t } = t$ we have for rectified flow. According to (Huang et al., 2021), the generalized reverse SDE takes the form:

$$
f ( t ) = - { \frac { 1 } { 1 - t } } , \quad g ^ { 2 } ( t ) = { \frac { 2 t } { 1 - t } }
$$

$$
\mathrm { d } \pmb { x } _ { t } = \left[ f ( t ) \pmb { x } _ { t } - \frac { 1 + \lambda _ { t } ^ { 2 } } { 2 } g ^ { 2 } ( t ) \nabla _ { \pmb { x } _ { t } } \log \pi _ { t } ( \pmb { x } _ { t } ) \right] \mathrm { d } t + \lambda _ { t } g ( t ) \mathrm { d } \bar { \pmb { w } } _ { t }
$$

where $\lambda _ { t } \in [ 0 , 1 ]$ . Equivalently, it amounts to introducing Langevin dynamics on top of the diffusion ODE, with $\lambda _ { t } = 0$ corresponding to ODE, and $\lambda _ { t } = 1$ corresponding to the maximum variance SDE in Song et al. (2020b). The score function ${ \pmb s } _ { \theta } ( { \pmb x } _ { t } , t ) \approx \nabla _ { { \pmb x } _ { t } } \log \pi _ { t } ( { \pmb x } _ { t } )$ , noise predictor $\epsilon _ { \theta } ( x _ { t } , t )$ , data predictor ${ \pmb x } _ { \theta } ( { \pmb x } _ { t } , t )$ and velocity predictor ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t )$ are interconvertible under general noise schedules (Zheng et al., 2023b):

$$
{ \bf \nabla } _ { \theta } ( { \bf x } _ { t } , t ) = - \sigma _ { t } s _ { \theta } ( { \bf x } _ { t } , t ) , \quad { \bf x } _ { \theta } ( { \bf x } _ { t } , t ) = \frac { { \bf x } _ { t } - \sigma _ { t } \epsilon _ { \theta } ( { \bf x } _ { t } , t ) } { \alpha _ { t } } , \quad { \bf v } _ { \theta } ( { \bf x } _ { t } , t ) = \dot { \alpha } _ { t } x _ { \theta } ( { \bf x } _ { t } , t ) + \dot { \sigma } _ { t } \epsilon _ { \theta } ( { \bf x } _ { t } , t )
$$

Applying these relations to the rectified flow schedule, we can derive:

$$
\mathbf { \boldsymbol { s } } _ { \theta } ( \mathbf { \boldsymbol { x } } _ { t } , t ) = - \frac { \mathbf { \boldsymbol { x } } _ { t } + ( 1 - t ) \mathbf { \boldsymbol { v } } _ { \theta } ( \mathbf { \boldsymbol { x } } _ { t } , t ) } { t }
$$

Substituting Eq. (11) and Eq. (14) into Eq. (12), we have the diffusion SDE under rectified flow:

$$
\mathrm { d } \pmb { x } _ { t } = \left[ ( 1 + \lambda _ { t } ^ { 2 } ) \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) + \frac { \lambda _ { t } ^ { 2 } } { 1 - t } \pmb { x } _ { t } \right] \mathrm { d } t + \lambda _ { t } \sqrt { \frac { 2 t } { 1 - t } } \mathrm { d } \pmb { w }
$$

$\begin{array} { r } { g _ { t } = \lambda _ { t } \sqrt { \frac { 2 t } { 1 - t } } } \end{array}$ from the interpolation parameter $\lambda _ { t } \in [ 0 , 1 ]$ to the variance parameter $g _ { t }$ . This also explains the choice $g _ { t } =$ $a { \sqrt { \frac { t } { 1 - t } } }$ in FlowGRPO, where $a = \sqrt { 2 } \lambda _ { t }$ is a scaled version of $\lambda _ { t }$ with $a = { \sqrt { 2 } }$ corresponding to the maximum variance SDE. In comparison, DanceGRPO adopts a fixed variance $g _ { t }$ across timesteps, which is less effective on image models while more stable on video models. FlowGRPO and DanceGRPO directly take the Euler discretization of the flow SDE. In principle, there are more accurate ways, such as utilizing the idea of diffusion implicit models (Song et al., 2020a; Zheng et al., 2024), which is equivalent to the first-order discretization after applying exponential integrators (Hochbruck & Ostermann, 2010; Zhang & Chen, 2022; Gonzalez et al., 2023). Specifically, the sampling step from $t$ to $s < t$ can be derived as:

$$
\begin{array} { r } { \mathbf { r } _ { s } = \left[ ( 1 - s ) + \sqrt { s ^ { 2 } - \rho _ { t } ^ { 2 } } \right] x _ { t } - \left[ ( 1 - s ) t - \sqrt { s ^ { 2 } - \rho _ { t } ^ { 2 } } ( 1 - t ) \right] v _ { \theta } ( x _ { t } , t ) + \rho _ { t } \epsilon , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } \end{array}
$$

where $\begin{array} { r } { \rho _ { t } = \eta _ { t } s \sqrt { 1 - \frac { s ^ { 2 } ( 1 - t ) ^ { 2 } } { t ^ { 2 } ( 1 - s ) ^ { 2 } } } } \end{array}$ $\eta _ { t } \in [ 0 , 1 ]$ SDE. Compared to the Euler discretization, the DDIM-style discretization avoids singularities at boundaries and is expected to reduce sampling errors. However, we did not observe notable advantages by replacing the SDE sampler with stochastic DDIM. Concurrent work (Wang & Yu, 2025) improves the SDE sampler through the Coefficients-Preserving Sampling (CPS) principle.

# B.2 HIgH-ORDER FLOW ODE SAMPLER

We implement the 2nd-order ODE sampler for flow models based on the DPM-Solver series (Lu et al., 2022a;b; Zheng et al., 2023a), which uses the multistep method and half the log signal-to-noise ratio (SNR) $\lambda _ { t } = \log ( \alpha _ { t } / \sigma _ { t } )$ for time discretization. Specifically, for three consecutive timesteps $t _ { i } < t _ { i - 1 } < t _ { i - 2 }$ , where ${ \pmb x } _ { t _ { i - 1 } } , { \pmb x } _ { t _ { i - 2 } }$ are already obtained, the update rule for $\mathbf { x } _ { t _ { i } }$ is:

$$
x _ { t _ { i } } = \frac { \sigma _ { t _ { i } } } { \sigma _ { t _ { i - 1 } } } x _ { t _ { i - 1 } } - \alpha _ { t _ { i } } ( e ^ { - h _ { i } } - 1 ) \left[ \left( 1 + \frac { 1 } { 2 r _ { i } } \right) x _ { \theta } ( x _ { t _ { i - 1 } } , t _ { i - 1 } ) - \frac { 1 } { 2 r _ { i } } x _ { \theta } ( x _ { t _ { i - 2 } } , t _ { i - 2 } ) \right]
$$

where $\begin{array} { r } { h _ { i } = \lambda _ { t _ { i } } - \lambda _ { t _ { i - 1 } } , r _ { i } = \frac { h _ { i - 1 } } { h _ { i } } } \end{array}$ and the data predictor ${ \pmb x } _ { \theta } = { \pmb x } _ { t } - t { \pmb v } _ { \theta }$ for rectified flow. Highorder solvers are also adopted in MixGRPO (Li et al., 2025b) but only for certain steps. Adopting the 2nd-order solver throughout the entire sampling process is infeasible, as $\lambda _ { t }$ will be infinity at boundaries $t = 0$ or $t = 1$ . Following common practices, the first and last steps degrade to the first-order solver, which is the default Euler discretization for flow models.

# B.3 INTUITION BEHIND THE FLOWGRPO OBJECTIVE

We provide some insight into reverse-process diffusion RL by inspecting the FlowGRPO objective in a sampler-agnostic manner. For any first-order SDE sampler, the reverse sampling step from $t$ to $s < t$ can be expressed as where $l ( s , t ) , m ( s , t ) , n ( s , t )$ depend only on $s , t$ and the sampler. Consider the on-policy case and the branching strategy in MixGRPO. Starting from a shared $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ , a group of $N$ noises $\epsilon ^ { ( 1 ) } , \dots , \epsilon ^ { ( N ) }$ aresampled and incorporated into the reverse step to produce multiple samples $\pmb { x } _ { s } ^ { ( 1 ) } , \ldots , \pmb { x } _ { s } ^ { ( N ) }$ .

$$
\pmb { x } _ { s } = l ( s , t ) \pmb { x } _ { t } - m ( s , t ) \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) + n ( s , t ) \pmb { \epsilon } , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )
$$

They go through further sampling, yielding $N$ clean samples and corresponding advantages $A ^ { ( 1 ) } , \dotsc , A ^ { ( N ) }$ On-policy GRPO minimizes the negative advantage-weighted log likelihoods:

$$
\mathcal { L } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } A ^ { ( i ) } \log p _ { \theta } ( x _ { s } ^ { ( i ) } | \pmb { x } _ { t } )
$$

where erge bee the pes $\pmb { x } _ { s } ^ { ( 1 ) } , \ldots , \pmb { x } _ { s } ^ { ( N ) }$ log likelihood w.r.t. can be surprisingly reduced to a simple form:

$$
\begin{array} { c } { { \log p _ { \theta } ( x _ { s } ^ { ( i ) } | x _ { t } ) = - \frac { \| x _ { s } ^ { ( i ) } - ( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) ) \| _ { 2 } ^ { 2 } } { 2 n ^ { 2 } ( s , t ) } + C } } \\ { { = - \frac { \| m ( s , t ) v _ { \theta } ( x _ { t } , t ) - m ( s , t ) v _ { \mathrm { s g } ( \theta ) } ( x _ { t } , t ) + n ( s , t ) \epsilon ^ { ( i ) } \| _ { 2 } ^ { 2 } } { 2 n ^ { 2 } ( s , t ) } + C } } \end{array}
$$

$$
\nabla _ { \boldsymbol { \theta } } \log { p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { s } ^ { ( i ) } | \mathbf { x } _ { t } ) } = - \frac { m ( s , t ) } { n ( s , t ) } \nabla _ { \boldsymbol { \theta } } ( ( \epsilon ^ { ( i ) } ) ^ { \top } \mathbf { v } _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t ) )
$$

and

$$
\nabla _ { \theta } \mathcal { L } ( \theta ) = \frac { m ( s , t ) } { n ( s , t ) } \nabla _ { \theta } \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( A ^ { ( i ) } \epsilon ^ { ( i ) } ) ^ { \top } \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) \right]
$$

Therefore, FlowGRPO essentially aligns the velocity field with the advantage-weighted noise, while $\textstyle { \frac { m ( s , t ) } { n ( s , t ) } }$ across sampling steps. In the following, we show a further conclusion that FlowGRPO can be viewed as $a$ gradient estimation of reward backpropagation. Denote $r _ { t } ( \pmb { x } _ { t } )$ as the implicit gradient-free function that solves the PF-ODE from $t$ to 0 and fetches the reward on the cleaned sample. The rewards can be expressed as

$$
r ^ { ( i ) } = r _ { s } \Big ( l ( s , t ) \pmb { x } _ { t } - m ( s , t ) \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) + n ( s , t ) \pmb { \epsilon } ^ { ( i ) } \Big )
$$

According to Stein's identity, we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r ^ { ( i ) } \epsilon ^ { ( i ) } \approx \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \epsilon \right] } \\ & { \quad \quad \quad \quad = n ( s , t ) \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ \nabla r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \right] } \end{array}
$$

Therefore, where $\sigma$ is the global std used in GRPO normalization. Therefore, the GRPO loss gradient is

$$
\begin{array} { r l } & { \quad \nabla _ { \theta } \left[ \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( A ^ { ( i ) } \epsilon ^ { ( i ) } ) ^ { \top } v _ { \theta } ( x _ { t } , t ) \right] } \\ & { \approx \frac { n ( s , t ) } { \sigma } \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ \nabla r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \nabla _ { \theta } v _ { \theta } ( x _ { t } , t ) \right] } \\ & { = - \displaystyle \frac { n ( s , t ) } { m ( s , t ) \sigma } \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ \nabla _ { \theta } r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \right] } \end{array}
$$

$$
\nabla _ { \theta } \mathcal { L } ( \theta ) \approx - \frac { 1 } { \sigma } \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) } \left[ \nabla _ { \theta } r _ { s } \left( l ( s , t ) x _ { t } - m ( s , t ) v _ { \theta } ( x _ { t } , t ) + n ( s , t ) \epsilon \right) \right]
$$

From the above gradient, GRPO optimizes the reverse transition $t  s$ when the remaining trajectory $s  0$ is gradient-free. Compared to works like ReFL (Xu et al., 2023), which conduct direct gradient backpropagation and approximate $s \to 0$ with a single forward pass ( ${ \bf \delta x } _ { 0 }$ -prediction), GRPO introduces higher estimation variance but avoids backpropagation through the $s \to 0$ process, allowing larger $s$ and a longer sampling chain for $s \to 0$ .

# C Experiment Details

Training Configurations. Our setup largely follows FlowGRPO, adopting the same number of groups per epoch (48), group size (24), LoRA configuration $( \alpha = 6 4 , r = 3 2 )$ , and learning rate $( 3 e \mathrm { ~ - ~ } 4 )$ . For each collected clean image, forward noising and loss computation are performed exactly on the corresponding sampling timesteps. We employ a 2nd-order ODE sampler for data collection and enable adaptive time weighting by default. Single-Reward. For a head-to-head comparison with FlowGRPO under single-reward settings, we fix the number of sampling steps to 10 to ensure fairness. By default, we set $\beta = 1$ and $\eta _ { i } ~ =$ $\operatorname* { m i n } ( 0 . 0 0 1 i , 0 . 5 )$ , which work stably for most reward models. In the case of OCR, the reward rapidly approaches 1 within 100 iterations but suffers from instability. To address this, we adopt a more conservative soft-update strategy with $\eta _ { \mathrm { m a x } } = 0 . 9 9 9$ .

Multi-Reward. To comprehensively improve the base model across multiple rewards, we adopt a multi-stage training scheme. The training setup involves three categories of rewards and datasets: (1) PickScore, CLIPScore, and HPSv2.1 rewards on the Pick-a-Pic dataset; (2) GenEval reward with the three rewards above on the GenEval dataset; and (3) OCR reward with the three rewards above on the OCR dataset. Since the initial CFG-free generation is of low quality, we first train on (1) for 800 iterations to enhance image quality, followed by (2) for 300 iterations, (1) for 200 iterations, (2) for 200 iterations, and finally (3) for 100 iterations. All rewards are equally weighted, with PickScore divided by 26 for normalization to [0, 1]. By default, we use $\beta = 0 . 1$ and $\eta _ { i } = \operatorname* { m i n } ( 0 . 0 0 1 i , 0 . 5 )$ , while setting $\eta _ { \mathrm { m a x } } = 0 . 9 5$ for OCR to stabilize training. The number of sampling steps is fixed to 40 to ensure high-fidelity data collection.

# D ADDITIONAL RESULTS

Table 2: Evaluation results of FlowGRPO and DiffusionNFT trained on single rewards, both initialized from CFG-free base model.Gray-colored: In-domain reward. We observe that training exclusively on the OCR reward impairs generalization to other metrics; to compensate this, we enable CFG when evaluating non-OCR rewards for OCR-trained models.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Iter</td><td colspan="2">Rule-Based</td><td colspan="6">Model-Based</td></tr><tr><td>GenEval</td><td>OCR</td><td>PickScore</td><td>ClipScore</td><td>HPSv2.1</td><td>Aesthetic</td><td>ImgRwd</td><td>UniRwd</td></tr><tr><td>SD3.5-M (w/o CFG)</td><td></td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td>+ CFG</td><td></td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td>+ FlowGRPO</td><td>4k</td><td>0.97</td><td>0.30</td><td>21.78</td><td>0.277</td><td>0.248</td><td>5.15</td><td>0.74</td><td>2.87</td></tr><tr><td rowspan="5">+ Ours</td><td>1k</td><td>0.66</td><td>0.96</td><td>21.94</td><td>0.280</td><td>0.257</td><td>5.18</td><td>0.31</td><td>2.86</td></tr><tr><td>4k</td><td>0.54</td><td>0.60</td><td>23.62</td><td>0.257</td><td>0.295</td><td>6.42</td><td>1.17</td><td>3.17</td></tr><tr><td>1k</td><td>0.98</td><td>0.36</td><td>21.92</td><td>0.271</td><td>0.251</td><td>5.33</td><td>0.68</td><td>2.91</td></tr><tr><td>150</td><td>0.54</td><td>0.97</td><td>21.63</td><td>0.281</td><td>0.246</td><td>5.19</td><td>0.37</td><td>2.81</td></tr><tr><td>2k</td><td>0.53</td><td>0.64</td><td>24.03</td><td>0.270</td><td>0.315</td><td>6.17</td><td>1.29</td><td>3.40</td></tr></table>

We provide more qualitative comparison between the base model, FlowGRPO and our multi-reward optimized model in Figure 11, Figure 12 and Figure 13.

![](images/11.jpg)  
a photo of a brown hot dog and a purple pizza   

Figure 11: Qualitative comparison between FlowGRPO and our model on GenEval prompts.

![](images/12.jpg)  
A close-po amedicine bottle with a prominent warning label that reads "Consul Doctor", set agaist a neutral background, emphasizing the clarity and visibility of the text.

![](images/13.jpg)  
A courtroom scene with a judge's gavel resting on a wooden plaque that reads "Orderin the Cour", s against the backdrop of a quiet, solemn courtroom.

![](images/14.jpg)  
A realistic photo of a tech campus courtyard at night, featuring a glowing "AI Training Zone" hologram fl the futuristic atmosphere.

![](images/15.jpg)  
Anqu ypewrir wi hee  pape nser proe isplayngheyped wors "Chap1 It Wa Dark NighThe en  dimy e ud wi glesk lacg ao over the typewriter.

![](images/16.jpg)  
A ba  a coru e wihher "LieBe  ,

Figure 12: Qualitative comparison between FlowGRPO and our model on OCR prompts.

SD3.5-M SD3.5-M +FlowGRPO +DiffusionNFT (w/o CFG) (w/ CFG) (w/ CFG) (w/o CFG)

![](images/17.jpg)  
A side view of an owl sitting in a field.   

Figure 13: Qualitative comparison between FlowGRPO and our model on DrawBench prompts.