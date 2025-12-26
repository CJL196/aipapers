# Agent-based Video Trimming

Lingfeng $\mathrm { Y a n g ^ { 1 \dagger } }$ , Zhenyuan Chen2†, Xiang $_ { \mathrm { L i ^ { 3 , 2 * } } }$ , Peiyang Jia4, Liangqu Long4, Jian Yang1\* 1Nanjing University of Science and Technology, VCIP, CS, Nankai University, 3NKIARI, Shenzhen Futian, 4Insta360 {yanglfnjust, csjyang}@njust.edu.cn, {zhenyuanchen, xiang.li.implus@}@nankai.edu.cn jiapeiyang@insta360.com, liangqu.long@gmail.com

# Abstract

As information becomes more accessible, user-generated videos are increasing in length, placing a burden on viewers to sift through vast content for valuable insights. This trend underscores the need for an algorithm to extract key video information efficiently. Despite significant advancements in highlight detection, moment retrieval, and video summarization, current approaches primarily focus on selecting specific time intervals, often overlooking the relevance between segments and the potential for segment arranging. In this paper, we introduce a novel task called Video Trimming (VT), which focuses on detecting wasted footage, selecting valuable segments, and composing them into a final video with a coherent story. To address this task, we propose Agent-based Video Trimming (AVT), structured into three phases: Video Structuring, Clip Filtering, and Story Composition. Specifically, we employ a Video Captioning Agent to convert video slices into structured textual descriptions, a Filtering Module to dynamically discard low-quality footage based on the structured information of each clip, and a Video Arrangement Agent to select and compile valid clips into a coherent final narrative. For evaluation, we develop a Video Evaluation Agent to assess trimmed videos, conducting assessments in parallel with human evaluations. Additionally, we curate a new benchmark dataset for video trimming using raw user videos from the internet. As a result, AVT received more favorable evaluations in user studies and demonstrated superior mAP and precision on the YouTube Highlights, TVSum, and our own dataset for the highlight detection task. The code and models are available at https://ylingfeng.github.io/AVT.

# 1. Introduction

With the ever-increasing volume of visual information and daily video content, there is an urgent need for algorithms that can perform video comprehension and effectively distill critical information from redundant content.

![](images/1.jpg)  
Figure 1. A comparison between our new task and existing video tasks: (a) Highlight Detection retrieves clips above a saliency threshold. (b) Moment Retrieval identifies the start and end for intervals related to a given query. (c) Video Summarization extracts keyframes for each theme of the video. (d) Video Trimming addresses more than just a retrieval task by also filtering wasted footage and logically composing the selected segments.

Understanding the video is essential to reducing excessive content. Substantial advancements are made in the field of video understanding [25, 32, 44, 60, 69, 71]. Building on these foundations, methods for highlight detection [22, 35, 41, 43, 58] focus on predicting saliency scores to identify and extract significant segments from videos, thereby reducing the amount of redundant information. Moment retrieval [3, 13, 21, 34, 35] seeks to identify specific moments in a video that correspond to a given query, while video summarization [4, 16, 72] compiles keyframes to capture detected themes within the video. However, current approaches focus solely on content extraction and retrieval, without considering the relationships and coherence between segments. To overcome this limitation, we propose a novel task for the first time: Video Trimming (VT). This task involves not only selecting highsaliency segments but also filtering out wasted footage and arranging the remaining clips, resulting in a logically structured and cohesive video output. A comparison with existing tasks is shown in Fig. 1.

![](images/2.jpg)  
(b) discards defective clips, and finally (c) organizes the remaining clips into a coherent final cut.

To establish a feasible baseline for this task, we consider leveraging the concept of agents. Recent observations indicate that multimodal large language models [1, 2, 9, 45] (MLLMs) exhibit robust capabilities in in-context communication and formatting interactions, positioning them as effective, training-free agents for video understanding tasks such as video captioning [23, 60] and question answering [44, 50, 67]. Agent-based approaches utilize MLLMs with specially designed prompting pipelines to summarize videos into text and organize the summaries based on given inputs [26, 46, 49]. Alternatively, these models can also function as controllers, coordinating various executors, such as tracking and captioning systems, to address complex multimodal video tasks [14, 30, 44, 55, 62]. Despite the extensive use of agent-based approaches in video understanding tasks, this study aims to harness their capabilities to develop the first video trimming algorithm. By integrating MLLMs, our approach targets the editing of long-form videos, which may extend up to half an hour or longer, and trims them into shorter, viewable final cuts. Specifically, Agent-based Video Trimming (AVT) presents innovations across three key phases: Video Structuring, Clip Filtering, and Story Composition (Fig. 2). In the Video Structuring phase, the video is divided into smaller units. A Video Captioning Agent then converts these units into structured textual descriptions, enabling detailed semantic analysis of each segment. Therefore, the subsequent processes are free of visual content, allowing for faster and more efficient results. In addition to generating basic descriptions, we incorporate attributes that denote defects in video segments, such as occlusion, jittering, overexposure, and meaningless content, to evaluate frame quality. The Clip Filtering phase employs a dynamic module to select useful clips by analyzing structured textual descriptions, differentiating between valuable and irrelevant content. In the Story Composition phase, a Video Arrangement Agent assembles the selected clips into a cohesive final video, ensuring a coherent and engaging narrative flow. Additionally, we design a Video Evaluation Agent to assess the quality of the trimmed videos. We also create a new benchmark for video trimming, which includes raw user videos annotated with waste and highlight labels. For evaluation, we conduct user studies and quantitative experiments on zero-shot highlight detection across three benchmarks: YouTube Highlights [43], TVSum [41], and our dataset. Our contributions can be summarized as follows: To the best of our knowledge, we are the first to introduce Video Trimming (VT), a novel task that extracts key intentions from long-form user-shot videos to generate condensed videos with a coherent storyline. To establish a baseline, we propose the AVT algorithm, which converts video content into structured descriptions, filters out wasted clips, and arranges the selected clips into a coherent final narrative video. • We propose a new video trimming benchmark by integrating web videos and using a Video Evaluation Agent to assess video quality alongside human evaluation. Our method demonstrates superior performance in video trimming and zero-shot highlight detection, as evidenced by both user studies and various benchmarks.

# 2. Related Work

# 2.1. Video Understanding

Leveraging MLLMs, a wide range of research studies are conducted to advance video understanding. The existing methods encompass one or multiple tasks, such as video question answering [24, 31, 50, 60, 71], long video understanding [54, 54], and moment localization [67]. InternVid [51] builds a large-scale video-text model through contrastive learning and multi-stage pre-training. InternVideo [50, 52] uses multimodal data to scale video understanding. Models like LaViLa [71] and Valley [31] improve video-based instruction understanding via fine-tuning. Merlin [66] and MovieChat [40] enhance video question answering and long video comprehension. PaLM-E [11] integrates real-world sensory data into language models, while SeViLA [67] uses keyframe localization for event prediction. Vid2Seq [60] and VideoChat [24] achieve chat-centric video understanding through fine-tuning. Recently, models like LongVLM [54] and VTimeLLM [19] improve comprehension of long videos by segmenting and identifying moments. In contrast to addressing generic video tasks, our method specifically targets video trimming, utilizing foundation models as integral components.

# 2.2. Video Agent

Existing video agent methods fall into two main development types. The first type leverages large language models (LLMs) in combination with external tools and code executors. DoraemonGPT [63] enhances VQA tasks by using symbolic memory for better retrieval and summarization. Similarly, InternGPT [30] improves reasoning through interactive polling, while MM-ReAct [62] extends the REACT [64] mechanism to multimodal tasks. Video ChatCaptioner [8] strengthens video understanding with multi-agent iterative polling. The second targets specific video understanding tasks by converting video content into a textual corpus for subsequent analysis. AssistGPT [14] improves VQA and moment retrieval through a cycle of planning, execution, inspection, and learning. ChatVideo [46] structures video content into a text database for efficient querying, while LLoVi [68] focuses on fine-grained VQA and interval retrieval using captions. MM-Vid [26] treats multimodal information as textual data, and VideoAgent [49] improves moment retrieval with iterative polling and similarity matching. The first type of video agent is inadequate for video trimming, as they lack specialized models or tools to solve this task. For the second type, although video retrieval agents can extract segments based on queries, they often neglect crucial global content, compromising video coherence. In contrast, our approach is the first to tackle this challenge by creating an innovative video processing pipeline that incorporates a video agent system.

# 2.3. Video Temporal Grounding

This task aims to ground target clips from videos in the form of continuous intervals or discrete keyframes, encompassing highlight detection, moment retrieval, and video summarization. Firstly, Highlight Detection [5, 6, 17, 41, 43, 58] (HD) predicts saliency scores to extract highlight segments, capturing key visual or contextual moments. However, these highlight-based methods lack the temporal context and event relationships needed for coherent video trimming. Next, Moment Retrieval [3, 15, 18, 21, 34] (MR) selects moments based on a given query. Datasets such as DiDeMo [3], ActivityNet Caption [21], and Charades-STA [15] provide regional captions for video slices to facilitate retrieval tasks. Further, methods such as Moment-DETR [22], QD-DETR [35], TR-DETR [42], UniVTG [27], and UVCOM [57] aim to address both moment and saliency prediction through reciprocal module designs. However, the retrieved segments often lack comprehensive video coverage and context, and they require a prior user query. Finally, Video Summarization condenses videos by selecting key shots that best represent the content of the raw video. Generic summarization [16, 20, 33, 41] relies solely on visual cues, while query-based summarization [27, 36, 38, 56] allows users to tailor the summary by specifying text-based keywords. Although summarization condenses the video, the selected segments are discrete and fail to create a coherent, viewable video. Above all, the video temporal grounding task focuses solely on segment selection, often neglecting narrative flow. In contrast, our proposed video trimming task emphasizes both segment selection and composition, preserving narrative integrity while shortening video duration.

# 3. Method

In this section, we introduce the Video Trimming (VT) task, which extends beyond highlight selection to include filtering out extraneous footage and creating cohesive, narratively coherent video outputs. To establish a baseline for this task, we propose Agent-based Video Trimming (AVT), an algorithm that leverages MLLMs [1, 2, 9, 45] as trainingfree agents across three phases: Video Structuring, Clip Filtering, and Story Composition. Finally, we present an agent-based evaluation metric to assess the final cut, complemented by a user study.

![](images/3.jpg)  
Figure 3. Keyframes from a mountain biking video. Clips marked with red boxes are discarded due to higher defect scores, while clips with green boxes are selected despite minor shaking, as they highlight the dynamic scene of cycling on a mountain path.

# 3.1. Video Structuring

Recent multimodal video tasks perform video understanding by extracting information from visual contexts to derive semantic features [7, 24, 25, 32, 69] or directly generating descriptive text [26, 46, 49]. In our case, we adopt the latter to ensure compatibility with multimodal agents such as GPT-4 [2] or Gemini [45]. Notably, once videos are processed as text, subsequent operations become independent of the visual content, which enhances processing speed and reduces computational costs by handling only text inputs.

To structure the video content, we divide the frames into clips, each with a default duration of 3 seconds. For each clip, frames are sampled at a rate of one per second to provide visual inputs. In contrast to previous works that merely derive generic descriptions of video content, we aim to assess the quality of each clip based on its filming characteristics. Empirically, a handheld recording may contain flawed footage, such as occlusion of the target by obstacles or excessive camera shake, both of which detract from the viewing experience. These defects are only weakly related to the events depicted in the visual cues, therefore they are typically not captured by general summaries. As a result, we need to specifically extract defect attributes to enable a thorough assessment of clip quality. To this end, we identify four primary defects: occlusion, jittering, overexposure, and meaningless content, as illustrated in Fig. 2 (b). Specifically, meaningless content refers to scenes characterized by simple transitions, empty shots devoid of substantive infor

# Algorithm 1 Dynamic Filtering Algorithm

Input: List of Keys keys, List of Numbers nums   
Output:(filter_flag, highlightflag, highlight_score)   
1: Initialize $s c o r e \gets 0$ $m a x \_ k e y \gets ]$ None   
2: for each (key, num) in zip(keys, nums) do   
3: if $n u m \ge s c o r e$ then   
4: $s c o r e \gets n u m$   
5: $m a x _ { - } k e y \gets k e y$   
6: end if   
7: end for   
8: if max_key $=$ [Highlight] then   
9: return (False, True, score)   
10: else   
11: return (score = 0, False, score)   
12:end if mation, or extended static frames. Moreover, to handle lengthy raw videos, it is essential to eliminate redundant segments while preserving highlights and engaging parts. Although raw captions offer detailed information, they are inadequate for segment trimming, as similar visual content can yield diverse textual descriptions. To address this, we introduce contextual attributes that summarize video content across four dimensions: what, where, when, and who, offering brief insights into the activity, location, timing, and potential characters. Additionally, we design a "Highlight" attribute to measure the overall excitement level of each clip. To obtain the aforementioned attributes, we employ the MLLMs as the Video Captioning Agent to extract the Raw Caption, Contextual Attributes, and Defect Attributes for every clip, as shown in Fig. 2 (a). The Clip $I D$ follows a natural sequence related to the length of the video. We expect the structured text information to consist of short sentences or phrases, except for the "Highlight" attribute and all defect attributes, which should return a float value ranging from 0 to 1, indicating the degree to which a clip is a highlight or exhibits specific flaws. If the value is 0, it indicates a negative attribute. These scores will be utilized in the next section to dynamically filter out wasted clips.

# 3.2. Clip Filtering

In contrast to existing moment retrieval methods that score videos based on their alignment with a specific query, we focus exclusively on visual properties to assess the quality of the footage. We gather defect attributes and the highlight score from the output of the Video Captioning Agent, formatted as strings: "[Occlusion]: 0.8; [Jittering]: 0.7; [Overexposure]: 0.0; [Meaningless]: 0.0; [Highlight]: $\it 0 . 9 ^ { * }$ . This output includes four defect indicators and one highlight score, used as inputs for the filtering mechanism. Specifically, a common practice is to filter out all clips

# User Prompt

# User Input

# Story Composition

# Task Introduction

You are a professional video editor specializing in video trimming. Given a series of video clip descriptions, your goal is to select suitable clips that can be composed into a video with a complete story, including a beginning, development, and ending.

# Composition Steps

# 1. General Comprehension

# Clip Information

# Clip ID: 1

Raw Caption: A woman introduces her vlog with a city skyline and continues speaking in her kitchen. Contextual Attributes: [What]: prepare for vlogging; [Where]: kitchen; [When]: evening; [Who]: woman. Highlight Flag: 0.7; Clip ID: 2 . Cip ID: 3 .. Agents   
User Prompt User Input   

Overall comprehension: The video collection   
features a woman introducing her vlog, engaging in daily kitchen activities, and caring for her pet dog. According to the theme, it can be separated into: [2]: Vlog Introduction; [3, 39, 40, 41, 42, 49, 50, 51, 54]: Kitchen Activities and Pet Care; [67, 68, 70, 72, 75, 81]: Urban Exploration and Interaction; [125, 126, 127, 129]: Playtime in the Living Room. Beginning is about a woman introduces her vlog with city skyline and speaking in her kitchen; choose clip id: [2]   
Development follows the summarized themes; choose clip id: [3, 39, 40, 41, 42, 49, 50, 51, 54, ,70,,7,8, 1, 12, 127]   
Ending is about A small white dog energetically plays in a living room, closing the sequence with a lively and cheerful scene; choose clip id: [129]   
Agent Output with any negative defect scores greater than zero. However, this is not always practical. In videos filmed from a first-person perspective, camera shake is inevitable, and the agent would mark clips as jittery, resulting in the exclusion of useful content. To address this, we introduce a positive indicator to balance against the negative ones. The hypothesis is to ensure the algorithm focuses more on the intense video content itself rather than on minor filming imperfections. Based on this strategy, a clip is selected as valid for the final composition only when its "Highlight" score exceeds all defect scores. This mechanism termed the Dynamic Filter, balances content richness with filming defects. As shown in Fig. 3, we visualize diverse frames from clips to demonstrate how this filtering rule is applied.

![](images/4.jpg)

![](images/5.jpg)

For a detailed understanding, we present the string processing algorithm in Alg. 1. The algorithm processes the structured data by parsing it into attribute-score pairs. For the returned value, the defect flag determines whether a clip should be filtered out, while the highlight flag and score are further used in the next phase of story composition.

# 3.3. Story Composition

In this section, we introduce an agent for story composition, which arranges filtered clips into a coherent order. For the user prompt, we present the video agent with an introduction to the task and involve Chain of Thought [53] (CoT) to generate the video composition steps, taking into account the global concept, clip selection, and composition arrangement (Fig. 4). We denote the entire user prompt as $P$ .Then, assuming we obtain $M$ valid clips with indices $C = \{ C _ { 1 } , C _ { 2 } , \dots , C _ { M } \}$ , we format the user input $I$ by concatenating the structured information as follows:

$$
\begin{array} { c } { { I = \{ \{ C l i p I D \} _ { k } , \{ H i g h l i g h t F l a g \} _ { k } ( \{ S c o r e \} _ { k } ) , } } \\ { { \{ C o n t e x t u a l A t t r i b u t e s \} _ { k } , } } \\ { { \{ R a w C a p t i o n \} _ { k } \} \mid _ { C l i p - k } \{ C _ { 1 } \sim C _ { M } \} , } } \end{array}
$$

where the Highlight Flag and the corresponding Score are derived from the filtering phase, while the remaining information is obtained from the structuring phase. Next, we prompt the Video Arrangement Agent with $P$ and $I$ , expecting the output storyline to consist of the preferred sequence, which is generated through a mapping operation from each sentence to its corresponding clip index, arranged in a sensible order. After processing, we expect outputs that include the sequence of composite clips, denoted as $C ^ { t }$ , along with the narrative and the reasoning behind their organization, as illustrated in Fig. 2 (c). The composition phase may be iterated until the desired length of the output video is achieved. It is crucial to note that processing too many clips at once can result in ambiguous outputs, as LLMs face difficulties with long contexts and are prone to distraction [39]. Therefore, we group the clips and call the agent in parallel for the initial processing. As the number of clips decreases, all information is integrated into the final composition. Subsequently, we map the final clip indices back to their respective video durations and assemble them to form the final video. It is important to note that not all clips will be selected, and the sequence of clips may not strictly adhere to chronological order; rather, they will be arranged according to the storyline organized by the agent. For details on the prompt design, see the Supplementary Materials.

Table 1. User study through blind testing, using a scale from 1 to 10, of different methods on the video trimming dataset, comprising 30 final cuts from 42 raw videos and involving 17 participants.   

<table><tr><td>Method</td><td>Richness</td><td>Appeal</td><td>Excitement</td><td>Wasted</td><td>Overall</td><td> Agent</td></tr><tr><td>UniVTG [27]</td><td>6.41</td><td>7.15</td><td>4.74</td><td>6.04</td><td>6.30</td><td>3.03</td></tr><tr><td>UVCOM [57]</td><td>6.15</td><td>7.12</td><td>4.69</td><td>6.47</td><td>6.23</td><td>2.91</td></tr><tr><td>AVT (ours)</td><td>7.21</td><td>7.78</td><td>5.57</td><td>6.72</td><td>7.15</td><td>3.32</td></tr></table>

Existing approaches primarily focus on image-text matching, emphasizing the accuracy of retrieved video moments and aiming to maximize their overall salience. However, in the context of video trimming, this is not the sole objective. A well-told story should prioritize the most prominent activities, while also incorporating the introductory and concluding segments. Although the beginning and end may appear less prominent than the main content, they are essential. We emphasize the importance of frame composition, which is achieved not only through the arrangement of clips but also by incorporating slightly less highlighted segments. These segments can effectively serve as transitions, bridging the preceding and subsequent content.

# 3.4. Final Cut Evaluation

As indicated in G-Eval [29], LLM-based evaluators possess the ability to assess the quality of natural language generation. We extend this automatic evaluation to multimodal tasks by utilizing LLM as the Video Evaluation Agent to assess final videos. Directly prompting MLLMs for aesthetic evaluation often aligns poorly with human assessments [12, 47]. To improve precision, we define evaluation criteria and create CoT instructions for the Video Evaluation Agent. The criteria, rated from 1 to 5, include Material Richness, assessing diversity and narrative coherence; Appeal, measuring engagement, length, and entertainment; Exciting Segments, evaluating highlight quality and frequency; and Amount of Wasted Footage, considering irrelevant content, with higher scores indicating fewer distractions and better viewing. The Video Evaluation Agent uses only video content as input and outputs scores for each metric, along with justifications. For example: "[Material Richness]: {Reason} (2.5); [Appeal]: {Reason} (3.0); [Exciting Segments]: {Reason} (3.5); [Amount of Wasted Footage]: $\{ R e a s o n \}$ (2.0);." We calculate the average of all scores to determine the final rating of a video cut.

# 4. Experiments

In this section, we first introduce the dataset and implementation details. We then compare the quality of the trimmed videos, along with quantitative experiments on highlight detection. Next, ablation studies on the main components of AVT are presented. Lastly, we provide visualizations of the results and case studies for further discussion.

# 4.1. Datasets

Existing Dataset. The YouTube Highlights [43] and TVSum [41] datasets are two well-established benchmarks for evaluating video temporal grounding tasks. We tested on $20 \%$ of the data, following the same split as in [27, 28]. For evaluation metrics, we followed the methodology described in [28], using mAP for the YouTube Highlights dataset and Top-5 mAP for the TVSum dataset. Video Trimming Dataset. Additionally, we collect webcrawled videos from YouTube and construct a benchmark specifically for video trimming. We categorize common user video types into three groups: daily life, sports, and travel vlogs. For each category, we select 10 video uploaders and choose one or more videos filmed around a consistent event, meaning the algorithm may be requested to composite video cuts from multiple source videos. In total, we compile 30 topics with 42 videos, each averaging 10 minutes in length. We annotate each video with four ranks of scores: 0 for wasted, 1 for ambiguous, 2 for normal, and 3 for highlight footage. Detailed illustrations are provided in the Supplementary Materials.

# 4.2. Implementation Details

To enhance multimodal interaction capabilities and ensure restricted output formatting, we implement all agents in our AVT algorithm using the GPT-4o model [37]. For the visual inputs, the video is divided into 3-second segments, with one keyframe sampled per second by default. All frame images are resized to 512 pixels on the shorter side. The text inputs include both our designed prompting instructions and the structured video captions and attributes. This configuration results in approximately 153,000 input image tokens, 100,000 input text tokens, and 20,000 output text tokens for a single 10-minute raw video, which costs approximately $\$ 0.83$ on the API. The output videos are restricted to around one minute to ensure fair comparisons. Further details can be found in the Supplementary Materials.

# 4.3. Comparisons on Video Trimming

Human Evaluation. We conduct a user study based on our constructed video trimming dataset. We design a blind test by randomly shuffling the order of output videos from each method. For the existing highlight detection methods, we utilize their pretrained models to generate saliency scores and obtain final videos by concatenating the video intervals with top scores. Seventeen participants are asked to score these videos across five aspects, similar to the criteria of the Video Evaluation Agent: Material Richness, Appeal, Exciting Segments, Amount of Wasted Footage, and an overall perception score. Tab. 1 shows that our AVT achieves overall improvements in the quality of the final cut, benefiting from the filtering of wasted footage and the clip composition process. We also display the agent evaluation score in the rightmost column, which is consistent with the ranking from human evaluation.

Table 2. Evaluation agent performance of different methods on the validation set of YouTube Highlights and TVSum dataset.   

<table><tr><td>Dataset</td><td>| Method</td><td>Richness</td><td></td><td>Appeal Excitement</td><td>Wasted</td><td>Average</td></tr><tr><td rowspan="4">e</td><td> UMT [28]</td><td>2.70</td><td>3.08</td><td>3.40</td><td>3.44</td><td>3.15</td></tr><tr><td>UniVTG [27]</td><td>2.67</td><td>3.06</td><td>3.35</td><td>3.39</td><td>3.12</td></tr><tr><td>UVCOM [57]</td><td>2.72</td><td>3.10</td><td>3.45</td><td>3.45</td><td>3.18</td></tr><tr><td>AVT (ours)</td><td>2.79</td><td>3.17</td><td>3.53</td><td>3.44</td><td>3.23</td></tr><tr><td rowspan="5">unsaL</td><td>| PGL-SUM [4]</td><td>2.75</td><td>3.05</td><td>3.10</td><td>3.10</td><td>3.00</td></tr><tr><td>UniVTG [27]</td><td>2.65</td><td>2.95</td><td>2.85</td><td>3.15</td><td>2.90</td></tr><tr><td>UVCOM [57]</td><td>2.50</td><td>2.80</td><td>2.70</td><td>3.30</td><td>2.83</td></tr><tr><td>AVT (ours)</td><td>3.15</td><td>3.35</td><td>3.25</td><td>3.70</td><td>3.36</td></tr></table>

Agent Evaluation. Following Sec. 3.4, we evaluate the quality of the generated video using our designed Video Evaluation Agent. We conduct experiments on the validation sets of the YouTube Highlights and TVSum datasets, using 150 and 10 videos, respectively. We compare our approach with three related previous methods. Tab. 2 show that our AVT achieves higher metrics than existing methods. Notably, the ablation study on the consistency with human evaluation is further elaborated in Sec. 4.5.

# 4.4. Comparisons on Highlight Detection

In this section, we compare our method with previous highlight detection and video summarization approaches. We denote the saliency score of AVT as follows:

$$
S _ { i } = { \left\{ \begin{array} { l l } { S _ { h } } & { { \mathrm { i f } } \ S _ { h } > m a x ( S _ { d } ) , } \\ { S _ { h } - m a x ( S _ { d } ) } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

where $i$ represents the $C l i p ~ I D$ , and $S _ { h }$ and $S _ { d }$ denote the highlight and defect scores in the structured clip description. We first conduct experiments on YouTube Highlights and TVSum. We report the results from the original papers of the fully/weakly supervised methods. Then, we compare the methods in terms of zero-shot transfer. Notably, for UniVTG [27], we directly copy their zero-shot result. For others, we infer with their models pretrained on the largest scaled datasets, such as QVHighlights [22] and CharadesSTA [15]. We follow the validation splits of [27, 28] to compare on the YouTube Highlights dataset. As the scale of TVSum is small, with its validation set containing only 10 videos, which may yield inconsistent scores, we measure all videos up to 50 for the zero-shot settings. As shown in Tab. 3 and Tab. 4, our AVT achieves state-of-the-art highlight detection performance under zero-shot transfer and is comparable to partially supervised methods with training.

![](images/6.jpg)  
Figure 5. Highlight detection results of mAP and precision on our collected video trimming dataset.

Next, using our constructed video trimming dataset, we show the mAP of highlight detection with existing methods and the precision of the highlight segments in the selected clips for the final video. In Fig. 5, we observe that there is less wasted footage selected in AVT videos than in previous methods, as they do not focus on footage filtering. Additionally, our method derives more highlight segments.

# 4.5. Ablation Study

Components of AVT. In this section, we analyze the effectiveness of each component within AVT, including the structuring phase, filtering phase, and dynamic module. Additionally, we compare the results of replacing the story composition phase with a simple concatenation of video slices in temporal order. For all experiments without the composition phase, clips with the top saliency scores are selected. For the control condition, with all components disabled, we randomly select video clips. We conducted a user study and quantitative precision measurements on the waste/highlight footage ratios to assess the quality of the generated videos. Tab. 5 shows that clip filtering significantly reduces wasted footage, while the composition process enhances overall video impressions. Notably, without the dynamic filter module, particularly for sports content, highlight segments may be discarded by the defect attribute. Further analysis is in the Supplementary Materials. Human Correlation of Evaluation Agent. Following GEval [29], we adopt three meta-evaluation metrics: Pearson $( r )$ , Spearman $( \rho )$ , and Kendall-Tau $( \tau )$ , to measure the correlation between our evaluation agent and human preferences. We perform ablation studies on the prompt settings to investigate the impact of requesting the agent to provide reasons alongside the score, as well as the effect of using the diverse criteria outlined in Sec. 3.4. With both strategies activated, we achieve an average correlation of 0.5247 with human ratings, as shown in Tab. 6.

# 4.6. Visualization

We visualize the saliency scores and the selected intervals of each method in Fig. 6. Since existing methods do not include a composition phase, their final video is constructed by concatenating high-salient footage, resulting in an inconsistent viewing experience and limited content richness. AVT surpasses previous works by selecting more highlight footage and less wasted footage while maintaining a consistent storyline with the raw videos.

<table><tr><td>Method</td><td>Sup</td><td>Dog</td><td>Gym.</td><td>Par.</td><td>Ska.</td><td>Ski.</td><td>Sur.</td><td>Avg.</td></tr><tr><td>LSVM [43]</td><td>FS</td><td>60.0</td><td>41.0</td><td>61.0</td><td>62.0</td><td>36.0</td><td>61.0</td><td>53.6</td></tr><tr><td>Trailer [48]</td><td>FS</td><td>63.3</td><td>82.5</td><td>62.3</td><td>52.9</td><td>74.5</td><td>79.3</td><td>69.1</td></tr><tr><td>SL-Module [59]</td><td>FS</td><td>70.8</td><td>53.2</td><td>77.2</td><td>72.5</td><td>66.1</td><td>76.2</td><td>69.3</td></tr><tr><td>Joint-VA† [5]</td><td>FS</td><td>64.5</td><td>71.9</td><td>80.8</td><td>62.0</td><td>73.2</td><td>78.3</td><td>71.8</td></tr><tr><td>UMT† [28]</td><td>FS</td><td>65.9</td><td>75.2</td><td>81.6</td><td>71.8</td><td>72.3</td><td>82.7</td><td>74.9</td></tr><tr><td>UniVTG [27]</td><td>FS</td><td>74.3</td><td>79.0</td><td>74.4</td><td>84.9</td><td>75.1</td><td>83.9</td><td>78.6</td></tr><tr><td>UVCOM [57]</td><td>FS</td><td>73.8</td><td>77.1</td><td>75.7</td><td>75.3</td><td>74.0</td><td>82.7</td><td>76.4</td></tr><tr><td>LIM-S [58]</td><td>WS</td><td>57.9</td><td>41.7</td><td>67.0</td><td>57.8</td><td>48.6</td><td>65.1</td><td>56.4</td></tr><tr><td>MINI-Net† [18]</td><td>WS</td><td>58.2</td><td>61.7</td><td>70.2</td><td>72.2</td><td>58.7</td><td>65.1</td><td>64.4</td></tr><tr><td>TCG† [65]</td><td>WS</td><td>55.4</td><td>62.7</td><td>70.9</td><td>69.1</td><td>60.1</td><td>59.8</td><td>63.0</td></tr><tr><td>RRAE [61]</td><td>ZS</td><td>49.0</td><td>35.0</td><td>50.0</td><td>25.0</td><td>22.0</td><td>49.0</td><td>38.3</td></tr><tr><td>UniVTG [27]</td><td>ZS</td><td>48.8</td><td>57.5</td><td>59.4 39.7</td><td></td><td>57.4</td><td>49.1</td><td>52.0</td></tr><tr><td>UVCOM [57]</td><td>ZS</td><td>46.6</td><td>67.4</td><td>61.4</td><td>57.2</td><td>63.5</td><td>60.9</td><td>59.5</td></tr><tr><td>AVT (ours)</td><td>ZS</td><td>58.0</td><td>62.1</td><td>76.1</td><td>32.0</td><td>67.1</td><td>67.9</td><td>60.5</td></tr></table>

<table><tr><td>Method</td><td>Sup</td><td>VT</td><td>VU</td><td>GA</td><td>MS</td><td>PK</td><td>PR</td><td>FM</td><td>BK</td><td>BT</td><td>DS</td><td>Avg.</td></tr><tr><td>sLSTM [70]</td><td>FS</td><td>41.1</td><td>46.2</td><td>46.3</td><td>47.7</td><td>44.8</td><td>46.1</td><td>45.2</td><td>40.6</td><td>47.1</td><td>45.5</td><td>45.1</td></tr><tr><td>Trailer [48]</td><td>FS</td><td>61.3</td><td>54.6</td><td>65.7</td><td>60.8</td><td>59.1</td><td>70.1</td><td>58.2</td><td>64.7</td><td>65.6</td><td>68.1</td><td>62.8</td></tr><tr><td>SL-Module [59]</td><td>FS</td><td>86.5</td><td>68.7</td><td>74.9</td><td>86.2</td><td>79.0</td><td>63.2</td><td>58.9</td><td>72.6</td><td>78.9</td><td>64.0</td><td>73.3</td></tr><tr><td>Joint-VA† [5]</td><td>FS</td><td>83.7</td><td>57.3</td><td>78.5</td><td>86.1</td><td>80.1</td><td>69.2</td><td>70.0</td><td>73.0</td><td>97.4</td><td>67.5</td><td>76.3</td></tr><tr><td>UMT [28]</td><td>FS</td><td>87.5</td><td>81.5</td><td>88.2</td><td>78.8</td><td>81.5</td><td>87.0</td><td>76.0</td><td>86.9</td><td>84.4</td><td>79.6</td><td>83.1</td></tr><tr><td>UniVTG [27]</td><td>FS</td><td>92.0</td><td>77.8</td><td>89.8</td><td>83.8</td><td>82.2</td><td>85.8</td><td>74.3</td><td>91.8</td><td>90.5</td><td>77.6</td><td>84.6</td></tr><tr><td>UVCOM [57]</td><td>FS</td><td>87.6</td><td>91.6</td><td>91.4</td><td>86.7</td><td>86.9</td><td>86.9</td><td>76.9</td><td>92.3</td><td>87.4</td><td>75.6</td><td>86.3</td></tr><tr><td>LIM-S [58]</td><td>WS</td><td>55.9</td><td>42.9</td><td>61.2</td><td>54.0</td><td>60.4</td><td>47.5</td><td>43.2</td><td>66.3</td><td>69.1</td><td>62.6</td><td>56.3</td></tr><tr><td>MINI-Net† [18]</td><td>WS</td><td>80.6</td><td>68.3</td><td>78.2</td><td>81.8</td><td>78.1</td><td>65.8</td><td>57.8</td><td>75.0</td><td>80.2</td><td>65.5</td><td>73.2</td></tr><tr><td>TCG† [65]</td><td>WS</td><td>85.0</td><td>71.4</td><td>81.9</td><td>78.6</td><td>80.2</td><td>75.5</td><td>71.6</td><td>77.3</td><td>78.6</td><td>68.1</td><td>76.8</td></tr><tr><td>SG [33]</td><td>ZS</td><td>42.3</td><td>47.2</td><td>47.5</td><td>48.9</td><td>45.6</td><td>47.3</td><td>46.4</td><td>41.7</td><td>48.3</td><td>46.6</td><td>46.2</td></tr><tr><td>UniVTG [27]</td><td>ZS</td><td>52.0</td><td>48.1</td><td>50.9</td><td>56.9</td><td>51.6</td><td>43.3 60.0 64.0 59.2 54.9</td><td></td><td></td><td></td><td></td><td>54.1</td></tr><tr><td>UVCOM [57]</td><td>ZS</td><td>63.4</td><td>44.5</td><td>50.6</td><td>67.6</td><td>55.1</td><td>42.0</td><td>47.5</td><td>56.9</td><td>58.6</td><td>39.3</td><td>52.5</td></tr><tr><td>AVT (ours)</td><td>ZS</td><td>76.6</td><td>75.9</td><td>62.4</td><td>63.9</td><td>76.6</td><td>68.8</td><td>39.4</td><td>45.6</td><td>43.4</td><td></td><td>62.9 61.6</td></tr></table>

![](images/7.jpg)  
Table 3. Highlight detection results of mAP on YouTube Highlights. $\dagger$ denotes using audio modality.   
Table 4. Highlight detection results of Top-5 mAP on TVSum. $^ \dagger$ denotes using audio modality. FS: Fully supervised. WS: Weakly supervised. ZS: Zero-shot.   
footage and less wasted footage.

Table 5. Ablation study on the effectiveness of AVT components. VS: Video Structuring. CF: Clip Filtering. DF: Dynamic Filter. SC: Story Composition.   

<table><tr><td>Method</td><td>VS</td><td>CF</td><td>DF</td><td>SC</td><td>User ↑</td><td>Waste ↓</td><td>Highlight ↑</td></tr><tr><td rowspan="2">UniVTG [27] UVCOM [57]</td><td></td><td>-</td><td>-</td><td>-</td><td>6.30</td><td>0.276</td><td>0.066</td></tr><tr><td></td><td>-</td><td></td><td></td><td>6.23</td><td>0.175</td><td>0.066</td></tr><tr><td rowspan="6">AVT (ours)</td><td></td><td></td><td>-</td><td>-</td><td>3.70</td><td>0.337</td><td>0.083</td></tr><tr><td></td><td></td><td></td><td>-</td><td>6.19</td><td>0.135</td><td>0.110</td></tr><tr><td></td><td></td><td>-</td><td></td><td>6.45</td><td>0.165</td><td>0.096</td></tr><tr><td></td><td></td><td></td><td></td><td>6.70</td><td>0.141</td><td>0.109</td></tr><tr><td></td><td></td><td></td><td>L</td><td>5.23</td><td>0.199</td><td>0.107</td></tr><tr><td></td><td></td><td>V</td><td>L</td><td>7.15</td><td>0.083</td><td>0.108</td></tr></table>

Table 6. Pearson $( r )$ , Spearman $( \rho )$ , and Kendall-Tau $( \tau )$ correlations of different metrics on video trimming benchmark.   

<table><tr><td>Output Reason Diverse Criteria</td><td></td><td>r</td><td>ρ</td><td>τ</td><td>Avg.</td></tr><tr><td>-</td><td>-</td><td>0.2675</td><td>0.2451</td><td>0.1723</td><td>0.2283</td></tr><tr><td>-</td><td>✓</td><td>0.4082</td><td>0.4119</td><td>0.3067</td><td>0.3756</td></tr><tr><td>✓</td><td>-</td><td>0.5260</td><td>0.4990</td><td>0.3738</td><td>0.4663</td></tr><tr><td>✓</td><td>✓</td><td>0.5616</td><td>0.5667</td><td>0.4457</td><td>0.5247</td></tr></table>

# 5. Conclusion

In this paper, we introduce the novel task of Video Trimming (VT), which focuses on segment selection and narrative preservation to extract meaningful insights from redundant content. To tackle this task, we propose Agent-based Video Trimming (AVT), a baseline framework with three key phases: Video Structuring, where a Video Captioning Agent provides segment descriptions; Clip Filtering, which dynamically selects clips using a filtering module; and Story Composition, where a Video Arrangement Agent creates a cohesive narrative. Further, a Video Evaluation Agent is designed to assess video quality. We construct a benchmark annotated for video trimming tasks. Our approach outperforms existing methods in highlight detection and demonstrates superior human preference in user studies.

# References

[1] Claude 3. Introducing the next generation of claude. https://www.anthropic.com/news/claude-3- family, 2024. 2, 3 [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,   
2023. 2, 3, 4 [3] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, 2017. 1,   
3 [4] Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris, and Ioannis Patras. Combining global and local attention with positional encoding for video summarization. In ISM, 2021. 1, 7 [5] Taivanbat Badamdorj, Mrigank Rochan, Yang Wang, and Li Cheng. Joint visual and audio learning for video highlight detection. In ICCV, 2021. 3, 8 [6] Taivanbat Badamdorj, Mrigank Rochan, Yang Wang, and Li Cheng. Contrastive learning for unsupervised video highlight detection. In CVPR, 2022. 3 [7] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, et al. Videollm: Modeling video sequence with large language models. arXiv preprint arXiv:2305.13292, 2023. 4 [8] Jun Chen, Deyao Zhu, Kilichbek Haydarov, Xiang Li, and Mohamed Elhoseiny. Video chatcaptioner: Towards enriched spatiotemporal descriptions. arXiv preprint arXiv:2304.04227, 2023. 3 [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 2, 3   
10] Sandra Eliza Fontes De Avila, Ana Paula Brandao Lopes, Antonio da Luz Jr, and Arnaldo de Albuquerque Araújo. Vsumm: A mechanism designed to produce static video summaries and a novel evaluation method. Pattern recognition letters, 2011. 13   
11] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 3   
[12] Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166, 2023. 6   
[13] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal transformer for video retrieval. In ECCV, 2020. 1   
[14] Difei Gao, Lei Ji, Luowei Zhou, Kevin Qinghong Lin, Joya Chen, Zihan Fan, and Mike Zheng Shou. Assistgpt: A general multi-modal assistant that can plan, execute, inspect, and learn. arXiv preprint arXiv:2306.08640, 2023. 2, 3   
[15] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In ICCV, 2017. 3, 7, 13   
[16] Michael Gygli, Helmut Grabner, Hayko Riemenschneider, and Luc Van Gool. Creating summaries from user videos. In ECCV, 2014. 1, 3, 13   
[17] Michael Gygli, Yale Song, and Liangliang Cao. Video2gif: Automatic generation of animated gifs from video. In CVPR, 2016.3   
[18] Fa-Ting Hong, Xuanteng Huang, Wei-Hong Li, and WeiShi Zheng. Mini-net: Multiple instance ranking network for video highlight detection. In ECCV, 2020. 3, 8   
[19] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In CVPR, 2024. 3   
[20] Hao Jiang and Yadong Mu. Joint video summarization and moment localization by cross-task sample transfer. In CVPR, 2022. 3   
[21] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In ICCV, 2017. 1, 3   
[22] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. NeurIPS, 2021. 1, 3, 7   
[23] Chenliang Li, Haiyang Xu, Junfeng Tian, Wei Wang, Ming Yan, Bin Bi, Jiabo Ye, Hehong Chen, Guohai Xu, Zheng Cao, et al. mplug: Effective and efficient vision-language learning by cross-modal skip-connections. arXiv preprint arXiv:2205.12005, 2022. 2   
[24] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 3, 4   
[25] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 1, 4   
[26] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). arXiv preprint arXiv:2310.19773, 2023. 2, 3, 4   
[27] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding. In ICCV, 2023. 3, 6, 7, 8, 17   
[28] Ye Liu, Siyuan Li, Yang Wu, Chang-Wen Chen, Ying Shan, and Xiaohu Qie. Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. In CVPR, 2022. 6, 7, 8   
[29] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634, 2023. 6, 7   
[30] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, et al. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language. arXiv preprint arXiv:2305.05662, 2023. 2, 3   
[31] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Da Li, Pengcheng Lu, Tao Wang, Linmei Hu, Minghui Qiu, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207, 2023. 3   
[32] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024. 1, 4   
[33] Behrooz Mahasseni, Michael Lam, and Sinisa Todorovic. Unsupervised video summarization with adversarial lstm networks. In CVPR, 2017. 3, 8   
[34] Niluthpol Chowdhury Mithun, Sujoy Paul, and Amit K RoyChowdhury. Weakly supervised video moment retrieval from text queries. In CVPR, 2019. 1, 3   
[35] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. Query-dependent video representation for moment retrieval and highlight detection. In CVPR, 2023. 1, 3   
[36] Saiteja Nalla, Mohit Agrawal, Vishal Kaushal, Ganesh Ramakrishnan, and Rishabh Iyer. Watch hours in minutes: Summarizing videos with user intent. In ECCV, 2020. 3   
[37] OpenAI. Gpt-4o. https://openai.com/index/ hello-gpt-40, 2024. 6, 12   
[38] Aidean Sharghi, Jacob S Laurel, and Boqing Gong. Queryfocused video summarization: Dataset, evaluation, and a memory network based approach. In CVPR, 2017. 3   
[39] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In ICML, 2023. 5   
[40] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In CVPR, 2024. 3   
[41] Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. Tvsum: Summarizing web videos using titles. In CVPR, 2015. 1, 2, 3, 6, 13   
[42] Hao Sun, Mingyao Zhou, Wenjing Chen, and Wei Xie. Trdetr: Task-reciprocal transformer for joint moment retrieval and highlight detection. arXiv preprint arXiv:2401.02309, 2024. 3   
[43] Min Sun, Ali Farhadi, and Steve Seitz. Ranking domainspecific highlights by analyzing edited videos. In ECCV, 2014. 1, 2, 3, 6, 8, 13   
[44] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In ICCV, 2023. 1, 2   
[45] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 3, 4   
[46] Junke Wang, Dongdong Chen, Chong Luo, Xiyang Dai, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. Chatvideo: A tracklet-centric multimodal and versatile video understanding system. arXiv preprint arXiv:2304.14407, 2023. 2, 3, 4   
[47] Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. Is chatgpt a good nlg evaluator? a preliminary study. arXiv preprint arXiv:2303.04048, 2023. 6   
[48] Lezi Wang, Dong Liu, Rohit Puri, and Dimitris N Metaxas. Learning trailer moments in full-length movies with cocontrastive attention. In ECCV, 2020. 8   
[49] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena YeungLevy. Videoagent: Long-form video understanding with large language model as agent. ECCV, 2024. 2, 3, 4   
[50] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 2, 3   
[51] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 3, 17   
[52] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. ECCV, 2024. 3, 17   
[53] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022. 5   
[54] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. ECCV, 2024. 3   
[55] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visal chatt: Talkig, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023. 2   
[56] Guande Wu, Jianzhe Lin, and Claudio T Silva. Intentvizor: Towards generic query guided interactive video summarization. In CVPR. 2022. 3 [72] Kaiyang Zhou, Yu Qiao, and Tao Xiang. Deep reinforcement learning for unsupervised video summarization with diversity-representativeness reward. In AAAI, 2018. 1

[57] Yicheng Xiao, Zhuoyan Luo, Yong Liu, Yue Ma, Hengwei X  : A unified video comprehension framework for moment retrieval and highlight detection. In CVPR, 2024. 3, 6, 7, 8,   
17 [58] Bo Xiong, Yannis Kalantidis, Deepti Ghadiyaram, and Kristen Grauman. Less is more: Learning highlight detection from video duration. In CVPR, 2019. 1, 3, 8 [59] Minghao Xu, Hang Wang, Bingbing Ni, Riheng Zhu, Zhenbang Sun, and Changhu Wang. Cross-category video highlight detection via set-based learning. In ICCV, 2021. 8 [60] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In CVPR,   
2023. 1, 2, 3 [61] Huan Yang, Baoyuan Wang, Stephen Lin, David Wipf, Minyi Guo, and Baining Guo. Unsupervised extraction of video highlights via robust recurrent auto-encoders. In ICCV,   
2015.8 [62] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023. 2, 3 [63] Zongxin Yang, Guikun Chen, Xiaodi Li, Wenguan Wang, and Yi Yang. Doraemongpt: Toward understanding dynamic scenes with large language models. arXiv preprint arXiv:2401.08392, 2024. 3 [64] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In ICLR, 2023. 3 [65] Qinghao Ye, Xiyue Shen, Yuan Gao, Zirui Wang, Qi Bi, Ping Li, and Guang Yang. Temporal cue guided video highlight detection with low-rank audio-visual fusion. In ICCV, 2021.   
8 [66] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. ECCV, 2024. 3 [67] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. NeurIPS, 2023. 2, 3 [68] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. EMNLP, 2024. 3 [69] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1, 4 [70] Ke Zhang, Wei-Lun Chao, Fei Sha, and Kristen Grauman. Video summarization with long short-term memory. In ECCV, 2016. 8 [71] Yue Zhao, Ishan Misra, Philipp Krähenbühl, and Rohit Girdhar. Learning video representations from large language models. In CVPR, 2023. 1, 3

# Agent-based Video Trimming

Supplementary Material

# A. Video Trimming Dataset

We collect user-generated videos from YouTube and construct a benchmark for video trimming specifically. The data collection process adheres to three key principles: 1. The current dataset predominantly consists of videos already edited by their creators. In contrast, for video trimming scenarios, we aim to use raw, unedited videos. These raw videos, typically filmed by individuals, often contain imperfections such as occlusion, jitter, or overexposure, reflecting real-world filming conditions. 2. The raw videos selected should be in long form, with durations exceeding 5 minutes, rather than short, pre-edited montages. 3.In practical video trimming tasks, input videos may originate from multiple sources. As a result, algorithms are expected to generate cuts from various videos, whereas existing datasets typically focus on a single video per topic. Following these principles, we curated a collection of 42 videos uploaded by 30 different users. A comparison with existing datasets is shown in Tab. S2, which highlights that our dataset boasts the longest average video duration, approximately 10 minutes per video. To ensure diversity in trimming scenarios, we selected videos spanning a range of topics, categorized into three groups: daily life, sports, and travel vlogs. For each category, we chose 10 video uploaders and included one or more videos that revolve around a consistent event. Detailed topics and corresponding YouTube IDs are listed in Tab. S1. Additionally, we annotated each video with 10 annotators using four ranking levels to evaluate footage quality: 0 for wasted, 1 for ambiguous, 2 for normal, and 3 for highlight footage. Examples of these ground-truth annotations can be found in Sec E.

# B. Prompt Design

This section provides the detailed prompts for AVT, utilizing GPT-4o [37]. It covers prompts for video structuring, story composition, and video evaluation. Figure S1 illustrates the structuring prompt, which includes general captions, defect attributes, and contextual attributes. Next, we describe the story composition process, which occurs in two stages. The first stage focuses on grouping clips to prevent overwhelming input lengths, prioritizing the selection of highlight segments (Fig. S2). In the second stage, the selected clips are gathered and arranged into a coherent final video, with an emphasis on narrative flow (Fig. S3). Finally, Figure S4 presents the prompt used by GPT to evaluate video quality. Table S1. Our curated video trimming dataset includes 42 YouTube videos, contributed by 30 users, and spans a wide variety of topics.   

<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Sub-Class</td><td rowspan=1 colspan=1> User</td><td rowspan=1 colspan=1>YouTube ID(s)</td></tr><tr><td rowspan=5 colspan=1>daily life</td><td rowspan=1 colspan=1>family</td><td rowspan=1 colspan=1>CapperCoolCooperEarls Family VlogsJason BoonThe Semps</td><td rowspan=1 colspan=1>KRqR6eLSoP8MyLwV1V19WYPcxuFef17PYYoIkzzpQjKM</td></tr><tr><td rowspan=1 colspan=1>food</td><td rowspan=1 colspan=1>Soon Films </td><td rowspan=1 colspan=1>McjBGfacfc</td></tr><tr><td rowspan=1 colspan=1>friend</td><td rowspan=1 colspan=1>Shawn Roscoe</td><td rowspan=1 colspan=1>JOnA4VgnoCo</td></tr><tr><td rowspan=1 colspan=1>light show</td><td rowspan=1 colspan=1>World In Nature</td><td rowspan=1 colspan=1>nFPJMj0tq9G</td></tr><tr><td rowspan=1 colspan=1>pets</td><td rowspan=1 colspan=1>Cats with GoProGone to the Snow DogsMs Kendall G</td><td rowspan=1 colspan=1>a98Ra7PaTeEiISfWRiDelgBhxk-O1Y7Ho</td></tr><tr><td rowspan=10 colspan=1>sports</td><td rowspan=1 colspan=1>badminton</td><td rowspan=1 colspan=1>SHUTTLE&amp;MORE</td><td rowspan=1 colspan=1>lyhfZy7tShU</td></tr><tr><td rowspan=1 colspan=1>basketball</td><td rowspan=1 colspan=1>June young Kim</td><td rowspan=1 colspan=1>UNemjRp6YJg</td></tr><tr><td rowspan=2 colspan=1>cycling</td><td rowspan=1 colspan=1>Erkan Sakallioglu</td><td rowspan=1 colspan=1>A06p0jlOd6UiwCSAYGwPq4xLhHU8uo2aY</td></tr><tr><td rowspan=1 colspan=1>Richard Whittle</td><td rowspan=1 colspan=1>g--B5HBR1Y</td></tr><tr><td rowspan=1 colspan=1>motorcycle</td><td rowspan=1 colspan=1>Skaily Production</td><td rowspan=1 colspan=1>MXAzBe7PZOQ</td></tr><tr><td rowspan=1 colspan=1>skateboard</td><td rowspan=1 colspan=1>TOW TRUCK BOB</td><td rowspan=1 colspan=1>VVK1KkIKCYQ</td></tr><tr><td rowspan=1 colspan=1>skating</td><td rowspan=1 colspan=1>HC+</td><td rowspan=1 colspan=1>a8M-5nTrl18bA3CxZllhsIdZ3i-HuhQXM</td></tr><tr><td rowspan=3 colspan=1>skiing</td><td rowspan=1 colspan=1>Alex E</td><td rowspan=1 colspan=1>E50qGoDzNtgdFrfsgW1M98ddxw58h5YAfB8zm1hTvgAh7aeRrf-m-8mqNZjVDZcfYpuAxGH6aWMYxnsTcvtttfY</td></tr><tr><td rowspan=1 colspan=1>Emerson Nishi</td><td rowspan=1 colspan=1>WL4TA--CVcA</td></tr><tr><td rowspan=1 colspan=1>ecosnowsportsTV</td><td rowspan=1 colspan=1>5FE871Ij1DQ</td></tr><tr><td rowspan=10 colspan=1>travel</td><td rowspan=1 colspan=1>amusement park</td><td rowspan=1 colspan=1>Informative Topics Vlogs</td><td rowspan=1 colspan=1>hmImxd681YI</td></tr><tr><td rowspan=1 colspan=1>city walk</td><td rowspan=1 colspan=1>Jahaar views</td><td rowspan=1 colspan=1>tp912d19x4E</td></tr><tr><td rowspan=1 colspan=1>hiking</td><td rowspan=1 colspan=1>thePOVchannel</td><td rowspan=1 colspan=1>I1gGZa4-h_U</td></tr><tr><td rowspan=1 colspan=1>luge</td><td rowspan=1 colspan=1>Travel&amp;Adventure Junkies</td><td rowspan=1 colspan=1>dxz00qnhrPo</td></tr><tr><td rowspan=1 colspan=1>mountaineering</td><td rowspan=1 colspan=1>stivn</td><td rowspan=1 colspan=1>iUMBQugUtVQ</td></tr><tr><td rowspan=1 colspan=1>rafting</td><td rowspan=1 colspan=1>All About Shenzhen</td><td rowspan=1 colspan=1>A7Ys5d-Zwro</td></tr><tr><td rowspan=1 colspan=1>road trip</td><td rowspan=1 colspan=1>Mojo Rides</td><td rowspan=1 colspan=1>ePKyNYP7uNg</td></tr><tr><td rowspan=2 colspan=1>show</td><td rowspan=1 colspan=1>HetfieldMustaine22</td><td rowspan=1 colspan=1>McC9gB5Cr60</td></tr><tr><td rowspan=1 colspan=1>KriyaLv</td><td rowspan=1 colspan=1>10LM0_Jzt5M</td></tr><tr><td rowspan=1 colspan=1>water park</td><td rowspan=1 colspan=1>Gezen Adam</td><td rowspan=1 colspan=1>3iz5SmEQj9AWgbe-WTp_QI</td></tr></table>

Table S2. Comparisons of existing datasets with our video trimming dataset.

<table><tr><td>Dataset</td><td>#Video</td><td>#User</td><td>Content</td><td>Annotation type</td><td>Query</td><td>Duration (Min, Max, Avg)</td></tr><tr><td>YouTube Highlights [43]</td><td>423</td><td>5</td><td>Web videos</td><td>Frame-level scores</td><td>Title</td><td>7s, 1483s, 102s</td></tr><tr><td>SumMe [16]</td><td>25</td><td>15~18</td><td>User-generated videos</td><td>Frame-level scores</td><td>N/A</td><td>32s, 324s, 146s</td></tr><tr><td>TVSum [41]</td><td>50</td><td>20</td><td>Web videos</td><td>Frame-level scores</td><td>Title</td><td>83s, 647s, 235s</td></tr><tr><td>Charades-STA [15]</td><td>9,848</td><td>267</td><td>Web videos</td><td>Time intervals</td><td>Local caption</td><td>2s, 194s, 30s</td></tr><tr><td>OVP [10]</td><td>50</td><td>5</td><td>Various genre videos</td><td>Time intervals</td><td>N/A</td><td>83s, 647s, 235s</td></tr><tr><td>YouTube [10]</td><td>39</td><td>5</td><td>Web videos</td><td>Time intervals</td><td>N/A</td><td>83s, 647s, 235s</td></tr><tr><td>Video Trimming (ours)</td><td>42</td><td>10</td><td>Web videos</td><td>Frame-level scores</td><td>N/A</td><td>141s, 1483s, 556s</td></tr></table>

# system ul detailed answers.

# user:

y well as purposeful tracking shots, should not be considered as obstructions. considered as jitter. Shots with clear actions or behaviors should not be considered shaky.   
colored horizontal stripes, colored vertical stripes, green fringing, pink screen, or purple screen.   
on the top, bottom, or sides of the frame, it indicates that the video has been edited.   
walls are ineffective and meaningless. considered highlights.

# # Useful attribute:

longer descriptions. Above all, don't use general, non-discriminative descriptions.   
[What]: Describe the main actions or events occurring in the scene.   
phrases such as 'outdoor'   
[When]: Determine the time of day, season, or any relevant time period depicted.   
rather than ambiguous phrases such as 'person'. Py summarize the content of the video clip. It should not exceed two sentences.

# Answer given questions with the following restrictions.   
(1) If you are not sure about the answer, say you do not know honestly.   
(2) Do not imagine any contents that are Not in the video.   
( Do not add information.   
(4) Do not describe each frame individually and do not mention the frame.   
(5) Do not summarize negative or uncertain answers. # Output format constraints   
The overall output format is as follows:   
{"atibuteuseless":Uselesattribute, "attributeuseful:Useul attribute, "racaption":Vido captin}   

- Useless attribute output format constraints:   
.    ] [Meaningless]: 0.8; [Highlight]: 0.9;   
exist, while a score of 1 indicates absolute reliability." - Useful attribute output format constraints:   
Each scene should contain the above four attributes.   
It is recommended to use one word or one short phrase to summarize each attribute.   
o ul  a ge, h y nd nse couple"} - Video caption output format constraints:   
summarize the content of the video clip. It should not exceed two sentences.

system # IDENTITY and PURPOSE Y <clip id>, <highlight score>, <clip caption>, <clip attribute>. Y segments. The <clip id> represents the temporal sequence of the original video. Think step-by-step about how to achieve the best possible results by following the steps below. # STEPS onsiderptianriuheideonszhmacoplthemh score> ranging from 0 to 1, with higher scores being prioritized. structure of beginning, development, and ending. # RULES - Include segments of the beginning and the end, focus on choosing continuous brilliant clips. Avoid duplicate clips or clips with similar sceneries. rather than fixed scenes. - The number of selected clips should be no less than half of the inputs clip length. more closely indexed clips should be considered for merging first. # OUTPUT INSTRUCTIONS Only output Markdown. Do not imagine any contents that are Not in the clip captions. Do not output the markdown code syntax, only the content. Do not use bold or italics formatting in the markdown output. Do not list clip id in HIGHLIGHTS. You use the following format in exciting video collection: [<clip id>: sentence], .…, [<clip id>: sentence], where each houlbewiten gsndae wit nc sntehunotehcio - Do not repeat ideas, quotes, facts, or resources. Do not start items with the same opening words. Ensure you follow ALL these instructions when creating your output. ser: # INPUT INPUT:

# system

# IDENTITY and PURPOSE   
attributes and descriptions provided for each clip.   
series of segments that capture multiple complete action scenes.   
The final merged video needs to consider the input video sequence and satisfy logical rationality.   
Think step by step about how to achieve the best possible results by following the steps below.   
Select clip from the start and the end of the input <clip id> as beginning and ending.

# STEPS clips according to each theme.   

2. The clips of each theme should contain the development of the event.   
the temporal sequence of the original video.

# RULE

Avoid duplicate clips or clips with similar sceneries. - The selected clips should all of the themes and ensure content diversity. -The chosen <clip id> should cover the clips from the start, middle, end of the inputs sequence. rather than fixed scenes - Ensure that the selected segments for the final story generation do not exceed 20 and no less than 15. narrative integrity. critical and should be selected. These segments should be consistently included in the final output.

# OUTPUT INSTRUCTIONS   

- Only output Markdown. Do not imagine any contents that are Not in the clip captions. Do not output the markdown code syntax, only the content. - Do not use bold or italics formatting in the markdown output. - You use the following format in output: [<clip id>: sentence], [<clip id>: sentence], where each sentence should be   
written in English and wrapped with [] and each sentence should note the clip id coming from. Do not repeat ideas, quotes, facts, or resources. Do not start items with the same opening words. Ensure you follow ALL these instructions when creating your output. user: # INPUT INPUT:

# system detailed answers.

# user:

highest score (best). should contain specific event content. Scores range from 1 to 5, with 5 being the highest score (best). h talking to the camera, static scenes, ec.Scores range from 1 to , with 5 being the highest score (best).

# Answer given questions with the following restrictions.   
(1) If you are not sure about the answer, say you do not know honestly.   
(2) Do not imagine any contents that are Not in the video.   
(3) Do not add information.   
(4) Do not describe each frame individually and do not mention the frame.   
(5) Do not summarize negative or uncertain answers.

# Output format constraints.

T ; [Content of Exciting Segments]: Reason (3.5); [Amount of Waste Footage]: Reason (2.0); Table S3. Ablation study on the impact of sampling ratio and prompt design on performance and cost.   

<table><tr><td>Frame Sampling Ratio</td><td>Prompt</td><td>Input Image Token</td><td>Input Text Token</td><td>Output Text Token</td><td>API Cost</td><td>Agent Metric</td></tr><tr><td>4/1s</td><td>Isolated</td><td>1,836,000</td><td>100,000</td><td>20,000</td><td>$5.04</td><td>3.34</td></tr><tr><td>4 /1s</td><td>Unified</td><td>612,000</td><td>100,000</td><td>20,000</td><td>$1.98</td><td>3.33</td></tr><tr><td>1/1s</td><td>Isolated</td><td>459,000</td><td>100,000</td><td>20,000</td><td>$1.60</td><td>3.34</td></tr><tr><td>1 /1s</td><td>Unified</td><td>153,000</td><td>100,000</td><td>20,000</td><td>$0.83</td><td>3.32</td></tr></table>

Table S4. Comparison of the fidelity between the final videos and the raw videos.

<table><tr><td>Method</td><td>ViCLIP</td><td>InternVideo2</td><td>Avg.</td></tr><tr><td>UniVTG [27]</td><td>0.877</td><td>0.941</td><td>0.909</td></tr><tr><td>UVCOM [57]</td><td>0.852</td><td>0.928</td><td>0.890</td></tr><tr><td>AVT (ours)</td><td>0.906</td><td>0.951</td><td>0.929</td></tr></table>

# C. Implementation and Efficiency

We analyze the efficiency and cost of different implementations by varying the video sampling ratio and prompt design. Specifically, we compare a sampling ratio of 1 frame per second (fps) with 4 fps. As shown in Fig. S1, three components: raw captions, defect attributes, and contextual attributes, are typically generated together using a unified prompt. Alternatively, these components can be extracted separately using isolated prompts, which require processing three times the visual content. The current GPT API pricing is $\$ 2.50$ per million input tokens and $\$ 10.00$ per million output tokens. Each sampled keyframe resized to $5 1 2 \times 5 1 2$ generates approximately 255 tokens in GPT-4o. Metrics are evaluated using the Video Evaluation Agent. Tab. S3 highlights that adopting a 1 fps sampling ratio and a unified prompt reduces the cost of processing a 10-minute video from $\$ 5.04$ to $\$ 0.83$ while maintaining comparable performance to configurations with higher sampling rates and isolated prompts.

# D. Fidelity Evaluation

For the quantitative experiments on video trimming, we also introduce a fidelity evaluation to assess the visual content similarity between the generated videos from different methods. For previous methods, we directly concatenate intervals with the highest saliency scores. In this experiment, we measure the feature similarity between the final video and the raw videos. A well-trimmed video should preserve the full content of the original video while maintaining its narrative coherence. We utilize two benchmarks, leveraging video features extracted by ViCLIP [51] and InternVideo2 [52]. For both raw and trimmed videos from each method, an equal number of keyframes are sampled and processed through vision encoders. The feature similarity between the raw and trimmed videos is subsequently evaluated. As shown in Tab. S4, our method consistently improves content fidelity across various feature extraction models.

# E. More Visualization

In the main paper, we present visualizations of saliency scores and selected intervals for each method, demonstrating the effectiveness of our waste filtering operation and composition phase. In this supplementary section, we expand the analysis by incorporating additional visualizations and conducting case studies to highlight the significance of the AVT module designs.

# E.1. Clip selection

We present a case study that visualizes the clip selection results when incorporating different AVT modules. Figure S5 illustrates the impact of AVT's clip filtering module by comparing performance with and without it. Without filtering, story composition is applied to all intervals, resulting in a full row of light green segments in the visualization. This lack of candidate narrowing leads to the inclusion of more wasted footage in the final video. Figure S6 highlights the consequences of omitting the dynamic filtering module. Without this module, the clip filter discards most segments, especially in sports content, where intense activity often introduces jitter or other visual defects. As a result, highlight segments are misclassified as defects and excluded from the composition. The second row in the visualization shows significantly fewer filtered clips (light green) compared to the first row, emphasizing the importance of the dynamic filtering module. The joint design of the AVT modules substantially enhances the viewing experience and enriches the content. By selecting more highlight footage and minimizing wasted footage, AVT not only outperforms prior approaches but also preserves a coherent storyline that aligns with the raw video material.

# E.2. Storyline

We create the final video by constructing a corresponding storyline that outlines the rationale behind selecting each clip as the beginning, development, or ending, referred to as clip-wise captions. Additionally, we generate clustered themes, each representing a group of selected segments, as outlined in the story composition prompts. Ultimately, this results in a global storyline that captures the entire content of the trimmed videos. These captions, presented at various levels, are visualized in Fig. S7.

# E.3. More Visualization with Existing Methods

We present additional visualizations of the saliency scores and selected intervals for each method in our video trimming dataset, as shown in Fig. S8. Overall, AVT outperforms previous approaches by selecting more highlight footage, reducing wasted footage, and maintaining a consistent storyline, ultimately enhancing the viewing experience. For instance, AVT excels at retrieving dynamic scenes like mountain biking, while existing methods tend to select more mundane clips. In another scenario, for plain vlogs such as food videos or dolphin shows, AVT efficiently trims the complete story across the entire timeline of the source video, while other methods may overlook key content.

![](images/8.jpg)  
Figure S5. Effect of clip filtering on visualization of trimmed videos.

![](images/9.jpg)  
Figure S6. Effect of dynamic filter module on visualization of trimmed videos.

![](images/10.jpg)  
Figure S7. Visualization of the multi-level storyline of the trimmed final video.

T c

![](images/11.jpg)

![](images/12.jpg)  
Figure S8. Visualization of trimmed videos on the video trimming dataset.