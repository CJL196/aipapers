# 1. Bibliographic Information

*   **Title:** Classroom Simulacra: Building Contextual Student Generative Agents in Online Education for Learning Behavioral Simulation
*   **Authors:** Songlin Xu, Hao-Ning Wen, Hongyi Pan, Dallas Dominguez, Dongyin Hu, and Xinyu Zhang. The authors are affiliated with the University of California San Diego and the University of Pennsylvania.
*   **Journal/Conference:** Published in the proceedings of the CHI Conference on Human Factors in Computing Systems (CHI '25). CHI is a premier international conference in the field of Human-Computer Interaction (HCI), known for its high impact and rigorous peer-review process.
*   **Publication Year:** 2025
*   **Abstract:** The paper addresses the challenge of creating realistic virtual students for educational simulations. Existing methods often fail to account for how specific course materials affect student learning. This is due to two key problems: a lack of detailed datasets that link course content to student performance, and the inability of current AI models to process very long texts (like lecture materials). To solve this, the authors first conducted a 6-week online workshop with 60 students to collect fine-grained behavioral data using a custom-built system. Second, they propose a novel method called **Transferable Iterative Reflection (TIR)**. This TIR module enhances Large Language Models (LLMs) for student simulation. Experiments show that LLMs augmented with TIR are more accurate than traditional deep learning models, even with limited data. The TIR approach successfully captures the dynamic nature of student performance and the relationships between students, advancing the goal of creating a "digital twin" for online classrooms.
*   **Original Source Link:** The paper is available at `/files/papers/68f1e93c4752e3ebe9d48550/paper.pdf`. It is slated for formal publication at CHI '25.

# 2. Executive Summary

*   **Background & Motivation (Why):** The ultimate goal is to build a "digital twin" classroom—a high-fidelity virtual sandbox where educators can test and refine their teaching strategies without risk. A core component of this is creating realistic "student generative agents" that mimic how real students learn and behave. However, current simulators are limited. They often predict student performance based only on past question-answer history, completely ignoring the crucial context of the **course materials** (lecture slides, readings) that students interact with.

    The paper identifies two primary roadblocks:
    1.  **Data Scarcity:** There are no publicly available datasets that granularly map detailed course materials to students' real-time learning behaviors and performance.
    2.  **Model Limitations:** Large Language Models (LLMs), which are promising for this task, struggle with extremely long inputs. Fine-tuning them on long texts is computationally expensive, and their ability to learn from examples (in-context learning) degrades as the length of the input prompt increases.

*   **Main Contributions / Findings (What):**
    1.  **A Novel Dataset and System:** The authors developed a custom online education system called `CogEdu` that monitors student engagement (using gaze tracking and facial expression analysis) to ensure high-quality data collection. Using this system, they ran a 6-week workshop with 60 students, creating a unique dataset that links lengthy course materials to fine-grained student performance data.
    2.  **Transferable Iterative Reflection (TIR) Module:** This is the core technical innovation. TIR is a framework that helps LLMs to "think" more deeply about the learning process. It distills key insights from long course materials and student histories into concise, useful "reflections." This compressed knowledge makes the simulation process more efficient and accurate. Crucially, these reflections are "transferable," meaning insights learned from one set of students can be used to simulate new, unseen students.
    3.  **Comprehensive Evaluation:** The study demonstrates that LLMs augmented with TIR significantly outperform both standard LLM approaches and state-of-the-art deep learning models in simulating student performance. The evaluation goes beyond simple accuracy, showing that the TIR-based agents better capture the complex dynamics of learning, including performance variations at the individual, lecture, and question levels, as well as the correlations in performance between different students.

# 3. Prerequisite Knowledge & Related Work

To understand this paper, it's helpful to be familiar with the following concepts:

*   **Generative Agents:** These are AI systems, typically powered by LLMs, designed to simulate believable human behavior. Unlike simple chatbots, they can maintain a memory of past events and make decisions within a simulated environment. The famous "Generative Agents" paper from Stanford [58] showed this with agents living in a virtual town. This paper applies the concept to an educational setting.
*   **Digital Twin:** A virtual replica of a physical object, process, or system. A digital twin of a classroom would be a dynamic simulation that mirrors the real classroom's state and behavior, allowing for what-if analyses.
*   **Knowledge Tracing (KT):** A classic problem in educational data mining. KT models aim to track a student's evolving knowledge of different concepts ("skills") over time by observing their answers to a sequence of questions. Early models used Bayesian methods, while modern approaches use deep learning.
*   **Large Language Models (LLMs):** Models like GPT-4, trained on massive amounts of text. Their strength lies in understanding context and generating human-like text.
    *   **Prompting (In-Context Learning):** Giving an LLM instructions and a few examples within its input prompt to guide its output, without changing the model's internal weights.
    *   **Fine-tuning:** Taking a pre-trained LLM and further training it on a smaller, task-specific dataset. This updates the model's weights to specialize it for the new task.
*   **Chain of Thought (CoT) Prompting:** A prompting technique where the LLM is instructed to "think step by step" and write out its reasoning process before giving a final answer. This often improves performance on complex reasoning tasks.
*   **BERT (Bidirectional Encoder Representations from Transformers):** A powerful language model that excels at understanding the context of words in a sentence. However, it has a strict input length limit (typically 512 tokens), making it unsuitable for processing long documents directly.

**Previous Works & Differentiation:**

*   **Generative Agents in Education:** Previous work like `GPTeach` [53] (for training teaching assistants) and `MATHVC` [92] (for math education) used LLMs to create virtual students, but they did not rigorously evaluate how realistic these simulated students were.
*   **LLM-based Knowledge Tracing:** Other studies [15, 31, 36] have used LLMs for knowledge tracing. However, they treat it as a simple sequence prediction problem (predicting the next answer based on past answers), ignoring the influence of the actual lecture content that students are learning from. The paper argues that this misses a key opportunity, as LLMs excel at contextual learning.
*   **`EduAgent` Dataset [86]:** This was one of the first attempts to include course materials. However, the lectures were only 5 minutes long, which is too short to observe complex learning dynamics like fatigue or the build-up of knowledge. This paper's dataset, collected over 1-hour lectures for 6 weeks, provides a much richer view.

    This paper's key innovation is bridging the gap. It introduces the **`TIR` module** as a method to effectively incorporate long, contextual course materials into student simulation, something no prior work had successfully done, and provides a rich new dataset to validate this approach.

# 4. Methodology (Core Technology & Implementation)

The paper's goal is to create a simulator that can predict a student's future performance (correctness on post-lecture test questions) based on their past performance and the relevant course materials.

![该图像是一个示意图，展示了基于转移迭代反思（TIR）模块的学生行为模拟过程，包括课程刺激、学习过程与结果、反思主体间的交互，以及利用学习历史进行学生模拟。](images/1.jpg)
*该图像是一个示意图，展示了基于转移迭代反思（TIR）模块的学生行为模拟过程，包括课程刺激、学习过程与结果、反思主体间的交互，以及利用学习历史进行学生模拟。*

**Figure 1:** The image provides a high-level overview. A real classroom provides a `Learning History` (data). The proposed `Transferable Iterative Reflection (TIR)` module processes this history, involving a `Reflective Agent` and a `Novice Agent`, to generate a `Student Simulation` of `Virtual Students` interacting with a `New Course Stimuli`.

## 3.1 Problem Formulation

The simulation model is given two sets of information:
*   **Past Learning History ($l_{past}$):** This includes the content of past questions ($q_{past}$), whether the student answered them correctly ($y_{past}$), and the course materials associated with those questions ($c_{past}$).
*   **Future Learning Information ($l_{future}$):** This includes the content of future questions ($q_{future}$) and their associated course materials ($c_{future}$).

    The model's task is to predict the correctness of the student's answers to the future questions, denoted as $\hat{y}_{future}$.

## 3.3 Transferable Iterative Reflection (TIR)

TIR is the core contribution, designed to help the LLM learn more effectively from the data, especially the long course materials. It is an iterative process involving two agents: a `reflective agent` and a `novice agent`.

The process unfolds in four phases during the **training stage**:

1.  **Initial Prediction:** A `reflective agent` (an LLM) is given the student's history ($l_{past}$) and future context ($l_{future}$) and makes an initial prediction ($\hat{y}_{future}$) of the student's performance.
2.  **Reflection:** The agent is then shown the *actual* outcome (the ground truth, $y_{future}$). It is prompted to reflect on the discrepancies between its prediction and the reality, explaining *why* it made mistakes. This generates a reflection, $r_k$.
3.  **Testing (Transferability Check):** A separate `novice agent` (another LLM instance that has **not** seen the ground truth) is given the original problem context *plus* the reflection $r_k$ from the reflective agent. It then makes a new prediction. This step is crucial because it tests whether the reflection is a genuinely useful, generalizable insight, rather than just an overfitted explanation.
4.  **Iteration:** If the `novice agent`'s accuracy improves, the reflection is considered good, and the process continues to refine it. If not, the `reflective agent` is prompted to reflect from a different angle. This loop continues until the novice agent achieves perfect accuracy or a maximum number of iterations is reached. The best-performing reflection ($r_{best}$) is saved to a `successful reflection database`.

    ![Figure 4: Prompt examples in the Transferable Iterative Reflection process.](images/4.jpg)
    *该图像是论文中关于可迁移迭代反思（TIR）过程的提示示例示意图，展示了多轮LLM（大语言模型）和用户交互反思改进预测的具体文本流程，突出模型如何通过不断反思提升预测准确率。*

    **Figure 4: Prompt examples in the Transferable Iterative Reflection process.** This image vividly illustrates the TIR process. The left panel shows the initial prompt. The right panels show three iterations ($r_1, r_2, r_3$). In $r_1$, the LLM makes an incorrect prediction. After receiving feedback, it tries again in $r_2$. After more feedback, in $r_3$, it finally identifies a "misunderstanding or oversight" that led to the initial error. This successful reflection ($r_3$) is then stored, as it represents a useful insight that improved the prediction.

## 3.4-3.6 Applying TIR to Different Models

The `TIR` module is versatile and can augment three types of LLM-based simulators.

![该图像是一个架构示意图，展示了训练（a）和测试（b）学生数据集时基于转移迭代反思（TIR）模块的生成代理学习行为模拟流程。](images/2.jpg)
*该图像是一个架构示意图，展示了训练（a）和测试（b）学生数据集时基于转移迭代反思（TIR）模块的生成代理学习行为模拟流程。*

**Figure 2:** This diagram shows the training and testing schemes for prompting-based models.
*   **(a) Training:** The TIR process (involving the reflective and novice agents) is run on the training dataset to generate a `Successful Reflection Database`.
*   **(b) Testing:** For a new student in the test set, the system retrieves relevant reflections from the database. These reflections act as in-context examples for a new `Reflective Agent`, which then makes a final prediction for the test student.

*   **TIR for Standard & CoT Prompts:** For prompting-based models, TIR serves as a powerful method for creating few-shot examples. Instead of naively stuffing long, raw data into the prompt (which would exceed token limits), the system uses the concise, insightful reflections from the database. During testing, reflections from a few similar students in the training set are retrieved and included in the prompt for a new student. This provides the LLM with high-quality, distilled context to guide its prediction. The process is the same for Chain of Thought (CoT), but the LLM is also asked to reason step-by-step.

    ![该图像是论文中图2的结构示意图，展示了训练集(a)和测试集(b)中基于转移迭代反思（TIR）模块的学生生成代理行为模拟流程。图中详细说明了输入数据、反思代理和分类器的交互机制，突出LLMs与BERT分类器的结合及动态反馈迭代。](images/3.jpg)
    *该图像是论文中图2的结构示意图，展示了训练集(a)和测试集(b)中基于转移迭代反思（TIR）模块的学生生成代理行为模拟流程。图中详细说明了输入数据、反思代理和分类器的交互机制，突出LLMs与BERT分类器的结合及动态反馈迭代。*

    **Figure 3:** This diagram shows the scheme for finetuning-based models (`BertKT`).
*   **(a) Training:** The TIR process is run to generate a reflection. This reflection, along with the initial LLM prediction and future question content, becomes the input to a `BERT Classifier`, which is then fine-tuned.
*   **(b) Testing:** For a new test student, a reflection is generated using the retrieval method from the `Successful Reflection Database`. This reflection, the initial prediction, and question content are fed into the already `Trained BERT Classifier` to get the final prediction.

*   **TIR for Finetuning-based Models (`BertKT`):** Here, TIR acts as a **knowledge compression** tool. `BERT` has a small 512-token limit and cannot handle the full context of course materials. The TIR module first processes all the long-form data and distills it into a short, text-based reflection. The input to the `BERT` model is then a combination of: (1) the future question content, (2) an initial prediction from an LLM, and (3) the compressed reflection from TIR. This allows the powerful classification abilities of a fine-tuned `BERT` model to be leveraged without being constrained by its token limit.

# 5. Experimental Setup

## Datasets

1.  **`EduAgent` [86]:** A public dataset from 301 students watching 5-minute online course videos. Used for initial validation.
2.  **New `CogEdu` Dataset:** The paper's main dataset, collected from a 6-week online workshop.
    *   **Participants:** 60 students (30 elementary, 30 high school) and 8 instructors.
    *   **Procedure:** 12 one-hour lectures on AI topics. After each lecture, students took a 10-12 question post-test.
    *   **Data Collection System (`CogEdu`):** A custom-built web platform that integrated Zoom for video conferencing. It used webcams to track student **gaze** and **facial expressions** (e.g., confusion). This data was used in real-time to provide feedback to instructors (e.g., heatmaps of where students were looking on a slide) and suggest actions to improve engagement, ensuring the quality of the collected learning data.

        ![Figure 7: The procedure to use our online learning system. (a)(b)(c). Gaze calibration process for gaze tracking. (d)(e). Facial expression model training data collection process. (f)(g). Students an…](images/7.jpg)
        *该图像是示意图，展示了论文中 Fig.7 在线学习系统的使用流程。包括(a)(b)(c)中的视线校准过程，(d)(e)中的面部表情模型训练数据采集过程，以及(f)(g)中学生和教师通过各自客户端加入在线视频通话的界面。*

        **Figure 7: The procedure to use our online learning system.** This image shows the user interface for the `CogEdu` system. Panels (a)-(c) show the gaze calibration process, where the user looks at dots on the screen. Panels (d)-(e) show the system capturing neutral and confused facial expressions to train a personalized confusion detector. Panels (f)-(g) show the interface for joining the online class.

## Evaluation Metrics

The models' predictions (correct/incorrect) are compared against the ground truth using two standard metrics for binary classification:

1.  **Accuracy:**
    *   **Conceptual Definition:** The percentage of predictions that were correct. It measures overall correctness.
    *   **Mathematical Formula:**
        $$
        \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
        $$
    *   **Symbol Explanation:**
        *   `TP` (True Positives): Student answered correctly, and the model predicted correctly.
        *   `TN` (True Negatives): Student answered incorrectly, and the model predicted incorrectly.
        *   `FP` (False Positives): Student answered incorrectly, but the model predicted correctly.
        *   `FN` (False Negatives): Student answered correctly, but the model predicted incorrectly.

2.  **F1 Score:**
    *   **Conceptual Definition:** The harmonic mean of Precision and Recall. It provides a more balanced measure than accuracy, especially if the number of correct and incorrect answers is unequal.
    *   **Mathematical Formula:**
        $$
        F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{where} \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad \text{and} \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
        $$
    *   **Symbol Explanation:**
        *   `Precision`: Of all the times the model predicted "correct," what fraction were actually correct?
        *   `Recall` (Sensitivity): Of all the questions the student actually answered correctly, what fraction did the model predict correctly?

## Baselines

The proposed `TIR`-augmented models were compared against:
*   **LLM-based Models (without TIR):** `Standard` prompting, `CoT` prompting, and `BertKT`.
*   **Deep Learning Models:** Five state-of-the-art knowledge tracing models: `DKT` (Deep Knowledge Tracing), `AKT` (Attentive Knowledge Tracing), `ATKT` (Adversarial Attentive KT), `DKVMN` (Dynamic Key-Value Memory Networks), and `SimpleKT`.

# 6. Results & Analysis

## Simulation on the `EduAgent` Dataset

The authors first validated their approach on the public `EduAgent` dataset.

*This is a transcription of the data from Table 1 in the paper.*
**Table 1:** Simulation results on EduAgent dataset

<table>
<thead>
<tr>
<th rowspan="2">Metric</th>
<th colspan="3">GPT4o-Mini (Without/With TIR)</th>
<th colspan="5">Deep Learning Models</th>
</tr>
<tr>
<th>Standard</th>
<th>CoT</th>
<th>BertKT</th>
<th>DKT</th>
<th>AKT</th>
<th>ATKT</th>
<th>DKVMN</th>
<th>SimpleKT</th>
</tr>
</thead>
<tbody>
<tr>
<td>Accuracy</td>
<td>0.6025+0.0469</td>
<td>0.6222-0.0049</td>
<td><strong>0.6074+0.0938</strong></td>
<td>0.6351</td>
<td>0.6171</td>
<td>0.6396</td>
<td>0.6171</td>
<td>0.6772</td>
</tr>
<tr>
<td>F1 score</td>
<td>0.5128+0.1346</td>
<td>0.5610+0.0341</td>
<td><strong>0.6110+0.0770</strong></td>
<td>0.6352</td>
<td>0.6051</td>
<td>0.6390</td>
<td>0.6051</td>
<td>0.6698</td>
</tr>
</tbody>
</table>

*(Note: The table shows results for models with and without TIR. The `+` or `-` indicates the change after applying TIR. For example, `BertKT` with TIR achieves an accuracy of 0.6074 + 0.0938 = **0.7012**.)*

**Key Finding:** The best deep learning model, `SimpleKT`, achieved an accuracy of 0.6772. The LLM models without TIR performed worse. However, after augmentation with TIR, the $BertKT+TIR$ model achieved an accuracy of **0.7012** and an F1 score of **0.6880**, surpassing all baselines. This provides initial evidence of TIR's effectiveness.

## Simulation on the New `CogEdu` Dataset

The main experiments were conducted on the richer dataset collected from the 6-week workshop.

![Figure 9: Model accuracy and F1 score comparison on our newly collected dataset. Left $^ { ( \\mathbf { a } , \\mathbf { c } ) }$ shows comparison (a: accuracy, c: F1 score) among deep learning models…](images/10.jpg)

**Figure 9: Model accuracy and F1 score comparison on our newly collected dataset.**
*   **(a) and (c):** These bar charts compare all models. The deep learning models (`akt`, `atkt`, `dkt`, `dkvmn`, `simpleKT`) are clustered on the left. The `TIR`-augmented models ($CoT+TIR$ and $BertKT+TIR$) clearly achieve the highest accuracy and F1 scores, outperforming all deep learning models and the non-TIR LLM variants. For example, $CoT+TIR$ reaches 0.676 accuracy, higher than the best deep learning model `SimpleKT` (0.656).
*   **(b) and (d):** These charts compare different LLMs. They show that the larger `gpt4o-0806` model generally performs better than the smaller `gpt4o-mini`, and that in all cases, applying TIR (the green bars) improves performance over not using it (the orange bars).

## Analysis of Granular Learning Dynamics

The paper goes deeper to show *how* the simulation is more realistic.

![该图像是一个热力图和统计显著性标注，展示了不同学生模型（如BertKT with TIR, AKT等）在多个学生ID上的表现相关性，图中顶部显示了方差分析结果 $F(5,85)=5.16, p=0.0004, \\eta_p^2=0.18$，右侧包括多个模型间显著性比较标记。](images/11.jpg)

**Figure 10: Heatmap of simulation accuracy per student.**
This heatmap shows the simulation accuracy for individual students (u1, u10, etc.). Each row is a different model, and each column is a student. Darker red indicates higher accuracy. The top row, `BertKT with TIR`, has more dark red cells compared to the other models, indicating it simulates a wider range of individual students more accurately. The statistical analysis on the right ($p<0.01$, $p<0.05$) confirms that its performance is significantly better than the deep learning models.

![Figure 11: Heatmap to show the average simulation accuracy (each cell) for each specific lecture using each model.](images/12.jpg)

**Figure 11: Heatmap of simulation accuracy per lecture.**
This heatmap shows accuracy broken down by lecture. This tests if the model can adapt to different topics and materials. `BertKT with TIR` again shows the most consistent high performance across all 12 lectures, demonstrating its ability to capture the modulating effect of different course stimuli.

![Figure 12: Heatmap to show the average simulation accuracy (each cell) for each post-test question ID using each model.](images/13.jpg)

**Figure 12: Heatmap of simulation accuracy per question.**
This heatmap breaks down accuracy by post-test question ID. It assesses the model's ability to handle questions of varying difficulty. While performance varies, `BertKT with TIR` maintains a strong performance compared to the other models, which show more variability and lower accuracy on more difficult questions (e.g., questions 11 and 12).

![该图像是论文中的多子图图表，展示了采用TIR模块与不采用TIR模块对学生模拟行为的多维度性能对比，包括问答准确率的相关性分析、差异分布、模型表现箱型图、Bland-Altman差异分析及学生个体的模拟准确率和F1分数对比。](images/14.jpg)

**Figure 14: Deeper analysis of TIR's impact.**
This multi-panel figure provides a comprehensive look at the benefits of TIR.
*   **(a) Correlation:** The line plot shows that the accuracy trend of the simulation `With TIR` (blue line, Pearson r=0.42) more closely follows the trend of the real student data (`Label`, gray line) compared to the simulation `No TIR` (orange line, Pearson r=0.02).
*   **(b, d, e) Statistical Comparison:** The histogram of differences, Bland-Altman plot, and bar chart with error bars all statistically demonstrate that the `With TIR` model's predictions are closer to the ground truth and have less variance.
*   **(f) Per-Student Improvement:** The bar charts on the right clearly show that for nearly every individual student, both simulation accuracy and F1 score are higher `With TIR` (blue bars) than `No TIR` (orange bars).

# 7. Conclusion & Reflections

*   **Conclusion Summary:** The paper successfully demonstrates a method for building more realistic student generative agents by incorporating the context of course materials. The two main contributions—a high-quality dataset from the `CogEdu` system and the novel `Transferable Iterative Reflection (TIR)` module—address key challenges in the field. The `TIR` module effectively compresses long contextual information, enabling both prompting-based and fine-tuning-based LLMs to achieve state-of-the-art performance in student simulation, surpassing traditional deep learning methods. This work is a significant step towards creating a "digital twin" for online education.

*   **Limitations & Future Work:** The authors acknowledge that the `TIR` process for generating the reflection database is done offline for each lecture. A potential future direction is to explore whether reflections learned from one lecture can be generalized to an entirely new lecture, which would increase the method's scalability. Further, running the iterative reflection is computationally intensive during the training phase.

*   **Personal Insights & Critique:**
    *   **Significance:** This paper's main strength is its focus on **context**. By moving beyond simple sequence prediction and modeling the influence of course materials, it makes student simulation significantly more plausible and useful. The "digital twin" classroom is no longer just a buzzword but a tangible research goal. This could empower educators to experiment with pedagogical innovations safely and efficiently.
    *   **Novelty of TIR:** The `TIR` module is a clever and practical solution to the long-context problem plaguing LLMs. The "transferability" check using a novice agent is particularly insightful, as it ensures that the generated reflections are genuinely useful and not just post-hoc rationalizations. This technique could potentially be applied to other domains beyond education where AI agents need to reason about long, complex documents.
    *   **Critique:** While powerful, the scalability of generating the `successful reflection database` could be a concern for very large-scale applications with thousands of courses. The process seems to require a fair amount of computation and LLM calls for each student in the training set. However, the authors rightly point out that this is a one-time, offline cost for each lecture, which is a reasonable trade-off for the significant boost in simulation fidelity. The reliance on `GPT-4o-mini` also ties the method to a specific proprietary model, though the framework itself is model-agnostic.