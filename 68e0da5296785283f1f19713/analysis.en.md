# 1. Bibliographic Information

*   <strong>Title:</strong> VideoForest: Person-Anchored Hierarchical Reasoning for Cross-Video Question Answering
*   <strong>Authors:</strong> Yiran Meng, Junhong Ye, Wei Zhou, Guanghui Yue, Xudong Mao, Ruomei Wang, and Baoquan Zhao. The authors are affiliated with Sun Yat-Sen University, Cardiff University, and Shenzhen University.
*   <strong>Journal/Conference:</strong> Published in the *Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)*. ACM Multimedia is a premier international conference in the field of multimedia, known for its high standards and significant impact on the community.
*   <strong>Publication Year:</strong> 2025
*   <strong>Abstract:</strong> The paper introduces `VideoForest`, a new framework designed for cross-video question answering (QA). This task is more complex than single-video QA because it requires connecting information across multiple video streams. `VideoForest` tackles this by using people as "anchors" or bridge points between videos. The framework has three main innovations: (1) A feature extraction method using person Re-Identification (ReID) and tracking to link individuals across videos. (2) A hierarchical tree structure that organizes video content around person trajectories at different levels of detail. (3) A multi-agent reasoning system that navigates this tree structure to answer complex questions efficiently. The authors also created a new benchmark dataset, `CrossVideoQA`, to evaluate such systems. Experiments show that `VideoForest` significantly outperforms existing methods in tasks like person recognition, behavior analysis, and complex reasoning.
*   <strong>Original Source Link:</strong> https://arxiv.org/pdf/2508.03039 (This is an arXiv preprint link for a paper submitted to a future conference).

# 2. Executive Summary

*   <strong>Background & Motivation (Why):</strong>
    *   <strong>Core Problem:</strong> Modern video understanding systems are very good at analyzing a single video. However, they struggle with questions that require information from multiple videos, such as tracking a person across several security cameras in a building. This is a critical limitation for real-world applications like surveillance and security.
    *   <strong>Gap in Prior Work:</strong> Existing video QA models, even advanced ones, operate within a "single-stream constraint." They can't build semantic bridges between different video sources to understand a larger, cohesive event. For example, answering "Which individual traversed all three campus buildings between 2-4 PM?" is impossible for a model that can only look at one video at a time.
    *   <strong>Innovation:</strong> The paper's key insight is to use <strong>people as natural anchors</strong> to connect disparate video streams. A person's identity is a consistent piece of information that can be tracked across different cameras and locations. `VideoForest` builds a unified, hierarchical representation of all video data centered around these person trajectories.

*   <strong>Main Contributions / Findings (What):</strong>
    1.  <strong>A Novel Person-Anchored Framework:</strong> The paper introduces the first hierarchical, tree-based framework that uses human subjects to connect and reason across multiple videos. This creates a unified "forest" of information from individual video "trees."
    2.  <strong>Efficient Multi-Agent Reasoning System:</strong> It proposes an efficient method to organize video content at multiple granularities (from whole scenes down to fine-grained actions) and couples it with a multi-agent system. This allows for complex reasoning without being computationally expensive.
    3.  <strong>A New Benchmark Dataset (`CrossVideoQA`):</strong> The authors created and released `CrossVideoQA`, the first benchmark specifically designed for person-centric, cross-video question answering. This provides a standardized way to measure and compare the performance of models on this challenging task.

# 3. Prerequisite Knowledge & Related Work

*   <strong>Foundational Concepts:</strong>
    *   <strong>Video Question Answering (VideoQA):</strong> A task where a model is given a video and a natural language question about its content and must provide a correct answer.
    *   <strong>Person Re-Identification (ReID):</strong> A computer vision technique used to identify the same person across different cameras or in different video frames. It's crucial for tracking individuals in a multi-camera setup.
    *   <strong>Object Tracking:</strong> Algorithms that follow a specific object (in this case, a person) through a video sequence.
    *   <strong>Hierarchical Data Structure:</strong> A way of organizing data in a tree-like structure with different levels. In this paper, it means organizing video content from coarse (e.g., "office scene") to fine-grained (e.g., "person writing on whiteboard").
    *   <strong>Multi-Agent System:</strong> A computational system where multiple autonomous "agents" (specialized software modules) interact with each other to solve a problem that is beyond the capacity of any single agent.
    *   <strong>Large Language Models (LLMs) & Multimodal Large Language Models (MLLMs):</strong> LLMs are AI models (like GPT-4) trained on vast amounts of text to understand and generate language. MLLMs are an extension that can process information from multiple modalities, including text, images, and video.

*   <strong>Previous Works:</strong>
    *   <strong>Single-Video QA Models:</strong> The paper cites models like `VideoAgent` and `Chat-Video`. While powerful, these models are fundamentally designed to work on one video at a time. `VideoAgent` uses an agent to search within a single video, and `Chat-Video` analyzes motion trajectories within an isolated video. They lack the architecture to bridge information across different video files.
    *   <strong>Structured Video Representation:</strong> Methods like `VideoTree` have explored hierarchical representations to understand long videos more efficiently. However, their focus remains on structuring a *single* video, not on creating connections *between* videos.

*   <strong>Differentiation:</strong>
    `VideoForest`'s primary innovation is its <strong>person-anchored cross-video linking mechanism</strong>. While prior works focused on getting better at understanding a single video, `VideoForest` re-architects the entire data representation to solve the multi-source problem. As illustrated in <strong>Image 4</strong>, it shifts the paradigm from answering "Who entered the library?" (single video) to "Who traversed the library, science building, and rec center?" (multiple videos). The multi-agent system is another key differentiator, enabling a more dynamic and efficient reasoning process compared to monolithic models.

    ![Comparison of single-video vs. cross-video question answering paradigms.](images/1.jpg)

# 4. Methodology (Core Technology & Implementation)

The `VideoForest` framework is built in several stages, from processing raw videos to answering complex questions. <strong>Image 1</strong> provides a high-level overview of the entire architecture.

![VideoForest architecture for cross-video question answering.](images/2.jpg)

*   <strong>Step 1: Problem Definition</strong>
    The task is defined as taking a collection of videos $$\mathcal{V} = \{V_1, V_2, ..., V_n\}$$ and a query with temporal ($$\mathcal{T}$$) and spatial ($$\mathcal{L}$$) constraints, and producing an answer $$\mathcal{A}$$. The core of the method is to represent the videos in a way that facilitates this.
    For each frame $$f_{i,j}$$, two types of information are extracted:
    1.  <strong>Visual Embeddings:</strong> A dense vector $$\mathbf{v}(f_{i,j})$$ that captures the overall visual content of the frame.
    2.  <strong>Person Detections:</strong> A structured set $$\mathbf{p}(f_{i,j})$$ containing information for each person detected: their timestamp, spatial coordinates, and a unique person identifier (`id`).

*   <strong>Step 2: Dual-Stream Feature Extraction and Adaptive Segmentation</strong>
    To get the representations above, a dual-stream architecture is used:
    1.  <strong>Visual Content Stream:</strong> Uses the `ViCLIP` encoder to generate the frame-level visual embeddings $$\mathbf{v}(f_{i,j})$$.
    2.  <strong>Person-Centric Stream:</strong> Uses a tracking and ReID model to extract person information $$\mathbf{p}(f_{i,j})$$.

        Next, the continuous video streams are broken down into semantically coherent segments. A segment boundary is created if any of three conditions are met:
    1.  $$C_1:$$ The visual difference between consecutive frames is too large (a sharp scene change).
    2.  $$C_2:$$ A frame deviates significantly from the average look of the current segment (a gradual but significant change).
    3.  $$C_3:$$ The set of people in the frame changes significantly (someone enters or leaves the scene).
        The condition is formalized as:
    $$
    S(f_{i,j}) = \mathbb{I}[C_1(f_{i,j}) \vee C_2(f_{i,j}) \vee C_3(f_{i,j})]
    $$
    where $$\mathbb{I}[\cdot]$$ is the indicator function. This adaptive segmentation creates meaningful chunks for the hierarchical tree.

*   <strong>Step 3: Multi-Level Semantic Representation</strong>
    Each video segment $$S_{i,k}$$ is then encoded into a single rich representation $$\mathbf{C}(S_{i,k})$$. This is done by a function $$\eta$$ that combines the visual embedding of the segment's keyframe with the aggregated person trajectory data from all frames within that segment. This ensures each segment's representation captures both the scene's appearance and the human activities within it.

*   <strong>Step 4: VideoForest Construction</strong>
    The core data structure, the `VideoForest`, is built from these semantic segments. Each video $$V_i$$ is organized into a tree $$\mathcal{T}_i$$. The collection of all these trees forms the "forest".
    *   <strong>Nodes:</strong> Each node $$v$$ in a tree is a tuple:
        $$
        \boldsymbol{v} = (t_{start}, t_{end}, \mathcal{R}_v, \mathbf{C}_v, \Gamma_v)
        $$
        *   $$[t_{start}, t_{end}]$$: The time interval the node covers.
        *   $$\mathcal{R}_v$$: A set of person identifiers and their trajectories within this time interval. <strong>This is the key component for linking across videos.</strong>
        *   $$\mathbf{C}_v$$: The semantic content representation of the node.
        *   $$\Gamma_v$$: The set of child nodes.
    *   <strong>Structure:</strong> The root node of each tree covers the entire video. This node is recursively split into child nodes covering smaller, non-overlapping time intervals, down to the leaf nodes which correspond to the fine-grained segments from Step 2. This hierarchical structure allows for efficient search, from coarse to fine.

*   <strong>Step 5: Collaborative Multi-Agent System for Reasoning</strong>
    To answer a question, a multi-agent system traverses the `VideoForest`. <strong>Image 2</strong> illustrates this dynamic reasoning process.

    ![Architecture of our distributed multi-agent framework for cross-video reasoning.](images/3.jpg)

    The system has four specialized agents:
    1.  $$\mathcal{A}_{filter}$$: Parses the user's query to identify relevant videos (based on time, location, etc.) from the forest.
    2.  $$\mathcal{A}_{retrieval}$$: Manages a <strong>global knowledge base ($$\mathcal{K}$$)</strong>. This knowledge base stores previously derived facts (e.g., "On July 1st, no one appeared in the office") with a confidence score. This prevents re-computing information for repeated queries. The confidence scores are updated based on new evidence, allowing the system to self-correct.
    3.  $$\mathcal{A}_{navigate}$$: Traverses the hierarchical trees of the selected videos. It starts at the root and moves down to more detailed nodes only if the parent node is deemed relevant to the query. This top-down search is highly efficient. When a person ID is mentioned in the query, it uses the $$\mathcal{R}_v$$ information to jump directly to relevant nodes across different video trees.
    4.  $$\mathcal{A}_{integrate}$$: Synthesizes all the retrieved information from the knowledge base and the tree traversal to formulate a final, comprehensive answer.

# 5. Experimental Setup

*   <strong>Datasets:</strong>
    *   <strong>`CrossVideoQA`:</strong> A new benchmark created by the authors for this task. It's built by combining two existing datasets:
        1.  <strong>Edinburgh Office Surveillance Dataset (EOSD):</strong> Contains videos from 3 locations over 12 days, ideal for tracking structured human behavior in an office environment.
        2.  <strong>HACS Dataset:</strong> A large-scale dataset with diverse human actions, providing more variety.
    *   The benchmark includes questions and answers that require reasoning across these videos.

*   <strong>Evaluation Metrics:</strong>
    The primary metric is <strong>Accuracy</strong>, measuring the percentage of correctly answered questions. Performance is evaluated across three reasoning tasks and four spatio-temporal configurations.
    *   <strong>Reasoning Tasks:</strong>
        1.  `Person Recognition`: Identifying and tracking individuals.
        2.  `Behavior Analysis`: Understanding actions and interactions.
        3.  `Summarization and Reasoning`: Synthesizing information to draw complex conclusions.
    *   <strong>Evaluation Modalities:</strong>
        1.  `M_single`: Questions about a single video (same day, same location).
        2.  `M_cross-temporal`: Questions about the same location but across different times/days.
        3.  `M_cross-spatial`: Questions about the same time period but across different locations.
        4.  `M_cross-spatiotemporal`: The most complex scenario, requiring reasoning across both different times and different locations.

*   <strong>Baselines:</strong>
    The paper compares `VideoForest` against several state-of-the-art MLLMs, including `ShareGPT4Video`, `Video-CCAM`, and `InternVL 2.5`. Since these models are designed for single-video input, a special protocol was used for a fair comparison: they were prompted to first identify relevant videos from the set, extract information from each one sequentially, and then synthesize a final answer.

# 6. Results & Analysis

*   <strong>Core Results:</strong>
    *   <strong>Task-Specific Performance (Table 1):</strong> `VideoForest` significantly outperforms all baseline models across all three reasoning tasks.
        *   In <strong>Person Recognition</strong>, it achieves <strong>71.93%</strong> accuracy, which is over 13% higher than the best baseline (`InternVL-2.5` at 58.93%). This directly validates the effectiveness of the person-anchored design with explicit ReID and tracking.
        *   In <strong>Behavior Analysis</strong>, it scores <strong>83.75%</strong>, a ~12% improvement over the next best models. This shows that by correctly linking people, the model can better understand their collective actions.
        *   The overall accuracy of <strong>69.12%</strong> is more than 10% higher than the closest competitors, demonstrating the superiority of its structured, cross-video approach.

    *   <strong>Spatio-Temporal Performance (Table 2):</strong> `VideoForest` demonstrates robust and balanced performance across all four evaluation modalities, whereas baseline models often excel in one area but fail in others.
        *   For `cross-temporal` and `cross-spatial` tasks, `VideoForest` leads with <strong>72.00%</strong> and <strong>69.23%</strong> accuracy, respectively. This highlights the strength of the hierarchical tree for temporal reasoning and the person anchors for spatial linking.
        *   Baselines like `InternVL 2.5` perform well on `single-video` tasks (73.08%) but drop significantly in `cross-temporal` scenarios (52.00%), confirming their single-stream limitation. `VideoForest`'s performance is much more consistent, showcasing its general applicability.

*   <strong>Qualitative Analysis:</strong>
    <strong>Image 3</strong> shows examples of `VideoForest`'s reasoning process.
    *   For the <strong>Person Recognition</strong> question, it correctly identifies that a person wearing glasses entered Room 201 only on July 1st by checking videos from multiple days.
    *   For the <strong>Behavior Analysis</strong> question, it calculates the duration of a discussion by locating the relevant video segment and analyzing its timestamps.
    *   The model successfully synthesizes information from multiple rooms to answer the <strong>Summarization and Reasoning</strong> question.
    *   The analysis reveals that the primary failure mode is in recognizing very <strong>fine-grained actions</strong> (e.g., "writing on paper"), which can be limited by video resolution and the inherent ambiguity of some actions.

        ![Exemplars from CrossVideoQA illustrating VideoForest's multi-modal reasoning architecture](images/4.jpg)

*   <strong>Ablations / Parameter Sensitivity:</strong>
    Ablation studies were conducted to measure the contribution of each key component.
    *   <strong>Knowledge Base and Reflection (Table 3):</strong> Removing the knowledge base (`w/o Retrieval`) or the reflection mechanism (`w/o Reflection`) caused significant performance drops, especially in complex cross-spatiotemporal scenarios. This confirms that caching previously computed knowledge and having a self-correction mechanism are crucial for efficiency and accuracy.
    *   <strong>Search Mechanisms (Table 4):</strong>
        *   Disabling the use of person ReID during the search (`w/o ReID in Search`) caused a major drop (average of 14.15%), proving that person IDs are critical anchors for navigating the forest.
        *   Removing the initial video filtering step (`w/o Video Filter`) hurt performance badly in multi-hop scenarios, as the system had to search through much more irrelevant data.
        *   Using only a shallow tree traversal (`w/o Deep Tree Traversal`) also reduced accuracy, showing that the fine-grained details in the lower levels of the tree are necessary for answering specific questions.

# 7. Conclusion & Reflections

*   <strong>Conclusion Summary:</strong>
    The paper successfully introduces `VideoForest`, a novel and effective framework for cross-video question answering. Its core strength lies in its person-anchored hierarchical structure, which creates meaningful bridges across multiple video streams. Combined with an efficient multi-agent reasoning system, it sets a new state-of-the-art on the newly proposed `CrossVideoQA` benchmark. The work effectively moves beyond the limitations of single-video processing and provides a computationally tractable solution for complex, real-world reasoning tasks.

*   <strong>Limitations & Future Work:</strong>
    *   <strong>Dependency on Upstream Models:</strong> The framework's performance is heavily dependent on the accuracy of the underlying person tracking and ReID models. Errors in these initial stages (e.g., failing to re-identify a person correctly) will propagate and lead to incorrect answers.
    *   <strong>Fine-Grained Action Recognition:</strong> As noted in the qualitative analysis, the model struggles with very detailed action recognition, which may require higher-resolution video or more sophisticated action models.
    *   <strong>Generalization Beyond People:</strong> The current framework is person-anchored. While highly effective for surveillance, it might be less applicable to cross-video tasks centered around objects or general events where people are not the main focus. Future work could explore using objects or scenes as alternative anchors.

*   <strong>Personal Insights & Critique:</strong>
    *   <strong>Strong, Intuitive Core Idea:</strong> Using people as anchors is an elegant and powerful concept that directly addresses the fundamental challenge of cross-video understanding. It mirrors how a human security officer would approach the task: "follow that person."
    *   <strong>Modular and Interpretable:</strong> The modular design (feature extraction, tree construction, multi-agent reasoning) is a major strength. It makes the system more interpretable than end-to-end black-box models. One can inspect the generated tree or the knowledge base to understand the model's reasoning. This modularity also allows for individual components to be upgraded independently (e.g., plugging in a better ReID model).
    *   <strong>The Benchmark is a Key Contribution:</strong> In machine learning, progress is often driven by good benchmarks. The creation of `CrossVideoQA` is as important as the model itself, as it will enable the field to systematically study and solve the cross-video reasoning problem.
    *   <strong>Non-End-to-End Trade-off:</strong> The fact that the system is not trained end-to-end is both a pro and a con. It avoids the need for massive, perfectly annotated cross-video datasets, which are extremely difficult to create. However, it may be less optimized than a potential future end-to-end model, as errors can accumulate between stages. Overall, for the current state of the field, this pragmatic approach is highly effective.