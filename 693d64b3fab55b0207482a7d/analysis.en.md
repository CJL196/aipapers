# 1. Bibliographic Information

## 1.1. Title
Immersive Phobia Therapy through Adaptive Virtual Reality and Biofeedback

## 1.2. Authors
Alin Moldoveanu, Oana Mitrut, Nicolae Jinga, Cătălin Petrescu, Florica Moldoveanu, Victor Asavei, Ana Magdalena Anghel, and Livia Petrescu.
The authors are affiliated with the Faculty of Automatic Control and Computers, University POLITEHNICA of Bucharest (A.M., O.M., N.J., C.P., F.M., V.A., A.M.A.), and the Faculty of Biology, University of Bucharest (L.P.). This indicates a multidisciplinary team involved in the research, combining expertise in computer science/engineering and biology/psychology.

## 1.3. Journal/Conference
The paper was published as an 'Article' but the specific journal or conference is not explicitly stated in the provided text, only that it is an 'Article'. However, the structure and academic format suggest publication in a peer-reviewed journal.

## 1.4. Publication Year
The paper was published on September 16, 2023.

## 1.5. Abstract
This paper introduces PhoVR (Phobia therapy through Virtual Reality), a system developed over two years in a collaborative project involving a technical university, a biology faculty, and a 3D application design industry partner. PhoVR offers immersive virtual reality exposure therapy (VRET) for specific phobias such as `acrophobia` (fear of heights), `claustrophobia` (fear of enclosed spaces), and `glossophobia` (fear of public speaking). The system integrates `gamified tasks` and `virtual reality environments` with `biophysical data acquisition` (electrodermal activity and heart rate), `automatic anxiety level classification`, `biofeedback` embedded in scene elements, `dynamic adaptation` of virtual environments, and `relaxation techniques`. A dedicated control panel allows `psychotherapists` to manage patient profiles and therapy sessions. Qualitative surveys with subjects and psychotherapists validated the prototype and provided suggestions for refinement, indicating its potential as an effective tool for phobia treatment.

## 1.6. Original Source Link
/files/papers/693d64b3fab55b0207482a7d/paper.pdf
The publication status is 'officially published'.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem addressed by this paper is the prevalence and impact of anxiety disorders, particularly phobias, and the limitations of traditional therapy methods. `Anxiety disorders` affect a significant portion of the global population (275 million people), ranking sixth in global disability. `Phobias`, such as social anxiety disorder, `agoraphobia` (fear of open or crowded spaces), `acrophobia`, and `claustrophobia`, affect approximately 13% of the global population. Traditional treatments include `medication` and `cognitive behavioral therapy (CBT)`, with `exposure therapy` being highly effective (80-90% response rates). However, traditional `in vivo exposure` (direct confrontation with feared stimuli) can be inconvenient, costly, or impractical, and `in sensu exposure` (imaginal) might lack realism.

The paper highlights the emergence of `virtual reality exposure therapy (VRET)` as a safer, more convenient, and equally successful alternative. VRET provides a controlled environment for `exposure therapy`, allowing for gradual and customizable exposure to phobic stimuli without the logistical challenges or perceived dangers of real-world scenarios. Despite its promise, existing VRET systems often lack comprehensive `biofeedback integration`, `automatic adaptation` based on physiological responses, and `gamified elements` to enhance patient engagement.

The paper's entry point is to develop a comprehensive, adaptive VRET system that addresses these gaps. It leverages advancements in `virtual reality` technology and `biophysical data acquisition` to create an immersive, personalized, and engaging therapeutic experience.

## 2.2. Main Contributions / Findings
The primary contributions of the PhoVR system are:
*   **Integrated System Development:** The successful development and implementation of PhoVR, a complete VRET system, through a collaborative, multidisciplinary project, showcasing how various fields can contribute to mental health solutions.
*   **Multifaceted Phobia Support:** The creation of specific, immersive virtual environments for treating `acrophobia` (four scenarios), `claustrophobia` (three scenarios), and `glossophobia` (configurable scenes with adjustable room size, audience size, and personality factors).
*   **Biophysical Data Acquisition and Classification:** Integration of `electrodermal activity (EDA)` and `heart rate (HR)` monitoring during therapy sessions. A proprietary algorithm is used to classify physiological data, determine anxiety levels, and identify critical moments for the patient.
*   **Adaptive and Personalized Therapy:** Implementation of `dynamic adaptation` of the virtual environments (e.g., modifying bridge characteristics, illumination) and `biofeedback` directly integrated into scene elements (e.g., changing sky color based on anxiety) to provide gradual and personalized exposure.
*   **Gamified Engagement:** Introduction of `gamified tasks` and `quests` within the VR environments to increase patient engagement and motivation during therapy.
*   **Therapist Control and Monitoring:** Development of a dedicated `control panel` for `psychotherapists` to manage patient profiles, configure therapy sessions, monitor patients in real-time (live VR view, biometric data), record sessions, and analyze progress.
*   **Support Mechanisms:** Provision of `virtual companions` and a `relaxation menu` with songs, images, and videos, which can be automatically triggered during difficult moments, or manually accessed.
*   **Qualitative Validation:** Positive feedback from qualitative surveys with subjects and `psychotherapists` who evaluated the prototype, confirming the system's functionality, realism, utility, and ergonomics, and identifying areas for refinement.

    These findings suggest that PhoVR offers a promising, advanced approach to VRET, combining technological innovation with therapeutic principles to provide a more engaging, personalized, and effective treatment for phobias.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To fully understand the PhoVR system and its contributions, a grasp of several foundational concepts is essential for a beginner.

*   **Phobias:** A type of anxiety disorder characterized by an intense, irrational fear of a specific object, situation, or activity. This fear is often disproportionate to the actual danger posed and can lead to avoidance behavior, significantly impacting a person's life. Examples include `acrophobia` (fear of heights), `claustrophobia` (fear of enclosed spaces), `agoraphobia` (fear of open or crowded spaces), and `glossophobia` (fear of public speaking).

*   **Anxiety Disorders:** A broad category of mental health conditions characterized by persistent and excessive worry, fear, or nervousness. These disorders can manifest physically (e.g., increased heart rate, sweating) and psychologically (e.g., panic attacks, difficulty concentrating).

*   **Virtual Reality (VR):** A simulated experience that can be similar to or completely different from the real world. VR typically involves a `head-mounted display (HMD)` that provides immersive visual and auditory feedback, often combined with `controllers` for interaction, creating a sense of presence within the virtual environment.

*   **Exposure Therapy (ET):** A psychological treatment technique used to help individuals overcome fears and anxieties. It involves confronting the feared object or situation in a safe and controlled environment, gradually reducing the associated anxiety response. ET can be conducted `in vivo` (real-world), `in sensu` (imaginal), or `in virtuo` (using virtual reality).

*   **Virtual Reality Exposure Therapy (VRET):** A form of exposure therapy where patients are exposed to feared stimuli within a computer-generated `virtual environment`. This method offers control, safety, and customization that might be difficult to achieve in `in vivo` settings.

*   **Biofeedback:** A technique that trains individuals to control their physiological responses (e.g., heart rate, skin temperature, muscle tension, brainwaves) using real-time information about these responses. In the context of VRET, it helps patients become aware of and manage their anxiety by seeing their physiological data (e.g., `heart rate`, `electrodermal activity`) represented visually.

*   **Biophysical Data Acquisition:** The process of collecting physiological signals from the body using sensors. In this paper, `electrodermal activity (EDA)` and `heart rate (HR)` are key biophysical data points.

    *   **Electrodermal Activity (EDA):** Also known as galvanic skin response (GSR) or skin conductance, `EDA` measures changes in the electrical conductivity of the skin. These changes are primarily influenced by sweat gland activity, which is regulated by the `sympathetic nervous system` – the part of the nervous system responsible for the "fight or flight" response. An increase in `EDA` typically indicates increased physiological arousal or emotional stress. The `tonic component (EDA-T)` refers to the slow, baseline changes in skin conductance, reflecting general arousal levels.

    *   **Heart Rate (HR):** The number of times a person's heart beats per minute (`bpm`). `Heart rate` is a primary indicator of physiological arousal and stress. An increase in `HR` can signal heightened anxiety or excitement.

*   **Gamification:** The application of game-design elements and game principles in non-game contexts. In therapy, `gamification` can make sessions more engaging, motivating, and enjoyable by introducing tasks, rewards, and challenges.

*   **Head-Mounted Display (HMD):** A device worn on the head that provides a virtual reality experience by presenting virtual content directly to the user's eyes. Examples include `Oculus Quest 2`, `HTC Vive`, and `Valve Index`.

*   **Frames Per Second (FPS):** A measure of how many unique consecutive images (frames) are displayed per second in a video or animation. A higher `FPS` generally results in a smoother and more realistic visual experience, while a low `FPS` can lead to `motion sickness` or `dizziness` in VR.

*   **Occlusion Culling:** A graphics rendering optimization technique that prevents objects from being rendered when they are hidden from view by other objects. This reduces the number of calculations needed by the graphics processor, improving `performance` and `frame rate`.

*   **Light Maps:** Pre-calculated textures that store lighting information for static objects in a scene. Using `light maps` can significantly improve rendering performance by avoiding real-time lighting calculations, especially in complex 3D environments.

## 3.2. Previous Works
The paper reviews both commercial and academic `VRET` systems, establishing the state-of-the-art that PhoVR builds upon.

### 3.2.1. Commercial Virtual Reality Exposure Therapy Systems
*   **C2Care [11]:** A commercial system offering `VRET` for various disorders including phobias, `OCD` (Obsessive-Compulsive Disorder), `PTSD` (Post-Traumatic Stress Disorder), addictions, and eating disorders. It allows `therapists` to `orchestrate` scenarios by adjusting stimuli (e.g., crowd intensity, audience emotions) and directly communicating with patients within the VR environment. Features an administrative panel for screen feedback and data management.
*   **Amelia [12]:** Another commercial system for phobias and eating disorders, focusing on personalized interventions with greater `therapist control`.
*   **Virtual Reality Medical Center (VRMC) [13]:** Specialized in `PTSD` treatment using VR.
*   **Virtually Better [14]:** Offers `VRET` for phobias and addictions.
*   **XRHealth [15]:** Provides `VRET` with features like `therapist orchestration` and real-time monitoring of live `VR sessions`. `Therapists` can develop personalized care plans updated based on patient progress. It also includes a virtual companion, Luna, for pain management.

    Common features among these commercial systems include `therapist orchestration`, `real-time monitoring`, `environment customization`, and `administrative panels` for data management.

### 3.2.2. Virtual Reality Exposure Therapy Systems Developed in the Academic Context
*   **AcTiVity [16]:** This system explored the impact of `avatar` representation on `sense of presence`, `immersion`, and `motion sickness` in `acrophobia` therapy. A study with 42 non-acrophobic participants showed higher `presence` when users had an `avatar` representation, suggesting its utility for subjects with `acrophobic` tendencies.
*   **Coelho et al. [17]:** An `acrophobia` study involving 15 subjects exposed to heights in both `virtual` and `real-world` environments. Both groups achieved identical results, but `virtual reality` sessions were 50% shorter than `real-world` sessions.
*   **Emmelkamp et al. [18]:** Used a low-budget `VRET` system for `acrophobia` and `in vivo` treatment with 29 subjects. Patients rated their anxiety, and `therapists` adjusted difficulty. Results showed `VRET` to be as effective as `in vivo exposure` in reducing anxiety and avoidance.
*   **Study by Botella et al. [19]:** Presented a mountain and city environment for `acrophobia`. 10 subjects underwent `real-world exposure` and another 10 `VR exposure`. `VR exposure` significantly reduced anxiety levels from 64% to 40%.
*   **ZeroPhobia [20]:** A system with six animated modules, a `virtual therapist`, mobile `VR games`, and 360-degree videos for `acrophobia`. Tested with 66 subjects, it suggested an optimal `VR session` duration of around 25 minutes for increased therapy efficiency.
*   **Study by Krijn et al. [21]:** Exposed 10 subjects (5 `acrophobic`, 5 control) to a 3D virtual city. Found a positive correlation between anxious behavior (HR increase of at least 15 `bpm`) and reported `fear level`.
*   **Study by Wenderoth et al. [22]:** Simulated a hilly landscape with platforms at different heights in a `CAVE (Cave Automatic Virtual Environment)` device. With 99 subjects, it showed that projection on floors and walls increased `immersion` and `anxiety`, though not `sense of presence`.
*   **Study by Costa et al. [23]:** Six `acrophobic` users underwent `e-virtual reality` sessions followed by `in vivo exposure`. Data analysis revealed no significant difference between subjectively perceived anxiety and physiological responses (`heart rate`) in both conditions.
*   **VR simulator [24]:** A simulator, not for therapy, but for companies to assess `acrophobia` in employees, featuring a moving platform and ladder at 11m height.
*   **Studies on Glossophobia (Fear of Public Speaking):**
    *   **Pertaub et al. [25]:** Showed that `glossophobic` individuals are more nervous even in an empty room or with neutral `virtual audiences`.
    *   **Bailenson et al. [26]:** Demonstrated that a hostile `virtual audience` generates strong emotions in speakers.
    *   **DIVE [27]:** Used for creating scenarios with friendly and hostile `virtual audiences` for `glossophobia`. Physiological measurements and speaking performance were analyzed, linking `co-presence` to `avatar` attitude.
    *   **Game by Stupar-Rutenfrans et al. [28]:** Allowed players to modify `audience settings` and reactions (laughter, comments). `VRET` group (8 subjects) showed significant improvements after 5 weeks.
    *   **360-degree virtual environments [29]:** Used for `glossophobia` training (empty classroom, small audience, large audience). Experiment with 35 subjects showed reduced anxiety, especially for those with high baseline anxiety.
    *   **Study by Lindner et al. [30]:** Used `VRET` where a `psychotherapist` could `orchestrate` scenarios and manipulate `audience reactions`. Concluded that consistent therapy reduces anxiety.
    *   **AwareMe system [31]:** Incorporated `voice factor` monitoring (pitch, filler words) and visual/haptic feedback via a wristband. A large experiment (66 subjects) showed similar `stress patterns` in `real-world` and `VR conditions`.
    *   **PubVR system [32]:** Evaluated `live-controlled avatar interactions` versus `prerecorded animations` for `glossophobia`. Participants felt more comfortable with spontaneous answers and dialogue, highlighting the importance of direct interaction.

## 3.3. Technological Evolution
The use of `VR` for `acrophobia` therapy dates back to the 1990s with Rothbaum [10]. Since then, `VRET` has evolved from rudimentary, expensive setups to more sophisticated, accessible systems. Early systems focused on basic `exposure`, while later developments incorporated more `realistic graphics`, `interactive elements`, and gradual `difficulty adjustments`. The integration of `biophysical data acquisition` (like `EDA` and `HR`) marks a significant step, moving beyond self-reported anxiety to objective physiological measures. The trend is towards `adaptive systems` that automatically respond to a patient's emotional state, `gamification` for enhanced engagement, and comprehensive `therapist control panels` for personalized treatment. The advent of affordable `all-in-one HMDs` like the `Oculus Quest` series has democratized `VRET`, making it more widely accessible. PhoVR represents a culmination of these trends, aiming for a highly immersive, adaptive, and therapist-centric `VRET` solution.

## 3.4. Differentiation Analysis
Compared to the main methods in related work, PhoVR introduces several core innovations:
*   **Comprehensive Adaptive System:** While many systems offer `therapist orchestration` or `environment customization`, PhoVR emphasizes `dynamic adaptation` that can be automatically initiated by the system (based on `biometric data`) or manually adjusted by the `therapist`. This includes real-time changes to environmental elements like visibility, illumination, and structural features (e.g., bridge transparency).
*   **Integrated Biophysical Data and Automatic Classification:** PhoVR goes beyond mere `biophysical data acquisition` by incorporating a `proprietary algorithm` for `automatic anxiety level classification`. This allows the system to identify `difficult moments` and trigger `adaptive responses` or `support elements` autonomously.
*   **Biofeedback Integrated into Scene Elements:** Instead of just displaying `biofeedback` numerically, PhoVR integrates it directly into the `virtual environment` itself. For instance, the ambient light or sky color can change based on the patient's `HR` and `EDA`, offering a more immersive and intuitive form of `biofeedback` that encourages emotional regulation.
*   **Gamified Tasks for Engagement:** PhoVR extensively uses `gamified tasks` and `quests` (e.g., `checkpoint`, `photo`, `collection`, `pickup` tasks) to challenge patients and increase their engagement, distinguishing it from simpler `exposure scenarios`.
*   **Virtual Companion and Relaxation Techniques:** The system includes `virtual companions` that can provide psychological support and a readily available `menu with relaxation elements` (music, 360-degree media), offering immediate coping mechanisms during stressful situations.
*   **User-Centric Performance Optimization:** The paper details significant efforts in optimizing `VR environments` for `performance` on `all-in-one HMDs` (like `Oculus Quest 2`), including drastic reduction of `3D model complexity`, `occlusion culling`, and `light maps`, ensuring a smooth experience and reducing `motion sickness`.
*   **Speech-to-Text Integration for Glossophobia:** For `glossophobia`, PhoVR integrates `Microsoft Azure Speech to Text` for real-time analysis of `voice factors` (volume, speech rate, pauses), providing objective feedback to the user and `therapist`.

    While commercial systems like `C2Care` and `XRHealth` offer `therapist orchestration` and `personalized plans`, PhoVR's depth in `automatic adaptation` driven by `classified biophysical data` and the novel `environmental biofeedback` stand out. The `gamified approach` and `virtual companion` also enhance the user experience beyond what is typically described in academic `VRET` studies.

# 4. Methodology

## 4.1. Principles
The core idea behind the PhoVR system is to provide an immersive, controlled, and adaptive environment for `phobia therapy` using `virtual reality`. The theoretical basis is rooted in `exposure therapy`, where gradual and systematic confrontation with feared stimuli leads to a reduction in anxiety. The intuition is that by simulating real-world phobic situations in a safe `virtual environment`, coupled with real-time `biophysical monitoring` and `adaptive responses`, patients can effectively habituate to their fears and develop coping mechanisms. The system aims to make therapy more engaging through `gamification` and more effective through objective physiological data and personalized interventions, all managed by a `psychotherapist`.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. VR and Biophysical Acquisition Devices
The PhoVR system is designed for accessibility and flexibility, supporting both `all-in-one VR equipment` and `HMDs` requiring external processing.
*   **VR Playback Devices:**
    *   **All-in-one equipment:** `Oculus Quest series` (e.g., `Oculus Quest 2`) is preferred for its ease of use, simpler configuration, and reduced cost. However, the paper notes its lower processing power compared to desktop solutions.
    *   **HMDs with external processing:** `HTC Vive` and `Valve Index` are recommended for enhanced graphic quality and optimal `video flow` due to the higher processing power of connected `VR-compatible laptops` or `desktops`.
*   **Biophysical Data Acquisition Device:**
    *   **BITalino [36]:** Chosen for `electrodermal activity (EDA)` and `heart rate variability (HRV)` acquisition. Its selection was based on ease of use, ergonomics, and cost-effectiveness. This device captures physiological signals to objectively measure a patient's arousal levels during `VR exposure`.

### 4.2.2. Description of the Virtual Environments

#### 4.2.2.1. Tutorial Scene
The `tutorial scene` serves as an onboarding experience for new users, teaching them how to interact with the `VR environment` and `controllers`.
*   **Navigation:** Users learn how to move around the scene using the `joystick`.
*   **Controller Functions:**
    *   `Joystick`: For movement within the scene (up/down, right/left).
    *   `Joystick press`: Opens the `Main Menu` to modify `graphic quality`, `sound volume`, `movement speed`, `active scene`, etc.
    *   `A button`: Rotates the user to the left.
    *   `B button`: Rotates the user to the right.
    *   `Trigger`: Used for selecting or confirming actions (menus, `quests`).
    *   `Grip`: Used for `pickup tasks`.
*   **Task Types:** The `tutorial scene` introduces four core `gamified task` types:
    *   **Checkpoint:** The user must move to a specific `marker point`. A confirmation sound indicates successful completion.
    *   **Photo:** The user needs to take a picture of a designated `object` or `animal` (e.g., a statue, a seagull). An `outline` appears around the target when the user is in the vicinity. The `outline` changes from yellow to green when the `camera angle` and `object` location are correct, and the picture is taken by pressing the `Trigger` button.
    *   **Collection:** Users collect three or more blue `particles` by moving their `avatar` to collide with them.
    *   **Pickup:** Users pick up various `objects` (e.g., a candle, a bottle) by pointing their right hand at the `item` and pressing the `Grip` button. A `vibration` and a `semi-transparent circle` at the object's base confirm correct targeting.
        The following figure (Figure 1 from the original paper) shows an example of a `Collection task`:

        ![Figure 1. Collection task.](images/1.jpg)
        *该图像是一个虚拟环境中的收集任务示意图，展示了由蓝色光点表示的交互元素。用户在环境中完成特定任务，以优化其应对恐惧症的体验。*

    The following figure (Figure 2 from the original paper) shows an example of a `Pickup task`:

    ![Figure 2. Pickup task.](images/2.jpg)
    *该图像是一个示意图，展示了用户在虚拟现实环境中执行捡拾任务的场景。图中显示了一个棕色的圆柱形物体和一只白色手部模型，手部正准备抓取该物体。*

*   **Emotional Assessment:** During the scene, a panel periodically appears asking: "How easy was it to control your feelings?". This collects `subjective anxiety` data, which is later correlated with `biometric data`.
*   **Task Sequence Assessment:** After completing a `task sequence` (a series of tasks in a specific order), a form appears with three questions:
    *   "How easy was the task sequence?"
    *   "How easy was it to control your feelings?"
    *   "How enjoyable was the task sequence?"
        This feedback helps assess the difficulty and patient experience.
    The following figure (Figure 3 from the original paper) shows the form with assessment questions:

    ![Figure 3. Form with assessment questions.](images/3.jpg)
    *该图像是一个任务评估的界面，用户在此界面上需要对任务的难易程度、情绪控制的难易程度以及任务的喜好程度进行打分。界面包含了三个评分项，每个项都有从1到5的评分选项，并有一个提交按钮。*

#### 4.2.2.2. Virtual Environments for Acrophobia Therapy
Two natural and two urban environments are developed for `acrophobia` therapy.
*   **Natural Environment 1 (Mountain Scenario):**
    *   **Route:** Involves a `free walk` in an open area and a `cable car` ride.
    *   **Free Walk Options:** Walking on a `metal platform`, on the `edge of the main ridge`, or a `descent route` along a river. Paths are marked with special textures or `active task markers`.
        The following figure (Figure 4 from the original paper) shows the `metal platform`:

        ![Figure 4. Metal platform.](images/4.jpg)
        *该图像是一个虚拟现实场景，展示了一个金属平台延伸至壮观的山谷与湖泊之间。这个环境设计用于治疗高空恐惧症，用户可以在模拟的自然环境中进行治疗任务，从而减少对高度的恐惧感。*

    *   **Cable Car Route:** Divided into 10 segments, with stops at poles where tasks and questions are presented.
        The following figure (Figure 5 from the original paper) shows the `cable car`:

        ![Figure 5. Cable car.](images/5.jpg)
        *该图像是一个插图，展示了虚拟现实体验中的缆车视角，面朝一片山脉景观。用户在进行强迫症疗法时，可能会通过这个视角体验患有高处恐惧症的情况，旨在帮助他们克服对高度的恐惧。*

    *   **Performance Refinement:** Initially, this scene suffered from low `FPS` on `Oculus Quest`. Optimizations included drastically reducing the number of `trees` (expensive elements), replacing them with `optimized 3D models`, reducing terrain dimensions, and converting terrain to a `3D model` to lower `rendering operations`.
*   **Natural Environment 2 (Restricted Area):**
    *   **Layout:** A narrower area with two `steep cliffs` and a `metal bridge`. Tasks are performed on edges and the bridge, sometimes requiring `jumping` between cliffs.
        The following figure (Figure 6 from the original paper) shows the `metal bridge`:

        ![Figure 6. Metal bridge.](images/6.jpg)
        *该图像是一个插图，展示了一座金属桥的视图。桥面延伸至远方，背景模糊，营造出悬空的紧张感，可能用于治疗恐高症的虚拟现实场景。*

    *   **Atmosphere:** Enhanced with `low lighting`, `flocks of crows`, or `dense fog` to increase immersion and anxiety.
        The following figure (Figure 7 from the original paper) shows the `metal bridge covered in fog`:

        ![Figure 7. Metal bridge covered in fog.](images/7.jpg)
        *该图像是一个金属桥覆盖在雾气中的场景，突显了悬崖与崎岖地形之间的神秘氛围。桥的结构细致，加上飞翔的鸟群，使得整个画面充满了动感与紧张感，适合用作某心理治疗场景的视觉元素。*

*   **Urban Environment 1 (High-rise Building):**
    *   **Layout:** Populated with many buildings, but the user navigates only two `buildings`. Starts on a `terrace` and ascends to the `roof`.
        The following figure (Figure 8 from the original paper) shows the `initial building`:

        ![Figure 8. Initial building.](images/8.jpg)
        *该图像是一个三维建筑物的渲染图，展示了一个多层建筑的上方视角，屋顶平坦且与周围建筑物形成鲜明对比。该建筑与周边环境相结合，反映了对高处恐惧症（如高处恐惧症）的潜在治疗场景。*

    *   **Levels:**
        *   `Level 1`: `Terrace` at 66m (17th floor).
        *   `Level 2`: 108m (28th floor).
        *   `Level 3`: `Roof` at 131m (35th floor).
    *   **Navigation:** `Elevators` (metal platforms) change height levels. If the floor difference is too large, the `platform` stops midway for a task. `Bridges` connect the two buildings at `Level 2` and `Level 3`. These bridges have `thin metal bars` and `grated floors` to allow seeing the ground below.
        The following figure (Figure 9 from the original paper) shows the `metal bridge between the buildings`:

        ![Figure 9. Metal bridge between the buildings.](images/9.jpg)
        *该图像是一个示意图，展示了一座位于高层建筑之间的金属桥。桥面细密的网状结构和两旁的栏杆使得此场景认证出参与者在面对高处恐惧时的虚拟体验。*

    *   **Tasks:** Tasks are strategically placed to guide the user and test anxiety. Example tasks include `checkpoint`, `collecting objects` on a `metal platform` attached to the edge (forcing users to look down), and `taking a picture of a seagull` on a `metal bar`.
        The following figure (Figure 10 from the original paper) shows the `seagull on a metal bar`:

        ![Figure 10. Seagull on a metal bar.](images/10.jpg)
        *该图像是一个插图，描绘了一只海鸥栖息在一根金属横杆上，周围是城市建筑的环境。这幅插图可能与虚拟现实场景相关，用于治疗恐高症的虚拟环境展示。*

    *   **Duration:** A full run takes about 10 minutes, varying with patient speed, skill, and anxiety.
*   **Urban Environment 2 (Tilted Bridges):**
    *   **Layout:** Similar to the first urban scene but with different buildings, designed to induce higher anxiety through `transparent materials`.
    *   **Areas:** Three areas:
        *   **Area 1:** User starts on the first floor, uses an `elevator` to climb 40 floors to the `roof`, stopping at the 18th and 33rd floors for `photo tasks`.
        *   **Area 2:** Composed of `platforms` with `thin metal bars` and `grid floors`. `Bridges` are `tilted at different angles` to create an illusion of greater height.
            The following figure (Figure 11 from the original paper) shows a `tilted bridge`:

            ![Figure 11. Tilted bridge.](images/11.jpg)
            *该图像是一个插图，展示了一座倾斜的桥，连接着两座建筑物。图中可以看到桥的结构设计，以及周围城市环境的细节，为用户在虚拟现实中体验相关情境提供了视觉支持。*

        *   **Area 3:** `Metal platforms` attached to the `roof edge`, accessed via `elevator` and `bridge`.
    *   **Dynamic Adaptation Elements:** Bridges can be modified: `grating` replaced with `metal panel` or `glass`, `side bars` removed or replaced with `thicker bars` or `walls`. Similar changes apply to `elevators` and `platforms`.
    *   **Performance Optimization:** This scene also underwent significant optimization. Buildings with 12 million `vertices` were replaced with `3D models` having between 400,000 and 1,200,000 `vertices`, eventually reducing to 180,000-190,000 `vertices` for interactive buildings and 84 `vertices` for background buildings. This achieved an `FPS` of 20. `Occlusion culling` (avoiding rendering hidden objects) and `light maps` (pre-calculated lighting) were crucial for performance, especially for mobile systems.

#### 4.2.2.3. Virtual Environments for Claustrophobia Therapy
Three scenarios are designed for `claustrophobia` therapy.
*   **Tunnel Scene:**
    *   **Layout:** A dimly lit `tunnel` that gradually changes environmental parameters as the user progresses.
        The following figure (Figure 12 from the original paper) shows the `tunnel scene`:

        ![Figure 12. Tunnel scene.](images/12.jpg)
        *该图像是图示，展示了一个虚拟现实中的隧道场景。光线通过墙壁上的火把照亮了石墙，营造出一种阴暗而紧张的氛围，符合针对幽闭恐惧症治疗的场景设置。*

    *   **Dynamic Changes:** `Tunnel width` decreases, `ceiling level` lowers, `lighting` weakens, and `obstacles` become narrower.
    *   **Tasks:** A sequence of 10 tasks, including `photo tasks` (e.g., statue) and `pickup tasks` (e.g., candles). `Supplementary walls` were added as obstacles to further limit space.
*   **Labyrinth of Rooms Scene:**
    *   **Layout:** Similar atmosphere to the `tunnel` but with a different structure. User starts in a `large room` and navigates through `rooms of different sizes` that gradually decrease, culminating in a `small room` that can accommodate only one person.
        The following figure (Figure 13 from the original paper) shows the `labyrinth of rooms`:

        ![Figure 13. Labyrinth of rooms.](images/13.jpg)
        *该图像是一幅示意图，展示了一个迷宫般的房间环境，墙面由砖块构成，地面铺有石块。该环境可能用于治疗恐惧症的虚拟现实场景，用户需要在其中进行任务以促进治疗。*

*   **Cave Scenario:**
    *   **Layout:** Four scenes: three in different `cave areas` and one in an `open, relaxing space`. Difficulty increases across scenes.
        The following figure (Figure 14 from the original paper) shows the `cave`:

        ![Figure 14. Cave.](images/14.jpg)
        *该图像是一个虚拟环境的示意图，展示了一个洞穴内部的场景，墙壁是由大型岩石构成，地面覆盖着砖石。这个场景可能用于恐惧症治疗的虚拟现实体验，例如针对幽闭恐惧症的暴露疗法。*

    *   **Dynamic Changes:** `Cave height` gradually decreases from three times the player's height to a 1:1 ratio.
    *   **Tasks:** Player seeks a `portal` (wooden cabin) to the next level, randomly placed. Must stand in the cabin with the door closed for 15 seconds.
    *   **Gamification:** Collects `crystals` (blue, red, white) placed in hard-to-reach areas (e.g., behind `stalactites` and `stalagmites`) to encourage exploration. `Animals` (bats) and `objects` (skeletons, burning barrels) add atmosphere; bats fly if approached.

#### 4.2.2.4. Virtual Environment for Fear of Public Speaking Therapy
This environment places the user in a virtual room to give a presentation, with extensive configuration and analysis features.
*   **Features:**
    *   `Module for setting up the scene`: Configures room type, audience settings, and initial audience behavior.
    *   `Module for running the scene`: Immersion, free speaking or topic presentation.
    *   `Monitoring immersion factors`.
    *   `Real-time calculation` of `attention degree` for each `virtual audience` member.
    *   `Numerical display` and `color indicators` of `attention degree`.
    *   `Animations` corresponding to `attention degree`.
    *   `In-session analysis`.
    *   `Post-session analysis`.
*   **Scene Setup Module:**
    The following figure (Figure 15 from the original paper) shows the `module for setting up the scene`:

    ![Figure 15. Module for setting up the scene.](images/15.jpg)
    *该图像是一个示意图，展示了PhoVR系统中场景设置模块的界面。界面包括一个‘开始’按钮，以及可调节的观众设置选项，如观众分布模型和初始行为。用户可以通过这些控制选项来设置虚拟环境的参数，以便更好地进行恐惧症治疗。*

    *   **Room Types:**
        *   `Small room` (interview room): Max 4 seats.
        *   `Medium-sized room` (lecture hall): Max 27 seats.
        *   `Large room` (amphitheater): Max 124 seats.
            The following figure (Figure 16 from the original paper) shows the `amphitheater`:

            ![Figure 16. Amphitheater.](images/16.jpg)
            *该图像是一个虚拟教室场景，展现了具有不同发型和肤色的三维人群坐在教室内的情景。此设计可能用于公共演讲恐惧症的治疗，帮助用户适应在众人面前发言的环境。*

    *   **Audience Settings (Crowding):** Five predefined settings for `room population` (percentage of occupied seats): `Empty room (0%)`, `Small audience (25%)`, `Average audience (50%)`, `Large audience (75%)`, `Full room (100%)`.
    *   **Audience Distribution:** Configurable.
    *   **Initial Behaviors:** `Psychotherapist` can configure `personality factors` and their `distribution type` (`uniform` or `Gaussian`). Factors are: `Interest in the topic`, `Tiredness`, and `Distraction`.
*   **Session Steps:**
    1.  `Placing the user` within the configured scene.
    2.  `Speaking freely` or `presenting a topic`.
        The following figure (Figure 17 from the original paper) shows the `presentation scene`:

        ![Figure 17. Presentation scene.](images/17.jpg)
        *该图像是一个虚拟现实环境中的演示场景，展示了一块显示“Prezentare Test”字样的屏幕，旁边有一扇门和一块时钟，这个场景用于帮助用户克服公众演讲恐惧症。*

*   **Monitoring Factors (Real-time):**
    *   **Audience Factors:** (Configured initially) `Interest in the topic`, `Tiredness`, `Distraction`.
    *   **User Factors (Directly influence audience factors):**
        *   **Spatiality Factors:**
            *   `Position relative to the audience`: Distance between presenter and each audience member.
            *   `Hand movement`: Total length of hand movement trajectories over a defined time interval (e.g., last 10 seconds).
            *   `Gaze Direction`: Influences `audience factors` based on how far each audience member is from the center of the presenter's `field of view`. Only applies to audience members within a preset `angle of attention` (e.g., 60 degrees). `Attention factor` is calculated based on the angle between the presenter's `gaze direction` and the `audience member's position`.
                The following figure (Figure 18 from the original paper) shows the `attention angle`:

                ![Figure 18. Attention angle.](images/18.jpg)
                *该图像是示意图，展示了用户与观众之间的注意角度。图中黑色圆圈代表观众，中心为用户，红色和紫色线条描绘了注意力的转移角度及其范围，标记了注意角度的位置。*

        *   **Voice Factors:**
            *   `Voice volume`: `Decibel level` recorded by the microphone. Presenter adjusts volume; large or frequent fluctuations negatively affect score.
            *   `Speech rate`: Number of words per minute, calculated using `speech-to-text algorithm` (`Microsoft Azure Services [37]`).
            *   `Pauses between words`: Duration of silence. `Fluency` and `monotony` degrees are calculated empirically.
*   **Real-time Feedback to User:**
    *   `Animations` for `audience members` (corresponding to `attention score`).
    *   `Numerical values` and `color codes` (optional, can be activated).
    *   Other `informative displays`: session duration, average `attention score`, `speech-to-text` result.
    *   **Performance Statistics Display:** `General audience interest level`, `heart rate`, `voice volume`, `rhythm` are displayed behind the classroom for the presenter. Data shown numerically and with `color indicators` (green, yellow, red) for level.
        The following figure (Figure 19 from the original paper) shows the `performance statistics in real time`:

        ![Figure 19. Performance statistics in real time.](images/19.jpg)
        *该图像是一个虚拟现实场景，展示了一组参与者坐在椅子上，背景墙上显示了一些性能统计数据，包括兴趣、脉搏、音量、节奏和清晰度。数据如：兴趣73（感兴趣）、脉搏92、音量90（正常）、节奏1.4（正常）、清晰度78（一般）和双手0.09（非常平静）。*

*   **Post-session Analysis:**
    *   `Plots` for all monitored factors.
    *   `Comparative data` with `previous sessions` (history/evolution).

### 4.2.3. Technologies and Performance Evaluation
*   **Virtual Environment Development:**
    *   `Unity 3D [38]`: The primary `graphics engine` with `VR support`.
    *   `SteamVR plugin [39]`: An extension for `Unity 3D` for `VR` functionality.
    *   `Adobe Fuse [40]`: Used for building `3D models` of the audience.
    *   `Adobe Mixamo [41]`: Used for adding `animations` to `3D models` (e.g., "Interested" or "Not interested" states for audience members).
    *   `Microsoft Azure Speech to Text [42]`: Used for analyzing recorded voice, generating text, and then processing it for `speech rate` and `pauses`.
*   **Performance Evaluation:**
    *   **Metric:** `Frame rate` (`frames per second`, `fps`) and `processing time per frame`.
    *   **Target:** `33 milliseconds per frame` (corresponding to `30 fps`) to ensure stable performance and prevent `motion sickness`.
    *   **Test Systems:**
        *   **Oculus Quest 2 (All-in-one HMD):**
            *   Processor: `Qualcomm Snapdragon XR2 7 nm`
            *   Memory: `6 GB DDR5`
            *   Video card: `Adreno 650 x 7 nm`
            *   Storage: `256 GB`
            *   Display refresh rate: `72 Hz, 90 Hz, 120 Hz`
            *   Resolution: `1832 x 1920` (per eye)
        *   **Desktop System (with HMD VR device):**
            *   Processor: `Ryzen 7 3700X 3.6 GHz`
            *   Memory: `32 GB DDR4 3200 MHz`
            *   Video card: $1060 GTX - 6 GB$
    *   **Results:**
        The following are the results from Table 1 of the original paper:

        <table>
        <thead>
        <tr>
        <td rowspan="2">Scene</td>
        <td colspan="2">Test Performance (fps)</td>
        <td colspan="2">Oculus Quest 2</td>
        </tr>
        <tr>
        <td>Minimum Value</td>
        <td>Average Value</td>
        <td>Minimum Value</td>
        <td>Average Value</td>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td>Main menu</td>
        <td>60</td>
        <td>72</td>
        <td>74</td>
        <td>119</td>
        </tr>
        <tr>
        <td>Tutorial scene</td>
        <td>55</td>
        <td>60</td>
        <td>71</td>
        <td>120</td>
        </tr>
        <tr>
        <td>Acrophobia—natural environment</td>
        <td>30</td>
        <td>35</td>
        <td>60</td>
        <td>86</td>
        </tr>
        <tr>
        <td>Acrophobia—urban environment</td>
        <td>35</td>
        <td>40</td>
        <td>68</td>
        <td>90</td>
        </tr>
        <tr>
        <td>Claustrophobia—tunnel</td>
        <td>57</td>
        <td>60</td>
        <td>74</td>
        <td>114</td>
        </tr>
        <tr>
        <td>Claustrophobia—cave</td>
        <td>58</td>
        <td>60</td>
        <td>70</td>
        <td>105</td>
        </tr>
        <tr>
        <td>Interview scenario</td>
        <td>-</td>
        <td>-</td>
        <td>87</td>
        <td>115</td>
        </tr>
        <tr>
        <td>Classroom scenario</td>
        <td></td>
        <td></td>
        <td>98</td>
        <td>117</td>
        </tr>
        <tr>
        <td>Amphitheater scenario</td>
        <td></td>
        <td></td>
        <td>64</td>
        <td>91</td>
        </tr>
        </tbody>
        </table>

    *   **Interpretation:** The table shows that minimum and average `FPS` values are generally adequate for safe `VR` operation, aiming to prevent `dizziness` from low `update rates`. Even on `Oculus Quest 2`, most scenes maintain acceptable performance, with `acrophobia` natural environment having a minimum of 30 `fps`, which is the threshold for smooth experience.
*   **Microsoft Azure Speech to Text:**
    *   **Functionality:** Enables real-time transcription of audio to text.
    *   **Integration:** Used in the `glossophobia` therapy application to process spoken words, sending requests to the `cloud server` for transcription. The resulting text is then analyzed for `speech rate`, `clarity`, etc.
    *   **Performance:** Achieves over 98% accuracy. Request and analysis durations are typically within one second, ensuring real-time feedback.

### 4.2.4. Biofeedback Integrated in the Virtual Environment
`Biofeedback` is provided in two ways:
*   **Direct Visual Biofeedback:** Displayed on the `HMD` for the patient and on the `control panel` for the `therapist`. This allows direct observation of `HR` and `EDA` data.
*   **Environmental Biofeedback:** For natural environments, `biophysical data` (`HR` and `EDA`) dynamically alter scene properties. This includes:
    *   `Color of ambient light`
    *   `Predominant color of the sky`
    *   `Density of clouds` and their `color`
        The model aims to ensure `immersion` and encourage the patient to maintain a relaxed emotional state.
    The following figure (Figure 20 from the original paper) shows `biofeedback integrated into the scene's elements`:

    ![Figure 20. Biofeedback integrated into the scene's elements.](images/20.jpg)
    *该图像是插图，展示了在虚拟现实环境中不同天空场景的模拟效果。左侧为晴朗的蓝天，中间为多云场景，右侧为日落时分的天空，反映了PhoVR系统中环境的动态适应性。*

### 4.2.5. Signaling Difficult Moments
The system identifies critical moments of `anxiety` with high confidence.
*   **Virtual Companion:** An animated `virtual companion` (e.g., an `eagle` or a `fox`) can accompany the user permanently, be hidden on demand, or appear automatically when needed. This companion provides simple, general advice, reminds of `relaxation methods`, offers support, and distracts from `phobic stimuli`. It helps reduce feelings of `isolation` and increases `engagement`, especially for younger users.
    The following figure (Figure 21 from the original paper) shows `virtual companions`:

    ![Figure 21. Virtual companions.](images/21.jpg)
    *该图像是虚拟现实环境中的生物伴侣，左侧展示了一只展翅的猫头鹰，右侧则是一只带着小猎物的狐狸。它们的设计旨在帮助用户在治疗过程中感受到安全与放松。*

*   **Support Menu:** When the system detects a critical moment (e.g., `biometric information` exceeds a `critical threshold`), the user is alerted and can open a menu with `support elements for relaxation`.
    *   **Content:** Relaxing songs, 360-degree images or videos.
    *   **Automation:** The `AI biometric information monitoring system` can automatically open this panel or even terminate the session if `biometric values` remain above the `critical threshold` for a prolonged period.
        The following figure (Figure 22 from the original paper) shows the `menu with support elements for relaxation`:

        ![Figure 22. Menu with support elements for relaxation.](images/22.jpg)
        *该图像是一个展示放松内容的界面，其中包含放松音乐、视频和图像的选项。界面显示了四个放松内容的缩略图，包括“山峰视图”、“森林视图”、“放松钢琴声”和“河流景观”。每个项目还附有时长信息，用于帮助用户选择适合的放松媒介。*

### 4.2.6. Dynamic Adaptation of the Virtual Environment
The system allows `gradual exposure` to `fear-inducing elements` through `dynamic adaptation`, which can be system-initiated or therapist/patient-controlled.
*   **Acrophobia Scenes:** Elements like `bridges` can be modified:
    *   `Gratings` can be replaced with `metal panels` (obscuring view below) or `glass` (increasing transparency).
    *   `Metal side bars` can be removed or replaced with `thicker bars` or `walls of various heights`.
    *   Similar modifications apply to `elevators` and `platforms`.
*   **Claustrophobia Scenes:** `Dynamic adaptation` primarily involves automatically changing the `degree of illumination` of the scene.
*   **Scene Recommendation:** At the end of a session, based on patient `feedback` and `biometric information analysis`, a new scene with an `appropriate level of difficulty` is recommended.

### 4.2.7. The Control Panel
A separate application for `psychotherapists` to manage patients and therapy sessions.
*   **Development Technologies:** `Electron.js platform` (using `HTML5 technologies`), `object-relational mapping (ORM)`, `relational databases`, and `Scrum` software development methodology.
*   **Functionalities:**
    *   `Administration`: Through a `therapist account`.
    *   `Patient management`: Adding and deleting patients.
    *   `Patient profile configuration`: Defining and editing `clinical profile sheets`, which are flexible and based on simple title/description entries.
    *   `Recording therapy sessions`: Saving session data in the `cloud`.
    *   `Displaying patient's therapy history`.
        The following figure (Figure 23 from the original paper) shows the `session history`:

        ![Figure 23. Session history.](images/23.jpg)
        *该图像是一个图表，展示了在两个不同虚拟环境（房间迷宫和隧道）中进行暴露治疗时的生理数据。图表中包含心率（HR）和皮肤电反应（EDA）随时间的变化，时间分别为1分20秒和1分44秒。*

    *   `Visualization of therapy session results`: With options to add `comments`, `notes`, `scores`, and `progress information`.
        The following figure (Figure 24 from the original paper) shows a `detailed view of a session`:

        ![Figure 24. Detailed view of a session.](images/24.jpg)
        *该图像是一个界面截图，展示了2022年5月12日的一个虚拟现实疗法会话信息。会话持续时间为2分钟22秒，界面左上方显示心率（HR）和皮肤电反应（EDA）的变化曲线，右下方则呈现场景事件记录，直观反映了用户在疗法过程中的生理反应与行为轨迹。*

    *   `Visualization of patient progress`: Using easy-to-interpret graphs.
        The following figure (Figure 25 from the original paper) shows a `graphic representation of heart rate data`:

        ![Figure 25. Graphic representation of heart rate data.](images/25.jpg)
        *该图像是一个图表，展示了心率数据的可视化表示，当前心率为95。图表中以绿色区间表示正常心率范围，黄色和红色区间表示心率偏高和过高的状态。*

    The following figure (Figure 26 from the original paper) shows a `graphic representation of electrodermal activity data`:

    ![Figure 26. Graphic representation of electrodermal activity data.](images/26.jpg)
    *该图像是一个仪表图，用于图示电皮肤反应的数据，其单位为微西门子 (µS)。图中显示的当前电导值为9.8 µS，指针指向该值，采用多彩色区域来表示不同的电导水平。*

    *   `Monitoring patients' therapy sessions`: `Live screen-casting` from the `VR device` and `real-time biometric data visualization`.
        The following figure (Figure 29 from the original paper) shows `monitoring patient therapy sessions—live view from the VR device`:

        ![Figure 29. Monitoring patient therapy sessions—live view from the VR device.](images/29.jpg)
        *该图像是一个虚拟现实场景，展示了一座悬索桥，桥梁位于高悬的悬崖之上，整体环境呈现淡雅的灰色调。这一场景被用于针对恐高症的治疗，使用户能够在安全的虚拟环境中体验恐惧情境。*

    *   `Session recordings' playback`: A desktop application reconstructs the scene and patient actions using saved information. Includes `time control bar` for scrolling. `Special sessions` with only `biometric data monitoring` (no `VR environments`) can also be recorded and played back.
        The following figure (Figure 27 from the original paper) shows `session playback`:

        ![Figure 27. Session playback.](images/27.jpg)
        *该图像是一个虚拟现实场景，展示了高楼之间的绳索走廊，意在模拟针对恐高症的暴露疗法。用户可以在这个模拟环境中进行心理治疗，体验克服高处恐惧的过程。*

*   **Validation of Technical Capabilities:** Tested `administration`, `patient management`, `profile configuration`, `communication with PhoVR hardware` (checking `WebSocket connection` with `Oculus Quest 2` and `HMD connected to PC`), `live monitoring`, `session recording/saving`, `history viewing`, and `playback`.
    The following figure (Figure 28 from the original paper) shows an `HMD connected to a laptop computer`:

    ![Figure 28. HMD connected to a laptop computer.](images/28.jpg)
    *该图像是一个展示虚拟现实设备的照片，其中一名用户佩戴着头戴式显示器（HMD），正在与一台连接的笔记本电脑互动，显示屏上可能是虚拟环境的场景。设备的连接线和其他电子设备也在桌子上可见。*

# 5. Experimental Setup

## 5.1. Datasets
The human subjects used for testing the PhoVR system consisted of `psychotherapists` and `subjects` with varying degrees of anxiety.
*   **Participants:**
    *   **Psychotherapists:** Seven agreed to participate. They first tested the system as patients to understand its functionality and then supervised sessions with subjects.
    *   **Subjects:** 19 individuals (4 male, average age $= 20.25 (\pm 0.96)$; 15 female, average age $= 19.86 (\pm 0.95)$). Critically, these subjects were *not diagnosed* with `acrophobia` or `claustrophobia`. This is an important detail for interpreting the physiological stress responses, as their reactions might be lower than those of diagnosed phobic patients.
*   **Exclusion Criteria for Subjects:**
    1.  Disorders that could mimic or exacerbate `anxiety symptoms` (e.g., `thyroid disorders`, `epilepsy`, `traumatic brain injury`).
    2.  Individuals currently taking substances or medications that could influence `anxiety symptoms` (e.g., `alcohol`, `benzodiazepines`, `antidepressants`, or `antipsychotics`).
*   **Testing Location:** Anthropology and Ethology Laboratory of the University of Bucharest, Faculty of Biology.
*   **Ethics Approval:** Obtained from the University of Bucharest Ethics Committee (protocol code number 28/18.06.2021). Informed consent was obtained from all subjects.
*   **Testing Procedure for Subjects:**
    1.  Filling in the `STAI questionnaire` (State-Trait Anxiety Inventory).
    2.  Presentation of `VR equipment`.
    3.  Presentation of the `biometric data acquisition device`.
    4.  Short training session on navigation and tasks.
    5.  Brief presentation of testing scenarios.
    6.  Three main stages of functionality testing: `acrophobia` and `claustrophobia` therapy, `public speaking therapy`.
    7.  System evaluations by filling in an `evaluation questionnaire`.
*   **Public Speaking Testing Session:**
    *   Involved four subjects under a specialized therapist's supervision.
    *   Protocol: Giving a presentation on a familiar topic (e.g., anatomy of the spine). Presentation files provided one day prior.
    *   Scenario: A classroom with 50% seat occupancy was chosen.
    *   Exposure: Each participant underwent two exposure sessions, with a 10-minute break. Average presentation duration was 5 minutes.

## 5.2. Evaluation Metrics
The evaluation employed a combination of subjective and objective measures.

*   **1. STAI (State-Trait Anxiety Inventory)**
    *   **Conceptual Definition:** The `STAI` is a widely used psychological self-report assessment tool designed to measure two distinct types of anxiety: `state anxiety` and `trait anxiety`. `State anxiety` refers to the transient, situational anxiety experienced at a particular moment, reflecting current feelings of nervousness, apprehension, and arousal. `Trait anxiety` refers to an individual's stable predisposition to experience anxiety over time, reflecting a general tendency to be anxious across various situations. It helps distinguish between a temporary anxious state and a more enduring anxious personality trait.
    *   **Mathematical Formula:** The paper describes the `STAI` as comprising 40 items, 20 for each subscale (state and trait), scored on a 4-point `Likert scale`: 1—not at all, 2—somewhat, 3—moderately, and 4—very much. While a specific mathematical formula for calculating the raw score isn't provided, it typically involves summing or averaging the scores of relevant items, with some items being reverse-scored.
        For illustrative purposes, if $N_S$ is the number of items for the state anxiety scale and $N_T$ is the number of items for the trait anxiety scale, and $s_{i}$ is the score for item $i$, the raw score for each scale would generally be:
        $$
        \text{Raw Score}_{\text{State}} = \sum_{i=1}^{N_S} s_i \quad \text{(after reverse-scoring negative items)}
        $$
        $$
        \text{Raw Score}_{\text{Trait}} = \sum_{i=1}^{N_T} s_i \quad \text{(after reverse-scoring negative items)}
        $$
    *   **Symbol Explanation:**
        *   $N_S$: Number of items in the state anxiety subscale (20 items).
        *   $N_T$: Number of items in the trait anxiety subscale (20 items).
        *   $s_i$: Score for the $i$-th item on the 4-point `Likert scale` (1 to 4).
        *   $\sum$: Summation operator.
        *   `Raw Score`: The total score for a given subscale.
    *   **Interpretation:** Raw scores are converted to `percentiles` by referring to normative data. Scores above the mean `percentile` are considered high, indicating the presence of `state anxiety` (S-Anxiety) or `trait anxiety` (T-Anxiety). Scores below the mean indicate no anxiety.

*   **2. EDA-T (Electrodermal Activity - Tonic Component)**
    *   **Conceptual Definition:** `EDA-T` measures the slow-changing, baseline level of skin conductance, reflecting general physiological arousal mediated by the `sympathetic nervous system`. It represents an individual's overall level of wakefulness, attention, or stress, rather than rapid, event-specific responses. An increase in `EDA-T` suggests heightened arousal or stress.
    *   **Mathematical Formula:** The paper describes `EDA-T` as the tonic component of the `electrodermal activity signal`. While `EDA` is measured in `microsiemens (`\mu S`)`, the extraction of the tonic component typically involves signal processing techniques like low-pass filtering or decomposition algorithms (e.g., continuous decomposition analysis or deconvolution) to separate slow-varying (`tonic`) from rapid-varying (`phasic`) components. The formula for `EDA-T` is not explicitly provided as a direct calculation, but rather as an extracted component of the overall `EDA` signal. It is then averaged over time windows.
        If `SC(t)` is the raw skin conductance signal over time, `EDA-T` would be a component `T(t)` such that $SC(t) = T(t) + P(t) + n(t)$, where `P(t)` is the `phasic component` and `n(t)` is noise. The average value over a `sliding window` of width $\Delta t$ and displacement step $\delta t$ would be:
        $$
        \overline{\text{EDA-T}}_{k} = \frac{1}{\Delta t} \int_{t_k}^{t_k+\Delta t} T(t) \, dt
        $$
    *   **Symbol Explanation:**
        *   `SC(t)`: Raw skin conductance signal at time $t$.
        *   `T(t)`: Tonic component of the skin conductance signal at time $t$.
        *   `P(t)`: Phasic component of the skin conductance signal at time $t$.
        *   `n(t)`: Noise component.
        *   $\overline{\text{EDA-T}}_{k}$: Average tonic `electrodermal activity` during the $k$-th time window.
        *   $\Delta t$: Width of the `sliding window` (10 seconds in the study).
        *   $t_k$: Start time of the $k$-th `sliding window`.
        *   $\int$: Integral operator.
    *   **Measurement:** The average values of `EDA-T` were calculated using a `sliding window` of width 10 seconds and a displacement step of 5 seconds.

*   **3. HR (Heart Rate)**
    *   **Conceptual Definition:** `Heart rate` measures the number of heartbeats per minute (`bpm`). It is a direct physiological indicator of cardiovascular activity and a widely recognized proxy for physiological arousal, stress, or excitement. An elevated `HR` generally correlates with increased emotional or physical stress.
    *   **Mathematical Formula:** `HR` is calculated from the `photoplethysmography (PPG)` signal, which measures blood volume changes in the microvasculature. The `PPG` signal allows for the detection of individual heartbeats. If $t_i$ and $t_{i+1}$ are the times of two consecutive heartbeat peaks (R-peaks), the `instantaneous heart rate` is:
        $$
        \text{HR}_{\text{instantaneous}} = \frac{60}{(t_{i+1} - t_i)} \quad \text{[bpm]}
        $$
        The average `HR` over a `sliding window` of width $\Delta t$ and displacement step $\delta t$ would be the average of these instantaneous rates or derived from the number of peaks within the window.
    *   **Symbol Explanation:**
        *   $\text{HR}_{\text{instantaneous}}$: Instantaneous heart rate in beats per minute.
        *   $t_i$: Time of the $i$-th heartbeat peak.
        *   $t_{i+1}$: Time of the $(i+1)$-th heartbeat peak.
        *   $\overline{\text{HR}}_{k}$: Average `heart rate` during the $k$-th time window.
        *   $\Delta t$: Width of the `sliding window` (10 seconds in the study).
        *   $t_k$: Start time of the $k$-th `sliding window`.
    *   **Measurement:** The average values of `HR` were calculated using a `sliding window` of width 10 seconds and a displacement step of 5 seconds. A `PPG signal` quality criterion was applied: the `PPG signal` amplitude must be greater than a `threshold value` at least 90% of the time, determined by visual inspection to ensure accurate `HR` calculation. 66 out of 76 recorded sessions met this criterion.

## 5.3. Baselines
The paper does not compare PhoVR against other established `VRET` systems or traditional `in vivo exposure therapy` in a quantitative experimental setup within this publication. Instead, the `training sessions` (considered free of `phobic stimuli`) served as a `baseline` for comparing the `biophysical signal recordings` from the actual `VR exposure environments`. This allowed the researchers to assess the `stress-inducing potential` of their virtual scenarios. The effectiveness of the system in therapy will be quantitatively evaluated in a future, longer-term study with diagnosed patients, where comparisons might be made to other therapeutic approaches.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. STAI Questionnaire Results
*   **S-Anxiety (State Anxiety):** Most subjects (16 out of 19) showed high scores on the `S-Anxiety scale`. This indicates that they were prone to experiencing anxiety as a temporary state, particularly when exposed to situations perceived as physically dangerous or psychologically stressful. This is a desirable characteristic for participants in an `exposure therapy` study, as it suggests they would likely exhibit measurable anxious responses in the `VR environments`.
*   **T-Anxiety (Trait Anxiety):** Similarly, most subjects (16 out of 19) also recorded high scores on the `T-Anxiety scale`. High `T-Anxiety` scores suggest a stable predisposition towards anxiety. The paper notes that high scores on this scale warrant a more complex evaluation as they can sometimes indicate depression.
*   **Individual Cases:**
    *   One subject with a high `T-Anxiety` score (58) had an average `S-Anxiety` score (40), indicating good `self-control` in specific situations despite a general anxious disposition.
    *   Another subject with an average `T-Anxiety` score (38) had a high `S-Anxiety` score (50), implying that certain situations could have a strong emotional impact on them.
    *   One subject showed average scores on both scales, suggesting a balanced emotional state regarding anxiety.

        These `STAI` results were crucial for `psychotherapists` to select the `optimal virtual therapy scenario` for each individual, ensuring that the chosen `VR environment` was appropriate for their baseline anxiety levels.

### 6.1.2. Biophysical Signal Analysis (EDA-T and HR)
This analysis aimed to objectively assess the `stress-inducing potential` of the `VR environments` by comparing `electrodermal activity (EDA-T)` and `heart rate (HR)` during `training sessions` (considered non-phobic) and `VR exposure` scenarios.
*   **Methodology:**
    *   `EDA-T` and `HR` average values were calculated using a `sliding window` of 10 seconds with a 5-second displacement step.
    *   `PPG signal` quality was ensured; only sessions where `PPG amplitude` was above a `threshold` for at least 90% of the time were included (66 out of 76 sessions).
    *   A `one-way ANOVA` was conducted to compare `EDA-T` and `HR` values across `environment types` (`Training`, `City`, `Bridge`, `Tunnel`, `Labyrinth`).
    *   `Post hoc Bonferroni-corrected tests` were used for `pairwise comparisons`.

*   **EDA-T Results:**
    *   Mean values: $\text{M}_{\text{Training}} = 8.97 (\pm 6.23) \mu S$; $\text{M}_{\text{City}} = 8.53 (\pm 6.86) \mu S$; $\text{M}_{\text{Bridge}} = 11.52 (\pm 6.76) \mu S$; $\text{M}_{\text{Tunnel}} = 11.70 (\pm 6.77) \mu S$; and $\text{M}_{\text{Labyrinth}} = 8.10 (\pm 2.89) \mu S$.
    *   `Unifactorial analysis` revealed a significant difference between these values: $\operatorname {F} (5, 1871) = 16.72, p < 0.001$. This indicates that the type of `VR environment` had a statistically significant effect on `EDA-T`.
    *   `Significant mean differences (MDs)` from `post hoc tests`:
        *   `Training` vs. `Bridge`: $\text{MD} = -2.55 \mu S, p < 0.001$ (Bridge significantly higher).
        *   `Training` vs. `Tunnel`: $\text{MD} = -2.72 \mu S, p < 0.001$ (Tunnel significantly higher).
        *   `City` vs. `Bridge`: $\text{MD} = -2.99 \mu S, p < 0.001$ (Bridge significantly higher).
        *   `City` vs. `Tunnel`: $\text{MD} = -3.17 \mu S, p < 0.001$ (Tunnel significantly higher).
        *   `Bridge` vs. `Labyrinth`: $\text{MD} = +3.42 \mu S, p = 0.002$ (Bridge significantly higher).
        *   `Tunnel` vs. `Labyrinth`: $\text{MD} = +3.60 \mu S, p < 0.001$ (Tunnel significantly higher).
    *   **Interpretation:** The `Bridge` (acrophobia) and `Tunnel` (claustrophobia) environments induced significantly higher `EDA-T` (indicating higher arousal) compared to the `Training` and `City` environments. The `Labyrinth` environment showed lower `EDA-T` than `Bridge` and `Tunnel`. This confirms that at least three of the `VR environments` (City, Bridge, and Tunnel) have a significant potential to induce stress, even in non-phobic subjects. The `City` environment, surprisingly, showed a lower mean `EDA-T` than `Training`, but this difference was not identified as significant in the pairwise comparisons.

*   **HR Results:**
    *   Mean values: $\text{M}_{\text{Training}} = 83.52 (\pm 10.56)$ `bpm`; $\text{M}_{\text{City}} = 85.39 (\pm 9.78)$ `bpm`; $\text{M}_{\text{Bridge}} = 82.73 (\pm 9.77)$ `bpm`; $\text{M}_{\text{Tunnel}} = 82.90 (\pm 10.38)$ `bpm`; and $\text{M}_{\text{Labyrinth}} = 83.78 (\pm 5.36)$ `bpm`.
    *   `Unifactorial analysis` revealed a significant difference between these values: $\operatorname {F} (5, 1871) = 4.81, p < 0.001$.
    *   `Significant mean differences (MDs)` from `post hoc tests`:
        *   `Training` vs. `City`: $\text{MD} = -1.87 \text{ bpm}, p = 0.019$ (City significantly higher).
        *   `City` vs. `Bridge`: $\text{MD} = +2.66 \text{ bpm}, p = 0.017$ (City significantly higher).
        *   `City` vs. `Tunnel`: $\text{MD} = +2.49 \text{ bpm}, p = 0.002$ (City significantly higher).
    *   **Interpretation:** The `City` environment induced a significantly higher `HR` compared to `Training`, `Bridge`, and `Tunnel`. This suggests that the `City` environment, despite showing lower `EDA-T` than some other scenarios, still elicited a measurable physiological stress response in terms of `heart rate`. The `Bridge` and `Tunnel` scenarios did not show significantly higher `HR` compared to `Training` in these pairwise comparisons, which contrasts slightly with the `EDA-T` findings. However, the overall `unifactorial analysis` confirmed that environment type influences `HR`. The presence of stress in these non-phobic subjects validates the scenarios' ability to trigger physiological responses, which would likely be amplified in truly phobic individuals.

### 6.1.3. Psychotherapist Feedback
*   **VR Adoption:** None of the seven `psychotherapists` had prior `VR` experience in therapy. Six expressed interest in integrating `VR technology` occasionally, indicating a positive reception and perceived potential.
*   **Control Panel:** Four found the `control panel` functionalities sufficient, while three requested additional features.
*   **Biophysical Data:** All seven `psychotherapists` found the `biophysical data acquisition system` useful for monitoring `emotional states`.
*   **VR Equipment Impact:** Four believed `VR equipment` didn't hinder therapy, while three noted it could hinder certain patients.
*   **Scenario Realism:** Five considered scenarios "a little artificial" but accurate and adequate for therapy. One found `phobic stimuli` only effective for `severe phobias`, and another deemed them "completely inefficient."
*   **System Effectiveness:** One `psychotherapist` found the system highly effective, another stated it could be "real help" with improvements, and five considered it useful for `certain patients` with `certain phobic pathologies`.
*   **Feasibility:** The majority suggested a `clinical center` model for `PhoVR` due to the small number of `acrophobia` and `claustrophobia` patients in individual practices. They anticipated individual office adoption if `VR equipment` costs decrease and performance increases.
*   **Therapist Control:** Psychologists requested greater control over the scenario flow, shifting from automatic adaptation to `therapist-orchestrated psychotherapy`.
*   **Public Speaking Scenarios:** Both participants and `psychotherapists` gave positive feedback regarding the `realism` and `effectiveness` of `public speaking scenarios`.

### 6.1.4. System Weaknesses & Suggestions for Improvement
*   **Identified Weaknesses:**
    *   Limited range of `phobias` addressed.
    *   `Familiarization time` for the system.
    *   Difficulty in performing `certain movements`.
    *   `Dizziness` caused by `VR equipment`.
*   **Suggestions for Improvement:**
    *   Address `multiple phobias`.
    *   Improve `feeling of realism`.
    *   Greater `freedom of movement` (left hand, body).
    *   Longer `virtual environments` with more dispersed tasks.
    *   Use `real images` from `natural environments`.
    *   `Visual qualitative improvement`.
    *   `Psychologist control` over the entire therapy course.
    *   Expanding to `agoraphobia` and `aerophobia`.
    *   Customizing `virtual environments` based on therapist/patient descriptions.
    *   Improving `patient performance analysis` and `progress tracking` with multiple plots and comparative analyses, and customizable dashboards.
    *   Optimizing `real-time playback performance`, especially in `locally complex` areas.
    *   Reducing `purchase cost` for individual users (e.g., integrating `commercial fitness trackers` for `biometric data`).
    *   Allowing `therapists` to `embody audience members` and interact directly with patients.
    *   More `varied` and less `predictable scenarios`, potentially using `randomized elements` in pre-orchestrated scenes.

## 6.2. Data Presentation (Tables)
The paper provided one table, which has been presented in the Methodology section (Section 4.2.3).

## 6.3. Ablation Studies / Parameter Analysis
The paper does not present explicit `ablation studies` to quantify the contribution of individual components (e.g., `gamification`, `biofeedback integration`, `dynamic adaptation`) to the system's overall effectiveness or anxiety reduction. Similarly, a detailed `parameter analysis` (e.g., how different `angle of attention` values affect `glossophobia` outcomes) is not provided. The performance optimization efforts described for `acrophobia` and `claustrophobia` scenes (`3D model simplification`, `occlusion culling`, `light maps`) are mentioned as technical improvements for `frame rate` but not explored as part of therapeutic efficacy. The evaluation in this paper focuses on qualitative feedback and the system's ability to induce physiological responses rather than component-level effectiveness.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper concludes that PhoVR is a robust `virtual reality exposure therapy (VRET)` system designed for treating `acrophobia`, `claustrophobia`, and `glossophobia`. It successfully integrates `realistic` and `immersive environments` with `gamified elements`, `biophysical data acquisition` (`EDA` and `HR`), `automatic anxiety classification`, `biofeedback` embedded in scene elements, `dynamic environment adaptation`, and `relaxation techniques`. A dedicated `control panel` empowers `psychotherapists` with patient and session management, real-time monitoring, and advanced analysis tools. The qualitative validation through `psychotherapists` and test subjects yielded overwhelmingly positive feedback on the system's `concept`, `realization`, `ergonomics`, and `utility`, positioning `PhoVR` as a functional prototype with high potential for a `commercial product`. The physiological data collected also confirmed the scenarios' ability to induce stress even in non-phobic subjects.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations of the current PhoVR prototype:
*   **Controller-based Locomotion:** This provides a sense of security and control but can increase `motion sickness` and reduce the `sense of presence`.
*   **Limited Audience Interaction (Glossophobia):** In `public speaking therapy`, `audience members` can change behavior but cannot directly interact by asking questions or initiating dialogue.
*   **Scenario Variety and Predictability:** The `virtual environments` are less varied and more predictable than the real world, potentially limiting long-term effectiveness.

    Based on feedback and identified opportunities, future work aims to:
*   **Expand Phobia Coverage:** Target `agoraphobia` and `aerophobia`.
*   **Customizable Environments:** Investigate `generating virtual environments` based on `therapist` or `patient descriptions` for enhanced personalization.
*   **Advanced Data Analysis:** Improve the `control panel's` `visual analysis` with more plots, comparative analyses, and customizable `statistics widgets`.
*   **Local Performance Optimization:** Conduct detailed analysis of `scene complexity` at `local task points` to ensure consistent `real-time playback performance`.
*   **Cost Reduction for Individual Users:** Investigate integrating `commercial devices` (e.g., `fitness trackers`) for `biometric data acquisition` to lower the system's overall `purchase cost`.
*   **Enhanced Therapist Interaction:** Consider allowing `therapists` to `embody a person from the audience` and interact directly with the patient as an `interlocutor` within the `VR environment`.
*   **Increased Scenario Variety:** Develop a `library of pre-orchestrated scenes` with `randomized elements` to reduce predictability, especially for `home-based therapy`.
*   **Quantitative Efficacy Evaluation:** A major future step is a long-term, quantitative study with a large number of diagnosed `phobic patients` and `control groups` using diverse `therapists` to rigorously evaluate the system's `therapeutic effectiveness`.

## 7.3. Personal Insights & Critique
This paper presents a comprehensive and well-thought-out `VRET` system, `PhoVR`, that leverages a strong multidisciplinary approach. The collaboration between a technical university, a biology faculty, and an industry partner is commendable, as it ensures both technological sophistication and therapeutic relevance.

One of the most inspiring aspects is the integration of **environmental biofeedback**. Changing the `sky color` or `cloud density` based on a patient's `HR` and `EDA` is an intuitive and immersive way to provide `biofeedback` without breaking `presence` by showing numerical displays. This could significantly enhance a patient's ability to self-regulate. The emphasis on `gamified tasks` is also a critical design choice, as it addresses a common challenge in therapy: patient engagement and motivation. Making therapy feel less like a chore and more like an interactive experience can improve adherence and outcomes.

However, some potential issues and areas for improvement, beyond what the authors already noted, could be considered:
*   **Individual Variability in VR Sickness:** While the authors optimized for `FPS`, `motion sickness` remains a significant challenge for some `VR` users. More research into individual susceptibility and personalized mitigation strategies (e.g., dynamic field of view adjustments, specific training protocols) might be beneficial.
*   **Ethical Considerations of Autonomous Adaptation:** The system's ability to `automatically classify anxiety` and `dynamically adapt environments` is powerful but also raises ethical questions. While `psychotherapists` oversee the process, the degree of autonomy and the potential for unintended `stress induction` or `misinterpretation` of `biometric data` require careful monitoring and robust safety protocols. The request from `psychotherapists` for more `therapist control` over `automatic adaptation` highlights this concern.
*   **Long-Term Efficacy and Generalization:** The current paper reports qualitative validation and physiological responses in non-phobic subjects. The true test of `PhoVR's` value will come from its `quantitative evaluation` in long-term clinical trials with diagnosed phobic patients. It will be crucial to demonstrate that the learned coping mechanisms `generalize` effectively to `real-world situations` and are maintained over time.
*   **Accessibility for Diverse Populations:** While affordable `all-in-one HMDs` improve accessibility, considerations for users with physical disabilities (beyond `controller-based locomotion`), cognitive impairments, or cultural sensitivities in `virtual environments` would be valuable for wider application.

    The methods and conclusions of this paper could be transferred to other domains requiring controlled `exposure` or `stress management`, such as `training for high-stress professions` (e.g., firefighters, pilots), `rehabilitation for PTSD`, or even `educational settings` to manage `performance anxiety`. The multi-sensor approach and adaptive `VR environment` could also be adapted for `pain management` or `relaxation therapies`. The `control panel's` design, with its focus on `therapist management` and `data visualization`, sets a high standard for clinical `VR` applications.