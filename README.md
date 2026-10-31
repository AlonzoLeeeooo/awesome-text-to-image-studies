<p align="center">
  <h1 align="center">A Collection of Text-to-Image Generation Studies</h1>

This GitHub repository summarizes papers and resources related to the text-to-image (T2I) generation task. 

> [!NOTE]
> This document serves as the `homepage` of the whole GitHub repo. Papers are summarized according to **different research directions, published years, and conferences.** 
> 
> [The `topics` section](topics/topics.md) summarizes papers that are highly related to T2I generation according to different properties, e.g., prerequisites of T2I generation, diffusion models with other techniques (e.g., Diffusion Transformer, LLMs, Mamba, etc.), and diffusion models for other tasks. 

If you have any suggestions about this reåpository, please feel free to [start a new issue](https://github.com/AlonzoLeeeooo/awesome-text-to-image-generation-studies/issues/new) or [pull requests](https://github.com/AlonzoLeeeooo/awesome-text-to-image-generation-studies/pulls).

Recent news of this GitHub repo are listed as follows.

🔥 [Dec. 11th, 2025] Our paper titled ["StableV2V: Stablizing Shape Consistency in Video-to-Video Editing"](https://ieeexplore.ieee.org/document/11272911) is accepted at TCSVT 2025!

🔥 [Nov. 19th] We have released our latest paper titled ["StableV2V: Stablizing Shape Consistency in Video-to-Video Editing"](https://arxiv.org/abs/2411.11045), with the correponding [code](https://github.com/AlonzoLeeeooo/StableV2V), [model weights](https://huggingface.co/AlonzoLeeeooo/StableV2V), and [a testing benchmark `DAVIS-Edit`](https://huggingface.co/datasets/AlonzoLeeeooo/DAVIS-Edit) open-sourced. Feel free to check them out from the links!
<details> <summary> Click to see more information. </summary>

- [Apr. 26th] Update a new topic: **Diffusion Models Meet Federated Learning.** See [the `topics` section](topics/topics.md) for more details!
- [Mar. 28th] The official **AAAI 2024** paper list are released! Official version of PDFs and BibTeX references are updated accordingly.
- [Mar. 21th] [The `topics` section](topics/topics.md) has been updated. This section aims to offer **paper lists that are summarized according to other properties of diffusion models**, e.g., Diffusion Transformer-based methods, diffusion models for NLP, diffusion models integrated with LLMs, etc. The corresponding references of these papers are also concluded in `reference.bib`.
- [Mar. 7th] All available **CVPR, ICLR, and AAAI 2024 papers and references** are updated.
- [Mar. 1st] Websites of [**the off-the-shelf text-to-image generation products**](#available-products) and [**toolkits**](#toolkits) are summarized.

</details>


<!-- omit in toc -->
# <span id="contents">Contents</span>
- [Products](#available-products)
- [To-Do Lists](#to-do-lists)
- [Papers](#papers)
  - [Survey Papers](#survey-papers)
  - [Text-to-Image Generation](#text-to-image-generation)
    - [Year 2025](#text-year-2025)
    - [Year 2024](#text-year-2024)
    - [Year 2023](#text-year-2023)
    - [Year 2022](#text-year-2022)
    - [Year 2021](#text-year-2021)
    - [Year 2020](#text-year-2020)
  - [Conditional Text-to-Image Generation](#conditional-text-to-image-generation)
    - [Year 2025](#conditional-year-2025)
    - [Year 2024](#conditional-year-2024)
    - [Year 2023](#conditional-year-2023)
    - [Year 2022](#conditional-year-2022)
  - [Personalized Text-to-Image Generation](#personalized-text-to-image-generation)
    - [Year 2025](#personalized-year-2025)
    - [Year 2024](#personalized-year-2024)
    - [Year 2023](#personalized-year-2023)
  - [Text-Guided Image Editing](#text-guided-image-editing) 
    - [Year 2025](#editing-year-2025)
    - [Year 2024](#editing-year-2024)
    - [Year 2023](#editing-year-2023)
    - [Year 2022](#editing-year-2022)
  - [Text Image Generation](#text-image-generation)
    - [Year 2024](#gentext-year-2024)
- [Datasets](#datasets)
- [Toolkits](#toolkits)
- [Q&A](#qa)
- [References](#references)
- [Star History](#star-history)
- [WeChat Group](#wechat-group)

<!-- omit in toc -->
# To-Do Lists
- Published Papers on Conferences
  - [ ] Update NeurIPS 2025 Papers
  - [x] Update ICCV 2025 Papers
  - [x] Update CVPR 2025 Papers
  - [x] Update ICLR 2025 Papers
  - [x] Update NeurIPS 2024 Papers
  - [x] Update ECCV 2024 Papers
  - [x] Update CVPR 2024 Papers
    - [x] Update ⚠️ Papers and References
    - [ ] Update arXiv References into the Official Version
  - [x] Update AAAI 2024 Papers
    - [x] Update ⚠️ Papers and References
    - [x] Update arXiv References into the Official Version
  - [x] Update ICLR 2024 Papers
  - [x] Update NeurIPS 2023 Papers
- Regular Maintenance of Preprint arXiv Papers and Missed Papers

[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
# Products

|Name|Year|Website|Specialties|
|-|-|-|-|
|AI Art Generator|2025|[link](https://aiart-generator.art/)|AI art generation platform with resolution presets, style transfer, upscaling, and video creation.|
|Nano Image Art|2025|[link](https://nanoimage.art/)|Create stunning AI images — powered by Google's Nano Banana Pro for next-gen quality, smart editing, and intelligent prompts.|
|Fast Image AI|2025|[link](https://fastimage.ai/)|Fast Image AI instantly transforms your photos into stunning styles like Ghibli, Sketch, and Pixar. Effortlessly control image elements and create amazing effects with just one click.|
|Gempix2 (Nano Banana 2)|2025|[link](https://gempix2.site)|Free AI image generation platform with text-to-image, AI editing, and video generation support|
|Raphael AI|-|[link](https://raphael.app)|Free unlimited AI image generator (text-to-image, no signup)|
|Stable Diffusion 3|2024|[link](https://stability.ai/news/stable-diffusion-3)|Diffusion Transformer-based Stable Diffusion|
|Stable Video|2024|[link](https://www.stablevideo.com/)|High-quality high-resolution images|
|DALL-E 3|2023|[link](https://openai.com/dall-e-3)|Collaborate with [ChatGPT](https://chat.openai.com/)|
|Ideogram|2023|[link](https://ideogram.ai/login)|Text images|
|Playground|2023|[link](https://playground.com/)|Athestic images|
|HiDream.ai|2023|[link](https://hidreamai.com/#/)|-|
|Dashtoon|2023|[link](https://dashtoon.com/)|Text-to-Comic Generation|
|WHEE|2023|[link](https://www.whee.com/)|WHEE is an online AI generation tool, which can be applied for *T2I generation, I2I generation, SR, inpainting, outpainting, image variation, virtural try-on, etc.* |
|Vega AI|2023|[link](https://www.vegaai.net/)|Vega AI is an online AI generation tool, which can be applied for *T2I generation, I2I generation, SR, T2V generation, I2V generation, etc.*|
|Wujie AI|2022|[link](https://www.wujieai.com/)|The Chinese name is "无界AI", offering AIGC resources and online services|
|Midjourney|2022|[link](https://www.midjourney.com/home)|Powerful close-sourced generation tool|

[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
# Papers

<!-- omit in toc -->
## Survey Papers
- **Text-to-Image Generation**
  - Year 2024
    - **ACM Computing Surveys** 
      - Diffusion Models: A Comprehensive Survey of Methods and Applications [[Paper]](https://arxiv.org/pdf/2209.00796.pdf)
  - Year 2023
    - **TPAMI**
      - Diffusion Models in Vision: A Survey [[Paper]](https://arxiv.org/pdf/2209.04747v2) [[Code]](https://github.com/CroitoruAlin/Diffusion-Models-in-Vision-A-Survey)
    - **arXiv**
      - Text-to-image Diffusion Models in Generative AI: A Survey [[Paper]](https://arxiv.org/pdf/2303.07909)
      - State of the Art on Diffusion Models for Visual Computing [[Paper]](https://arxiv.org/pdf/2310.07204.pdf) 
  - Year 2022
    - **arXiv**
      - Efficient Diffusion Models for Vision: A Survey [[Paper]](https://arxiv.org/pdf/2210.09292)
- **Conditional Text-to-Image Generation**
  - Year 2024
    - **arXiv**
      - Controllable Generation with Text-to-Image Diffusion Models: A Survey [[Paper]](https://arxiv.org/pdf/2403.04279)
- **Text-Guided Image Editing**
  - Year 2024
    - **arXiv**
      - Diffusion Model-Based Image Editing: A Survey [[Paper]](https://arxiv.org/pdf/2402.17525.pdf) [[Code]](https://github.com/SiatMMLab/Awesome-Diffusion-Model-Based-Image-Editing-Methods)

[<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
## Text-to-Image Generation
- <span id="text-year-2025">**Year 2025**</span>
  - **CVPR**
    - ***PreciseCam:*** Precise Camera Control for Text-to-Image Generation [[Paper]](https://arxiv.org/apdf/2501.12910) [[Project]](https://graphics.unizar.es/projects/PreciseCam2024/) [[Code]](https://github.com/edurnebernal/PreciseCam)
    - ***Type-R:*** Automatically Retouching Typos for Text-to-Image Generation [[Paper]](https://arxiv.org/abs/2411.18159) [[Code]](https://github.com/CyberAgentAILab/Type-R)
    - ***Compass Control:*** Multi Object Orientation Control for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2504.06752)
    - ***Generative Photography:*** Scene-Consistent Camera Control for Realistic Text-to-Image Synthesis [[Paper]](https://arxiv.org/pdf/2412.02168) [[Project]](https://generative-photography.github.io/project/) [[Code]](https://github.com/pandayuanyu/generative-photography)
    -  ***One-Way Ticket:*** Time-Independent Unified Encoder for Distilling Text-to-Image Diffusion Models [[Paper]](https://cvpr.thecvf.com/virtual/2025/poster/32579) [[Code]](https://github.com/sen-mao/Loopfree)
    -  ***Build a Text-to-Image Generator (from Scratch):*** Build your own vision transformer and diffusion models for text-to-image generation–from scratch! [[Paid book]](https://www.manning.com/books/build-a-text-to-image-generator-from-scratch) 
    - Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps [[Paper]](https://arxiv.org/pdf/2411.15236)
    - Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2412.03178)
    - Plug-and-Play Interpretable Responsible Text-to-Image Generation via Dual-Space Multi-facet Concept Control [[Paper]](https://arxiv.org/pdf/2503.18324) [[Project]](https://basim-azam.github.io/responsiblediffusion/) [[Code]](https://github.com/basim-azam/responsiblediffusion)
    - Make It Count: Text-to-Image Generation with an Accurate Number of Objects [[Paper]](https://arxiv.org/pdf/2406.10210) [[Project]](https://make-it-count-paper.github.io/) [[Code]](https://github.com/Litalby1/make-it-count)
    - ***MCCD:*** Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2505.02648)
    - Rethinking Training for De-biasing Text-to-Image Generation: Unlocking the Potential of Stable Diffusion [[Paper]](https://arxiv.org/pdf/2408.12692)
    - ***ShapeWords:*** Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts [[Paper]](https://arxiv.org/pdf/2412.02912) [[Project]](https://lodurality.github.io/shapewords/)
    - ***SnapGen:*** Taming High-Resolution Text-to-Image Models for Mobile Devices with Efficient Architectures and Training [[Paper]](https://arxiv.org/pdf/2412.09619)
    - Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis [[Paper]](https://arxiv.org/pdf/2503.22168)
    - ***Focus-N-Fix:*** Region-Aware Fine-Tuning for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2501.06481)
    - ***SILMM:*** Self-Improving Large Multimodal Models for Compositional Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2412.05818)
    - Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation [[Paper]](https://arxiv.org/pdf/2503.12356)
    - Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects [[Paper]](https://arxiv.org/pdf/2411.18936)
    - Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis [[Paper]](https://arxiv.org/pdf/2411.16503) [[Code]](https://github.com/Bomingmiao/NoiseDiffusion)
    - Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2410.07838)
    - ***STEREO:*** A Two-Stage Framework for Adversarially Robust Concept Erasing from Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2408.16807)
    - Minority-Focused Text-to-Image Generation via Prompt Optimization [[Paper]](https://arxiv.org/pdf/2411.16503) [[Code]](https://github.com/soobin-um/MinorityPrompt)
    - Scaling Down Text Encoders of Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2503.19897) [[Code]](https://github.com/LifuWang-66/DistillT5)
    - ***Janus:*** Decoupling Visual Encoding for Unified Multimodal Understanding and Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Janus_Decoupling_Visual_Encoding_for_Unified_Multimodal_Understanding_and_Generation_CVPR_2025_paper.pdf)
    - Enhancing Creative Generation on Stable Diffusion-based Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Enhancing_Creative_Generation_on_Stable_Diffusion-based_Models_CVPR_2025_paper.pdf)
    - Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_Aesthetic_Post-Training_Diffusion_Models_from_Generic_Preferences_with_Step-by-step_Preference_CVPR_2025_paper.pdf)
    - ***Stretching Each Dollar:*** Diffusion Training from Scratch on a Micro-Budget [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sehwag_Stretching_Each_Dollar_Diffusion_Training_from_Scratch_on_a_Micro-Budget_CVPR_2025_paper.pdf)
    - ***Diffusion-4K:*** Ultra-High-Resolution Image Synthesis with Latent Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Diffusion-4K_Ultra-High-Resolution_Image_Synthesis_with_Latent_Diffusion_Models_CVPR_2025_paper.pdf)
    - Precise, Fast, and Low-cost Concept Erasure in Value Space: Orthogonal Complement Matters [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Precise_Fast_and_Low-cost_Concept_Erasure_in_Value_Space__CVPR_2025_paper.pdf)
    - ***T2ISafety:*** Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_T2ISafety_Benchmark_for_Assessing_Fairness_Toxicity_and_Privacy_in_Image_CVPR_2025_paper.pdf)
    - ***ChatGen:*** Automatic Text-to-Image Generation From FreeStyle Chatting [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_ChatGen_Automatic_Text-to-Image_Generation_From_FreeStyle_Chatting_CVPR_2025_paper.pdf)
    - Dual Diffusion for Unified Image Generation and Understanding [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Dual_Diffusion_for_Unified_Image_Generation_and_Understanding_CVPR_2025_paper.pdf)
    - ***DyMO:*** Training-Free Diffusion Model Alignment with Dynamic Multi-Objective Scheduling [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_DyMO_Training-Free_Diffusion_Model_Alignment_with_Dynamic_Multi-Objective_Scheduling_CVPR_2025_paper.pdf)
    - Erasing Undesirable Influence in Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Erasing_Undesirable_Influence_in_Diffusion_Models_CVPR_2025_paper.pdf)
    - Redefining &lt;Creative&gt; in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_Redefining_Creative_in_Dictionary_Towards_an_Enhanced_Semantic_Understanding_of_CVPR_2025_paper.pdf)
    - ***Six-CD:*** Benchmarking Concept Removals for Text-to-image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_Six-CD_Benchmarking_Concept_Removals_for_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf)
    - ***NitroFusion:*** High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_NitroFusion_High-Fidelity_Single-Step_Diffusion_through_Dynamic_Adversarial_Training_CVPR_2025_paper.pdf)
    - ***ReNeg:*** Learning Negative Embedding with Reward Guidance [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_ReNeg_Learning_Negative_Embedding_with_Reward_Guidance_CVPR_2025_paper.pdf)
    - ***UniReal:*** Universal Image Generation and Editing via Learning Real-world Dynamics [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_UniReal_Universal_Image_Generation_and_Editing_via_Learning_Real-world_Dynamics_CVPR_2025_paper.pdf)
    - ***OmniFlow:*** Any-to-Any Generation with Multi-Modal Rectified Flows [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_OmniFlow_Any-to-Any_Generation_with_Multi-Modal_Rectified_Flows_CVPR_2025_paper.pdf)
    - ***FAM Diffusion:*** Frequency and Attention Modulation for High-Resolution Image Generation with Stable Diffusion [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_FAM_Diffusion_Frequency_and_Attention_Modulation_for_High-Resolution_Image_Generation_CVPR_2025_paper.pdf)
    - ***DreamOmni:*** Unified Image Generation and Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xia_DreamOmni_Unified_Image_Generation_and_Editing_CVPR_2025_paper.pdf)
    - Curriculum Direct Preference Optimization for Diffusion and Consistency Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Croitoru_Curriculum_Direct_Preference_Optimization_for_Diffusion_and_Consistency_Models_CVPR_2025_paper.pdf)
    - Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_Dissecting_and_Mitigating_Diffusion_Bias_via_Mechanistic_Interpretability_CVPR_2025_paper.pdf)
    - ***ACE:*** Anti-Editing Concept Erasure in Text-to-Image Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_ACE_Anti-Editing_Concept_Erasure_in_Text-to-Image_Models_CVPR_2025_paper.pdf)
    - One Diffusion to Generate Them All [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Le_One_Diffusion_to_Generate_Them_All_CVPR_2025_paper.pdf)
    - Let's Verify and Reinforce Image Generation Step by Step [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Lets_Verify_and_Reinforce_Image_Generation_Step_by_Step_CVPR_2025_paper.pdf)
    - ***JanusFlow:*** Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_JanusFlow_Harmonizing_Autoregression_and_Rectified_Flow_for_Unified_Multimodal_Understanding_CVPR_2025_paper.pdf)
    - ***Schedule On the Fly:*** Diffusion Time Prediction for Faster and Better Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_Schedule_On_the_Fly_Diffusion_Time_Prediction_for_Faster_and_CVPR_2025_paper.pdf)
    - Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Thakral_Fine-Grained_Erasure_in_Text-to-Image_Diffusion-based_Foundation_Models_CVPR_2025_paper.pdf)
    - A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_A_Comprehensive_Study_of_Decoder-Only_LLMs_for_Text-to-Image_Generation_CVPR_2025_paper.pdf)
    - Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Exploring_Sparse_MoE_in_GANs_for_Text-conditioned_Image_Synthesis_CVPR_2025_paper.pdf)
    - ***WeGen:*** A Unified Model for Interactive Multimodal Generation as We Chat [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_WeGen_A_Unified_Model_for_Interactive_Multimodal_Generation_as_We_CVPR_2025_paper.pdf)
    - ***VerbDiff:*** Text-Only Diffusion Models with Enhanced Interaction Awareness [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Cha_VerbDiff_Text-Only_Diffusion_Models_with_Enhanced_Interaction_Awareness_CVPR_2025_paper.pdf)
    - Memories of Forgotten Concepts [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Rusanovsky_Memories_of_Forgotten_Concepts_CVPR_2025_paper.pdf)
    - ***PQPP:*** A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Poesina_PQPP_A_Joint_Benchmark_for_Text-to-Image_Prompt_and_Query_Performance_CVPR_2025_paper.pdf)
    - Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Reward_Fine-Tuning_Two-Step_Diffusion_Models_via_Learning_Differentiable_Latent-Space_Surrogate_CVPR_2025_paper.pdf)
    - Calibrated Multi-Preference Optimization for Aligning Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Calibrated_Multi-Preference_Optimization_for_Aligning_Diffusion_Models_CVPR_2025_paper.pdf)
    - ***InPO:*** Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lu_InPO_Inversion_Preference_Optimization_with_Reparametrized_DDIM_for_Efficient_Diffusion_CVPR_2025_paper.pdf)
    - Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Enhancing_Privacy-Utility_Trade-offs_to_Mitigate_Memorization_in_Diffusion_Models_CVPR_2025_paper.pdf)
    - Spherical Manifold Guided Diffusion Model for Panoramic Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Spherical_Manifold_Guided_Diffusion_Model_for_Panoramic_Image_Generation_CVPR_2025_paper.pdf)
    - Towards Transformer-Based Aligned Generation with Self-Coherence Guidance [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Towards_Transformer-Based_Aligned_Generation_with_Self-Coherence_Guidance_CVPR_2025_paper.pdf)
    - ***Infinity:*** Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Infinity_Scaling_Bitwise_AutoRegressive_Modeling_for_High-Resolution_Image_Synthesis_CVPR_2025_paper.pdf)
    - ***Science-T2I:*** Addressing Scientific Illusions in Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Science-T2I_Addressing_Scientific_Illusions_in_Image_Synthesis_CVPR_2025_paper.pdf)
    - Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf)
    - ⚠️ The Illusion of Unlearning: The Unstable Nature of Machine Unlearning in Text-to-Image Diffusion Models [Paper]
    - ⚠️ Exploring the Deep Fusion of Large Language Models and Diffusion Transformers for Text-to-Image Synthesis [Paper]
    - ⚠️ Detect-and-Guide: Self-regulation of Diffusion Models for Safe Text-to-Image Generation via Guideline Token Optimization [Paper]
    - ⚠️ Multi-Group Proportional Representations for Text-to-Image Models [Paper]
    - ⚠️ ***VODiff:*** Controlling Object Visibility Order in Text-to-Image Generation [Paper]
  - **ICCV**
    - ***SANA-Sprint:*** One-Step Diffusion with Continuous-Time Consistency Distillation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_SANA-Sprint_One-Step_Diffusion_with_Continuous-Time_Consistency_Distillation_ICCV_2025_paper.pdf)
    - Erasing More Than Intended? How Concept Erasure Degrades the Generation of Non-Target Concepts [[Paper]](https://arxiv.org/abs/2501.09833)
    - Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis [[Paper]](https://arxiv.org/abs/2507.18569)
    - Golden Noise for Diffusion Models: A Learning Framework [[Paper]](https://arxiv.org/abs/2411.09502)
    - What Makes for Text to 360-degree Panorama Generation with Stable Diffusion? [[Paper]](https://arxiv.org/abs/2505.22129)
    - ***EDiT:*** Efficient Diffusion Transformers with Linear Compressed Attention [[Paper]](https://arxiv.org/abs/2503.16726)
    - ***TRCE:*** Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/abs/2503.07389)
    - Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment [[Paper]](https://arxiv.org/abs/2507.19002)
    - Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences [[Paper]](https://arxiv.org/abs/2506.02095)
    - ***CAP:*** Evaluation of Persuasive and Creative Image Generation [[Paper]](https://arxiv.org/abs/2412.10426)
    - ***FairGen:*** Enhancing Fairness in Text-to-Image Diffusion Models via Self-Discovering Latent Directions [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_FairGen_Enhancing_Fairness_in_Text-to-Image_Diffusion_Models_via_Self-Discovering_Latent_ICCV_2025_paper.pdf)
    - Trade-offs in Image Generation: How Do Different Dimensions Interact? [[Paper]](https://arxiv.org/abs/2507.22100)
    - ***TCFG:*** Truncated Classifier-Free Guidance for Efficient and Scalable Text-to-Image Acceleration [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Fu_TCFG_Truncated_Classifier-Free_Guidance_for_Efficient_and_Scalable_Text-to-Image_Acceleration_ICCV_2025_paper.pdf)
    - ***Dense2MoE:*** Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation [[Paper]](https://arxiv.org/abs/2510.09094)
    - ***Holistic Unlearning Benchmark:*** A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning [[Paper]](https://arxiv.org/abs/2410.05664)
    - Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention [[Paper]](https://arxiv.org/abs/2506.13298)
    - ***DIMCIM:*** A Quantitative Evaluation Framework for Default-mode Diversity and Generalization in Text-to-Image Generative Models [[Paper]](https://arxiv.org/abs/2506.05108)
    - Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers [[Paper]](https://arxiv.org/abs/2506.07986)
    - ***UniVG:*** A Generalist Diffusion Model for Unified Image Generation and Editing [[Paper]](https://arxiv.org/abs/2503.12652)
    - ***FlowTok:*** Flowing Seamlessly Across Text and Image Tokens [[Paper]](https://arxiv.org/abs/2503.10772)
    - From Reflection to Perfection: Scaling Inference-Time Optimization for Text-to-Image Diffusion Models via Reflection Tuning [[Paper]](https://arxiv.org/abs/2504.16080)
    - ***CuRe:*** Cultural Gaps in the Long Tail of Text-to-Image Systems [[Paper]](https://arxiv.org/abs/2506.08071)
    - Penalizing Boundary Activation for Object Completeness in Diffusion Models [[Paper]](https://arxiv.org/abs/2509.16968)
    - Decoding Correlation-Induced Misalignment in the Stable Diffusion Workflow for Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Tong_Decoding_Correlation-Induced_Misalignment_in_the_Stable_Diffusion_Workflow_for_Text-to-Image_ICCV_2025_paper.pdf)
    - Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model [[Paper]](https://arxiv.org/abs/2411.13787)
    - ***SuMa:*** A Subspace Mapping Approach for Robust and Effective Concept Erasure in Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/abs/2509.05625)
    - Leveraging Panoptic Scene Graph for Evaluating Fine-Grained Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Deng_Leveraging_Panoptic_Scene_Graph_for_Evaluating_Fine-Grained_Text-to-Image_Generation_ICCV_2025_paper.pdf)
    - Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts [[Paper]](https://arxiv.org/abs/2410.12777)
    - ***Sculpting Memory:*** Multi-Concept Forgetting in Diffusion Models via Dynamic Mask and Concept-Aware Optimization [[Paper]](https://arxiv.org/abs/2504.09039)
    - LLM Thought Divergence and Convergence for Dialogue-Based Image Generation Control [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_LLM_Thought_Divergence_and_Convergence_for_Dialogue-Based_Image_Generation_Control_ICCV_2025_paper.pdf)
    - ***Imbalance in Balance:*** Online Concept Balancing in Generation Models [[Paper]](https://arxiv.org/abs/2507.13345)
    - Multimodal LLMs as Customized Reward Models for Text-to-Image Generation [[Paper]](https://arxiv.org/abs/2507.21391)
    - The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation [[Paper]](https://arxiv.org/abs/2412.05101)
    - ***ShortFT:*** Diffusion Model Alignment via Shortcut-based Fine-Tuning [[Paper]](https://arxiv.org/abs/2507.22604)
    - Rethinking DPO-style Diffusion Aligning Frameworks [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_Rethinking_DPO-style_Diffusion_Aligning_Frameworks_ICCV_2025_paper.pdf)
    - ***VSC:*** Visual Search Compositional Text-to-Image Diffusion Model [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Dat_VSC_Visual_Search_Compositional_Text-to-Image_Diffusion_Model_ICCV_2025_paper.pdf)
    - Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models [[Paper]](https://arxiv.org/abs/2501.08727)
    - ***InfGen:*** A Resolution-Agnostic Paradigm for Scalable Image Synthesis [[Paper]](https://arxiv.org/abs/2509.10441)
    - ***FreeScale:*** Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion [[Paper]](https://arxiv.org/abs/2412.09626)
    - Text Embedding Knows How to Quantize Text-Guided Diffusion Models [[Paper]](https://arxiv.org/abs/2507.10340)
    - ***HPSv3:*** Towards Wide-Spectrum Human Preference Score [[Paper]](https://arxiv.org/abs/2508.03789)
    - ***DC-AR:*** Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_DC-AR_Efficient_Masked_Autoregressive_Image_Generation_with_Deep_Compression_Hybrid_ICCV_2025_paper.pdf)
    - ***DiTFastAttnV2:*** Head-wise Attention Compression for Multi-Modality Diffusion Transformers [[Paper]](https://arxiv.org/abs/2503.22796)
    - ***IQA-Adapter:*** Exploring Knowledge Transfer from Image Quality Assessment to Diffusion-based Generative Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Abud_IQA-Adapter_Exploring_Knowledge_Transfer_from_Image_Quality_Assessment_to_Diffusion-based_ICCV_2025_paper.pdf)
    - Guiding Diffusion Models with Adaptive Negative Sampling Without External Resources [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Desai_Guiding_Diffusion_Models_with_Adaptive_Negative_Sampling_Without_External_Resources_ICCV_2025_paper.pdf)
    - ***AlignGuard:*** Scalable Safety Alignment for Text-to-Image Generation [[Paper]](https://arxiv.org/abs/2412.10493)
    - ***ICE-Bench:*** A Unified and Comprehensive Benchmark for Image Creating and Editing [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Pan_ICE-Bench_A_Unified_and_Comprehensive_Benchmark_for_Image_Creating_and_ICCV_2025_paper.pdf)
    - ***X2I:*** Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation [[Paper]](https://arxiv.org/abs/2503.06134)
    - ***T2I-Copilot:*** A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_T2I-Copilot_A_Training-Free_Multi-Agent_Text-to-Image_System_for_Enhanced_Prompt_Interpretation_ICCV_2025_paper.pdf)
    - ***Lumina-Image 2.0:*** A Unified and Efficient Image Generative Framework [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Qin_Lumina-Image_2.0_A_Unified_and_Efficient_Image_Generative_Framework_ICCV_2025_paper.pdf)
    - ***Scendi Score:*** Prompt-Aware Diversity Evaluation via Schur Complement of CLIP Embeddings [[Paper]](https://arxiv.org/abs/2412.18645)
    - ***PUMA:*** Empowering Unified MLLM with Multi-granular Visual Generation [[Paper]](https://arxiv.org/abs/2410.13861)
    - Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens [[Paper]](https://arxiv.org/abs/2501.07730)
    - ***Reflect-DiT:*** Inference-Time Scaling for Text-to-Image Diffusion Transformers via In-Context Reflection [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Reflect-DiT_Inference-Time_Scaling_for_Text-to-Image_Diffusion_Transformers_via_In-Context_Reflection_ICCV_2025_paper.pdf)
    - Supercharged One-step Text-to-Image Diffusion Models with Negative Prompts [[Paper]](https://arxiv.org/abs/2412.02687)
    - ***IMG:*** Calibrating Diffusion Models via Implicit Multimodal Guidance [[Paper]](https://arxiv.org/abs/2509.26231)
    - ***FairHuman:*** Boosting Hand and Face Quality in Human Image Generation with Minimum Potential Delay Fairness in Diffusion Models [[Paper]](https://arxiv.org/abs/2507.02714)
    - Importance-Based Token Merging for Efficient Image and Video Generation [[Paper]](https://arxiv.org/abs/2411.16720)
    - ***Scale Your Instructions:*** Enhance the Instruction-Following Fidelity of Unified Image Generation Model by Self-Adaptive Attention Scaling [[Paper]](https://arxiv.org/abs/2507.16240)
    - Translation of Text Embedding via Delta Vector to Suppress Strongly Entangled Content in Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/abs/2508.10407)
    - Dual-Process Image Generation [[Paper]](https://arxiv.org/abs/2506.01955)
    - ***LMM4LMM:*** Benchmarking and Evaluating Large-multimodal Image Generation with LMMs [[Paper]](https://arxiv.org/abs/2504.08358)
    - Reusing Computation in Text-to-Image Diffusion for Efficient Generation of Image Sets [[Paper]](https://arxiv.org/abs/2508.21032)
    - Scalable Ranked Preference Optimization for Text-to-Image Generation [[Paper]](https://arxiv.org/abs/2410.18013)
    - Frequency-Aware Autoregressive Modeling for Efficient High-Resolution Image Synthesis [[Paper]](https://arxiv.org/abs/2507.20454)
    - ***PLADIS:*** Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity [[Paper]](https://arxiv.org/abs/2503.07677)
  - **NeurIPS**
    - Rectified-CFG++ for Flow Based Models [[Paper]](https://arxiv.org/abs/2510.07631) [[Code]](https://github.com/shreshthsaini/Rectified-CFGpp)
  - **ICLR**
    - Improving Long-Text Alignment for Text-to-Image Diffusion Models [[Paper]](https://openreview.net/forum?id=2ZK8zyIt7o)
    - Information Theoretic Text-to-Image Alignment [[Paper]](https://openreview.net/forum?id=Ugs2W5XFFo)
    - ***Meissonic:*** Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis [[Paper]](https://openreview.net/forum?id=GJsuYHhAga)
    - ***PaRa:*** Personalizing Text-to-Image Diffusion via Parameter Rank Reduction [[Paper]](https://openreview.net/forum?id=KZgo2YQbhc)
    - ***Fluid:*** Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens [[Paper]](https://openreview.net/forum?id=jQP5o1VAVc)
    - Not All Prompts Are Made Equal: Prompt-based Pruning of Text-to-Image Diffusion Models [[Paper]](https://openreview.net/forum?id=3BhZCfJ73Y)
    - Denoising Autoregressive Transformers for Scalable Text-to-Image Generation [[Paper]](https://openreview.net/forum?id=amDkNPVWcn)
    - Progressive Compositionality in Text-to-Image Generative Models [[Paper]](https://openreview.net/forum?id=S85PP4xjFD)
    - Mining your own secrets: Diffusion Classifier Scores for Continual Personalization of Text-to-Image Diffusion Models [[Paper]](https://openreview.net/forum?id=hUdLs6TqZL)
    - Measuring And Improving Engagement of Text-to-Image Generation Models [[Paper]](https://openreview.net/forum?id=TmCcNuo03f)
    - Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate [[Paper]](https://openreview.net/forum?id=ZRDhBwKs7l)
    - Enhancing Compositional Text-to-Image Generation with Reliable Random Seeds [[Paper]](https://openreview.net/forum?id=5BSlakturs)
    - ***One-Prompt-One-Story:*** Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt [[Paper]](https://openreview.net/forum?id=cD1kl2QKv1)
    - You Only Sample Once: Taming One-Step Text-to-Image Synthesis by Self-Cooperative Diffusion GANs [[Paper]](https://openreview.net/forum?id=T7bmHkwzS6)
    - Rethinking Artistic Copyright Infringements In the Era Of Text-to-Image Generative Models [[Paper]](https://openreview.net/forum?id=0OTVNEm9N4)
    - Erasing Concept Combination from Text-to-Image Diffusion Model [[Paper]](https://openreview.net/forum?id=OBjF5I4PWg)
    - Cross-Attention Head Position Patterns Can Align with Human Visual Concepts in Text-to-Image Generative Models [[Paper]](https://openreview.net/forum?id=1vggIT5vvj)
    - ***TIGeR:*** Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models [[Paper]](https://openreview.net/forum?id=mr2icR6dpD)
    - ***DGQ:*** Distribution-Aware Group Quantization for Text-to-Image Diffusion Models [[Paper]](https://openreview.net/forum?id=ZyNEr7Xw5L)
    - Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding [[Paper]](https://openreview.net/forum?id=LZfjxvqw0N)
    - ***PT-T2I/V:*** An Efficient Proxy-Tokenized Diffusion Transformer for Text-to-Image/Video-Task [[Paper]](https://openreview.net/forum?id=lTrrnNdkOX)
    - Revisiting text-to-image evaluation with Gecko: on metrics, prompts, and human rating [[Paper]](https://openreview.net/forum?id=Im2neAMlre)
    - ***SANA:*** Efficient High-Resolution Text-to-Image Synthesis with Linear Diffusion Transformers [[Paper]](https://openreview.net/forum?id=N8Oj1XhtYZ)
    - Text-to-Image Rectified Flow as Plug-and-Play Priors [[Paper]](https://openreview.net/forum?id=SzPZK856iI)
    - Automated Filtering of Human Feedback Data for Aligning Text-to-Image Diffusion Models [[Paper]](https://openreview.net/forum?id=8jvVNPHtVJ)
    - ***SAFREE:*** Training-Free and Adaptive Guard for Safe Text-to-Image And Video Generation [[Paper]](https://openreview.net/forum?id=hgTFotBRKl)
    - ***IterComp:*** Iterative Composition-Aware Feedback Learning from Model Gallery for Text-to-Image Generation [[Paper]](https://openreview.net/forum?id=4w99NAikOE)
    - ***ScImage:*** How good are multimodal large language models at scientific text-to-image generation? [[Paper]](https://openreview.net/forum?id=ugyqNEOjoU)
    - Guided Score identity Distillation for Data-Free One-Step Text-to-Image Generation [[Paper]](https://openreview.net/forum?id=HMVDiaWMwM)
    - Evaluating Semantic Variation in Text-to-Image Synthesis: A Causal Perspective [[Paper]](https://openreview.net/forum?id=NWb128pSCb)
    - Test-time Alignment of Diffusion Models without Reward Over-optimization [[Paper]](https://openreview.net/forum?id=vi3DjUhFVm)
    - ***JetFormer:*** An autoregressive generative model of raw images and text [[Paper]](https://openreview.net/forum?id=sgAp2qG86e)
    - ***Lumina-T2X:*** Scalable Flow-based Large Diffusion Transformer for Flexible Resolution Generation [[Paper]](https://openreview.net/forum?id=EbWf36quzd)
    - ***Visually Guided Decoding:*** Gradient-Free Hard Prompt Inversion with Language Models [[Paper]](https://openreview.net/forum?id=mQ55y4s5hj)
    - ***VILA-U:*** a Unified Foundation Model Integrating Visual Understanding and Generation [[Paper]](https://openreview.net/forum?id=02haSpO453)
    - ***HART:*** Efficient Visual Generation with Hybrid Autoregressive Transformer [[Paper]](https://openreview.net/forum?id=q5sOv4xQe4)
    - ***A-Bench:*** Are LMMs Masters at Evaluating AI-generated Images? [[Paper]](https://openreview.net/forum?id=4muXQ5r8Ol)
    - Halton Scheduler for Masked Generative Image Transformer [[Paper]](https://openreview.net/forum?id=RDVrlWAb7K)
    - ***EC-DIT:*** Scaling Diffusion Transformers with Adaptive Expert-Choice Routing [[Paper]](https://openreview.net/forum?id=PxlfzEePC0)
    - Fantastic Copyrighted Beasts and How (Not) to Generate Them [[Paper]](https://openreview.net/forum?id=ftHNJmogT1)
    - ***Score Forgetting Distillation:*** A Swift, Data-Free Method for Machine Unlearning in Diffusion Models [[Paper]](https://openreview.net/forum?id=gjwhDHeAsz)
    - Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them [[Paper]](https://openreview.net/forum?id=tZdqL5FH7w)
    - ***ViDiT-Q:*** Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation [[Paper]](https://openreview.net/forum?id=E1N1oxd63b)
    - Training-Free Diffusion Model Alignment with Sampling Demons [[Paper]](https://openreview.net/forum?id=tfemquulED)
    - ***OASIS Uncovers:*** High-Quality T2I Models, Same Old Stereotypes [[Paper]](https://openreview.net/forum?id=L6IgkJvcgV)
    - Accelerating Diffusion Transformers with Token-wise Feature Caching [[Paper]](https://openreview.net/forum?id=yYZbZGo4ei)
    - ***SVDQuant:*** Absorbing Outliers by Low-Rank Component for 4-Bit Diffusion Models [[Paper]](https://openreview.net/forum?id=vWR3KuiQur)
    - ***Rare-to-Frequent:*** Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance [[Paper]](https://openreview.net/forum?id=BgxsmpVoOX)
    - ***FreCaS:*** Efficient Higher-Resolution Image Generation via Frequency-aware Cascaded Sampling [[Paper]](https://openreview.net/forum?id=TsBDfe8Ra5)
    - ***HERO:*** Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning [[Paper]](https://openreview.net/forum?id=yMHe9SRvxk)
    - ***Show-o:*** One Single Transformer to Unify Multimodal Understanding and Generation [[Paper]](https://openreview.net/forum?id=o6Ynz6OIQ6)
    - Efficient Diversity-Preserving Diffusion Alignment via Gradient-Informed GFlowNets [[Paper]](https://openreview.net/forum?id=Aye5wL6TCn)
    - Exploring Local Memorization in Diffusion Models via Bright Ending Attention [[Paper]](https://openreview.net/forum?id=p4cLtzk4oe)
    - ***Zigzag Diffusion Sampling:*** Diffusion Models Can Self-Improve via Self-Reflection [[Paper]](https://openreview.net/forum?id=MKvQH1ekeY)
    - Growth Inhibitors for Suppressing Inappropriate Image Concepts in Diffusion Models [[Paper]](https://openreview.net/forum?id=w4C4z80w59)
    - The Crystal Ball Hypothesis in diffusion models: Anticipating object positions from initial noise [[Paper]](https://openreview.net/forum?id=GpdO9r73xT)
    - ***DSPO:*** Direct Score Preference Optimization for Diffusion Model Alignment [[Paper]](https://openreview.net/forum?id=xyfb9HHvMe)
    - Long-horizon Visual Instruction Generation with Logic and Attribute Self-reflection [[Paper]](https://openreview.net/forum?id=EdMb9TqqDY)
    - Tuning Timestep-Distilled Diffusion Model Using Pairwise Sample Optimization [[Paper]](https://openreview.net/forum?id=fXnE4gB64o)
    - ***ConceptPrune:*** Concept Editing in Diffusion Models via Skilled Neuron Pruning [[Paper]](https://openreview.net/forum?id=kSdWcw5mkp)
  - **ICML**
    - ***SAeUron:*** Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/cywinski25a/cywinski25a.pdf) [[Code]](https://github.com/cywinski/SAeUron)
    - ***CHATS:*** Combining Human-Aligned Optimization and Test-Time Sampling for Text-to-Image Generation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/fu25g/fu25g.pdf) [[Code]](https://github.com/ATH-MaaS/CHATS)
    - ***EraseAnything:*** Enabling Concept Erasure in Rectified Flow Transformers [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/gao25j/gao25j.pdf) [[Code]](https://github.com/tomguluson92/eraseanything)
    - Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/hahn25a/hahn25a.pdf) [[Code]](https://github.com/google-deepmind/proactive_t2i_agents)
    - ***Adaptive Median Smoothing:*** Adversarial Defense for Unlearned Text-to-Image Diffusion Models at Inference Time [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/han25k/han25k.pdf)
    - ***ZipAR:*** Parallel Autoregressive Image Generation through Spatial Locality [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/he25b/he25b.pdf) [[Code]](https://github.com/ThisisBillhe/ZipAR)
    - ***D-Fusion:*** Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/hu25ab/hu25ab.pdf)
    - ***Angle Domain Guidance:*** Latent Diffusion Requires Rotation Rather Than Extrapolation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/jin25j/jin25j.pdf) [[Code]](https://github.com/jinc7461/ADG)
    - ***Shielded Diffusion:*** Generating Novel and Diverse Images using Sparse Repellency [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/kirchhof25a/kirchhof25a.pdf)
    - ***Orthus:*** Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/kou25a/kou25a.pdf)
    - ***One Image is Worth a Thousand Words:*** A Usability Preservable Text-Image Collaborative Erasing Framework [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25de/li25de.pdf) [[Code]](https://github.com/Ferry-Li/Co-Erasing)
    - Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/lu25l/lu25l.pdf) [[Code]](https://github.com/JaydenLyh/SmPO)
    - ***David and Goliath:*** Small One-step Model Beats Large Diffusion with Score Post-training [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/luo25s/luo25s.pdf) [[Code]](https://github.com/pkulwj1994/diff_instruct_star)
    - Preference Adaptive and Sequential Text-to-Image Generation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/nabati25a/nabati25a.pdf)
    - ***Concept Reachability in Diffusion Models:*** Beyond Dataset Constraints [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/rodriguez25a/rodriguez25a.pdf) [[Code]](https://github.com/martaaparod/concept_reachability)
    - ***Skrr:*** Skip and Re-use Text Encoder Layers for Memory Efficient Text-to-Image Generation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/seo25a/seo25a.pdf)
    - Inference-Time Alignment of Diffusion Models with Direct Noise Optimization [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/tang25h/tang25h.pdf) [[Code]](https://github.com/TZW1998/Direct-Noise-Optimization)
    - How to Evaluate and Mitigate IP Infringement in Visual Generative AI? [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25o/wang25o.pdf)
    - ***SANA 1.5:*** Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/xie25b/xie25b.pdf) [[Code]](https://github.com/NVlabs/Sana)
    - ***MuLan:*** Adapting Multilingual Diffusion Models for Hundreds of Languages with Negligible Cost [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/xing25d/xing25d.pdf) [[Code]](https://github.com/mulanai/MuLan)
    - Improving Compositional Generation with Diffusion Models Using Lift Scores [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/yu25e/yu25e.pdf) [[Project]](https://chenningyu.com/complift/)
    - Minimalist Concept Erasure in Generative Models [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhang25ai/zhang25ai.pdf)
  - **SIGGRAPH**
    - ***Be Decisive:*** Noise-Induced Layouts for Multi-Subject Generation [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3721238.3730631)
    - ***InstanceGen:*** Image Generation with Instance-level Instructions [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3721238.3730613)
  - **AAAI**
    - ***AGFSync:*** Leveraging AI-Generated Feedback for Preference Optimization in Text-to-Image Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32168)
    - ***Adaptive Guidance:*** Training-free Acceleration of Conditional Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32192)
    - Self-Corrected Flow Distillation for Consistent One-Step and Few-Step Image Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32269)
    - ***DuMo:*** Dual Encoder Modulation Network for Precise Concept Erasure [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32343)
    - ***DiffuseHigh:*** Training-Free Progressive High-Resolution Image Synthesis Through Structure Guidance [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32456)
    - ***SCott:*** Accelerating Diffusion Models with Stochastic Consistency Distillation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32580)
    - ***LLM4GEN:*** Leveraging Semantic Representation of LLMs for Text-to-Image Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32588)
    - ***Bridge Diffusion Model:*** Bridge Chinese Text-to-Image Diffusion Model with English Communities [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32590)
    - ***Storynizor:*** Consistent Story Generation via Inter-Frame Synchronized and Shuffled ID Injection [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32644)
    - ***Image Regeneration:*** Evaluating Text-to-Image Model via Generating Identical Image with Multimodal Large Language Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32651)
    - ***Qua2SeDiMo:*** Quantifiable Quantization Sensitivity of Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32658)
    - Boosting Consistency in Story Visualization with Rich-Contextual Conditional Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32728)
    - ***ResMaster:*** Mastering High-Resolution Image Generation via Structural and Fine-Grained Guidance [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32739)
    - Guided and Variance-Corrected Fusion with One-shot Style Alignment for Large-Content Image Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32764)
    - Unlearning Concepts in Diffusion Model via Concept Domain Correction and Concept Preserving Gradient [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32917)
    - ***FreqTS:*** Frequency-Aware Token Selection for Accelerating Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33008)
    - Training-Free and Hardware-Friendly Acceleration for Diffusion Models via Similarity-based Token Pruning [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33071)
    - ***Hand1000:*** Generating Realistic Hands from Text with Only 1,000 Images [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33074)
    - ***AdaDiff:*** Adaptive Step Selection for Fast Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33075)
    - ***StoryWeaver:*** A Unified World Model for Knowledge-Enhanced Story Character Customization [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33079)
    - ***ContextualStory:*** Consistent Visual Storytelling with Spatially-Enhanced and Storyline Context [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33153)
    - ***Flash Diffusion:*** Accelerating Any Conditional Diffusion Model for Few Steps Image Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33722)
    - ***DiffExp:*** Efficient Exploration in Reward Fine-tuning for Text-to-Image Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33723)
    - ***MARS:*** Mixture of Auto-Regressive Models for Fine-grained Text-to-image Synthesis [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33882)
    - Fair Text-to-Image Diffusion via Fair Mapping [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34823)
    - Evaluating Image Hallucination in Text-to-Image Generation with Question-Answering [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34827)
- <span id="text-year-2024">**Year 2024**</span>
  - **CVPR**
    - ***DistriFusion:*** Distributed Parallel Inference for High-Resolution Diffusion Models [[Paper]](https://arxiv.org/pdf/2402.19481.pdf) [[Code]](https://github.com/mit-han-lab/distrifuser)
    - ***InstanceDiffusion:*** Instance-level Control for Image Generation [[Paper]](https://arxiv.org/pdf/2402.03290.pdf) [[Code]](https://github.com/frank-xwang/InstanceDiffusion) [[Project]](https://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/)
    - ***ECLIPSE:*** A Resource-Efficient Text-to-Image Prior for Image Generations [[Paper]](https://arxiv.org/pdf/2312.04655.pdf) [[Code]](https://eclipse-t2i.vercel.app/) [[Project]](https://github.com/eclipse-t2i/eclipse-inference) [[Demo]](https://huggingface.co/spaces/ECLIPSE-Community/ECLIPSE-Kandinsky-v2.2)
    - ***Instruct-Imagen:*** Image Generation with Multi-modal Instruction [[Paper]](https://arxiv.org/pdf/2401.01952.pdf)
    - Learning Continuous 3D Words for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2402.08654.pdf) [[Code]](https://github.com/ttchengab/continuous_3d_words_code/)
    - ***HanDiffuser:*** Text-to-Image Generation With Realistic Hand Appearances [[Paper]](https://arxiv.org/pdf/2403.01693.pdf)
    - Rich Human Feedback for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2312.10240.pdf)
    - ***MarkovGen:*** Structured Prediction for Efficient Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2308.10997.pdf)
    - Customization Assistant for Text-to-image Generation [[Paper]](https://arxiv.org/pdf/2312.03045.pdf)
    - ***ADI:*** Learning Disentangled Identifiers for Action-Customized Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2311.15841.pdf) [[Project]](https://adi-t2i.github.io/ADI/)
    - ***UFOGen:*** You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs [[Paper]](https://arxiv.org/pdf/2311.09257.pdf)
    - Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2311.17216.pdf)
    - ***Tailored Visions:*** Enhancing Text-to-Image Generation with Personalized Prompt Rewriting [[Paper]](https://arxiv.org/pdf/2310.08129.pdf) [[Code]](https://github.com/zzjchen/Tailored-Visions)
    - ***CoDi:*** Conditional Diffusion Distillation for Higher-Fidelity and Faster Image Generation [[Paper]](https://arxiv.org/pdf/2310.01407.pdf) [[Code]](https://github.com/fast-codi/CoDi) [[Project]](https://fast-codi.github.io/) [[Demo]](https://huggingface.co/spaces/MKFMIKU/CoDi)
    - Arbitrary-Scale Image Generation and Upsampling using Latent Diffusion Model and Implicit Neural Decoder [[Paper]](https://arxiv.org/pdf/2403.10255.pdf)
    - Towards Effective Usage of Human-Centric Priors in Diffusion Models for Text-based Human Image Generation [[Paper]](https://arxiv.org/pdf/2403.05239)
    -  ***ElasticDiffusion:*** Training-free Arbitrary Size Image Generation [[Paper]](https://arxiv.org/pdf/2311.18822) [[Code]](https://github.com/MoayedHajiAli/ElasticDiffusion-official) [[Project]](https://elasticdiffusion.github.io/) [[Demo]](https://replicate.com/moayedhajiali/elasticdiffusion)
    - ***CosmicMan:*** A Text-to-Image Foundation Model for Humans [[Paper]](https://arxiv.org/pdf/2404.01294) [[Code]](https://github.com/cosmicman-cvpr2024/CosmicMan) [[Project]](https://cosmicman-cvpr2024.github.io/)
    - ***PanFusion:*** Taming Stable Diffusion for Text to 360° Panorama Image Generation [[Paper]](https://arxiv.org/pdf/2404.07949) [[Code]](https://github.com/chengzhag/PanFusion) [[Project]](https://chengzhag.github.io/publication/panfusion)
    - ***Intelligent Grimm:*** Open-ended Visual Storytelling via Latent Diffusion Models [[Paper]](https://arxiv.org/pdf/2306.00973) [[Code]](https://github.com/haoningwu3639/StoryGen) [[Project]](https://haoningwu3639.github.io/StoryGen_Webpage/)
    - On the Scalability of Diffusion-based Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2404.02883)
    - ***MuLAn:*** A Multi Layer Annotated Dataset for Controllable Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2404.02790) [[Project]](https://mulan-dataset.github.io/) [[Dataset]](https://huggingface.co/datasets/mulan-dataset/v1.0)
    - Learning Multi-dimensional Human Preference for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2405.14705)
    - Dynamic Prompt Optimizing for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2404.04095)
    - Training Diffusion Models Towards Diverse Image Generation with Reinforcement Learning [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Miao_Training_Diffusion_Models_Towards_Diverse_Image_Generation_with_Reinforcement_Learning_CVPR_2024_paper.pdf)
    - Adversarial Text to Continuous Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Haydarov_Adversarial_Text_to_Continuous_Image_Generation_CVPR_2024_paper.pdf) [[Project]](https://kilichbek.github.io/webpage/hypercgan/) [[Video]](https://kilichbek.github.io/webpage/hypercgan/#)
    - ***EmoGen:*** Emotional Image Content Generation with Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_EmoGen_Emotional_Image_Content_Generation_with_Text-to-Image_Diffusion_Models_CVPR_2024_paper.pdf) [[Code]](https://github.com/JingyuanYY/EmoGen)
  - **ECCV**
    - Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2403.07860) [[Code]](https://github.com/ShihaoZhaoZSH/LaVi-Bridge) [[Project]](https://shihaozhaozsh.github.io/LaVi-Bridge/)
    - Exploring Phrase-Level Grounding with Text-to-Image Diffusion Model [[Paper]](https://arxiv.org/pdf/2407.05352v1) [[Code]](https://github.com/nini0919/DiffPNG)
    - Getting it Right: Improving Spatial Consistency in Text-to-Image Models [[Paper]](https://arxiv.org/pdf/2404.01197) [[Code]](https://github.com/SPRIGHT-T2I/SPRIGHT) [[Project]](https://spright-t2i.github.io/)
    - Navigating Text-to-Image Generative Bias across Indic Languages [[Paper]](https://arxiv.org/pdf/2408.00283v1) [[Project]](https://iab-rubric.org/resources/other-databases/indictti)
    - Safeguard Text-to-Image Diffusion Models with Human Feedback Inversion [[Paper]](https://arxiv.org/pdf/2407.21032)
    - The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation [[Paper]](https://arxiv.org/pdf/2407.12579) [[Code]](https://leo81005.github.io/Reality-and-Fantasy/) [[Project]](https://leo81005.github.io/Reality-and-Fantasy/) [[Dataset]](https://leo81005.github.io/Reality-and-Fantasy/)
    - Reliable and Efficient Concept Erasure of Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2407.12383v1) [[Code]](https://github.com/CharlesGong12/RECE)
    - Exploring Phrase-Level Grounding with Text-to-Image Diffusion Model [[Paper]](https://arxiv.org/pdf/2407.05352v1) [[Code]](https://github.com/nini0919/DiffPNG)
    - ***StyleTokenizer:*** Defining Image Style by a Single Instance for Controlling Diffusion Models [[Paper]](https://arxiv.org/pdf/2409.02543) [[Code]](https://github.com/alipay/style-tokenizer)
    - ***PEA-Diffusion:*** Parameter-Efficient Adapter with Knowledge Distillation in non-English Text-to-Image Generation [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08492.pdf) [[Code]](https://github.com/OPPO-Mente-Lab/PEA-Diffusion)
    - Skews in the Phenomenon Space Hinder Generalization in Text-to-Image Generation [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11936.pdf) [[Code]](https://github.com/zdxdsw/skewed_relations_T2I)
    - ***Parrot:*** Pareto-optimal Multi-Reward Reinforcement Learning Framework for Text-to-Image Generation [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05562.pdf)
    - Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10495.pdf) [[Code]](https://github.com/ShihaoZhaoZSH/LaVi-Bridge) [[Project]](https://shihaozhaozsh.github.io/LaVi-Bridge/)
    - ***MobileDiffusion:*** Instant Text-to-Image Generation on Mobile Devices [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07923.pdf)
    - ***PixArt-Σ:*** Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2403.04692) [[Code]](https://github.com/PixArt-alpha/PixArt-sigma) [[Project]](https://pixart-alpha.github.io/PixArt-sigma-project/)
    - ***CogView3:*** Finer and Faster Text-to-Image Generation via Relay Diffusion [[Paper]](https://arxiv.org/pdf/2403.05121) [[Code]](https://github.com/THUDM/CogView)
  - **ICLR**
    - Patched Denoising Diffusion Models For High-Resolution Image Synthesis [[Paper]](https://arxiv.org/pdf/2308.01316.pdf) [[Code]](https://github.com/mlpc-ucsd/patch-dm)
    - ***Relay Diffusion:*** Unifying diffusion process across resolutions for image synthesis [[Paper]](https://arxiv.org/pdf/2309.03350.pdf) [[Code]](https://github.com/THUDM/RelayDiffusion)
    - ***SDXL:*** Improving Latent Diffusion Models for High-Resolution Image Synthesis [[Paper]](https://arxiv.org/pdf/2307.01952.pdf) [[Code]](https://github.com/Stability-AI/generative-models)
    - Compose and Conquer: Diffusion-Based 3D Depth Aware Composable Image Synthesis [[Paper]](https://arxiv.org/pdf/2401.09048.pdf) [[Code]](https://github.com/tomtom1103/compose-and-conquer)
    - ***PixArt-α:*** Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis [[Paper]](https://arxiv.org/pdf/2310.00426.pdf) [[Code]](https://github.com/PixArt-alpha/PixArt-alpha) [[Project]](https://pixart-alpha.github.io/) [[Demo]](https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha)
  - **SIGGRAPH**
    - ***RGB↔X:*** Image Decomposition and Synthesis Using Material- and Lighting-aware Diffusion Models [[Paper]](https://zheng95z.github.io/assets/files/sig24-rgbx.pdf) [[Project]](https://zheng95z.github.io/publications/rgbx24)
  - **AAAI**
    - Semantic-aware Data Augmentation for Text-to-image Synthesis [[Paper]](https://arxiv.org/pdf/2312.07951.pdf)
    - Text-to-Image Generation for Abstract Concepts [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28122)
  - **arXiv**
    - Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2402.10210.pdf)
    - ***RPG:*** Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs [[Paper]](https://arxiv.org/pdf/2401.11708.pdf) [[Code]](https://github.com/YangLing0818/RPG-DiffusionMaster)
    - ***Playground v2.5:*** Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2402.17245.pdf) [[Code]](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
    - ***ResAdapter:*** Domain Consistent Resolution Adapter for Diffusion Models [[Paper]](https://arxiv.org/pdf/2403.02084.pdf) [[Code]](https://github.com/bytedance/res-adapter) [[Project]](https://res-adapter.github.io/)
    - ***InstantID:*** Zero-shot Identity-Preserving Generation in Seconds [[Paper]](https://arxiv.org/pdf/2401.07519.pdf) [[Code]](https://github.com/InstantID/InstantID) [[Project]](https://instantid.github.io/) [[Demo]](https://huggingface.co/spaces/InstantX/InstantID)
    - ***PIXART-δ:*** Fast and Controllable Image Generation with Latent Consistency Models [[Paper]](https://arxiv.org/pdf/2401.05252) [[Code]](b.com/PixArt-alpha/PixArt-alpha?tab=readme-ov-file)
    - ***ELLA:*** Equip Diffusion Models with LLM for Enhanced Semantic Alignment [[Paper]](https://arxiv.org/pdf/2403.05135) [[Code]](https://github.com/ELLA-Diffusion/ELLA) [[Project]](https://ella-diffusion.github.io/) 
    - ***Text2Street:*** Controllable Text-to-image Generation for Street Views [[Paper]](https://arxiv.org/pdf/2402.04504.pdf)
    - ***LayerDiffuse:*** Transparent Image Layer Diffusion using Latent Transparency [[Paper]](https://arxiv.org/pdf/2402.17113) [[Code]](https://github.com/layerdiffusion/LayerDiffuse)
    - ***SD3-Turbo:*** Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation [[Paper]](https://arxiv.org/pdf/2403.12015.pdf)
    - ***StreamMultiDiffusion:*** Real-Time Interactive Generation with Region-Based Semantic Control [[Paper]](https://arxiv.org/pdf/2403.09055) [[Code]](https://github.com/ironjr/StreamMultiDiffusion)
    - ***SVGDreamer:*** Text Guided SVG Generation with Diffusion Model [[Paper]](https://arxiv.org/pdf/2312.16476) [[Code]](https://github.com/ximinng/SVGDreamer) [[Project]](https://ximinng.github.io/SVGDreamer-project/)
    - ***PromptCharm:*** Text-to-Image Generation through Multi-modal Prompting and Refinement [[Paper]](https://arxiv.org/pdf/2403.04014.pdf)
    - ***YOSO:*** You Only Sample Once: Taming One-Step Text-to-Image Synthesis by Self-Cooperative Diffusion GANs [[Paper]](https://arxiv.org/pdf/2403.12931) [[Code]](https://github.com/mlpen/YOSO)
    - ***SingDiffusion:*** Tackling the Singularities at the Endpoints of Time Intervals in Diffusion Models [[Paper]](https://arxiv.org/pdf/2403.08381) [[Code]](https://github.com/PangzeCheung/SingDiffusion)
    - ***CoMat:*** Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching [[Paper]](https://arxiv.org/pdf/2404.03653) [[Code]](https://github.com/CaraJ7/CoMat) [[Project]](https://caraj7.github.io/comat/)
    - ***StoryDiffusion:*** Consistent Self-Attention for Long-Range Image and Video Generation [[Paper]](https://arxiv.org/pdf/2405.01434) [[Code]](https://github.com/HVision-NKU/StoryDiffusion) [[Project]](https://storydiffusion.github.io/) [[Demo]](https://huggingface.co/spaces/YupengZhou/StoryDiffusion)
    - Face Adapter for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control [[Paper]](https://arxiv.org/pdf/2405.12970) [[Project]](https://faceadapter.github.io/face-adapter.github.io/)
    - ***LinFusion:*** 1 GPU, 1 Minute, 16K Image [[Paper]](https://arxiv.org/pdf/2409.02097) [[Code]](https://github.com/Huage001/LinFusion) [[Project]](https://lv-linfusion.github.io/) [[Demo]](https://huggingface.co/spaces/Huage001/LinFusion-SD-v1.5)
    - ***OmniGen:*** Unified Image Generation [[Paper]](https://arxiv.org/pdf/2409.11340) [[Code]](https://github.com/VectorSpaceLab/OmniGen)
    - ***CoMPaSS:*** Enhancing Spatial Understanding in Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/abs/2412.13195) [[Code]](https://github.com/blurgyy/CoMPaSS)
  - **Others**
    - ***Stable Cascade*** [[Blog]](https://stability.ai/news/introducing-stable-cascade) [[Code]](https://github.com/Stability-AI/StableCascade)

[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="text-year-2023">**Year 2023**</span>
  - **CVPR**
    - ***GigaGAN:*** Scaling Up GANs for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Scaling_Up_GANs_for_Text-to-Image_Synthesis_CVPR_2023_paper.pdf) [[Reproduced Code]](https://github.com/lucidrains/gigagan-pytorch) [[Project]](https://mingukkang.github.io/GigaGAN/) [[Video]](https://www.youtube.com/watch?v=ZjxtuDQkOPY&feature=youtu.be)
    - ***ERNIE-ViLG 2.0:*** Improving Text-to-Image Diffusion Model With Knowledge-Enhanced Mixture-of-Denoising-Experts [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_ERNIE-ViLG_2.0_Improving_Text-to-Image_Diffusion_Model_With_Knowledge-Enhanced_Mixture-of-Denoising-Experts_CVPR_2023_paper.pdf)
    - Shifted Diffusion for Text-to-image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Shifted_Diffusion_for_Text-to-Image_Generation_CVPR_2023_paper.pdf) [[Code]](https://github.com/drboog/Shifted_Diffusion)
    - ***GALIP:*** Generative Adversarial CLIPs for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tao_GALIP_Generative_Adversarial_CLIPs_for_Text-to-Image_Synthesis_CVPR_2023_paper.pdf) [[Code]](https://github.com/tobran/GALIP)
    - ***Specialist Diffusion:*** Plug-and-Play Sample-Efficient Fine-Tuning of Text-to-Image Diffusion Models to Learn Any Unseen Style [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Specialist_Diffusion_Plug-and-Play_Sample-Efficient_Fine-Tuning_of_Text-to-Image_Diffusion_Models_To_CVPR_2023_paper.pdf) [[Code]](https://github.com/Picsart-AI-Research/Specialist-Diffusion)
    - Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.pdf)
    - ***RIATIG:*** Reliable and Imperceptible Adversarial Text-to-Image Generation with Natural Prompts [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_RIATIG_Reliable_and_Imperceptible_Adversarial_Text-to-Image_Generation_With_Natural_Prompts_CVPR_2023_paper.pdf) [[Code]](https://github.com/WUSTL-CSPL/RIATIG)
    - Multi-Concept Customization of Text-to-Image Diffusion [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf) [[Project]](https://www.cs.cmu.edu/~custom-diffusion/) [[Code]](https://github.com/adobe-research/custom-diffusion) 
  - **ICCV**
    - ***DiffFit:*** Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.pdf) [[Code]](https://github.com/mkshing/DiffFit-pytorch) [[Demo]](https://colab.research.google.com/github/mkshing/difffit-pytorch/blob/main/scripts/difffit_pytorch.ipynb)
  - **NeurIPS**
    - ***ImageReward:*** Learning and Evaluating Human Preferences for Text-to-Image Generation [[Paper]](https://openreview.net/pdf?id=JVzeOYEx6d) [[Code]](https://github.com/THUDM/ImageReward)
    - ***RAPHAEL***: Text-to-Image Generation via Large Mixture of Diffusion Paths [[Paper]](https://arxiv.org/pdf/2305.18295) [[Project]](https://raphael-painter.github.io/)
    - Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment [[Paper]](https://openreview.net/pdf?id=AOKU4nRw1W) [[Code]](https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models)
    - ***DenseDiffusion:*** Dense Text-to-Image Generation with Attention Modulation [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Dense_Text-to-Image_Generation_with_Attention_Modulation_ICCV_2023_paper.pdf) [[Code]](https://github.com/naver-ai/densediffusion)
  - **ICLR**
    - Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis [[Paper]](https://openreview.net/pdf?id=PUIqjT4rzq7) [[Code]](https://github.com/weixi-feng/Structured-Diffusion-Guidance)
  - **ICML**
    - ***StyleGAN-T:*** Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis [[Paper]](https://proceedings.mlr.press/v202/sauer23a/sauer23a.pdf) [[Code]](https://github.com/autonomousvision/stylegan-t) [[Project]](https://sites.google.com/view/stylegan-t/) [[Video]](https://www.youtube.com/watch?v=MMj8OTOUIok)
    - ***Muse:*** Text-To-Image Generation via Masked Generative Transformers [[Paper]](https://proceedings.mlr.press/v202/chang23b/chang23b.pdf) [[Reproduced Code]](https://github.com/lucidrains/muse-maskgit-pytorch) [[Project]](https://muse-icml.github.io/)
    - ***UniDiffusers:*** One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale [[Paper]](https://arxiv.org/pdf/2303.06555) [[Code]](https://github.com/thu-ml/unidiffuser)
  - **ACM MM**
    - ***SUR-adapter:*** Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models [[Paper]](https://arxiv.org/pdf/2305.05189.pdf) [[Code]](https://github.com/Qrange-group/SUR-adapter)
    - ***ControlStyle:*** Text-Driven Stylized Image Generation Using Diffusion Priors [[Paper]](https://arxiv.org/pdf/2311.05463.pdf)
  - **SIGGRAPH**
    - ***Attend-and-Excite:*** Attention-Based Semantic Guidance for Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2301.13826.pdf) [[Code]](https://github.com/yuval-alaluf/Attend-and-Excite) [[Project]](https://yuval-alaluf.github.io/Attend-and-Excite/) [[Demo]](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite)
  - **arXiv**
    - ***P+:*** Extended Textual Conditioning in Text-to-Image Generation [[Paper]](https://prompt-plus.github.io/files/PromptPlus.pdf)
    - ***SDXL-Turbo:*** Adversarial Diffusion Distillation [[Paper]](https://arxiv.org/pdf/2311.17042.pdf) [[Code]](https://github.com/Stability-AI/generative-models)
    - ***Wuerstchen:*** An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2306.00637.pdf) [[Code]](https://github.com/dome272/Wuerstchen)
    - ***StreamDiffusion:*** A Pipeline-level Solution for Real-time Interactive Generation [[Paper]](https://arxiv.org/pdf/2312.12491.pdf) [[Project]](https://github.com/cumulo-autumn/StreamDiffusion)
    - ***ParaDiffusion:*** Paragraph-to-Image Generation with Information-Enriched Diffusion Model [[Paper]](https://arxiv.org/pdf/2311.14284) [[Code]](https://github.com/weijiawu/ParaDiffusion)
  - **Others**
    - ***DALL-E 3:*** Improving Image Generation with Better Captions [[Paper]](https://cdn.openai.com/papers/dall-e-3.pdf)

[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="text-year-2022">**Year 2022**</span>
  - **CVPR**
    - 🔥 ***Stable Diffusion:*** High-Resolution Image Synthesis With Latent Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) [[Code]](https://github.com/CompVis/latent-diffusion) [[Project]](https://ommer-lab.com/research/latent-diffusion-models/)
    - Vector Quantized Diffusion Model for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf) [[Code]](https://github.com/cientgu/VQ-Diffusion)
    - ***DF-GAN:*** A Simple and Effective Baseline for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tao_DF-GAN_A_Simple_and_Effective_Baseline_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf) [[Code]](https://github.com/tobran/DF-GAN)
    - ***LAFITE:*** Towards Language-Free Training for Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Towards_Language-Free_Training_for_Text-to-Image_Generation_CVPR_2022_paper.pdf) [[Code]](https://github.com/drboog/Lafite)
    - Text-to-Image Synthesis based on Object-Guided Joint-Decoding Transformer [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Text-to-Image_Synthesis_Based_on_Object-Guided_Joint-Decoding_Transformer_CVPR_2022_paper.pdf)
    - ***StyleT2I:*** Toward Compositional and High-Fidelity Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_StyleT2I_Toward_Compositional_and_High-Fidelity_Text-to-Image_Synthesis_CVPR_2022_paper.pdf) [[Code]](https://github.com/zhihengli-UR/StyleT2I)
  - **ECCV**
    - ***Make-A-Scene:*** Scene-Based Text-to-Image Generation with Human Priors [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750087.pdf) [[Code]](https://github.com/CasualGANPapers/Make-A-Scene) [[Demo]](https://colab.research.google.com/drive/1SPyQ-epTsAOAu8BEohUokN4-b5RM_TnE?usp=sharing)
    - Trace Controlled Text to Image Generation [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960058.pdf)
    - Improved Masked Image Generation with Token-Critic [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830070.pdf)
    - ***VQGAN-CLIP:*** Open Domain Image Generation and Manipulation Using Natural Language [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970088.pdf) [[Code]](https://github.com/EleutherAI/vqgan-clip)
    - ***TISE:*** Bag of Metrics for Text-to-Image Synthesis Evaluation [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960585.pdf) [[Code]](https://github.com/VinAIResearch/tise-toolbox)
    - ***StoryDALL-E:*** Adapting Pretrained Text-to-image Transformers for Story Continuation [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970070.pdf) [[Code]](https://github.com/adymaharana/storydalle) [[Demo]](https://huggingface.co/spaces/ECCV2022/storydalle)
  - **NeurIPS**
    - ***CogView2:*** Faster and Better Text-to-Image Generation via Hierarchical Transformers [[Paper]](https://openreview.net/pdf?id=GkDbQb6qu_r) [[Code]](https://openreview.net/pdf?id=GkDbQb6qu_r)
    - ***Imagen:*** Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding [[Paper]](https://papers.nips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf) [[Reproduced Code]](https://github.com/lucidrains/imagen-pytorch) [[Project]](https://imagen.research.google/) [[***Imagen 2***]](https://deepmind.google/technologies/imagen-2/)
  - **ACM MM**
    - ***Adma-GAN:*** Attribute-Driven Memory Augmented GANs for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2209.14046.pdf) [[Code]](https://github.com/Hsintien-Ng/Adma-GAN)
    - Background Layout Generation and Object Knowledge Transfer for Text-to-Image Generation [[Paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3548154)
    - ***DSE-GAN:*** Dynamic Semantic Evolution Generative Adversarial Network for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2209.01339.pdf)
    - ***AtHom:*** Two Divergent Attentions Stimulated By Homomorphic Training in Text-to-Image Synthesis [[Paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3548159)
  - **arXiv**
    - ***DALLE-2:*** Hierarchical Text-Conditional Image Generation with CLIP Latents [[Paper]](https://cdn.openai.com/papers/dall-e-2.pdf)
    - ***PITI:*** Pretraining is All You Need for Image-to-Image Translation [[Paper]](https://arxiv.org/pdf/2205.12952.pdf) [[Code]](https://github.com/PITI-Synthesis/PITI)

[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="text-year-2021">**Year 2021**</span>
  - **ICCV**
    -  ***DAE-GAN:*** Dynamic Aspect-aware GAN for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Ruan_DAE-GAN_Dynamic_Aspect-Aware_GAN_for_Text-to-Image_Synthesis_ICCV_2021_paper.pdf) [[Code]](https://github.com/hiarsal/DAE-GAN)
  - **NeurIPS**
    - ***CogView:*** Mastering Text-to-Image Generation via Transformers [[Paper]](https://proceedings.neurips.cc/paper/2021/file/a4d92e2cd541fca87e4620aba658316d-Paper.pdf) [[Code]](https://github.com/THUDM/CogView) [[Demo]](https://thudm.github.io/CogView/index.html)
    - ***UFC-BERT:*** Unifying Multi-Modal Controls for Conditional Image Synthesis [[Paper]](https://proceedings.neurips.cc/paper/2021/file/e46bc064f8e92ac2c404b9871b2a4ef2-Paper.pdf)
  - **ICML**
    - ***DALLE-1:*** Zero-Shot Text-to-Image Generation [[Paper]](https://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf) [[Reproduced Code]](https://github.com/lucidrains/DALLE-pytorch)
   -  **ACM MM**
      -  Cycle-Consistent Inverse GAN for Text-to-Image Synthesis [[Paper]](https://arxiv.org/pdf/2108.01361.pdf)
      -  ***R-GAN:*** Exploring Human-like Way for Reasonable Text-to-Image Synthesis via Generative Adversarial Networks [[Paper]](https://dl.acm.org/doi/10.1145/3474085.3475363)

[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="text-year-2020">**Year 2020**</span>
  - **ACM MM**
    - Text-to-Image Synthesis via Aesthetic Layout [[Paper]](https://dl.acm.org/doi/10.1145/3394171.3414357)

[<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
## Conditional Text-to-Image Generation
- <span id="conditional-year-2025">**Year 2025**</span>
  - **CVPR**
    - Training-free Dense-Aligned Diffusion Guidance for Modular Conditional Image Synthesis [[Paper]](https://arxiv.org/pdf/2504.01515) [[Code]](https://github.com/ZixuanWang0525/DADG)
    - Devil is in the Detail: Towards Injecting Fine Details of Image Prompt in Image Generation via Conflict-free Guidance and Stratified Attention [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jo_Devil_is_in_the_Detail_Towards_Injecting_Fine_Details_of_CVPR_2025_paper.pdf)
    - Color Alignment in Diffusion [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Shum_Color_Alignment_in_Diffusion_CVPR_2025_paper.pdf)
    - GPS as a Control Signal for Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_GPS_as_a_Control_Signal_for_Image_Generation_CVPR_2025_paper.pdf)
    - ***ART:*** Anonymous Region Transformer for Variable Multi-Layer Transparent Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Pu_ART_Anonymous_Region_Transformer_for_Variable_Multi-Layer_Transparent_Image_Generation_CVPR_2025_paper.pdf)
    - ***StyleStudio:*** Text-Driven Style Transfer with Selective Control of Style Elements [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_StyleStudio_Text-Driven_Style_Transfer_with_Selective_Control_of_Style_Elements_CVPR_2025_paper.pdf)
    - ***TKG-DM:*** Training-free Chroma Key Content Generation Diffusion Model [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Morita_TKG-DM_Training-free_Chroma_Key_Content_Generation_Diffusion_Model_CVPR_2025_paper.pdf)
    - ***Attention Distillation:*** A Unified Approach to Visual Characteristics Transfer [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_Attention_Distillation_A_Unified_Approach_to_Visual_Characteristics_Transfer_CVPR_2025_paper.pdf)
    - ***PTDiffusion:*** Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Model [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_PTDiffusion_Free_Lunch_for_Generating_Optical_Illusion_Hidden_Pictures_with_CVPR_2025_paper.pdf)
    - Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Cohen_Conditional_Balance_Improving_Multi-Conditioning_Trade-Offs_in_Image_Generation_CVPR_2025_paper.pdf)
    - ***UNIC-Adapter:*** Unified Image-instruction Adapter with Multi-modal Transformer for Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_UNIC-Adapter_Unified_Image-instruction_Adapter_with_Multi-modal_Transformer_for_Image_Generation_CVPR_2025_paper.pdf)
    - ***ROICtrl:*** Boosting Instance Control for Visual Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Gu_ROICtrl_Boosting_Instance_Control_for_Visual_Generation_CVPR_2025_paper.pdf)
    - Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Baumann_Continuous_Subject-Specific_Attribute_Control_in_T2I_Models_by_Identifying_Semantic_CVPR_2025_paper.pdf)
  - **ICCV**
    - ***UNO:*** A Universal Customization Method for Both Single and Multi‑Subject Conditioning [[Paper]](https://arxiv.org/abs/2504.02160) [[Project]](https://bytedance.github.io/UNO) [[Code]](https://github.com/bytedance/UNO)
    - ***CoMPaSS:*** Enhancing Spatial Understanding in Text‑to‑Image Diffusion Models [[Paper]](https://arxiv.org/abs/2412.13195) [[Project]](https://compass.blurgy.xyz) [[Code]](https://github.com/blurgyy/CoMPaSS)
    - ***SP‑Ctrl:*** Rethink Sparse Signals for Pose‑Guided Text‑to‑Image Generation [[Paper]](https://arxiv.org/abs/2506.20983) [[Code]](https://github.com/DREAMXFAR/SP-Ctrl)
    - ***CompCon:*** Discovering Divergent Representations Between Text‑to‑Image Models [[Paper]](https://arxiv.org/abs/2509.08940) [[Code]](https://github.com/adobe-research/CompCon)
    - ***C2OT:*** The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow‑Based Generation [[Paper]](https://arxiv.org/abs/2503.10636) [[Project]](https://hkchengrex.com/C2OT) [[Code]](https://github.com/hkchengrex/C2OT)
    - ***RAG‑Diffusion:*** Region‑Aware Text‑to‑Image Generation via Hard Binding and Soft Refinement [[Paper]](https://arxiv.org/abs/2411.06558) [[Project]](https://huggingface.co/spaces/NJU/RAG-Diffusion) [[Code]](https://github.com/NJU-PCALab/RAG-Diffusion)
    - ***CharaConsist:*** Fine‑Grained Consistent Character Generation [[Paper]](https://arxiv.org/abs/2507.11533) [[Project]](https://murray-wang.github.io/CharaConsist) [[Code]](https://github.com/Murray-Wang/CharaConsist)
    - ***Shadow Director:*** Parametric Shadow Control for Portrait Generation in Text‑to‑Image Diffusion Models [[Paper]](https://arxiv.org/abs/2503.21943) [[Project]](https://hm-cai.com/ShadowDirector)
    - ***ImageGen‑CoT:*** Enhancing Text‑to‑Image In‑Context Learning with Chain‑of‑Thought Reasoning [[Paper]](https://arxiv.org/abs/2503.19312) [[Project]](https://imagegen-cot.github.io)
    - ***Lay2Story:*** Extending Diffusion Transformers for Layout-Togglable Story Generation [[Paper]](https://arxiv.org/abs/2508.08949)
    - Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion [[Paper]](https://arxiv.org/abs/2508.09575)
    - ***EmotiCrafter:*** Text-to-Emotional-Image Generation based on Valence-Arousal Model [[Paper]](https://arxiv.org/abs/2501.05710)
    - ***VisualCloze:*** A Universal Image Generation Framework via Visual In-Context Learning [[Paper]](https://arxiv.org/abs/2504.07960)
    - ***DreamLayer:*** Simultaneous Multi-Layer Generation via Diffusion Model [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_DreamLayer_Simultaneous_Multi-Layer_Generation_via_Diffusion_Model_ICCV_2025_paper.pdf)
    - ***CompSlider:*** Compositional Slider for Disentangled Multiple-Attribute Image Generation [[Paper]](https://arxiv.org/abs/2509.01028)
    - ***DC-ControlNet:*** Decoupling Inter- and Intra-Element Conditions in Image Generation with Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_DC-ControlNet_Decoupling_Inter-_and_Intra-Element_Conditions_in_Image_Generation_with_ICCV_2025_paper.pdf)
    - Wasserstein Style Distribution Analysis and Transform for Stylized Image Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Wasserstein_Style_Distribution_Analysis_and_Transform_for_Stylized_Image_Generation_ICCV_2025_paper.pdf)
    - ***Co-Painter:*** Fine-Grained Controllable Image Stylization via Implicit Decoupling and Adaptive Injection [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Fu_Co-Painter_Fine-Grained_Controllable_Image_Stylization_via_Implicit_Decoupling_and_Adaptive_ICCV_2025_paper.pdf)
    - Hybrid Layout Control for Diffusion Transformer: Fewer Annotations, Superior Aesthetics [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_Hybrid_Layout_Control_for_Diffusion_Transformer_Fewer_Annotations_Superior_Aesthetics_ICCV_2025_paper.pdf)
    - ***Omegance:*** A Single Parameter for Various Granularities in Diffusion-Based Synthesis [[Paper]](https://arxiv.org/abs/2411.17769)
    - ***LaRender:*** Training-Free Occlusion Control in Image Generation via Latent Rendering [[Paper]](https://arxiv.org/abs/2508.07647)
    - ***FICGen:*** Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation [[Paper]](https://arxiv.org/abs/2509.01107)
    - ***PlanGen:*** Towards Unified Layout Planning and Image Generation in Auto-Regressive Vision Language Models [[Paper]](https://arxiv.org/abs/2503.10127)
    - ***UniCombine:*** Unified Multi-Conditional Combination with Diffusion Transformer [[Paper]](https://arxiv.org/abs/2503.09277)
    - Cycle-Consistent Learning for Joint Layout-to-Image Generation and Object Detection [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Cai_Cycle-Consistent_Learning_for_Joint_Layout-to-Image_Generation_and_Object_Detection_ICCV_2025_paper.pdf)
    - ***RealGeneral:*** Unifying Visual Generation via Temporal In-Context Learning with Video Models [[Paper]](https://arxiv.org/abs/2503.10406)
    - ***IFAdapter:*** Instance Feature Control for Grounded Text-to-Image Generation [[Paper]](https://arxiv.org/abs/2409.08240)
    - ***Preserve Anything:*** Controllable Image Synthesis with Object Preservation [[Paper]](https://arxiv.org/abs/2506.22531)
    - ***EasyControl:*** Adding Efficient and Flexible Control for Diffusion Transformer [[Paper]](https://arxiv.org/abs/2503.07027)
    - ***CreatiLayout:*** Siamese Multimodal Diffusion Transformer for Creative Layout-to-Image Generation [[Paper]](https://arxiv.org/abs/2412.03859)
    - LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text Pairing [[Paper]](https://arxiv.org/abs/2507.22627)
    - ***DreamRenderer:*** Taming Multi-Instance Attribute Control in Large-Scale Text-to-Image Models [[Paper]](https://arxiv.org/abs/2503.12885)
    - ***GeoDiffusion:*** A Training-Free Framework for Accurate 3D Geometric Conditioning in Image Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Mueller_GeoDiffusion_A_Training-Free_Framework_for_Accurate_3D_Geometric_Conditioning_in_ICCV_2025_paper.pdf)
    - ***IntrinsicControlNet:*** Cross-distribution Image Generation with Real and Unreal [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_IntrinsicControlNet_Cross-distribution_Image_Generation_with_Real_and_Unreal_ICCV_2025_paper.pdf)
    - ***MUSE:*** Multi-Subject Unified Synthesis via Explicit Layout Semantic Expansion [[Paper]](https://arxiv.org/abs/2508.14440)
    - Adding Additional Control to One-Step Diffusion with Joint Distribution Matching [[Paper]](https://arxiv.org/abs/2503.06652)
    - ***MedSegFactory:*** Text-Guided Generation of Medical Image-Mask Pairs [[Paper]](https://arxiv.org/abs/2504.06897)
    - Scene Graph Guided Generation: Enable Accurate Relations Generation in Text-to-Image Models via Textural Rectification [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Shen_Scene_Graph_Guided_Generation_Enable_Accurate_Relations_Generation_in_Text-to-Image_ICCV_2025_paper.pdf)
    - ***TF-TI2I:*** Training-Free Text-and-Image-to-Image Generation via Multi-Modal Implicit-Context Learning In Text-to-Image Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Hsiao_TF-TI2I_Training-Free_Text-and-Image-to-Image_Generation_via_Multi-Modal_Implicit-Context_Learning_In_Text-to-Image_ICCV_2025_paper.pdf)
    - ***PathDiff:*** Histopathology Image Synthesis with Unpaired Text and Mask Conditions [[Paper]](https://arxiv.org/abs/2506.23440)
    - ***Att-Adapter:*** A Robust and Precise Domain-Specific Multi-Attributes T2I Diffusion Adapter via Conditional Variational Autoencoder [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Cho_Att-Adapter_A_Robust_and_Precise_Domain-Specific_Multi-Attributes_T2I_Diffusion_Adapter_ICCV_2025_paper.pdf)
  - **ICLR**
    - ***3DIS:*** Depth-Driven Decoupled Image Synthesis for Universal Multi-Instance Generation [[Paper]](https://openreview.net/forum?id=MagmwodCAB)
    - ***ControlAR:*** Controllable Image Generation with Autoregressive Models [[Paper]](https://openreview.net/forum?id=BWuBDdXVnH)
    - ***Ctrl-Adapter:*** An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model [[Paper]](https://openreview.net/forum?id=ny8T8OuNHe)
    - ***Build-A-Scene:*** Interactive 3D Layout Control for Diffusion-Based Image Generation [[Paper]](https://openreview.net/forum?id=gg6dPtdC1C)
    - ***ToddlerDiffusion:*** Interactive Structured Image Generation with Cascaded Schrödinger Bridge [[Paper]](https://openreview.net/forum?id=Jszf4et48m)
    - ***Hummingbird:*** High Fidelity Image Generation via Multimodal Context Alignment [[Paper]](https://openreview.net/forum?id=6kPBThI6ZJ)
    - ***UniCon:*** Unidirectional Information Flow for Effective Control of Large-Scale Diffusion Models [[Paper]](https://openreview.net/forum?id=uJqKf24HGN)
    - ***Less is More:*** Masking Elements in Image Condition Features Avoids Content Leakages in Style Transfer Diffusion Models [[Paper]](https://openreview.net/forum?id=88JJjsLtqr)
  - **ICML**
    - ***FlexControl:*** Computation-Aware Conditional Control with Differentiable Router for Text-to-Image Generation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/fang25j/fang25j.pdf)
    - ***EcoMapper:*** Generative Modeling for Climate-Aware Satellite Imagery [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/goktepe25a/goktepe25a.pdf) [[Code]](https://github.com/maltevb/ecomapper)
    - ***UniMC:*** Taming Diffusion Transformer for Unified Keypoint-Guided Multi-Class Image Generation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/guo25s/guo25s.pdf)
    - ***Control and Realism:*** Best of Both Worlds in Layout-to-Image without Training [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25ab/li25ab.pdf)
    - ***TextCenGen:*** Attention-Guided Text-Centric Background Adaptation for Text-to-Image Generation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/liang25i/liang25i.pdf) [[Code]](https://github.com/tianyilt/TextCenGen_Background_Adapt)
    - ***RealRAG:*** Retrieval-augmented Realistic Image Generation via Self-reflective Contrastive Learning [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/lyu25c/lyu25c.pdf) [[Project]](https://qc-ly.github.io/RealRAG-page/)
    - Learning Vision and Language Concepts for Controllable Image Generation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/xie25g/xie25g.pdf)
  - **SIGGRAPH**
    - ***FashionComposer:*** Compositional Fashion Image Generation [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3721238.3730663)
    - ***IP-Composer:*** Semantic Composition of Visual Concepts [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3721238.3730624)
  - **AAAI**
    - ***VersaGen:*** Unleashing Versatile Visual Control for Text-to-Image Synthesis [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32240)
    - ***Muses:*** 3D-Controllable Image Generation via Multi-Modal Agent Collaboration [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32280)
    - Simplifying Control Mechanism in Text-to-Image Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32309)
    - Scene Graph-Grounded Image Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32823)
    - ***EMControl:*** Adding Conditional Control to Text-to-Image Diffusion Models via Expectation-Maximization [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32828)
    - ***GRPose:*** Learning Graph Relations for Human Image Generation with Pose Priors [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33032)
    - Local Conditional Controlling for Text-to-Image Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33139)
    - ***DialogDraw:*** Image Generation and Editing System Based on Multi-Turn Dialogue [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34661)

- <span id="conditional-year-2024">**Year 2024**</span>
  - **CVPR**
    - ***PLACE:*** Adaptive Layout-Semantic Fusion for Semantic Image Synthesis [[Paper]](https://arxiv.org/pdf/2403.01852.pdf)
    - One-Shot Structure-Aware Stylized Image Synthesis [[Paper]](https://arxiv.org/pdf/2402.17275.pdf)
    - Grounded Text-to-Image Synthesis with Attention Refocusing [[Paper]](https://arxiv.org/pdf/2306.05427.pdf) [[Code]](https://github.com/Attention-Refocusing/attention-refocusing) [[Project]](https://attention-refocusing.github.io/) [[Demo]](https://huggingface.co/spaces/attention-refocusing/Attention-refocusing)
    - Coarse-to-Fine Latent Diffusion for Pose-Guided Person Image Synthesis [[Paper]](https://arxiv.org/pdf/2402.18078.pdf) [[Code]](https://github.com/YanzuoLu/CFLD)
    - ***DetDiffusion:*** Synergizing Generative and Perceptive Models for Enhanced Data Generation and Perception [[Paper]](https://arxiv.org/pdf/2403.13304)
    - ***CAN:*** Condition-Aware Neural Network for Controlled Image Generation [[Paper]](https://arxiv.org/pdf/2404.01143.pdf)
    - ***SceneDiffusion:*** Move Anything with Layered Scene Diffusion [[Paper]](https://arxiv.org/pdf/2404.07178)
    - ***Zero-Painter:*** Training-Free Layout Control for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ohanyan_Zero-Painter_Training-Free_Layout_Control_for_Text-to-Image_Synthesis_CVPR_2024_paper.pdf) [[Code]](https://github.com/Picsart-AI-Research/Zero-Painter)
    - ***MIGC:*** Multi-Instance Generation Controller for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_MIGC_Multi-Instance_Generation_Controller_for_Text-to-Image_Synthesis_CVPR_2024_paper.pdf) [[Code]](https://github.com/limuloo/MIGC) [[Project]](https://migcproject.github.io/)
    - ***FreeControl:*** Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Mo_FreeControl_Training-Free_Spatial_Control_of_Any_Text-to-Image_Diffusion_Model_with_CVPR_2024_paper.pdf) [[Code]](https://github.com/genforce/freecontrol) [[Project]](https://openaccess.thecvf.com/content/CVPR2024/papers/Mo_FreeControl_Training-Free_Spatial_Control_of_Any_Text-to-Image_Diffusion_Model_with_CVPR_2024_paper.pdf)
  - **ECCV**
    - ***PreciseControl:*** Enhancing Text-To-Image Diffusion Models with Fine-Grained Attribute Control [[Paper]](https://arxiv.org/pdf/2408.05083) [[Code]](https://github.com/rishubhpar/PreciseControl) [[Project]](https://rishubhpar.github.io/PreciseControl.home/)
    - ***AnyControl:*** Create Your Artwork with Versatile Control on Text-to-Image Generation [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01706.pdf) [[Code]](https://github.com/open-mmlab/AnyControl)
  - **NeurIPS**
    - ***Ctrl-X:*** Controlling Structure and Appearance for Text-To-Image Generation Without Guidance [[Paper]](https://arxiv.org/pdf/2406.07540) [[Code]](https://github.com/genforce/ctrl-x) [[Project]](https://genforce.github.io/ctrl-x/)
  - **ICLR**
    - Advancing Pose-Guided Image Synthesis with Progressive Conditional Diffusion Models [[Paper]](https://arxiv.org/pdf/2310.06313.pdf) [[Code]](https://github.com/muzishen/PCDMs)
  - **WACV**
    - Training-Free Layout Control with Cross-Attention Guidance [[Paper]](https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Training-Free_Layout_Control_With_Cross-Attention_Guidance_WACV_2024_paper.pdf) [[Code]](https://github.com/silent-chen/layout-guidance) [[Project]](https://silent-chen.github.io/layout-guidance/) [[Demo]](https://huggingface.co/spaces/silentchen/layout-guidance)
  - **AAAI**
    - ***SSMG:*** Spatial-Semantic Map Guided Diffusion Model for Free-form Layout-to-image Generation [[Paper]](https://arxiv.org/pdf/2308.10156.pdf)
    - Compositional Text-to-Image Synthesis with Attention Map Control of Diffusion Models [[Paper]](https://arxiv.org/pdf/2305.13921.pdf) [[Code]](https://github.com/OPPO-Mente-Lab/attention-mask-control)
  - **arXiv**
    - ***DEADiff:*** An Efficient Stylization Diffusion Model with Disentangled Representations [[Paper]](https://arxiv.org/pdf/2403.06951)
    - ***InstantStyle:*** Free Lunch towards Style-Preserving in Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2404.02733.pdf) [[Code]](https://github.com/InstantStyle/InstantStyle) [[Project]](https://instantstyle.github.io/)
    - ***ControlNet++:*** Improving Conditional Controls with Efficient Consistency Feedback [[Paper]](https://arxiv.org/pdf/2404.07987.pdf) [[Project]](https://liming-ai.github.io/ControlNet_Plus_Plus/)
    - ***Hunyuan-DiT:*** A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding [[Paper]](https://arxiv.org/pdf/2405.08748) [[Code]](https://github.com/Tencent/HunyuanDiT) [[Project]](https://dit.hunyuan.tencent.com/)
    - ***DialogGen:*** Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2403.08857) [[Code]](https://github.com/Centaurusalpha/DialogGen) [[Project]](https://hunyuan-dialoggen.github.io/)
    - ***ControlNeXt:*** Powerful and Efficient Control for Image and Video Generation [[Paper]](https://arxiv.org/pdf/2408.06070) [[Code]](https://github.com/dvlab-research/ControlNeXt) [[Project]](https://pbihao.github.io/projects/controlnext/index.html)
    - ***UniPortrait:*** A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization [[Paper]](https://arxiv.org/pdf/2408.05939) [[Code]](https://github.com/junjiehe96/UniPortrait) [[Project]](https://aigcdesigngroup.github.io/UniPortrait-Page/) [[Demo]](https://huggingface.co/spaces/Junjie96/UniPortrait)
    - ***OmniControl:*** Minimal and Universal Control for Diffusion Transformer [[Paper]](https://arxiv.org/pdf/2411.15098) [[Code]](https://github.com/Yuanshi9815/OminiControl) [[Demo]](https://huggingface.co/spaces/Yuanshi/OminiControl)
    - ***UnZipLoRA:*** Separating Content and Style from a Single Image [[Paper]](https://arxiv.org/pdf/2412.04465) [[Project]](https://unziplora.github.io/)
    - ***CtrLoRA:*** An Extensible and Efficient Framework for Controllable Image Generation [[Paper]](https://arxiv.org/pdf/2410.09400) [[Code]](https://github.com/xyfJASON/ctrlora)
    - Region-Aware Text-to-Image Generation via Hard Binding and Soft Refinement [[Paper]](https://arxiv.org/pdf/2411.06558) [[Code]](https://github.com/NJU-PCALab/RAG-Diffusion)





[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="conditional-year-2023">**Year 2023**</span>
  - **CVPR**
    - ***GLIGEN:*** Open-Set Grounded Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_GLIGEN_Open-Set_Grounded_Text-to-Image_Generation_CVPR_2023_paper.pdf) [[Code]](https://github.com/gligen/GLIGEN) [[Project]](https://gligen.github.io/) [[Demo]](https://huggingface.co/spaces/gligen/demo) [[Video]](https://www.youtube.com/watch?v=-MCkU7IAGKs&feature=youtu.be)
    - Autoregressive Image Generation using Residual Quantization [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Autoregressive_Image_Generation_Using_Residual_Quantization_CVPR_2022_paper.pdf) [[Code]](https://github.com/kakaobrain/rq-vae-transformer)
    - ***SpaText:*** Spatio-Textual Representation for Controllable Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Avrahami_SpaText_Spatio-Textual_Representation_for_Controllable_Image_Generation_CVPR_2023_paper.pdf) [[Project]](https://omriavrahami.com/spatext/) [[Video]](https://www.youtube.com/watch?v=VlieNoCwHO4)
    - Text to Image Generation with Semantic-Spatial Aware GAN [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liao_Text_to_Image_Generation_With_Semantic-Spatial_Aware_GAN_CVPR_2022_paper.pdf)
    - ***ReCo:*** Region-Controlled Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_ReCo_Region-Controlled_Text-to-Image_Generation_CVPR_2023_paper.pdf) [[Code]](https://github.com/microsoft/ReCo)
    - ***LayoutDiffusion:*** Controllable Diffusion Model for
    Layout-to-image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_LayoutDiffusion_Controllable_Diffusion_Model_for_Layout-to-Image_Generation_CVPR_2023_paper.pdf) [[Code]](https://github.com/ZGCTroy/LayoutDiffusion)
  - **ICLR**
    - ***Ctrl-U:*** Robust Conditional Image Generation via Uncertainty-aware Reward Modeling [[Paper]](https://openreview.net/forum?id=eC2ICbECNM) [[Project]](https://grenoble-zhang.github.io/Ctrl-U-Page/) [[Code]](https://github.com/grenoble-zhang/Ctrl-U)
  - **ICCV**
    - ***ControlNet:*** Adding Conditional Control to Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf) [[Code]](https://github.com/lllyasviel/ControlNet)
    - ***SceneGenie:*** Scene Graph Guided Diffusion Models for Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/ICCV2023W/SG2RL/papers/Farshad_SceneGenie_Scene_Graph_Guided_Diffusion_Models_for_Image_Synthesis_ICCVW_2023_paper.pdf) [[Code]](https://openaccess.thecvf.com/content/ICCV2023W/SG2RL/papers/Farshad_SceneGenie_Scene_Graph_Guided_Diffusion_Models_for_Image_Synthesis_ICCVW_2023_paper.pdf)
    - ***ZestGuide:*** Zero-Shot Spatial Layout Conditioning for Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Couairon_Zero-Shot_Spatial_Layout_Conditioning_for_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf)
  - **ICML**
    - ***Composer:*** Creative and Controllable Image Synthesis with Composable Conditions [[Paper]](https://proceedings.mlr.press/v202/huang23b/huang23b.pdf) [[Code]](https://github.com/ali-vilab/composer) [[Project]](https://ali-vilab.github.io/composer-page/)
    - ***MultiDiffusion:*** Fusing Diffusion Paths for Controlled Image Generation [[Paper]](https://proceedings.mlr.press/v202/bar-tal23a/bar-tal23a.pdf) [[Code]](https://github.com/omerbt/MultiDiffusion) [[Video]](https://www.youtube.com/watch?v=D2Q0D1gIeqg) [[Project]](https://multidiffusion.github.io/) [[Demo]](https://huggingface.co/spaces/weizmannscience/MultiDiffusion)
  - **SIGGRAPH** 
    - Sketch-Guided Text-to-Image Diffusion Models [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3588432.3591560) [[Reproduced Code]](https://github.com/ogkalu2/Sketch-Guided-Stable-Diffusion) [[Project]](https://sketch-guided-diffusion.github.io/)
  - **NeurIPS**
    - ***Uni-ControlNet:*** All-in-One Control to Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2305.16322.pdf) [[Code]](https://github.com/ShihaoZhaoZSH/Uni-ControlNet) [[Project]](https://shihaozhaozsh.github.io/unicontrolnet/)
    - ***Prompt Diffusion:*** In-Context Learning Unlocked for Diffusion Models [[Paper]](https://openreview.net/pdf?id=6BZS2EAkns) [[Code]](https://github.com/Zhendong-Wang/Prompt-Diffusion) [[Project]](https://zhendong-wang.github.io/prompt-diffusion.github.io/)
  - **WACV**
    - More Control for Free! Image Synthesis with Semantic Diffusion Guidance [[Paper]](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_More_Control_for_Free_Image_Synthesis_With_Semantic_Diffusion_Guidance_WACV_2023_paper.pdf)
  - **ACM MM**
    -  ***LayoutLLM-T2I:*** Eliciting Layout Guidance from LLM for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2308.05095.pdf)
  - **arXiv**
    - ***T2I-Adapter:*** Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2302.08453.pdf) [[Code]](https://github.com/TencentARC/T2I-Adapter) [[Demo]](https://huggingface.co/spaces/TencentARC/T2I-Adapter-SDXL)
    - ***BLIP-Diffusion:*** Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing [[Paper]](https://arxiv.org/pdf/2305.14720.pdf) [[Code]](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion)
    - Late-Constraint Diffusion Guidance for Controllable Image Synthesis [[Paper]](https://arxiv.org/pdf/2305.11520) [[Code]](https://github.com/AlonzoLeeeooo/LCDG)
- <span id="conditional-year-2022">**Year 2022**</span>
  - **ICLR**
    - ***SDEdit:*** Guided Image Synthesis and Editing with Stochastic Differential Equations [[Paper]](https://openreview.net/pdf?id=aBsCjcPu_tE) [[Code]](https://github.com/ermongroup/SDEdit) [[Project]](https://sde-image-editing.github.io/)

[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
## Personalized Text-to-Image Generation
- <span id="personalized-year-2025">**Year 2025**</span>
  - **CVPR**
    - ***SerialGen:*** Personalized Image Generation by First Standardization Then Personalization [[Paper]](https://arxiv.org/pdf/2412.01485) [[Project]](https://serialgen.github.io/)
    - ***PatchDPO:*** Patch-level DPO for Finetuning-free Personalized Image Generation [[Paper]](https://arxiv.org/pdf/2412.03177) [[Code]](https://github.com/hqhQAQ/PatchDPO)
    - ***DreamCache:*** Finetuning-Free Lightweight Personalized Image Generation via Feature Caching [[Paper]](https://arxiv.org/pdf/2411.17786v1)
    - ***DiffSensei:*** Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_DiffSensei_Bridging_Multi-Modal_LLMs_and_Diffusion_Models_for_Customized_Manga_CVPR_2025_paper.pdf)
    - ***TFCustom:*** Customized Image Generation with Time-Aware Frequency Feature Guidance [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_TFCustom_Customized_Image_Generation_with_Time-Aware_Frequency_Feature_Guidance_CVPR_2025_paper.pdf)
    - Multi-party Collaborative Attention Control for Image Customization [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Multi-party_Collaborative_Attention_Control_for_Image_Customization_CVPR_2025_paper.pdf)
    - ***Yo'Chameleon:*** Personalized Vision and Language Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_YoChameleon_Personalized_Vision_and_Language_Generation_CVPR_2025_paper.pdf)
    - ***CustAny:*** Customizing Anything from A Single Example [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kong_CustAny_Customizing_Anything_from_A_Single_Example_CVPR_2025_paper.pdf)
    - ***PersonaHOI:*** Effortlessly Improving Face Personalization in Human-Object Interaction Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_PersonaHOI_Effortlessly_Improving_Face_Personalization_in_Human-Object_Interaction_Generation_CVPR_2025_paper.pdf)
    - ***Visual Persona:*** Foundation Model for Full-Body Human Customization [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Nam_Visual_Persona_Foundation_Model_for_Full-Body_Human_Customization_CVPR_2025_paper.pdf)
    - ***HiFi-Portrait:*** Zero-shot Identity-preserved Portrait Generation with High-fidelity Multi-face Fusion [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_HiFi-Portrait_Zero-shot_Identity-preserved_Portrait_Generation_with_High-fidelity_Multi-face_Fusion_CVPR_2025_paper.pdf)
    - ***APT:*** Adaptive Personalized Training for Diffusion Models with Limited Data [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chae_APT_Adaptive_Personalized_Training_for_Diffusion_Models_with_Limited_Data_CVPR_2025_paper.pdf)
    - ***ConceptGuard:*** Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_ConceptGuard_Continual_Personalized_Text-to-Image_Generation_with_Forgetting_and_Confusion_Mitigation_CVPR_2025_paper.pdf)
    - ***MC^2:*** Multi-concept Guidance for Customized Multi-concept Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_MC2_Multi-concept_Guidance_for_Customized__Multi-concept_Generation_CVPR_2025_paper.pdf)
    - Personalized Preference Fine-tuning of Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Dang_Personalized_Preference_Fine-tuning_of_Diffusion_Models_CVPR_2025_paper.pdf)
    - Diffusion Self-Distillation for Zero-Shot Customized Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Cai_Diffusion_Self-Distillation_for_Zero-Shot_Customized_Image_Generation_CVPR_2025_paper.pdf)
    - ***LoRACLR:*** Contrastive Adaptation for Customization of Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Simsar_LoRACLR_Contrastive_Adaptation_for_Customization_of_Diffusion_Models_CVPR_2025_paper.pdf)
    - ***HyperLoRA:*** Parameter-Efficient Adaptive Generation for Portrait Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_HyperLoRA_Parameter-Efficient_Adaptive_Generation_for_Portrait_Synthesis_CVPR_2025_paper.pdf)
    - ***LaTexBlend:*** Scaling Multi-concept Customized Generation with Latent Textual Blending [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jin_LaTexBlend_Scaling_Multi-concept_Customized_Generation_with_Latent_Textual_Blending_CVPR_2025_paper.pdf)
    - ***DreamRelation:*** Bridging Customization and Relation Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_DreamRelation_Bridging_Customization_and_Relation_Generation_CVPR_2025_paper.pdf)
    - ***Multitwine:*** Multi-Object Compositing with Text and Layout Control [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Tarres_Multitwine_Multi-Object_Compositing_with_Text_and_Layout_Control_CVPR_2025_paper.pdf)
    - Efficient Personalization of Quantized Diffusion Model without Backpropagation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Seo_Efficient_Personalization_of_Quantized_Diffusion_Model_without_Backpropagation_CVPR_2025_paper.pdf)
    - ***ArtiFade:*** Learning to Generate High-quality Subject from Blemished Images [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_ArtiFade_Learning_to_Generate_High-quality_Subject_from_Blemished_Images_CVPR_2025_paper.pdf)
    - ***ID-Patch:*** Robust ID Association for Group Photo Personalization [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_ID-Patch_Robust_ID_Association_for_Group_Photo_Personalization_CVPR_2025_paper.pdf)
    - Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Shin_Large-Scale_Text-to-Image_Model_with_Inpainting_is_a_Zero-Shot_Subject-Driven_Image_CVPR_2025_paper.pdf)
  - **ICCV**
    - ***DrUM:*** Draw Your Mind: Personalized Generation via Condition‑Level Modeling in Text‑to‑Image Diffusion Models [[Paper]](https://arxiv.org/abs/2508.03481) [[Code]](https://github.com/Burf/DrUM)
    - ***PersonaCraft:*** Personalized and Controllable Full‑Body Multi‑Human Scene Generation Using Occlusion‑Aware 3D‑Conditioned Diffusion [[Paper]](https://arxiv.org/abs/2411.18068) [[Project]](https://gwang-kim.github.io/persona_craft/) [[Code]](https://github.com/gwang-kim/PersonaCraft)
    - ***Steering Guidance:*** Steering Guidance for Personalized Text‑to‑Image Diffusion Models [[Paper]](https://arxiv.org/abs/2508.00319)
    - ***FreeCus:*** FreeCus: Free Lunch Subject‑Driven Customization in Diffusion Transformers [[Paper]](https://arxiv.org/abs/2507.15249) [[Code]](https://github.com/Monalissaa/FreeCus)
    - ***PromptDresser:*** Improving the Quality and Controllability of Virtual Try‑On via Generative Textual Prompt and Prompt‑aware Mask [[Paper]](https://arxiv.org/abs/2412.16978) [[Code]](https://github.com/rlawjdghek/PromptDresser)
    - ***DynamicID:*** Zero‑Shot Multi‑ID Image Personalization with Flexible Facial Editability [[Paper]](https://arxiv.org/abs/2503.06505) [[Code]](https://github.com/ByteCat-bot/DynamicID)
    - ***UniversalBooth:*** Model‑Agnostic Personalized Text‑to‑Image Generation
    - ***ARBooth:*** Fine‑Tuning Visual Autoregressive Models for Subject‑Driven Generation [[Paper]](https://arxiv.org/abs/2504.02612) [[Project]](https://jiwoogit.github.io/ARBooth/) [[Code]](https://github.com/jiwoogit/ARBooth)
    - ⚠️ ***ConceptSplit:*** Decoupled Multi‑Concept Personalization of Diffusion Models via Token‑wise Adaptation and Attention Disentanglement [[Code]](https://github.com/KU-VGI/ConceptSplit)
    - ⚠️ ***ObjectMate:*** A Recurrence Prior for Object Insertion and Subject‑Driven Generation [[Project]](https://object-mate.com)
    - ***LoRAverse:*** A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Sonmezer_LoRAverse_A_Submodular_Framework_to_Retrieve_Diverse_Adapters_for_Diffusion_ICCV_2025_paper.pdf)
    - ***CSD-VAR:*** Content-Style Decomposition in Visual Autoregressive Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Nguyen_CSD-VAR_Content-Style_Decomposition_in_Visual_Autoregressive_Models_ICCV_2025_paper.pdf)
    - ***ImageGem:*** In-the-wild Generative Image Interaction Dataset for Generative Model Personalization [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_ImageGem_In-the-wild_Generative_Image_Interaction_Dataset_for_Generative_Model_Personalization_ICCV_2025_paper.pdf)
    - ***DuoLoRA:*** Cycle-consistent and Rank-disentangled Content-Style Personalization [[Paper]](https://arxiv.org/abs/2504.13206)
    - ***F-Bench:*** Rethinking Human Preference Evaluation Metrics for Benchmarking Face Generation, Customization, and Restoration [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_F-Bench_Rethinking_Human_Preference_Evaluation_Metrics_for_Benchmarking_Face_Generation_ICCV_2025_paper.pdf)
    - Generating Multi-Image Synthetic Data for Text-to-Image Customization [[Paper]](https://arxiv.org/abs/2502.01720)
    - Contrastive Test-Time Composition of Multiple LoRA Models for Image Generation [[Paper]](https://arxiv.org/abs/2403.19776)
    - ***LoRA.rar:*** Learning to Merge LoRAs via Hypernetworks for Subject-Style Conditioned Image Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Shenaj_LoRA.rar_Learning_to_Merge_LoRAs_via_Hypernetworks_for_Subject-Style_Conditioned_ICCV_2025_paper.pdf)
    - Continual Personalization for Diffusion Models [[Paper]](https://arxiv.org/abs/2510.02296)
    - ***QR-LoRA:*** Efficient and Disentangled Fine-tuning via QR Decomposition for Customized Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_QR-LoRA_Efficient_and_Disentangled_Fine-tuning_via_QR_Decomposition_for_Customized_ICCV_2025_paper.pdf)
  - **NeurIPS**
    - ***MS-Diffusion:*** Multi-Subject Zero-Shot Image Personalized with Layout Guidance [[Paper]](https://arxiv.org/pdf/2406.07209) [[Project]](https://ms-diffusion.github.io/) [[Code]](https://github.com/MS-Diffusion/MS-Diffusion)
    - ***ClassDiffusion:*** More Aligned Personalization Tuning with Explicit Class Guidance [[Paper]](https://openreview.net/pdf?id=iTm4H6N4aG) [[Project]](https://classdiffusion.github.io/) [[Code]](https://github.com/Rbrq03/ClassDiffusion)
    - ***DreamBench++:*** A Human-Aligned Benchmark for Personalized Image Generation [[Paper]](https://arxiv.org/pdf/2406.16855) [[Project]](https://dreambenchplus.github.io/)
    - ***TweedieMix:*** Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation [[Paper]](https://arxiv.org/pdf/2410.05591) [[Code]](https://github.com/KwonGihyun/TweedieMix)
  - **ICLR**
    - Cached Multi-Lora Composition for Multi-Concept Image Generation [[Paper]](https://openreview.net/forum?id=4iFSBgxvIO)
    - ***DreamDistribution:*** Learning Prompt Distribution for Diverse In-distribution Generation [[Paper]](https://openreview.net/forum?id=oQoQ4u6MQC)
    - ***Storybooth:*** Training-Free Multi-Subject Consistency for Improved Visual Storytelling [[Paper]](https://openreview.net/forum?id=JZLon6cvx8)
    - ***DisEnvisioner:*** Disentangled and Enriched Visual Prompt for Customized Image Generation [[Paper]](https://openreview.net/forum?id=vQxqcVGrhR)
  - **ICML**
    - Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/farhadzadeh25a/farhadzadeh25a.pdf)
    - Event-Customized Image Generation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25ar/wang25ar.pdf)
    - ***EasyRef:*** Omni-Generalized Group Image Reference for Diffusion Models via Multimodal LLM [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zong25a/zong25a.pdf) [[Code]](https://github.com/TempleX98/EasyRef)
  - **SIGGRAPH**
    - ***Nested Attention:*** Semantic-aware Attention Values for Concept Personalization [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3721238.3730634)
    - ***TokenVerse:*** Versatile Multi-concept Personalization in Token Modulation Space [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3730843)
    - ***IP-Prompter:*** Training-Free Theme-Specific Image Generation via Dynamic Visual Prompting [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3721238.3730670)
  - **AAAI**
    - ***CustomContrast:*** A Multilevel Contrastive Perspective for Subject-Driven Text-to-Image Customization [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32210)
    - ***DynASyn:*** Multi-Subject Personalization Enabling Dynamic Action Synthesis [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32259)
    - ***Dis²Booth:*** Learning Image Distribution with Disentangled Features for Text-to-Image Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32279)
    - Resolving Multi-Condition Confusion for Finetuning-Free Personalized Image Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32386)
    - ***PersonaMagic:*** Stage-Regulated High-Fidelity Face Customization with Tandem Equilibrium [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32529)
    - ***RealisID:*** Scale-Robust and Fine-Controllable Identity Customization via Local and Global Complementation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32769)
    - ***SpotActor:*** Training-Free Layout-Controlled Consistent Image Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32831)
    - ***Concept Conductor:*** Orchestrating Multiple Personalized Concepts in Text-to-Image Synthesis [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33021)
    - ***MagicNaming:*** Consistent Identity Generation by Finding a “Name Space” in T2I Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33133)
    - ***MultiBooth:*** Towards Generating All Your Concepts in an Image from Text [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33187)
    - Harmonizing Visual and Textual Embeddings for Zero-Shot Text-to-Image Customization [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34264)
- <span id="personalized-year-2024">**Year 2024**</span>
  - **CVPR**
    - Cross Initialization for Personalized Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2312.15905.pdf)
    - When StyleGAN Meets Stable Diffusion: a W+ Adapter for Personalized Image Generation [[Paper]](https://arxiv.org/pdf/2311.17461.pdf) [[Code]](https://github.com/csxmli2016/w-plus-adapter) [[Project]](https://csxmli2016.github.io/projects/w-plus-adapter/)
    - Style Aligned Image Generation via Shared Attention [[Paper]](https://arxiv.org/pdf/2312.02133.pdf) [[Code]](https://github.com/google/style-aligned) [[Project]](https://style-aligned-gen.github.io/)
    - ***InstantBooth:*** Personalized Text-to-Image Generation without Test-Time Finetuning [[Paper]](https://arxiv.org/pdf/2304.03411.pdf) [[Project]](https://jshi31.github.io/InstantBooth/)
    - High Fidelity Person-centric Subject-to-Image Synthesis [[Paper]](https://arxiv.org/pdf/2311.10329.pdf)
    - ***RealCustom:*** Narrowing Real Text Word for Real-Time Open-Domain Text-to-Image Customization [[Paper]](https://arxiv.org/pdf/2403.00483.pdf) [[Project]](https://corleone-huang.github.io/realcustom/)
    - ***DisenDiff:*** Attention Calibration for Disentangled Text-to-Image Personalization [[Paper]](https://arxiv.org/pdf/2403.18551) [[Code]](https://github.com/Monalissaa/DisenDiff)
    - ***FreeCustom:*** Tuning-Free Customized Image Generation for Multi-Concept Composition [[Paper]](https://arxiv.org/pdf/2405.13870v1) [[Code]](https://github.com/aim-uofa/FreeCustom) [[Project]](https://aim-uofa.github.io/FreeCustom/)
    - Personalized Residuals for Concept-Driven Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2405.12978)
    - Improving Subject-Driven Image Synthesis with Subject-Agnostic Guidance [[Paper]](https://arxiv.org/pdf/2405.01356)
    - ***JeDi:*** Joint-Image Diffusion Models for Finetuning-Free Personalized Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zeng_JeDi_Joint-Image_Diffusion_Models_for_Finetuning-Free_Personalized_Text-to-Image_Generation_CVPR_2024_paper.pdf)
    - Countering Personalized Text-to-Image Generation with Influence Watermarks [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Countering_Personalized_Text-to-Image_Generation_with_Influence_Watermarks_CVPR_2024_paper.pdf)
    - ***PIA:*** Your Personalized Image Animator via Plug-and-Play Modules in Text-to-Image Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_PIA_Your_Personalized_Image_Animator_via_Plug-and-Play_Modules_in_Text-to-Image_CVPR_2024_paper.pdf) [[Project]](https://pi-animator.github.io/) [[Code]](https://github.com/open-mmlab/PIA)
    - ***SSR-Encoder:*** Encoding Selective Subject Representation for Subject-Driven Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_SSR-Encoder_Encoding_Selective_Subject_Representation_for_Subject-Driven_Generation_CVPR_2024_paper.pdf) [[Code]](https://github.com/Xiaojiu-z/SSR_Encoder)
  - **ECCV**
    - Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2403.16990) [[Project]](https://omer11a.github.io/bounded-attention/)
    - Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning [[Paper]](http://arxiv.org/pdf/2407.06642v1) [[Code]](https://github.com/wfanyue/DPG-T2I-Personalization)
    - ***TIGC:*** Tuning-Free Image Customization with Image and Text Guidance [[Paper]](https://arxiv.org/pdf/2403.12658) [[Code]](https://github.com/zrealli/TIGIC) [[Project]](https://zrealli.github.io/TIGIC/)
    - ***MasterWeaver:*** Taming Editability and Face Identity for Personalized Text-to-Image Generation [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06786.pdf) [[Code]](https://github.com/csyxwei/MasterWeaver) [[Project]](https://masterweaver.github.io/)
  - **NeurIPS**
    - ***RectifID:*** Personalizing Rectified Flow with Anchored Classifier Guidance [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/afa58a5b6adc0845e0fd632132a64c39-Paper-Conference.pdf) [[Code]](https://github.com/feifeiobama/RectifID)
    - ***AttnDreamBooth:*** Towards Text-Aligned Personalized Image Generation [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/465a13a95741fab2e912f98adb07df1d-Paper-Conference.pdf) [[Project]](https://attndreambooth.github.io/) [[Code]](https://github.com/lyuPang/AttnDreamBooth)
  - **AAAI**
    - Decoupled Textual Embeddings for Customized Image Generation [[Paper]](https://arxiv.org/pdf/2312.11826.pdf)
  - **arXiv**
    - ***FlashFace:*** Human Image Personalization with High-fidelity Identity Preservation [[Paper]](https://arxiv.org/pdf/2403.17008) [[Code]](https://github.com/jshilong/FlashFace) [[Project]](https://jshilong.github.io/flashface-page)
    - ***MoMA:*** Multimodal LLM Adapter for Fast Personalized Image Generation [[Paper]](https://arxiv.org/pdf/2404.05674)
    - ***IDAdapter:*** Learning Mixed Features for Tuning-Free Personalization of Text-to-Image Models [[Paper]](https://arxiv.org/pdf/2403.13535)
    - ***CoRe:*** Context-Regularized Text Embedding Learning for Text-to-Image Personalization [[Paper]](https://www.arxiv.org/pdf/2408.15914)
    - ***Imagine yourself:*** Tuning-Free Personalized Image Generation [[Paper]](https://arxiv.org/abs/2409.13346) [[Project]](https://ai.meta.com/research/publications/imagine-yourself-tuning-free-personalized-image-generation/)
- <span id="personalized-year-2023">**Year 2023**</span>
  - **CVPR**
    - ***Custom Diffusion:*** Multi-Concept Customization of Text-to-Image Diffusion [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf) [[Code]](https://github.com/adobe-research/custom-diffusion) [[Project]](https://www.cs.cmu.edu/~custom-diffusion/)
    - ***DreamBooth:*** Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf) [[Code]](https://github.com/google/dreambooth) [[Project]](https://dreambooth.github.io/)
  - **ICCV**
    - ***ELITE:*** Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_ELITE_Encoding_Visual_Concepts_into_Textual_Embeddings_for_Customized_Text-to-Image_ICCV_2023_paper.pdf) [[Code]](https://github.com/csyxwei/ELITE)
  - **ICLR**
    - ***Textual Inversion:*** An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion [[Paper]](https://openreview.net/pdf?id=NAQvF08TcyG) [[Code]](https://github.com/rinongal/textual_inversion) [[Project]](https://textual-inversion.github.io/)
  - **SIGGRAPH**
    - ***Break-A-Scene:*** Extracting Multiple Concepts from a Single Image [[Paper]](https://arxiv.org/pdf/2305.16311.pdf) [[Code]](https://github.com/google/break-a-scene)
    - Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models [[Paper]](https://arxiv.org/pdf/2302.12228.pdf) [[Project]](https://tuning-encoder.github.io/)
    - ***LayerDiffusion:*** Layered Controlled Image Editing with Diffusion Models [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3610543.3626172)
  - **arXiv**
    - ***DreamTuner:*** Single Image is Enough for Subject-Driven Generation [[Paper]](https://arxiv.org/pdf/2312.13691.pdf) [[Project]](https://dreamtuner-diffusion.github.io/)
    - ***PhotoMaker:*** Customizing Realistic Human Photos via Stacked ID Embedding [[Paper]](https://arxiv.org/pdf/2312.04461.pdf) [[Code]](https://github.com/TencentARC/PhotoMaker)
    - ***IP-Adapter:*** Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2308.06721.pdf) [[Code]](https://github.com/tencent-ailab/IP-Adapter) [[Project]](https://ip-adapter.github.io/)
    - ***FastComposer:*** Tuning-Free Multi-Subject Image Generation with Localized Attention [[Paper]](https://arxiv.org/pdf/2305.10431.pdf) [[Code]](https://github.com/mit-han-lab/fastcomposer)

[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
## Text-Guided Image Editing
- <span id="editing-year-2025">**Year 2025**</span>
  - **CVPR**
    - ***FDS:*** Frequency-Aware Denoising Score for Text-Guided Latent Diffusion Image Editing [[Paper]](https://arxiv.org/pdf/2503.19191)
    - Reference-Based 3D-Aware Image Editing with Triplanes [[Paper]](https://arxiv.org/pdf/2404.03632)
    - ***MoEdit:*** On Learning Quantity Perception for Multi-object Image Editing [[Paper]](https://arxiv.org/pdf/2503.10112)
    - Instruction-based Image Manipulation by Watching How Things Move [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Cao_Instruction-based_Image_Manipulation_by_Watching_How_Things_Move_CVPR_2025_paper.pdf)
    - ***h-Edit:*** Effective and Flexible Diffusion-Based Editing via Doob's h-Transform [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_h-Edit_Effective_and_Flexible_Diffusion-Based_Editing_via_Doobs_h-Transform_CVPR_2025_paper.pdf)
    - Identity-preserving Distillation Sampling by Fixed-Point Iterator [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Identity-preserving_Distillation_Sampling_by_Fixed-Point_Iterator_CVPR_2025_paper.pdf)
    - ***FluxSpace:*** Disentangled Semantic Editing in Rectified Flow Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Dalva_FluxSpace_Disentangled_Semantic_Editing_in_Rectified_Flow_Models_CVPR_2025_paper.pdf)
    - ***EmoEdit:*** Evoking Emotions through Image Manipulation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_EmoEdit_Evoking_Emotions_through_Image_Manipulation_CVPR_2025_paper.pdf)
    - ***ATA:*** Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Inpainting [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_ATA_Adaptive_Transformation_Agent_for_Text-Guided_Subject-Position_Variable_Background_Inpainting_CVPR_2025_paper.pdf)
    - Text-Driven Fashion Image Editing with Compositional Concept Learning and Counterfactual Abduction [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Text-Driven_Fashion_Image_Editing_with_Compositional_Concept_Learning_and_Counterfactual_CVPR_2025_paper.pdf)
    - Preserve or Modify? Context-Aware Evaluation for Balancing Preservation and Modification in Text-Guided Image Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Preserve_or_Modify_Context-Aware_Evaluation_for_Balancing_Preservation_and_Modification_CVPR_2025_paper.pdf)
    - ***SwiftEdit:*** Lightning Fast Text-Guided Image Editing via One-Step Diffusion [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf)
    - ***RealEdit:*** Reddit Edits As a Large-scale Empirical Dataset for Image Transformations [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Sushko_RealEdit_Reddit_Edits_As_a_Large-scale_Empirical_Dataset_for_Image_CVPR_2025_paper.pdf)
    - ***Instruct-CLIP:*** Improving Instruction-Guided Image Editing with Automated Data Refinement Using Contrastive Learning [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Instruct-CLIP_Improving_Instruction-Guided_Image_Editing_with_Automated_Data_Refinement_Using_CVPR_2025_paper.pdf)
    - ***Concept Lancet:*** Image Editing with Compositional Representation Transplant [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Concept_Lancet_Image_Editing_with_Compositional_Representation_Transplant_CVPR_2025_paper.pdf)
    - ***TurboFill:*** Adapting Few-step Text-to-image Model for Fast Image Inpainting [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_TurboFill_Adapting_Few-step_Text-to-image_Model_for_Fast_Image_Inpainting_CVPR_2025_paper.pdf)
    - ***InsightEdit:*** Towards Better Instruction Following for Image Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_InsightEdit_Towards_Better_Instruction_Following_for_Image_Editing_CVPR_2025_paper.pdf)
    - ***Style-Editor:*** Text-driven Object-centric Style Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_Style-Editor_Text-driven_Object-centric_Style_Editing_CVPR_2025_paper.pdf)
    - ***MagicQuill:*** An Intelligent Interactive Image Editing System [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_MagicQuill_An_Intelligent_Interactive_Image_Editing_System_CVPR_2025_paper.pdf)
    - Pathways on the Image Manifold: Image Editing via Video Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Rotstein_Pathways_on_the_Image_Manifold_Image_Editing_via_Video_Generation_CVPR_2025_paper.pdf)
    - ***Stable Flow:*** Vital Layers for Training-Free Image Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Avrahami_Stable_Flow_Vital_Layers_for_Training-Free_Image_Editing_CVPR_2025_paper.pdf)
    - Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Unveil_Inversion_and_Invariance_in_Flow_Transformer_for_Versatile_Image_CVPR_2025_paper.pdf)
    - ***AnyEdit:*** Mastering Unified High-Quality Image Editing for Any Idea [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_AnyEdit_Mastering_Unified_High-Quality_Image_Editing_for_Any_Idea_CVPR_2025_paper.pdf)
    - Improving Editability in Image Generation with Layer-wise Memory [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Improving_Editability_in_Image_Generation_with_Layer-wise_Memory_CVPR_2025_paper.pdf)
    - ***PhyS-EdiT:*** Physics-aware Semantic Image Editing with Text Description [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Cai_PhyS-EdiT_Physics-aware_Semantic_Image_Editing_with_Text_Description_CVPR_2025_paper.pdf)
    - Unleashing In-context Learning of Autoregressive Models for Few-shot Image Manipulation [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Lai_Unleashing_In-context_Learning_of_Autoregressive_Models_for_Few-shot_Image_Manipulation_CVPR_2025_paper.pdf)
    - ***MTADiffusion:*** Mask Text Alignment Diffusion Model for Object Inpainting [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_MTADiffusion_Mask_Text_Alignment_Diffusion_Model_for_Object_Inpainting_CVPR_2025_paper.pdf)
    - ***FireEdit:*** Fine-grained Instruction-based Image Editing via Region-aware Vision Language Model [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_FireEdit_Fine-grained_Instruction-based_Image_Editing_via_Region-aware_Vision_Language_Model_CVPR_2025_paper.pdf)
    - Towards Scalable Human-aligned Benchmark for Text-guided Image Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Ryu_Towards_Scalable_Human-aligned_Benchmark_for_Text-guided_Image_Editing_CVPR_2025_paper.pdf)
    - ⚠️ FeedEdit: Text-Based Image Editing with Dynamic Feedback Regulation [Paper]
  - **ICCV**
    - ***In-Context Edit:*** Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer [[Paper]](https://arxiv.org/abs/2504.20690) [[Project]](https://river-zhang.github.io/ICEdit-gh-pages/) [[Code]](https://github.com/River-Zhang/ICEdit?tab=readme-ov-file)
    - ***Dual‑Conditional Inversion:*** for Boosting Diffusion‑Based Image Editing [[Paper]](https://arxiv.org/abs/2506.02560)
    - ***CAMILA:*** Context‑Aware Masking for Image Editing with Language Alignment [[Paper]](https://neurips.cc/virtual/2025/poster/119101)
    - ***EditInfinity:*** Image Editing with Binary‑Quantized Generative Models [[Paper]](https://neurips.cc/virtual/2025/poster/115392)
    - ***KRIS‑Bench:*** Benchmarking Knowledge‑Based Reasoning in Image Editing Systems [[Paper]](https://arxiv.org/abs/2505.16707) [[Project]](https://yongliang-wu.github.io/kris_bench_project_page/) [[Code]](https://github.com/mercurystraw/Kris_Bench)
    - ***LoongX:*** Neural-Driven Image Editing [[Paper]](https://arxiv.org/abs/2507.05397) [[Project]](https://loongx1.github.io) [[Code]](https://github.com/LanceZPF/loongx)
    - ***CREA:*** CREA: A Collaborative Multi‑Agent Framework for Creative Image Editing and Generation [[Paper]](https://arxiv.org/abs/2504.05306) [[Project]](https://crea-diffusion.github.io)
    - ***IEAP:*** Image Editing As Programs with Diffusion Models [[Paper]](https://arxiv.org/abs/2506.04158) [[Project]](https://yujiahu1109.github.io/IEAP/) [[Code]](https://github.com/YujiaHu1109/IEAP)
    - ***ArtEditor:*** Learning Customized Instructional Image Editor from Few-Shot Examples [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_ArtEditor_Learning_Customized_Instructional_Image_Editor_from_Few-Shot_Examples_ICCV_2025_paper.pdf)
    - ***ADIEE:*** Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation [[Paper]](https://arxiv.org/abs/2507.07317)
    - ***Describe, Don't Dictate:*** Semantic Image Editing with Natural Language Intent [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Ci_Describe_Dont_Dictate_Semantic_Image_Editing_with_Natural_Language_Intent_ICCV_2025_paper.pdf)
    - What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models [[Paper]](https://arxiv.org/abs/2505.20405)
    - Multi-turn Consistent Image Editing [[Paper]](https://arxiv.org/abs/2505.04320)
    - ***LUSD:*** Localized Update Score Distillation for Text-Guided Image Editing [[Paper]](https://arxiv.org/abs/2503.11054)
    - ***QK-Edit:*** Revisiting Attention-based Injection in MM-DiT for Image and Video Editing [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Shen_QK-Edit_Revisiting_Attention-based_Injection_in_MM-DiT_for_Image_and_Video_ICCV_2025_paper.pdf)
    - ***RefEdit:*** A Benchmark and Method for Improving Instruction-based Image Editing Model on Referring Expressions [[Paper]](https://arxiv.org/abs/2506.03448)
    - ***ReFlex:*** Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation [[Paper]](https://arxiv.org/abs/2507.01496)
    - ***FreeFlux:*** Understanding and Exploiting Layer-Specific Roles in RoPE-Based MMDiT for Versatile Image Editing [[Paper]](https://arxiv.org/abs/2503.16153)
    - Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing [[Paper]](https://arxiv.org/abs/2508.07519)
    - ***Anchor Token Matching:*** Implicit Structure Locking for Training-free AR Image Editing [[Paper]](https://arxiv.org/abs/2504.10434)
    - Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing [[Paper]](https://arxiv.org/abs/2504.13490)
    - Addressing Text Embedding Leakage in Diffusion-based Image Editing [[Paper]](https://arxiv.org/abs/2412.04715)
    - ***EEdit:*** Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing [[Paper]](https://arxiv.org/abs/2503.10270)
    - ***SuperEdit:*** Rectifying and Facilitating Supervision for Instruction-Based Image Editing [[Paper]](https://arxiv.org/abs/2505.02370)
    - Frequency-Guided Diffusion for Training-Free Text-Driven Image Translation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_Frequency-Guided_Diffusion_for_Training-Free_Text-Driven_Image_Translation_ICCV_2025_paper.pdf)
    - Streamlining Image Editing with Layered Diffusion Brushes [[Paper]](https://arxiv.org/abs/2405.00313)
    - ***InstantEdit:*** Text-Guided Few-Step Image Editing with Piecewise Rectified Flow [[Paper]](https://arxiv.org/abs/2508.06033)
    - Training-Free Text-Guided Image Editing with Visual Autoregressive Model [[Paper]](https://arxiv.org/abs/2503.23897)
    - ***LOCATEdit:*** Graph Laplacian Optimized Cross Attention for Localized Text-Guided Image Editing [[Paper]](https://arxiv.org/abs/2503.21541)
    - ***FlowChef:*** Steering of Rectified Flow Models for Controlled Generations [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Patel_FlowChef_Steering_of_Rectified_Flow_Models_for_Controlled_Generations_ICCV_2025_paper.pdf)
    - ***PixTalk:*** Controlling Photorealistic Image Processing and Editing with Language [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Conde_PixTalk_Controlling_Photorealistic_Image_Processing_and_Editing_with_Language_ICCV_2025_paper.pdf)
    - ***Pose-Star:*** Anatomy-Aware Editing for Open-World Fashion Images [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Dong_Pose-Star_Anatomy-Aware_Editing_for_Open-World_Fashion_Images_ICCV_2025_paper.pdf)
    - ***UIP2P:*** Unsupervised Instruction-based Image Editing via Edit Reversibility Constraint [[Paper]](https://arxiv.org/abs/2412.15216)
    - ***Edicho:*** Consistent Image Editing in the Wild [[Paper]](https://arxiv.org/abs/2412.21079)
    - Instruction-based Image Editing with Planning, Reasoning, and Generation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Ji_Instruction-based_Image_Editing_with_Planning_Reasoning_and_Generation_ICCV_2025_paper.pdf)
    - ***KV-Edit:*** Training-Free Image Editing for Precise Background Preservation [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_KV-Edit_Training-Free_Image_Editing_for_Precise_Background_Preservation_ICCV_2025_paper.pdf)
  - **ICLR**
    - Lightning-Fast Image Inversion and Editing for Text-to-Image Diffusion Models [[Paper]](https://openreview.net/forum?id=t9l63huPRt)
    - Multi-Reward as Condition for Instruction-based Image Editing [[Paper]](https://openreview.net/forum?id=9RFocgIccP) [[Code]](https://github.com/bytedance/Multi-Reward-Editing)
    - ***HQ-Edit:*** A High-Quality Dataset for Instruction-based Image Editing [[Paper]](https://openreview.net/forum?id=mZptYYttFj) [[Dataset]](https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit) [[Code]](https://github.com/UCSC-VLAA/HQ-Edit)
    - ***CLIPDrag:*** Combining Text-based and Drag-based Instructions for Image Editing [[Paper]](https://openreview.net/forum?id=2HjRezQ1nj) [[Code]](https://github.com/HKUST-LongGroup/CLIPDrag)
    - Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations [[Paper]](https://openreview.net/forum?id=Hu0FSOSEyS) [[Project]](https://rf-inversion.github.io/) [[Code]](https://github.com/LituRout/RF-Inversion)
    - ***PostEdit:*** Posterior Sampling for Efficient Zero-Shot Image Editing [[Paper]](https://openreview.net/forum?id=J8YWCBPgx7) [[Code]](https://github.com/TFNTF/PostEdit)
    - ***OmniEdit:*** Building Image Editing Generalist Models Through Specialist Supervision [[Paper]](https://openreview.net/forum?id=Hlm0cga0sv) [[Project]](https://tiger-ai-lab.github.io/OmniEdit/) [[Code]](https://github.com/TIGER-AI-Lab/OmniEdit) [[Dataset]](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M)
    - ***InstantPortrait:*** One-Step Portrait Editing via Diffusion Multi-Objective Distillation [[Paper]](https://openreview.net/forum?id=ZkFMe3OPfw)
    - Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing Consistent Light Transport [[Paper]](https://openreview.net/forum?id=u1cQYxRI1H)
    - ***InstantSwap:*** Fast Customized Concept Swapping across Sharp Shape Differences [[Paper]](https://openreview.net/forum?id=UFrHWzZENz)
    - ***Add-it:*** Training-Free Object Insertion in Images With Pretrained Diffusion Models [[Paper]](https://openreview.net/forum?id=ZeaTvXw080)
    - ***PixWizard:*** Versatile Image-to-Image Visual Assistant with Open-Language Instructions [[Paper]](https://openreview.net/forum?id=xuQSp75HmP)
  - **SIGGRAPH**
    - ***Cora:*** Correspondence-aware Image Editing Using Few-Step Diffusion [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3721238.3730650) [[Project]](https://cora-edit.github.io/) [[Code]](https://github.com/alimohammadiamirhossein/cora)
    - ***PartEdit:*** Fine-Grained Image Editing using Pre-Trained Diffusion Models [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3721238.3730747)
  - **ICML**
    - Highly Compressed Tokenizer Can Generate Without Training [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/beyer25a/beyer25a.pdf) [[Code]](https://github.com/lukaslaobeyer/token-opt)
    - ***FireFlow:*** Fast Inversion of Rectified Flow for Image Semantic Editing [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/deng25c/deng25c.pdf) [[Code]](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing)
    - ***Editable Noise Map Inversion:*** Encoding Target-image into Noise For High-Fidelity Image Manipulation [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/kang25b/kang25b.pdf)
    - ***Balancing Preservation and Modification:*** A Region and Semantic Aware Metric for Instruction-Based Image Editing [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25dh/li25dh.pdf) [[Project]](https://joyli-x.github.io/BPM/)
    - Taming Rectified Flow for Inversion and Editing [[Paper]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25ce/wang25ce.pdf) [[Code]](https://github.com/wangjiangshan0725/RF-Solver-Edit)
  - **AAAI**
    - ***Text2Relight:*** Creative Portrait Relighting with Text Guidance [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32194)
    - ***DiT4Edit:*** Diffusion Transformer for Image Editing [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32304)
    - ***DesignEdit:*** Unify Spatial-Aware Image Editing via Training-free Inpainting with a Multi-Layered Latent Diffusion Framework [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32414)
    - ***Move and Act:*** Enhanced Object Manipulation and Background Integrity for Image Editing [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32423)
    - Energy-Guided Optimization for Personalized Image Editing with Pretrained Text-to-Image Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32424)
    - ***Click2Mask:*** Local Editing with Dynamic Mask Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32720)
    - ***FunEditor:*** Achieving Complex Image Edits via Function Aggregation with Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32725)
    - ***SigStyle:*** Signature Style Transfer via Personalized Text-to-Image Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32868)
    - Style Nursing with Spatial and Semantic Guidance for Zero-Shot Traffic Scene Style Transfer [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32886)
    - ***CA-Edit:*** Causality-Aware Condition Adapter for High-Fidelity Local Facial Attribute Editing [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32928)
    - ***VersaFusion:*** A Versatile Diffusion-Based Framework for Fine-Grained Image Editing and Enhancement [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33023)
    - Textualize Visual Prompt for Image Editing via Diffusion Bridge [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/35483)

- <span id="editing-year-2024">**Year 2024**</span>
  - **CVPR**
    - ***InfEdit:*** Inversion-Free Image Editing with Natural Language [[Paper]](https://arxiv.org/pdf/2312.04965.pdf) [[Code]](https://github.com/sled-group/InfEdit) [[Project]](https://sled-group.github.io/InfEdit/)
    - Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing [[Paper]](https://arxiv.org/pdf/2403.03431.pdf)
    - Doubly Abductive Counterfactual Inference for Text-based Image Editing [[Paper]](https://arxiv.org/pdf/2403.02981.pdf) [[Code]](https://github.com/xuesong39/DAC)
    - Focus on Your Instruction: Fine-grained and Multi-instruction Image Editing by Attention Modulation [[Paper]](https://arxiv.org/pdf/2312.10113.pdf) [[Code]](https://github.com/guoqincode/Focus-on-Your-Instruction)
    - Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing [[Paper]](https://arxiv.org/pdf/2311.18608.pdf)
    - ***DragDiffusion:*** Harnessing Diffusion Models for Interactive Point-based Image Editing [[Paper]](https://arxiv.org/pdf/2306.14435.pdf) [[Code]](https://github.com/Yujun-Shi/DragDiffusion)
    - ***DiffEditor:*** Boosting Accuracy and Flexibility on Diffusion-based Image Editing [[Paper]](https://arxiv.org/pdf/2402.02583.pdf)
    - ***FreeDrag:*** Feature Dragging for Reliable Point-based Image Editing [[Paper]](https://arxiv.org/pdf/2307.04684.pdf) [[Code]](https://github.com/LPengYang/FreeDrag)
    - Text-Driven Image Editing via Learnable Regions [[Paper]](https://arxiv.org/pdf/2311.16432.pdf) [[Code]](https://github.com/yuanze-lin/Learnable_Regions) [[Project]](https://yuanze-lin.me/LearnableRegions_page/) [[Video]](https://www.youtube.com/watch?v=FpMWRXFraK8&feature=youtu.be)
    - ***LEDITS++:*** Limitless Image Editing using Text-to-Image Models [[Paper]](https://arxiv.org/pdf/2311.16711.pdf) [[Code]](https://huggingface.co/spaces/editing-images/ledtisplusplus/tree/main) [[Project]](https://leditsplusplus-project.static.hf.space/index.html) [[Demo]](https://huggingface.co/spaces/editing-images/leditsplusplus)
    - ***SmartEdit:*** Exploring Complex Instruction-based Image Editing with Large Language Models [[Paper]](https://arxiv.org/pdf/2312.06739.pdf) [[Code]](https://github.com/TencentARC/SmartEdit) [[Project]](https://yuzhou914.github.io/SmartEdit/)
    - ***Edit One for All:*** Interactive Batch Image Editing [[Paper]](https://arxiv.org/pdf/2401.10219.pdf) [[Code]](https://github.com/thaoshibe/edit-one-for-all) [[Project]](https://thaoshibe.github.io/edit-one-for-all/)
    - ***DiffMorpher:*** Unleashing the Capability of Diffusion Models for Image Morphing [[Paper]](https://arxiv.org/pdf/2312.07409) [[Code]](https://github.com/Kevin-thu/DiffMorpher) [[Project]](https://kevin-thu.github.io/DiffMorpher_page/) [[Demo]](https://openxlab.org.cn/apps/detail/KaiwenZhang/DiffMorpher)
    - ***TiNO-Edit:*** Timestep and Noise Optimization for Robust Diffusion-Based Image Editing [[Paper]](https://arxiv.org/pdf/2404.11120) [[Code]](https://github.com/SherryXTChen/TiNO-Edit)
    - Person in Place: Generating Associative Skeleton-Guidance Maps for Human-Object Interaction Image Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Person_in_Place_Generating_Associative_Skeleton-Guidance_Maps_for_Human-Object_Interaction_CVPR_2024_paper.pdf) [[Project]](https://yangchanghee.github.io/Person-in-Place_page/) [[Code]](https://github.com/YangChangHee/CVPR2024_Person-In-Place_RELEASE)
    - Referring Image Editing: Object-level Image Editing via Referring Expressions [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Referring_Image_Editing_Object-level_Image_Editing_via_Referring_Expressions_CVPR_2024_paper.pdf)
    - Prompt Augmentation for Self-supervised Text-guided Image Manipulation [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Bodur_Prompt_Augmentation_for_Self-supervised_Text-guided_Image_Manipulation_CVPR_2024_paper.pdf)
    - The Devil is in the Details: StyleFeatureEditor for Detail-Rich StyleGAN Inversion and High Quality Image Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Bobkov_The_Devil_is_in_the_Details_StyleFeatureEditor_for_Detail-Rich_StyleGAN_CVPR_2024_paper.pdf) [[Code]](https://github.com/AIRI-Institute/StyleFeatureEditor)
  

  - **ECCV**
    - ***RegionDrag:*** Fast Region-Based Image Editing with Diffusion Models [[Paper]](http://arxiv.org/pdf/2407.18247v1) [[Code]](https://github.com/Visual-AI/RegionDrag) [[Project]](https://visual-ai.github.io/regiondrag/) [[Demo]](https://colab.research.google.com/drive/1pnq9t_1zZ8yL_Oba20eBLVZLp3glniBR?usp=sharing)
    - ***TurboEdit:*** Instant text-based image editing [[Paper]](https://arxiv.org/pdf/2408.08332v1) [[Project]](https://betterze.github.io/TurboEdit/)
    - ***InstructGIE:*** Towards Generalizable Image Editing [[Paper]](https://arxiv.org/pdf/2403.05018)
    - ***StableDrag:*** Stable Dragging for Point-based Image Editing [[Paper]](https://arxiv.org/pdf/2403.04437)
    - ***Eta Inversion:*** Designing an Optimal Eta Function for Diffusion-based Real Image Editing [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02157.pdf) [[Code]](https://github.com/furiosa-ai/eta-inversion) [[Project]](https://github.com/furiosa-ai/eta-inversion)
    - ***SwapAnything:*** Enabling Arbitrary Object Swapping in Personalized Image Editing [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04768.pdf) [[Code]](https://github.com/eric-ai-lab/swap-anything) [[Project]](https://swap-anything.github.io/)
    - ***Guide-and-Rescale:*** Self-Guidance Mechanism for Effective Tuning-Free Real Image Editing [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08987.pdf)
    - ***FreeDiff:*** Progressive Frequency Truncation for Image Editing with Diffusion Models [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00759.pdf) [[Code]](https://github.com/Thermal-Dynamics/FreeDiff)
    - Lazy Diffusion Transformer for Interactive Image Editing [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03436.pdf) [[Project]](https://lazydiffusion.github.io/)
    - ***ByteEdit:*** Boost, Comply and Accelerate Generative Image Editing [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00359.pdf) [[Project]](https://byte-edit.github.io/)
  - **ICLR**
    - Guiding Instruction-based Image Editing via Multimodal Large Language Models [[Paper]](https://arxiv.org/pdf/2309.17102.pdf) [[Code]](https://github.com/apple/ml-mgie) [[Project]](https://mllm-ie.github.io/)
    - The Blessing of Randomness: SDE Beats ODE in General Diffusion-based Image Editing [[Paper]](https://arxiv.org/pdf/2311.01410.pdf) [[Code]](https://github.com/ML-GSAI/SDE-Drag) [[Project]](https://ml-gsai.github.io/SDE-Drag-demo/)
    - ***Motion Guidance:*** Diffusion-Based Image Editing with Differentiable Motion Estimators [[Paper]](https://arxiv.org/pdf/2401.18085.pdf) [[Code]](https://github.com/dangeng/motion_guidance) [[Project]](https://dangeng.github.io/motion_guidance/)
    - Object-Aware Inversion and Reassembly for Image Editing [[Paper]](https://arxiv.org/pdf/2310.12149.pdf) [[Code]](https://github.com/aim-uofa/OIR) [[Project]](https://aim-uofa.github.io/OIR-Diffusion/)
    - ***Noise Map Guidance:*** Inversion with Spatial Context for Real Image Editing [[Paper]](https://arxiv.org/pdf/2402.04625.pdf)
  - **AAAI**
    - Tuning-Free Inversion-Enhanced Control for Consistent Image Editing [[Paper]](https://arxiv.org/pdf/2312.14611)
    - ***BARET:*** Balanced Attention based Real image Editing driven by Target-text Inversion [[Paper]](https://arxiv.org/pdf/2312.05482)
    - Accelerating Text-to-Image Editing via Cache-Enabled Sparse Diffusion Inference [[Paper]](https://arxiv.org/pdf/2305.17423)
    - High-Fidelity Diffusion-based Image Editing [[Paper]](https://arxiv.org/pdf/2312.15707)
    - ***AdapEdit:*** Spatio-Temporal Guided Adaptive Editing Algorithm for Text-Based Continuity-Sensitive Image Editing [[Paper]](https://arxiv.org/pdf/2312.08019)
    - ***TexFit:*** Text-Driven Fashion Image Editing with Diffusion Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28885)
  - **arXiv**
    - ***An Item is Worth a Prompt:*** Versatile Image Editing with Disentangled Control [[Paper]](https://arxiv.org/pdf/2403.04880.pdf) [[Code]](https://github.com/asFeng/d-edit)
    - One-Dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications [[Paper]](https://arxiv.org/pdf/2312.16145) [[Code]](https://github.com/Con6924/SPM) [[Project]](https://lyumengyao.github.io/projects/spm)
    - ***EditWorld:*** Simulating World Dynamics for Instruction-Following Image Editing [[Paper]](https://arxiv.org/pdf/2405.14785) [[Code]](https://github.com/YangLing0818/EditWorld) [[Project]](https://github.com/YangLing0818/EditWorld)
    - ***ReasonPix2Pix:*** Instruction Reasoning Dataset for Advanced Image Editing [[Paper]](https://arxiv.org/pdf/2405.11190)
    - ***FlowEdit:*** Inversion-Free Text-Based Editing Using Pre-Trained Flow Models [[Paper]](https://arxiv.org/pdf/2412.08629) [[Code]](https://github.com/fallenshock/FlowEdit) [[Project]](https://matankleiner.github.io/flowedit/) [[Demo]](https://huggingface.co/spaces/fallenshock/FlowEdit/)
- <span id="editing-year-2023">**Year 2023**</span>
  - **CVPR**
    - Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Uncovering_the_Disentanglement_Capability_in_Text-to-Image_Diffusion_Models_CVPR_2023_paper.pdf) [[Code]](https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement)
    - ***SINE:*** SINgle Image Editing with Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_SINE_SINgle_Image_Editing_With_Text-to-Image_Diffusion_Models_CVPR_2023_paper.pdf) [[Code]](https://github.com/zhang-zx/SINE)
    - ***Imagic:*** Text-Based Real Image Editing with Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kawar_Imagic_Text-Based_Real_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf)
    - ***InstructPix2Pix:*** Learning to Follow Image Editing Instructions [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf) [[Code]](https://github.com/timothybrooks/instruct-pix2pix) [[Dataset]](https://instruct-pix2pix.eecs.berkeley.edu/) [[Project]](https://www.timothybrooks.com/instruct-pix2pix/) [[Demo]](https://huggingface.co/spaces/timbrooks/instruct-pix2pix)
    - ***Null-text Inversion*** for Editing Real Images using Guided Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Mokady_NULL-Text_Inversion_for_Editing_Real_Images_Using_Guided_Diffusion_Models_CVPR_2023_paper.pdf) [[Code]](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images)
  - **ICCV**
    - ***MasaCtrl:*** Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_MasaCtrl_Tuning-Free_Mutual_Self-Attention_Control_for_Consistent_Image_Synthesis_and_ICCV_2023_paper.pdf) [[Code]](https://github.com/TencentARC/MasaCtrl) [[Project]](https://ljzycmd.github.io/projects/MasaCtrl/) [[Demo]](https://colab.research.google.com/drive/1DZeQn2WvRBsNg4feS1bJrwWnIzw1zLJq?usp=sharing)
    - Localizing Object-level Shape Variations with Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Patashnik_Localizing_Object-Level_Shape_Variations_with_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf) [[Code]](https://github.com/orpatashnik/local-prompt-mixing) [[Project]](https://orpatashnik.github.io/local-prompt-mixing/) [[Demo]](https://huggingface.co/spaces/orpatashnik/local-prompt-mixing)
  - **ICLR**
    - ***SDEdit:*** Guided Image Synthesis and Editing with Stochastic Differential Equations [[Paper]](https://arxiv.org/pdf/2108.01073.pdf) [[Code]](https://github.com/ermongroup/SDEdit) [[Project]](https://sde-image-editing.github.io/)
- <span id="editing-year-2022">**Year 2022**</span>
  - **CVPR**
    - ***DiffusionCLIP:*** Text-Guided Diffusion Models for Robust Image Manipulation [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.pdf) [[Code]](https://github.com/gwang-kim/DiffusionCLIP)

[<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
## Text Image Generation
- <span id="gentext-year-2025">**Year 2025**</span>
  - **ICCV**
    - ***TextMaster:*** A Unified Framework for Realistic Text Editing via Glyph-Style Dual-Control [[Paper]](https://arxiv.org/abs/2410.09879)
    - ***FonTS:*** Text Rendering With Typography and Style Controls [[Paper]](https://arxiv.org/abs/2412.00136)
  - **ICLR**
    - Precise Parameter Localization for Textual Generation in Diffusion Models [[Paper]](https://openreview.net/forum?id=gdHtZlaaSo)
  - **AAAI**
    - ***HDLayout:*** Hierarchical and Directional Layout Planning for Arbitrary Shaped Visual Text Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32307)
    - ***GlyphDraw2:*** Automatic Generation of Complex Glyph Posters with Diffusion Models and Large Language Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32636)
- <span id="gentext-year-2024">**Year 2024**</span>
  - **arXiv**
    - ***AnyText:*** Multilingual Visual Text Generation And Editing [[Paper]](https://arxiv.org/pdf/2311.03054.pdf) [[Code]](https://github.com/tyxsspa/AnyText) [[Project]](https://anytext.pics/)
  - **CVPR**
    - ***SceneTextGen:*** Layout-Agnostic Scene Text Image Synthesis with Integrated Character-Level Diffusion and Contextual Consistency [[Paper]](https://arxiv.org/pdf/2406.01062v2)

[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
# Datasets
- ***Microsoft COCO:*** Common Objects in Context [[Paper]](https://arxiv.org/pdf/1405.0312.pdf) [[Dataset]](https://cocodataset.org/#home)
- ***Conceptual Captions:*** A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning [[Paper]](https://aclanthology.org/P18-1238.pdf) [[Dataset]](https://ai.google.com/research/ConceptualCaptions/)
- ***LAION-5B:*** An Open Large-Scale Dataset for Training Next Generation Image-Text Models [[Paper]](https://openreview.net/pdf?id=M3Y74vmsMcY) [[Dataset]](https://laion.ai/)
- ***PartiPrompts:*** Scaling Autoregressive Models for Content-Rich Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2206.10789) [[Dataset]](https://github.com/google-research/parti?tab=readme-ov-file) [[Project]](https://sites.research.google/parti/)
- ***ROVI:*** A VLM-LLM Re-Captioned Dataset for Open-Vocabulary Instance-Grounded Text-to-Image Generation [[Paper]](https://arxiv.org/abs/2508.01008)


[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
# Toolkits
|Name|Website|Description|
|-|-|-|
|Stable Diffusion WebUI|[link](https://github.com/AUTOMATIC1111/stable-diffusion-webui)|Built based on Gradio, deployed locally to run Stable Diffusion checkpoints, LoRA weights, ControlNet weights, etc.|
|Stable Diffusion WebUI-forge|[link](https://github.com/lllyasviel/stable-diffusion-webui-forge)|Built based on Gradio, deployed locally to run Stable Diffusion checkpoints, LoRA weights, ControlNet weights, etc.|
|Fooocus|[link](https://github.com/lllyasviel/Fooocus)|Built based on Gradio, offline, open source, and free.  <br />The manual tweaking is not needed, and users only need to focus on the prompts and images.|
|ComfyUI|[link](https://github.com/comfyanonymous/ComfyUI)|Deployed locally to enable customized workflows with Stable Diffusion|
|Civitai|[link](https://civitai.com/)|Websites for community Stable Diffusion and LoRA checkpoints|

[<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
# Q&A
- **Q: The conference sequence of this paper list?**
  - This paper list is organized according to the following sequence:
    - CVPR
    - ICCV
    - ECCV
    - WACV
    - NeurIPS
    - ICLR
    - ICML
    - ACM MM
    - SIGGRAPH
    - AAAI
    - arXiv
    - Others
- **Q: What does `Others` refers to?**
  - Some of the following studies (e.g., `Stable Casacade`) does not publish their technical report on arXiv. Instead, they tend to write a blog in their official websites. The `Others` category refers to such kind of studies.

[<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
# References

The `reference.bib` file summarizes bibtex references of up-to-date image inpainting papers, widely used datasets, and toolkits.
Based on the original references, I have made the following modifications to make their results look nice in the `LaTeX` manuscripts:
- Refereces are normally constructed in the form of `author-etal-year-nickname`. Particularly, references of datasets and toolkits are directly constructed as `nickname`, e.g., `imagenet`.
- In each reference, all names of conferences/journals are converted into abbreviations, e.g., `Computer Vision and Pattern Recognition -> CVPR`.
- The `url`, `doi`, `publisher`, `organization`, `editor`, `series` in all references are removed.
- The `pages` of all references are added if they are missing.
- All paper names are in title case. Besides, I have added an additional `{}` to make sure that the title case would also work well in some particular templates. 

If you have other demands of reference formats, you may refer to the original references of papers by searching their names in [DBLP](https://dblp.org/) or [Google Scholar](https://scholar.google.com/).

> [!NOTE]
> Note that references in the `homepage` and [the `topic` section](topics/topics.md) can be repeated in `reference.bib`. Personally, I recommend using `"Ctrl+F" / "Command+F"` to search your desired `BibTeX` reference.

 [<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
# Star History

<p align="center">
    <a href="https://api.star-history.com/svg?repos=AlonzoLeeeooo/awesome-text-to-image-studies&type=Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=AlonzoLeeeooo/awesome-text-to-image-studies&type=Date" alt="Star History Chart">
    </a>
<p>

[<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
# WeChat Group

<div align="center">
  <a href="YOUR_OFFICIAL_WEBSITE_URL">
    <img src="github-materials/group.png" alt="group">
  </a>
</div>

[<u><small><🎯Back to Top></small></u>](#contents)
