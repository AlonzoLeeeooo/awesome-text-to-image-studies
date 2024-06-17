<p align="center">
  <h1 align="center">A Collection of Text-to-Image Generation Studies</h1>

This GitHub repository summarizes papers and resources related to the text-to-image (T2I) generation task. 

> [!NOTE]
> This document serves as the `homepage` of the whole GitHub repo. Papers are summarized according to **different research directions, published years, and conferences.** 
> 
> [The `topics` section](topics/topics.md) summarizes papers that are highly related to T2I generation according to different properties, e.g., prerequisites of T2I generation, diffusion models with other techniques (e.g., Diffusion Transformer, LLMs, Mamba, etc.), and diffusion models for other tasks. 

If you have any suggestions about this repository, please feel free to [start a new issue](https://github.com/AlonzoLeeeooo/awesome-text-to-image-generation-studies/issues/new) or [pull requests](https://github.com/AlonzoLeeeooo/awesome-text-to-image-generation-studies/pulls).

Recent news of this GitHub repo are listed as follows.
<details> <summary> 🔥 Click to see more information. </summary>

- [Apr. 26th] Update a new topic: **Diffusion Models Meet Federated Learning.** See [the `topics` section](topics/topics.md) for more details!
- [Mar. 28th] The official **AAAI 2024** paper list are released! Official version of PDFs and BibTeX references are updated accordingly.
- [Mar. 21th] [The `topics` section](topics/topics.md) has been updated. This section aims to offer **paper lists that are summarized according to other properties of diffusion models**, e.g., Diffusion Transformer-based methods, diffusion models for NLP, diffusion models integrated with LLMs, etc. The corresponding references of these papers are also concluded in `reference.bib`.
- [Mar. 7th] All available **CVPR, ICLR, and AAAI 2024 papers and references** are updated. Papers highlighted with ⚠️ will be updated as soon as their metadata is avaiable.
- [Mar. 1st] Websites of [**the off-the-shelf text-to-image generation products**](#available-products) and [**toolkits**](#toolkits) are summarized.

</details>


<!-- omit in toc -->
# <span id="contents">Contents</span>
- [Products](#available-products)
- [To-Do Lists](#to-do-lists)
- [Papers](#papers)
  - [Survey Papers](#survey-papers)
  - [Text-to-Image Generation](#text-to-image-generation)
    - [Year 2024](#text-year-2024)
    - [Year 2023](#text-year-2023)
    - [Year 2022](#text-year-2022)
    - [Year 2021](#text-year-2021)
    - [Year 2020](#text-year-2020)
  - [Conditional Text-to-Image Generation](#conditional-text-to-image-generation)
    - [Year 2024](#conditional-year-2024)
    - [Year 2023](#conditional-year-2023)
    - [Year 2022](#conditional-year-2022)
  - [Personalized Text-to-Image Generation](#personalized-text-to-image-generation)
    - [Year 2024](#personalized-year-2024)
    - [Year 2023](#personalized-year-2023)
  - [Text-Guided Image Editing](#text-guided-image-editing) 
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


<!-- omit in toc -->
# To-Do Lists
- Published Papers on Conferences
  - [x] Update CVPR 2024 Papers
    - [ ] Update ⚠️ Papers and References
    - [ ] Update arXiv References into the Official Version
  - [x] Update AAAI 2024 Papers
    - [x] Update ⚠️ Papers and References
    - [x] Update arXiv References into the Official Version
  - [x] Update ICLR 2024 Papers
  - [ ] Update NeurIPS 2023 Papers
- Regular Maintenance of Preprint arXiv Papers and Missed Papers

[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
# Products
|Name|Year|Website|Specialties|
|-|-|-|-|
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
    - ***PixArt-Σ:*** Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2403.04692) [[Code]](https://github.com/PixArt-alpha/PixArt-sigma) [[Project]](https://pixart-alpha.github.io/PixArt-sigma-project/)
    - ***CogView3:*** Finer and Faster Text-to-Image Generation via Relay Diffusion [[Paper]](https://arxiv.org/pdf/2403.05121) [[Code]](https://github.com/THUDM/CogView)
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
  - **ICCV**
    - ***DiffFit:*** Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.pdf) [[Code]](https://github.com/mkshing/DiffFit-pytorch) [[Demo]](https://colab.research.google.com/github/mkshing/difffit-pytorch/blob/main/scripts/difffit_pytorch.ipynb)
  - **NeurIPS**
    - ***ImageReward:*** Learning and Evaluating Human Preferences for Text-to-Image Generation [[Paper]](https://openreview.net/pdf?id=JVzeOYEx6d) [[Code]](https://github.com/THUDM/ImageReward)
    - ***RAPHAEL***: Text-to-Image Generation via Large Mixture of Diffusion Paths [[Paper]](https://arxiv.org/pdf/2305.18295) [[Project]](https://raphael-painter.github.io/)
    - Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment [[Paper]](https://openreview.net/pdf?id=AOKU4nRw1W) [[Code]](https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models)
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
    - ⚠️ Countering Personalized Text-to-Image Generation with Influence Watermarks [Paper]
  - **AAAI**
    - Decoupled Textual Embeddings for Customized Image Generation [[Paper]](https://arxiv.org/pdf/2312.11826.pdf)
  - **arXiv**
    - ***FlashFace:*** Human Image Personalization with High-fidelity Identity Preservation [[Paper]](https://arxiv.org/pdf/2403.17008) [[Code]](https://github.com/jshilong/FlashFace) [[Project]](https://jshilong.github.io/flashface-page)
    - ***MoMA:*** Multimodal LLM Adapter for Fast Personalized Image Generation [[Paper]](https://arxiv.org/pdf/2404.05674)
    - ***IDAdapter:*** Learning Mixed Features for Tuning-Free Personalization of Text-to-Image Models [[Paper]](https://arxiv.org/pdf/2403.13535)
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
    - Person in Place: Generating Associative Skeleton-Guidance Maps for Human-Object Interaction Image Editing [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Person_in_Place_Generating_Associative_Skeleton-Guidance_Maps_for_Human-Object_Interaction_CVPR_2024_paper.pdf) [[Project]](https://yangchanghee.github.io/Person-in-Place_page/)
    - Referring Image Editing: Object-level Image Editing via Referring Expressions [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Referring_Image_Editing_Object-level_Image_Editing_via_Referring_Expressions_CVPR_2024_paper.pdf)
    - Prompt Augmentation for Self-supervised Text-guided Image Manipulation [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Bodur_Prompt_Augmentation_for_Self-supervised_Text-guided_Image_Manipulation_CVPR_2024_paper.pdf)
    - ⚠️ The Devil is in the Details: StyleFeatureEditor for Detail-Rich StyleGAN Inversion and High Quality Image Editing [Paper]
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
    - ***StableDrag:*** Stable Dragging for Point-based Image Editing [[Paper]](https://arxiv.org/pdf/2403.04437)
    - One-Dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications [[Paper]](https://arxiv.org/pdf/2312.16145) [[Code]](https://github.com/Con6924/SPM) [[Project]](https://lyumengyao.github.io/projects/spm)
    - ***EditWorld:*** Simulating World Dynamics for Instruction-Following Image Editing [[Paper]](https://arxiv.org/pdf/2405.14785) [[Code]](https://github.com/YangLing0818/EditWorld) [[Project]](https://github.com/YangLing0818/EditWorld)
    - ***ReasonPix2Pix:*** Instruction Reasoning Dataset for Advanced Image Editing [[Paper]](https://arxiv.org/pdf/2405.11190)
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
