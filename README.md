<p align="center">
  <h1 align="center">A Collection of Text-to-Image Generation Studies</h1>

This GitHub repository summarizes papers and resources related to the text-to-image generation task. 

If you have any suggestions about this repository, please feel free to [start a new issue](https://github.com/AlonzoLeeeooo/awesome-text-to-image-generation-studies/issues/new) or [pull requests](https://github.com/AlonzoLeeeooo/awesome-text-to-image-generation-studies/pulls).


<!-- omit in toc -->
# <span id="contents">Contents</span>
- [References](#references)
- [Papers](#papers)
  - [Text-to-Image Generation](#text-to-image-generation)
    - [Year 2024](#text-year-2024)
    - [Year 2023](#text-year-2023)
    - [Year 2022](#text-year-2022)
    - [Year 2021](#text-year-2021)
  - [Conditional Text-to-Image Generation](#conditional-text-to-image-generation)
    - [Year 2024](#conditional-year-2024)
    - [Year 2023](#conditional-year-2023)
    - [Year 2022](#conditional-year-2022)
  - [Personalized Text-to-Image Generation](#personalized-text-to-image-generation)
    - [Year 2023](#attribute-year-2023)
    - [Year 2022](#attribute-year-2022)
  - [Text-Guided Image Manipulation](#text-guided-image-manipulation) 
    - [Year 2023](#manipulation-year-2023)
    - [Year 2022](#manipulation-year-2022)
- [Datasets](#datasets)
- [To-Do Lists](#to-do-lists)


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

 [<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
# Papers

<!-- omit in toc -->
## Text-to-Image Generation
- <span id="text-year-2024">**Year 2024**</span>
  - **arXiv**
    - Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation [[Paper]](https://arxiv.org/pdf/2402.10210.pdf)
    - ***RPG:*** Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs [[Paper]](https://arxiv.org/pdf/2401.11708.pdf) [[Code]](https://github.com/YangLing0818/RPG-DiffusionMaster)
  - **Others**
    - **Stable Cascade** [[Code]](https://github.com/Stability-AI/StableCascade) [[Project]](https://stability.ai/news/introducing-stable-cascade)

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
  - **NeurIPS**
    - ***ImageReward:*** Learning and Evaluating Human Preferences for Text-to-Image Generation [[Paper]](https://openreview.net/pdf?id=JVzeOYEx6d) [[Code]](https://github.com/THUDM/ImageReward)
  - **ICML**
    - ***StyleGAN-T:*** Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis [[Paper]](https://proceedings.mlr.press/v202/sauer23a/sauer23a.pdf) [[Code]](https://github.com/autonomousvision/stylegan-t) [[Project]](https://sites.google.com/view/stylegan-t/) [[Video]](https://www.youtube.com/watch?v=MMj8OTOUIok)
    - ***Muse:*** Text-To-Image Generation via Masked Generative Transformers [[Paper]](https://proceedings.mlr.press/v202/chang23b/chang23b.pdf) [[Reproduced Code]](https://github.com/lucidrains/muse-maskgit-pytorch) [[Project]](https://muse-icml.github.io/)
  - **ACM MM**
    - ***SUR-adapter:*** Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models [[Paper]](https://arxiv.org/pdf/2305.05189.pdf) [[Code]](https://github.com/Qrange-group/SUR-adapter)
  - **SIGGRAPH**
    - ***Attend-and-Excite:*** Attention-Based Semantic Guidance for Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2301.13826.pdf) [[Code]](https://github.com/yuval-alaluf/Attend-and-Excite) [[Project]](https://yuval-alaluf.github.io/Attend-and-Excite/) [[Demo]](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite)
  - **arXiv**
    - ***P+:*** Extended Textual Conditioning in Text-to-Image Generation [[Paper]](https://prompt-plus.github.io/files/PromptPlus.pdf)
    - ***SDXL:*** Improving Latent Diffusion Models for High-Resolution Image Synthesis [[Paper]](https://arxiv.org/pdf/2307.01952.pdf) [[Code]](https://github.com/Stability-AI/generative-models)
    - ***SDXL-Turbo:*** Adversarial Diffusion Distillation [[Paper]](https://arxiv.org/pdf/2311.17042.pdf) [[Code]](https://github.com/Stability-AI/generative-models)
    - ***Wuerstchen:*** An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2306.00637.pdf) [[Code]](https://github.com/dome272/Wuerstchen)
    - ***StreamDiffusion:*** A Pipeline-level Solution for Real-time Interactive Generation [[Paper]](https://arxiv.org/pdf/2312.12491.pdf) [[Project]](https://github.com/cumulo-autumn/StreamDiffusion)
  - **Others**
    - **DALLE-3:** Improving Image Generation with Better Captions [[Paper]](https://cdn.openai.com/papers/dall-e-3.pdf)

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
  - **arXiv**
    - ***DALLE-2:*** Hierarchical Text-Conditional Image Generation with CLIP Latents [[Paper]](https://cdn.openai.com/papers/dall-e-2.pdf)
    - ***PITI:*** Pretraining is All You Need for Image-to-Image Translation [[Paper]](https://arxiv.org/pdf/2205.12952.pdf) [[Code]](https://github.com/PITI-Synthesis/PITI)

[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="text-year-2021">**Year 2021**</span>
  - **NeurIPS**
    - ***CogView:*** Mastering Text-to-Image Generation via Transformers [[Paper]](https://proceedings.neurips.cc/paper/2021/file/a4d92e2cd541fca87e4620aba658316d-Paper.pdf) [[Code]](https://github.com/THUDM/CogView) [[Demo]](https://thudm.github.io/CogView/index.html)
  - **ICML**
    - ***DALLE-1:*** Zero-Shot Text-to-Image Generation [[Paper]](https://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf) [[Reproduced Code]](https://github.com/lucidrains/DALLE-pytorch)
  - **ICCV**
    -  ***DAE-GAN:*** Dynamic Aspect-aware GAN for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Ruan_DAE-GAN_Dynamic_Aspect-Aware_GAN_for_Text-to-Image_Synthesis_ICCV_2021_paper.pdf) [[Code]](https://github.com/hiarsal/DAE-GAN)



<!-- omit in toc -->
## Conditional Text-to-Image Generation
- <span id="conditional-year-2024">**Year 2024**</span>
  - **WACV**
    - Training-Free Layout Control with Cross-Attention Guidance [[Paper]](https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Training-Free_Layout_Control_With_Cross-Attention_Guidance_WACV_2024_paper.pdf) [[Code]](https://github.com/silent-chen/layout-guidance) [[Project]](https://silent-chen.github.io/layout-guidance/) [[Demo]](https://huggingface.co/spaces/silentchen/layout-guidance)

[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="conditional-year-2023">**Year 2023**</span>
  - **CVPR**
    - ***GLIGEN:*** Open-Set Grounded Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_GLIGEN_Open-Set_Grounded_Text-to-Image_Generation_CVPR_2023_paper.pdf) [[Code]](https://github.com/gligen/GLIGEN) [[Project]](https://gligen.github.io/) [[Demo]](https://huggingface.co/spaces/gligen/demo) [[Video]](https://www.youtube.com/watch?v=-MCkU7IAGKs&feature=youtu.be)
    - Autoregressive Image Generation using Residual Quantization [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Autoregressive_Image_Generation_Using_Residual_Quantization_CVPR_2022_paper.pdf) [[Code]](https://github.com/kakaobrain/rq-vae-transformer)
    - ***SpaText:*** Spatio-Textual Representation for Controllable Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Avrahami_SpaText_Spatio-Textual_Representation_for_Controllable_Image_Generation_CVPR_2023_paper.pdf) [[Project]](https://omriavrahami.com/spatext/) [[Video]](https://www.youtube.com/watch?v=VlieNoCwHO4)
    - Text to Image Generation with Semantic-Spatial Aware GAN [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liao_Text_to_Image_Generation_With_Semantic-Spatial_Aware_GAN_CVPR_2022_paper.pdf)
    - ***ReCo:*** Region-Controlled Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_ReCo_Region-Controlled_Text-to-Image_Generation_CVPR_2023_paper.pdf) [[Code]](https://github.com/microsoft/ReCo)
  - **ICCV**
    - ***ControlNet:*** Adding Conditional Control to Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf) [[Code]](https://github.com/lllyasviel/ControlNet)
    - ***SceneGenie:*** Scene Graph Guided Diffusion Models for Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/ICCV2023W/SG2RL/papers/Farshad_SceneGenie_Scene_Graph_Guided_Diffusion_Models_for_Image_Synthesis_ICCVW_2023_paper.pdf) [[Code]](https://openaccess.thecvf.com/content/ICCV2023W/SG2RL/papers/Farshad_SceneGenie_Scene_Graph_Guided_Diffusion_Models_for_Image_Synthesis_ICCVW_2023_paper.pdf)
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
  - **arXiv**
    - ***T2I-Adapter:*** Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models [[Paper]](https://arxiv.org/pdf/2302.08453.pdf) [[Code]](https://github.com/TencentARC/T2I-Adapter) [[Demo]](https://huggingface.co/spaces/TencentARC/T2I-Adapter-SDXL)

[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="conditional-year-2022">**Year 2022**</span>

[<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
## Personalized Text-to-Image Generation
- <span id="attribute-year-2023">**Year 2023**</span>
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
  - **arXiv**
    - ***InstantBooth:*** Personalized Text-to-Image Generation without Test-Time Finetuning [[Paper]](https://arxiv.org/pdf/2304.03411.pdf) [[Project]](https://jshi31.github.io/InstantBooth/)
    - ***DreamTuner:*** Single Image is Enough for Subject-Driven Generation [[Paper]](https://arxiv.org/pdf/2312.13691.pdf) [[Project]](https://dreamtuner-diffusion.github.io/)
    - ***PhotoMaker:*** Customizing Realistic Human Photos via Stacked ID Embedding [[Paper]](https://arxiv.org/pdf/2312.04461.pdf) [[Code]](https://github.com/TencentARC/PhotoMaker)

[<u><small><🎯Back to Top></small></u>](#contents)

- <span id="attribute-year-2022">**Year 2022**</span>

[<u><small><🎯Back to Top></small></u>](#contents)

<!-- omit in toc -->
## Text-Guided Image Manipulation
- <span id="manipulation-year-2023">**Year 2023**</span>
  - **CVPR**
    - Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Uncovering_the_Disentanglement_Capability_in_Text-to-Image_Diffusion_Models_CVPR_2023_paper.pdf) [[Code]](https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement)
    - ***SINE:*** SINgle Image Editing with Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_SINE_SINgle_Image_Editing_With_Text-to-Image_Diffusion_Models_CVPR_2023_paper.pdf) [[Code]](https://github.com/zhang-zx/SINE)
    - ***Imagic:*** Text-Based Real Image Editing with Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kawar_Imagic_Text-Based_Real_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf)
    - ***InstructPix2Pix:*** Learning to Follow Image Editing Instructions [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf) [[Code]](https://github.com/timothybrooks/instruct-pix2pix) [[Dataset]](https://instruct-pix2pix.eecs.berkeley.edu/) [[Project]](https://www.timothybrooks.com/instruct-pix2pix/) [[Demo]](https://huggingface.co/spaces/timbrooks/instruct-pix2pix)
  - **ICCV**
    - ***MasaCtrl:*** Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_MasaCtrl_Tuning-Free_Mutual_Self-Attention_Control_for_Consistent_Image_Synthesis_and_ICCV_2023_paper.pdf) [[Code]](https://github.com/TencentARC/MasaCtrl) [[Project]](https://ljzycmd.github.io/projects/MasaCtrl/) [[Demo]](https://colab.research.google.com/drive/1DZeQn2WvRBsNg4feS1bJrwWnIzw1zLJq?usp=sharing)
    - Localizing Object-level Shape Variations with Text-to-Image Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Patashnik_Localizing_Object-Level_Shape_Variations_with_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf) [[Code]](https://github.com/orpatashnik/local-prompt-mixing) [[Project]](https://orpatashnik.github.io/local-prompt-mixing/) [[Demo]](https://huggingface.co/spaces/orpatashnik/local-prompt-mixing)
- <span id="manipulation-year-2022">**Year 2022**</span>
  - **CVPR**
    - ***DiffusionCLIP:*** Text-Guided Diffusion Models for Robust Image Manipulation [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.pdf) [[Code]](https://github.com/gwang-kim/DiffusionCLIP)

[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
# Datasets
- ***Microsoft COCO:*** Common Objects in Context [[Paper]](https://arxiv.org/pdf/1405.0312.pdf) [[Dataset]](https://cocodataset.org/#home)
- ***Conceptual Captions:*** A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning [[Paper]](https://aclanthology.org/P18-1238.pdf) [[Dataset]](https://ai.google.com/research/ConceptualCaptions/)
- ***LAION-5B:*** An Open Large-Scale Dataset for Training Next Generation Image-Text Models [[Paper]](https://openreview.net/pdf?id=M3Y74vmsMcY) [[Dataset]](https://laion.ai/)


[<u><small><🎯Back to Top></small></u>](#contents)


<!-- omit in toc -->
# To-Do Lists
- Published Papers on Conferences
  - [x] Update CVPR papers
  - [x] Update ICCV papers
  - [x] Update ECCV papers
  - [ ] Update ACM MM papers
  - [ ] Update NeurIPS papers
  - [ ] Update ICLR papers
- Regular Maintenance of Preprint arXiv Papers and Missed Papers

[<u><small><🎯Back to Top></small></u>](#contents)