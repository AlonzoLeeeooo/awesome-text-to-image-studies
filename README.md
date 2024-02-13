<p align="center">
  <h1 align="center">A Collection of Text-to-Image Generation Studies</h1>

This GitHub repository summarizes papers and resources related to the text-to-image generation task. 

If you have any suggestions about this repository, please feel free to [start a new issue](https://github.com/AlonzoLeeeooo/awesome-text-to-image-generation-studies/issues/new) or [pull requests](https://github.com/AlonzoLeeeooo/awesome-text-to-image-generation-studies/pulls).


<!-- omit in toc -->
# Contents
- [References](#references)
- [Papers](#papers)
  - [Text-to-Image Generation](#text-to-image-generation)
    - [Year 2023](#text-year-2023)
    - [Year 2022](#text-year-2022)
  - [Conditional Text-to-Image Generation](#conditional-text-to-image-generation)
    - [Year 2023](#conditional-year-2023)
    - [Year 2022](#conditional-year-2022)
  - [Personalized Text-to-Image Generation](#personalized-text-to-image-generation)
    - [Year 2023](#attribute-year-2023)
    - [Year 2022](#attribute-year-2022)

<!-- omit in toc -->
# References

The `reference.bib` file summarizes bibtex references of up-to-date image inpainting papers, widely used datasets, and toolkits.
Based on the original references, I have made the following modifications to make their results look nice in the `LaTeX` manuscripts:
- Refereces are normally constructed in the form of `author-etal-year-nickname`. Particularly, references of datasets and toolkits are directly constructed as `nickname`, e.g., `imagenet`.
- In each reference, all names of conferences/journals are converted into abbreviations, e.g., `{IEEE/CVF} Conference on Computer Vision and Pattern Recognition -> CVPR`.
- The `url`, `doi`, `publisher`, `organization`, `editor`, `series` in all references are removed.
- The `pages` of all references are added if they are missing.
- All paper names are in title case. Besides, I have added an additional `{}` to make sure that the title case would also work well in some particular templates. 

If you have other demands of reference formats, you may refer to the original references of papers by searching their names in [DBLP](https://dblp.org/) or [Google Scholar](https://scholar.google.com/).



<!-- omit in toc -->
# Papers

<!-- omit in toc -->
## Text-to-Image Generation
- <span id="text-year-2023">**Year 2023**</span>
  - **NeurIPS**
    - ***ImageReward:*** Learning and Evaluating Human Preferences for Text-to-Image Generation [[Paper]](https://openreview.net/pdf?id=JVzeOYEx6d) [[Code]](https://github.com/THUDM/ImageReward)
  - **arXiv**
    - ***P+:*** Extended Textual Conditioning in Text-to-Image Generation [[Paper]](https://prompt-plus.github.io/files/PromptPlus.pdf)
  - **Others**
    - **DALLE-3:** Improving Image Generation with Better Captions [[Paper]](https://cdn.openai.com/papers/dall-e-3.pdf)
- <span id="text-year-2022">**Year 2022**</span>
  - **CVPR**
    - 🔥 ***Stable Diffusion:*** High-Resolution Image Synthesis With Latent Diffusion Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) [[Code]](https://github.com/CompVis/latent-diffusion) [[Project]](https://ommer-lab.com/research/latent-diffusion-models/)
    - Vector Quantized Diffusion Model for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf) [[Code]](https://github.com/cientgu/VQ-Diffusion)
    - ***DF-GAN:*** A Simple and Effective Baseline for Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tao_DF-GAN_A_Simple_and_Effective_Baseline_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf) [[Code]](https://github.com/tobran/DF-GAN)
    - ***LAFITE:*** Towards Language-Free Training for Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Towards_Language-Free_Training_for_Text-to-Image_Generation_CVPR_2022_paper.pdf) [[Code]](https://github.com/drboog/Lafite)
    - Text-to-Image Synthesis based on Object-Guided Joint-Decoding Transformer [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Text-to-Image_Synthesis_Based_on_Object-Guided_Joint-Decoding_Transformer_CVPR_2022_paper.pdf)
    - ***StyleT2I:*** Toward Compositional and High-Fidelity Text-to-Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_StyleT2I_Toward_Compositional_and_High-Fidelity_Text-to-Image_Synthesis_CVPR_2022_paper.pdf) [[Code]](https://github.com/zhihengli-UR/StyleT2I)
  - **NeurIPS**
    - ***CogView2:*** Faster and Better Text-to-Image Generation via Hierarchical Transformers [[Paper]](https://openreview.net/pdf?id=GkDbQb6qu_r) [[Code]](https://openreview.net/pdf?id=GkDbQb6qu_r)
    - ***Imagen:*** Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding [[Paper]](https://papers.nips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf) [[Reproduced Code]](https://github.com/lucidrains/imagen-pytorch) [[Project]](https://imagen.research.google/) [[***Imagen 2***]](https://deepmind.google/technologies/imagen-2/)
  - **arXiv**
    - ***DALLE-2:*** Hierarchical Text-Conditional Image Generation with CLIP Latents [[Paper]](https://cdn.openai.com/papers/dall-e-2.pdf)
- <span id="text-year-2021">**Year 2021**</span>
  - **NeurIPS**
    - ***CogView:*** Mastering Text-to-Image Generation via Transformers [[Paper]](https://proceedings.neurips.cc/paper/2021/file/a4d92e2cd541fca87e4620aba658316d-Paper.pdf) [[Code]](https://github.com/THUDM/CogView) [[Demo]](https://thudm.github.io/CogView/index.html)
  - **ICML**
    - ***DALLE-1:*** Zero-Shot Text-to-Image Generation [[Paper]](https://proceedings.mlr.press/v139/ramesh21a/ramesh21a.pdf) [[Reproduced Code]](https://github.com/lucidrains/DALLE-pytorch)
<!-- omit in toc -->
## Conditional Text-to-Image Generation
- <span id="conditional-year-2023">**Year 2023**</span>
  - **CVPR**
    - ***GLIGEN:*** Open-Set Grounded Text-to-Image Generation [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_GLIGEN_Open-Set_Grounded_Text-to-Image_Generation_CVPR_2023_paper.pdf) [[Code]](https://github.com/gligen/GLIGEN) [[Project]](https://gligen.github.io/) [[Demo]](https://huggingface.co/spaces/gligen/demo) [[Video]](https://www.youtube.com/watch?v=-MCkU7IAGKs&feature=youtu.be)
- <span id="conditional-year-2022">**Year 2022**</span>

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
  - **ACM Trans. Graph**
    - Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models [[Paper]](https://arxiv.org/pdf/2302.12228.pdf) [[Project]](https://tuning-encoder.github.io/)
  - **arXiv**
    - ***InstantBooth:*** Personalized Text-to-Image Generation without Test-Time Finetuning [[Paper]](https://arxiv.org/pdf/2304.03411.pdf) [[Project]](https://jshi31.github.io/InstantBooth/)
- <span id="attribute-year-2022">**Year 2022**</span>

<!-- omit in toc -->
## Text-Guided Image Manipulation
- <span id="manipulation-year-2023">**Year 2023**</span>
- <span id="manipulation-year-2022">**Year 2022**</span>
  - **CVPR**
    - ***DiffusionCLIP:*** Text-Guided Diffusion Models for Robust Image Manipulation [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.pdf) [[Code]](https://github.com/gwang-kim/DiffusionCLIP)


<!-- omit in toc -->
# To-Do Lists
- Published Papers on Conferences
  - [ ] Update CVPR papers
  - [ ] Update ICCV papers
  - [ ] Update ECCV papers
  - [ ] Update ACM MM papers
  - [ ] Update NeurIPS papers
  - [ ] Update ICLR papers
- Regular Maintenance of Preprint arXiv Papers and Missed Papers
