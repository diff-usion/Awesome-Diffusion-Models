[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/hee9joon/Awesome-Diffusion-Models) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

This repository contains a collection of resources and papers on ***Diffusion Models***.

Please refer to [this page](https://diff-usion.github.io/Awesome-Diffusion-Models/) as this page may not contain all the information due to page constraints.

## Contents
- [Resources](#resources)
  - [Introductory Posts](#introductory-posts)
  - [Introductory Papers](#introductory-papers)
  - [Introductory Videos](#introductory-videos)
  - [Introductory Lectures](#introductory-lectures)
  - [Tutorial and Jupyter Notebook](#tutorial-and-jupyter-notebook)
- [Papers](#papers)
  - [Survey](#survey)
  - [Vision](#vision)
    - [Generation](#generation)
    - [Classification](#classification)
    - [Segmentation](#segmentation)
    - [Image Translation](#image-translation)
    - [Inverse Problems](#inverse-problems)
    - [Medical Imaging](#medical-imaging)
    - [Multi-modal Learning](#multi-modal-learning)
    - [3D Vision](#3d-vision)
    - [Adversarial Attack](#adversarial-attack)
    - [Miscellany](#miscellany)
  - [Audio](#audio)
    - [Generation](#generation-1)
    - [Conversion](#conversion)
    - [Enhancement](#enhancement)
    - [Separation](#separation)
    - [Text-to-Speech](#text-to-speech)
    - [Miscellany](#miscellany-1)
  - [Natural Language](#natural-language)
  - [Tabular and Time Series](#tabular-and-time-series)
    - [Generation](#generation-2)
    - [Forecasting](#forecasting)
    - [Imputation](#imputation)
    - [Miscellany](#miscellany-2)
  - [Graph](#graph)
    - [Generation](#generation-3)
    - [Molecular and Material Generation](#molecular-and-material-generation)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Theory](#theory)
  - [Applications](#applications)


# Resources
## Introductory Posts

**:fast_forward: DiffusionFastForward: 01-Diffusion-Theory** \
*Mikolaj Czerkawski (@mikonvergence)* \
[[Website](https://github.com/mikonvergence/DiffusionFastForward/blob/master/notes/01-Diffusion-Theory.md)] \
4 Feb 2023

**How diffusion models work: the math from scratch** \
*Sergios Karagiannakos,Nikolas Adaloglou* \
[[Website](https://theaisummer.com/diffusion-models/?fbclid=IwAR1BIeNHqa3NtC8SL0sKXHATHklJYphNH-8IGNoO3xZhSKM_GYcvrrQgB0o)] \
24 Sep 2022

**A Path to the Variational Diffusion Loss** \
*Alex Alemi* \
[[Website](https://blog.alexalemi.com/diffusion.html)] [[Colab](https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb)] \
15 Sep 2022

**The Annotated Diffusion Model** \
*Niels Rogge, Kashif Rasul* \
[[Website](https://huggingface.co/blog/annotated-diffusion)] \
06 Jun 2022

**The recent rise of diffusion-based models** \
*Maciej Domagała* \
[[Website](https://maciejdomagala.github.io/generative_models/2022/06/06/The-recent-rise-of-diffusion-based-models.html)] \
06 Jun 2022

**Introduction to Diffusion Models for Machine Learning** \
*Ryan O'Connor* \
[[Website](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)] \
12 May 2022

**Improving Diffusion Models as an Alternative To GANs** \
*Arash Vahdat and Karsten Kreis* \
[[Website-Part 1](https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/)] [[Website-Part 2](https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-2/)] \
26 Apr 2022

**An introduction to Diffusion Probabilistic Models** \
*Ayan Das* \
[[Website](https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html)] \
04 Dec 2021

**Introduction to deep generative modeling: Diffusion-based Deep Generative Models** \
*Jakub Tomczak* \
[[Website](https://jmtomczak.github.io/blog/10/10_ddgms_lvm_p2.html)] \
30 Aug 2021

**What are Diffusion Models?** \
*Lilian Weng* \
[[Website](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html)] \
11 Jul 2021

**Diffusion Models as a kind of VAE** \
*Angus Turner* \
[[Website](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html)] \
29 Jun 2021

**Generative Modeling by Estimating Gradients of the Data Distribution** \
*Yang Song* \
[[Website](https://yang-song.github.io/blog/2021/score/)] \
5 May 2021

- [guardian-agent-prompts](https://github.com/milkomida77/guardian-agent-prompts) - 49 production-tested AI agent system prompts for diffusion model workflow orchestration, automated image generation pipeline management, and multi-agent coordination. MIT licensed.
## Introductory Papers

**Understanding Diffusion Models: A Unified Perspective** \
*Calvin Luo* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.11970)] \
25 Aug 2022

**How to Train Your Energy-Based Models** \
*Yang Song, Diederik P. Kingma* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2101.03288)] \
9 Jan 2021

## Introductory Videos

**:fast_forward: DiffusionFastForward** \
*Mikolaj Czerkawski (@mikonvergence)* \
[[Video](https://www.youtube.com/playlist?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)] \
4 Mar 2023

**Diffusion models from scratch in PyTorch** \
*DeepFindr* \
[[Video](https://www.youtube.com/watch?v=a4Yfz2FxXiY)] \
18 Jul 2022

**Diffusion Models | Paper Explanation | Math Explained** \
*Outlier* \
[[Video](https://www.youtube.com/watch?v=HoKDTa5jHvg)] \
6 Jun 2022

**What are Diffusion Models?** \
*Ari Seff* \
[[Video](https://www.youtube.com/watch?v=fbLgFrlTnGU&list=LL&index=2)] \
20 Apr 2022

**Diffusion models explained** \
*AI Coffee Break with Letitia* \
[[Video](https://www.youtube.com/watch?v=344w5h24-h8&ab_channel=AICoffeeBreakwithLetitia)] \
23 Mar 2022

## Introductory Lectures

**Denoising Diffusion-based Generative Modeling: Foundations and Applications** \
*Karsten Kreis, Ruiqi Gao, Arash Vahdat* \
[[Page](https://cvpr2022-tutorial-diffusion-models.github.io/)] \
19 Jun 2022

**Diffusion Probabilistic Models** \
*Jascha Sohl-Dickstein, MIT 6.S192 - Lecture 22* \
[[Video](https://www.youtube.com/watch?v=XCUlnHP1TNM)] \
19 Apr 2022

## Tutorial and Jupyter Notebook

**:fast_forward: DiffusionFastForward: train from scratch in colab** \
*Mikolaj Czerkawski (@mikonvergence)* \
[[Github](https://github.com/mikonvergence/DiffusionFastForward)]
[[notebook](https://github.com/mikonvergence/DiffusionFastForward#computer-code)]

**diffusion-for-beginners** \
*ozanciga* \
[[Github](https://github.com/ozanciga/diffusion-for-beginners)]

**Beyond Diffusion: What is Personalized Image Generation and How Can You Customize Image Synthesis?** \
*J. Rafid Siddiqui* \
[[Github](https://github.com/azad-academy/personalized-diffusion)] [[Medium](https://medium.com/mlearning-ai/beyond-diffusion-what-is-personalized-image-generation-and-how-can-you-customize-image-synthesis-26a89d5b335)]

**Diffusion_models_tutorial** \
*FilippoMB* \
[[Github](https://github.com/FilippoMB/Diffusion_models_tutorial)]

**ScoreDiffusionModel** \
*JeongJiHeon* \
[[Github](https://github.com/JeongJiHeon/ScoreDiffusionModel)]

**Minimal implementation of diffusion models** \
*VSehwag* \
[[Github](https://github.com/VSehwag/minimal-diffusion)]

**diffusion_tutorial** \
*sunlin-ai* \
[[Github](https://github.com/sunlin-ai/diffusion_tutorial)] 

**Denoising diffusion probabilistic models** \
*acids-ircam* \
[[Github](https://github.com/acids-ircam/diffusion_models)] 


**Centipede Diffusion** \
*Zalring* \
[[Notebook](https://colab.research.google.com/github/Zalring/Centipede_Diffusion/blob/main/Centipede_Diffusion.ipynb)]

**Deforum Stable Diffusion** \
*deforum* \
[[Notebook](https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb)]

**Stable Diffusion Interpolation** \
*None* \
[[Notebook](https://colab.research.google.com/drive/1EHZtFjQoRr-bns1It5mTcOVyZzZD9bBc?usp=sharing)]

**Keras Stable Diffusion: GPU starter example** \
*None* \
[[Notebook](https://colab.research.google.com/drive/1zVTa4mLeM_w44WaFwl7utTaa6JcaH1zK)]

**Huemin Jax Diffusion** \
*huemin-art* \
[[Notebook](https://colab.research.google.com/github/huemin-art/jax-guided-diffusion/blob/v2.7/Huemin_Jax_Diffusion_2_7.ipynb)]

**Disco Diffusion** \
*alembics* \
[[Notebook](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb)]

**Simplified Disco Diffusion** \
*entmike* \
[[Notebook](https://colab.research.google.com/github/entmike/disco-diffusion-1/blob/main/Simplified_Disco_Diffusion.ipynb)]

**WAS's Disco Diffusion - Portrait Generator Playground** \
*WASasquatch* \
[[Notebook](https://colab.research.google.com/github/WASasquatch/disco-diffusion-portrait-playground/blob/main/WAS's_Disco_Diffusion_v5_6_9_%5BPortrait_Generator_Playground%5D.ipynb)]

**Diffusers - Hugging Face** \
*huggingface* \
[[Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)] 


# Papers

## Survey

**A Survey on Video Diffusion Models** \
*Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu and Yu-Gang Jiang*
arXiv 2023. [[Paper](https://arxiv.org/pdf/2310.10647.pdf)] \
16 Oct 2023

**State of the Art on Diffusion Models for Visual Computing** \
*Ryan Po, Wang Yifan, Vladislav Golyanik, Kfir Aberman, Jonathan T. Barron, Amit H. Bermano, Eric Ryan Chan, Tali Dekel, Aleksander Holynski, Angjoo Kanazawa, C. Karen Liu, Lingjie Liu, Ben Mildenhall, Matthias Nießner, Björn Ommer, Christian Theobalt, Peter Wonka, Gordon Wetzstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07204)] \
11 Oct 2023

**Memory in Plain Sight: A Survey of the Uncanny Resemblances between Diffusion Models and Associative Memories** \
*Benjamin Hoover, Hendrik Strobelt, Dmitry Krotov, Judy Hoffman, Zsolt Kira, Duen Horng Chau* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.16750)] \
28 Sep 2023

**A Survey of Diffusion Based Image Generation Models: Issues and Their Solutions** \
*Tianyi Zhang, Zheng Wang, Jing Huang, Mohiuddin Muhammad Tasnim, Wei Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13142)] \
25 Aug 2023

**Diffusion Models for Image Restoration and Enhancement -- A Comprehensive Survey** \
*Xin Li, Yulin Ren, Xin Jin, Cuiling Lan, Xingrui Wang, Wenjun Zeng, Xinchao Wang, Zhibo Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09388)] \
18 Aug 2023

**A Comprehensive Survey on Generative Diffusion Models for Structured Data** \
*Heejoon Koo, To Eun Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04139)] \
7 Jun 2023

**On the Design Fundamentals of Diffusion Models: A Survey** \
*Ziyi Chang, George A. Koulieris, Hubert P. H. Shum* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04542)] \
7 Jun 2023

**Diffusion Models in NLP: A Survey** \
*Hao Zou, Zae Myung Kim, Dongyeop Kang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14671)] \
24 May 2023

**Diffusion Models for Time Series Applications: A Survey** \
*Lequan Lin, Zhengkun Li, Ruikun Li, Xuliang Li, Junbin Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.00624)] \
1 May 2023

**A Comprehensive Survey on Knowledge Distillation of Diffusion Models** \
*Weijian Luo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04262)] \
9 Apr 2023

**A Survey on Graph Diffusion Models: Generative AI in Science for Molecule, Protein and Material** \
*Mengchun Zhang, Maryam Qamar, Taegoo Kang, Yuna Jung, Chenshuang Zhang, Sung-Ho Bae, Chaoning Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01565)] \
4 Apr 2023

**Audio Diffusion Model for Speech Synthesis: A Survey on Text To Speech and Speech Enhancement in Generative AI** \
*Chenshuang Zhang, Chaoning Zhang, Sheng Zheng, Mengchun Zhang, Maryam Qamar, Sung-Ho Bae, In So Kweon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13336)] \
23 Mar 2023

**Diffusion Models in NLP: A Survey** \
*Yuansong Zhu, Yu Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07576)] \
14 Mar 2023

**Text-to-image Diffusion Model in Generative AI: A Survey** \
*Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, In So Kweon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07909)] \
14 Mar 2023

**Diffusion Models for Non-autoregressive Text Generation: A Survey** \
*Yifan Li, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06574)] \
12 Mar 2023

**Diffusion Models in Bioinformatics: A New Wave of Deep Learning Revolution in Action** \
*Zhiye Guo, Jian Liu, Yanli Wang, Mengrui Chen, Duolin Wang, Dong Xu, Jianlin Cheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10907)] \
13 Feb 2023

**Generative Diffusion Models on Graphs: Methods and Applications** \
*Wenqi Fan, Chengyi Liu, Yunqing Liu, Jiatong Li, Hang Li, Hui Liu, Jiliang Tang, Qing Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02591)] \
6 Feb 2023

**Diffusion Models for Medical Image Analysis: A Comprehensive Survey** \
*Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Moein Heidari, Reza Azad, Mohsen Fayyaz, Ilker Hacihaliloglu, Dorit Merhof* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.07804)] [[Github](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging)] \
14 Nov 2022

**Efficient Diffusion Models for Vision: A Survey** \
*Anwaar Ulhaq, Naveed Akhtar, Ganna Pogrebna* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.09292)] \
7 Oct 2022

**Diffusion Models in Vision: A Survey** \
*Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Mubarak Shah* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.04747)] \
10 Sep 2022

**A Survey on Generative Diffusion Model** \
*Hanqun Cao, Cheng Tan, Zhangyang Gao, Guangyong Chen, Pheng-Ann Heng, Stan Z. Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.02646)] \
6 Sep 2022

**Diffusion Models: A Comprehensive Survey of Methods and Applications** \
*Ling Yang, Zhilong Zhang, Shenda Hong, Wentao Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.00796)] \
2 Sep 2022

## Vision
### Generation

**DiffEnc: Variational Diffusion with a Learned Encoder** \
*Beatrix M. G. Nielsen, Anders Christensen, Andrea Dittadi, Ole Winther* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19789)] \
30 Oct 2023

**Upgrading VAE Training With Unlimited Data Plans Provided by Diffusion Models** \
*Tim Z. Xiao, Johannes Zenn, Robert Bamler* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19653)] \
30 Oct 2023

**Successfully Applying Lottery Ticket Hypothesis to Diffusion Model** \
*Chao Jiang, Bo Hui, Bohan Liu, Da Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.18823)] \
28 Oct 2023

**Noise-Free Score Distillation** \
*Oren Katzir, Or Patashnik, Daniel Cohen-Or, Dani Lischinski* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17590)] \
26 Oct 2023

**The statistical thermodynamics of generative diffusion models** \
*Luca Ambrogioni* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17467)] \
26 Oct 2023

**Improving Denoising Diffusion Models via Simultaneous Estimation of Image and Noise** \
*Zhenkai Zhang, Krista A. Ehinger, Tom Drummond* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17167)] \
26 Oct 2023

**Hierarchical Semi-Implicit Variational Inference with Application to Diffusion Model Acceleration** \
*Longlin Yu, Tianyu Xie, Yu Zhu, Tong Yang, Xiangyu Zhang, Cheng Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17153)] [[Github](https://github.com/longinyu/hsivi)] \
26 Oct 2023

**RePoseDM: Recurrent Pose Alignment and Gradient Guidance for Pose Guided Image Synthesis** \
*Anant Khandelwal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16074)] \
24 Oct 2023

**Improved Techniques for Training Consistency Models** \
*Yang Song, Prafulla Dhariwal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.14189)] \
22 Oct 2023

**ScaleLong: Towards More Stable Training of Diffusion Model via Scaling Network Long Skip Connection** \
*Zhongzhan Huang, Pan Zhou, Shuicheng Yan, Liang Lin* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.13545)] [[Github](https://github.com/sail-sg/ScaleLong)] \
20 Oct 2023


**Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models** \
*Gabriele Corso, Yilun Xu, Valentin de Bortoli, Regina Barzilay, Tommi Jaakkola* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.13102)] [[Github](https://github.com/gcorso/particle-guidance)] \
19 Oct 2023

**Closed-Form Diffusion Models** \
*Christopher Scarvelis, Haitz Sáez de Ocáriz Borde, Justin Solomon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12395)] \
19 Oct 2023

**Elucidating The Design Space of Classifier-Guided Diffusion Generation** \
*Jiajun Ma, Tianyang Hu, Wenjia Wang, Jiacheng Sun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11311)] [[Github](https://github.com/alexmaols/elucd)] \
17 Oct 2023


**BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference** \
*Siqi Kou, Lei Gan, Dequan Wang, Chongxuan Li, Zhijie Deng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11142)] \
17 Oct 2023

**Unsupervised Discovery of Interpretable Directions in h-space of Pre-trained Diffusion Models** \
*Zijian Zhang, Luping Liu. Zhijie Lin, Yichen Zhu, Zhou Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09912)] \
15 Oct 2023

**Towards More Accurate Diffusion Model Acceleration with A Timestep Aligner** \
*Mengfei Xia, Yujun Shen, Changsong Lei, Yu Zhou, Ran Yi, Deli Zhao, Wenping Wang, Yong-jin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09469)] \
14 Oct 2023

**Unseen Image Synthesis with Diffusion Models** \
*Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, Yan Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09213)] \
13 Oct 2023

**Debias the Training of Diffusion Models** \
*Hu Yu, Li Shen, Jie Huang, Man Zhou, Hongsheng Li, Feng Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08442)] \
12 Oct 2023

**Neural Diffusion Models** \
*Grigory Bartosh, Dmitry Vetrov, Christian A. Naesseth* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08337)] \
12 Oct 2023

**Efficient Integrators for Diffusion Generative Models** \
*Kushagra Pandey, Maja Rudolph, Stephan Mandt* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07894)] \
11 Oct 2023


**Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling** \
*Huangjie Zheng, Zhendong Wang, Jianbo Yuan, Guanghan Ning, Pengcheng He, Quanzeng You, Hongxia Yang, Mingyuan Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06389)] \
10 Oct 2023

**Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation** \
*Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, Lu Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05737)] [[Github](https://github.com/lucidrains/magvit2-pytorch)] \
9 Oct 2023

**The Emergence of Reproducibility and Consistency in Diffusion Models** \
*Huijie Zhang, Jinfan Zhou, Yifu Lu, Minzhe Guo, Liyue Shen, Qing Qu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05264)] \
8 Oct 2023

**DiffNAS: Bootstrapping Diffusion Models by Prompting for Better Architectures** \
*Wenhao Li, Xiu Su, Shan You, Fei Wang, Chen Qian, Chang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.04750)] \
7 Oct 2023

**Observation-Guided Diffusion Probabilistic Models** \
*Junoh Kang, Jinyoung Choi, Sungik Choi, Bohyung Han* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.04041)] \
6 Oct 2023

**Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference** \
*Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, Hang Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.04378)] \
6 Oct 2023

**Denoising Diffusion Step-aware Models** \
*Shuai Yang, Yukang Chen, Luozhou Wang, Shu Liu, Yingcong Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03337)] \
5 Oct 2023


**EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models** \
*Yefei He, Jing Liu, Weijia Wu, Hong Zhou, Bohan Zhuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03270)] \
5 Oct 2023

**Learning Energy-Based Prior Model with Diffusion-Amortized MCMC** \
*Peiyu Yu, Yaxuan Zhu, Sirui Xie, Xiaojian Ma, Ruiqi Gao, Song-Chun Zhu, Ying Nian Wu* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.03218)] [[Github](https://github.com/yuPeiyu98/Diffusion-Amortized-MCMC)] \
5 Oct 2023

**On Memorization in Diffusion Models** \
*Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, Ye Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02664)] [[Github](https://github.com/sail-sg/DiffMemorize)] \
4 Oct 2023


**Sequential Data Generation with Groupwise Diffusion Process** \
*Sangyun Lee, Gayoung Lee, Hyunsu Kim, Junho Kim, Youngjung Uh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01400)] \
2 Oct 2023

**Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion** \
*Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, Stefano Ermon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02279)] \
1 Oct 2023

**Completing Visual Objects via Bridging Generation and Segmentation** \
*Xiang Li, Yinpeng Chen, Chung-Ching Lin, Rita Singh, Bhiksha Raj, Zicheng Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00808)] \
1 Oct 2023

**Decoding Realistic Images from Brain Activity with Contrastive Self-supervision and Latent Diffusion** \
*Jingyuan Sun, Mingxiao Li, Marie-Francine Moens* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00318)] \
30 Sep 2023

**FashionFlow: Leveraging Diffusion Models for Dynamic Fashion Video Synthesis from Static Imagery** \
*Tasin Islam, Alina Miron, XiaoHui Liu, Yongmin Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00106)] \
29 Sep 2023

**Denoising Diffusion Bridge Models** \
*Linqi Zhou, Aaron Lou, Samar Khanna, Stefano Ermon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.16948)] \
29 Sep 2023


**DeeDiff: Dynamic Uncertainty-Aware Early Exiting for Accelerating Diffusion Model Generation** \
*Shengkun Tang, Yaqing Wang, Caiwen Ding, Yi Liang, Yao Li, Dongkuan Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.17074)] \
29 Sep 2023


**Distilling ODE Solvers of Diffusion Models into Smaller Steps** \
*Sanghwan Kim, Hao Tang, Fisher Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.16421)] \
28 Sep 2023

**Factorized Diffusion Architectures for Unsupervised Image Generation and Segmentation** \
*Xin Yuan, Michael Maire* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.15726)] \
27 Sep 2023

**Generative Escher Meshes** \
*Noam Aigerman, Thibault Groueix* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14564)] \
25 Sep 2023

**Soft Mixture Denoising: Beyond the Expressive Bottleneck of Diffusion Models** \
*Yangming Li, Boris van Breugel, Mihaela van der Schaar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14068)] \
25 Sep 2023

**GLOBER: Coherent Non-autoregressive Video Generation via GLOBal Guided Video DecodER** \
*Mingzhen Sun, Weining Wang, Zihan Qin, Jiahui Sun, Sihan Chen, Jing Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.13274)] [[Github](https://github.com/iva-mzsun/glober)] \
23 Sep 2023

**Score Mismatching for Generative Modeling** \
*Senmao Ye, Fei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11043)] \
20 Sep 2023

**Generalised Probabilistic Diffusion Scale-Spaces** \
*Pascal Peter* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.08511)] \
15 Sep 2023

**Generative Image Dynamics** \
*Zhengqi Li, Richard Tucker, Noah Snavely, Aleksander Holynski* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07906)] [[Project](https://generative-dynamics.github.io/)] \
14 Sep 2023

**Beta Diffusion** \
*Mingyuan Zhou, Tianqi Chen, Zhendong Wang, Huangjie Zheng* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2309.07867)] \
14 Sep 2023

**Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models** \
*Zalan Fabian, Berk Tinaz, Mahdi Soltanolkotabi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.06642)] \
12 Sep 2023

**Elucidating the solution space of extended reverse-time SDE for diffusion models** \
*Qinpeng Cui, Xinyi Zhang, Zongqing Lu, Qingmin Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.06169)] \
12 Sep 2023


**Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood** \
*Yaxuan Zhu, Jianwen Xie, Yingnian Wu, Ruiqi Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05153)] \
10 Sep 2023

**Relay Diffusion: Unifying diffusion process across resolutions for image synthesis** \
*Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, Jie Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03350)] \
4 Sep 2023

**Gradient Domain Diffusion Models for Image Synthesis** \
*Yuanhao Gong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01875)] \
5 Sep 2023


**Hierarchical Masked 3D Diffusion Model for Video Outpainting** \
*Fanda Fan, Chaoxu Guo, Litong Gong, Biao Wang, Tiezheng Ge, Yuning Jiang, Chunjie Luo, Jianfeng Zhan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.02119)] [[Github](https://fanfanda.github.io/M3DDM/)] \
5 Sep 2023

**Diffusion Models with Deterministic Normalizing Flow Priors** \
*Mohsen Zand, Ali Etemad, Michael Greenspan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01274)] [[Github](https://github.com/MohsenZand/DiNof)] \
3 Sep 2023

**Diffusion Inertial Poser: Human Motion Reconstruction from Arbitrary Sparse IMU Configurations** \
*Tom Van Wouwe, Seunghwan Lee, Antoine Falisse, Scott Delp, C. Karen Liu* \
AAAI 2024. [[Paper](https://arxiv.org/abs/2308.16682)] \
31 Aug 2023

**Conditioning Score-Based Generative Models by Neuro-Symbolic Constraints** \
*Davide Scassola, Sebastiano Saccani, Ginevra Carbone, Luca Bortolussi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16534)] \
31 Aug 2023

**Elucidating the Exposure Bias in Diffusion Models** \
*Mang Ning, Mingxiao Li, Jianlin Su, Albert Ali Salah, Itir Onal Ertugrul* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15321)] \
29 Aug 2023

**Residual Denoising Diffusion Models** \
*Jiawei Liu, Qiang Wang, Huijie Fan, Yinong Wang, Yandong Tang, Liangqiong Qu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13712)] [[Github](https://github.com/nachifur/RDDM)] \
25 Aug 2023

**Efficient Transfer Learning in Diffusion Models via Adversarial Noise** \
*Xiyu Wang, Baijiong Lin, Daochang Liu, Chang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.11948)] \
23 Aug 2023

**Boosting Diffusion Models with an Adaptive Momentum Sampler** \
*Xiyu Wang, Anh-Dung Dinh, Daochang Liu, Chang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.11941)] \
23 Aug 2023

**Make-It-4D: Synthesizing a Consistent Long-Term Dynamic Scene Video from a Single Image** \
*Liao Shen, Xingyi Li, Huiqiang Sun, Juewen Peng, Ke Xian, Zhiguo Cao, Guosheng Lin* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.10257)] \
20 Aug 2023

**Spiking-Diffusion: Vector Quantized Discrete Diffusion Model with Spiking Neural Networks** \
*Mingxuan Liu, Rui Wen, Hong Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.10187)] \
20 Aug 2023

**SciRE-Solver: Efficient Sampling of Diffusion Probabilistic Models by Score-integrand Solver with Recursive Derivative Estimation** \
*Shigui Li, Wei Chen, Delu Zeng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07896)] \
15 Aug 2023

**Improved Order Analysis and Design of Exponential Integrator for Diffusion Models Sampling** \
*Qinsheng Zhang, Jiaming Song, Yongxin Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02157)] \
4 Aug 2023

**Patched Denoising Diffusion Models For High-Resolution Image Synthesis** \
*Zheng Ding, Mengqi Zhang, Jiajun Wu, Zhuowen Tu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01316)] \
2 Aug 2023

**Spatial-Frequency U-Net for Denoising Diffusion Probabilistic Models** \
*Xin Yuan, Linjie Li, Jianfeng Wang, Zhengyuan Yang, Kevin Lin, Zicheng Liu, Lijuan Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.14648)] \
27 Jul 2023

**Synthesis of Batik Motifs using a Diffusion -- Generative Adversarial Network** \
*One Octadion, Novanto Yudistira, Diva Kurnianingtyas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.12122)] \
22 Jul 2023

**DPM-OT: A New Diffusion Probabilistic Model Based on Optimal Transport** \
*Zezeng Li, ShengHao Li, Zhanpeng Wang, Na Lei, Zhongxuan Luo, Xianfeng Gu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.11308)] [[Github](https://github.com/cognaclee/DPM-OT)] \
21 Jul 2023

**Diffusion Sampling with Momentum for Mitigating Divergence Artifacts** \
*Suttisak Wizadwongsa, Worameth Chinchuthakun, Pramook Khungurn, Amit Raj, Supasorn Suwajanakorn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.11118)] \
20 Jul 2023

**Flow Matching in Latent Space** \
*Quan Dao, Hao Phung, Binh Nguyen, Anh Tran* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08698)] [[Project](https://vinairesearch.github.io/LFM/)] \
17 Jul 2023

**Manifold-Guided Sampling in Diffusion Models for Unbiased Image Generation** \
*Xingzhe Su, Wenwen Qiang, Zeen Song, Hang Gao, Fengge Wu, Changwen Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08199)] \
17 Jul 2023

**Complexity Matters: Rethinking the Latent Space for Generative Modeling** \
*Tianyang Hu, Fei Chen, Haonan Wang, Jiawei Li, Wenjia Wang, Jiacheng Sun, Zhenguo Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08283)] \
17 Jul 2023

**Collaborative Score Distillation for Consistent Visual Synthesis** \
*Subin Kim, Kyungmin Lee, June Suk Choi, Jongheon Jeong, Kihyuk Sohn, Jinwoo Shin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04787)] [[Project](https://subin-kim-cv.github.io/CSD/)] [[Github](https://github.com/subin-kim-cv/CSD)] \
4 Jul 2023

**ProtoDiffusion: Classifier-Free Diffusion Guidance with Prototype Learning** \
*Gulcin Baykal, Halil Faruk Karagoz, Taha Binhuraib, Gozde Unal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.01924)] \
4 Jul 2023

**SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis** \
*Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.01952)] [[Github](https://github.com/Stability-AI/generative-models)] \
4 Jul 2023

**Bidirectional Temporal Diffusion Model for Temporally Consistent Human Animation** \
*Tserendorj Adiya, Sanghun Kim, Jung Eun Lee, Jae Shin Yoon, Hwasup Lim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.00574)] \
2 Jul 2023


**Spiking Denoising Diffusion Probabilistic Models** \
*Jiahang Cao, Ziqing Wang, Hanzhong Guo, Hao Cheng, Qiang Zhang, Renjing Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17046)] \
29 Jun 2023

**DomainStudio: Fine-Tuning Diffusion Models for Domain-Driven Image Generation using Limited Data** \
*Jingyuan Zhu, Huimin Ma, Jiansheng Chen, Jian Yuan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.14153)] \
25 Jun 2023

**Decoupled Diffusion Models with Explicit Transition Probability** \
*Yuhang Huang, Zheng Qin, Xinwang Liu, Kai Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.13720)] \
23 Jun 2023

**Continuous Layout Editing of Single Images with Diffusion Models** \
*Zhiyuan Zhang, Zhitong Huang, Jing Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.13078)] \
22 Jun 2023

**Semi-Implicit Denoising Diffusion Models (SIDDMs)** \
*Yanwu Xu, Mingming Gong, Shaoan Xie, Wei Wei, Matthias Grundmann, kayhan Batmanghelich, Tingbo Hou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12511)] \
21 Jun 2023

**Eliminating Lipschitz Singularities in Diffusion Models** \
*Zhantao Yang, Ruili Feng, Han Zhang, Yujun Shen, Kai Zhu, Lianghua Huang, Yifei Zhang, Yu Liu, Deli Zhao, Jingren Zhou, Fan Cheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11251)] \
20 Jun 2023

**GD-VDM: Generated Depth for better Diffusion-based Video Generation** \
*Ariel Lapid, Idan Achituve, Lior Bracha, Ethan Fetaya* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11173)] \
19 Jun 2023

**Image Harmonization with Diffusion Model** \
*Jiajie Li, Jian Wang, Chen Wang, Jinjun Xiong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.10441)] \
17 Jun 2023


**Training Diffusion Classifiers with Denoising Assistance** \
*Chandramouli Sastry, Sri Harsha Dumpala, Sageev Oore* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09192)] \
15 Jun 2023


**Conditional Human Sketch Synthesis with Explicit Abstraction Control** \
*Dar-Yen Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09274)] \
15 Jun 2023

**Fast Training of Diffusion Models with Masked Transformers** \
*Hongkai Zheng, Weili Nie, Arash Vahdat, Anima Anandkumar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09305)] [[Github](https://github.com/Anima-Lab/MaskDiT)] \
15 Jun 2023


**Relation-Aware Diffusion Model for Controllable Poster Layout Generation** \
*Fengheng Li, An Liu, Wei Feng, Honghe Zhu, Yaoyu Li, Zheng Zhang, Jingjing Lv, Xin Zhu, Junjie Shen, Zhangang Lin, Jingping Shao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09086)] \
15 Jun 2023

**OMS-DPM: Optimizing the Model Schedule for Diffusion Probabilistic Models** \
*Enshu Liu, Xuefei Ning, Zinan Lin, Huazhong Yang, Yu Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08860)] \
15 Jun 2023


**DORSal: Diffusion for Object-centric Representations of Scenes $\textit{et al.}$** \
*Allan Jabri, Sjoerd van Steenkiste, Emiel Hoogeboom, Mehdi S. M. Sajjadi, Thomas Kipf* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08068)] \
13 Jun 2023


**Fast Diffusion Model** \
*Zike Wu, Pan Zhou, Kenji Kawaguchi, Hanwang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.06991)] [[Github](https://github.com/sail-sg/FDM)] \
12 Jun 2023

**ADDP: Learning General Representations for Image Recognition and Generation with Alternating Denoising Diffusion Process** \
*Changyao Tian, Chenxin Tao, Jifeng Dai, Hao Li, Ziheng Li, Lewei Lu, Xiaogang Wang, Hongsheng Li, Gao Huang, Xizhou Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05423)] \
8 Jun 2023

**Multi-Architecture Multi-Expert Diffusion Models** \
*Yunsung Lee, Jin-Young Kim, Hyojun Go, Myeongho Jeong, Shinhyeok Oh, Seungtaek Choi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04990)] \
8 Jun 2023

**Interpreting and Improving Diffusion Models Using the Euclidean Distance Function** \
*Frank Permenter, Chenyang Yuan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04848)] \
8 Jun 2023

**Video Diffusion Models with Local-Global Context Guidance** \
*Siyuan Yang, Lu Zhang, Yu Liu, Zhizhuo Jiang, You He* \
IJCAI 2023. [[Paper](https://arxiv.org/abs/2306.02562)] [[Github](https://github.com/exisas/LGC-VD)] \
5 Jun 2023

**Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models** \
*Andrew F. Luo, Margaret M. Henderson, Leila Wehbe, Michael J. Tarr* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03089)] \
5 Jun 2023

**Faster Training of Diffusion Models and Improved Density Estimation via Parallel Score Matching** \
*Etrit Haxholli, Marco Lorenzi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02658)] \
5 Jun 2023

**Temporal Dynamic Quantization for Diffusion Models** \
*Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, Eunhyeok Park* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02316)] \
4 Jun 2023

**Conditional Generation from Unconditional Diffusion Models using Denoiser Representations** \
*Alexandros Graikos, Srikar Yellapragada, Dimitris Samaras* \
BMVC 2023. [[Paper](https://arxiv.org/abs/2306.01900)] [[Github](https://github.com/cvlab-stonybrook/fewshot-conditional-diffusion)] \
2 Jun 2023

**Conditioning Diffusion Models via Attributes and Semantic Masks for Face Generation** \
*Nico Giambi, Giuseppe Lisanti* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00914)] \
1 Jun 2023

**Differential Diffusion: Giving Each Pixel Its Strength** \
*Eran Levin, Ohad Fried* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00950)] \
1 Jun 2023


**Addressing Discrepancies in Semantic and Visual Alignment in Neural Networks** \
*Natalie Abreu, Nathan Vaska, Victoria Helus* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01148)] \
1 Jun 2023


**Addressing Negative Transfer in Diffusion Models** \
*Hyojun Go, JinYoung Kim, Yunsung Lee, Seunghyun Lee, Shinhyeok Oh, Hyeongdon Moon, Seungtaek Choi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00354)] \
1 Jun 2023

**A Geometric Perspective on Diffusion Models** \
*Defang Chen, Zhenyu Zhou, Jian-Ping Mei, Chunhua Shen, Chun Chen, Can Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19947)] \
31 May 2023



**Spontaneous symmetry breaking in generative diffusion models** \
*Gabriel Raya, Luca Ambrogioni* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19693)] \
31 May 2023

**Perturbation-Assisted Sample Synthesis: A Novel Approach for Uncertainty Quantification** \
*Yifei Liu, Rex Shen, Xiaotong Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18671)] \
30 May 2023

**One-Line-of-Code Data Mollification Improves Optimization of Likelihood-based Generative Models** \
*Ba-Hien Tran, Giulio Franzese, Pietro Michiardi, Maurizio Filippone* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18900)] \
30 May 2023

**Ambient Diffusion: Learning Clean Distributions from Corrupted Data** \
*Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alexandros G. Dimakis, Adam Klivans* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19256)] \
30 May 2023

**Towards Accurate Data-free Quantization for Diffusion Models** \
*Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, Jiwen Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18723)] \
30 May 2023

**BRIGHT: Bi-level Feature Representation of Image Collections using Groups of Hash Tables** \
*Dingdong Yang, Yizhi Wang, Ali Mahdavi-Amiri, Hao Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18601)] [[Project](https://bright-project01.github.io/)] \
29 May 2023

**Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models** \
*Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhihua Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18455)] \
29 May 2023

**Learning to Jump: Thinning and Thickening Latent Counts for Generative Modeling** \
*Tianqi Chen, Mingyuan Zhou* \
ICML 2023. [[Paper](https://arxiv.org/abs/2305.18375)] [[Github](https://github.com/tqch/poisson-jump)] \
28 May 2023

**Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors** \
*Paul S. Scotti, Atmadeep Banerjee, Jimmie Goode, Stepan Shabalin, Alex Nguyen, Ethan Cohen, Aidan J. Dempster, Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth A. Norman, Tanishq Mathew Abraham* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18274)] [[Github](https://medarc-ai.github.io/mindeye/)] \
29 May 2023

**Contrast, Attend and Diffuse to Decode High-Resolution Images from Brain Activities** \
*Jingyuan Sun, Mingxiao Li, Zijiao Chen, Yunhao Zhang, Shaonan Wang, Marie-Francine Moens* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17214)] \
26 May 2023

**Parallel Sampling of Diffusion Models** \
*Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16317)] [[Github](https://github.com/AndyShih12/paradigms)] \
25 May 2023

**Trans-Dimensional Generative Modeling via Jump Diffusion Models** \
*Andrew Campbell, William Harvey, Christian Weilbach, Valentin De Bortoli, Tom Rainforth, Arnaud Doucet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16261)] \
25 May 2023

**UDPM: Upsampling Diffusion Probabilistic Models** \
*Shady Abu-Hussein, Raja Giryes* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16269)] \
25 May 2023


**Unifying GANs and Score-Based Diffusion as Generative Particle Models** \
*Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, Alain Rakotomamonjy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16150)] \
25 May 2023

**DuDGAN: Improving Class-Conditional GANs via Dual-Diffusion** \
*Taesun Yeom, Minhyeok Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14849)] \
24 May 2023

**Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps** \
*Mingxiao Li, Tingyu Qu, Wei Sun, Marie-Francine Moens* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15583)] \
24 May 2023


**Robust Classification via a Single Diffusion Model** \
*Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15241)] \
24 May 2023

**On the Generalization of Diffusion Model** \
*Mingyang Yi, Jiacheng Sun, Zhenguo Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14712)] \
24 May 2023

**VDT: An Empirical Study on Video Diffusion with Transformers** \
*Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, Mingyu Ding* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13311)] [[Github](https://github.com/RERV/VDT)] \
22 May 2023

**Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity** \
*Zijiao Chen, Jiaxin Qing, Juan Helen Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11675)] [[Project](https://mind-video.com/)] \
19 May 2023

**PTQD: Accurate Post-Training Quantization for Diffusion Models** \
*Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, Bohan Zhuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10657)] \
18 May 2023

**Blackout Diffusion: Generative Diffusion Models in Discrete-State Spaces** \
*Javier E Santos, Zachary R. Fox, Nicholas Lubbers, Yen Ting Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11089)] \
18 May 2023

**Structural Pruning for Diffusion Models** \
*Gongfan Fang, Xinyin Ma, Xinchao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10924)] [[Github](https://github.com/VainF/Diff-Pruning)] \
18 May 2023


**Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling** \
*Shitong Shao, Xu Dai, Shouyi Yin, Lujun Li, Huanran Chen, Yang Hu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10769)] \
18 May 2023

**Controllable Mind Visual Diffusion Model** \
*Bohan Zeng, Shanglin Li, Xuhui Liu, Sicheng Gao, Xiaolong Jiang, Xu Tang, Yao Hu, Jianzhuang Liu, Baochang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10135)] \
17 May 2023

**Analyzing Bias in Diffusion-based Face Generation Models** \
*Malsha V. Perera, Vishal M. Patel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06402)] \
10 May 2023


**Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs** \
*Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu* \
ICML 2023. [[Paper](https://arxiv.org/abs/2305.03935)] \
6 May 2023

**LEO: Generative Latent Image Animator for Human Video Synthesis** \
*Yaohui Wang, Xin Ma, Xinyuan Chen, Antitza Dantcheva, Bo Dai, Yu Qiao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03989)] [[Project](https://wyhsirius.github.io/LEO-project/)] [[Github](https://github.com/wyhsirius/LEO)] \
6 May 2023

**Iterative α-(de)Blending: a Minimalist Deterministic Diffusion Model** \
*Eric Heitz, Laurent Belcour, Thomas Chambon* \
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2305.03486)] \
5 May 2023


**Reconstructing seen images from human brain activity via guided stochastic search** \
*Reese Kneeland, Jordyn Ojeda, Ghislain St-Yves, Thomas Naselaris* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.00556)] \
30 Apr 2023


**Motion-Conditioned Diffusion Model for Controllable Video Synthesis** \
*Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, Tsung-Yi Lin, Ming-Hsuan Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.14404)] [[Project](https://tsaishien-chen.github.io/MCDiff/)] \
27 Apr 2023

**Score-based Generative Modeling Through Backward Stochastic Differential Equations: Inversion and Generation** \
*Zihao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13224)] \
26 Apr 2023

**Exploring Compositional Visual Generation with Latent Classifier Guidance** \
*Changhao Shi, Haomiao Ni, Kai Li, Shaobo Han, Mingfu Liang, Martin Renqiang Min* \
CVPR Workshop 2023. [[Paper](https://arxiv.org/abs/2304.12536)] \
25 Apr 2023

**Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models** \
*Zhendong Wang, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang Wang, Weizhu Chen, Mingyuan Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12526)] \
25 Apr 2023


**Variational Diffusion Auto-encoder: Deep Latent Variable Model with Unconditional Diffusion Prior** \
*Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12141)] \
24 Apr 2023


**LaMD: Latent Motion Diffusion for Video Generation** \
*Yaosi Hu, Zhenzhong Chen, Chong Luo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11603)] \
23 Apr 2023


**Lookahead Diffusion Probabilistic Models for Refining Mean Estimation** \
*Guoqiang Zhang, Niwa Kenta, W. Bastiaan Kleijn* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.11312)] [[Github](https://github.com/guoqiang-zhang-x/LA-DPM)] \
22 Apr 2023

**NeuralField-LDM: Scene Generation with Hierarchical Latent Diffusion Models** \
*Seung Wook Kim, Bradley Brown, Kangxue Yin, Karsten Kreis, Katja Schwarz, Daiqing Li, Robin Rombach, Antonio Torralba, Sanja Fidler* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.09787)] \
19 Apr 2023

**Attributing Image Generative Models using Latent Fingerprints** \
*Guangyu Nie, Changhoon Kim, Yezhou Yang, Yi Ren* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.09752)] \
17 Apr 2023


**Identity Encoder for Personalized Diffusion** \
*Yu-Chuan Su, Kelvin C.K. Chan, Yandong Li, Yang Zhao, Han Zhang, Boqing Gong, Huisheng Wang, Xuhui Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.07429)] \
14 Apr 2023

**Memory Efficient Diffusion Probabilistic Models via Patch-based Generation** \
*Shinei Arakawa, Hideki Tsunashima, Daichi Horita, Keitaro Tanaka, Shigeo Morishima* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.07087)] \
14 Apr 2023

**DCFace: Synthetic Face Generation with Dual Condition Diffusion Model** \
*Minchul Kim, Feng Liu, Anil Jain, Xiaoming Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.07060)] [[Github](https://github.com/mk-minchul/dcface)] \
14 Apr 2023

**DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning** \
*Enze Xie, Lewei Yao, Han Shi, Zhili Liu, Daquan Zhou, Zhaoqiang Liu, Jiawei Li, Zhenguo Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06648)] \
13 Apr 2023

**RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment** \
*Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, Tong Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06767)] \
13 Apr 2023

**DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion** \
*Johanna Karras, Aleksander Holynski, Ting-Chun Wang, Ira Kemelmacher-Shlizerman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06025)] [[Project](https://grail.cs.washington.edu/projects/dreampose/)] [[Github](https://github.com/johannakarras/DreamPose)] \
12 Apr 2023

**Reflected Diffusion Models** \
*Aaron Lou, Stefano Ermon* \
ICML 2023. [[Paper](https://arxiv.org/abs/2304.04740)] [[Project](https://aaronlou.com/blog/2023/reflected-diffusion/)] [[Github](https://github.com/louaaron/Reflected-Diffusion)] \
10 Apr 2023

**Binary Latent Diffusion** \
*Ze Wang, Jiang Wang, Zicheng Liu, Qiang Qiu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04820)] \
10 Apr 2023


**Diffusion Models as Masked Autoencoders** \
*Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, Christoph Feichtenhofer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03283)] [[Project](https://weichen582.github.io/diffmae.html)] \
6 Apr 2023

**Few-shot Semantic Image Synthesis with Class Affinity Transfer** \
*Marlène Careil, Jakob Verbeek, Stéphane Lathuilière* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.02321)] \
5 Apr 2023


**EGC: Image Generation and Classification via a Diffusion Energy-Based Model** \
*Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, Ping Luo* \
arxiv 2023. [[Paper](https://arxiv.org/abs/2304.02012)] [[Project](https://guoqiushan.github.io/egc.github.io/)] \
4 Apr 2023



**Token Merging for Fast Stable Diffusion** \
*Daniel Bolya, Judy Hoffman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17604)] [[Github](https://github.com/dbolya/tomesd)] \
30 Mar 2023

**A Closer Look at Parameter-Efficient Tuning in Diffusion Models** \
*Chendong Xiang, Fan Bao, Chongxuan Li, Hang Su, Jun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.18181)] \
31 Mar 2023

**-Diff: Infinite Resolution Diffusion with Subsampled Mollified States** \
*Sam Bond-Taylor, Chris G. Willcocks* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.18242)] \
31 Mar 2023

**3D-aware Image Generation using 2D Diffusion Models** \
*Jianfeng Xiang, Jiaolong Yang, Binbin Huang, Xin Tong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17905)] [[Project](https://jeffreyxiang.github.io/ivid/)] \
31 Mar 2023

**Consistent View Synthesis with Pose-Guided Diffusion Models** \
*Hung-Yu Tseng, Qinbo Li, Changil Kim, Suhib Alsisan, Jia-Bin Huang, Johannes Kopf* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.17598)] \
30 Mar 2023


**DiffCollage: Parallel Generation of Large Content with Diffusion Models** \
*Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, Ming-Yu Liu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.17076)] [[Project](https://research.nvidia.com/labs/dir/diffcollage/)] \
30 Mar 2023

**Masked Diffusion Transformer is a Strong Image Synthesizer** \
*Shanghua Gao, Pan Zhou, Ming-Ming Cheng, Shuicheng Yan* \
arXiv 2023.  [[Paper](https://arxiv.org/abs/2303.14389)] [[Github](https://github.com/sail-sg/MDT)] \
25 Mar 2023

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models** \
*Haomiao Ni, Changhao Shi, Kai Li, Sharon X. Huang, Martin Renqiang Min* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.13744)] [[Github](https://github.com/nihaomiao/CVPR23_LFDM)] \
24 Mar 2023

**NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation** \
*Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, Jianlong Fu, Gong Ming, Lijuan Wang, Zicheng Liu, Houqiang Li, Nan Duan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12346)] [[Project](https://msra-nuwa.azurewebsites.net/#/)] \
22 Mar 2023

**Object-Centric Slot Diffusion** \
*Jindong Jiang, Fei Deng, Gautam Singh, Sungjin Ahn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10834)] \
20 Mar 2023


**LDMVFI: Video Frame Interpolation with Latent Diffusion Models** \
*Duolikun Danier, Fan Zhang, David Bull* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09508)] \
16 Mar 2023

**Efficient Diffusion Training via Min-SNR Weighting Strategy** \
*Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, Baining Guo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09556)] \
16 Mar 2023

**VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation** \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.08320)] \
15 Mar 2023

**Interpretable ODE-style Generative Diffusion Model via Force Field Construction** \
*Weiyang Jin, Yongpei Zhu, Yuxi Peng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08063)] \
14 Mar 2023

**Regularized Vector Quantization for Tokenized Image Synthesis** \
*Jiahui Zhang, Fangneng Zhan, Christian Theobalt, Shijian Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06424)] \
11 Mar 2023


**PARASOL: Parametric Style Control for Diffusion Image Synthesis** \
*Gemma Canet Tarrés, Dan Ruta, Tu Bui, John Collomosse* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06464)] \
11 Mar 2023

**Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion** \
*Furkan Ozcelik, Rufin VanRullen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05334)] \
9 Mar 2023

**Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation** \
*Paul Hagemann, Lars Ruthotto, Gabriele Steidl, Nicole Tianjiao Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04772)] \
8 Mar 2023


**TRACT: Denoising Diffusion Models with Transitive Closure Time-Distillation** \
*David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbott, Eric Gu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04248)] \
7 Mar 2023

**Generative Diffusions in Augmented Spaces: A Complete Recipe** \
*Kushagra Pandey, Stephan Mandt* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.01748)] \
3 Mar 2023

**Consistency Models** \
*Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.01469)] \
2 Mar 2023

**Diffusion Probabilistic Fields** \
*Peiye Zhuang, Samira Abnar, Jiatao Gu, Alex Schwing, Joshua M. Susskind, Miguel Ángel Bautista* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2303.00165)] \
1 Mar 2023

**Unsupervised Discovery of Semantic Latent Directions in Diffusion Models** \
*Yong-Hyun Park, Mingi Kwon, Junghyo Jo, Youngjung Uh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12469)] \
24 Feb 2023

**Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC** \
*Yilun Du, Conor Durkan, Robin Strudel, Joshua B. Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, Will Grathwohl* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.11552)] [[Project](https://energy-based-model.github.io/reduce-reuse-recycle/)] \
22 Feb 2023

**Learning 3D Photography Videos via Self-supervised Diffusion on Single Images** \
*Xiaodong Wang, Chenfei Wu, Shengming Yin, Minheng Ni, Jianfeng Wang, Linjie Li, Zhengyuan Yang, Fan Yang, Lijuan Wang, Zicheng Liu, Yuejian Fang, Nan Duan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10781)] \
21 Feb 2023

**On Calibrating Diffusion Probabilistic Models** \
*Tianyu Pang, Cheng Lu, Chao Du, Min Lin, Shuicheng Yan, Zhijie Deng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10688)] [[Github](https://github.com/thudzj/Calibrated-DPMs)] \
21 Feb 2023

**Diffusion Models and Semi-Supervised Learners Benefit Mutually with Few Labels** \
*Zebin You, Yong Zhong, Fan Bao, Jiacheng Sun, Chongxuan Li, Jun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10586)] \
21 Feb 2023

**Cross-domain Compositing with Pretrained Diffusion Models** \
*Roy Hachnochi, Mingrui Zhao, Nadav Orzech, Rinon Gal, Ali Mahdavi-Amiri, Daniel Cohen-Or, Amit Haim Bermano* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10167)] [[Github](https://github.com/cross-domain-compositing/cross-domain-compositing)] \
20 Feb 2023



**Restoration based Generative Models** \
*Jaemoo Choi, Yesom Park, Myungjoo Kang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05456)] \
20 Feb 2023



**Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent** \
*Giannis Daras, Yuval Dagan, Alexandros G. Dimakis, Constantinos Daskalakis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.09057)] [[Github](https://github.com/giannisdaras/cdm)] \
17 Feb 2023

**LayoutDiffuse: Adapting Foundational Diffusion Models for Layout-to-Image Generation** \
*Jiaxin Cheng, Xiao Liang, Xingjian Shi, Tong He, Tianjun Xiao, Mu Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08908)] \
16 Feb 2023

**Video Probabilistic Diffusion Models in Projected Latent Space** \
*Sihyun Yu, Kihyuk Sohn, Subin Kim, Jinwoo Shin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07685)] [[Github](https://sihyun.me/PVDM/)] \
15 Feb 2023

**DiffFaceSketch: High-Fidelity Face Image Synthesis with Sketch-Guided Latent Diffusion Model** \
*Yichen Peng, Chunqi Zhao, Haoran Xie, Tsukasa Fukusato, Kazunori Miyata* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.06908)] \
14 Feb 2023

**Where to Diffuse, How to Diffuse, and How to Get Back: Automated Learning for Multivariate Diffusions** \
*Raghav Singhal, Mark Goldstein, Rajesh Ranganath* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07261)] \
14 Feb 2023


**Preconditioned Score-based Generative Models** \
*Li Zhang, Hengyuan Ma, Xiatian Zhu, Jianfeng Feng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.06504)] [Github](https://github.com/fudan-zvg/PDS)] \
13 Feb 2023

**Star-Shaped Denoising Diffusion Probabilistic Models** \
*Andrey Okhotin, Dmitry Molchanov, Vladimir Arkhipkin, Grigory Bartosh, Aibek Alanov, Dmitry Vetrov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05259)] \
10 Feb 2023 


**UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models** \
*Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04867)] [[Project](https://unipc.ivg-research.xyz)] [[Github](https://github.com/wl-zhao/UniPC)] \
9 Feb 2023

**Geometry of Score Based Generative Models** \
*Sandesh Ghimire, Jinyang Liu, Armand Comas, Davin Hill, Aria Masoomi, Octavia Camps, Jennifer Dy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04411)] \
9 Feb 2023

**Q-Diffusion: Quantizing Diffusion Models** \
*Xiuyu Li, Long Lian, Yijiang Liu, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, Kurt Keutzer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04304)] \
8 Feb 2023

**PFGM++: Unlocking the Potential of Physics-Inspired Generative Models** \
*Yilun Xu, Ziming Liu, Yonglong Tian, Shangyuan Tong, Max Tegmark, Tommi Jaakkola* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04265)] [[Github](https://github.com/Newbeeer/pfgmpp)] \
8 Feb 2023

**Long Horizon Temperature Scaling** \
*Andy Shih, Dorsa Sadigh, Stefano Ermon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03686)] \
7 Feb 2023

**Spatial Functa: Scaling Functa to ImageNet Classification and Generation** \
*Matthias Bauer, Emilien Dupont, Andy Brock, Dan Rosenbaum, Jonathan Schwarz, Hyunjik Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03130)] \
6 Feb 2023

**ShiftDDPMs: Exploring Conditional Diffusion Models by Shifting Diffusion Trajectories** \
*Zijian Zhang, Zhou Zhao, Jun Yu, Qi Tian* \
AAAI 2023. [[Paper](https://arxiv.org/abs/2302.02373)] \
5 Feb 2023

**Divide and Compose with Score Based Generative Models** \
*Sandesh Ghimire, Armand Comas, Davin Hill, Aria Masoomi, Octavia Camps, Jennifer Dy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02272)] [[Github](https://github.com/sandeshgh/Score-based-disentanglement)] \
5 Feb 2023


**Stable Target Field for Reduced Variance Score Estimation in Diffusion Models** \
*Yilun Xu, Shangyuan Tong, Tommi Jaakkola* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2302.00670)] [[Github](https://github.com/Newbeeer/stf)] \
1 Feb 2023

**DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models** \
*Tao Yang, Yuwang Wang, Yan Lv, Nanning Zheng* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2301.13721)] \
31 Jan 2023


**Optimizing DDPM Sampling with Shortcut Fine-Tuning** \
*Ying Fan, Kangwook Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13362)] \
31 Jan 2023

**Learning Data Representations with Joint Diffusion Models** \
*Kamil Deja, Tomasz Trzcinski, Jakub M. Tomczak* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13622)] \
31 Jan 2023

**ERA-Solver: Error-Robust Adams Solver for Fast Sampling of Diffusion Probabilistic Models** \
*Shengmeng Li, Luping Liu, Zenghao Chai, Runnan Li, Xu Tan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12935)] \
30 Jan 2023

**Don't Play Favorites: Minority Guidance for Diffusion Models** \
*Soobin Um, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12334)] [[Github](https://github.com/sangyun884/fast-ode)] \
29 Jan 2023

**Accelerating Guided Diffusion Sampling with Splitting Numerical Methods** \
*Suttisak Wizadwongsa, Supasorn Suwajanakorn* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2301.11558)] \
27 Jan 2023

**Input Perturbation Reduces Exposure Bias in Diffusion Models** \
*Mang Ning, Enver Sangineto, Angelo Porrello, Simone Calderara, Rita Cucchiara* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11706)] [[Github](https://github.com/forever208/DDPM-IP)] \
27 Jan 2023

**Minimizing Trajectory Curvature of ODE-based Generative Models** \
*Sangyun Lee, Beomsu Kim, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12003)] \
27 Jan 2023


**On the Importance of Noise Scheduling for Diffusion Models** \
*Ting Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.10972)] \
26 Jan 2023

**simple diffusion: End-to-end diffusion for high resolution images** \
*Emiel Hoogeboom, Jonathan Heek, Tim Salimans* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11093)] \
26 Jan 2023

**Fast Inference in Denoising Diffusion Models via MMD Finetuning** \
*Emanuele Aiello, Diego Valsesia, Enrico Magli* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.07969)] [[Github](https://github.com/diegovalsesia/MMD-DDM)] \
19 Jan 2023

**Exploring Transformer Backbones for Image Diffusion Models** \
*Princy Chahal* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.14678)] \
27 Dec 2022

**Unsupervised Representation Learning from Pre-trained Diffusion Probabilistic Models** \
*Zijian Zhang, Zhou Zhao, Zhijie Lin* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.12990)] \
26 Dec 2022


**Scalable Adaptive Computation for Iterative Generation** \
*Allan Jabri, David Fleet, Ting Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.11972)] \
22 Dec 2022

**Hierarchically branched diffusion models for efficient and interpretable multi-class conditional generation** \
*Alex M. Tseng, Tommaso Biancalani, Max Shen, Gabriele Scalia* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10777)] \
21 Dec 2022


**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation** \
*Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, Baining Guo* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09478)] [[Github](https://github.com/researchmm/MM-Diffusion)] \
19 Dec 2022


**Scalable Diffusion Models with Transformers** \
*William Peebles, Saining Xie* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09748)] [[Project](https://www.wpeebles.com/DiT)] [[Github](https://github.com/facebookresearch/DiT)] \
19 Dec 2022


**DAG: Depth-Aware Guidance with Denoising Diffusion Probabilistic Models** \
*Gyeongnyeon Kim, Wooseok Jang, Gyuseong Lee, Susung Hong, Junyoung Seo, Seungryong Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08861)] [[Project](https://ku-cvlab.github.io/DAG/)] \
17 Dec 2022


**Towards Practical Plug-and-Play Diffusion Models** \
*Hyojun Go, Yunsung Lee, Jin-Young Kim, Seunghyun Lee, Myeongho Jeong, Hyun Seung Lee, Seungtaek Choi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.05973)] \
12 Dec 2022

**Semantic Brain Decoding: from fMRI to conceptually similar image reconstruction of visual stimuli** \
*Matteo Ferrante, Tommaso Boccato, Nicola Toschi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06726)] \
13 Dec 2022

**MAGVIT: Masked Generative Video Transformer** \
*Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, Lu Jiang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.05199)] [Project](https://magvit.cs.cmu.edu/)] \
10 Dec 2022

**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding** \
*Gyeongman Kim, Hajin Shim, Hyunsu Kim, Yunjey Choi, Junho Kim, Eunho Yang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02802)] \
6 Dec 2022

**Fine-grained Image Editing by Pixel-wise Guidance Using Diffusion Models** \
*Naoki Matsunaga, Masato Ishii, Akio Hayakawa, Kenji Suzuki, Takuya Narihira* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02024)] \
5 Dec 2022


**VIDM: Video Implicit Diffusion Models** \
*Kangfu Mei, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00235)] [[Project](https://kfmei.page/vidm/)] [[Github](https://github.com/MKFMIKU/VIDM)] \
1 Dec 2022

**Why Are Conditional Generative Models Better Than Unconditional Ones?** \
*Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00362)] \
1 Dec 2022


**High-Fidelity Guided Image Synthesis with Latent Diffusion Models** \
*Jaskirat Singh, Stephen Gould, Liang Zheng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.17084)] [[Project](https://1jsingh.github.io/gradop)] \
30 Nov 2022


**Score-based Continuous-time Discrete Diffusion Models** \
*Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, Hanjun Dai* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16750)] \
30 Nov 2022

**Wavelet Diffusion Models are fast and scalable Image Generators** \
*Hao Phung, Quan Dao, Anh Tran* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16152)] \
29 Nov 2022


**Dimensionality-Varying Diffusion Process** \
*Han Zhang, Ruili Feng, Zhantao Yang, Lianghua Huang, Yu Liu, Yifei Zhang, Yujun Shen, Deli Zhao, Jingren Zhou, Fan Cheng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16032)] \
29 Nov 2022

**Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models** \
*Dongjun Kim, Yeongmin Kim, Wanmo Kang, Il-Chul Moon* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.17091)] \
28 Nov 2022



**Diffusion Probabilistic Model Made Slim** \
*Xingyi Yang, Daquan Zhou, Jiashi Feng, Xinchao Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.17106)] \
27 Nov 2022


**Fast Sampling of Diffusion Models via Operator Learning** \
*Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, Anima Anandkumar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13449)] \
24 Nov 2022

**Latent Video Diffusion Models for High-Fidelity Video Generation with Arbitrary Lengths** \
*Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, Qifeng Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13221)] \
23 Nov 2022



**Paint by Example: Exemplar-based Image Editing with Diffusion Models** \
*Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, Fang Wen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13227)] \
23 Nov 2022


**SinDiffusion: Learning a Diffusion Model from a Single Natural Image** \
*Weilun Wang, Jianmin Bao, Wengang Zhou, Dongdong Chen, Dong Chen, Lu Yuan, Houqiang Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12445)] [[Github](https://github.com/WeilunWang/SinDiffusion)] \
22 Nov 2022

**Accelerating Diffusion Sampling with Classifier-based Feature Distillation** \
*Wujie Sun, Defang Chen, Can Wang, Deshi Ye, Yan Feng, Chun Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12039)] \
22 Nov 2022

**SceneComposer: Any-Level Semantic Image Synthesis** \
*Yu Zeng, Zhe Lin, Jianming Zhang, Qing Liu, John Collomosse, Jason Kuen, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11742)] [[Project](https://zengyu.me/scenec/)] \
21 Nov 2022

**Diffusion-Based Scene Graph to Image Generation with Masked Contrastive Pre-Training** \
*Ling Yang, Zhilin Huang, Yang Song, Shenda Hong, Guohao Li, Wentao Zhang, Bin Cui, Bernard Ghanem, Ming-Hsuan Yang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11138)] \
21 Nov 2022

**SinFusion: Training Diffusion Models on a Single Image or Video** \
*Yaniv Nikankin, Niv Haim, Michal Irani* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11743)] \
21 Nov 2022

**MagicVideo: Efficient Video Generation With Latent Diffusion Models** \
*Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, Jiashi Feng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11018)] [[Project](https://magicvideo.github.io/)] \
20 Nov 2022

**Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding** \
*Zijiao Chen, Jiaxin Qing, Tiange Xiang, Wan Lin Yue, Juan Helen Zhou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.06956)] [[Project](https://mind-vis.github.io/)] [[Github](https://github.com/zjc062/mind-vis)] \
13 Nov 2022

**Few-shot Image Generation with Diffusion Models** \
*Jingyuan Zhu, Huimin Ma, Jiansheng Chen, Jian Yuan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.03264)] \
7 Nov 2022

**From Denoising Diffusions to Denoising Markov Models** \
*Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, Arnaud Doucet* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.03595)] [[Github](https://github.com/yuyang-shi/generalized-diffusion)] \
7 Nov 2022


**Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models** \
*Muyang Li, Ji Lin, Chenlin Meng, Stefano Ermon, Song Han, Jun-Yan Zhu* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2211.02048)] [[Github](https://github.com/lmxyy/sige)] \
4 Nov 2022

**An optimal control perspective on diffusion-based generative modeling** \
*Julius Berner, Lorenz Richter, Karen Ullrich* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2211.01364)] \
2 Nov 2022

**Entropic Neural Optimal Transport via Diffusion Processes** \
*Nikita Gushchin, Alexander Kolesov, Alexander Korotin, Dmitry Vetrov, Evgeny Burnaev* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01156)] \
2 Nov 2022

**DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models** \
*Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu* \
NeurIPS 2022 (Oral). [[Paper](https://arxiv.org/abs/2211.01095)] [[Github](https://github.com/LuChengTHU/dpm-solver)] \
2 Nov 2022

**Score-based Denoising Diffusion with Non-Isotropic Gaussian Noise Models** \
*Vikram Voleti, Christopher Pal, Adam Oberman* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2210.12254)] \
21 Oct 2022


**Deep Equilibrium Approaches to Diffusion Models** \
*Ashwini Pokle, Zhengyang Geng, Zico Kolter* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2210.12867)] [[Github](https://github.com/locuslab/deq-ddim)] \
23 Oct 2022

**Representation Learning with Diffusion Models** \
*Jeremias Traub* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11058)] [[Github](https://github.com/jeremiastraub/diffusion)] \
20 Oct 2022

**Self-Guided Diffusion Models** \
*Vincent Tao Hu, David W Zhang, Yuki M. Asano, Gertjan J. Burghouts, Cees G. M. Snoek* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.06462)] [[Project](http://taohu.me/sgdm/)] \
12 Oct 2022

**GENIE: Higher-Order Denoising Diffusion Solvers** \
*Tim Dockhorn, Arash Vahdat, Karsten Kreis* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2210.05475)] [[Project](https://nv-tlabs.github.io/GENIE/) [[Github](https://github.com/nv-tlabs/GENIE)] \
11 Oct 2022

**f-DM: A Multi-stage Diffusion Model via Progressive Signal Transformation** \
*Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Miguel Angel Bautista, Josh Susskind* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.04955)] [[Project](http://jiataogu.me/fdm/)] \
10 Oct 2022

**On Distillation of Guided Diffusion Models** \
*Chenlin Meng, Ruiqi Gao, Diederik P. Kingma, Stefano Ermon, Jonathan Ho, Tim Salimans* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.03142)] \
6 Oct 2022


**Improving Sample Quality of Diffusion Model Using Self-Attention Guidance** \
*Susung Hong, Gyuseong Lee, Wooseok Jang, Seungryong Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.00939)] [[Project](https://ku-cvlab.github.io/Self-Attention-Guidance/)] \
3 Oct 2022

**OCD: Learning to Overfit with Conditional Diffusion Models** \
*Shahar Shlomo Lutati, Lior Wolf* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.00471)] [[Github](https://github.com/ShaharLutatiPersonal/OCD)] \
2 Oct 2022

**Generated Faces in the Wild: Quantitative Comparison of Stable Diffusion, Midjourney and DALL-E 2** \
*Ali Borji* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.00586)] [[Github](https://github.com/aliborji/GFW)] \
2 Oct 2022

**Denoising MCMC for Accelerating Diffusion-Based Generative Models** \
*Beomsu Kim, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14593)] [[Github](https://github.com/1202kbs/DMCMC)] \
29 Sep 2022

**All are Worth Words: a ViT Backbone for Score-based Diffusion Models** \
*Fan Bao, Chongxuan Li, Yue Cao, Jun Zhu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.12152)] \
25 Sep 2022


**Neural Wavelet-domain Diffusion for 3D Shape Generation** \
*Ka-Hei Hui, Ruihui Li, Jingyu Hu, Chi-Wing Fu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.08725)] \
19 Sep 2022

**Can segmentation models be trained with fully synthetically generated data?** \
*Virginia Fernandez, Walter Hugo Lopez Pinaya, Pedro Borges, Petru-Daniel Tudosiu, Mark S Graham, Tom Vercauteren, M Jorge Cardoso* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.08256)] \
17 Sep 2022

**Blurring Diffusion Models** \
*Emiel Hoogeboom, Tim Salimans* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.05557)] \
12 Sep 2022

**Soft Diffusion: Score Matching for General Corruptions** \
*Giannis Daras, Mauricio Delbracio, Hossein Talebi, Alexandros G. Dimakis, Peyman Milanfar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.05442)] \
12 Sep 2022

**Improved Masked Image Generation with Token-Critic** \
*José Lezama, Huiwen Chang, Lu Jiang, Irfan Essa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.04439)] \
9 Sep 2022


**Let us Build Bridges: Understanding and Extending Diffusion Generative Models** \
*Xingchao Liu, Lemeng Wu, Mao Ye, Qiang Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.14699)] \
31 Aug 2022

**Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis** \
*Wan-Cyuan Fan, Yen-Chun Chen, DongDong Chen, Yu Cheng, Lu Yuan, Yu-Chiang Frank Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.13753)] \
29 Aug 2022


**Adaptively-Realistic Image Generation from Stroke and Sketch with Diffusion Model** \
*Shin-I Cheng, Yu-Jie Chen, Wei-Chen Chiu, Hsin-Ying Lee, Hung-Yu Tseng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.12675)] [[Project](https://cyj407.github.io/DiSS/)] \
26 Aug 2022

**Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise** \
*Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.09392)] [[Github](https://github.com/arpitbansal297/Cold-Diffusion-Models)] \
19 Aug 2022

**Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance** \
*Bahjat Kawar, Roy Ganz, Michael Elad* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.08664)] \
18 Aug 2022

**Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model** \
*Xiulong Yang, Sheng-Min Shih, Yinlin Fu, Xiaoting Zhao, Shihao Ji* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.07791)] [[Github](https://github.com/sndnyang/Diffusion_ViT)] \
16 Aug 2022



**Applying Regularized Schrödinger-Bridge-Based Stochastic Process in Generative Modeling** \
*Ki-Ung Song* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.07131)] [[Github](https://github.com/KiUngSong/RSB)] \
15 Aug 2022

**Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning** \
*Ting Chen, Ruixiang Zhang, Geoffrey Hinton* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.04202)] \
8 Aug 2022


**Pyramidal Denoising Diffusion Probabilistic Models** \
*Dohoon Ryu, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.01864)] \
3 Aug 2022

**Progressive Deblurring of Diffusion Models for Coarse-to-Fine Image Synthesis** \
*Sangyun Lee, Hyungjin Chung, Jaehyeon Kim, Jong Chul Ye* \
arxiv 2022. [[Paper](https://arxiv.org/abs/2207.11192)] [[Github](https://github.com/sangyun884/blur-diffusion)] \
16 Jul 2022

**Improving Diffusion Model Efficiency Through Patching** \
*Troy Luhman, Eric Luhman* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.04316)] [[Github](https://github.com/ericl122333/PatchDiffusion-Pytorch)] \
9 Jul 2022

**Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling** \
*Hengyuan Ma, Li Zhang, Xiatian Zhu, Jianfeng Feng* \
ECCV 2022. [[Paper](https://arxiv.org/abs/2207.02196)] \
5 Jul 2022

**SPI-GAN: Distilling Score-based Generative Models with Straight-Path Interpolations** \
*Jinsung Jeon, Noseong Park* \
arxiv 2022. [[Paper](https://arxiv.org/abs/2206.14464)] \
29 Jun 2022

**Entropy-driven Sampling and Training Scheme for Conditional Diffusion Generation** \
*Shengming Li, Guangcong Zheng, Hui Wang, Taiping Yao, Yang Chen, Shoudong Ding, Xi Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.11474)] \
23 Jun 2022

**Generative Modelling With Inverse Heat Dissipation** \
*Severi Rissanen, Markus Heinonen, Arno Solin* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.13397)] [[Project](https://aaltoml.github.io/generative-inverse-heat-dissipation/)] \
21 Jun 2022

**Diffusion models as plug-and-play priors** \
*Alexandros Graikos, Nikolay Malkin, Nebojsa Jojic, Dimitris Samaras* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.09012)] [[Github](https://github.com/alexgraikos/diffusion_priors)] \
17 Jun 2022

**A Flexible Diffusion Model** \
*Weitao Du, Tao Yang, He Zhang, Yuanqi Du* \
ICML 2023. [[Paper](https://arxiv.org/abs/2206.10365)] \
17 Jun 2022

**Lossy Compression with Gaussian Diffusion** \
*Lucas Theis, Tim Salimans, Matthew D. Hoffman, Fabian Mentzer* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.08889)] \
17 Jun 2022

**Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching** \
*Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu* \
ICML 2022. [[Paper](https://arxiv.org/abs/2206.08265)] [[Github](https://github.com/LuChengTHU/mle_score_ode)] \
16 Jun 2022

**Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models** \
*Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, Bo Zhang* \
ICML 2022. [[Paper](https://arxiv.org/abs/2206.07309)] [[Github](https://github.com/baofff/Extended-Analytic-DPM)] \
15 Jun 2022


**Diffusion Models for Video Prediction and Infilling** \
*Tobias Höppe, Arash Mehrjou, Stefan Bauer, Didrik Nielsen, Andrea Dittadi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.07696)] \
15 Jun 2022

**Discrete Contrastive Diffusion for Cross-Modal and Conditional Generation** \
*Ye Zhu, Yu Wu, Kyle Olszewski, Jian Ren, Sergey Tulyakov, Yan Yan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.07771)] [[Github](https://github.com/L-YeZhu/CDCD)] \
15 Jun 2022

**gDDIM: Generalized denoising diffusion implicit models** \
*Qinsheng Zhang, Molei Tao, Yongxin Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.05564)] [[Github](https://github.com/qsh-zh/gDDIM)] \
11 Jun 2022

**How Much is Enough? A Study on Diffusion Times in Score-based Generative Models** \
*Giulio Franzese, Simone Rossi, Lixuan Yang, Alessandro Finamore, Dario Rossi, Maurizio Filippone, Pietro Michiardi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.05173)] \
10 Jun 2022

**Image Generation with Multimodal Priors using Denoising Diffusion Probabilistic Models** \
*Nithin Gopalakrishnan Nair, Wele Gedara Chaminda Bandara, Vishal M Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.05039)] \
10 Jun 2022

**Accelerating Score-based Generative Models for High-Resolution Image Synthesis** \
*Hengyuan Ma, Li Zhang, Xiatian Zhu, Jingfeng Zhang, Jianfeng Feng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.04029)] \
8 Jun 2022

**Diffusion-GAN: Training GANs with Diffusion** \
*Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, Mingyuan Zhou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.02262)] \
5 Jun 2022

**DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps** \
*Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu* \
NeurrIPS 2022. [[Paper](https://arxiv.org/abs/2206.00927)] [[Github](https://github.com/LuChengTHU/dpm-solver)] \
2 Jun 2022

**Elucidating the Design Space of Diffusion-Based Generative Models** \
*Tero Karras, Miika Aittala, Timo Aila, Samuli Laine* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.00364)] \
1 Jun 2022

**On Analyzing Generative and Denoising Capabilities of Diffusion-based Deep Generative Models** \
*Kamil Deja, Anna Kuzina, Tomasz Trzciński, Jakub M. Tomczak* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.00070)] \
31 May 2022

**Few-Shot Diffusion Models** \
*Giorgio Giannone, Didrik Nielsen, Ole Winther* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.15463)] \
30 May 2022

**A Continuous Time Framework for Discrete Denoising Models** \
*Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, Arnaud Doucet* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.14987)] \
30 May 2022

**Maximum Likelihood Training of Implicit Nonlinear Diffusion Models** \
*Dongjun Kim, Byeonghu Na, Se Jung Kwon, Dongsoo Lee, Wanmo Kang, Il-Chul Moon* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.13699)] \
27 May 2022

**Accelerating Diffusion Models via Early Stop of the Diffusion Process** \
*Zhaoyang Lyu, Xudong XU, Ceyuan Yang, Dahua Lin, Bo Dai* \
ICML 2022. [[Paper](https://arxiv.org/abs/2205.12524)] \
25 May 2022



**Flexible Diffusion Modeling of Long Videos** \
*William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, Frank Wood* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.11495)] [[Github](https://github.com/plai-group/flexible-video-diffusion-modeling)] \
23 May 2022

**MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation** \
*Vikram Voleti, Alexia Jolicoeur-Martineau, Christopher Pal* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.09853)] [[Github](https://github.com/voletiv/mcvd-pytorch)] \
19 May 2022

**On Conditioning the Input Noise for Controlled Image Generation with Diffusion Models** \
*Vedant Singh, Surgan Jandial, Ayush Chopra, Siddharth Ramesh, Balaji Krishnamurthy, Vineeth N. Balasubramanian* \
CVPR Workshop 2022. [[Paper](https://arxiv.org/abs/2205.03859)] \
8 May 2022

**Subspace Diffusion Generative Models** \
*Bowen Jing, Gabriele Corso, Renato Berlinghieri, Tommi Jaakkola* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.01490)] [[Github](https://github.com/bjing2016/subspace-diffusion)] \
3 May 2022

**Fast Sampling of Diffusion Models with Exponential Integrator** \
*Qinsheng Zhang, Yongxin Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.13902)] \
29 Apr 2022

**Semi-Parametric Neural Image Synthesis** \
*Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas Müller, Björn Ommer* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2204.11824)] \
25 Apr 2022


**Video Diffusion Models** \
*Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2204.03458)] \
7 Apr 2022

**Perception Prioritized Training of Diffusion Models** \
*Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, Sungroh Yoon* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2204.00227)] [[Github](https://github.com/jychoi118/P2-weighting)] \
1 Apr 2022

**Generating High Fidelity Data from Low-density Regions using Diffusion Models** \
*Vikash Sehwag, Caner Hazirbas, Albert Gordo, Firat Ozgenel, Cristian Canton Ferrer* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.17260)] \
31 Mar 2022

**Diffusion Models for Counterfactual Explanations** \
*Guillaume Jeanneret, Loïc Simon, Frédéric Jurie* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.15636)] \
29 Mar 2022

**Denoising Likelihood Score Matching for Conditional Score-based Data Generation** \
*Chen-Hao Chao, Wei-Fang Sun, Bo-Wun Cheng, Yi-Chen Lo, Chia-Che Chang, Yu-Lun Liu, Yu-Lin Chang, Chia-Ping Chen, Chun-Yi Lee* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2203.14206)] \
27 Mar 2022

**Diffusion Probabilistic Modeling for Video Generation** \
*Ruihan Yang, Prakhar Srivastava, Stephan Mandt* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.09481)] [[Github](https://github.com/buggyyang/rvd)] \
16 Mar 2022

**Dynamic Dual-Output Diffusion Models** \
*Yaniv Benny, Lior Wolf* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2203.04304)] \
8 Mar 2022

**Conditional Simulation Using Diffusion Schrödinger Bridges** \
*Yuyang Shi, Valentin De Bortoli, George Deligiannidis, Arnaud Doucet* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.13460)] \
27 Feb 2022

**Diffusion Causal Models for Counterfactual Estimation** \
*Pedro Sanchez, Sotirios A. Tsaftaris* \
PMLR 2022. [[Paper](https://arxiv.org/abs/2202.10166)] \
21 Feb 2022

**Pseudo Numerical Methods for Diffusion Models on Manifolds** \
*Luping Liu, Yi Ren, Zhijie Lin, Zhou Zhao* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2202.09778)] [[Github](https://github.com/luping-liu/PNDM)] \
20 Feb 2022

**Truncated Diffusion Probabilistic Models** \
*Huangjie Zheng, Pengcheng He, Weizhu Chen, Mingyuan Zhou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.09671)] \
19 Feb 2022

**Understanding DDPM Latent Codes Through Optimal Transport** \
*Valentin Khrulkov, Ivan Oseledets* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.07477)] \
14 Feb 2022

**Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality** \
*Daniel Watson, William Chan, Jonathan Ho, Mohammad Norouzi* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2202.05830)] \
11 Feb 2022


**Diffusion bridges vector quantized Variational AutoEncoders** \
*Max Cohen, Guillaume Quispe, Sylvain Le Corff, Charles Ollion, Eric Moulines* \
ICML 2022. [[Paper](https://arxiv.org/abs/2202.04895)] \
10 Feb 2022

**Progressive Distillation for Fast Sampling of Diffusion Models** \
*Tim Salimans, Jonathan Ho* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2202.00512)] \
1 Feb 2022

**Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models** \
*Fan Bao, Chongxuan Li, Jun Zhu, Bo Zhang* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2201.06503)] \
17 Jan 2022

**DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents** \
*Kushagra Pandey, Avideep Mukherjee, Piyush Rai, Abhishek Kumar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.00308)] [[Github](https://github.com/kpandey008/DiffuseVAE)] \
2 Jan 2022

**Itô-Taylor Sampling Scheme for Denoising Diffusion Probabilistic Models using Ideal Derivatives** \
*Hideyuki Tachibana, Mocho Go, Muneyoshi Inahara, Yotaro Katayama, Yotaro Watanabe* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.13339)] \
26 Dec 2021

**GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models** \
*Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen* \
ICML 2021. [[Paper](https://arxiv.org/abs/2112.10741)] [[Github](https://github.com/openai/glide-text2im)] \
20 Dec 2021

**High-Resolution Image Synthesis with Latent Diffusion Models** \
*Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.10752)] [[Github](https://github.com/CompVis/latent-diffusion)] \
20 Dec 2021

**Heavy-tailed denoising score matching** \
*Jacob Deasy, Nikola Simidjievski, Pietro Liò* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.09788)] \
17 Dec 2021

**High Fidelity Visualization of What Your Self-Supervised Representation Knows About** \
*Florian Bordes, Randall Balestriero, Pascal Vincent* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.09164)] \
16 Dec 2021

**Tackling the Generative Learning Trilemma with Denoising Diffusion GANs** \
*Zhisheng Xiao, Karsten Kreis, Arash Vahdat* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.07804)] [[Project](https://nvlabs.github.io/denoising-diffusion-gan)] \
15 Dec 2021

**Score-Based Generative Modeling with Critically-Damped Langevin Diffusion** \
*Tim Dockhorn, Arash Vahdat, Karsten Kreis* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2112.07068)] [[Project](https://nv-tlabs.github.io/CLD-SGM/)] \
14 Dec 2021

**More Control for Free! Image Synthesis with Semantic Diffusion Guidance** \
*Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, Trevor Darrell* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.05744)] \
10 Dec 2021

**Global Context with Discrete Diffusion in Vector Quantised Modelling for Image Generation** \
*Minghui Hu, Yujie Wang, Tat-Jen Cham, Jianfei Yang, P.N.Suganthan* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.01799)] \
3 Dec 2021

**Diffusion Autoencoders: Toward a Meaningful and Decodable Representation** \
*Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, Supasorn Suwajanakorn* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.15640)] [[Project](https://diff-ae.github.io/)] [[Github](https://github.com/phizaz/diffae)] \
30 Dec 2021

**Conditional Image Generation with Score-Based Diffusion Models** \
*Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, Christian Etmann* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.13606)] \
26 Nov 2021

**Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes** \
*Sam Bond-Taylor, Peter Hessey, Hiroshi Sasaki, Toby P. Breckon, Chris G. Willcocks* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.12701)] [[Github](https://github.com/samb-t/unleashing-transformers)] \
24 Nov 2021

**Diffusion Normalizing Flow** \
*Qinsheng Zhang, Yongxin Chen* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2110.07579)] [[Github](https://github.com/qsh-zh/DiffFlow)] \
14 Oct 2021

**Denoising Diffusion Gamma Models** \
*Eliya Nachmani, Robin San Roman, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2110.05948)] \
10 Oct 2021

**Score-based Generative Neural Networks for Large-Scale Optimal Transport** \
*Max Daniels, Tyler Maunu, Paul Hand* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2110.03237)] \
7 Oct 2021

**Score-Based Generative Classifiers** \
*Roland S. Zimmermann, Lukas Schott, Yang Song, Benjamin A. Dunn, David A. Klindt* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2110.00473)] \
1 Oct 2021

**Classifier-Free Diffusion Guidance** \
*Jonathan Ho, Tim Salimans* \
NeurIPS Workshop 2021. [[Paper](https://arxiv.org/abs/2207.12598)] \
28 Sep 2021


**Bilateral Denoising Diffusion Models** \
*Max W. Y. Lam, Jun Wang, Rongjie Huang, Dan Su, Dong Yu* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2108.11514)] [[Project](https://bilateral-denoising-diffusion-model.github.io)] \
26 Aug 2021

**ImageBART: Bidirectional Context with Multinomial Diffusion for Autoregressive Image Synthesis** \
*Patrick Esser, Robin Rombach, Andreas Blattmann, Björn Ommer* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2108.08827)] [[Project](https://compvis.github.io/imagebart/)] \
19 Aug 2021

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021

**SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** \
*Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon* \
ICLR  2022. [[Paper](https://arxiv.org/abs/2108.01073)] [[Project](https://sde-image-editing.github.io/)] [[Github](https://github.com/ermongroup/SDEdit)] \
2 Aug 2021

**Structured Denoising Diffusion Models in Discrete State-Spaces** \
*Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2107.03006)] \
7 Jul 2021 

**Variational Diffusion Models** \
*Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.00630)] [[Github](https://github.com/google-research/vdm)] \
1 Jul 2021 

**Diffusion Priors In Variational Autoencoders** \
*Antoine Wehenkel, Gilles Louppe* \
ICML Workshop 2021. [[Paper](https://arxiv.org/abs/2106.15671)] \
29 Jun 2021

**Deep Generative Learning via Schrödinger Bridge** \
*Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, Can Yang* \
ICML 2021. [[Paper](https://arxiv.org/abs/2106.10410)] \
19 Jun 2021

**Non Gaussian Denoising Diffusion Models** \
*Eliya Nachmani, Robin San Roman, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.07582)] [[Project](https://enk100.github.io/Non-Gaussian-Denoising-Diffusion-Models/)] \
14 Jun 2021 

**D2C: Diffusion-Denoising Models for Few-shot Conditional Generation** \
*Abhishek Sinha, Jiaming Song, Chenlin Meng, Stefano Ermon* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2106.06819)] [[Project](https://d2c-model.github.io/)] [[Github](https://github.com/d2c-model/d2c-model.github.io)] \
12 Jun 2021

**Score-based Generative Modeling in Latent Space** \
*Arash Vahdat, Karsten Kreis, Jan Kautz* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.05931)] \
10 Jun 2021

**Learning to Efficiently Sample from Diffusion Probabilistic Models** \
*Daniel Watson, Jonathan Ho, Mohammad Norouzi, William Chan* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.03802)] \
7 Jun 2021 

**A Variational Perspective on Diffusion-Based Generative Models and Score Matching** \
*Chin-Wei Huang, Jae Hyun Lim, Aaron Courville* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2106.02808)] [[Github](https://github.com/CW-Huang/sdeflow-light)] \
5 Jun 2021 

**Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation** \
*Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, Il-Chul Moon* \
ICML 2022. [[Paper](https://arxiv.org/abs/2106.05527)] \
10 Jun 2021

**Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling** \
*Valentin De Bortoli, James Thornton, Jeremy Heng, Arnaud Doucet* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.01357)] [[Project](https://jtt94.github.io/papers/schrodinger_bridge)] [[Github](https://github.com/JTT94/diffusion_schrodinger_bridge)] \
1 Jun 2021

**On Fast Sampling of Diffusion Probabilistic Models** \
*Zhifeng Kong, Wei Ping* \
ICML Workshop 2021. [[Paper](https://arxiv.org/abs/2106.00132)] [[Github](https://github.com/FengNiMa/FastDPM_pytorch)] \
31 May 2021 

**Cascaded Diffusion Models for High Fidelity Image Generation** \
*Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, Tim Salimans* \
JMLR 2021. [[Paper](https://arxiv.org/abs/2106.15282)] [[Project](https://cascaded-diffusion.github.io/)] \
30 May 2021 

**Gotta Go Fast When Generating Data with Score-Based Models** \
*Alexia Jolicoeur-Martineau, Ke Li, Rémi Piché-Taillefer, Tal Kachman, Ioannis Mitliagkas* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.14080)] [[Github](https://github.com/AlexiaJM/score_sde_fast_sampling)] \
28 May 2021

**Diffusion Models Beat GANs on Image Synthesis** \
*Prafulla Dhariwal, Alex Nichol* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.05233)] [[Github](https://github.com/openai/guided-diffusion)] \
11 May 2021 

**Image Super-Resolution via Iterative Refinement** \
*Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.07636)] [[Project](https://iterative-refinement.github.io/)] [[Github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] \
15 Apr 2021 

**Noise Estimation for Generative Diffusion Models** \
*Robin San-Roman, Eliya Nachmani, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.02600)] \
6 Apr 2021 

**Improved Denoising Diffusion Probabilistic Models** \
*Alex Nichol, Prafulla Dhariwal* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2102.09672)] [[Github](https://github.com/openai/improved-diffusion)] \
18 Feb 2021 

**Maximum Likelihood Training of Score-Based Diffusion Models** \
*Yang Song, Conor Durkan, Iain Murray, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2101.09258)] \
22 Jan 2021 

**Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed** \
*Eric Luhman, Troy Luhman* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2101.02388)] [[Github](https://github.com/tcl9876/Denoising_Student)] \
7 Jan 2021

**Learning Energy-Based Models by Diffusion Recovery Likelihood** \
*Ruiqi Gao, Yang Song, Ben Poole, Ying Nian Wu, Diederik P. Kingma* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2012.08125)] [[Github](https://github.com/ruiqigao/recovery_likelihood)] \
15 Dec 2020 

**Score-Based Generative Modeling through Stochastic Differential Equations** \
*Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole* \
ICLR 2021 (Oral). [[Paper](https://arxiv.org/abs/2011.13456)] [[Github](https://github.com/yang-song/score_sde)] \
26 Nov 2020 

**Variational (Gradient) Estimate of the Score Function in Energy-based Latent Variable Models** \
*Fan Bao, Kun Xu, Chongxuan Li, Lanqing Hong, Jun Zhu, Bo Zhang* \
ICML 2021. [[Paper](https://arxiv.org/abs/2010.08258)] \
16 Oct 2020

**Denoising Diffusion Implicit Models**  \
*Jiaming Song, Chenlin Meng, Stefano Ermon* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2010.02502)] [[Github](https://github.com/ermongroup/ddim)] \
6 Oct 2020

**Adversarial score matching and improved sampling for image generation** \
*Alexia Jolicoeur-Martineau, Rémi Piché-Taillefer, Rémi Tachet des Combes, Ioannis Mitliagkas* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2009.05475)] [[Github](https://github.com/AlexiaJM/AdversarialConsistentScoreMatching)] \
11 Sep 2020

**Denoising Diffusion Probabilistic Models** \
*Jonathan Ho, Ajay Jain, Pieter Abbeel* \
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2006.11239)] [[Github](https://github.com/hojonathanho/diffusion)] [[Github2](https://github.com/pesser/pytorch_diffusion)] \
19 Jun 2020 

**Improved Techniques for Training Score-Based Generative Models** \
*Yang Song, Stefano Ermon* \
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2006.09011)] [[Github](https://github.com/ermongroup/ncsnv2)] \
16 Jun 2020 

**Generative Modeling by Estimating Gradients of the Data Distribution** \
*Yang Song, Stefano Ermon* \
NeurIPS 2019. [[Paper](https://arxiv.org/abs/1907.05600)] [[Project](https://yang-song.github.io/blog/2021/score/)] [[Github](https://github.com/ermongroup/ncsn)] \
12 Jul 2019 

**Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit** \
*Belinda Tzen, Maxim Raginsky* \
arXiv 2019. [[Paper](https://arxiv.org/abs/1905.09883)] \
23 May 2019 

**Deep Unsupervised Learning using Nonequilibrium Thermodynamics** \
*Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli* \
ICML 2015. [[Paper](https://arxiv.org/abs/1503.03585)] [[Github](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)] \
2 Mar 2015

### Classification

**Likelihood-based Out-of-Distribution Detection with Denoising Diffusion Probabilistic Models** \
*Joseph Goodier, Neill D. F. Campbell* \
BMVC 2023. [[Paper](https://arxiv.org/abs/2310.17432)] \
26 Oct 2023

**Multi-scale Diffusion Denoised Smoothing** \
*Jongheon Jeong, Jinwoo Shin* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.16779)] \
25 Oct 2023

**DiffRef3D: A Diffusion-based Proposal Refinement Framework for 3D Object Detection** \
*Se-Ho Kim, Inyong Koo, Inyoung Lee, Byeongjun Park, Changick Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16349)] \
25 Oct 2023

**Denoising Task Routing for Diffusion Models** \
*Byeongjun Park, Sangmin Woo, Hyojun Go, Jin-Young Kim, Changick Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07138)] \
11 Oct 2023

**Leveraging Diffusion-Based Image Variations for Robust Training on Poisoned Data** \
*Lukas Struppek, Martin B. Hentschel, Clifton Poth, Dominik Hintersdorf, Kristian Kersting* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06372)] [[Github](https://github.com/LukasStruppek/Robust_Training_on_Poisoned_Samples)] \
10 Oct 2023

**Dream the Impossible: Outlier Imagination with Diffusion Models** \
*Xuefeng Du, Yiyou Sun, Xiaojin Zhu, Yixuan Li* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2309.13415)] [[Github](https://github.com/deeplearning-wisc/dream-ood)] \
23 Sep 2023

**Zero-Shot Object Counting with Language-Vision Models** \
*Jingyi Xu, Hieu Le, Dimitris Samaras* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2309.13097)] [[Github](https://github.com/cvlab-stonybrook/zero-shot-counting)] \
22 Sep 2023

**PSDiff: Diffusion Model for Person Search with Iterative and Collaborative Refinement** \
*Chengyou Jia, Minnan Luo, Zhuohang Dang, Guang Dai, Xiaojun Chang, Jingdong Wang, Qinghua Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11125)] \
20 Sep 2023

**Beyond Generation: Harnessing Text to Image Models for Object Detection and Segmentation** \
*Yunhao Ge, Jiashu Xu, Brian Nlong Zhao, Neel Joshi, Laurent Itti, Vibhav Vineet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05956)] [[Github](https://github.com/gyhandy/Text2Image-for-Detection)] \
12 Sep 2023

**DiffusionEngine: Diffusion Model is Scalable Data Engine for Object Detection** \
*Manlin Zhang, Jie Wu, Yuxi Ren, Ming Li, Jie Qin, Xuefeng Xiao, Wei Liu, Rui Wang, Min Zheng, Andy J. Ma* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03893)] [[Project](https://mettyz.github.io/DiffusionEngine/)] [[Github](https://github.com/bytedance/DiffusionEngine)] \
7 Sep 2023

**Diffusion-based 3D Object Detection with Random Boxes** \
*Xin Zhou, Jinghua Hou, Tingting Yao, Dingkang Liang, Zhe Liu, Zhikang Zou, Xiaoqing Ye, Jianwei Cheng, Xiang Bai* \
PRCV 2023. [[Paper](https://arxiv.org/abs/2309.02049)] \
5 Sep 2023

**Diffusion Model as Representation Learner** \
*Xingyi Yang, Xinchao Wang* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.10916)] \
21 Aug 2023

**DiffusionTrack: Diffusion Model For Multi-Object Tracking** \
*Run Luo, Zikai Song, Lintao Ma, Jinlin Wei, Wei Yang, Min Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09905)] \
19 Aug 2023

**DiffGuard: Semantic Mismatch-Guided Out-of-Distribution Detection using Pre-trained Diffusion Models** \
*Ruiyuan Gao, Chenchen Zhao, Lanqing Hong, Qiang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07687)] \
15 Aug 2023

**IDiff-Face: Synthetic-based Face Recognition through Fizzy Identity-Conditioned Diffusion Models** \
*Fadi Boutros, Jonas Henry Grebe, Arjan Kuijper, Naser Damer* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.04995)] \
9 Aug 2023

**Exploiting Synthetic Data for Data Imbalance Problems: Baselines from a Data Perspective** \
*Moon Ye-Bin, Nam Hyeon-Woo, Wonseok Choi, Nayeong Kim, Suha Kwak, Tae-Hyun Oh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.00994)] \
2 Aug 2023

**Diffusion Model for Camouflaged Object Detection** \
*Zhennan Chen, Rongrong Gao, Tian-Zhu Xiang, Fan Lin* \
ECAI 2023. [[Paper](https://arxiv.org/abs/2308.00303)] \
1 Aug 2023

**DiffPose: SpatioTemporal Diffusion Model for Video-Based Human Pose Estimation** \
*Runyang Feng, Yixing Gao, Tze Ho Elden Tse, Xueqing Ma, Hyung Jin Chang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16687)] \
31 Jul 2023

**MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning** \
*Baoquan Zhang, Demin Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16424)] \
31 Jul 2023

**Generative Prompt Model for Weakly Supervised Object Localization** \
*Yuzhong Zhao, Qixiang Ye, Weijia Wu, Chunhua Shen, Fang Wan* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2307.09756)] [[Github](https://github.com/callsys/GenPromp)] \
19 Jul 2023

**Diffusion Models Beat GANs on Image Classification** \
*Soumik Mukhopadhyay, Matthew Gwilliam, Vatsal Agarwal, Namitha Padmanabhan, Archana Swaminathan, Srinidhi Hegde, Tianyi Zhou, Abhinav Shrivastava* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08702)] \
17 Jul 2023

**Diffusion to Confusion: Naturalistic Adversarial Patch Generation Based on Diffusion Model for Object Detector** \
*Shuo-Yen Lin, Ernie Chu, Che-Hsien Lin, Jun-Cheng Chen, Jia-Ching Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08076)] \
16 Jul 2023

**DreamTeacher: Pretraining Image Backbones with Deep Generative Models** \
*Daiqing Li, Huan Ling, Amlan Kar, David Acuna, Seung Wook Kim, Karsten Kreis, Antonio Torralba, Sanja Fidler* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.07487)] [[Project](https://research.nvidia.com/labs/toronto-ai/DreamTeacher/)] \
14 Jul 2023

**ProtoDiff: Learning to Learn Prototypical Networks by Task-Guided Diffusion** \
*Yingjun Du, Zehao Xiao, Shengcai Liao, Cees Snoek* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.14770)] \
26 Jun 2023

**Masked Diffusion Models are Fast Learners** \
*Jiachen Lei, Peng Cheng, Zhongjie Ba, Kui Ren* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11363)] \
20 Jun 2023


**Renderers are Good Zero-Shot Representation Learners: Exploring Diffusion Latents for Metric Learning** \
*Michael Tang, David Shustin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.10721)] \
19 Jun 2023

**The Big Data Myth: Using Diffusion Models for Dataset Generation to Train Deep Detection Models** \
*Roy Voetman, Maya Aghaei, Klaas Dijkstra* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09762)] \
16 Jun 2023

**When Hyperspectral Image Classification Meets Diffusion Models: An Unsupervised Feature Learning Framework** \
*Jingyi Zhou, Jiamu Sheng, Jiayuan Fan, Peng Ye, Tong He, Bin Wang, Tao Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08964)] \
15 Jun 2023

**DDLP: Unsupervised Object-Centric Video Prediction with Deep Dynamic Latent Particles** \
*Tal Daniel, Aviv Tamar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05957)] \
9 Jun 2023

**ADDP: Learning General Representations for Image Recognition and Generation with Alternating Denoising Diffusion Process** \
*Changyao Tian, Chenxin Tao, Jifeng Dai, Hao Li, Ziheng Li, Lewei Lu, Xiaogang Wang, Hongsheng Li, Gao Huang, Xizhou Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05423)] \
8 Jun 2023

**Conditional Generation from Unconditional Diffusion Models using Denoiser Representations** \
*Alexandros Graikos, Srikar Yellapragada, Dimitris Samaras* \
BMVC 2023. [[Paper](https://arxiv.org/abs/2306.01900)] [[Github](https://github.com/cvlab-stonybrook/fewshot-conditional-diffusion)] \
2 Jun 2023

**DiffCLIP: Leveraging Stable Diffusion for Language Grounded 3D Classification** \
*Sitian Shen, Zilin Zhu, Linqian Fan, Harry Zhang, Xinxiao Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15957)] \
25 May 2023


**Training on Thin Air: Improve Image Classification with Generated Data** \
*Yongchao Zhou, Hshmat Sahak, Jimmy Ba* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15316)] [[Project](https://sites.google.com/view/diffusion-inversion)] [[Github](https://github.com/yongchao97/diffusion_inversion)] \
24 May 2023

**Is Synthetic Data From Diffusion Models Ready for Knowledge Distillation?** \
*Zheng Li, Yuxuan Li, Penghai Zhao, Renjie Song, Xiang Li, Jian Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12954)] [[Github](https://github.com/zhengli97/DM-KD)] \
22 May 2023

**Boosting Human-Object Interaction Detection with Text-to-Image Diffusion Model** \
*Jie Yang, Bingliang Li, Fengyu Yang, Ailing Zeng, Lei Zhang, Ruimao Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12252)] \
20 May 2023


**Meta-DM: Applications of Diffusion Models on Few-Shot Learning** \
*Wentao Hu, Xiurong Jiang, Jiarun Liu, Yuqi Yang, Hui Tian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08092)] \
14 May 2023

**Class-Balancing Diffusion Models** \
*Yiming Qin, Huangjie Zheng, Jiangchao Yao, Mingyuan Zhou, Ya Zhang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2305.00562)] \
30 Apr 2023

**Synthetic Data from Diffusion Models Improves ImageNet Classification** \
*Shekoofeh Azizi, Simon Kornblith, Chitwan Saharia, Mohammad Norouzi, David J. Fleet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08466)] \
17 Apr 2023



**OVTrack: Open-Vocabulary Multiple Object Tracking** \
*Siyuan Li, Tobias Fischer, Lei Ke, Henghui Ding, Martin Danelljan, Fisher Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08408)] \
17 Apr 2023

**Your Diffusion Model is Secretly a Zero-Shot Classifier** \
*Alexander C. Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, Deepak Pathak* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16203)] [[Project](https://diffusion-classifier.github.io/)] \
28 Mar 2023


**Text-to-Image Diffusion Models are Zero-Shot Classifiers** \
*Kevin Clark, Priyank Jaini* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15233)] \
27 Mar 2023

**Diffusion Denoised Smoothing for Certified and Adversarial Robust Out-Of-Distribution Detection** \
*Nicola Franco, Daniel Korth, Jeanette Miriam Lorenz, Karsten Roscher, Stephan Guennemann* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14961)] \
27 Mar 2023



**CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images** \
*Jordan J. Bird, Ahmad Lotfi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14126)] \
24 Mar 2023



**Denoising Diffusion Autoencoders are Unified Self-supervised Learners** \
*Weilai Xiang, Hongyu Yang, Di Huang, Yunhong Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09769)] )] \
17 Mar 2023

**Boosting Zero-shot Classification with Synthetic Data Diversity via Stable Diffusion** \
*Jordan Shipard, Arnold Wiliem, Kien Nguyen Thanh, Wei Xiang, Clinton Fookes* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03298)] \
7 Feb 2023

**Fake it till you make it: Learning(s) from a synthetic ImageNet clone** \
*Mert Bulent Sariyildiz, Karteek Alahari, Diane Larlus, Yannis Kalantidis* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.08420)] [[Project](https://europe.naverlabs.com/research/computer-vision/imagenet-sd/)] \
16 Dec 2022

**DiffAlign : Few-shot learning using diffusion based synthesis and alignment** \
*Aniket Roy, Anshul Shah, Ketul Shah, Anirban Roy, Rama Chellappa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.05404)] \
11 Dec 2022


**Diffusion Denoising Process for Perceptron Bias in Out-of-distribution Detection** \
*Luping Liu, Yi Ren, Xize Cheng, Zhou Zhao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11255)] [[Github](https://github.com/luping-liu/DiffOOD)] \
21 Nov 2022


**DiffusionDet: Diffusion Model for Object Detection** \
*Shoufa Chen, Peize Sun, Yibing Song, Ping Luo* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09788)] [[Github](https://github.com/ShoufaChen/DiffusionDet)] \
17 Nov 2022



**Denoising Diffusion Models for Out-of-Distribution Detection** \
*Mark S. Graham, Walter H.L. Pinaya, Petru-Daniel Tudosiu, Parashkev Nachev, Sebastien Ourselin, M. Jorge Cardoso* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.07740)] [[Github](https://github.com/marksgraham/ddpm-ood)] \
14 Nov 2022



**A simple, efficient and scalable contrastive masked autoencoder for learning visual representations** \
*Shlok Mishra, Joshua Robinson, Huiwen Chang, David Jacobs, Aaron Sarna, Aaron Maschinot, Dilip Krishnan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.16870)] \
30 Oct 2022

**From Points to Functions: Infinite-dimensional Representations in Diffusion Models** \
*Sarthak Mittal, Guillaume Lajoie, Stefan Bauer, Arash Mehrjou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.13774)] [[Github](https://github.com/sarthmit/traj_drl)] \
25 Oct 2022


**Boomerang: Local sampling on image manifolds using diffusion models** \
*Lorenzo Luzi, Ali Siahkoohi, Paul M Mayer, Josue Casco-Rodriguez, Richard Baraniuk* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.12100)] [[Colab](https://colab.research.google.com/drive/1PV5Z6b14HYZNx1lHCaEVhId-Y4baKXwt)] \
21 Oct 2022


**Meta-Learning via Classifier(-free) Guidance** \
*Elvis Nava, Seijin Kobayashi, Yifei Yin, Robert K. Katzschmann, Benjamin F. Grewe* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.08942)] \
17 Oct 2022


### Segmentation

**One-shot Localization and Segmentation of Medical Images with Foundation Models** \
*Deepa Anand, Gurunath Reddy M, Vanika Singhal, Dattesh D. Shanbhag, Shriram KS, Uday Patil, Chitresh Bhushan, Kavitha Manickam, Dawei Gui, Rakesh Mullick, Avinash Gopal, Parminder Bhatia, Taha Kass-Hout* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.18642)] \
28 Oct 2023

**Semantic-preserving image coding based on Conditional Diffusion models** \
*Francesco Pezone, Osman Musa, Giuseppe Caire, Sergio Barbarossa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.15737)] \
24 Oct 2023

**Diffusion-based Data Augmentation for Nuclei Image Segmentation** \
*Xinyi Yu, Guanbin Li, Wei Lou, Siqi Liu, Xiang Wan, Yan Chen, Haofeng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.14197)] \
22 Oct 2023

**EMIT-Diff: Enhancing Medical Image Segmentation via Text-Guided Diffusion Model** \
*Zheyuan Zhang, Lanhong Yao, Bin Wang, Debesh Jha, Elif Keles, Alpay Medetalibeyoglu, Ulas Bagci* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12868)] \
19 Oct 2023

**Towards Training-free Open-world Segmentation via Image Prompting Foundation Models** \
*Lv Tang, Peng-Tao Jiang, Hao-Ke Xiao, Bo Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10912)] \
17 Oct 2023

**Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation** \
*Haonan Wang, Xiaomeng Li* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.11320)] [[Github](https://github.com/xmed-lab/GenericSSL)] \
17 Oct 2023

**Image Augmentation with Controlled Diffusion for Weakly-Supervised Semantic Segmentation** \
*Wangyu Wu, Tianhong Dai, Xiaowei Huang, Fei Ma, Jimin Xiao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09760)] \
15 Oct 2023


**Steered Diffusion: A Generalized Framework for Plug-and-Play Conditional Image Synthesis** \
*Nithin Gopalakrishnan Nair, Anoop Cherian, Suhas Lohit, Ye Wang, Toshiaki Koike-Akino, Vishal M. Patel, Tim K. Marks* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2310.00224)] \
30 Sep 2023

**Factorized Diffusion Architectures for Unsupervised Image Generation and Segmentation** \
*Xin Yuan, Michael Maire* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.15726)] \
27 Sep 2023

**Dataset Diffusion: Diffusion-based Synthetic Dataset Generation for Pixel-Level Semantic Segmentation** \
*Quang Nguyen, Truong Vu, Anh Tran, Khoi Nguyen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14303)] \
25 Sep 2023

**MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation** \
*Jiahao Xie, Wei Li, Xiangtai Li, Ziwei Liu, Yew Soon Ong, Chen Change Loy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.13042)] [[Github](https://github.com/Jiahao000/MosaicFusion)] \
22 Sep 2023

**Beyond Generation: Harnessing Text to Image Models for Object Detection and Segmentation** \
*Yunhao Ge, Jiashu Xu, Brian Nlong Zhao, Neel Joshi, Laurent Itti, Vibhav Vineet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05956)] [[Github](https://github.com/gyhandy/Text2Image-for-Detection)] \
12 Sep 2023

**Introducing Shape Prior Module in Diffusion Model for Medical Image Segmentation** \
*Zhiqing Zhang, Guojia Fan, Tianyong Liu, Nan Li, Yuyang Liu, Ziyu Liu, Canwei Dong, Shoujun Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05929)] \
12 Sep 2023

**From Text to Mask: Localizing Entities Using the Attention of Text-to-Image Diffusion Models** \
*Changming Xiao, Qi Yang, Feng Zhou, Changshui Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.04109)] \
8 Sep 2023

**SLiMe: Segment Like Me** \
*Aliasghar Khani, Saeid Asgari Taghanaki, Aditya Sanghi, Ali Mahdavi Amiri, Ghassan Hamarneh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03179)] [[Github](https://github.com/aliasgharkhani/SLiMe)] \
6 Sep 2023

**Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter** \
*Jinglong Wang, Xiawei Li, Jing Zhang, Qingyuan Xu, Qin Zhou, Qian Yu, Lu Sheng, Dong Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.02773)] \
6 Sep 2023


**GenSelfDiff-HIS: Generative Self-Supervision Using Diffusion for Histopathological Image Segmentation** \
*Vishnuvardhan Purma, Suhas Srinath, Seshan Srirangarajan, Aanchal Kakkar, Prathosh A. P* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01487)] [[Github](https://github.com/PurmaVishnuVardhanReddy/GenSelfDiff-HIS)] \
4 Sep 2023

**Attention as Annotation: Generating Images and Pseudo-masks for Weakly Supervised Semantic Segmentation with Diffusion** \
*Ryota Yoshihashi, Yuya Otsuka, Kenji Doi, Tomohiro Tanaka* \
AAAI 2022. [[Paper](https://arxiv.org/abs/2309.01369)] \
4 Sep 2023

**ArSDM: Colonoscopy Images Synthesis with Adaptive Refinement Semantic Diffusion Models** \
*Yuhao Du, Yuncheng Jiang, Shuangyi Tan, Xusheng Wu, Qi Dou, Zhen Li, Guanbin Li, Xiang Wan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01111)] \
3 Sep 2023

**Ref-Diff: Zero-shot Referring Image Segmentation with Generative Models** \
*Minheng Ni, Yabo Zhang, Kailai Feng, Xiaoming Li, Yiwen Guo, Wangmeng Zuo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16777)] \
31 Aug 2023

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16150)] \
30 Aug 2023

**A Recycling Training Strategy for Medical Image Segmentation with Diffusion Denoising Models** \
*Yunguan Fu, Yiwen Li, Shaheer U Saeed, Matthew J Clarkson, Yipeng Hu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16355)] [[Github](https://github.com/mathpluscode/ImgX-DiffSeg)] \
30 Aug 2023

**Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion** \
*Junjiao Tian, Lavisha Aggarwal, Andrea Colaco, Zsolt Kira, Mar Gonzalez-Franco* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.12469)] \
23 Aug 2023

**Diffusion-based Image Translation with Label Guidance for Domain Adaptive Semantic Segmentation** \
*Duo Peng, Ping Hu, Qiuhong Ke, Jun Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.12350)] \
23 Aug 2023

**DMCVR: Morphology-Guided Diffusion Model for 3D Cardiac Volume Reconstruction** \
*Xiaoxiao He, Chaowei Tan, Ligong Han, Bo Liu, Leon Axel, Kang Li, Dimitris N. Metaxas* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2308.09223)] [[Github](https://github.com/hexiaoxiao-cs/DMCVR)] \
18 Aug 2023

**Masked Diffusion as Self-supervised Representation Learner** \
*Zixuan Pan, Jianxu Chen, Yiyu Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.05695)] \
10 Aug 2023

**DermoSegDiff: A Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation** \
*Afshin Bozorgpour, Yousef Sadegheih, Amirhossein Kazerouni, Reza Azad, Dorit Merhof* \
MICCAI Workshop 2023. [[Paper](https://arxiv.org/abs/2308.02959)] [[Github](https://github.com/mindflow-institue/dermosegdiff)] \
5 Aug 2023

**DiffusePast: Diffusion-based Generative Replay for Class Incremental Semantic Segmentation** \
*Jingfan Chen, Yuxi Wang, Pengfei Wang, Xiao Chen, Zhaoxiang Zhang, Zhen Lei, Qing Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01127)] \
2 Aug 2023

**DAVIS: High-Quality Audio-Visual Separation with Generative Diffusion Models** \
*Chao Huang, Susan Liang, Yapeng Tian, Anurag Kumar, Chenliang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.00122)] \
31 Jul 2023

**Pre-Training with Diffusion models for Dental Radiography segmentation** \
*Jérémy Rousseau, Christian Alaka, Emma Covili, Hippolyte Mayard, Laura Misrachi, Willy Au* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.14066)] \
26 Jul 2023

**FEDD -- Fair, Efficient, and Diverse Diffusion-based Lesion Segmentation and Malignancy Classification** \
*Héctor Carrión, Narges Norouzi* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2307.11654)] [[Github](https://github.com/hectorcarrion/fedd)] \
21 Jul 2023

**DreamTeacher: Pretraining Image Backbones with Deep Generative Models** \
*Daiqing Li, Huan Ling, Amlan Kar, David Acuna, Seung Wook Kim, Karsten Kreis, Antonio Torralba, Sanja Fidler* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.07487)] [[Project](https://research.nvidia.com/labs/toronto-ai/DreamTeacher/)] \
14 Jul 2023

**Prompting Diffusion Representations for Cross-Domain Semantic Segmentation** \
*Rui Gong, Martin Danelljan, Han Sun, Julio Delgado Mangas, Luc Van Gool* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.02138)] \
5 Jul 2023

**DifFSS: Diffusion Model for Few-Shot Semantic Segmentation** \
*Weimin Tan, Siyuan Chen, Bo Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.00773)] \
3 Jul 2023



**Towards Better Certified Segmentation via Diffusion Models** \
*Othmane Laousy, Alexandre Araujo, Guillaume Chassagnon, Marie-Pierre Revel, Siddharth Garg, Farshad Khorrami, Maria Vakalopoulou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09949)] \
16 Jun 2023

**Diffusion Models for Zero-Shot Open-Vocabulary Segmentation** \
*Laurynas Karazija, Iro Laina, Andrea Vedaldi, Christian Rupprecht* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09316)] \
15 Jun 2023


**Annotator Consensus Prediction for Medical Image Segmentation with Diffusion Models** \
*Tomer Amit, Shmuel Shichrur, Tal Shaharabany, Lior Wolf* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09004)] \
15 Jun 2023


**Generative Semantic Communication: Diffusion Models Beyond Bit Recovery** \
*Eleonora Grassucci, Sergio Barbarossa, Danilo Comminiello* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04321)] [[Github](https://github.com/ispamm/GESCO)] \
7 Jun 2023

**Conditional Diffusion Models for Weakly Supervised Medical Image Segmentation** \
*Xinrong Hu, Yu-Jen Chen, Tsung-Yi Ho, Yiyu Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03878)] \
6 Jun 2023

**DFormer: Diffusion-guided Transformer for Universal Image Segmentation** \
*Hefeng Wang, Jiale Cao, Rao Muhammad Anwer, Jin Xie, Fahad Shahbaz Khan, Yanwei Pang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03437)] [[Github](https://github.com/cp3wan/DFormer)] \
6 Jun 2023

**Denoising Diffusion Semantic Segmentation with Mask Prior Modeling** \
*Zeqiang Lai, Yuchen Duan, Jifeng Dai, Ziheng Li, Ying Fu, Hongsheng Li, Yu Qiao, Wenhai Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01721)] \
2 Jun 2023

**Multi-Level Global Context Cross Consistency Model for Semi-Supervised Ultrasound Image Segmentation with Diffusion Model** \
*Fenghe Tang, Jianrui Ding, Lingtao Wang, Min Xian, Chunping Ning* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09447)] [[Github](https://github.com/FengheTan9/Multi-Level-Global-Context-Cross-Consistency)] \
16 May 2023

**Echo from noise: synthetic ultrasound image generation using diffusion models for real image segmentation** \
*David Stojanovski, Uxio Hermida, Pablo Lamata, Arian Beqiri, Alberto Gomez* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05424)] \
9 May 2023

**Personalize Segment Anything Model with One Shot** \
*Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Hao Dong, Peng Gao, Hongsheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03048)] [[Github](https://github.com/ZrrSkywalker/Personalize-SAM)] \
4 May 2023


**Personalize Segment Anything Model with One Shot** \
*Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Hao Dong, Peng Gao, Hongsheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03048)] [[Github](https://github.com/ZrrSkywalker/Personalize-SAM)] \
4 May 2023

**Unsupervised Discovery of 3D Hierarchical Structure with Generative Diffusion Features** \
*Nurislam Tursynbek, Marc Niethammer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.00067)] \
28 Apr 2023

**DiffuseExpand: Expanding dataset for 2D medical image segmentation using diffusion models** \
*Shitong Shao, Xiaohan Yuan, Zhen Huang, Ziming Qiu, Shuai Wang, Kevin Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13416)] [[Github](https://anonymous.4open.science/r/DiffuseExpand/README.md)] \
26 Apr 2023



**Realistic Data Enrichment for Robust Image Segmentation in Histopathology** \
*Sarah Cechnicka, James Ball, Callum Arthurs, Candice Roufosse, Bernhard Kainz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.09534)] \
19 Apr 2023

**Denoising Diffusion Medical Models** \
*Pham Ngoc Huy, Tran Minh Quan* \
IEEE ISBI 2023. [[Paper](https://arxiv.org/abs/2304.09383)] \
19 Apr 2023


**Ambiguous Medical Image Segmentation using Diffusion Models** \
*Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, Vishal M Patel* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.04745)] [[Github](https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models)] \
10 Apr 2023

**BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation** \
*Tao Chen, Chenhui Wang, Hongming Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04429)] \
10 Apr 2023


**Distribution Aligned Diffusion and Prototype-guided network for Unsupervised Domain Adaptive Segmentation** \
*Haipeng Zhou, Lei Zhu, Yuyin Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12313)] \
22 Mar 2023

**Semantic Latent Space Regression of Diffusion Autoencoders for Vertebral Fracture Grading** \
*Matthias Keicher, Matan Atad, David Schinz, Alexandra S. Gersing, Sarah C. Foreman, Sophia S. Goller, Juergen Weissinger, Jon Rischewski, Anna-Sophia Dietrich, Benedikt Wiestler, Jan S. Kirschke, Nassir Navab* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12031)] \
21 Mar 2023

**LD-ZNet: A Latent Diffusion Approach for Text-Based Image Segmentation** \
*Koutilya Pnvr, Bharat Singh, Pallabi Ghosh, Behjat Siddiquie, David Jacobs* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12343)] \
22 Mar 2023

**DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models** \
*Weijia Wu, Yuzhong Zhao, Mike Zheng Shou, Hong Zhou, Chunhua Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11681)] [[Project](https://weijiawu.github.io/DiffusionMask/)] \
21 Mar 2023

**Object-Centric Slot Diffusion** \
*Jindong Jiang, Fei Deng, Gautam Singh, Sungjin Ahn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10834)] \
20 Mar 2023


**Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation** \
*Zhaohu Xing, Liang Wan, Huazhu Fu, Guang Yang, Lei Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10326)] [[Github](https://github.com/ge-xing/Diff-UNet)] \
18 Mar 2023

**DiffusionSeg: Adapting Diffusion Towards Unsupervised Object Discovery** \
*Chaofan Ma, Yuhuan Yang, Chen Ju, Fei Zhang, Jinxiang Liu, Yu Wang, Ya Zhang, Yanfeng Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09813)] \
17 Mar 2023

**Stochastic Segmentation with Conditional Categorical Diffusion Models** \
*Lukas Zbinden, Lars Doorenbos, Theodoros Pissas, Raphael Sznitman, Pablo Márquez-Neila* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2303.08888)] [[Github](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)] \
15 Mar 2023

**DiffBEV: Conditional Diffusion Model for Bird's Eye View Perception** \
*Jiayu Zou, Zheng Zhu, Yun Ye, Xingang Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08333)] \
15 Mar 2023

**Importance of Aligning Training Strategy with Evaluation for Diffusion Models in 3D Multiclass Segmentation** \
*Yunguan Fu, Yiwen Li, Shaheer U. Saeed, Matthew J. Clarkson, Yipeng Hu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06040)] [[Github](https://github.com/mathpluscode/ImgX-DiffSeg)] \
10 Mar 2023

**MaskDiff: Modeling Mask Distribution with Diffusion Probabilistic Model for Few-Shot Instance Segmentation** \
*Minh-Quan Le, Tam V. Nguyen, Trung-Nghia Le, Thanh-Toan Do, Minh N. Do, Minh-Triet Tran* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05105)] \
9 Mar 2023


**Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models** \
*Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, Shalini De Mello* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04803)] [[Project](https://jerryxu.net/ODISE/)] \
8 Mar 2023


**MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer** \
*Junde Wu, Rao Fu, Huihui Fang, Yu Zhang, Yanwu Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11798)] \
19 Jan 2023

**DiffusionInst: Diffusion Model for Instance Segmentation** \
*Zhangxuan Gu, Haoxing Chen, Zhuoer Xu, Jun Lan, Changhua Meng, Weiqiang Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02773)] [[Github](https://github.com/chenhaoxing/DiffusionInst)] \
6 DEc 2022

**Multi-Class Segmentation from Aerial Views using Recursive Noise Diffusion** \
*Benedikt Kolbeinsson, Krystian Mikolajczyk* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00787)] \
1 Dec 2022

**Peekaboo: Text to Image Diffusion Models are Zero-Shot Segmentors** \
*Ryan Burgert, Kanchana Ranasinghe, Xiang Li, Michael S. Ryoo* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13224)] \
23 Nov 2022

**Improved HER2 Tumor Segmentation with Subtype Balancing using Deep Generative Networks** \
*Mathias Öttl, Jana Mönius, Matthias Rübner, Carol I. Geppert, Jingna Qiu, Frauke Wilm, Arndt Hartmann, Matthias W. Beckmann, Peter A. Fasching, Andreas Maier, Ramona Erber, Katharina Breininger* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.06150)] \
11 Nov 2022

**MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model** \
*Junde Wu, Huihui Fang, Yu Zhang, Yehui Yang, Yanwu Xu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.00611)] \
1 Nov 2022

**Accelerating Diffusion Models via Pre-segmentation Diffusion Sampling for Medical Image Segmentation** \
*Xutao Guo, Yanwu Yang, Chenfei Ye, Shang Lu, Yang Xiang, Ting Ma* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.17408)] \
27 Oct 2022

**Anatomically constrained CT image translation for heterogeneous blood vessel segmentation** \
*Giammarco La Barbera, Haithem Boussaid, Francesco Maso, Sabine Sarnacki, Laurence Rouet, Pietro Gori, Isabelle Bloch* \
BMVC 2022. [[Paper](https://arxiv.org/abs/2210.01713)] \
4 Oct 2022

**Diffusion Adversarial Representation Learning for Self-supervised Vessel Segmentation** \
*Boah Kim, Yujin Oh, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14566)] \
29 Sep 2022

**Can segmentation models be trained with fully synthetically generated data?** \
*Virginia Fernandez, Walter Hugo Lopez Pinaya, Pedro Borges, Petru-Daniel Tudosiu, Mark S Graham, Tom Vercauteren, M Jorge Cardoso* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.08256)] \
17 Sep 2022

**Let us Build Bridges: Understanding and Extending Diffusion Generative Models** \
*Xingchao Liu, Lemeng Wu, Mao Ye, Qiang Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.14699)] \
31 Aug 2022


**Semantic Image Synthesis via Diffusion Models** \
*Weilun Wang, Jianmin Bao, Wengang Zhou, Dongdong Chen, Dong Chen, Lu Yuan, Houqiang Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.00050)] \
30 Jun 2022


**Remote Sensing Change Detection (Segmentation) using Denoising Diffusion Probabilistic Models** \
*Wele Gedara Chaminda Bandara, Nithin Gopalakrishnan Nair, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.11892)] [[Github](https://github.com/wgcban/ddpm-cd)] \
23 Jun 2022



**Diffusion models as plug-and-play priors** \
*Alexandros Graikos, Nikolay Malkin, Nebojsa Jojic, Dimitris Samaras* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.09012)] \
17 Jun 2022


**Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models** \
*Walter H. L. Pinaya, Mark S. Graham, Robert Gray, Pedro F Da Costa, Petru-Daniel Tudosiu, Paul Wright, Yee H. Mah, Andrew D. MacKinnon, James T. Teo, Rolf Jager, David Werring, Geraint Rees, Parashkev Nachev, Sebastien Ourselin, M. Jorge Cardos* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2206.03461)] \
7 Jun 2022


**Decoder Denoising Pretraining for Semantic Segmentation** \
*Emmanuel Brempong Asiedu, Simon Kornblith, Ting Chen, Niki Parmar, Matthias Minderer, Mohammad Norouzi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.11423)] \
23 May 2022

**Diffusion Models for Implicit Image Segmentation Ensembles** \
*Julia Wolleb, Robin Sandkühler, Florentin Bieder, Philippe Valmaggia, Philippe C. Cattin* \
MIDL 2021. [[Paper](https://arxiv.org/abs/2112.03145)] \
6 Dec 2021

**Label-Efficient Semantic Segmentation with Diffusion Models** \
*Dmitry Baranchuk, Ivan Rubachev, Andrey Voynov, Valentin Khrulkov, Artem Babenko* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2112.03126)] [[Github](https://github.com/yandex-research/ddpm-segmentation)] \
6 Dec 2021

**SegDiff: Image Segmentation with Diffusion Probabilistic Models** \
*Tomer Amit, Eliya Nachmani, Tal Shaharbany, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.00390)] \
1 Dec 2021


**Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions** \
*Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, Max Welling* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2102.05379)] \
10 Feb 2021




### Image Translation

**Latent Diffusion Counterfactual Explanations** \
*Karim Farid, Simon Schrodi, Max Argus, Thomas Brox* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06668)] \
10 Oct 2023

**Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption** \
*Teng Hu, Jiangning Zhang, Liang Liu, Ran Yi, Siqi Kou, Haokun Zhu, Xu Chen, Yabiao Wang, Chengjie Wang, Lizhuang Ma* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2309.03729)] \
7 Sep 2023

**Latent Painter** \
*Shih-Chieh Su* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16490)] \
31 Aug 2023

**Zero-shot Inversion Process for Image Attribute Editing with Diffusion Models** \
*Zhanbo Feng, Zenan Ling, Ci Gong, Feng Zhou, Jie Li, Robert C. Qiu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15854)] \
30 Aug 2023

**DiffI2I: Efficient Diffusion Model for Image-to-Image Translation** \
*Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xinglong Wu, Yapeng Tian, Wenming Yang, Radu Timotfe, Luc Van Gool* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13767)] \
26 Aug 2023

**SSMG: Spatial-Semantic Map Guided Diffusion Model for Free-form Layout-to-Image Generation** \
*Chengyou Jia, Minnan Luo, Zhuohang Dang, Guang Dai, Xiaojun Chang, Mengmeng Wang, Jingdong Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.10156)] \
20 Aug 2023

**MeDM: Mediating Image Diffusion Models for Video-to-Video Translation with Temporal Correspondence Guidance** \
*Ernie Chu, Tzuhsuan Huang, Shuo-Yen Lin, Jun-Cheng Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.10079)] [[Project](https://medm2023.github.io/)] \
19 Aug 2023

**StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models** \
*Zhizhong Wang, Lei Zhao, Wei Xing* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07863)] \
15 Aug 2023

**Inversion-by-Inversion: Exemplar-based Sketch-to-Photo Synthesis via Stochastic Differential Equations without Training** \
*Ximing Xing, Chuang Wang, Haitao Zhou, Zhihao Hu, Chongxuan Li, Dong Xu, Qian Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07665)] \
15 Aug 2023

**Taming the Power of Diffusion Models for High-Quality Virtual Try-On with Appearance Flow** \
*Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, Liqing Zhang* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.06101)] \
11 Aug 2023

**Head Rotation in Denoising Diffusion Models** \
*Andrea Asperti, Gabriele Colasuonno, Antonio Guerra* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06057)] \
11 Aug 2023

**Photorealistic and Identity-Preserving Image-Based Emotion Manipulation with Latent Diffusion Models** \
*Ioannis Pikoulis, Panagiotis P. Filntisis, Petros Maragos* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.03183)] \
6 Aug 2023

**SDDM: Score-Decomposed Diffusion Models on Manifolds for Unpaired Image-to-Image Translation** \
*Shikun Sun, Longhui Wei, Junliang Xing, Jia Jia, Qi Tian* \
ICML 2023. [[Paper](https://arxiv.org/abs/2308.02154)] \
4 Aug 2023

**Interpolating between Images with Diffusion Models** \
*Clinton J. Wang, Polina Golland* \
ICML Workshop 2023. [[Paper](https://arxiv.org/abs/2307.12560)] [[Project](https://clintonjwang.github.io/interpolation)] [[Github](https://github.com/clintonjwang/ControlNet)] \
24 Jul 2023

**TF-ICON: Diffusion-Based Training-Free Cross-Domain Image Composition** \
*Shilin Lu, Yanzhu Liu, Adams Wai-Kin Kong* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2307.12493)] [[Github](https://github.com/Shilin-LU/TF-ICON)] \
24 Jul 2023

**DiffuseGAE: Controllable and High-fidelity Image Manipulation from Disentangled Representation** \
*Yipeng Leng, Qiangjuan Huang, Zhiyuan Wang, Yangyang Liu, Haoyu Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.05899)] \
12 Jul 2023


**DIFF-NST: Diffusion Interleaving For deFormable Neural Style Transfer** \
*Dan Ruta, Gemma Canet Tarrés, Andrew Gilbert, Eli Shechtman, Nicholas Kolkin, John Collomosse* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04157)] \
9 Jul 2023

**Applying a Color Palette with Local Control using Diffusion Models** \
*Vaibhav Vavilala, David Forsyth* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.02698)] \
6 Jul 2023


**DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models** \
*Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, Jian Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.02421)] [[Project](https://mc-e.github.io/project/DragonDiffusion/)] \
5 Jul 2023

**DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing** \
*Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent Y. F. Tan, Song Bai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.14435)] \
26 Jun 2023

**ArtFusion: Controllable Arbitrary Style Transfer using Dual Conditional Latent Diffusion Models** \
*Dar-Yen Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09330)] [[Github](https://github.com/ChenDarYen/ArtFusion)] \
15 Jun 2023


**InfoDiffusion: Representation Learning Using Information Maximizing Diffusion Models** \
*Yingheng Wang, Yair Schiff, Aaron Gokaslan, Weishen Pan, Fei Wang, Christopher De Sa, Volodymyr Kuleshov* \
ICML 2023. [[Paper](https://arxiv.org/abs/2306.08757)] \
14 Jun 2023

**TryOnDiffusion: A Tale of Two UNets** \
*Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, Ira Kemelmacher-Shlizerman* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2306.08276)] \
14 Jun 2023


**Improving Diffusion-based Image Translation using Asymmetric Gradient Guidance** \
*Gihyun Kwon, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04396)] \
7 Jun 2023


**DiffSketching: Sketch Control Image Synthesis with Diffusion Models** \
*Qiang Wang, Di Kong, Fengyin Lin, Yonggang Qi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18812)] \
30 May 2023

**Real-World Image Variation by Aligning Diffusion Inversion Chain** \
*Yuechen Zhang, Jinbo Xing, Eric Lo, Jiaya Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18729)] \
30 May 2023

**Photoswap: Personalized Subject Swapping in Images** \
*Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, Xin Eric Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18286)] [[Project](https://photoswap.github.io/)] \
29 May 2023

**Diversify Your Vision Datasets with Automatic Diffusion-Based Augmentation** \
*Lisa Dunlap, Alyssa Umino, Han Zhang, Jiezhi Yang, Joseph E. Gonzalez, Trevor Darrell* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16289)] [[Github](https://github.com/lisadunlap/ALIA)] \
25 May 2023

**Unpaired Image-to-Image Translation via Neural Schrödinger Bridge** \
*Beomsu Kim, Gihyun Kwon, Kwanyoung Kim, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15086)] [[Github](https://github.com/cyclomon/UNSB)] \
24 May 2023

**SAR-to-Optical Image Translation via Thermodynamics-inspired Network** \
*Mingjin Zhang, Jiamin Xu, Chengyu He, Wenteng Shang, Yunsong Li, Xinbo Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13839)] \
23 May 2023


**Null-text Guidance in Diffusion Models is Secretly a Cartoon-style Creator** \
*Jing Zhao, Heliang Zheng, Chaoyue Wang, Long Lan, Wanrong Huang, Wenjing Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06710)] [[Project](https://nulltextforcartoon.github.io/)] [[Github](https://github.com/NullTextforCartoon/NullTextforCartoon)] \
11 May 2023


**ReGeneration Learning of Diffusion Models with Rich Prompts for Zero-Shot Image Translation** \
*Yupei Lin, Sen Zhang, Xiaojun Yang, Xiao Wang, Yukai Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04651)] [[Project](https://yupeilin2388.github.io/publication/ReDiffuser)] \
8 May 2023

**Hierarchical Diffusion Autoencoders and Disentangled Image Manipulation** \
*Zeyu Lu, Chengyue Wu, Xinyuan Chen, Yaohui Wang, Yu Qiao, Xihui Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11829)] \
24 Apr 2023

**DiffusionRig: Learning Personalized Priors for Facial Appearance Editing** \
*Zheng Ding, Xuaner Zhang, Zhihao Xia, Lars Jebe, Zhuowen Tu, Xiuming Zhang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.06711)] [[Project](https://diffusionrig.github.io/)] [[Github](https://github.com/adobe-research/diffusion-rig)] \
13 Apr 2023



**Face Animation with an Attribute-Guided Diffusion Model** \
*Bohan Zeng, Xuhui Liu, Sicheng Gao, Boyu Liu, Hong Li, Jianzhuang Liu, Baochang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03199)] \
6 Apr 2023



**Reference-based Image Composition with Sketch via Structure-aware Diffusion Model** \
*Kangyeol Kim, Sunghyun Park, Junsoo Lee, Jaegul Choo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.09748)] \
31 Mar 2023

**Training-free Style Transfer Emerges from h-space in Diffusion models** \
*Jaeseok Jeong, Mingi Kwon, Youngjung Uh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15403)] [[Project](https://curryjung.github.io/DiffStyle/)] [[Github](https://github.com/curryjung/DiffStyle_official)] \
27 Mar 2023

**Diffusion-based Target Sampler for Unsupervised Domain Adaptation** \
*Yulong Zhang, Shuhao Chen, Yu Zhang, Jiangang Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12724)] \
17 Mar 2023

**StyO: Stylize Your Face in Only One-Shot** \
*Bonan Li, Zicheng Zhang, Xuecheng Nie, Congying Han, Yinhan Hu, Tiande Guo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03231)] \
6 Mar 2023


**DiffFashion: Reference-based Fashion Design with Structure-aware Transfer by Diffusion Models** \
*Shidong Cao, Wenhao Chai, Shengyu Hao, Yanting Zhang, Hangyue Chen, Gaoang Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.06826)] \
14 Feb 2023

**I2SB: Image-to-Image Schrödinger Bridge** \
*Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, Anima Anandkumar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05872)] [[Project](https://i2sb.github.io/)] \
12 Feb 2023

**Zero-shot-Learning Cross-Modality Data Translation Through Mutual Information Guided Stochastic Diffusion** \
*Zihao Wang, Yingyu Yang, Maxime Sermesant, Hervé Delingette, Ona Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13743)] \
31 Jan 2023

**DiffFace: Diffusion-based Face Swapping with Facial Guidance** \
*Kihong Kim, Yunho Kim, Seokju Cho, Junyoung Seo, Jisu Nam, Kychul Lee, Seungryong Kim, KwangHee Lee* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.13344)] [[Project](https://hxngiee.github.io/DiffFace/)] \
27 Dec 2022

**HS-Diffusion: Learning a Semantic-Guided Diffusion Model for Head Swapping** \
*Qinghe Wang, Lijie Liu, Miao Hua, Qian He, Pengfei Zhu, Bing Cao, Qinghua Hu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06458)] \
13 Dec 2022


**Inversion-Based Creativity Transfer with Diffusion Models** \
*Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, Changsheng Xu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.13203)] [[Github](https://github.com/zyxElsa/InST)] \
23 Nov 2022


**Person Image Synthesis via Denoising Diffusion Model** \
*Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Jorma Laaksonen, Mubarak Shah, Fahad Shahbaz Khan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12500)] \
22 Nov 2022

**Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance** \
*Chen Henry Wu, Fernando De la Torre* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.05559)] [[Github-1](https://github.com/ChenWu98/cycle-diffusion)] [[Github-2](https://github.com/ChenWu98/unified-generative-zoo)] \
11 Oct 2022


**Anatomically constrained CT image translation for heterogeneous blood vessel segmentation** \
*Giammarco La Barbera, Haithem Boussaid, Francesco Maso, Sabine Sarnacki, Laurence Rouet, Pietro Gori, Isabelle Bloch* \
BMVC 2022. [[Paper](https://arxiv.org/abs/2210.01713)] \
4 Oct 2022


**Diffusion-based Image Translation using Disentangled Style and Content Representation** \
*Gihyun Kwon, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.15264)] \
30 Sep 2022


**MIDMs: Matching Interleaved Diffusion Models for Exemplar-based Image Translation** \
*Junyoung Seo, Gyuseong Lee, Seokju Cho, Jiyoung Lee, Seungryong Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.11047)] [[Project](https://ku-cvlab.github.io/MIDMs/)] \
22 Sep 2022


**Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models** \
*Ozan Özdenizci, Robert Legenstein* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.14626)] \
29 Jul 2022

**Non-Uniform Diffusion Models** \
*Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, Christian Etmann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.09786)] \
20 Jul 2022

**Unsupervised Medical Image Translation with Adversarial Diffusion Models** \
*Muzaffer Özbey, Salman UH Dar, Hasan A Bedel, Onat Dalmaz, Şaban Özturk, Alper Güngör, Tolga Çukur* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.08208)] \
17 Jul 2022

**EGSDE: Unpaired Image-to-Image Translation via Energy-Guided Stochastic Differential Equations** \
*Min Zhao, Fan Bao, Chongxuan Li, Jun Zhu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.06635)] \
14 Jul 2022

**Discrete Contrastive Diffusion for Cross-Modal and Conditional Generation** \
*Ye Zhu, Yu Wu, Kyle Olszewski, Jian Ren, Sergey Tulyakov, Yan Yan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.07771)] [[Github](https://github.com/L-YeZhu/CDCD)] \
15 Jun 2022

**Pretraining is All You Need for Image-to-Image Translation** \
*Tengfei Wang, Ting Zhang, Bo Zhang, Hao Ouyang, Dong Chen, Qifeng Chen, Fang Wen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.12952)] [[Project](https://tengfei-wang.github.io/PITI/index.html)] [[Github](https://github.com/PITI-Synthesis/PITI)] \
25 May 2022

**VQBB: Image-to-image Translation with Vector Quantized Brownian Bridge** \
*Bo Li, Kaitao Xue, Bin Liu, Yu-Kun Lai* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.07680)] \
16 May 2022

**The Swiss Army Knife for Image-to-Image Translation: Multi-Task Diffusion Models** \
*Julia Wolleb, Robin Sandkühler, Florentin Bieder, Philippe C. Cattin* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.02641)] \
6 Apr 2022


**Dual Diffusion Implicit Bridges for Image-to-Image Translation** \
*Xuan Su, Jiaming Song, Chenlin Meng, Stefano Ermon* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.08382)] \
16 Mar 2022

**Denoising Diffusion Restoration Models** \
*Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2201.11793)] \
27 Jan 2022

**DiffuseMorph: Unsupervised Deformable Image Registration Along Continuous Trajectory Using Diffusion Models** \
*Boah Kim, Inhwa Han, Jong Chul Ye* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.05149)] \
9 Dec 2021

**Diffusion Autoencoders: Toward a Meaningful and Decodable Representation** \
*Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, Supasorn Suwajanakorn* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.15640)] [[Project](https://diff-ae.github.io/)] \
30 Dec 2021

**Conditional Image Generation with Score-Based Diffusion Models** \
*Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, Christian Etmann* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.13606)] \
26 Nov 2021

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021

**UNIT-DDPM: UNpaired Image Translation with Denoising Diffusion Probabilistic Models**  \
*Hiroshi Sasaki, Chris G. Willcocks, Toby P. Breckon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.05358)] \
12 Apr 2021

### Inverse Problems

**EDiffSR: An Efficient Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution** \
*Yi Xiao, Qiangqiang Yuan, Kui Jiang, Jiang He, Xianyu Jin, Liangpei Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19288)] \
30 Oct 2023


**Global Structure-Aware Diffusion Process for Low-Light Image Enhancement** \
*Jinhui Hou, Zhiyu Zhu, Junhui Hou, Hui Liu, Huanqiang Zeng, Hui Yuan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17577)] \
26 Oct 2023


**From Posterior Sampling to Meaningful Diversity in Image Restoration** \
*Noa Cohen, Hila Manor, Yuval Bahat, Tomer Michaeli* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16047)] \
24 Oct 2023

**Diffusion-Model-Assisted Supervised Learning of Generative Models for Density Estimation** \
*Yanfang Liu, Minglei Yang, Zezhong Zhang, Feng Bao, Yanzhao Cao, Guannan Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.14458)] \
22 Oct 2023

**High-Quality 3D Face Reconstruction with Affine Convolutional Networks** \
*Zhiqian Lin, Jiangke Lin, Lincheng Li, Yi Yuan, Zhengxia Zou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.14237)] \
22 Oct 2023

**Image Super-resolution Via Latent Diffusion: A Sampling-space Mixture Of Experts And Frequency-augmented Decoder Approach** \
*Feng Luo, Jinxi Xiang, Jun Zhang, Xiao Han, Wei Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12004)] \
18 Oct 2023

**Towards image compression with perfect realism at ultra-low bitrates** \
*Marlène Careil, Matthew J. Muckley, Jakob Verbeek, Stéphane Lathuilière* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10325)] \
16 Oct 2023

**AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion** \
*Yitong Jiang, Zhaoyang Zhang, Tianfan Xue, Jinwei Gu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10123)] \
16 Oct 2023

**Exploring the Design Space of Diffusion Autoencoders for Face Morphing** \
*Zander Blasingame, Chen Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09484)] \
14 Oct 2023

**Diffusion Prior Regularized Iterative Reconstruction for Low-dose CT** \
*Wenjun Xia, Yongyi Shi, Chuang Niu, Wenxiang Cong, Ge Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06949)] \
10 Oct 2023


**SMRD: SURE-based Robust MRI Reconstruction with Diffusion Models** \
*Batu Ozturkler, Chao Liu, Benjamin Eckart, Morteza Mardani, Jiaming Song, Jan Kautz* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2310.01799)] [[Github](https://github.com/NVlabs/SMRD)] \
3 Oct 2023

**Conditional Diffusion Distillation** \
*Kangfu Mei, Mauricio Delbracio, Hossein Talebi, Zhengzhong Tu, Vishal M. Patel, Peyman Milanfar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01407)] \
2 Oct 2023

**CommIN: Semantic Image Communications as an Inverse Problem with INN-Guided Diffusion Models** \
*Jiakang Chen, Di You, Deniz Gündüz, Pier Luigi Dragotti* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01130)] \
2 Oct 2023


**Prompt-tuning latent diffusion models for inverse problems** \
*Hyungjin Chung, Jong Chul Ye, Peyman Milanfar, Mauricio Delbracio* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01110)] \
2 Oct 2023

**Steered Diffusion: A Generalized Framework for Plug-and-Play Conditional Image Synthesis** \
*Nithin Gopalakrishnan Nair, Anoop Cherian, Suhas Lohit, Ye Wang, Toshiaki Koike-Akino, Vishal M. Patel, Tim K. Marks* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2310.00224)] \
30 Sep 2023

**Generating Visual Scenes from Touch** \
*Fengyu Yang, Jiacheng Zhang, Andrew Owens* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2309.15117)] [[Project](https://fredfyyang.github.io/vision-from-touch/)] \
26 Sep 2023

**Bootstrap Diffusion Model Curve Estimation for High Resolution Low-Light Image Enhancement** \
*Jiancheng Huang, Yifan Liu, Shifeng Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14709)] \
26 Sep 2023

**Multiple Noises in Diffusion Model for Semi-Supervised Multi-Domain Translation** \
*Tsiry Mayet, Simon Bernard, Clement Chatelain, Romain Herault* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14394)] \
25 Sep 2023

**Domain-Guided Conditional Diffusion Model for Unsupervised Domain Adaptation** \
*Yulong Zhang, Shuhao Chen, Weisen Jiang, Yu Zhang, Jiangang Lu, James T. Kwok* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14360)] \
23 Sep 2023

**License Plate Super-Resolution Using Diffusion Models** \
*Sawsan AlHalawani, Bilel Benjdira, Adel Ammar, Anis Koubaa, Anas M. Ali* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.12506)] \
21 Sep 2023

**Deshadow-Anything: When Segment Anything Model Meets Zero-shot shadow removal** \
*Xiao Feng Zhang, Tian Yi Song, Jia Wei Yao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11715)] \
21 Sep 2023


**Face Aging via Diffusion-based Editing** \
*Xiangyi Chen, Stéphane Lathuilière* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11321)] \
20 Sep 2023

**PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance** \
*Peiqing Yang, Shangchen Zhou, Qingyi Tao, Chen Change Loy* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2309.10810)] [[Github](https://github.com/pq-yang/PGDiff)] \
19 Sep 2023

**Reconstruct-and-Generate Diffusion Model for Detail-Preserving Image Denoising** \
*Yujin Wang, Lingen Li, Tianfan Xue, Jinwei Gu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.10714)] \
19 Sep 2023

**Gradpaint: Gradient-Guided Inpainting with Diffusion Models** \
*Asya Grechka, Guillaume Couairon, Matthieu Cord* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.09614)] \
18 Sep 2023



**AdBooster: Personalized Ad Creative Generation using Stable Diffusion Outpainting** \
*Veronika Shilova, Ludovic Dos Santos, Flavian Vasile, Gaëtan Racic, Ugo Tanielian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11507)] \
8 Sep 2023

**Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy** \
*Yi Tang, Takafumi Iwaguchi, Hiroshi Kawasaki* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2309.03445)] [[Github](https://github.com/piggy2009/DM_underwater)] \
7 Sep 2023

**Efficient Bayesian Computational Imaging with a Surrogate Score-Based Prior** \
*Berthy T. Feng, Katherine L. Bouman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01949)] \
5 Sep 2023

**Diffusion Modeling with Domain-conditioned Prior Guidance for Accelerated MRI and qMRI Reconstruction** \
*Wanyu Bian, Albert Jang, Fang Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00783)] \
2 Sep 2023

**Correlated and Multi-frequency Diffusion Modeling for Highly Under-sampled MRI Reconstruction** \
*Yu Guan, Chuanming Yu, Shiyu Lu, Zhuoxu Cui, Dong Liang, Qiegen Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00853)] [[Github](https://github.com/yqx7150/CM-DM)] \
2 Sep 2023

**Fast Diffusion EM: a diffusion model for blind inverse problems with application to deconvolution** \
*Charles Laroche, Andrés Almansa, Eva Coupete* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00287)] [[Github](https://anonymous.4open.science/r/FastDiffusionEM-26BE/README.md)] \
1 Sep 2023

**Unsupervised CT Metal Artifact Reduction by Plugging Diffusion Priors in Dual Domains** \
*Xuan Liu, Yaoqin Xie, Songhui Diao, Shan Tan, Xiaokun Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16742)] \
31 Aug 2023

**Stage-by-stage Wavelet Optimization Refinement Diffusion Model for Sparse-View CT Reconstruction** \
*Kai Xu, Shiyu Lu, Bin Huang, Weiwen Wu, Qiegen Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15942)] \
30 Aug 2023

**Physics-Informed DeepMRI: Bridging the Gap from Heat Diffusion to k-Space Interpolation** \
*Zhuo-Xu Cui, Congcong Liu, Xiaohong Fan, Chentao Cao, Jing Cheng, Qingyong Zhu, Yuanyuan Liu, Sen Jia, Yihang Zhou, Haifeng Wang, Yanjie Zhu, Jianping Zhang, Qiegen Liu, Dong Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15918)] \
30 Aug 2023

**DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior** \
*Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Ben Fei, Bo Dai, Wanli Ouyang, Yu Qiao, Chao Dong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15070)] [[Github](https://github.com/XPixelGroup/DiffBIR)] \
29 Aug 2023

**Pixel-Aware Stable Diffusion for Realistic Image Super-resolution and Personalized Stylization** \
*Tao Yang, Peiran Ren, Xuansong Xie, Lei Zhang* \
AAAI 2024. [[Paper](https://arxiv.org/abs/2308.14469)] \
28 Aug 2023

**Data-iterative Optimization Score Model for Stable Ultra-Sparse-View CT Reconstruction** \
*Weiwen Wu, Yanyang Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.14437)] \
28 Aug 2023

**Residual Denoising Diffusion Models** \
*Jiawei Liu, Qiang Wang, Huijie Fan, Yinong Wang, Yandong Tang, Liangqiong Qu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13712)] [[Github](https://github.com/nachifur/RDDM)] \
25 Aug 2023

**Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model** \
*Xunpeng Yi, Han Xu, Hao Zhang, Linfeng Tang, Jiayi Ma* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.13164)] \
25 Aug 2023

**Full-dose PET Synthesis from Low-dose PET Using High-efficiency Diffusion Denoising Probabilistic Model** \
*Shaoyan Pan, Elham Abouei, Junbo Peng, Joshua Qian, Jacob F Wynne, Tonghe Wang, Chih-Wei Chang, Justin Roper, Jonathon A Nye, Hui Mao, Xiaofeng Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13072)] \
24 Aug 2023

**InverseSR: 3D Brain MRI Super-Resolution Using a Latent Diffusion Model** \
*Jueqi Wang, Jacob Levman, Walter Hugo Lopez Pinaya, Petru-Daniel Tudosiu, M. Jorge Cardoso, Razvan Marinescu* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2308.12465)] [[Github](https://github.com/BioMedAI-UCSC/InverseSR)] \
23 Aug 2023

**High-quality Image Dehazing with Diffusion Model** \
*Hu Yu, Jie Huang, Kaiwen Zheng, Man Zhou, Feng Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.11949)] \
23 Aug 2023

**Frequency Compensated Diffusion Model for Real-scene Dehazing** \
*Jing Wang, Songtao Wu, Kuanhong Xu, Zhiqiang Yuan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.10510)] \
21 Aug 2023



**Contrastive Diffusion Model with Auxiliary Guidance for Coarse-to-Fine PET Reconstruction** \
*Zeyu Han, Yuhan Wang, Luping Zhou, Peng Wang, Binyu Yan, Jiliu Zhou, Yan Wang, Dinggang Shen* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2308.10157)] [[Github](https://github.com/Show-han/PET-Reconstruction)] \
20 Aug 2023

**DiffLLE: Diffusion-guided Domain Calibration for Unsupervised Low-light Image Enhancement** \
*Shuzhou Yang, Xuanyu Zhang, Yinhuai Wang, Jiwen Yu, Yuhan Wang, Jian Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09279)] \
18 Aug 2023

**Learning A Coarse-to-Fine Diffusion Transformer for Image Restoration** \
*Liyan Wang, Qinyu Yang, Cong Wang, Wei Wang, Jinshan Pan, Zhixun Su* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.08730)] \
17 Aug 2023

**Monte Carlo guided Diffusion for Bayesian linear inverse problems** \
*Gabriel Cardoso, Yazid Janati El Idrissi, Sylvain Le Corff, Eric Moulines* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07983)] \
15 Aug 2023

**Geometry of the Visual Cortex with Applications to Image Inpainting and Enhancement** \
*Francesco Ballerin, Erlend Grong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07652)] [[Github](https://github.com/ballerin/v1diffusion)] \
15 Aug 2023

**YODA: You Only Diffuse Areas. An Area-Masked Diffusion Approach For Image Super-Resolution** \
*Brian B. Moser, Stanislav Frolov, Federico Raue, Sebastian Palacio, Andreas Dengel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07977)] \
15 Aug 2023

**TextDiff: Mask-Guided Residual Diffusion Models for Scene Text Image Super-Resolution** \
*Baolin Liu, Zongyuan Yang, Pengfei Wang, Junjie Zhou, Ziqi Liu, Ziyi Song, Yan Liu, Yongping Xiong* \
AAAI 2024. [[Paper](https://arxiv.org/abs/2308.06743)] \
13 Aug 2023

**CLE Diffusion: Controllable Light Enhancement Diffusion Model** \
*Yuyang Yin, Dejia Xu, Chuangchuang Tan, Ping Liu, Yao Zhao, Yunchao Wei* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06725)] [[Project](https://yuyangyin.github.io/CLEDiffusion/)] [[Github](https://github.com/YuyangYin/CLEDiffusion)] \
13 Aug 2023

**Diffusion-Augmented Depth Prediction with Sparse Annotations** \
*Jiaqi Li, Yiran Wang, Zihao Huang, Jinghong Zheng, Ke Xian, Zhiguo Cao, Jianming Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02283)] \
4 Aug 2023

**Painterly Image Harmonization using Diffusion Model** \
*Lingxiao Lu, Jiangtong Li, Junyan Cao, Li Niu, Liqing Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02228)] \
4 Aug 2023

**Reference-Free Isotropic 3D EM Reconstruction using Diffusion Models** \
*Kyungryun Lee, Won-Ki Jeong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01594)] \
3 Aug 2023



**Learning Fourier-Constrained Diffusion Bridges for MRI Reconstruction** \
*Muhammad U. Mirza, Onat Dalmaz, Hasan A. Bedel, Gokberk Elmas, Yilmaz Korkmaz, Alper Gungor, Salman UH Dar, Tolga Çukur* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01096)] \
2 Aug 2023


**Ultrasound Image Reconstruction with Denoising Diffusion Restoration Models** \
*Yuxin Zhang, Clément Huneau, Jérôme Idier, Diana Mateus* \
MICCAI Workshop 2023. [[Paper](https://arxiv.org/abs/2307.15990)] [[Github](https://github.com/Yuxin-Zhang-Jasmine/DRUS-v1)] \
29 Jul 2023

**LLDiffusion: Learning Degradation Representations in Diffusion Models for Low-Light Image Enhancement** \
*Tao Wang, Kaihao Zhang, Ziqian Shao, Wenhan Luo, Bjorn Stenger, Tae-Kyun Kim, Wei Liu, Hongdong Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.14659)] \
27 Jul 2023

**Artifact Restoration in Histology Images with Diffusion Probabilistic Models** \
*Zhenqi He, Junjun He, Jin Ye, Yiqing Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.14262)] [[Github](https://github.com/zhenqi-he/ArtiFusion)] \
26 Jul 2023

**ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting** \
*Zongsheng Yue, Jianyi Wang, Chen Change Loy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.12348)] [[Github](https://github.com/zsyOAOA/ResShift)] \
23 Jul 2023


**Iterative Reconstruction Based on Latent Diffusion Model for Sparse Data Reconstruction** \
*Linchao He, Hongyu Yan, Mengting Luo, Kunming Luo, Wang Wang, Wenchao Du, Hu Chen, Hongyu Yang, Yi Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.12070)] \
22 Jul 2023

**PartDiff: Image Super-resolution with Partial Diffusion Models** \
*Kai Zhao, Alex Ling Yu Hung, Kaifeng Pang, Haoxin Zheng, Kyunghyun Sung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.11926)] \
21 Jul 2023

**Reference-based Painterly Inpainting via Diffusion: Crossing the Wild Reference Domain Gap** \
*Dejia Xu, Xingqian Xu, Wenyan Cong, Humphrey Shi, Zhangyang Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.10584)] [[Project](https://vita-group.github.io/RefPaint/)] \
20 Jul 2023

**AnyDoor: Zero-shot Object-level Image Customization** \
*Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, Hengshuang Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.09481)] [[Project](https://damo-vilab.github.io/AnyDoor-Page/)] \
18 Jul 2023

**Towards Authentic Face Restoration with Iterative Diffusion Models and Beyond** \
*Yang Zhao, Tingbo Hou, Yu-Chuan Su, Xuhui Jia. Yandong Li, Matthias Grundmann* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2307.08996)] \
18 Jul 2023



**Flow Matching in Latent Space** \
*Quan Dao, Hao Phung, Binh Nguyen, Anh Tran* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08698)] [[Project](https://vinairesearch.github.io/LFM/)] \
17 Jul 2023

**Identity-Preserving Aging of Face Images via Latent Diffusion Models** \
*Sudipta Banerjee, Govind Mittal, Ameya Joshi, Chinmay Hegde, Nasir Memon* \
IJCB 2023. [[Paper](https://arxiv.org/abs/2307.08585)] \
17 Jul 2023



**Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency** \
*Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, Liyue Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08123)] \
16 Jul 2023

**ExposureDiffusion: Learning to Expose for Low-light Image Enhancement** \
*Yufei Wang, Yi Yu, Wenhan Yang, Lanqing Guo, Lap-Pui Chau, Alex C. Kot, Bihan Wen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.07710)] \
15 Jul 2023

**DDGM: Solving inverse problems by Diffusive Denoising of Gradient-based Minimization** \
*Kyle Luther, H. Sebastian Seung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04946)] \
11 Jul 2023

**Stimulating the Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling** \
*Tong Li, Hansen Feng, Lizhi Wang, Zhiwei Xiong, Hua Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.03992)] \
8 Jul 2023

**IPO-LDM: Depth-aided 360-degree Indoor RGB Panorama Outpainting via Latent Diffusion Model** \
*Tianhao Wu, Chuanxia Zheng, Tat-Jen Cham* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.03177)] [[Github](https://sm0kywu.github.io/ipoldm/)] \
6 Jul 2023

**Single Image LDR to HDR Conversion using Conditional Diffusion** \
*Dwip Dalal, Gautam Vashishtha, Prajwal Singh, Shanmuganathan Raman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.02814)] \
6 Jul 2023

**ACDMSR: Accelerated Conditional Diffusion Models for Single Image Super-Resolution** \
*Axi Niu, Pham Xuan Trung, Kang Zhang, Jinqiu Sun, Yu Zhu, In So Kweon, Yanning Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.00781)] \
3 Jul 2023

**LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance** \
*Linoy Tsaban, Apolinário Passos* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.00522)] \
2 Jul 2023


**Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models** \
*Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alexandros G. Dimakis, Sanjay Shakkottai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.00619)] [[Github](https://github.com/LituRout/PSLD)] \
2 Jul 2023

**Content-Preserving Diffusion Model for Unsupervised AS-OCT image Despeckling** \
*Li Sanqian, Higashita Risa, Fu Huazhu, Li Heng, Niu Jingxuan, Liu Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17717)] \
30 Jun 2023


**Self-Supervised MRI Reconstruction with Unrolled Diffusion Models** \
*Yilmaz Korkmaz, Tolga Cukur, Vishal Patel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.16654)] \
29 Jun 2023

**SVNR: Spatially-variant Noise Removal with Denoising Diffusion** \
*Naama Pearl, Yaron Brodsky, Dana Berman, Assaf Zomet, Alex Rav Acha, Daniel Cohen-Or, Dani Lischinski* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.16052)] \
28 Jun 2023


**Easing Color Shifts in Score-Based Diffusion Models** \
*Katherine Deck, Tobias Bischoff* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.15832)] \
27 Jun 2023

**Diffusion Model Based Low-Light Image Enhancement for Space Satellite** \
*Yiman Zhu, Lu Wang, Jingyi Yuan, Yu Guo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.14227)] \
25 Jun 2023

**DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology** \
*Marco Aversa, Gabriel Nobis, Miriam Hägele, Kai Standvoss, Mihaela Chirica, Roderick Murray-Smith, Ahmed Alaa, Lukas Ruff, Daniela Ivanova, Wojciech Samek, Frederick Klauschen, Bruno Sanguinetti, Luis Oala* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.13384)] \
23 Jun 2023


**Wind Noise Reduction with a Diffusion-based Stochastic Regeneration Model** \
*Jean-Marie Lemercier, Joachim Thiemann, Raphael Koning, Timo Gerkmann* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12867)] \
22 Jun 2023


**DiffuseIR:Diffusion Models For Isotropic Reconstruction of 3D Microscopic Images** \
*Mingjie Pan, Yulu Gan, Fangxu Zhou, Jiaming Liu, Aimin Wang, Shanghang Zhang, Dawei Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12109)] \
21 Jun 2023


**HSR-Diff:Hyperspectral Image Super-Resolution via Conditional Diffusion Models** \
*Chanyue Wu, Dong Wang, Hanyu Mao, Ying Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12085)] \
21 Jun 2023


**Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision** \
*Ayush Tewari, Tianwei Yin, George Cazenavette, Semon Rezchikov, Joshua B. Tenenbaum, Frédo Durand, William T. Freeman, Vincent Sitzmann* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11719)] \
20 Jun 2023

**Deep Ultrasound Denoising Using Diffusion Probabilistic Models** \
*Hojat Asgariandehkordi, Sobhan Goudarzi, Adrian Basarab, Hassan Rivaz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07440)] \
12 Jun 2023

**Towards Visual Foundational Models of Physical Scenes** \
*Chethan Parameshwara, Alessandro Achille, Matthew Trager, Xiaolong Li, Jiawei Mo, Matthew Trager, Ashwin Swaminathan, CJ Taylor, Dheera Venkatraman, Xiaohan Fei, Stefano Soatto* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03727)] \
6 Jun 2023


**INDigo: An INN-Guided Probabilistic Diffusion Algorithm for Inverse Problems** \
*Di You, Andreas Floros, Pier Luigi Dragotti* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02949)] \
5 Jun 2023

**The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation** \
*Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, David J. Fleet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01923)] \
2 Jun 2023

**Dissecting Arbitrary-scale Super-resolution Capability from Pre-trained Diffusion Generative Models** \
*Ruibin Li, Qihua Zhou, Song Guo, Jie Zhang, Jingcai Guo, Xinyang Jiang, Yifei Shen, Zhenhua Han* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00714)] \
1 Jun 2023

**Low-Light Image Enhancement with Wavelet-based Diffusion Models** \
*Hai Jiang, Ao Luo, Songchen Han, Haoqiang Fan, Shuaicheng Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00306)] \
1 Jun 2023

**A Unified Conditional Framework for Diffusion-based Image Restoration** \
*Yi Zhang, Xiaoyu Shi, Dasong Li, Xiaogang Wang, Jian Wang, Hongsheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.20049)] \
31 May 2023

**Direct Diffusion Bridge using Data Consistency for Inverse Problems** \
*Hyungjin Chung, Jeongsol Kim, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19809)] \
31 May 2023

**Accelerating Diffusion Models for Inverse Problems through Shortcut Sampling** \
*Gongye Liu, Haoze Sun, Jiayi Li, Fei Yin, Yujiu Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16965)] \
26 May 2023

**Look Ma, No Hands! Agent-Environment Factorization of Egocentric Videos** \
*Matthew Chang, Aditya Prakash, Saurabh Gupta* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16301)] [[Project](https://matthewchang.github.io/vidm/)] \
25 May 2023

**A Diffusion Probabilistic Prior for Low-Dose CT Image Denoising** \
*Xuan Liu, Yaoqin Xie, Songhui Diao, Shan Tan, Xiaokun Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15887)] \
25 May 2023

**Solving Diffusion ODEs with Optimal Boundary Conditions for Better Image Super-Resolution** \
*Yiyang Ma, Huan Yang, Wenhan Yang, Jianlong Fu, Jiaying Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15357)] \
24 May 2023

**WaveDM: Wavelet-Based Diffusion Models for Image Restoration** \
*Yi Huang, Jiancheng Huang, Jianzhuang Liu, Yu Dong, Jiaxi Lv, Shifeng Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13819)] \
23 May 2023

**Dual-Diffusion: Dual Conditional Denoising Diffusion Probabilistic Models for Blind Super-Resolution Reconstruction in RSIs** \
*Mengze Xu, Jie Ma, Yuanyuan Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12170)] [[Github](https://github.com/Lincoln20030413/DDSR)] \
20 May 2023

**UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild** \
*Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, Ran Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11147)] \
18 May 2023

**Pyramid Diffusion Models For Low-light Image Enhancement** \
*Dewei Zhou, Zongxin Yang, Yi Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10028)] \
17 May 2023

**A Conditional Denoising Diffusion Probabilistic Model for Radio Interferometric Image Reconstruction** \
*Ruoqi Wang, Zhuoyang Chen, Qiong Luo, Feng Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09121)]
16 May 2023

**Denoising Diffusion Models for Plug-and-Play Image Restoration** \
*Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, Luc Van Gool* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08995)] [[Github](https://github.com/yuanzhi-zhu/DiffPIR)] \
15 May 2023


**Exploiting Diffusion Prior for Real-World Image Super-Resolution** \
*Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C.K. Chan, Chen Change Loy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.07015)] [[Project](https://iceclear.github.io/projects/stablesr/)] [[Github](https://github.com/IceClear/StableSR)] \
11 May 2023

**Atmospheric Turbulence Correction via Variational Deep Diffusion** \
*Xijun Wang, Santiago López-Tapia, Aggelos K. Katsaggelos* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05077)] \
8 May 2023

**Controllable Light Diffusion for Portraits** \
*David Futschik, Kelvin Ritland, James Vecore, Sean Fanello, Sergio Orts-Escolano, Brian Curless, Daniel Sýkora, Rohit Pandey* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04745)] \
8 May 2023

**DiffBFR: Bootstrapping Diffusion Model Towards Blind Face Restoration** \
*Xinmin Qiu, Congying Han, ZiCheng Zhang, Bonan Li, Tiande Guo, Xuecheng Nie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04517)] \
8 May 2023

**Real-World Denoising via Diffusion Model** \
*Cheng Yang, Lijing Liang, Zhixun Su* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04457)] \
8 May 2023

**A Variational Perspective on Solving Inverse Problems with Diffusion Models** \
*Morteza Mardani, Jiaming Song, Jan Kautz, Arash Vahdat* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04391)] \
7 May 2023

**Synthesizing PET images from High-field and Ultra-high-field MR images Using Joint Diffusion Attention Model** \
*Taofeng Xie, Chentao Cao, Zhuoxu Cui, Yu Guo, Caiying Wu, Xuemei Wang, Qingneng Li, Zhanli Hu, Tao Sun, Ziru Sang, Yihang Zhou, Yanjie Zhu, Dong Liang, Qiyu Jin, Guoqing Chen, Haifeng Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03901)] \
6 May 2023



**DocDiff: Document Enhancement via Residual Diffusion Models** \
*Zongyuan Yang, Baolin Liu, Yongping Xiong, Lan Yi, Guibin Wu, Xiaojun Tang, Ziqi Liu, Junjie Zhou, Xing Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03892)] [[Github](https://github.com/Royalvice/DocDiff)] \
6 May 2023

**Solving Inverse Problems with Score-Based Generative Priors learned from Noisy Data** \
*Asad Aali, Marius Arvinte, Sidharth Kumar, Jonathan I. Tamir* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01166)] \
2 May 2023

**Self-similarity-based super-resolution of photoacoustic angiography from hand-drawn doodles** \
*Yuanzheng Ma, Wangting Zhou, Rui Ma, Sihua Yang, Yansong Tang, Xun Guan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01165)] \
2 May 2023

**Score-Based Diffusion Models as Principled Priors for Inverse Imaging** \
*Berthy T. Feng, Jamie Smith, Michael Rubinstein, Huiwen Chang, Katherine L. Bouman, William T. Freeman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11751)] \
23 Apr 2023

**Improved Diffusion-based Image Colorization via Piggybacked Models** \
*Hanyuan Liu, Jinbo Xing, Minshan Xie, Chengze Li, Tien-Tsin Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11105)] [[Project](https://piggyback-color.github.io/)] \
21 Apr 2023

**DiFaReli: Diffusion Face Relighting** \
*Puntawat Ponglertnapakorn, Nontawat Tritrong, Supasorn Suwajanakorn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.09479)] [[Project](https://diffusion-face-relighting.github.io/)] \
19 Apr 2023

**Inpaint Anything: Segment Anything Meets Image Inpainting** \
*Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, Zhibo Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06790)] [[Github](https://github.com/geekyutao/Inpaint-Anything)] \
13 Apr 2023

**Refusion: Enabling Large-Size Realistic Image Restoration with Latent-Space Diffusion Models** \
*Ziwei Luo, Fredrik K. Gustafsson, Zheng Zhao, Jens Sjölund, Thomas B. Schön* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08291)] [[Github](https://github.com/Algolzw/image-restoration-sde)] \
17 Apr 2023


**SPIRiT-Diffusion: Self-Consistency Driven Diffusion Model for Accelerated MRI** \
*Zhuo-Xu Cui, Chentao Cao, Jing Cheng, Sen Jia, Hairong Zheng, Dong Liang, Yanjie Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05060)] \
11 Apr 2023

**Zero-shot CT Field-of-view Completion with Unconditional Generative Diffusion Prior** \
*Kaiwen Xu, Aravind R. Krishnan, Thomas Z. Li, Yuankai Huo, Kim L. Sandler, Fabien Maldonado, Bennett A. Landman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03760)] \
7 Apr 2023

**SketchFFusion: Sketch-guided image editing with diffusion model** \
*Weihang Mao, Bo Han, Zihao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03174)] \
6 Apr 2023


**Inst-Inpaint: Instructing to Remove Objects with Diffusion Models** \
*Ahmet Burak Yildirim, Vedat Baday, Erkut Erdem, Aykut Erdem, Aysegul Dundar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03246)] [[Project](http://instinpaint.abyildirim.com/)] \
6 Apr 2023

**Towards Coherent Image Inpainting Using Denoising Diffusion Implicit Models** \
*Guanhua Zhang, Jiabao Ji, Yang Zhang, Mo Yu, Tommi Jaakkola, Shiyu Chang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03322)] [[Github](https://github.com/UCSB-NLP-Chang/CoPaint/)] \
6 Apr 2023

**Zero-shot Medical Image Translation via Frequency-Guided Diffusion Models** \
*Yunxiang Li, Hua-Chieh Shao, Xiao Liang, Liyuan Chen, Ruiqi Li, Steve Jiang, Jing Wang, You Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02742)] \
5 Apr 2023

**Waving Goodbye to Low-Res: A Diffusion-Wavelet Approach for Image Super-Resolution** \
*Brian Moser, Stanislav Frolov, Federico Raue, Sebastian Palacio, Andreas Dengel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01994)] \
4 Apr 2023


**CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization** \
*Qi Gao, Zilong Li, Junping Zhang, Yi Zhang, Hongming Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01814)] \
4 Apr 2023


**Generative Diffusion Prior for Unified Image Restoration and Enhancement** \
*Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, Bo Dai* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.01247)] \
3 Apr 2023

**Implicit Diffusion Models for Continuous Super-Resolution** \
*Sicheng Gao, Xuhui Liu, Bohan Zeng, Sheng Xu, Yanjing Li, Xiaoyan Luo, Jianzhuang Liu, Xiantong Zhen, Baochang Zhang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.16491)] \
29 Mar 2023


**DiracDiffusion: Denoising and Incremental Reconstruction with Assured Data-Consistency** \
*Zalan Fabian, Berk Tinaz, Mahdi Soltanolkotabi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14353)] \
25 Mar 2023

**MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion** \
*Yizhuo Lu, Changde Du, Dianpeng Wang, Huiguang He* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14139)] \
24 Mar 2023

**DisC-Diff: Disentangled Conditional Diffusion Model for Multi-Contrast MRI Super-Resolution** \
*Ye Mao, Lan Jiang, Xi Chen, Chao Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13933)] \
23 Mar 2023


**Sub-volume-based Denoising Diffusion Probabilistic Model for Cone-beam CT Reconstruction from Incomplete Data** \
*Wenjun Xia, Chuang Niu, Wenxiang Cong, Ge Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12861)] \
22 Mar 2023



**A Perceptual Quality Assessment Exploration for AIGC Images** \
*Zicheng Zhang, Chunyi Li, Wei Sun, Xiaohong Liu, Xiongkuo Min, Guangtao Zhai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12618)] \
22 Mar 2023

**Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration** \
*Mauricio Delbracio, Peyman Milanfar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11435)] \
20 Mar 2023

**Efficient Neural Generation of 4K Masks for Homogeneous Diffusion Inpainting** \
*Karl Schrader, Pascal Peter, Niklas Kämper, Joachim Weickert* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10096)] \
17 Mar 2023

**Denoising Diffusion Post-Processing for Low-Light Image Enhancement** \
*Savvas Panagiotou, Anna S. Bosman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09627)] \
16 Mar 2023

**SUD2: Supervision by Denoising Diffusion Models for Image Reconstruction** \
*Matthew A. Chan, Sean I. Young, Christopher A. Metzler* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09642)] \
16 Mar 2023

**DiffIR: Efficient Diffusion Model for Image Restoration** \
*Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xinglong Wu, Yapeng Tian, Wenming Yang, Luc Van Gool* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09472)] \
16 Mar 2023

**ResDiff: Combining CNN and Diffusion Model for Image Super-Resolution** \
*Shuyao Shang, Zhengyang Shan, Guangxing Liu, Jinglin Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08714)] \
15 Mar 2023

**Class-Guided Image-to-Image Diffusion: Cell Painting from Brightfield Images with Class Labels** \
*Jan Oscar Cross-Zamirski, Praveen Anand, Guy Williams, Elizabeth Mouchet, Yinhai Wang, Carola-Bibiane Schönlieb* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08863)] [[Github](https://github.com/crosszamirski/guided-I2I)] \
15 Mar 2023


**Diffusion Models for Contrast Harmonization of Magnetic Resonance Images** \
*Alicia Durrer, Julia Wolleb, Florentin Bieder, Tim Sinnecker, Matthias Weigel, Robin Sandkühler, Cristina Granziera, Özgür Yaldizli, Philippe C. Cattin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08189)] \
14 Mar 2023

**Synthesizing Realistic Image Restoration Training Pairs: A Diffusion Approach** \
*Tao Yang, Peiran Ren, Xuansong xie, Lei Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06994)] \
13 Mar 2023

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration** \
*Zhixin Wang, Xiaoyun Zhang, Ziying Zhang, Huangjie Zheng, Mingyuan Zhou, Ya Zhang, Yanfeng Wang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.06885)] \
13 Mar 2023

**DDS2M: Self-Supervised Denoising Diffusion Spatio-Spectral Model for Hyperspectral Image Restoration** \
*Yuchun Miao, Lefei Zhang, Liangpei Zhang, Dacheng Tao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06682)] \
12 Mar 2023


**Fast Diffusion Sampler for Inverse Problems by Geometric Decomposition** \
*Hyungjin Chung, Suhyeon Lee, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05754)] \
10 Mar 2023

**Generalized Diffusion MRI Denoising and Super-Resolution using Swin Transformers** \
*Amir Sadikov, Jamie Wren-Jarvis, Xinlei Pan, Lanya T. Cai, Pratik Mukherjee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05686)] \
10 Mar 2023

**DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation** \
*Yiqun Duan, Zheng Zhu, Xianda Guo* \
arxiv 2023. [[Paper](https://arxiv.org/abs/2303.05021)] [[Github](https://github.com/duanyiqun/DiffusionDepth)] \
9 Mar 2023

**Learning Enhancement From Degradation: A Diffusion Model For Fundus Image Enhancement** \
*Puijin Cheng, Li Lin, Yijin Huang, Huaqing He, Wenhan Luo, Xiaoying Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04603)] [[Github](https://github.com/QtacierP/LED)] \
8 Mar 2023

**Unlimited-Size Diffusion Restoration** \
*Yinhuai Wang, Jiwen Yu, Runyi Yu, Jian Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.00354)] \
1 Mar 2023

**Unsupervised Out-of-Distribution Detection with Diffusion Inpainting** \
*Zhenzhen Liu, Jin Peng Zhou, Yufan Wang, Kilian Q. Weinberger* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10326)] \
20 Feb 2023

**Restoration based Generative Models** \
*Jaemoo Choi, Yesom Park, Myungjoo Kang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05456)] \
20 Feb 2023

**Explicit Diffusion of Gaussian Mixture Model Based Image Priors** \
*Martin Zach, Thomas Pock, Erich Kobler, Antonin Chambolle* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08411)] \
16 Feb 2023

**Denoising Diffusion Probabilistic Models for Robust Image Super-Resolution in the Wild** \
*Hshmat Sahak, Daniel Watson, Chitwan Saharia, David Fleet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07864)] \
15 Feb 2023



**CDPMSR: Conditional Diffusion Probabilistic Models for Single Image Super-Resolution** \
*Axi Niu, Kang Zhang, Trung X. Pham, Jinqiu Sun, Yu Zhu, In So Kweon, Yanning Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12831)] \
14 Feb 2023

**How to Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control** \
*Jacopo Teneggi, Matt Tivnan, J Webster Stayman, Jeremias Sulam* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03791)] \
7 Feb 2023

**DDM2: Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models** \
*Tiange Xiang, Mahmut Yurt, Ali B Syed, Kawin Setsompop, Akshay Chaudhari* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2302.03018)] [[Github](https://github.com/StanfordMIMI/DDM2)] \
6 Feb 2023




**Diffusion Model for Generative Image Denoising** \
*Yutong Xie, Minne Yuan, Bin Dong, Quanzheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02398)] \
5 Feb 2023



**A Theoretical Justification for Image Inpainting using Denoising Diffusion Probabilistic Models** \
*Litu Rout, Advait Parulekar, Constantine Caramanis, Sanjay Shakkottai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01217)] \
2 Feb 2023

**GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration** \
*Naoki Murata, Koichi Saito, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Yuki Mitsufuji, Stefano Ermon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12686)] \
30 Jan 2023


**Accelerating Guided Diffusion Sampling with Splitting Numerical Methods** \
*Suttisak Wizadwongsa, Supasorn Suwajanakorn* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2301.11558)] \
27 Jan 2023

**Diffusion Denoising for Low-Dose-CT Model** \
*Runyi Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11482)] \
27 Jan 2023

**Screen Space Indirect Lighting with Visibility Bitmask** \
*Olivier Therrien, Yannick Levesque, Guillaume Gilet* \
Visual Computer 2023. [[Paper](https://arxiv.org/abs/2301.11376)] \
26 Jan 2023


**Dual Diffusion Architecture for Fisheye Image Rectification: Synthetic-to-Real Generalization** \
*Shangrong Yang, Chunyu Lin, Kang Liao, Yao Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11785)] \
26 Jan 2023

**RainDiffusion:When Unsupervised Learning Meets Diffusion Models for Real-world Image Deraining** \
*Mingqiang Wei, Yiyang Shen, Yongzhen Wang, Haoran Xie, Fu Lee Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.09430)] \
23 Jan 2023

**Dif-Fusion: Towards High Color Fidelity in Infrared and Visible Image Fusion with Diffusion Models** \
*Mingqiang Wei, Yiyang Shen, Yongzhen Wang, Haoran Xie, Fu Lee Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.09430)] \
23 Jan 2023

**Removing Structured Noise with Diffusion Models** \
*Tristan S.W. Stevens, Jean-Luc Robert, Faik C. Meral Jason Yu, Jun Seob Shin, Ruud J.G. van Sloun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05290)] \
20 Jan 2023

**Image Restoration with Mean-Reverting Stochastic Differential Equations** \
*Ziwei Luo, Fredrik K. Gustafsson, Zheng Zhao, Jens Sjölund, Thomas B. Schön* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11699)] [[Github](https://github.com/Algolzw/image-restoration-sde)] \
20 Jan 2023


**DiffusionCT: Latent Diffusion Model for CT Image Standardization** \
*Md Selim, Jie Zhang, Michael A. Brooks, Ge Wang, Jin Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.08815)] \
20 Jan 2023

**Targeted Image Reconstruction by Sampling Pre-trained Diffusion Model** \
*Jiageng Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.07557)] \
18 Jan 2023

**Annealed Score-Based Diffusion Model for MR Motion Artifact Reduction** \
*Gyutaek Oh, Jeong Eun Lee, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.03027)] \
8 Jan 2023

**Exploring Vision Transformers as Diffusion Learners** \
*He Cao, Jianan Wang, Tianhe Ren, Xianbiao Qi, Yihao Chen, Yuan Yao, Lei Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.13771)] \
28 Dec 2022


**Towards Blind Watermarking: Combining Invertible and Non-invertible Mechanisms** \
*Rui Ma, Mengxi Guo, Yi Hou, Fan Yang, Yuan Li, Huizhu Jia, Xiaodong Xie* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.12678)] [[Github](https://github.com/rmpku/CIN)] \
24 Dec 2022

**Bi-Noising Diffusion: Towards Conditional Diffusion Models with Generative Restoration Priors** \
*Kangfu Mei, Nithin Gopalakrishnan Nair, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.07352)] [[Project](https://kfmei.page/bi-noising/)] \
14 Dec 2022

**SPIRiT-Diffusion: SPIRiT-driven Score-Based Generative Modeling for Vessel Wall imaging** \
*Chentao Cao, Zhuo-Xu Cui, Jing Cheng, Sen Jia, Hairong Zheng, Dong Liang, Yanjie Zhu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.11274)] \
14 Dec 2022

**Universal Generative Modeling in Dual-domain for Dynamic MR Imaging** \
*Chuanming Yu, Yu Guan, Ziwen Ke, Dong Liang, Qiegen Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.07599)] \
15 Dec 2022

**DifFace: Blind Face Restoration with Diffused Error Contraction** \
*Zongsheng Yue, Chen Change Loy* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06512)] [[Github](https://github.com/zsyOAOA/DifFace)] \
13 Dec 2022

**ShadowDiffusion: When Degradation Prior Meets Diffusion Model for Shadow Removal** \
*Lanqing Guo, Chong Wang, Wenhan Yang, Siyu Huang, Yufei Wang, Hanspeter Pfister, Bihan Wen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04711)] \
9 Dec 2022


**One Sample Diffusion Model in Projection Domain for Low-Dose CT Imaging** \
*Bin Huang, Liu Zhang, Shiyu Lu, Boyu Lin, Weiwen Wu, Qiegen Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03630)] \
7 Dec 2022

**SDM: Spatial Diffusion Model for Large Hole Image Inpainting** \
*Wenbo Li, Xin Yu, Kun Zhou, Yibing Song, Zhe Lin, Jiaya Jia* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02963)] \
6 Dec 2022

**ADIR: Adaptive Diffusion for Image Reconstruction** \
*Shady Abu-Hussein, Tom Tirer, Raja Giryes* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03221)] [[Project](https://shadyabh.github.io/ADIR/)] \
6 Dec 2022

**Image Deblurring with Domain Generalizable Diffusion Models** \
*Mengwei Ren, Mauricio Delbracio, Hossein Talebi, Guido Gerig, Peyman Milanfar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.01789)] \
4 Dec 2022


**Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model** \
*Yinhuai Wang, Jiwen Yu, Jian Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00490)] [[Github](https://github.com/wyhuai/DDNM)] \
1 Dec 2022


**FREDSR: Fourier Residual Efficient Diffusive GAN for Single Image Super Resolution** \
*Kyoungwan Woo, Achyuta Rajaram* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16678)] \
30 Nov 2022

**CHIMLE: Conditional Hierarchical IMLE for Multimodal Conditional Image Synthesis** \
*Shichong Peng, Alireza Moazeni, Ke Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.14286)] \
25 Nov 2022



**DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction** \
*Jiaming Liu, Rushil Anirudh, Jayaraman J. Thiagarajan, Stewart He, K. Aditya Mohan, Ulugbek S. Kamilov, Hyojin Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12340)] \
22 Nov 2022

**Diffusion Model Based Posterior Sampling for Noisy Linear Inverse Problems** \
*Xiangming Meng, Yoshiyuki Kabashima* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12343)] [[Github](https://github.com/mengxiangming/dmps)] \
20 Nov 2022

**Parallel Diffusion Models of Operator and Image for Blind Inverse Problems** \
*Hyungjin Chung, Jeongsol Kim, Sehui Kim, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10656)] \
19 Nov 2022


**Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models** \
*Hyungjin Chung, Dohoon Ryu, Michael T. McCann, Marc L. Klasky, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10655)] \
19 Nov 2022

**Patch-Based Denoising Diffusion Probabilistic Model for Sparse-View CT Reconstruction** \
*Wenjun Xia, Wenxiang Cong, Ge Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10388)] \
18 Nov 2022


**A Structure-Guided Diffusion Model for Large-Hole Diverse Image Completion** \
*Daichi Horita, Jiaolong Yang, Dong Chen, Yuki Koyama, Kiyoharu Aizawa* \
BMVC 2023. [[Paper](https://arxiv.org/abs/2211.10437)] \
18 Nov 2022


**Conffusion: Confidence Intervals for Diffusion Models** \
*Eliahu Horwitz, Yedid Hoshen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09795)] \
17 Nov 2022

**Superresolution Reconstruction of Single Image for Latent features** \
*Xin Wang, Jing-Ke Yan, Jing-Ye Cai, Jian-Hua Deng, Qin Qin, Qin Wang, Heng Xiao, Yao Cheng, Peng-Fei Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12845)] \
16 Nov 2022



**Learning to Kindle the Starlight** \
*Yu Yuan, Jiaqi Wu, Lindong Wang, Zhongliang Jing, Henry Leung, Shuyuan Zhu, Han Pan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09206)] \
16 Nov 2022


**ShadowDiffusion: Diffusion-based Shadow Removal using Classifier-driven Attention and Structure Preservation** \
*Yeying Jin, Wenhan Yang, Wei Ye, Yuan Yuan, Robby T. Tan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.08089)] \
15 Nov 2022


**DriftRec: Adapting diffusion models to blind image restoration tasks** \
*Simon Welker, Henry N. Chapman, Timo Gerkmann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.06757)] \
12 Nov 2022

**From Denoising Diffusions to Denoising Markov Models** \
*Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, Arnaud Doucet* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.03595)] [[Github](https://github.com/yuyang-shi/generalized-diffusion)] \
7 Nov 2022



**Quantized Compressed Sensing with Score-Based Generative Models** \
*Xiangming Meng, Yoshiyuki Kabashima* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13006)] [[Github](https://github.com/mengxiangming/QCS-SGM)] \
2 Nov 2022




**Intelligent Painter: Picture Composition With Resampling Diffusion Model** \
*Wing-Fung Ku, Wan-Chi Siu, Xi Cheng, H. Anthony Chan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.17106)] \
31 Oct 2022

**Multitask Brain Tumor Inpainting with Diffusion Models: A Methodological Report** \
*Pouria Rouzrokh, Bardia Khosravi, Shahriar Faghani, Mana Moassefi, Sanaz Vahdati, Bradley J. Erickson* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.12113)] [[Github](https://github.com/Mayo-Radiology-Informatics-Lab/MBTI)] \
21 Oct 2022


**DiffGAR: Model-Agnostic Restoration from Generative Artifacts Using Image-to-Image Diffusion Models** \
*Yueqin Yin, Lianghua Huang, Yu Liu, Kaiqi Huang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.08573)] \
16 Oct 2022

**Low-Dose CT Using Denoising Diffusion Probabilistic Model for 20× Speedup** \
*Wenjun Xia, Qing Lyu, Ge Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.15136)] \
29 Sep 2022

**Diffusion Posterior Sampling for General Noisy Inverse Problems** \
*Hyungjin Chung, Jeongsol Kim, Michael T. Mccann, Marc L. Klasky, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14687)] [[Github](https://github.com/DPS2022/diffusion-posterior-sampling)] \
29 Sep 2022

**Face Super-Resolution Using Stochastic Differential Equations** \
*Marcelo dos Santos, Rayson Laroca, Rafael O. Ribeiro, João Neves, Hugo Proença, David Menotti* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.12064)] [[Github](https://github.com/marcelowds/sr-sde)] \
24 Sep 2022


**JPEG Artifact Correction using Denoising Diffusion Restoration Models** \
*Bahjat Kawar, Jiaming Song, Stefano Ermon, Michael Elad* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.11888)] \
23 Sep 2022




**T2V-DDPM: Thermal to Visible Face Translation using Denoising Diffusion Probabilistic Models** \
*Nithin Gopalakrishnan Nair, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.08814)] \
19 Sep 2022


**Delving Globally into Texture and Structure for Image Inpainting** \
*Haipeng Liu, Yang Wang, Meng Wang, Yong Rui* \
ACM 2022. [[Paper](https://arxiv.org/abs/2209.08217)] [[Github](https://github.com/htyjers/DGTS-Inpainting)] \
17 Sep 2022


**PET image denoising based on denoising diffusion probabilistic models** \
*Kuang Gong, Keith A. Johnson, Georges El Fakhri, Quanzheng Li, Tinsu Pan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.06167)] \
13 Sep 2022

**Self-Score: Self-Supervised Learning on Score-Based Models for MRI Reconstruction** \
*Zhuo-Xu Cui, Chentao Cao, Shaonan Liu, Qingyong Zhu, Jing Cheng, Haifeng Wang, Yanjie Zhu, Dong Liang* \
IEEE TMI 2022. [[Paper](https://arxiv.org/abs/2209.00835)] \
2 Sep 2022

**AT-DDPM: Restoring Faces degraded by Atmospheric Turbulence using Denoising Diffusion Probabilistic Models** \
*Nithin Gopalakrishnan Nair, Kangfu Mei, Vishal M Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.11284)] \
24 Aug 2022

**Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise** \
*Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.09392)] [[Github](https://github.com/arpitbansal297/Cold-Diffusion-Models)] \
19 Aug 2022



**High-Frequency Space Diffusion Models for Accelerated MRI** \
*Chentao Cao, Zhuo-Xu Cui, Shaonan Liu, Dong Liang, Yanjie Zhu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.05481)] \
10 Aug 2022


**Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models** \
*Ozan Özdenizci, Robert Legenstein* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.14626)] [[Github](https://github.com/IGITUGraz/WeatherDiffusion)] \
29 Jul 2022





**Non-Uniform Diffusion Models** \
*Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, Christian Etmann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.09786)] \
20 Jul 2022


**Unsupervised Medical Image Translation with Adversarial Diffusion Models** \
*Muzaffer Özbey, Salman UH Dar, Hasan A Bedel, Onat Dalmaz, Şaban Özturk, Alper Güngör, Tolga Çukur* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.08208)] \
17 Jul 2022

**Adaptive Diffusion Priors for Accelerated MRI Reconstruction** \
*Salman UH Dar, Şaban Öztürk, Yilmaz Korkmaz, Gokberk Elmas, Muzaffer Özbey, Alper Güngör, Tolga Çukur* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.05876)] \
12 Jul 2022

**A Novel Unified Conditional Score-based Generative Framework for Multi-modal Medical Image Completion** \
*Xiangxi Meng, Yuning Gu, Yongsheng Pan, Nizhuan Wang, Peng Xue, Mengkang Lu, Xuming He, Yiqiang Zhan, Dinggang Shen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.03430)] \
7 Jul 2022


**SAR Despeckling using a Denoising Diffusion Probabilistic Model** \
*Malsha V. Perera, Nithin Gopalakrishnan Nair, Wele Gedara Chaminda Bandara, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.04514)] \
9 Jun 2022


**Improving Diffusion Models for Inverse Problems using Manifold Constraints** \
*Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00941)] \
2 Jun 2022


**The Swiss Army Knife for Image-to-Image Translation: Multi-Task Diffusion Models** \
*Julia Wolleb, Robin Sandkühler, Florentin Bieder, Philippe C. Cattin* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.02641)] \
6 Apr 2022

**MR Image Denoising and Super-Resolution Using Regularized Reverse Diffusion** \
*Hyungjin Chung, Eun Sun Lee, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.12621)] \
23 Mar 2022


**Towards performant and reliable undersampled MR reconstruction via diffusion model sampling** \
*Cheng Peng, Pengfei Guo, S. Kevin Zhou, Vishal Patel, Rama Chellappa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.04292)] [[Github](https://github.com/cpeng93/diffuserecon)] \
8 Mar 2022

**Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction** \
*Yutong Xie, Quanzheng Li* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2203.03623)] [[Github](https://github.com/Theodore-PKU/MC-DDPM)] \
5 Mar 2022

**MRI Reconstruction via Data Driven Markov Chain with Joint Uncertainty Estimation** \
*Guanxiong Luo, Martin Heide, Martin Uecker* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.01479)] [[Github](https://github.com/mrirecon/spreco)] \
3 Feb 2022

**Unsupervised Denoising of Retinal OCT with Diffusion Probabilistic Model** \
*Dewei Hu, Yuankai K. Tao, Ipek Oguz* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11760)] [[Github](https://github.com/DeweiHu/OCT_DDPM)] \
27 Jan 2022

**Denoising Diffusion Restoration Models** \
*Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song* \
ICLR 2022 Workshop (Oral). [[Paper](https://arxiv.org/abs/2201.11793)] \
27 Jan 2022



**RePaint: Inpainting using Denoising Diffusion Probabilistic Models** \
*Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2201.09865)] [[Github](https://github.com/andreas128/RePaint)] \
24 Jan 2022

**DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents** \
*Kushagra Pandey, Avideep Mukherjee, Piyush Rai, Abhishek Kumar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.00308)] [[Github](https://github.com/kpandey008/DiffuseVAE)] \
2 Jan 2022

**High-Resolution Image Synthesis with Latent Diffusion Models** \
*Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.10752)] [[Github](https://github.com/CompVis/latent-diffusion)] \
20 Dec 2021


**Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction** \
*Hyungjin Chung, Byeongsu Sim, Jong Chul Ye* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.05146)] \
9 Dec 2021

**Deblurring via Stochastic Refinement** \
*Jay Whang, Mauricio Delbracio, Hossein Talebi, Chitwan Saharia, Alexandros G. Dimakis, Peyman Milanfar* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.02475)]  \
5 Dec 2021

**Conditional Image Generation with Score-Based Diffusion Models** \
*Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, Christian Etmann* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.13606)] \
26 Nov 2021

**Solving Inverse Problems in Medical Imaging with Score-Based Generative Models** \
*Yang Song, Liyue Shen, Lei Xing, Stefano Ermon* \
NeurIPS Workshop 2021. [[Paper](https://arxiv.org/abs/2111.08005)] [[Github](https://github.com/yang-song/score_inverse_problems)] \
15 Nov 2021


**S3RP: Self-Supervised Super-Resolution and Prediction for Advection-Diffusion Process** \
*Chulin Wang, Kyongmin Yeo, Xiao Jin, Andres Codas, Levente J. Klein, Bruce Elmegreen* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2111.04639)] \
8 Nov 2021



**Score-based diffusion models for accelerated MRI** \
*Hyungjin Chung, Jong chul Ye* \
MIA 2021. [[Paper](https://arxiv.org/abs/2110.05243)] [[Github](https://github.com/HJ-harry/score-MRI)] \
8 Oct 2021

**Autoregressive Diffusion Models** \
*Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, Tim Salimans* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2110.02037)] \
5 Oct 2021

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021 

**Cascaded Diffusion Models for High Fidelity Image Generation**  \
*Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, Tim Salimans* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.15282)] [[Project](https://cascaded-diffusion.github.io/)] \
30 May 2021

**SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models** \
*Haoying Li, Yifan Yang, Meng Chang, Huajun Feng, Zhihai Xu, Qi Li, Yueting Chen* \
ACM 2022. [[Paper](https://arxiv.org/abs/2104.14951)] \
30 Apr 2021


**Image Super-Resolution via Iterative Refinement**  \
*Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.07636)] [[Project](https://iterative-refinement.github.io/)] [[Github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] \
15 Apr 2021



### Medical Imaging

**Diffusion-based Data Augmentation for Nuclei Image Segmentation** \
*Xinyi Yu, Guanbin Li, Wei Lou, Siqi Liu, Xiang Wan, Yan Chen, Haofeng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.14197)] \
22 Oct 2023

**EMIT-Diff: Enhancing Medical Image Segmentation via Text-Guided Diffusion Model** \
*Zheyuan Zhang, Lanhong Yao, Bin Wang, Debesh Jha, Elif Keles, Alpay Medetalibeyoglu, Ulas Bagci* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12868)] \
19 Oct 2023



**Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation** \
*Haonan Wang, Xiaomeng Li* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.11320)] [[Github](https://github.com/xmed-lab/GenericSSL)] \
17 Oct 2023

**Self-supervised Fetal MRI 3D Reconstruction Based on Radiation Diffusion Generation Model** \
*Junpeng Tan, Xin Zhang, Yao Lv, Xiangmin Xu, Gang Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10209)] \
16 Oct 2023

**JSMoCo: Joint Coil Sensitivity and Motion Correction in Parallel MRI with a Self-Calibrating Score-Based Diffusion Model** \
*Lixuan Chen, Xuanyu Tian, Jiangjie Wu, Ruimin Feng, Guoyan Lao, Yuyao Zhang, Hongjiang Wei* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09625)] \
14 Oct 2023

**Histogram- and Diffusion-Based Medical Out-of-Distribution Detection** \
*Evi M. C. Huijben, Sina Amirrajab, Josien P. W. Pluim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08654)] \
12 Oct 2023

**Echocardiography video synthesis from end diastolic semantic map via diffusion model** \
*Phi Nguyen Van, Duc Tran Minh, Hieu Pham Huy, Long Tran Quoc* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07131)] \
11 Oct 2023


**Diffusion Prior Regularized Iterative Reconstruction for Low-dose CT** \
*Wenjun Xia, Yongyi Shi, Chuang Niu, Wenxiang Cong, Ge Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06949)] \
10 Oct 2023

**Image Compression and Decompression Framework Based on Latent Diffusion Model for Breast Mammography** \
*InChan Hwang, MinJae Woo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05299)] \
8 Oct 2023


**Latent Diffusion Model for Medical Image Standardization and Enhancement** \
*Md Selim, Jie Zhang, Faraneh Fathi, Michael A. Brooks, Ge Wang, Guoqiang Yu, Jin Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05237)] \
8 Oct 2023

**Characterizing the Features of Mitotic Figures Using a Conditional Diffusion Probabilistic Model** \
*Cagla Deniz Bahadir, Benjamin Liechty, David J. Pisapia, Mert R. Sabuncu* \
MICCAI Workshop 2023. [[Paper](https://arxiv.org/abs/2310.03893)] \
5 Oct 2023

**MedSyn: Text-guided Anatomy-aware Synthesis of High-Fidelity 3D CT Images** \
*Yanwu Xu, Li Sun, Wei Peng, Shyam Visweswaran, Kayhan Batmanghelich* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03559)] \
5 Oct 2023


**Blind CT Image Quality Assessment Using DDPM-derived Content and Transformer-based Evaluator** \
*Yongyi Shi, Wenjun Xia, Ge Wang, Xuanqin Mou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03118)] \
4 Oct 2023


**SMRD: SURE-based Robust MRI Reconstruction with Diffusion Models** \
*Batu Ozturkler, Chao Liu, Benjamin Eckart, Morteza Mardani, Jiaming Song, Jan Kautz* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2310.01799)] [[Github](https://github.com/NVlabs/SMRD)] \
3 Oct 2023

**DiffGAN-F2S: Symmetric and Efficient Denoising Diffusion GANs for Structural Connectivity Prediction from Brain fMRI** \
*Qiankun Zuo, Ruiheng Li, Yi Di, Hao Tian, Changhong Jing, Xuhang Chen, Shuqiang Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.16205)] \
28 Sep 2023

**Enhancing Knee Osteoarthritis severity level classification using diffusion augmented images** \
*Paleti Nikhil Chowdary, Gorantla V N S L Vishnu Vardhan, Menta Sai Akshay, Menta Sai Aashish, Vadlapudi Sai Aravind, Garapati Venkata Krishna Rayalu, Aswathy P* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.09328)] \
17 Sep 2023

**Introducing Shape Prior Module in Diffusion Model for Medical Image Segmentation** \
*Zhiqing Zhang, Guojia Fan, Tianyong Liu, Nan Li, Yuyang Liu, Ziyu Liu, Canwei Dong, Shoujun Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05929)] \
12 Sep 2023

**Treatment-aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction** \
*Qinghui Liu, Elies Fuster-Garcia, Ivar Thokle Hovden, Donatas Sederevicius, Karoline Skogen, Bradley J MacIntosh, Edvard Grødem, Till Schellhorn, Petter Brandal, Atle Bjørnerud, Kyrre Eeg Emblem* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05406)] \
11 Sep 2023

**Efficient Bayesian Computational Imaging with a Surrogate Score-Based Prior** \
*Berthy T. Feng, Katherine L. Bouman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01949)] \
5 Sep 2023

**Segmentation of 3D pore space from CT images using curvilinear skeleton: application to numerical simulation of microbial decomposition** \
*Olivier Monga, Zakaria Belghali, Mouad Klai, Lucie Druoton, Dominique Michelucci, Valerie Pot* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01611)] \
4 Sep 2023



**GenSelfDiff-HIS: Generative Self-Supervision Using Diffusion for Histopathological Image Segmentation** \
*Vishnuvardhan Purma, Suhas Srinath, Seshan Srirangarajan, Aanchal Kakkar, Prathosh A. P* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01487)] [[Github](https://github.com/PurmaVishnuVardhanReddy/GenSelfDiff-HIS)] \
4 Sep 2023

**Correlated and Multi-frequency Diffusion Modeling for Highly Under-sampled MRI Reconstruction** \
*Yu Guan, Chuanming Yu, Shiyu Lu, Zhuoxu Cui, Dong Liang, Qiegen Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00853)] [[Github](https://github.com/yqx7150/CM-DM)] \
2 Sep 2023

**Diffusion Modeling with Domain-conditioned Prior Guidance for Accelerated MRI and qMRI Reconstruction** \
*Wanyu Bian, Albert Jang, Fang Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00783)] \
2 Sep 2023

**PathLDM: Text conditioned Latent Diffusion Model for Histopathology** \
*Srikar Yellapragada, Alexandros Graikos, Prateek Prasanna, Tahsin Kurc, Joel Saltz, Dimitris Samaras* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00748)] \
1 Sep 2023


**Unsupervised CT Metal Artifact Reduction by Plugging Diffusion Priors in Dual Domains** \
*Xuan Liu, Yaoqin Xie, Songhui Diao, Shan Tan, Xiaokun Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16742)] \
31 Aug 2023

**A Recycling Training Strategy for Medical Image Segmentation with Diffusion Denoising Models** \
*Yunguan Fu, Yiwen Li, Shaheer U Saeed, Matthew J Clarkson, Yipeng Hu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16355)] [[Github](https://github.com/mathpluscode/ImgX-DiffSeg)] \
30 Aug 2023

**Physics-Informed DeepMRI: Bridging the Gap from Heat Diffusion to k-Space Interpolation** \
*Zhuo-Xu Cui, Congcong Liu, Xiaohong Fan, Chentao Cao, Jing Cheng, Qingyong Zhu, Yuanyuan Liu, Sen Jia, Yihang Zhou, Haifeng Wang, Yanjie Zhu, Jianping Zhang, Qiegen Liu, Dong Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15918)] \
30 Aug 2023


**Stage-by-stage Wavelet Optimization Refinement Diffusion Model for Sparse-View CT Reconstruction** \
*Kai Xu, Shiyu Lu, Bin Huang, Weiwen Wu, Qiegen Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15942)] \
30 Aug 2023

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16150)] \
30 Aug 2023



**Data-iterative Optimization Score Model for Stable Ultra-Sparse-View CT Reconstruction** \
*Weiwen Wu, Yanyang Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.14437)] \
28 Aug 2023



**Full-dose PET Synthesis from Low-dose PET Using High-efficiency Diffusion Denoising Probabilistic Model** \
*Shaoyan Pan, Elham Abouei, Junbo Peng, Joshua Qian, Jacob F Wynne, Tonghe Wang, Chih-Wei Chang, Justin Roper, Jonathon A Nye, Hui Mao, Xiaofeng Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13072)] \
24 Aug 2023

**Augmenting medical image classifiers with synthetic data from latent diffusion models** \
*Luke W. Sagers, James A. Diao, Luke Melas-Kyriazi, Matthew Groh, Pranav Rajpurkar, Adewole S. Adamson, Veronica Rotemberg, Roxana Daneshjou, Arjun K. Manrai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.12453)] \
23 Aug 2023

**InverseSR: 3D Brain MRI Super-Resolution Using a Latent Diffusion Model** \
*Jueqi Wang, Jacob Levman, Walter Hugo Lopez Pinaya, Petru-Daniel Tudosiu, M. Jorge Cardoso, Razvan Marinescu* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2308.12465)] [[Github](https://github.com/BioMedAI-UCSC/InverseSR)] \
23 Aug 2023

**Texture Generation on 3D Meshes with Point-UV Diffusion** \
*Xin Yu, Peng Dai, Wenbo Li, Lan Ma, Zhengzhe Liu, Xiaojuan Qi* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.10490)] \
21 Aug 2023

**Contrastive Diffusion Model with Auxiliary Guidance for Coarse-to-Fine PET Reconstruction** \
*Zeyu Han, Yuhan Wang, Luping Zhou, Peng Wang, Binyu Yan, Jiliu Zhou, Yan Wang, Dinggang Shen* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2308.10157)] [[Github](https://github.com/Show-han/PET-Reconstruction)] \
20 Aug 2023

**Denoising diffusion-based MR to CT image translation enables whole spine vertebral segmentation in 2D and 3D without manual annotations** \
*Robert Graf, Joachim Schmitt, Sarah Schlaeger, Hendrik Kristian Möller, Vasiliki Sideri-Lampretsa, Anjany Sekuboyina, Sandro Manuel Krieg, Benedikt Wiestler, Bjoern Menze, Daniel Rueckert, Jan Stefan Kirschke* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09345)] \
18 Aug 2023

**DMCVR: Morphology-Guided Diffusion Model for 3D Cardiac Volume Reconstruction** \
*Xiaoxiao He, Chaowei Tan, Ligong Han, Bo Liu, Leon Axel, Kang Li, Dimitris N. Metaxas* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2308.09223)] [[Github](https://github.com/hexiaoxiao-cs/DMCVR)] \
18 Aug 2023

**Denoising Diffusion Probabilistic Model for Retinal Image Generation and Segmentation** \
*Alnur Alimanov, Md Baharul Islam* \
ICCP 2023. [[Paper](https://arxiv.org/abs/2308.08339)] \
16 Aug 2023

**Shape-guided Conditional Latent Diffusion Models for Synthesising Brain Vasculature** \
*Yash Deo, Haoran Dou, Nishant Ravikumar, Alejandro F. Frangi, Toni Lassila* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06781)] \
13 Aug 2023

**Masked Diffusion as Self-supervised Representation Learner** \
*Zixuan Pan, Jianxu Chen, Yiyu Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.05695)] \
10 Aug 2023

**Synthetic Augmentation with Large-scale Unconditional Pre-training** \
*Jiarong Ye, Haomiao Ni, Peng Jin, Sharon X. Huang, Yuan Xue* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2308.04020)] [[Github](https://github.com/karenyyy/HistoDiffAug)] \
8 Aug 2023

**Energy-Guided Diffusion Model for CBCT-to-CT Synthesis** \
*Linjie Fu, Xia Li, Xiuding Cai, Dong Miao, Yu Yao, Yali Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.03354)] \
7 Aug 2023


**DermoSegDiff: A Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation** \
*Afshin Bozorgpour, Yousef Sadegheih, Amirhossein Kazerouni, Reza Azad, Dorit Merhof* \
MICCAI Workshop 2023. [[Paper](https://arxiv.org/abs/2308.02959)] [[Github](https://github.com/mindflow-institue/dermosegdiff)] \
5 Aug 2023

**Synthesising Rare Cataract Surgery Samples with Guided Diffusion Models** \
*Yannik Frisch, Moritz Fuchs, Antoine Sanner, Felix Anton Ucar, Marius Frenzel, Joana Wasielica-Poslednik, Adrian Gericke, Felix Mathias Wagner, Thomas Dratsch, Anirban Mukhopadhyay* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02587)] \
3 Aug 2023

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02062)] \
3 Aug 2023

**Reference-Free Isotropic 3D EM Reconstruction using Diffusion Models** \
*Kyungryun Lee, Won-Ki Jeong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01594)] \
3 Aug 2023

**A vision transformer-based framework for knowledge transfer from multi-modal to mono-modal lymphoma subtyping models** \
*Bilel Guetarni, Feryal Windal, Halim Benhabiles, Marianne Petit, Romain Dubois, Emmanuelle Leteurtre, Dominique Collard* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01328)] \
2 Aug 2023

**Learning Fourier-Constrained Diffusion Bridges for MRI Reconstruction** \
*Muhammad U. Mirza, Onat Dalmaz, Hasan A. Bedel, Gokberk Elmas, Yilmaz Korkmaz, Alper Gungor, Salman UH Dar, Tolga Çukur* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01096)] \
2 Aug 2023


**C-DARL: Contrastive diffusion adversarial representation learning for label-free blood vessel segmentation** \
*Boah Kim, Yujin Oh, Bradford J. Wood, Ronald M. Summers, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.00193)] \
31 Jul 2023

**Ultrasound Image Reconstruction with Denoising Diffusion Restoration Models** \
*Yuxin Zhang, Clément Huneau, Jérôme Idier, Diana Mateus* \
MICCAI Workshop 2023. [[Paper](https://arxiv.org/abs/2307.15990)] [[Github](https://github.com/Yuxin-Zhang-Jasmine/DRUS-v1)] \
29 Jul 2023


**Pre-Training with Diffusion models for Dental Radiography segmentation** \
*Jérémy Rousseau, Christian Alaka, Emma Covili, Hippolyte Mayard, Laura Misrachi, Willy Au* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.14066)] \
26 Jul 2023


**Iterative Reconstruction Based on Latent Diffusion Model for Sparse Data Reconstruction** \
*Linchao He, Hongyu Yan, Mengting Luo, Kunming Luo, Wang Wang, Wenchao Du, Hu Chen, Hongyu Yang, Yi Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.12070)] \
22 Jul 2023

**FSDiffReg: Feature-wise and Score-wise Diffusion-guided Unsupervised Deformable Image Registration for Cardiac Images** \
*Yi Qin, Xiaomeng Li* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2307.12035)] [[Github](https://github.com/xmed-lab/FSDiffReg)] \
22 Jul 2023

**FEDD -- Fair, Efficient, and Diverse Diffusion-based Lesion Segmentation and Malignancy Classification** \
*Héctor Carrión, Narges Norouzi* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2307.11654)] [[Github](https://github.com/hectorcarrion/fedd)] \
21 Jul 2023



**PartDiff: Image Super-resolution with Partial Diffusion Models** \
*Kai Zhao, Alex Ling Yu Hung, Kaifeng Pang, Haoxin Zheng, Kyunghyun Sung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.11926)] \
21 Jul 2023

**Make-A-Volume: Leveraging Latent Diffusion Models for Cross-Modality 3D Brain MRI Synthesis** \
*Lingting Zhu, Zeyue Xue, Zhenchao Jin, Xian Liu, Jingzhen He, Ziwei Liu, Lequan Yu* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2307.10094)] \
19 Jul 2023

**DiffDP: Radiotherapy Dose Prediction via a Diffusion Model** \
*Zhenghao Feng, Lu Wen, Peng Wang, Binyu Yan, Xi Wu, Jiliu Zhou, Yan Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.09794)] \
19 Jul 2023

**DreaMR: Diffusion-driven Counterfactual Explanation for Functional MRI** \
*Hasan Atakan Bedel, Tolga Çukur* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.09547)] \
18 Jul 2023

**TractCloud: Registration-free tractography parcellation with a novel local-global streamline point cloud representation** \
*Tengfei Xue, Yuqian Chen, Chaoyi Zhang, Alexandra J. Golby, Nikos Makris, Yogesh Rathi, Weidong Cai, Fan Zhang, Lauren J. O'Donnell* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.09000)] [[Project](https://tractcloud.github.io/)] [[Github](https://github.com/SlicerDMRI/TractCloud)] \
18 Jul 2023


**Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency** \
*Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, Liyue Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08123)] \
16 Jul 2023

**Fast Adaptation with Bradley-Terry Preference Models in Text-To-Image Classification and Generation** \
*Victor Gallego* \
EYSM 2023. [[Paper](https://arxiv.org/abs/2308.07929)] \
15 Jul 2023

**Improving Nonalcoholic Fatty Liver Disease Classification Performance With Latent Diffusion Models** \
*Romain Hardy, Cornelia Ilin, Joe Klepich, Ryan Mitchell, Steve Hall, Jericho Villareal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.06507)] \
13 Jul 2023



**DDGM: Solving inverse problems by Diffusive Denoising of Gradient-based Minimization** \
*Kyle Luther, H. Sebastian Seung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04946)] \
11 Jul 2023

**LLCaps: Learning to Illuminate Low-Light Capsule Endoscopy with Curved Wavelet Attention and Reverse Diffusion** \
*Long Bai, Tong Chen, Yanan Wu, An Wang, Mobarakol Islam, Hongliang Ren* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.02452)] [[Github](https://github.com/longbai1006/LLCaps)] \
5 Jul 2023

**Synchronous Image-Label Diffusion Probability Model with Application to Stroke Lesion Segmentation on Non-contrast CT** \
*Jianhai Zhang, Tonghua Wan, Ethan MacDonald, Bijoy Menon, Aravind Ganesh, Qiu Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.01740)] \
4 Jul 2023

**Investigating Data Memorization in 3D Latent Diffusion Models for Medical Image Synthesis** \
*Salman Ul Hassan Dar, Arman Ghanaat, Jannik Kahmann, Isabelle Ayx, Theano Papavassiliu, Stefan O. Schoenberg, Sandy Engelhardt* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.01148)] \
3 Jul 2023

**Content-Preserving Diffusion Model for Unsupervised AS-OCT image Despeckling** \
*Li Sanqian, Higashita Risa, Fu Huazhu, Li Heng, Niu Jingxuan, Liu Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17717)] \
30 Jun 2023

**Self-Supervised MRI Reconstruction with Unrolled Diffusion Models** \
*Yilmaz Korkmaz, Tolga Cukur, Vishal Patel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.16654)] \
29 Jun 2023

**DoseDiff: Distance-aware Diffusion Model for Dose Prediction in Radiotherapy** \
*Yiwen Zhang, Chuanpu Li, Liming Zhong, Zeli Chen, Wei Yang, Xuetao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.16324)] \
28 Jun 2023

**DiffMix: Diffusion Model-based Data Synthesis for Nuclei Segmentation and Classification in Imbalanced Pathology Image Datasets** \
*Hyun-Jic Oh, Won-Ki Jeong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.14132)] \
25 Jun 2023


**DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology** \
*Marco Aversa, Gabriel Nobis, Miriam Hägele, Kai Standvoss, Mihaela Chirica, Roderick Murray-Smith, Ahmed Alaa, Lukas Ruff, Daniela Ivanova, Wojciech Samek, Frederick Klauschen, Bruno Sanguinetti, Luis Oala* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.13384)] \
23 Jun 2023


**DiffuseIR:Diffusion Models For Isotropic Reconstruction of 3D Microscopic Images** \
*Mingjie Pan, Yulu Gan, Fangxu Zhou, Jiaming Liu, Aimin Wang, Shanghang Zhang, Dawei Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12109)] \
21 Jun 2023

**TauPETGen: Text-Conditional Tau PET Image Synthesis Based on Latent Diffusion Models** \
*Se-In Jang, Cristina Lois, Emma Thibault, J. Alex Becker, Yafei Dong, Marc D. Normandin, Julie C. Price, Keith A. Johnson, Georges El Fakhri, Kuang Gong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11984)] \
21 Jun 2023

**SANO: Score-Based Diffusion Model for Anomaly Localization in Dermatology** \
*Alvaro Gonzalez-Jimenez, Simone Lionetti, Marc Pouly, Alexander A. Navarini* \
CVPR Workshop 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/html/Gonzalez-Jimenez_SANO_Score-Based_Diffusion_Model_for_Anomaly_Localization_in_Dermatology_CVPRW_2023_paper.html)] \
18 Jun 2023


**Aligning Synthetic Medical Images with Clinical Knowledge using Human Feedback** \
*Shenghuan Sun, Gregory M. Goldgof, Atul Butte, Ahmed M. Alaa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12438)] \
16 Jun 2023

**Annotator Consensus Prediction for Medical Image Segmentation with Diffusion Models** \
*Tomer Amit, Shmuel Shichrur, Tal Shaharabany, Lior Wolf* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09004)] \
15 Jun 2023

**Deep Ultrasound Denoising Using Diffusion Probabilistic Models** \
*Hojat Asgariandehkordi, Sobhan Goudarzi, Adrian Basarab, Hassan Rivaz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07440)] \
12 Jun 2023

**Conditional Diffusion Models for Weakly Supervised Medical Image Segmentation** \
*Xinrong Hu, Yu-Jen Chen, Tsung-Yi Ho, Yiyu Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03878)] \
6 Jun 2023

**Interpretable Alzheimer's Disease Classification Via a Contrastive Diffusion Autoencoder** \
*Ayodeji Ijishakin, Ahmed Abdulaal, Adamos Hadjivasiliou, Sophie Martin, James Cole* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03022)] \
5 Jun 2023

**Optimizing Sampling Patterns for Compressed Sensing MRI with Diffusion Generative Models** \
*Sriram Ravula, Brett Levac, Ajil Jalal, Jonathan I. Tamir, Alexandros G. Dimakis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03284)] \
5 Jun 2023

**Brain tumor segmentation using synthetic MR images -- A comparison of GANs and diffusion models** \
*Muhammad Usman Akbar, Måns Larsson, Anders Eklund* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02986)] \
5 Jun 2023


**Unsupervised Anomaly Detection in Medical Images Using Masked Diffusion Model** \
*Hasan Iqbal, Umar Khalid, Jing Hua, Chen Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19867)] \
31 May 2023

**Mask, Stitch, and Re-Sample: Enhancing Robustness and Generalizability in Anomaly Detection through Automatic Diffusion Models** \
*Cosmin I. Bercea, Michael Neumayr, Daniel Rueckert, Julia A. Schnabel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19643)] \
31 May 2023

**Synthetic CT Generation from MRI using 3D Transformer-based Denoising Diffusion Model** \
*Shaoyan Pan, Elham Abouei, Jacob Wynne, Tonghe Wang, Richard L. J. Qiu, Yuheng Li, Chih-Wei Chang, Junbo Peng, Justin Roper, Pretesh Patel, David S. Yu, Hui Mao, Xiaofeng Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19467)] \
31 May 2023


**Conditional Diffusion Models for Semantic 3D Medical Image Synthesis** \
*Zolnamar Dorjsembe, Hsing-Kuo Pao, Sodtavilan Odonchimed, Furen Xiao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18453)] \
29 May 2023


**GenerateCT: Text-Guided 3D Chest CT Generation** \
*Ibrahim Ethem Hamamci, Sezgin Er, Enis Simsar, Alperen Tezcan, Ayse Gulnihan Simsek, Furkan Almas, Sevval Nil Esirgun, Hadrien Reynaud, Sarthak Pati, Christian Bluethgen, Bjoern Menze* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16037)] [[Github](https://github.com/ibrahimethemhamamci/GenerateCT)] \
25 May 2023

**A Diffusion Probabilistic Prior for Low-Dose CT Image Denoising** \
*Xuan Liu, Yaoqin Xie, Songhui Diao, Shan Tan, Xiaokun Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15887)] \
25 May 2023

**Multi-Level Global Context Cross Consistency Model for Semi-Supervised Ultrasound Image Segmentation with Diffusion Model** \
*Fenghe Tang, Jianrui Ding, Lingtao Wang, Min Xian, Chunping Ning* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09447)] [[Github](https://github.com/FengheTan9/Multi-Level-Global-Context-Cross-Consistency)] \
16 May 2023

**Beware of diffusion models for synthesizing medical images -- A comparison with GANs in terms of memorizing brain tumor images** \
*Muhammad Usman Akbar, Wuhao Wang, Anders Eklund* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.07644)] \
12 May 2023

**Generation of Structurally Realistic Retinal Fundus Images with Diffusion Models** \
*Sojung Go, Younghoon Ji, Sang Jun Park, Soochahn Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06813)] \
11 May 2023

**Echo from noise: synthetic ultrasound image generation using diffusion models for real image segmentation** \
*David Stojanovski, Uxio Hermida, Pablo Lamata, Arian Beqiri, Alberto Gomez* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05424)] \
9 May 2023

**Synthesizing PET images from High-field and Ultra-high-field MR images Using Joint Diffusion Attention Model** \
*Taofeng Xie, Chentao Cao, Zhuoxu Cui, Yu Guo, Caiying Wu, Xuemei Wang, Qingneng Li, Zhanli Hu, Tao Sun, Ziru Sang, Yihang Zhou, Yanjie Zhu, Dong Liang, Qiyu Jin, Guoqing Chen, Haifeng Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03901)] \
6 May 2023

**Solving Inverse Problems with Score-Based Generative Priors learned from Noisy Data** \
*Asad Aali, Marius Arvinte, Sidharth Kumar, Jonathan I. Tamir* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01166)] \
2 May 2023

**Self-similarity-based super-resolution of photoacoustic angiography from hand-drawn doodles** \
*Yuanzheng Ma, Wangting Zhou, Rui Ma, Sihua Yang, Yansong Tang, Xun Guan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01165)] \
2 May 2023


**High-Fidelity Image Synthesis from Pulmonary Nodule Lesion Maps using Semantic Diffusion Model** \
*Xuan Zhao, Benjamin Hou* \
MIDL 2023. [[Paper](https://arxiv.org/abs/2305.01138)] \
2 May 2023

**Unsupervised Discovery of 3D Hierarchical Structure with Generative Diffusion Features** \
*Nurislam Tursynbek, Marc Niethammer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.00067)] \
28 Apr 2023

**Cycle-guided Denoising Diffusion Probability Model for 3D Cross-modality MRI Synthesis** \
*Shaoyan Pan, Chih-Wei Chang, Junbo Peng, Jiahan Zhang, Richard L.J. Qiu, Tonghe Wang, Justin Roper, Tian Liu, Hui Mao, Xiaofeng Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.00042)] \
28 Apr 2023

**DiffuseExpand: Expanding dataset for 2D medical image segmentation using diffusion models** \
*Shitong Shao, Xiaohan Yuan, Zhen Huang, Ziming Qiu, Shuai Wang, Kevin Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13416)] [[Github](https://anonymous.4open.science/r/DiffuseExpand/README.md)] \
26 Apr 2023


**Realistic Data Enrichment for Robust Image Segmentation in Histopathology** \
*Sarah Cechnicka, James Ball, Callum Arthurs, Candice Roufosse, Bernhard Kainz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.09534)] \
19 Apr 2023

**Denoising Diffusion Medical Models** \
*Pham Ngoc Huy, Tran Minh Quan* \
IEEE ISBI 2023. [[Paper](https://arxiv.org/abs/2304.09383)] \
19 Apr 2023

**A Multi-Institutional Open-Source Benchmark Dataset for Breast Cancer Clinical Decision Support using Synthetic Correlated Diffusion Imaging Data** \
*Chi-en Amy Tai, Hayden Gunraj, Alexander Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05623)] \
12 Apr 2023

**Cancer-Net BCa-S: Breast Cancer Grade Prediction using Volumetric Deep Radiomic Features from Synthetic Correlated Diffusion Imaging** \
*Chi-en Amy Tai, Hayden Gunraj, Alexander Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05899)] \
12 Apr 2023

**SPIRiT-Diffusion: Self-Consistency Driven Diffusion Model for Accelerated MRI** \
*Zhuo-Xu Cui, Chentao Cao, Jing Cheng, Sen Jia, Hairong Zheng, Dong Liang, Yanjie Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05060)] \
11 Apr 2023
 
**Mask-conditioned latent diffusion for generating gastrointestinal polyp images** \
*Roman Macháček, Leila Mozaffari, Zahra Sepasdar, Sravanthi Parasa, Pål Halvorsen, Michael A. Riegler, Vajira Thambawita* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05233)] \
11 Apr 2023



**BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation** \
*Tao Chen, Chenhui Wang, Hongming Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04429)] \
10 Apr 2023


**Ambiguous Medical Image Segmentation using Diffusion Models** \
*Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, Vishal M Patel* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.04745)] [[Github](https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models)] \
10 Apr 2023

**MedGen3D: A Deep Generative Framework for Paired 3D Image and Mask Generation** \
*Kun Han, Yifeng Xiong, Chenyu You, Pooya Khosravi, Shanlin Sun, Xiangyi Yan, James Duncan, Xiaohui Xie* \
arxiv 2023. [[Paper](https://arxiv.org/abs/2304.04106)] [[Project](https://krishan999.github.io/MedGen3D/)] \
8 Apr 2023

**Towards Realistic Ultrasound Fetal Brain Imaging Synthesis** \
*Michelle Iskandar, Harvey Mannering, Zhanxiang Sun, Jacqueline Matthew, Hamideh Kerdegari, Laura Peralta, Miguel Xochicale* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03941)] [[Gitub](https://github.com/budai4medtech/midl2023)] \
8 Apr 2023


**Zero-shot CT Field-of-view Completion with Unconditional Generative Diffusion Prior** \
*Kaiwen Xu, Aravind R. Krishnan, Thomas Z. Li, Yuankai Huo, Kim L. Sandler, Fabien Maldonado, Bennett A. Landman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03760)] \
7 Apr 2023


**Zero-shot Medical Image Translation via Frequency-Guided Diffusion Models** \
*Yunxiang Li, Hua-Chieh Shao, Xiao Liang, Liyuan Chen, Ruiqi Li, Steve Jiang, Jing Wang, You Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02742)] \
5 Apr 2023




**CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization** \
*Qi Gao, Zilong Li, Junping Zhang, Yi Zhang, Hongming Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01814)] \
4 Apr 2023

**ViT-DAE: Transformer-driven Diffusion Autoencoder for Histopathology Image Analysis** \
*Xuan Xu, Saarthak Kapse, Rajarsi Gupta, Prateek Prasanna* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2304.01053)] \
3 Apr 2023


**Pay Attention: Accuracy Versus Interpretability Trade-off in Fine-tuned Diffusion Models** \
*Mischa Dombrowski, Hadrien Reynaud, Johanna P. Müller, Matthew Baugh, Bernhard Kainz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17908)] \
31 Mar 2023

**DDMM-Synth: A Denoising Diffusion Model for Cross-modal Medical Image Synthesis with Sparse-view Measurement Embedding** \
*Xiaoyue Li, Kai Shang, Gaoang Wang, Mark D. Butala* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15770)] \
28 Mar 2023

**Diffusion Models for Memory-efficient Processing of 3D Medical Images** \
*Florentin Bieder, Julia Wolleb, Alicia Durrer, Robin Sandkühler, Philippe C. Cattin* \
MIDL 2023. [[Paper](https://arxiv.org/abs/2303.15288)] \
27 Mar 2023

**Multi-task Learning of Histology and Molecular Markers for Classifying Diffuse Glioma** \
*Xiaofei Wang, Stephen Price, Chao Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14845)] \
26 Mar 2023

**CoLa-Diff: Conditional Latent Diffusion Model for Multi-Modal MRI Synthesis** \
*Lan Jiang, Ye Mao, Xi Chen, Xiangfeng Wang, Chao Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14081)] \
24 Mar 2023

**DisC-Diff: Disentangled Conditional Diffusion Model for Multi-Contrast MRI Super-Resolution** \
*Ye Mao, Lan Jiang, Xi Chen, Chao Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13933)] \
23 Mar 2023

**Medical diffusion on a budget: textual inversion for medical image generation** \
*Bram de Wilde, Anindo Saha, Richard P.G. ten Broek, Henkjan Huisman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13430)] \
23 Mar 2023

**Sub-volume-based Denoising Diffusion Probabilistic Model for Cone-beam CT Reconstruction from Incomplete Data** \
*Wenjun Xia, Chuang Niu, Wenxiang Cong, Ge Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12861)] \
22 Mar 2023


**Feature-Conditioned Cascaded Video Diffusion Models for Precise Echocardiogram Synthesis** \
*Hadrien Reynaud, Mengyun Qiao, Mischa Dombrowski, Thomas Day, Reza Razavi, Alberto Gomez, Paul Leeson, Bernhard Kainz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12644)] \
22 Mar 2023




**Distribution Aligned Diffusion and Prototype-guided network for Unsupervised Domain Adaptive Segmentation** \
*Haipeng Zhou, Lei Zhu, Yuyin Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12313)] \
22 Mar 2023

**Semantic Latent Space Regression of Diffusion Autoencoders for Vertebral Fracture Grading** \
*Matthias Keicher, Matan Atad, David Schinz, Alexandra S. Gersing, Sarah C. Foreman, Sophia S. Goller, Juergen Weissinger, Jon Rischewski, Anna-Sophia Dietrich, Benedikt Wiestler, Jan S. Kirschke, Nassir Navab* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12031)] \
21 Mar 2023


**NASDM: Nuclei-Aware Semantic Histopathology Image Generation Using Diffusion Models** \
*Aman Shrivastava, P. Thomas Fletcher* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11477)] \
20 Mar 2023

**Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis** \
*Tobias Weber, Michael Ingrisch, Bernd Bischl, David Rügamer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11224)] \
20 Mar 2023

**DiffMIC: Dual-Guidance Diffusion Network for Medical Image Classification** \
*Yijun Yang, Huazhu Fu, Angelica Aviles-Rivero, Carola-Bibiane Schönlieb, Lei Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10610)] \
19 Mar 2023

**Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation** \
*Zhaohu Xing, Liang Wan, Huazhu Fu, Guang Yang, Lei Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10326)] [[Github](https://github.com/ge-xing/Diff-UNet)] \
18 Mar 2023

**Reversing the Abnormal: Pseudo-Healthy Generative Networks for Anomaly Detection** \
*Cosmin I Bercea, Benedikt Wiestler, Daniel Rueckert, Julia A Schnabel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08452)] \
15 Mar 2023



**Improving 3D Imaging with Pre-Trained Perpendicular 2D Diffusion Models** \
*Suhyeon Lee, Hyungjin Chung, Minyoung Park, Jonghyuk Park, Wi-Sun Ryu, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08440)] \
15 Mar 2023

**Class-Guided Image-to-Image Diffusion: Cell Painting from Brightfield Images with Class Labels** \
*Jan Oscar Cross-Zamirski, Praveen Anand, Guy Williams, Elizabeth Mouchet, Yinhai Wang, Carola-Bibiane Schönlieb* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08863)] [[Github](https://github.com/crosszamirski/guided-I2I)] \
15 Mar 2023


**Stochastic Segmentation with Conditional Categorical Diffusion Models** \
*Lukas Zbinden, Lars Doorenbos, Theodoros Pissas, Raphael Sznitman, Pablo Márquez-Neila* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2303.08888)] [[Github](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)] \
15 Mar 2023

**Diffusion Models for Contrast Harmonization of Magnetic Resonance Images** \
*Alicia Durrer, Julia Wolleb, Florentin Bieder, Tim Sinnecker, Matthias Weigel, Robin Sandkühler, Cristina Granziera, Özgür Yaldizli, Philippe C. Cattin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08189)] \
14 Mar 2023


**Efficiently Training Vision Transformers on Structural MRI Scans for Alzheimer's Disease Detection** \
*Nikhil J. Dhinagar, Sophia I. Thomopoulos, Emily Laltoo, Paul M. Thompson* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08216)] \
14 Mar 2023


**Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays** \
*Ibrahim Ethem Hamamci, Sezgin Er, Enis Simsar, Anjany Sekuboyina, Mustafa Gundogar, Bernd Stadlinger, Albert Mehl, Bjoern Menze* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06500)] \
11 Mar 2023

**AugDiff: Diffusion based Feature Augmentation for Multiple Instance Learning in Whole Slide Image** \
*Zhuchen Shao, Liuxi Dai, Yifeng Wang, Haoqian Wang, Yongbing Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06371)] \
11 Mar 2023

**Brain Diffuser: An End-to-End Brain Image to Brain Network Pipeline** \
*Xuhang Chen, Baiying Lei, Chi-Man Pun, Shuqiang Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06410)] \
11 Mar 2023

**Fast Diffusion Sampler for Inverse Problems by Geometric Decomposition** \
*Hyungjin Chung, Suhyeon Lee, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05754)] \
10 Mar 2023

**Generalized Diffusion MRI Denoising and Super-Resolution using Swin Transformers** \
*Amir Sadikov, Jamie Wren-Jarvis, Xinlei Pan, Lanya T. Cai, Pratik Mukherjee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05686)] \
10 Mar 2023

**Importance of Aligning Training Strategy with Evaluation for Diffusion Models in 3D Multiclass Segmentation** \
*Yunguan Fu, Yiwen Li, Shaheer U. Saeed, Matthew J. Clarkson, Yipeng Hu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06040)] [[Github](https://github.com/mathpluscode/ImgX-DiffSeg)] \
10 Mar 2023

**Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI** \
*Finn Behrendt, Debayan Bhattacharya, Julia Krüger, Roland Opfer, Alexander Schlaefer* \
MIDL 2023. [[Paper](https://arxiv.org/abs/2303.03758)] \
7 Mar 2023


**Bi-parametric prostate MR image synthesis using pathology and sequence-conditioned stable diffusion** \
*Shaheer U. Saeed, Tom Syer, Wen Yan, Qianye Yang, Mark Emberton, Shonit Punwani, Matthew J. Clarkson, Dean C. Barratt, Yipeng Hu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.02094)] \
3 Mar 2023

**Dissolving Is Amplifying: Towards Fine-Grained Anomaly Detection** \
*Jian Shi, Pengyi Zhang, Ni Zhang, Hakim Ghazzai, Yehia Massoud* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.14696)] \
28 Feb 2023

**DDM2: Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models** \
*Tiange Xiang, Mahmut Yurt, Ali B Syed, Kawin Setsompop, Akshay Chaudhari* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2302.03018)] [[Github](https://github.com/StanfordMIMI/DDM2)] \
6 Feb 2023


**Zero-shot-Learning Cross-Modality Data Translation Through Mutual Information Guided Stochastic Diffusion** \
*Zihao Wang, Yingyu Yang, Maxime Sermesant, Hervé Delingette, Ona Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13743)] \
31 Jan 2023

**Diffusion Denoising for Low-Dose-CT Model** \
*Runyi Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11482)] \
27 Jan 2023

**DiffusionCT: Latent Diffusion Model for CT Image Standardization** \
*Md Selim, Jie Zhang, Michael A. Brooks, Ge Wang, Jin Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.08815)] \
20 Jan 2023

**MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer** \
*Junde Wu, Rao Fu, Huihui Fang, Yu Zhang, Yanwu Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11798)] \
19 Jan 2023

**The role of noise in denoising models for anomaly detection in medical images** \
*Antanas Kascenas, Pedro Sanchez, Patrick Schrempf, Chaoyang Wang, William Clackett, Shadia S. Mikhael, Jeremy P. Voisey, Keith Goatman, Alexander Weir, Nicolas Pugeault, Sotirios A. Tsaftaris, Alison Q. O'Neil* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.08330)] [[Github](https://github.com/AntanasKascenas/DenoisingAE)] \
19 Jan 2023



**Diffusion-based Data Augmentation for Skin Disease Classification: Impact Across Original Medical Datasets to Fully Synthetic Images** \
*Mohamed Akrout, Bálint Gyepesi, Péter Holló, Adrienn Poór, Blága Kincső, Stephen Solis, Katrina Cirone, Jeremy Kawahara, Dekker Slade, Latif Abid, Máté Kovács, István Fazekas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.04802)] \
12 Jan 2023

**Annealed Score-Based Diffusion Model for MR Motion Artifact Reduction** \
*Gyutaek Oh, Jeong Eun Lee, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.03027)] \
8 Jan 2023

**Denoising Diffusion Probabilistic Models for Generation of Realistic Fully-Annotated Microscopy Image Data Sets** \
*Dennis Eschweiler, Johannes Stegmaier* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.10227)] \
2 Jan 2023

**Diffusion Model based Semi-supervised Learning on Brain Hemorrhage Images for Efficient Midline Shift Quantification** \
*Shizhan Gong, Cheng Chen, Yuqi Gong, Nga Yan Chan, Wenao Ma, Calvin Hoi-Kwan Mak, Jill Abrigo, Qi Dou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.00409)] \
1 Jan 2023


**SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image Generation** \
*Jee Seok Yoon, Chenghao Zhang, Heung-Il Suk, Jia Guo, Xiaoxiao Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08228)] \
16 Dec 2022

**Universal Generative Modeling in Dual-domain for Dynamic MR Imaging** \
*Chuanming Yu, Yu Guan, Ziwen Ke, Dong Liang, Qiegen Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.07599)] \
15 Dec 2022

**Generating Realistic 3D Brain MRIs Using a Conditional Diffusion Probabilistic Model** \
*Wei Peng, Ehsan Adeli, Qingyu Zhao, Kilian M. Pohl* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08034)] [[Github](https://github.com/Project-MONAI/GenerativeModels/tree/260-add-cdpm-model)] \
15 Dec 2022


**SPIRiT-Diffusion: SPIRiT-driven Score-Based Generative Modeling for Vessel Wall imaging** \
*Chentao Cao, Zhuo-Xu Cui, Jing Cheng, Sen Jia, Hairong Zheng, Dong Liang, Yanjie Zhu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.11274)] \
14 Dec 2022

**Diffusion Probabilistic Models beat GANs on Medical Images** \
*Gustav Müller-Franzes, Jan Moritz Niehues, Firas Khader, Soroosh Tayebi Arasteh, Christoph Haarburger, Christiane Kuhl, Tianci Wang, Tianyu Han, Sven Nebelung, Jakob Nikolas Kather, Daniel Truhn* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.07501)] \
14 Dec 2022


**One Sample Diffusion Model in Projection Domain for Low-Dose CT Imaging** \
*Bin Huang, Liu Zhang, Shiyu Lu, Boyu Lin, Weiwen Wu, Qiegen Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03630)] \
7 Dec 2022



**Neural Cell Video Synthesis via Optical-Flow Diffusion** \
*Manuel Serna-Aguilera, Khoa Luu, Nathaniel Harris, Min Zou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03250)] \
6 Dec 2022

**Improving dermatology classifiers across populations using images generated by large diffusion models** \
*Luke W. Sagers, James A. Diao, Matthew Groh, Pranav Rajpurkar, Adewole S. Adamson, Arjun K. Manrai* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2211.13352)] \
23 Nov 2022

**RoentGen: Vision-Language Foundation Model for Chest X-ray Generation** \
*Pierre Chambon, Christian Bluethgen, Jean-Benoit Delbrouck, Rogier Van der Sluijs, Małgorzata Połacin, Juan Manuel Zambrano Chaves, Tanishq Mathew Abraham, Shivanshu Purohit, Curtis P. Langlotz, Akshay Chaudhari* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12737)] \
23 Nov 2022

**DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction** \
*Jiaming Liu, Rushil Anirudh, Jayaraman J. Thiagarajan, Stewart He, K. Aditya Mohan, Ulugbek S. Kamilov, Hyojin Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12340)] \
22 Nov 2022



**Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models** \
*Hyungjin Chung, Dohoon Ryu, Michael T. McCann, Marc L. Klasky, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10655)] \
19 Nov 2022

**Patch-Based Denoising Diffusion Probabilistic Model for Sparse-View CT Reconstruction** \
*Wenjun Xia, Wenxiang Cong, Ge Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10388)] \
18 Nov 2022


**Brain PET Synthesis from MRI Using Joint Probability Distribution of Diffusion Model at Ultrahigh Fields** \
*Xie Taofeng, Cao Chentao, Cui Zhuoxu, Li Fanshi, Wei Zidong, Zhu Yanjie, Li Ye, Liang Dong, Jin Qiyu, Chen Guoqing, Wang Haifeng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.08901)] \
16 Nov 2022

**Improved HER2 Tumor Segmentation with Subtype Balancing using Deep Generative Networks** \
*Mathias Öttl, Jana Mönius, Matthias Rübner, Carol I. Geppert, Jingna Qiu, Frauke Wilm, Arndt Hartmann, Matthias W. Beckmann, Peter A. Fasching, Andreas Maier, Ramona Erber, Katharina Breininger* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.06150)] \
11 Nov 2022



**An unobtrusive quality supervision approach for medical image annotation** \
*Sonja Kunzmann, Mathias Öttl, Prathmesh Madhu, Felix Denzinger, Andreas Maier* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.06146)] \
11 Nov 2022


**Medical Diffusion -- Denoising Diffusion Probabilistic Models for 3D Medical Image Generation** \
*Firas Khader, Gustav Mueller-Franzes, Soroosh Tayebi Arasteh, Tianyu Han, Christoph Haarburger, Maximilian Schulze-Hagen, Philipp Schad, Sandy Engelhardt, Bettina Baessler, Sebastian Foersch, Johannes Stegmaier, Christiane Kuhl, Sven Nebelung, Jakob Nikolas Kather, Daniel Truhn* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.03364)] \
7 Nov 2022

**Generation of Anonymous Chest Radiographs Using Latent Diffusion Models for Training Thoracic Abnormality Classification Systems** \
*Kai Packhäuser, Lukas Folle, Florian Thamm, Andreas Maier* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01323)] \
2 Nov 2022

**Spot the fake lungs: Generating Synthetic Medical Images using Neural Diffusion Models** \
*Hazrat Ali, Shafaq Murad, Zubair Shah* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.00902)] [[Project](https://www.kaggle.com/datasets/hazrat/awesomelungs)] \
2 Nov 2022


**MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model** \
*Junde Wu, Huihui Fang, Yu Zhang, Yehui Yang, Yanwu Xu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.00611)] \
1 Nov 2022

**Accelerating Diffusion Models via Pre-segmentation Diffusion Sampling for Medical Image Segmentation** \
*Xutao Guo, Yanwu Yang, Chenfei Ye, Shang Lu, Yang Xiang, Ting Ma* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.17408)] \
27 Oct 2022

**Multitask Brain Tumor Inpainting with Diffusion Models: A Methodological Report** \
*Pouria Rouzrokh, Bardia Khosravi, Shahriar Faghani, Mana Moassefi, Sanaz Vahdati, Bradley J. Erickson* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.12113)] [[Github](https://github.com/Mayo-Radiology-Informatics-Lab/MBTI)] \
21 Oct 2022


**Adapting Pretrained Vision-Language Foundational Models to Medical Imaging Domains** \
*Pierre Chambon, Christian Bluethgen, Curtis P. Langlotz, Akshay Chaudhari* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.04133)] \
9 Oct 2022

**Anatomically constrained CT image translation for heterogeneous blood vessel segmentation** \
*Giammarco La Barbera, Haithem Boussaid, Francesco Maso, Sabine Sarnacki, Laurence Rouet, Pietro Gori, Isabelle Bloch* \
BMVC 2022. [[Paper](https://arxiv.org/abs/2210.01713)] \
4 Oct 2022

**Low-Dose CT Using Denoising Diffusion Probabilistic Model for 20× Speedup** \
*Wenjun Xia, Qing Lyu, Ge Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.15136)] \
29 Sep 2022


**Diffusion Adversarial Representation Learning for Self-supervised Vessel Segmentation** \
*Boah Kim, Yujin Oh, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14566)] \
29 Sep 2022

**Conversion Between CT and MRI Images Using Diffusion and Score-Matching Models** \
*Qing Lyu, Ge Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.12104)] \
24 Sep 2022

**Brain Imaging Generation with Latent Diffusion Models** \
*Walter H. L. Pinaya, Petru-Daniel Tudosiu, Jessica Dafflon, Pedro F da Costa, Virginia Fernandez, Parashkev Nachev, Sebastien Ourselin, M. Jorge Cardoso* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.07162)] \
15 Sep 2022

**PET image denoising based on denoising diffusion probabilistic models** \
*Kuang Gong, Keith A. Johnson, Georges El Fakhri, Quanzheng Li, Tinsu Pan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.06167)] \
13 Sep 2022

**Self-Score: Self-Supervised Learning on Score-Based Models for MRI Reconstruction** \
*Zhuo-Xu Cui, Chentao Cao, Shaonan Liu, Qingyong Zhu, Jing Cheng, Haifeng Wang, Yanjie Zhu, Dong Liang* \
IEEE TMI 2022. [[Paper](https://arxiv.org/abs/2209.00835)] \
2 Sep 2022

**High-Frequency Space Diffusion Models for Accelerated MRI** \
*Chentao Cao, Zhuo-Xu Cui, Shaonan Liu, Dong Liang, Yanjie Zhu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.05481)] \
10 Aug 2022

**What is Healthy? Generative Counterfactual Diffusion for Lesion Localization** \
*Pedro Sanchez, Antanas Kascenas, Xiao Liu, Alison Q. O'Neil, Sotirios A. Tsaftaris* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2207.12268)] [[Github](https://github.com/vios-s/Diff-SCM)] \
25 Jul 2022


**Unsupervised Medical Image Translation with Adversarial Diffusion Models** \
*Muzaffer Özbey, Salman UH Dar, Hasan A Bedel, Onat Dalmaz, Şaban Özturk, Alper Güngör, Tolga Çukur* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.08208)] \
17 Jul 2022

**Adaptive Diffusion Priors for Accelerated MRI Reconstruction** \
*Salman UH Dar, Şaban Öztürk, Yilmaz Korkmaz, Gokberk Elmas, Muzaffer Özbey, Alper Güngör, Tolga Çukur* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.05876)] \
12 Jul 2022

**A Novel Unified Conditional Score-based Generative Framework for Multi-modal Medical Image Completion** \
*Xiangxi Meng, Yuning Gu, Yongsheng Pan, Nizhuan Wang, Peng Xue, Mengkang Lu, Xuming He, Yiqiang Zhan, Dinggang Shen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.03430)] \
7 Jul 2022


**Cross-Modal Transformer GAN: A Brain Structure-Function Deep Fusing Framework for Alzheimer's Disease** \
*Junren Pan, Shuqiang Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.13393)] \
20 Jun 2022

**Diffusion Deformable Model for 4D Temporal Medical Image Generation** \
*Boah Kim, Jong Chul Ye* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2206.13295)] [[Github](https://github.com/torchddm/ddm)] \
27 Jun 2022


**Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models** \
*Walter H. L. Pinaya, Mark S. Graham, Robert Gray, Pedro F Da Costa, Petru-Daniel Tudosiu, Paul Wright, Yee H. Mah, Andrew D. MacKinnon, James T. Teo, Rolf Jager, David Werring, Geraint Rees, Parashkev Nachev, Sebastien Ourselin, M. Jorge Cardos* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2206.03461)] \
7 Jun 2022

**Improving Diffusion Models for Inverse Problems using Manifold Constraints** \
*Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00941)] \
2 Jun 2022

**AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise** \
*Julian Wyatt, Adam Leach, Sebastian M. Schmon, Chris G. Willcocks* \
CVPR Workshop 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf)] [[Github](https://github.com/Julian-Wyatt/AnoDDPM)] \
1 Jun 2022

**The Swiss Army Knife for Image-to-Image Translation: Multi-Task Diffusion Models** \
*Julia Wolleb, Robin Sandkühler, Florentin Bieder, Philippe C. Cattin* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.02641)] \
6 Apr 2022

**MR Image Denoising and Super-Resolution Using Regularized Reverse Diffusion** \
*Hyungjin Chung, Eun Sun Lee, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.12621)] \
23 Mar 2022

**Diffusion Models for Medical Anomaly Detection** \
*Julia Wolleb, Florentin Bieder, Robin Sandkühler, Philippe C. Cattin* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2203.04306)] [[Github](https://github.com/JuliaWolleb/diffusion-anomaly)] \
8 Mar 2022

**Towards performant and reliable undersampled MR reconstruction via diffusion model sampling** \
*Cheng Peng, Pengfei Guo, S. Kevin Zhou, Vishal Patel, Rama Chellappa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.04292)] [[Github](https://github.com/cpeng93/diffuserecon)] \
8 Mar 2022

**Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction** \
*Yutong Xie, Quanzheng Li* \
MICCAI 2022. [[Paper](https://arxiv.org/abs/2203.03623)] [[Github](https://github.com/Theodore-PKU/MC-DDPM)] \
5 Mar 2022

**MRI Reconstruction via Data Driven Markov Chain with Joint Uncertainty Estimation** \
*Guanxiong Luo, Martin Heide, Martin Uecker* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.01479)] [[Github](https://github.com/mrirecon/spreco)] \
3 Feb 2022

**Unsupervised Denoising of Retinal OCT with Diffusion Probabilistic Model** \
*Dewei Hu, Yuankai K. Tao, Ipek Oguz* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11760)] [[Github](https://github.com/DeweiHu/OCT_DDPM)] \
27 Jan 2022

**Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction** \
*Hyungjin Chung, Byeongsu Sim, Jong Chul Ye* \
CVPR 2021. [[Paper](https://arxiv.org/abs/2112.05146)] \
9 Dec 2021

**Solving Inverse Problems in Medical Imaging with Score-Based Generative Models** \
*Yang Song, Liyue Shen, Lei Xing, Stefano Ermon* \
NeurIPS Workshop 2021. [[Paper](https://arxiv.org/abs/2111.08005)] [[Github](https://github.com/yang-song/score_inverse_problems)] \
15 Nov 2021

**Score-based diffusion models for accelerated MRI** \
*Hyungjin Chung, Jong chul Ye* \
MIA 2021. [[Paper](https://arxiv.org/abs/2110.05243)] [[Github](https://github.com/HJ-harry/score-MRI)] \
8 Oct 2021




### Multi-modal Learning


**IterInv: Iterative Inversion for Pixel-Level T2I Models** \
*Chuanming Tang, Kai Wang, Joost van de Weijer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19540)] \
30 Oct 2023


**VideoCrafter1: Open Diffusion Models for High-Quality Video Generation** \
*Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, Ying Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19512)] \
30 Oct 2023

**IMPRESS: Evaluating the Resilience of Imperceptible Perturbations Against Unauthorized Data Usage in Diffusion-Based Generative AI** \
*Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, Jinghui Chen* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.19248)] \
30 Oct 2023


**CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models** \
*Ziyang Yuan, Mingdeng Cao, Xintao Wang, Zhongang Qi, Chun Yuan, Ying Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19784)] \
30 Oct 2023


**Seeing Through the Conversation: Audio-Visual Speech Separation based on Diffusion Model** \
*Suyeon Lee, Chaeyoung Jung, Youngjoon Jang, Jaehun Kim, Joon Son Chung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19581)] \
30 Oct 2023


**Text-to-3D with Classifier Score Distillation** \
*Xin Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Song-Hai Zhang, Xiaojuan Qi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19415)] \
30 Oct 2023

**Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models** \
*Hai Wang, Xiaoyu Xiang, Yuchen Fan, Jing-Hao Xue* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.18840)] \
28 Oct 2023

**SD4Match: Learning to Prompt Stable Diffusion Model for Semantic Matching** \
*Xinghui Li, Jingyi Lu, Kai Han, Victor Prisacariu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17569)] \
26 Oct 2023





**CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling** \
*Seyedmorteza Sadat, Jakob Buhmann, Derek Bradely, Otmar Hilliges, Romann M. Weber* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17347)] \
26 Oct 2023

**Exploring Iterative Refinement with Diffusion Models for Video Grounding** \
*Xiao Liang, Tao Shi, Yaoyuan Liang, Te Tao, Shao-Lun Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17189)] \
26 Oct 2023

**A Picture is Worth a Thousand Words: Principled Recaptioning Improves Image Generation** \
*Eyal Segalis, Dani Valevski, Danny Lumen, Yossi Matias, Yaniv Leviathan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16656)] \
25 Oct 2023



**CommonCanvas: An Open Diffusion Model Trained with Creative-Commons Images** \
*Aaron Gokaslan, A. Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, Volodymyr Kuleshov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16825)] \
25 Oct 2023


**On the Proactive Generation of Unsafe Images From Text-To-Image Models Using Benign Prompts** \
*Yixin Wu, Ning Yu, Michael Backes, Yun Shen, Yang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16613)] \
25 Oct 2023

**Fuse Your Latents: Video Editing with Multi-source Latent Diffusion Models** \
*Tianyi Lu, Xing Zhang, Jiaxi Gu, Hang Xu, Renjing Pei, Songcen Xu, Zuxuan Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16400)] \
25 Oct 2023



**Adapt Anything: Tailor Any Image Classifiers across Domains And Categories Using Text-to-Image Diffusion Models** \
*Weijie Chen, Haoyu Wang, Shicai Yang, Lei Zhang, Wei Wei, Yanning Zhang, Luojun Lin, Di Xie, Yueting Zhuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16573)] \
25 Oct 2023

**Text Guided Video Editing Competition** \
*Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, Rui He, Feng Hu, Junhua Hu, Hai Huang, Hanyu Zhu, Xu Cheng, Jie Tang, Mike Zheng Shou, Kurt Keutzer, Forrest Iandola* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16003)] \
24 Oct 2023

**Language-driven Scene Synthesis using Multi-conditional Diffusion Model** \
*An Vuong, Minh Nhat Vu, Toan Tien Nguyen, Baoru Huang, Dzung Nguyen, Thieu Vo, Anh Nguyen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.15948)] \
24 Oct 2023

**FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling** \
*Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.15169)] [[Project](http://haonanqiu.com/projects/FreeNoise.html)] \
23 Oct 2023

**SyncFusion: Multimodal Onset-synchronized Video-to-Audio Foley Synthesis** \
*Marco Comunità, Riccardo F. Gramaccioni, Emilian Postolache, Emanuele Rodolà, Danilo Comminiello, Joshua D. Reiss* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.15247)] \
23 Oct 2023


**Matryoshka Diffusion Models** \
*Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Josh Susskind, Navdeep Jaitly* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.15111)] \
23 Oct 2023

**Large Language Models can Share Images, Too!** \
*Young-Jun Lee, Jonghwan Hyeon, Ho-Jin Choi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.14804)] \
23 Oct 2023

**Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models** \
*Shawn Shan, Wenxin Ding, Josephine Passananti, Haitao Zheng, Ben Y. Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.13828)] \
20 Oct 2023


**TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models** \
*Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, Kangxue Yin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.13772)] \
20 Oct 2023



**DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics** \
*Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.13268)] [[Project](https://ml.cs.tsinghua.edu.cn/dpmv3/)] \
20 Oct 2023


**Localizing and Editing Knowledge in Text-to-Image Generative Models** \
*Samyadeep Basu, Nanxuan Zhao, Vlad Morariu, Soheil Feizi, Varun Manjunatha* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.13730)] \
20 Oct 2023

**TapMo: Shape-aware Motion Generation of Skeleton-free Characters** \
*Jiaxu Zhang, Shaoli Huang, Zhigang Tu, Xin Chen, Xiaohang Zhan, Gang Yu, Ying Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12678)] \
19 Oct 2023

**CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation** \
*Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, Joyce Chai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.13165)] \
19 Oct 2023


**DreamSpace: Dreaming Your Room Space with Text-Driven Panoramic Texture Propagation** \
*Bangbang Yang, Wenqi Dong, Lin Ma, Wenbo Hu, Xiao Liu, Zhaopeng Cui, Yuewen Ma* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.13119)] \
19 Oct 2023

**EMIT-Diff: Enhancing Medical Image Segmentation via Text-Guided Diffusion Model** \
*Zheyuan Zhang, Lanhong Yao, Bin Wang, Debesh Jha, Elif Keles, Alpay Medetalibeyoglu, Ulas Bagci* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12868)] \
19 Oct 2023

**Diverse Diffusion: Enhancing Image Diversity in Text-to-Image Generation** \
*Mariia Zameshina, Olivier Teytaud, Laurent Najman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12583)] \
19 Oct 2023

**DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors** \
*Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, Ying Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12190)] \
18 Oct 2023

**Progressive3D: Progressively Local Editing for Text-to-3D Content Creation with Complex Semantic Prompts** \
*Xinhua Cheng, Tianyu Yang, Jianan Wang, Yu Li, Lei Zhang, Jian Zhang, Li Yuan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11784)] \
18 Oct 2023


**Language Agents for Detecting Implicit Stereotypes in Text-to-image Models at Scale** \
*Qichao Wang, Tian Bian, Yian Yin, Tingyang Xu, Hong Cheng, Helen M. Meng, Zibin Zheng, Liang Chen, Bingzhe Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11778)] \
18 Oct 2023

**Elucidating The Design Space of Classifier-Guided Diffusion Generation** \
*Jiajun Ma, Tianyang Hu, Wenjia Wang, Jiacheng Sun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11311)] [[Github](https://github.com/alexmaols/elucd)] \
17 Oct 2023

**BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference** \
*Siqi Kou, Lei Gan, Dequan Wang, Chongxuan Li, Zhijie Deng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11142)] \
17 Oct 2023

**GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment** \
*Dhruba Ghosh, Hanna Hajishirzi, Ludwig Schmidt* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11513)] \
17 Oct 2023

**Towards Training-free Open-world Segmentation via Image Prompting Foundation Models** \
*Lv Tang, Peng-Tao Jiang, Hao-Ke Xiao, Bo Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10912)] \
17 Oct 2023

**LAMP: Learn A Motion Pattern for Few-Shot-Based Video Generation** \
*Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, Xiangyu Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10769)] [[Project](https://rq-wu.github.io/projects/LAMP/)] [[Github](https://github.com/RQ-Wu/LAMP)] \
16 Oct 2023

**Scene Graph Conditioning in Latent Diffusion** \
*Frank Fundel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10338)] [[Github](https://github.com/FrankFundel/SGCond)] \
16 Oct 2023


**Ring-A-Bell! How Reliable are Concept Removal Methods for Diffusion Models?** \
*Yu-Lin Tsai, Chia-Yi Hsu, Chulin Xie, Chih-Hsun Lin, Jia-You Chen, Bo Li, Pin-Yu Chen, Chia-Mu Yu, Chun-Ying Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10012)] \
16 Oct 2023

**Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models** \
*Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, Sergey Levine* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10639)] \
16 Oct 2023

**ViPE: Visualise Pretty-much Everything** \
*Hassan Shahmohammadi, Adhiraj Ghosh, Hendrik P. A. Lensch* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10543)] \
16 Oct 2023


**TOSS:High-quality Text-guided Novel View Synthesis from a Single Image** \
*Yukai Shi, Jianan Wang, He Cao, Boshi Tang, Xianbiao Qi, Tianyu Yang, Yukun Huang, Shilong Liu, Lei Zhang, Heung-Yeung Shum* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10644)] \
16 Oct 2023


**LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts** \
*Hanan Gani, Shariq Farooq Bhat, Muzammal Naseer, Salman Khan, Peter Wonka* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10640)] \
16 Oct 2023


**LOVECon: Text-driven Training-Free Long Video Editing with ControlNet** \
*Zhenyi Liao, Zhijie Deng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09711)] \
15 Oct 2023

**PaintHuman: Towards High-fidelity Text-to-3D Human Texturing via Denoised Score Distillation** \
*Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, Wayne Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09458)] \
14 Oct 2023

**Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task** \
*Maya Okawa, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka* \
ICML Workshop 2023. [[Paper](https://arxiv.org/abs/2310.09336)] \
13 Oct 2023

**Hypernymy Understanding Evaluation of Text-to-Image Models via WordNet Hierarchy** \
*Anton Baryshnikov, Max Ryabinin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09247)] \
13 Oct 2023

**Making Multimodal Generation Easier: When Diffusion Models Meet LLMs** \
*Xiangyu Zhao, Bo Liu, Qijiong Liu, Guangyuan Shi, Xiao-Ming Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08949)] \
13 Oct 2023


**R&B: Region and Boundary Aware Zero-shot Grounded Text-to-image Generation** \
*Jiayu Xiao, Liang Li, Henglei Lv, Shuhui Wang, Qingming Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08872)] \
13 Oct 2023

**DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing** \
*Yueming Lyu, Kang Zhao, Bo Peng, Yue Jiang, Yingya Zhang, Jing Dong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08785)] \
12 Oct 2023


**OmniControl: Control Any Joint at Any Time for Human Motion Generation** \
*Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, Huaizu Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08580)] [[Project](https://neu-vi.github.io/omnicontrol/)] \
12 Oct 2023


**HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion** \
*Xian Liu, Jian Ren, Aliaksandr Siarohin, Ivan Skorokhodov, Yanyu Li, Dahua Lin, Xihui Liu, Ziwei Liu, Sergey Tulyakov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08579)] [[Project](https://snap-research.github.io/HyperHuman/)] [[Github](https://github.com/snap-research/HyperHuman)] \
12 Oct 2023

**GaussianDreamer: Fast Generation from Text to 3D Gaussian Splatting with Point Cloud Priors** \
*Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, Xinggang Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08529)] \
12 Oct 2023


**MotionDirector: Motion Customization of Text-to-Video Diffusion Models** \
*Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, Mike Zheng Shou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08465)] \
12 Oct 2023

**Interpretable Diffusion via Information Decomposition** \
*Xianghao Kong, Ollie Liu, Han Li, Dani Yogatama, Greg Ver Steeg* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07972)] \
12 Oct 2023

**DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model** \
*Xiaofan Li, Yifu Zhang, Xiaoqing Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07771)] [[Project](https://drivingdiffusion.github.io/)] [[Github](https://github.com/shalfun/DrivingDiffusion)] \
11 Oct 2023

**ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models** \
*Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, Ying Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07702)] [[Project](https://yingqinghe.github.io/scalecrafter/)] [[Github](https://github.com/YingqingHe/ScaleCrafter)] \
11 Oct 2023

**ConditionVideo: Training-Free Condition-Guided Text-to-Video Generation** \
*Bo Peng, Xinyuan Chen, Yaohui Wang, Chaochao Lu, Yu Qiao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07697)] \
11 Oct 2023


**Mini-DALLE3: Interactive Text to Image by Prompting Large Language Models** \
*Zeqiang Lai, Xizhou Zhu, Jifeng Dai, Yu Qiao, Wenhai Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07653)] \
11 Oct 2023

**Multi-Concept T2I-Zero: Tweaking Only The Text Embeddings and Nothing Else** \
*Hazarapet Tunanyan, Dejia Xu, Shant Navasardyan, Zhangyang Wang, Humphrey Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07419)] \
11 Oct 2023

**Uni-paint: A Unified Framework for Multimodal Image Inpainting with Pretrained Diffusion Model** \
*Shiyuan Yang, Xiaodong Chen, Jing Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07222)] \
11 Oct 2023

**ObjectComposer: Consistent Generation of Multiple Objects Without Fine-tuning** \
*Alec Helbling, Evan Montoya, Duen Horng Chau* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06968)] \
10 Oct 2023


**JointNet: Extending Text-to-Image Diffusion for Dense Distribution Modeling** \
*Jingyang Zhang, Shiwei Li, Yuanxun Lu, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan, Yao Yao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06347)] \
10 Oct 2023

**Improving Compositional Text-to-image Generation with Large Vision-Language Models** \
*Song Wen, Guian Fang, Renrui Zhang, Peng Gao, Hao Dong, Dimitris Metaxas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06311)] \
10 Oct 2023

**Geom-Erasing: Geometry-Driven Removal of Implicit Concept in Diffusion Models** \
*Zhili Liu, Kai Chen, Yifan Zhang, Jianhua Han, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James Kwok* \
arXiv 2023 [[Paper](https://arxiv.org/abs/2310.05873)] \
9 Oct 2023


**FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing** \
*Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, Sen He* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05922)] \
9 Oct 2023

**Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation** \
*Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, Lu Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05737)] [[Github](https://github.com/lucidrains/magvit2-pytorch)] \
9 Oct 2023

**IPDreamer: Appearance-Controllable 3D Object Generation with Image Prompts** \
*Bohan Zeng, Shanglin Li, Yutang Feng, Hong Li, Sicheng Gao, Jiaming Liu, Huaxia Li, Xu Tang, Jianzhuang Liu, Baochang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05375)] \
9 Oct 2023


**Diffusion Models as Masked Audio-Video Learners** \
*Elvis Nunez, Yanzi Jin, Mohammad Rastegari, Sachin Mehta, Maxwell Horton* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03937)] \
5 Oct 2023

**Aligning Text-to-Image Diffusion Models with Reward Backpropagation** \
*Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, Katerina Fragkiadaki* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03739)] \
5 Oct 2023

**Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints** \
*Chuan Fang, Xiaotao Hu, Kunming Luo, Ping Tan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03602)] \
5 Oct 2023


**MedSyn: Text-guided Anatomy-aware Synthesis of High-Fidelity 3D CT Images** \
*Yanwu Xu, Li Sun, Wei Peng, Shyam Visweswaran, Kayhan Batmanghelich* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03559)] \
5 Oct 2023


**Kandinsky: an Improved Text-to-Image Synthesis with Image Prior and Latent Diffusion** \
*Anton Razzhigaev, Arseniy Shakhmatov, Anastasia Maltseva, Vladimir Arkhipkin, Igor Pavlov, Ilya Ryabov, Angelina Kuts, Alexander Panchenko, Andrey Kuznetsov, Denis Dimitrov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03502)] \
5 Oct 2023

**Realistic Speech-to-Face Generation with Speech-Conditioned Latent Diffusion Model with Face Prior** \
*Jinting Wang, Li Liu, Jun Wang, Hei Victor Cheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03363)] \
5 Oct 2023

**T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation** \
*Yuze He, Yushi Bai, Matthieu Lin, Wang Zhao, Yubin Hu, Jenny Sheng, Ran Yi, Juanzi Li, Yong-Jin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02977)] [[Project](https://t3bench.com/)] [[Github](https://github.com/THU-LYJ-Lab/T3Bench)] \
4 Oct 2023

**Boosting Dermatoscopic Lesion Segmentation via Diffusion Models with Visual and Textual Prompts** \
*Shiyi Du, Xiaosong Wang, Yongyi Lu, Yuyin Zhou, Shaoting Zhang, Alan Yuille, Kang Li, Zongwei Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02906)] \
4 Oct 2023


**Magicremover: Tuning-free Text-guided Image inpainting with Diffusion Models** \
*Siyuan Yang, Lu Zhang, Liqian Ma, Yu Liu, JingJing Fu, You He* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02848)] \
4 Oct 2023


**ED-NeRF: Efficient Text-Guided Editing of 3D Scene using Latent Space NeRF** \
*Jangho Park, Gihyun Kwon, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02712)] \
4 Oct 2023

**SweetDreamer: Aligning Geometric Priors in 2D Diffusion for Consistent Text-to-3D** \
*Weiyu Li, Rui Chen, Xuelin Chen, Ping Tan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02596)] [[Project](https://sweetdreamer3d.github.io/)] \
4 Oct 2023

**EditVal: Benchmarking Diffusion Based Text-Guided Image Editing Methods** \
*Samyadeep Basu, Mehrdad Saberi, Shweta Bhardwaj, Atoosa Malemir Chegini, Daniela Massiceti, Maziar Sanjabi, Shell Xu Hu, Soheil Feizi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02426)] [[Project](https://deep-ml-research.github.io/editval/)] [[Github](https://github.com/deep-ml-research/editval_code)] \
3 Oct 2023


**FT-Shield: A Watermark Against Unauthorized Fine-tuning in Text-to-Image Diffusion Models** \
*Yingqian Cui, Jie Ren, Yuping Lin, Han Xu, Pengfei He, Yue Xing, Wenqi Fan, Hui Liu, Jiliang Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02401)] \
3 Oct 2023


**Amazing Combinatorial Creation: Acceptable Swap-Sampling for Text-to-Image Generation** \
*Jun Li, Zedong Zhang, Jian Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01819)] [[Project](https://asst2i.github.io/anon/)] \
3 Oct 2023

**Transcending Domains through Text-to-Image Diffusion: A Source-Free Approach to Domain Adaptation** \
*Shivang Chopra, Suraj Kothawade, Houda Aynaou, Aman Chadha* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01701)] \
2 Oct 2023


**Conditional Diffusion Distillation** \
*Kangfu Mei, Mauricio Delbracio, Hossein Talebi, Zhengzhong Tu, Vishal M. Patel, Peyman Milanfar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01407)] \
2 Oct 2023

**Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code** \
*Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, Qiang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01506)] \
2 Oct 2023


**Prompt-tuning latent diffusion models for inverse problems** \
*Hyungjin Chung, Jong Chul Ye, Peyman Milanfar, Mauricio Delbracio* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01110)] \
2 Oct 2023


**DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models** \
*Yongchan Kwon, Eric Wu, Kevin Wu, James Zou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00902)] \
2 Oct 2023

**Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models** \
*Hyeonho Jeong, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01107)] [[Github](https://github.com/Ground-A-Video/Ground-A-Video)] \
2 Oct 2023

**Music- and Lyrics-driven Dance Synthesis** \
*Wenjie Yin, Qingyuan Yao, Yi Yu, Hang Yin, Danica Kragic, Mårten Björkman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00455)] \
30 Sep 2023

**DiffPoseTalk: Speech-Driven Stylistic 3D Facial Animation and Head Pose Generation via Diffusion Models** \
*Zhiyao Sun, Tian Lv, Sheng Ye, Matthieu Gaetan Lin, Jenny Sheng, Yu-Hui Wen, Minjing Yu, Yong-jin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00434)] [[Project](https://raineggplant.github.io/DiffPoseTalk/)] \
30 Sep 2023

**PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis** \
*Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, Zhenguo Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00426)] [[Project](https://pixart-alpha.github.io/)] [[Github](https://github.com/PixArt-alpha/PixArt-alpha)] \
30 Sep 2023


**InstructCV: Instruction-Tuned Text-to-Image Diffusion Models as Vision Generalists** \
*Yulu Gan, Sungwoo Park, Alexander Schubert, Anthony Philippakis, Ahmed M. Alaa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00390)] \
30 Sep 2023

**Steered Diffusion: A Generalized Framework for Plug-and-Play Conditional Image Synthesis** \
*Nithin Gopalakrishnan Nair, Anoop Cherian, Suhas Lohit, Ye Wang, Toshiaki Koike-Akino, Vishal M. Patel, Tim K. Marks* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2310.00224)] \
30 Sep 2023

**Directly Fine-Tuning Diffusion Models on Differentiable Rewards** \
*Kevin Clark, Paul Vicol, Kevin Swersky, David J Fleet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.17400)] \
29 Sep 2023

**Text-image Alignment for Diffusion-based Perception** \
*Neehar Kondapaneni, Markus Marks, Manuel Knott, Rogério Guimarães, Pietro Perona* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00031)] \
29 Sep 2023


**LLM-grounded Video Diffusion Models** \
*Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, Boyi Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.17444)] [[Project](https://llm-grounded-diffusion.github.io/)] [[Github](https://github.com/TonyLianLong/LLM-groundedDiffusion)] \
29 Sep 2023

**KV Inversion: KV Embeddings Learning for Text-Conditioned Real Image Action Editing** \
*Jiancheng Huang, Yifan Liu, Jin Qin, Shifeng Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.16608)] \
28 Sep 2023


**CCEdit: Creative and Controllable Video Editing via Diffusion Models** \
*Ruoyu Feng, Wenming Weng, Yanhui Wang, Yuhui Yuan, Jianmin Bao, Chong Luo, Zhibo Chen, Baining Guo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.16496)] \
28 Sep 2023

**Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation** \
*David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, Mike Zheng Shou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.15818)] \
27 Sep 2023

**Dynamic Prompt Learning: Addressing Cross-Attention Leakage for Text-Based Image Editing** \
*Kai Wang, Fei Yang, Shiqi Yang, Muhammad Atif Butt, Joost van de Weijer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.15664)] \
27 Sep 2023

**DreamCom: Finetuning Text-guided Inpainting Model for Image Composition** \
*Lingxiao Lu, Bo Zhang, Li Niu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.15508)] \
27 Sep 2023

**Learning Using Generated Privileged Information by Text-to-Image Diffusion Models** \
*Rafael-Edy Menadil, Mariana-Iuliana Georgescu, Radu Tudor Ionescu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.15238)] \
26 Sep 2023

**LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models** \
*Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, Yuwei Guo, Tianxing Wu, Chenyang Si, Yuming Jiang, Cunjian Chen, Chen Change Loy, Bo Dai, Dahua Lin, Yu Qiao, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.15103)] [[Project](https://vchitect.github.io/LaVie-project/)] \
26 Sep 2023

**Learning Using Generated Privileged Information by Text-to-Image Diffusion Models** \
*Rafael-Edy Menadil, Mariana-Iuliana Georgescu, Radu Tudor Ionescu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.15238)] \
26 Sep 2023

**FEC: Three Finetuning-free Methods to Enhance Consistency for Real Image Editing** \
*Songyan Chen, Jiancheng Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14934)] \
26 Sep 2023

**Navigating Text-To-Image Customization:From LyCORIS Fine-Tuning to Model Evaluation** \
*Shin-Ying Yeh, Yu-Guan Hsieh, Zhidong Gao, Bernard B W Yang, Giyeong Oh, Yanmin Gong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14859)] \
26 Sep 2023

**Text-image guided Diffusion Model for generating Deepfake celebrity interactions** \
*Yunzhuo Chen, Nur Al Hasan Haldar, Naveed Akhtar, Ajmal Mian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14751)] \
26 Sep 2023

**Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator** \
*Hanzhuo Huang, Yufan Feng, Cheng Shi, Lan Xu, Jingyi Yu, Sibei Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14494)] \
25 Sep 2023

**COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs** \
*Tiep Le, Vasudev Lal, Phillip Howard* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14356)] \
23 Sep 2023

**Zero-Shot Object Counting with Language-Vision Models** \
*Jingyi Xu, Hieu Le, Dimitris Samaras* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2309.13097)] [[Github](https://github.com/cvlab-stonybrook/zero-shot-counting)] \
22 Sep 2023

**MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation** \
*Jiahao Xie, Wei Li, Xiangtai Li, Ziwei Liu, Yew Soon Ong, Chen Change Loy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.13042)] [[Github](https://github.com/Jiahao000/MosaicFusion)] \
22 Sep 2023

**DurIAN-E: Duration Informed Attention Network For Expressive Text-to-Speech Synthesis** \
*Yu Gu, Yianrao Bian, Guangzhi Lei, Chao Weng, Dan Su* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.12792)] \
22 Sep 2023

**FreeU: Free Lunch in Diffusion U-Net** \
*Chenyang Si, Ziqi Huang, Yuming Jiang, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11497)] \
20 Sep 2023

**Investigating Personalization Methods in Text to Music Generation** \
*Manos Plitsis, Theodoros Kouzelis, Georgios Paraskevopoulos, Vassilis Katsouros, Yannis Panagakis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11140)] [[Project](https://zelaki.github.io/)] \
20 Sep 2023

**Accelerating Diffusion-Based Text-to-Audio Generation with Consistency Distillation** \
*Yatong Bai, Trung Dang, Dung Tran, Kazuhito Koishida, Somayeh Sojoudi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.10740)] \
19 Sep 2023


**Forgedit: Text Guided Image Editing via Learning and Forgetting** \
*Shiwen Zhang, Shuai Xiao, Weilin Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.10556)] [[Github](https://github.com/witcherofresearch/Forgedit)] \
19 Sep 2023

**What is a Fair Diffusion Model? Designing Generative Text-To-Image Models to Incorporate Various Worldviews** \
*Zoe De Simone, Angie Boggust, Arvind Satyanarayan, Ashia Wilson* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.09944)] \
18 Sep 2023

**Causal-Story: Local Causal Attention Utilizing Parameter-Efficient Tuning For Visual Story Synthesis** \
*Tianyi Song, Jiuxin Cao, Kun Wang, Bo Liu, Xiaofeng Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.09553)] \
18 Sep 2023

**Progressive Text-to-Image Diffusion with Soft Latent Direction** \
*YuTeng Ye, Jiale Cai, Hang Zhou, Guanwen Li, Youjia Zhang, Zikai Song, Chenxing Gao, Junqing Yu, Wei Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.09466)] \
18 Sep 2023

**LivelySpeaker: Towards Semantic-Aware Co-Speech Gesture Generation** \
*Yihao Zhi, Xiaodong Cun, Xuelin Chen, Xi Shen, Wen Guo, Shaoli Huang, Shenghua Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.09294)] \
17 Sep 2023

**PromptTTS++: Controlling Speaker Identity in Prompt-Based Text-to-Speech Using Natural Language Descriptions** \
*Reo Shimizu, Ryuichi Yamamoto, Masaya Kawamura, Yuma Shirahata, Hironori Doi, Tatsuya Komatsu, Kentaro Tachibana* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.08140)] \
15 Sep 2023

**AV2Wav: Diffusion-Based Re-synthesis from Continuous Self-supervised Features for Audio-Visual Speech Enhancement** \
*Ju-Chieh Chou, Chung-Ming Chien, Karen Livescu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.08030)] \
14 Sep 2023


**Viewpoint Textual Inversion: Unleashing Novel View Synthesis with Pretrained 2D Diffusion Models** \
*James Burgess, Kuan-Chieh Wang, Serena Yeung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07986)] [[Github](https://github.com/jmhb0/view_neti)] \
14 Sep 2023

**Text-to-Image Models for Counterfactual Explanations: a Black-Box Approach** \
*Guillaume Jeanneret, Loïc Simon, Frédéric Jurie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07944)] \
14 Sep 2023

**Large-Vocabulary 3D Diffusion Model with Transformer** \
*Ziang Cao, Fangzhou Hong, Tong Wu, Liang Pan, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07920)] [[Project](https://ziangcao0312.github.io/difftf_pages/)] [[Github](https://github.com/ziangcao0312/DiffTF)] \
14 Sep 2023


**DiffTalker: Co-driven audio-image diffusion for talking faces via intermediate landmarks** \
*Zipeng Qi, Xulong Zhang, Ning Cheng, Jing Xiao, Jianzong Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07509)] \
14 Sep 2023

**Diffusion models for audio semantic communication** \
*Eleonora Grassucci, Christian Marinoni, Andrea Rodriguez, Danilo Comminiello* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07195)] \
13 Sep 2023

**DreamStyler: Paint by Style Inversion with Text-to-Image Diffusion Models** \
*Namhyuk Ahn, Junsoo Lee, Chunggi Lee, Kunhee Kim, Daesik Kim, Seung-Hun Nam, Kibeom Hong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.06933)] \
13 Sep 2023

**DCTTS: Discrete Diffusion Model with Contrastive Learning for Text-to-speech Generation** \
*Zhichao Wu, Qiulin Li, Sixing Liu, Qun Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.06787)] \
13 Sep 2023


**InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation** \
*Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.06380)] [[Github](https://github.com/gnobitab/InstaFlow)] \
12 Sep 2023



**Fg-T2M: Fine-Grained Text-Driven Human Motion Generation via Diffusion Model** \
*Yin Wang, Zhiying Leng, Frederick W. B. Li, Shun-Cheng Wu, Xiaohui Liang* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2309.06284)] \
12 Sep 2023

**Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts** \
*Zhi-Yi Chin, Chieh-Ming Jiang, Ching-Chun Huang, Pin-Yu Chen, Wei-Chen Chiu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.06135)] \
12 Sep 2023

**PhotoVerse: Tuning-Free Image Customization with Text-to-Image Diffusion Models** \
*Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, Min Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05793)] [[Project](https://photoverse2d.github.io/)] \
11 Sep 2023


**PAI-Diffusion: Constructing and Serving a Family of Open Chinese Diffusion Models for Text-to-image Synthesis on the Cloud** \
*Chengyu Wang, Zhongjie Duan, Bingyan Liu, Xinyi Zou, Cen Chen, Kui Jia, Jun Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05534)] \
11 Sep 2023

**Diffusion-Based Co-Speech Gesture Generation Using Joint Text and Audio Representation** \
*Anna Deichler, Shivam Mehta, Simon Alexanderson, Jonas Beskow* \
ICMI 2023. [[Paper](https://arxiv.org/abs/2309.05455)] \
11 Sep 2023

**Effective Real Image Editing with Accelerated Iterative Diffusion Inversion** \
*Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, Stephen Huang* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2309.04907)] \
10 Sep 2023

**Prefix-diffusion: A Lightweight Diffusion Model for Diverse Image Captioning** \
*Guisheng Liu, Yi Li, Zhengcong Fei, Haiyan Fu, Xiangyang Luo, Yanqing Guo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.04965)] \
10 Sep 2023

**Text-driven Editing of 3D Scenes without Retraining** \
*Shuangkang Fang, Yufeng Wang, Yi Yang, Yi-Hsuan Tsai, Wenrui Ding, Shuchang Zhou, Ming-Hsuan Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.04917)] \
10 Sep 2023

**The Power of Sound (TPoS): Audio Reactive Video Generation with Stable Diffusion** \
*Yujin Jeong, Wonjeong Ryoo, Seunghyun Lee, Dabin Seo, Wonmin Byeon, Sangpil Kim, Jinkyu Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.04509)] \
8 Sep 2023

**Create Your World: Lifelong Text-to-Image Diffusion** \
*Gan Sun, Wenqi Liang, Jiahua Dong, Jun Li, Zhengming Ding, Yang Cong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.04430)] \
8 Sep 2023

**MaskDiffusion: Boosting Text-to-Image Consistency with Conditional Mask** \
*Yupeng Zhou, Daquan Zhou, Zuo-Liang Zhu, Yaxing Wang, Qibin Hou, Jiashi Feng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.04399)] \
8 Sep 2023

**MoEController: Instruction-based Arbitrary Image Manipulation with Mixture-of-Expert Controllers** \
*Sijia Li, Chen Chen, Haonan Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.04372)] [[Project](https://oppo-mente-lab.github.io/moe_controller/)] \
8 Sep 2023

**From Text to Mask: Localizing Entities Using the Attention of Text-to-Image Diffusion Models** \
*Changming Xiao, Qi Yang, Feng Zhou, Changshui Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.04109)] \
8 Sep 2023



**InstructDiffusion: A Generalist Modeling Interface for Vision Tasks** \
*Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Han Hu, Dong Chen, Baining Guo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03895)] [[Project](https://gengzigang.github.io/instructdiffusion.github.io/)] [[Github](https://github.com/cientgu/InstructDiffusion)] \
7 Sep 2023

**Text-to-feature diffusion for audio-visual few-shot learning** \
*Otniel-Bogdan Mercea, Thomas Hummel, A. Sophia Koepke, Zeynep Akata* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03869)] \
7 Sep 2023

**Text2Control3D: Controllable 3D Avatar Generation in Neural Radiance Fields using Geometry-Guided Text-to-Image Diffusion Model** \
*Sungwon Hwang, Junha Hyung, Jaegul Choo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03550)] [[Project](https://text2control3d.github.io/)] \
7 Sep 2023

**Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation** \
*Jiaxi Gu, Shicong Wang, Haoyu Zhao, Tianyi Lu, Xing Zhang, Zuxuan Wu, Songcen Xu, Wei Zhang, Yu-Gang Jiang, Hang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03549)] \
7 Sep 2023

**SyncDreamer: Generating Multiview-consistent Images from a Single-view Image** \
*Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, Wenping Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03453)] [[Project](https://liuyuan-pal.github.io/SyncDreamer/)] [[Github](https://github.com/liuyuan-pal/SyncDreamer)] \
7 Sep 2023

**MCM: Multi-condition Motion Synthesis Framework for Multi-scenario** \
*Zeyu Ling, Bo Han, Yongkang Wong, Mohan Kangkanhalli, Weidong Geng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03031)] \
6 Sep 2023

**Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter** \
*Jinglong Wang, Xiawei Li, Jing Zhang, Qingyuan Xu, Qin Zhou, Qian Yu, Lu Sheng, Dong Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.02773)] \
6 Sep 2023


**Generating Realistic Images from In-the-wild Sounds** \
*Taegyeong Lee, Jeonghun Kang, Hyeonyu Kim, Taehwan Kim* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2309.02405)] \
5 Sep 2023

**Generative-based Fusion Mechanism for Multi-Modal Tracking** \
*Zhangyong Tang, Tianyang Xu, Xuefeng Zhu, Xiao-Jun Wu, Josef Kittler* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01728)] \
4 Sep 2023

**VGDiffZero: Text-to-image Diffusion Models Can Be Zero-shot Visual Grounders** \
*Xuyang Liu, Siteng Huang, Yachen Kang, Honggang Chen, Donglin Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01141)] \
3 Sep 2023

**Bridge Diffusion Model: bridge non-English language-native text-to-image diffusion model with English communities** \
*Shanyuan Liu, Dawei Leng, Yuhui Yin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00952)] \
2 Sep 2023

**MagicProp: Diffusion-based Video Editing via Motion-aware Appearance Propagation** \
*Hanshu Yan, Jun Hao Liew, Long Mai, Shanchuan Lin, Jiashi Feng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00908)] \
2 Sep 2023

**Iterative Multi-granular Image Editing using Diffusion Models** \
*K J Joseph, Prateksha Udhayanan, Tripti Shukla, Aishwarya Agarwal, Srikrishna Karanam, Koustava Goswami, Balaji Vasan Srinivasan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00613)] \
1 Sep 2023

**DiffuGen: Adaptable Approach for Generating Labeled Image Datasets using Stable Diffusion Models** \
*Michael Shenoda, Edward Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00248)] \
1 Sep 2023

**PathLDM: Text conditioned Latent Diffusion Model for Histopathology** \
*Srikar Yellapragada, Alexandros Graikos, Prateek Prasanna, Tahsin Kurc, Joel Saltz, Dimitris Samaras* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00748)] \
1 Sep 2023

**VideoGen: A Reference-Guided Latent Diffusion Approach for High Definition Text-to-Video Generation** \
*Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, Jingdong Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.00398)] \
1 Sep 2023

**Detecting Out-of-Context Image-Caption Pairs in News: A Counter-Intuitive Method** \
*Eivind Moholdt, Sohail Ahmed Khan, Duc-Tien Dang-Nguyen* \
CBMI 2023. [[Paper](https://arxiv.org/abs/2308.16611)] \
31 Aug 2023

**Any-Size-Diffusion: Toward Efficient Text-Driven Synthesis for Any-Size HD Images** \
*Qingping Zheng, Yuanfan Guo, Jiankang Deng, Jianhua Han, Ying Li, Songcen Xu, Hang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16582)] \
31 Aug 2023

**MVDream: Multi-view Diffusion for 3D Generation** \
*Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, Xiao Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16512)] \
31 Aug 2023

**Intriguing Properties of Diffusion Models: A Large-Scale Dataset for Evaluating Natural Attack Capability in Text-to-Image Generative Models** \
*Takami Sato, Justin Yue, Nanze Chen, Ningfei Wang, Qi Alfred Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15692)] \
30 Aug 2023

**DiffusionVMR: Diffusion Model for Video Moment Retrieval** \
*Henghao Zhao, Kevin Qinghong Lin, Rui Yan, Zechao Li* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.15109)] \
29 Aug 2023

**C2G2: Controllable Co-speech Gesture Generation with Latent Diffusion Model** \
*Longbin Ji, Pengfei Wei, Yi Ren, Jinglin Liu, Chen Zhang, Xiang Yin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15016)] \
29 Aug 2023

**360-Degree Panorama Generation from Few Unregistered NFoV Images** \
*Jionghao Wang, Ziyu Chen, Jun Ling, Rong Xie, Li Song* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.14686)] [[Github](https://github.com/shanemankiw/Panodiff)] \
28 Aug 2023

**Priority-Centric Human Motion Generation in Discrete Latent Space** \
*Hanyang Kong, Kehong Gong, Dongze Lian, Michael Bi Mi, Xinchao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.14480)] \
28 Aug 2023

**SketchDreamer: Interactive Text-Augmented Creative Sketch Ideation** \
*Zhiyu Qu, Tao Xiang, Yi-Zhe Song* \
BMVC 2023. [[Paper](https://arxiv.org/abs/2308.14191)] [[Github](https://github.com/WinKawaks/SketchDreamer)] \
27 Aug 2023

**Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models** \
*Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Tat-Seng Chua* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13812)] [[Project](https://haofei.vip/Dysen-VDM/)] \
26 Aug 2023

**ORES: Open-vocabulary Responsible Visual Synthesis** \
*Minheng Ni, Chenfei Wu, Xiaodong Wang, Shengming Yin, Lijuan Wang, Zicheng Liu, Nan Duan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13785)] \
26 Aug 2023

**The DiffuseStyleGesture+ entry to the GENEA Challenge 2023** \
*Sicheng Yang, Haiwei Xue, Zhensong Zhang, Minglei Li, Zhiyong Wu, Xiaofei Wu, Songcen Xu, Zonghong Dai* \
ICMI 2023. [[Paper](https://arxiv.org/abs/2308.13879)] [[Github](https://github.com/YoungSeng/DiffuseStyleGesture/tree/DiffuseStyleGesturePlus/BEAT-TWH-main)] \
26 Aug 2023

**EfficientDreamer: High-Fidelity and Robust 3D Creation via Orthogonal-view Diffusion Prior** \
*Minda Zhao, Chaoyi Zhao, Xinyue Liang, Lincheng Li, Zeng Zhao, Zhipeng Hu, Changjie Fan, Xin Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13223)] \
25 Aug 2023

**Unified Concept Editing in Diffusion Models** \
*Rohit Gandikota, Hadas Orgad, Yonatan Belinkov, Joanna Materzyńska, David Bau* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.14761)] [[Project](https://unified.baulab.info/)] [[Github](https://github.com/rohitgandikota/unified-concept-editing)] \
25 Aug 2023

**Dense Text-to-Image Generation with Attention Modulation** \
*Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, Jun-Yan Zhu* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.12964)] [[Github](https://github.com/naver-ai/DenseDiffusion)] \
24 Aug 2023

**APLA: Additional Perturbation for Latent Noise with Adversarial Training Enables Consistency** \
*Yupu Yao, Shangqi Deng, Zihan Cao, Harry Zhang, Liang-Jian Deng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.12605)] \
24 Aug 2023

**Manipulating Embeddings of Stable Diffusion Prompts** \
*Niklas Deckers, Julia Peters, Martin Potthast* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.12059)] \
23 Aug 2023




**DF-3DFace: One-to-Many Speech Synchronized 3D Face Animation with Diffusion** \
*Se Jin Park, Joanna Hong, Minsu Kim, Yong Man Ro* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05934)] \
23 Aug 2023

**IT3D: Improved Text-to-3D Generation with Explicit View Synthesis** \
*Yiwen Chen, Chi Zhang, Xiaofeng Yang, Zhongang Cai, Gang Yu, Lei Yang, Guosheng Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.11473)] [[Github](https://github.com/buaacyw/IT3D-text-to-3D)] \
22 Aug 2023

**DiffCloth: Diffusion Based Garment Synthesis and Manipulation via Structural Cross-modal Semantic Alignment** \
*Xujie Zhang, Binbin Yang, Michael C. Kampffmeyer, Wenqing Zhang, Shiyue Zhang, Guansong Lu, Liang Lin, Hang Xu, Xiaodan Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.11206)] \
22 Aug 2023

**MusicJam: Visualizing Music Insights via Generated Narrative Illustrations** \
*Chuer Chen, Nan Cao, Jiani Hou, Yi Guo, Yulei Zhang, Yang Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.11329)] \
22 Aug 2023

**TADA! Text to Animatable Digital Avatars** \
*Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxaing Tang, Yangyi Huang, Justus Thies, Michael J. Black* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.10899)] \
21 Aug 2023

**EVE: Efficient zero-shot text-based Video Editing with Depth Map Guidance and Temporal Consistency Constraints** \
*Yutao Chen, Xingning Dong, Tian Gan, Chunluan Zhou, Ming Yang, Qingpei Guo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.10648)] \
21 Aug 2023

**Backdooring Textual Inversion for Concept Censorship** \
*Yutong Wu, Jie Zhang, Florian Kerschbaum, Tianwei Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.10718)] [[Project](https://concept-censorship.github.io/)] [[Github](https://github.com/concept-censorship/concept-censorship.github.io/tree/main/code)] \
21 Aug 2023

**AltDiffusion: A Multilingual Text-to-Image Diffusion Model** \
*Fulong Ye, Guang Liu, Xinya Wu, Ledell Wu* \
AAAI 2024. [[Paper](https://arxiv.org/abs/2308.09991)] [[Github](https://github.com/superhero-7/AltDiffuson)] \
19 Aug 2023

**DiffDis: Empowering Generative Diffusion Model with Cross-Modal Discrimination Capability** \
*Runhui Huang, Jianhua Han, Guansong Lu, Xiaodan Liang, Yihan Zeng, Wei Zhang, Hang Xu* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.09306)] \
18 Aug 2023

**MATLABER: Material-Aware Text-to-3D via LAtent BRDF auto-EncodeR** \
*Xudong Xu, Zhaoyang Lyu, Xingang Pan, Bo Dai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09278)] [[Project](https://sheldontsui.github.io/projects/Matlaber)] \
18 Aug 2023

**Diff2Lip: Audio Conditioned Diffusion Models for Lip-Synchronization** \
*Soumik Mukhopadhyay, Saksham Suri, Ravi Teja Gadde, Abhinav Shrivastava* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09716)] [[Project](https://soumik-kanad.github.io/diff2lip/)] [[Github](https://github.com/soumik-kanad/diff2lip)] \
18 Aug 2023

**Guide3D: Create 3D Avatars from Text and Image Guidance** \
*Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, Kwan-Yee K. Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09705)] \
18 Aug 2023

**Language-Guided Diffusion Model for Visual Grounding** \
*Sijia Chen, Baochun Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09599)] \
18 Aug 2023

**SimDA: Simple Diffusion Adapter for Efficient Video Generation** \
*Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, Yu-Gang Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09710)] [[Project](https://chenhsing.github.io/SimDA/)] \
18 Aug 2023

**StableVideo: Text-driven Consistency-aware Diffusion Video Editing** \
*Wenhao Chai, Xun Guo, Gaoang Wang, Yan Lu* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.09592)] [[Github](https://github.com/rese1f/StableVideo)] \
18 Aug 2023

**Edit Temporal-Consistent Videos with Image Diffusion Model** \
*Yuanzhi Wang, Yong Li, Xin Liu, Anbo Dai, Antoni Chan, Zhen Cui* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09091)] \
17 Aug 2023

**Watch Your Steps: Local Image and Scene Editing by Text Instructions** \
*Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, Igor Gilitschenski* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.08947)] [[Project](https://ashmrz.github.io/WatchYourSteps/)] \
17 Aug 2023

**Learning to Generate Semantic Layouts for Higher Text-Image Correspondence in Text-to-Image Synthesis** \
*Minho Park, Jooyeol Yun, Seunghwan Choi, Jaegul Choo* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.08157)] [[Project](https://pmh9960.github.io/research/GCDP/)] [[Github](https://github.com/pmh9960/GCDP/)] \
16 Aug 2023


**DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory** \
*Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, Nan Duan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.08089)] [[Project](https://www.microsoft.com/en-us/research/project/dragnuwa/)] \
16 Aug 2023

**Dual-Stream Diffusion Net for Text-to-Video Generation** \
*Binhui Liu, Xin Liu, Anbo Dai, Zhiyong Zeng, Zhen Cui, Jian Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.08316)] \
16 Aug 2023

**DiffV2S: Diffusion-based Video-to-Speech Synthesis with Vision-guided Speaker Embedding** \
*Jeongsoo Choi, Joanna Hong, Yong Man Ro* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07787)] \
15 Aug 2023

**SGDiff: A Style Guided Diffusion Model for Fashion Synthesis** \
*Zhengwentai Sun, Yanghong Zhou, Honghong He, P. Y. Mok* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.07605)] \
15 Aug 2023

**Dancing Avatar: Pose and Text-Guided Human Motion Videos Synthesis with Image Diffusion Model** \
*Bosheng Qin, Wentao Ye, Qifan Yu, Siliang Tang, Yueting Zhuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07749)] \
15 Aug 2023

**Diffusion Based Augmentation for Captioning and Retrieval in Cultural Heritage** \
*Dario Cioni, Lorenzo Berlincioni, Federico Becattini, Alberto del Bimbo* \
ICCV Workshop 2023. [[Paper](https://arxiv.org/abs/2308.07151)] \
14 Aug 2023

**Jurassic World Remake: Bringing Ancient Fossils Back to Life via Zero-Shot Long Image-to-Image Translation** \
*Alexander Martin, Haitian Zheng, Jie An, Jiebo Luo* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.07316)] \
14 Aug 2023

**UniBrain: Unify Image Reconstruction and Captioning All in One Diffusion Model from Human Brain Activity** \
*Weijian Mai, Zhijun Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07428)] \
14 Aug 2023

**Free-ATM: Exploring Unsupervised Learning on Diffusion-Generated Images with Free Attention Masks** \
*David Junhao Zhang, Mutian Xu, Chuhui Xue, Wenqing Zhang, Xiaoguang Han, Song Bai, Mike Zheng Shou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06739)] \
13 Aug 2023

**IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models** \
*Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, Wei Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06721)] [[Project](https://ip-adapter.github.io/)] [[Github](https://github.com/tencent-ailab/IP-Adapter)] \
13 Aug 2023

**LAW-Diffusion: Complex Scene Generation by Diffusion with Layouts** \
*Binbin Yang, Yi Luo, Ziliang Chen, Guangrun Wang, Xiaodan Liang, Liang Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06713)] \
13 Aug 2023

**ModelScope Text-to-Video Technical Report** \
*Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, Shiwei Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06571)] \
12 Aug 2023

**DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models** \
*Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, Chunhua Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06160)] [[Project](https://weijiawu.github.io/DatasetDM_page/)] [[Github](https://github.com/showlab/DatasetDM)] \
11 Aug 2023

**Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning** \
*Chun-Mei Feng, Kai Yu, Yong Liu, Salman Khan, Wangmeng Zuo* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.06038)] [[Github](https://github.com/chunmeifeng/DiffTPT)] \
11 Aug 2023

**Masked-Attention Diffusion Guidance for Spatially Controlling Text-to-Image Generation** \
*Yuki Endo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06027)] \
11 Aug 2023

**Audio is all in one: speech-driven gesture synthetics using WavLM pre-trained model** \
*Fan Zhang, Naye Ji, Fuxing Gao, Siyuan Zhao, Zhaohan Wang, Shunman Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.05995)] \
11 Aug 2023

**Zero-shot Text-driven Physically Interpretable Face Editing** \
*Yapeng Meng, Songru Yang, Xu Hu, Rui Zhao, Lincheng Li, Zhenwei Shi, Zhengxia Zou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.05976)] \
11 Aug 2023

**PromptPaint: Steering Text-to-Image Generation Through Paint Medium-like Interactions** \
*John Joon Young Chung, Eytan Adar* \
UIST 2023. [[Paper](https://arxiv.org/abs/2308.05184)] \
9 Aug 2023

**LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation** \
*Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, Tat-Seng Chua* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.05095)] [[Project](https://layoutllm-t2i.github.io/)] \
9 Aug 2023

**Cloth2Tex: A Customized Cloth Texture Generation Pipeline for 3D Virtual Try-On** \
*Daiheng Gao, Xu Chen, Xindi Zhang, Qi Wang, Ke Sun, Bang Zhang, Liefeng Bo, Qixing Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.04288)] \
8 Aug 2023

**MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion** \
*Yizhuo Lu, Changde Du, Qiongyi zhou, Dianpeng Wang, Huiguang He* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.04249)] \
8 Aug 2023

**FLIRT: Feedback Loop In-context Red Teaming** \
*Ninareh Mehrabi, Palash Goyal, Christophe Dupuy, Qian Hu, Shalini Ghosh, Richard Zemel, Kai-Wei Chang, Aram Galstyan, Rahul Gupta* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.04265)] \
8 Aug 2023

**DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis** \
*Zhongjie Duan, Lizhou You, Chengyu Wang, Cen Chen, Ziheng Wu, Weining Qian, Jun Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.03463)] [[Project](https://anonymous456852.github.io/)] [[Github](https://github.com/alibaba/EasyNLP/tree/master/diffusion)] \
7 Aug 2023

**AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose** \
*Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Kang Du, Min Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.03610)] [[Project](https://avatarverse3d.github.io/)] \
7 Aug 2023

**Towards Scene-Text to Scene-Text Translation** \
*Onkar Susladkar, Prajwal Gatti, Anand Mishra* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.03024)] \
6 Aug 2023

**Sketch and Text Guided Diffusion Model for Colored Point Cloud Generation** \
*Zijie Wu, Yaonan Wang, Mingtao Feng, He Xie, Ajmal Mian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02874)] \
5 Aug 2023

**ConceptLab: Creative Generation using Diffusion Prior Constraints** \
*Elad Richardson, Kfir Goldberg, Yuval Alaluf, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02669)] [[Project](https://kfirgoldberg.github.io/ConceptLab/)] [[Github](https://github.com/kfirgoldberg/ConceptLab)] \
3 Aug 2023

**DiffColor: Toward High Fidelity Text-Guided Image Colorization with Diffusion Models** \
*Jianxin Lin, Peng Xiao, Yijun Wang, Rongju Zhang, Xiangxiang Zeng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01655)] \
3 Aug 2023

**Synthesizing Long-Term Human Motions with Diffusion Models via Coherent Sampling** \
*Zhao Yang, Bing Su, Ji-Rong Wen* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.01850)] [[Github](https://github.com/yangzhao1230/PCMDM)] \
3 Aug 2023

**Reverse Stable Diffusion: What prompt was used to generate this image?** \
*Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Mubarak Shah* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.01472)] \
2 Aug 2023

**Degeneration-Tuning: Using Scrambled Grid shield Unwanted Concepts from Stable Diffusion** \
*Zixuan Ni, Longhui Wei, Jiacheng Li, Siliang Tang, Yueting Zhuang, Qi Tian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02552)] \
2 Aug 2023

**ImageBrush: Learning Visual In-Context Instructions for Exemplar-Based Image Manipulation** \
*Yasheng Sun, Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, Hideki Koike* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.00906)] \
2 Aug 2023

**The Bias Amplification Paradox in Text-to-Image Generation** \
*Preethi Seshadri, Sameer Singh, Yanai Elazar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.00755)] \
1 Aug 2023

**BAGM: A Backdoor Attack for Manipulating Text-to-Image Generative Models** \
*Jordan Vice, Naveed Akhtar, Richard Hartley, Ajmal Mian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16489)] [[Github](https://github.com/JJ-Vice/BAGM)] [[Dataset](https://ieee-dataport.org/documents/marketable-foods-mf-dataset)] \
31 Jul 2023

**MobileVidFactory: Automatic Diffusion-Based Social Media Video Generation for Mobile Devices from Text** \
*Junchen Zhu, Huan Yang, Wenjing Wang, Huiguo He, Zixi Tuo, Yongsheng Yu, Wen-Huang Cheng, Lianli Gao, Jingkuan Song, Jianlong Fu, Jiebo Luo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16371)] \
31 Jul 2023

**DAVIS: High-Quality Audio-Visual Separation with Generative Diffusion Models** \
*Chao Huang, Susan Liang, Yapeng Tian, Anurag Kumar, Chenliang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.00122)] \
31 Jul 2023

**Contrastive Conditional Latent Diffusion for Audio-visual Segmentation** \
*Yuxin Mao, Jing Zhang, Mochu Xiang, Yunqiu Lv, Yiran Zhong, Yuchao Dai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16579)] \
31 Jul 2023

**HD-Fusion: Detailed Text-to-3D Generation Leveraging Multiple Noise Estimation** \
*Jinbo Wu, Xiaobo Gao, Xing Liu, Zhengyang Shen, Chen Zhao, Haocheng Feng, Jingtuo Liu, Errui Ding* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16183)] \
30 Jul 2023

**Seeing through the Brain: Image Reconstruction of Visual Perception from Human Brain Signals** \
*Yu-Ting Lan, Kan Ren, Yansen Wang, Wei-Long Zheng, Dongsheng Li, Bao-Liang Lu, Lili Qiu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02510)] \
27 Jul 2023

**VideoControlNet: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet** \
*Zhihao Hu, Dong Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.14073)] [[Project](https://vcg-aigc.github.io/)] \
26 Jul 2023


**Points-to-3D: Bridging the Gap between Sparse Points and Shape-Controllable Text-to-3D Generation** \
*Chaohui Yu, Qiang Zhou, Jingliang Li, Zhe Zhang, Zhibin Wang, Fan Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.13908)] \
26 Jul 2023

**Visual Instruction Inversion: Image Editing via Visual Prompting** \
*Thao Nguyen, Yuheng Li, Utkarsh Ojha, Yong Jae Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.14331)] [[Project](https://thaoshibe.github.io/visii/)] [[Github](https://github.com/thaoshibe/visii)] \
26 Jul 2023


**Composite Diffusion | whole >= \Sigma parts** \
*Vikram Jamwal, Ramaneswaran S* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.13720)] \
25 Jul 2023


**Fashion Matrix: Editing Photos by Just Talking** \
*Zheng Chong, Xujie Zhang, Fuwei Zhao, Zhenyu Xie, Xiaodan Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.13240)] [[Project](https://zheng-chong.github.io/FashionMatrix/)] [[Github](https://github.com/Zheng-Chong/FashionMatrix)] \
25 Jul 2023

**Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry** \
*Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, Youngjung Uh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.12868)] \
24 Jul 2023

**InFusion: Inject and Attention Fusion for Multi Concept Zero-Shot Text-based Video Editing** \
*Anant Khandelwal* \
ICCV Workshop 2023. [[Paper](https://arxiv.org/abs/2308.00135)] \
22 Jul 2023

**Subject-Diffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning** \
*Jian Ma, Junhao Liang, Chen Chen, Haonan Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.11410)] [[Project](https://oppo-mente-lab.github.io/subject_diffusion/)] [[Github](https://github.com/OPPO-Mente-Lab/Subject-Diffusion)] \
21 Jul 2023

**Divide & Bind Your Attention for Improved Generative Semantic Nursing** \
*Yumeng Li, Margret Keuper, Dan Zhang, Anna Khoreva* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.10864)] [[Project](https://sites.google.com/view/divide-and-bind)] \
20 Jul 2023

**AdjointDPM: Adjoint Sensitivity Method for Gradient Backpropagation of Diffusion Probabilistic Models** \
*Jiachun Pan, Jun Hao Liew, Vincent Y. F. Tan, Jiashi Feng, Hanshu Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.10711)] \
20 Jul 2023

**BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion** \
*Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, Mike Zheng Shou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.10816)] [[Github](https://github.com/Sierkinhane/BoxDiff)] \
20 Jul 2023

**Text2Layer: Layered Image Generation using Latent Diffusion Model** \
*Xinyang Zhang, Wentian Zhao, Xin Lu, Jeff Chien* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.09781)] \
19 Jul 2023

**FABRIC: Personalizing Diffusion Models with Iterative Feedback** \
*Dimitri von Rütte, Elisabetta Fedele, Jonathan Thomm, Lukas Wolf* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.10159)] \
19 Jul 2023

**TokenFlow: Consistent Diffusion Features for Consistent Video Editing** \
*Michal Geyer, Omer Bar-Tal, Shai Bagon, Tali Dekel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.10373)] [[Project](https://diffusion-tokenflow.github.io/)] [[Github](https://github.com/omerbt/TokenFlow)] \
19 Jul 2023

**Multimodal Diffusion Segmentation Model for Object Segmentation from Manipulation Instructions** \
*Yui Iioka, Yu Yoshida, Yuiga Wada, Shumpei Hatanaka, Komei Sugiura* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08597)] \
17 Jul 2023

**Not All Steps are Created Equal: Selective Diffusion Distillation for Image Manipulation** \
*Luozhou Wang, Shuai Yang, Shu Liu, Ying-cong Chen* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2307.08448)] [[Github](https://github.com/AndysonYs/Selective-Diffusion-Distillation)] \
17 Jul 2023

**Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection** \
*Alessandro Flaborea, Luca Collorone, Guido D'Amely, Stefano D'Arrigo, Bardh Prenkaj, Fabio Galasso* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.07205)] \
14 Jul 2023

**HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models** \
*Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, Kfir Aberman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.06949)] [[Project](https://hyperdreambooth.github.io/)] [[Github](https://github.com/JiauZhang/hyperdreambooth)] \
13 Jul 2023

**Exact Diffusion Inversion via Bi-directional Integration Approximation** \
*Guoqiang Zhang, J. P. Lewis, W. Bastiaan Kleijn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.10829)] \
10 Jul 2023

**AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning** \
*Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, Bo Dai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04725)] [[Project](https://animatediff.github.io/)] [[Github](https://github.com/guoyww/animatediff/)] \
10 Jul 2023


**Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback** \
*Jaskirat Singh, Liang Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04749)] [[Project](https://1jsingh.github.io/divide-evaluate-and-refine)] [[Github](https://github.com/1jsingh/Divide-Evaluate-and-Refine)] \
10 Jul 2023

**Augmenters at SemEval-2023 Task 1: Enhancing CLIP in Handling Compositionality and Ambiguity for Zero-Shot Visual WSD through Prompt Augmentation and Text-To-Image Diffusion** \
*Jie S. Li, Yow-Ting Shiue, Yong-Siang Shih, Jonas Geiping* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.05564)] \
9 Jul 2023

**Measuring the Success of Diffusion Models at Imitating Human Artists** \
*Stephen Casper, Zifan Guo, Shreya Mogulothu, Zachary Marinov, Chinmay Deshpande, Rui-Jie Yew, Zheng Dai, Dylan Hadfield-Menell* \
ICML Workshop 2023. [[Paper](https://arxiv.org/abs/2307.04028)] \
8 Jul 2023

**How to Detect Unauthorized Data Usages in Text-to-image Diffusion Models** \
*Zhenting Wang, Chen Chen, Yuchen Liu, Lingjuan Lyu, Dimitris Metaxas, Shiqing Ma* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.03108)] \
6 Jul 2023


**Collaborative Score Distillation for Consistent Visual Synthesis** \
*Subin Kim, Kyungmin Lee, June Suk Choi, Jongheon Jeong, Kihyuk Sohn, Jinwoo Shin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04787)] [[Project](https://subin-kim-cv.github.io/CSD/)] [[Github](https://github.com/subin-kim-cv/CSD)] \
4 Jul 2023



**SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis** \
*Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.01952)] [[Github](https://github.com/Stability-AI/generative-models)] \
4 Jul 2023



**MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion** \
*Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, Yasutaka Furukawa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.01097)] [[Project](https://mvdiffusion.github.io/)] \
3 Jul 2023

**Counting Guidance for High Fidelity Text-to-Image Synthesis** \
*Wonjun Kang, Kevin Galim, Hyung Il Koo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17567)] \
30 Jun 2023

**Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation** \
*Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, Shenghua Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17115)] \
29 Jun 2023

**Generate Anything Anywhere in Any Scene** \
*Yuheng Li, Haotian Liu, Yangming Wen, Yong Jae Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17154)] [[Project](https://yuheng-li.github.io/PACGen/)] \
29 Jun 2023

**Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models** \
*Simian Luo, Chuanhao Yan, Chenxu Hu, Hang Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17203)] [[Github](https://github.com/luosiallen/Diff-Foley)] \
29 Jun 2023

**PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing** \
*Wenjing Huang, Shikui Tu, Lei Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.16894)] \
28 Jun 2023

**DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models** \
*Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, Dong Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.14685)] \
26 Jun 2023

**A-STAR: Test-time Attention Segregation and Retention for Text-to-image Synthesis** \
*Aishwarya Agarwal, Srikrishna Karanam, K J Joseph, Apoorv Saxena, Koustava Goswami, Balaji Vasan Srinivasan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.14544)] \
26 Jun 2023

**Decompose and Realign: Tackling Condition Misalignment in Text-to-Image Diffusion Models** \
*Luozhou Wang, Guibao Shen, Yijun Li, Ying-cong Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.14408)] \
26 Jun 2023

**Zero-shot spatial layout conditioning for text-to-image diffusion models** \
*Guillaume Couairon, Marlène Careil, Matthieu Cord, Stéphane Lathuilière, Jakob Verbeek* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.13754)] \
23 Jun 2023

**DreamTime: An Improved Optimization Strategy for Text-to-3D Content Creation** \
*Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, Zheng-Jun Zha, Lei Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12422)] \
21 Jun 2023

**Align, Adapt and Inject: Sound-guided Unified Image Generation** \
*Yue Yang, Kaipeng Zhang, Yuying Ge, Wenqi Shao, Zeyue Xue, Yu Qiao, Ping Luo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11504)] \
20 Jun 2023


**EMoG: Synthesizing Emotive Co-speech 3D Gesture with Diffusion Model** \
*Lianying Yin, Yijun Wang, Tianyu He, Jinming Liu, Wei Zhao, Bohan Li, Xin Jin, Jianxin Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11496)] \
20 Jun 2023

**RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model** \
*Zilun Zhang, Tiancheng Zhao, Yulong Guo, Jianwei Yin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11300)] \
20 Jun 2023


**Instruct-NeuralTalker: Editing Audio-Driven Talking Radiance Fields with Instructions** \
*Yuqi Sun, Reian He, Weimin Tan, Bo Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.10813)] \
19 Jun 2023

**Conditional Text Image Generation with Diffusion Models** \
*Yuanzhi Zhu, Zhaohai Li, Tianwei Wang, Mengchao He, Cong Yao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.10804)] \
19 Jun 2023

**Point-Cloud Completion with Pretrained Text-to-image Diffusion Models** \
*Yoni Kasten, Ohad Rahamim, Gal Chechik* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.10533)] \
18 Jun 2023

**Energy-Based Cross Attention for Bayesian Context Update in Text-to-Image Diffusion Models** \
*Geon Yeong Park, Jeongsol Kim, Beomsu Kim, Sang Wan Lee, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09869)] \
16 Jun 2023



**Evaluating the Robustness of Text-to-image Diffusion Models against Real-world Attacks** \
*Hongcheng Gao, Hao Zhang, Yinpeng Dong, Zhijie Deng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.13103)] \
16 Jun 2023

**CLIPSonic: Text-to-Audio Synthesis with Unlabeled Videos and Pretrained Language-Vision Models** \
*Hao-Wen Dong, Xiaoyu Liu, Jordi Pons, Gautam Bhattacharya, Santiago Pascual, Joan Serrà, Taylor Berg-Kirkpatrick, Julian McAuley* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09635)] \
16 Jun 2023

**Taming Diffusion Models for Music-driven Conducting Motion Generation** \
*Zhuoran Zhao, Jinbin Bai, Delong Chen, Debang Wang, Yubo Pan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.10065)] \
15 Jun 2023


**Diff-TTSG: Denoising probabilistic integrated speech and gesture synthesis** \
*Shivam Mehta, Siyang Wang, Simon Alexanderson, Jonas Beskow, Éva Székely, Gustav Eje Henter* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09417)] \
15 Jun 2023

**Diffusion Models for Zero-Shot Open-Vocabulary Segmentation** \
*Laurynas Karazija, Iro Laina, Andrea Vedaldi, Christian Rupprecht* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09316)] \
15 Jun 2023

**Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment** \
*Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, Gal Chechik* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08877)] \
15 Jun 2023

**Training Multimedia Event Extraction With Generated Images and Captions** \
*Zilin Du, Yunxin Li, Xu Guo, Yidan Sun, Boyang Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08966)] \
15 Jun 2023

**VidEdit: Zero-Shot and Spatially Aware Text-Driven Video Editing** \
*Paul Couairon, Clément Rambour, Jean-Emmanuel Haugeard, Nicolas Thome* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08707)] \
14 Jun 2023


**Norm-guided latent space exploration for text-to-image generation** \
*Dvir Samuel, Rami Ben-Ari, Nir Darshan, Haggai Maron, Gal Chechik* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08687)] \
14 Jun 2023


**Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis** \
*Zhiyu Jin, Xuli Shen, Bin Li, Xiangyang Xue* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08645)] \
14 Jun 2023

**GBSD: Generative Bokeh with Stage Diffusion** \
*Jieren Deng, Xin Zhou, Hao Tian, Zhihong Pan, Derek Aguiar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08251)] \
14 Jun 2023

**Diffusion in Diffusion: Cyclic One-Way Diffusion for Text-Vision-Conditioned Generation** \
*Yongqi Yang, Ruoyu Wang, Zhihao Qian, Ye Zhu, Yu Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08247)] \
14 Jun 2023

**Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation** \
*Shuai Yang, Yifan Zhou, Ziwei Liu, Chen Change Loy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07954)] \
13 Jun 2023

**Paste, Inpaint and Harmonize via Denoising: Subject-Driven Image Editing with Pre-Trained Diffusion Model** \
*Xin Zhang, Jiaxian Guo, Paul Yoo, Yutaka Matsuo, Yusuke Iwasawa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07596)] \
13 Jun 2023

**Controlling Text-to-Image Diffusion by Orthogonal Finetuning** \
*Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, Bernhard Schölkopf* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07280)] \
12 Jun 2023


**MovieFactory: Automatic Movie Creation from Text using Large Generative Models for Language and Images** \
*Junchen Zhu, Huan Yang, Huiguo He, Wenjing Wang, Zixi Tuo, Wen-Huang Cheng, Lianli Gao, Jingkuan Song, Jianlong Fu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07257)] \
12 Jun 2023

**InstructP2P: Learning to Edit 3D Point Clouds with Text Instructions** \
*Jiale Xu, Xintao Wang, Yan-Pei Cao, Weihao Cheng, Ying Shan, Shenghua Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07154)] \
12 Jun 2023


**Language-Guided Traffic Simulation via Scene-Level Diffusion** \
*Ziyuan Zhong, Davis Rempe, Yuxiao Chen, Boris Ivanovic, Yulong Cao, Danfei Xu, Marco Pavone, Baishakhi Ray* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.06344)] \
10 Jun 2023

**BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping** \
*Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, Josh Susskind* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05544)] \
8 Jun 2023


**Grounded Text-to-Image Synthesis with Attention Refocusing** \
*Quynh Phung, Songwei Ge, Jia-Bin Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05427)] \
8 Jun 2023

**SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions** \
*Yuseung Lee, Kunho Kim, Hyunjin Kim, Minhyuk Sung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05178)] [[Project](https://syncdiffusion.github.io/)] [[Github](https://github.com/KAIST-Geometric-AI-Group/SyncDiffusion)] \
8 Jun 2023

**Improving Tuning-Free Real Image Editing with Proximal Guidance** \
*Ligong Han, Song Wen, Qi Chen, Zhixing Zhang, Kunpeng Song, Mengwei Ren, Ruijiang Gao, Yuxiao Chen, Di Liu, Qilong Zhangli, Anastasis Stathopoulos, Jindong Jiang, Zhaoyang Xia, Akash Srivastava, Dimitris Metaxas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05414)] \
8 Jun 2023

**WOUAF: Weight Modulation for User Attribution and Fingerprinting in Text-to-Image Diffusion Models** \
*Changhoon Kim, Kyle Min, Maitreya Patel, Sheng Cheng, Yezhou Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04744)] \
7 Jun 2023


**ConceptBed: Evaluating Concept Learning Abilities of Text-to-Image Diffusion Models** \
*Maitreya Patel, Tejas Gokhale, Chitta Baral, Yezhou Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04695)] \
7 Jun 2023

**Designing a Better Asymmetric VQGAN for StableDiffusion** \
*Zixin Zhu, Xuelu Feng, Dongdong Chen, Jianmin Bao, Le Wang, Yinpeng Chen, Lu Yuan, Gang Hua* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04632)] [[Github](https://github.com/buxiangzhiren/Asymmetric_VQGAN)] \
7 Jun 2023

**Multi-modal Latent Diffusion** \
*Mustapha Bounoua, Giulio Franzese, Pietro Michiardi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04445)] \
7 Jun 2023

**Integrating Geometric Control into Text-to-Image Diffusion Models for High-Quality Detection Data Generation via Text Prompt** \
*Kai Chen, Enze Xie, Zhe Chen, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04607)] \
7 Jun 2023

**Improving Diffusion-based Image Translation using Asymmetric Gradient Guidance** \
*Gihyun Kwon, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04396)] \
7 Jun 2023

**Stable Diffusion is Unstable** \
*Chengbin Du, Yanxi Li, Zhongwei Qiu, Chang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02583)] \
5 Jun 2023

**LipVoicer: Generating Speech from Silent Videos Guided by Lip Reading** \
*Yochai Yemini, Aviv Shamsian, Lior Bracha, Sharon Gannot, Ethan Fetaya* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03258)] [[Project](https://lipvoicer.github.io/)] \
5 Jun 2023

**HeadSculpt: Crafting 3D Head Avatars with Text** \
*Xiao Han, Yukang Cao, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, Kwan-Yee K. Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03038)] [[Project](https://brandonhan.uk/HeadSculpt/)] \
5 Jun 2023

**Instruct-Video2Avatar: Video-to-Avatar Generation with Instructions** \
*Shaoxu Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02903)] \
5 Jun 2023

**Towards Unified Text-based Person Retrieval: A Large-scale Multi-Attribute and Language Search Benchmark** \
*Shuyu Yang, Yinan Zhou, Yaxiong Wang, Yujiao Wu, Li Zhu, Zhedong Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02898)] \
5 Jun 2023

**User-friendly Image Editing with Minimal Text Input: Leveraging Captioning and Injection Techniques** \
*Sunwoo Kim, Wooseok Jang, Hyunsu Kim, Junho Kim, Yunjey Choi, Seungryong Kim, Gayeong Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02717)] \
5 Jun 2023

**Detector Guidance for Multi-Object Text-to-Image Generation** \
*Luping Liu, Zijian Zhang, Yi Ren, Rongjie Huang, Xiang Yin, Zhou Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02236)] \
4 Jun 2023

**VideoComposer: Compositional Video Synthesis with Motion Controllability** \
*Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou* \
NeruIPS 2023. [[Paper](https://arxiv.org/abs/2306.02018)] [[Project](https://videocomposer.github.io/)] [[Github](https://github.com/damo-vilab/videocomposer)] \
3 Jun 2023


**Word-Level Explanations for Analyzing Bias in Text-to-Image Models** \
*Alexander Lin, Lucas Monteiro Paes, Sree Harsha Tanneru, Suraj Srinivas, Himabindu Lakkaraju* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05500)] \
3 Jun 2023


**Efficient Text-Guided 3D-Aware Portrait Generation with Score Distillation Sampling on Distribution** \
*Yiji Cheng, Fei Yin, Xiaoke Huang, Xintong Yu, Jiaxiang Liu, Shikun Feng, Yujiu Yang, Yansong Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02083)] \
3 Jun 2023

**Probabilistic Adaptation of Text-to-Video Models** \
*Mengjiao Yang, Yilun Du, Bo Dai, Dale Schuurmans, Joshua B. Tenenbaum, Pieter Abbeel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01872)] [[Project](https://video-adapter.github.io/video-adapter/)] \
2 Jun 2023


**Video Colorization with Pre-trained Text-to-Image Diffusion Models** \
*Hanyuan Liu, Minshan Xie, Jinbo Xing, Chengze Li, Tien-Tsin Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01732)] \
2 Jun 2023


**Audio-Visual Speech Enhancement with Score-Based Generative Models** \
*Julius Richter, Simone Frintrop, Timo Gerkmann* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01432)] \
2 Jun 2023

**Privacy Distillation: Reducing Re-identification Risk of Multimodal Diffusion Models** \
*Virginia Fernandez, Pedro Sanchez, Walter Hugo Lopez Pinaya, Grzegorz Jacenków, Sotirios A. Tsaftaris, Jorge Cardoso* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01322)] \
2 Jun 2023

**StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners** \
*Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, Dilip Krishnan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00984)] \
1 Jun 2023

**Diffusion Self-Guidance for Controllable Image Generation** \
*Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, Aleksander Holynski* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00986)] [[Project](https://dave.ml/selfguidance/)] \
1 Jun 2023


**StyleDrop: Text-to-Image Generation in Any Style** \
*Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, Yuan Hao, Irfan Essa, Michael Rubinstein, Dilip Krishnan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00983)] [[Project](https://styledrop.github.io/)] \
1 Jun 2023


**Intriguing Properties of Text-guided Diffusion Models** \
*Qihao Liu, Adam Kortylewski, Yutong Bai, Song Bai, Alan Yuille* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00974)] \
1 Jun 2023


**Intelligent Grimm -- Open-ended Visual Storytelling via Latent Diffusion Models** \
*Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Weidi Xie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00973)] [[Project](https://haoningwu3639.github.io/StoryGen_Webpage/)] \
1 Jun 2023


**ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation** \
*Shaozhe Hao, Kai Han, Shihao Zhao, Kwan-Yee K. Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00971)] [[Github](https://github.com/haoosz/ViCo)] \
1 Jun 2023

**The Hidden Language of Diffusion Models** \
*Hila Chefer, Oran Lang, Mor Geva, Volodymyr Polosukhin, Assaf Shocher, Michal Irani, Inbar Mosseri, Lior Wolf* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00966)] [[Project](https://hila-chefer.github.io/Conceptor/)] \
1 Jun 2023

**Cocktail: Mixing Multi-Modality Controls for Text-Conditional Image Generation** \
*Minghui Hu, Jianbin Zheng, Daqing Liu, Chuanxia Zheng, Chaoyue Wang, Dacheng Tao, Tat-Jen Cham* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00964)] [[Project](https://mhh0318.github.io/cocktail/)] [[Github](https://github.com/mhh0318/Cocktail)] \
1 Jun 2023

**Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance** \
*Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, Ying Shan, Tien-Tsin Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00943)] [[Project](https://doubiiu.github.io/projects/Make-Your-Video/)] \
1 Jun 2023

**Inserting Anybody in Diffusion Models via Celeb Basis** \
*Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, Huicheng Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00926)] [[Project](https://celeb-basis.github.io/)] \
1 Jun 2023

**Wuerstchen: Efficient Pretraining of Text-to-Image Models** \
*Pablo Pernias, Dominic Rampas, Marc Aubreville* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00637)] \
1 Jun 2023

**UniDiff: Advancing Vision-Language Models with Generative and Discriminative Learning** \
*Xiao Dong, Runhui Huang, Xiaoyong Wei, Zequn Jie, Jianxing Yu, Jian Yin, Xiaodan Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00813)] \
1 Jun 2023

**FigGen: Text to Scientific Figure Generation** \
*Juan A. Rodriguez, David Vazquez, Issam Laradji, Marco Pedersoli, Pau Rodriguez* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2306.00800)] \
1 Jun 2023


**Diffusion Brush: A Latent Diffusion Model-based Editing Tool for AI-generated Images** \
*Peyman Gholami, Robert Xiao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00219)] \
31 May 2023

**Understanding and Mitigating Copying in Diffusion Models** \
*Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2305.20086)] [[Github](https://github.com/somepago/DCR)] \
31 May 2023


**Control4D: Dynamic Portrait Editing by Learning 4D GAN from 2D Diffusion-based Editor** \
*Ruizhi Shao, Jingxiang Sun, Cheng Peng, Zerong Zheng, Boyao Zhou, Hongwen Zhang, Yebin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.20082)] [[Project](https://control4darxiv.github.io/)] \
31 May 2023


**Boosting Text-to-Image Diffusion Models with Fine-Grained Semantic Rewards** \
*Guian Fang, Zutao Jiang, Jianhua Han, Guansong Lu, Hang Xu, Xiaodan Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19599)] [[Github](https://github.com/Enderfga/FineRewards)] \
31 May 2023



**Perturbation-Assisted Sample Synthesis: A Novel Approach for Uncertainty Quantification** \
*Yifei Liu, Rex Shen, Xiaotong Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18671)] \
30 May 2023

**PanoGen: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation** \
*Jialu Li, Mohit Bansal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19195)] [[Project](https://pano-gen.github.io/)] [[Github](https://github.com/jialuli-luka/PanoGen)] \
30 May 2023

**Video ControlNet: Towards Temporally Consistent Synthetic-to-Real Video Translation Using Conditional Image Diffusion Models** \
*Ernie Chu, Shuo-Yen Lin, Jun-Cheng Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19193)] \
30 May 2023

**Nested Diffusion Processes for Anytime Image Generation** \
*Noam Elata, Bahjat Kawar, Tomer Michaeli, Michael Elad* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19066)] \
30 May 2023

**StyleAvatar3D: Leveraging Image-Text Diffusion Models for High-Fidelity 3D Avatar Generation** \
*Chi Zhang, Yiwen Chen, Yijun Fu, Zhenglin Zhou, Gang YU, Billzb Wang, Bin Fu, Tao Chen, Guosheng Lin, Chunhua Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19012)] \
30 May 2023

**HiFA: High-fidelity Text-to-3D with Advanced Diffusion Guidance** \
*Junzhe Zhu, Peiye Zhuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18766)] \
30 May 2023

**LayerDiffusion: Layered Controlled Image Editing with Diffusion Models** \
*Pengzhi Li, QInxuan Huang, Yikang Ding, Zhiheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18676)] \
30 May 2023

**Controllable Text-to-Image Generation with GPT-4** \
*Tianjun Zhang, Yi Zhang, Vibhav Vineet, Neel Joshi, Xin Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18583)] \
29 May 2023

**Cognitively Inspired Cross-Modal Data Generation Using Diffusion Models** \
*Zizhao Hu, Mohammad Rostami* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2305.18433)] \
28 May 2023

**RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths** \
*Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, Ping Luo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18295)] \
29 May 2023

**Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models** \
*Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, Mike Zheng Shou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18292)] [[Project](https://showlab.github.io/Mix-of-Show/)] \
29 May 2023

**Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising** \
*Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, Hongsheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18264)] [[Github](https://github.com/G-U-N/Gen-L-Video)] \
29 May 2023


**Text-Only Image Captioning with Multi-Context Data Generation** \
*Feipeng Ma, Yizhou Zhou, Fengyun Rao, Yueyi Zhang, Xiaoyan Sun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18072)] \
29 May 2023

**InstructEdit: Improving Automatic Masks for Diffusion-based Image Editing With User Instructions** \
*Qian Wang, Biao Zhang, Michael Birsak, Peter Wonka* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18047)] \
29 May 2023


**Conditional Score Guidance for Text-Driven Image-to-Image Translation** \
*Hyunsoo Lee, Minsoo Kang, Bohyung Han* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18007)] \
29 May 2023

**Text-to-image Editing by Image Information Removal** \
*Zhongping Zhang, Jian Zheng, Jacob Zhiyuan Fang, Bryan A. Plummer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17489)] \
27 May 2023

**Towards Consistent Video Editing with Text-to-Image Diffusion Models** \
*Zicheng Zhang, Bonan Li, Xuecheng Nie, Congying Han, Tiande Guo, Luoqi Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17431)] \
27 May 2023


**FISEdit: Accelerating Text-to-image Editing via Cache-enabled Sparse Diffusion Inference** \
*Zihao Yu, Haoyang Li, Fangcheng Fu, Xupeng Miao, Bin Cui* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17423)] \
27 May 2023

**ControlVideo: Adding Conditional Control for One Shot Text-to-Video Editing** \
*Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, Jun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17098)] [[Project](https://ml.cs.tsinghua.edu.cn/controlvideo/)] \
26 May 2023


**Improved Visual Story Generation with Adaptive Context Modeling** \
*Zhangyin Feng, Yuchen Ren, Xinmiao Yu, Xiaocheng Feng, Duyu Tang, Shuming Shi, Bing Qin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16811)] \
26 May 2023


**Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models** \
*Daiki Miyake, Akihiro Iohara, Yu Saito, Toshiyuki Tanaka* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16807)] \
26 May 2023

**Are Diffusion Models Vision-And-Language Reasoners?** \
*Benno Krojer, Elinor Poole-Dayan, Vikram Voleti, Christopher Pal, Siva Reddy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16397)] [[Github](https://github.com/McGill-NLP/diffusion-itm)] \
25 May 2023


**DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models** \
*Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, Kimin Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16381)] \
25 May 2023

**Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models** \
*Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, Kwan-Yee K. Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16322)] [[Project](https://shihaozhaozsh.github.io/unicontrolnet/)] [[Github](https://github.com/ShihaoZhaoZSH/Uni-ControlNet)] \
25 May 2023


**Parallel Sampling of Diffusion Models** \
*Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16317)] [[Github](https://github.com/AndyShih12/paradigms)] \
25 May 2023

**Break-A-Scene: Extracting Multiple Concepts from a Single Image** \
*Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel Cohen-Or, Dani Lischinski* \
SIGGRAPH Asia 2023. [[Paper](https://arxiv.org/abs/2305.16311)] [[Project](https://omriavrahami.com/break-a-scene/)] [[Github](https://github.com/google/break-a-scene)] \
25 May 2023

**Diversify Your Vision Datasets with Automatic Diffusion-Based Augmentation** \
*Lisa Dunlap, Alyssa Umino, Han Zhang, Jiezhi Yang, Joseph E. Gonzalez, Trevor Darrell* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16289)] [[Github](https://github.com/lisadunlap/ALIA)] \
25 May 2023

**Prompt-Free Diffusion: Taking "Text" out of Text-to-Image Diffusion Models** \
*Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, Humphrey Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16223)] [[Github](https://github.com/SHI-Labs/Prompt-Free-Diffusion)] \
25 May 2023 

**ProSpect: Expanded Conditioning for the Personalization of Attribute-aware Image Generation** \
*Yuxin Zhang, Weiming Dong, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Tong-Yee Lee, Oliver Deussen, Changsheng Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16225)] \
25 May 2023

**ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation** \
*Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, Jun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16213)] [[Project](https://ml.cs.tsinghua.edu.cn/prolificdreamer/)] \
25 May 2023

**On Architectural Compression of Text-to-Image Diffusion Models** \
*Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, Shinkook Choi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15798)] \
25 May 2023


**Custom-Edit: Text-Guided Image Editing with Customized Diffusion Models** \
*Jooyoung Choi, Yunjey Choi, Yunji Kim, Junho Kim, Sungroh Yoon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15779)] \
25 May 2023

**MultiFusion: Fusing Pre-Trained Models for Multi-Lingual, Multi-Modal Image Generation** \
*Marco Bellagente, Manuel Brack, Hannah Teufel, Felix Friedrich, Björn Deiseroth, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Koen Oostermeijer, Andres Felipe Cruz-Salinas, Patrick Schramowski, Kristian Kersting, Samuel Weinbach* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15296)] \
24 May 2023

**ChatFace: Chat-Guided Real Face Editing via Diffusion Latent Space Manipulation** \
*Dongxu Yue, Qin Guo, Munan Ning, Jiaxi Cui, Yuesheng Zhu, Li Yuan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14742)] \
24 May 2023

**DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models** \
*Sungnyun Kim, Junsoo Lee, Kibeom Hong, Daesik Kim, Namhyuk Ahn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15194)] [[Github](https://github.com/sungnyun/diffblender)] \
24 May 2023

**I Spy a Metaphor: Large Language Models and Diffusion Models Co-Create Visual Metaphors** \
*Tuhin Chakrabarty, Arkadiy Saakyan, Olivia Winn, Artemis Panagopoulou, Yue Yang, Marianna Apidianaki, Smaranda Muresan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14724)] \
24 May 2023

**BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing** \
*Dongxu Li, Junnan Li, Steven C. H. Hoi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14720)] \
24 May 2023

**Adversarial Nibbler: A Data-Centric Challenge for Improving the Safety of Text-to-Image Models** \
*Alicia Parrish, Hannah Rose Kirk, Jessica Quaye, Charvi Rastogi, Max Bartolo, Oana Inel, Juan Ciro, Rafael Mosquera, Addison Howard, Will Cukierski, D. Sculley, Vijay Janapa Reddi, Lora Aroyo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14384)] \
22 May 2023

**Compositional Text-to-Image Synthesis with Attention Map Control of Diffusion Models** \
*Ruichen Wang, Zekang Chen, Chen Chen, Jian Ma, Haonan Lu, Xiaodong Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13921)] \
23 May 2023

**Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models** \
*Yiting Qu, Xinyue Shen, Xinlei He, Michael Backes, Savvas Zannettou, Yang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13873)] \
23 May 2023


**Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models** \
*Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, Liang Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13840)] \
23 May 2023

**Understanding Text-driven Motion Synthesis with Keyframe Collaboration via Diffusion Models** \
*Dong Wei, Xiaoning Sun, Huaijiang Sun, Bin Li, Shengxiang Hu, Weiqing Li, Jianfeng Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13773)] \
23 May 2023

**LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models** \
*Long Lian, Boyi Li, Adam Yala, Trevor Darrell* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13655)] \
23 May 2023

**LaDI-VTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On** \
*Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, Rita Cucchiara* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13501)] \
22 May 2023



**FACTIFY3M: A Benchmark for Multimodal Fact Verification with Explainability through 5W Question-Answering** \
*Megha Chakraborty, Khusbu Pahwa, Anku Rani, Adarsh Mahor, Aditya Pakala, Arghya Sarkar, Harshit Dave, Ishan Paul, Janvita Reddy, Preethi Gurumurthy, Ritvik G, Samahriti Mukherjee, Shreyas Chatterjee, Kinjal Sensharma, Dwip Dalal, Suryavardan S, Shreyash Mishra, Parth Patwa, Aman Chadha, Amit Sheth, Amitava Das* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05523)] \
22 May 2023

**Training Diffusion Models with Reinforcement Learning** \
*Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13301)] \
22 May 2023


**If at First You Don't Succeed, Try, Try Again: Faithful Diffusion-based Text-to-Image Generation by Selection** \
*Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, Zeynep Akata* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13308)] [[Project](https://rl-diffusion.github.io/)] \
22 May 2023

**ControlVideo: Training-free Controllable Text-to-Video Generation** \
*Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, Qi Tian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13077)] [[Github](https://github.com/YBYBZhang/ControlVideo)] \
22 May 2023

**AudioToken: Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation** \
*Guy Yariv, Itai Gat, Lior Wolf, Yossi Adi, Idan Schwartz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13050)] \
22 May 2023

**The CLIP Model is Secretly an Image-to-Prompt Converter** \
*Yuxuan Ding, Chunna Tian, Haoxuan Ding, Lingqiao Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12716)] \
22 May 2023

**InstructVid2Vid: Controllable Video Editing with Natural Language Instructions** \
*Bosheng Qin, Juncheng Li, Siliang Tang, Tat-Seng Chua, Yueting Zhuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12328)] \
21 May 2023

**SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters** \
*Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12082)] \
20 May 2023

**Late-Constraint Diffusion Guidance for Controllable Image Synthesis** \
*Chang Liu, Dong Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11520)] [[Project](https://alonzoleeeooo.github.io/LCDG/)] [[Github](https://github.com/AlonzoLeeeooo/LCDG)] \
19 May 2023

**Any-to-Any Generation via Composable Diffusion** \
*Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11846)] [[Project](https://codi-gen.github.io/)] [[Github](https://github.com/microsoft/i-Code/tree/main/i-Code-V3)] \
19 May 2023

**Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields** \
*Jingbo Zhang, Xiaoyu Li, Ziyu Wan, Can Wang, Jing Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11588)] \
19 May 2023

**Brain Captioning: Decoding human brain activity into images and text** \
*Matteo Ferrante, Furkan Ozcelik, Tommaso Boccato, Rufin VanRullen, Nicola Toschi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11560)] \
19 May 2023


**Efficient Cross-Lingual Transfer for Chinese Stable Diffusion with Images as Pivots** \
*Jinyi Hu, Xu Han, Xiaoyuan Yi, Yutong Chen, Wenhao Li, Zhiyuan Liu, Maosong Sun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11540)] \
19 May 2023

**Discriminative Diffusion Models as Few-shot Vision and Language Learners** \
*Xuehai He, Weixi Feng, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, William Yang Wang, Xin Eric Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10722)] \
18 May 2023

**Zero-Day Backdoor Attack against Text-to-Image Diffusion Models via Personalization** \
*Yihao Huang, Qing Guo, Felix Juefei-Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10701)] \
18 May 2023


**AIwriting: Relations Between Image Generation and Digital Writing** \
*Scott Rettberg, Talan Memmott, Jill Walker Rettberg, Jason Nelson, Patrick Lichty* \
ISEA 2023. [[Paper](https://arxiv.org/abs/2305.10834)] \
18 May 2023

**TextDiffuser: Diffusion Models as Text Painters** \
*Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, Furu Wei* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10855)] \
18 May 2023

**VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation** \
*Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, Jiaying Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10874)] \
18 May 2023

**LDM3D: Latent Diffusion Model for 3D** \
*Gabriela Ben Melech Stan, Diana Wofk, Scottie Fox, Alex Redden, Will Saxton, Jean Yu, Estelle Aflalo, Shao-Yen Tseng, Fabio Nonato, Matthias Muller, Vasudev Lal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10853)] \
18 May 2023

**X-IQE: eXplainable Image Quality Evaluation for Text-to-Image Generation with Visual Large Language Models** \
*Yixiong Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10843)] [[Github](https://github.com/Schuture/Benchmarking-Awesome-Diffusion-Models)] \
18 May 2023

**Inspecting the Geographical Representativeness of Images from Text-to-Image Models** \
*Abhipsa Basu, R. Venkatesh Babu, Danish Pruthi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11080)] \
18 May 2023

**Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models** \
*Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, Yogesh Balaji* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10474)] [[Project](https://research.nvidia.com/labs/dir/pyoco/)] \
17 May 2023

**AMD: Autoregressive Motion Diffusion** \
*Bo Han, Hao Peng, Minjing Dong, Chang Xu, Yi Ren, Yixuan Shen, Yuheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09381)] \
16 May 2023

**Generating coherent comic with rich story using ChatGPT and Stable Diffusion** \
*Ze Jin, Zorina Song* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11067)] \
16 May 2023



**Make-An-Animation: Large-Scale Text-conditional 3D Human Motion Generation** \
*Samaneh Azadi, Akbar Shah, Thomas Hayes, Devi Parikh, Sonal Gupta* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09662)] [[Project](https://azadis.github.io/make-an-animation/)] \
16 May 2023

**Make-A-Protagonist: Generic Video Editing with An Ensemble of Experts** \
*Yuyang Zhao, Enze Xie, Lanqing Hong, Zhenguo Li, Gim Hee Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08850)] [[Project](https://make-a-protagonist.github.io/)] [[Github](https://github.com/Make-A-Protagonist/Make-A-Protagonist)] \
15 May 2023

**Common Diffusion Noise Schedules and Sample Steps are Flawed** \
*Shanchuan Lin, Bingchen Liu, Jiashi Li, Xiao Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08891)] \
15 May 2023



**Interactive Fashion Content Generation Using LLMs and Latent Diffusion Models** \
*Krishna Sri Ipsit Mantri, Nevasini Sasikumar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05182)] \
15 May 2023


**Null-text Guidance in Diffusion Models is Secretly a Cartoon-style Creator** \
*Jing Zhao, Heliang Zheng, Chaoyue Wang, Long Lan, Wanrong Huang, Wenjing Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06710)] [[Project](https://nulltextforcartoon.github.io/)] [[Github](https://github.com/NullTextforCartoon/NullTextforCartoon)] \
11 May 2023

**iEdit: Localised Text-guided Image Editing with Weak Supervision** \
*Rumeysa Bodur, Erhan Gundogdu, Binod Bhattarai, Tae-Kyun Kim, Michael Donoser, Loris Bazzani* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05947)] \
10 May 2023

**SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models** \
*Shanshan Zhong, Zhongzhan Huang, Wushao Wen, Jinghui Qin, Liang Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05189)] [[Github](https://github.com/Qrange-group/SUR-adapter)] \
9 May 2023

**Style-A-Video: Agile Diffusion for Arbitrary Text-based Video Style Transfer** \
*Nisha Huang, Yuxin Zhang, Weiming Dong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05464)] \
9 May 2023

**DiffuseStyleGesture: Stylized Audio-Driven Co-Speech Gesture Generation with Diffusion Models** \
*Sicheng Yang, Zhiyong Wu, Minglei Li, Zhensong Zhang, Lei Hao, Weihong Bao, Ming Cheng, Long Xiao* \
IJCAI 2023. [[Paper](https://arxiv.org/abs/2305.04919)] [[Github](https://github.com/YoungSeng/DiffuseStyleGesture)] \
8 May 2023

**IIITD-20K: Dense captioning for Text-Image ReID** \
*A V Subramanyam, Niranjan Sundararajan, Vibhu Dubey, Brejesh Lall* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04497)] \
8 May 2023

**ReGeneration Learning of Diffusion Models with Rich Prompts for Zero-Shot Image Translation** \
*Yupei Lin, Sen Zhang, Xiaojun Yang, Xiao Wang, Yukai Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04651)] [[Project](https://yupeilin2388.github.io/publication/ReDiffuser)] \
8 May 2023

**Prompt Tuning Inversion for Text-Driven Image Editing Using Diffusion Models** \
*Wenkai Dong, Song Xue, Xiaoyue Duan, Shumin Han* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04441)] \
8 May 2023


**Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning** \
*Shengfang Zhai, Yinpeng Dong, Qingni Shen, Shi Pu, Yuejian Fang, Hang Su* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04175)] \
7 May 2023


**AADiff: Audio-Aligned Video Synthesis with Text-to-Image Diffusion** \
*Seungwoo Lee, Chaerin Kong, Donghyeon Jeon, Nojun Kwak* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04001)] \
6 May 2023

**Data Curation for Image Captioning with Text-to-Image Generative Models** \
*Wenyan Li, Jonas F. Lotz, Chen Qiu, Desmond Elliott* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03610)] \
5 May 2023

**DisenBooth: Identity-Preserving Disentangled Tuning for Subject-Driven Text-to-Image Generation** \
*Hong Chen, Yipeng Zhang, Xin Wang, Xuguang Duan, Yuwei Zhou, Wenwu Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03374)] [[Project](https://disenbooth.github.io/)] \
5 May 2023

**Guided Image Synthesis via Initial Image Editing in Diffusion Model** \
*Jiafeng Mao, Xueting Wang, Kiyoharu Aizawa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03382)] \
5 May 2023

**Diffusion Explainer: Visual Explanation for Text-to-image Stable Diffusion** \
*Seongmin Lee, Benjamin Hoover, Hendrik Strobelt, Zijie J. Wang, ShengYun Peng, Austin Wright, Kevin Li, Haekyu Park, Haoyang Yang, Duen Horng Chau* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03509)] [[Project](https://poloclub.github.io/diffusion-explainer/)] \
4 May 2023


**Multimodal-driven Talking Face Generation, Face Swapping, Diffusion Model** \
*Chao Xu, Shaoting Zhu, Junwei Zhu, Tianxin Huang, Jiangning Zhang, Ying Tai, Yong Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.02594)] \
4 May 2023


**Multimodal Data Augmentation for Image Captioning using Diffusion Models** \
*Changrong Xiao, Sean Xin Xu, Kunpeng Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01855)] \
3 May 2023

**In-Context Learning Unlocked for Diffusion Models** \
*Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, Mingyuan Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01115)] [[Project](https://zhendong-wang.github.io/prompt-diffusion.github.io/)] [[Github](https://github.com/Zhendong-Wang/Prompt-Diffusion)] \
1 May 2023

**SceneGenie: Scene Graph Guided Diffusion Models for Image Synthesis** \
*Azade Farshad, Yousef Yeganeh, Yu Chi, Chengzhi Shen, Björn Ommer, Nassir Navab* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.14573)] \
28 Apr 2023

**It is all about where you start: Text-to-image generation with seed selection** \
*Dvir Samuel, Rami Ben-Ari, Simon Raviv, Nir Darshan, Gal Chechik* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.14530)] \
27 Apr 2023

**Edit Everything: A Text-Guided Generative System for Images Editing** \
*Defeng Xie, Ruichen Wang, Jian Ma, Chen Chen, Haonan Lu, Dong Yang, Fobo Shi, Xiaodong Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.14006)] [[Github](https://github.com/DefengXie/Edit_Everything)] \
27 Apr 2023

**Training-Free Location-Aware Text-to-Image Synthesis** \
*Jiafeng Mao, Xueting Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13427)] \
26 Apr 2023

**TextMesh: Generation of Realistic 3D Meshes From Text Prompts** \
*Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, Federico Tombari* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12439)] \
24 Apr 2023

**Using Text-to-Image Generation for Architectural Design Ideation** \
*Ville Paananen, Jonas Oppenlaender, Aku Visuri* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.10182)] \
20 Apr 2023


**Anything-3D: Towards Single-view Anything Reconstruction in the Wild** \
*Qiuhong Shen, Xingyi Yang, Xinchao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.10261)] [[Github](https://github.com/Anything-of-anything/Anything-3D)] \
19 Apr 2023


**UPGPT: Universal Diffusion Model for Person Image Generation, Editing and Pose Transfer** \
*Soon Yau Cheong, Armin Mustafa, Andrew Gilbert* \
ICCV Workshop 2023. [[Paper](https://arxiv.org/abs/2304.08870)] [[Github](https://github.com/soon-yau/upgpt)] \
18 Apr 2023

**TTIDA: Controllable Generative Data Augmentation via Text-to-Text and Text-to-Image Models** \
*Yuwei Yin, Jean Kaddour, Xiang Zhang, Yixin Nie, Zhenguang Liu, Lingpeng Kong, Qi Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08821)] \
18 Apr 2023

**Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models** \
*Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, Karsten Kreis* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.08818)] [[Project](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)] \
18 Apr 2023

**Text2Performer: Text-Driven Human Video Generation** \
*Yuming Jiang, Shuai Yang, Tong Liang Koh, Wayne Wu, Chen Change Loy, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08483)] [[Project](https://yumingj.github.io/projects/Text2Performer.html)] \
17 Apr 2023

**Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation** \
*Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, Xi Yin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08477)] [[Project](https://latent-shift.github.io/)] \
17 Apr 2023

**MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing** \
*Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, Yinqiang Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08465)] [[Github](https://github.com/TencentARC/MasaCtrl)] \
17 Apr 2023

**Text-Conditional Contextualized Avatars For Zero-Shot Personalization** \
*Samaneh Azadi, Thomas Hayes, Akbar Shah, Guan Pang, Devi Parikh, Sonal Gupta* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.07410)] \
14 Apr 2023

**Delta Denoising Score** \
*Amir Hertz, Kfir Aberman, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.07090)] [[Project](https://delta-denoising-score.github.io/)] \
14 Apr 2023

**Expressive Text-to-Image Generation with Rich Text** \
*Songwei Ge, Taesung Park, Jun-Yan Zhu, Jia-Bin Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06720)] [[Project](https://rich-text-to-image.github.io/)] [[Github](https://github.com/SongweiGe/rich-text-to-image)] \
13 Apr 2023



**Soundini: Sound-Guided Diffusion for Natural Video Editing** \
*Seung Hyun Lee, Sieun Kim, Innfarn Yoo, Feng Yang, Donghyeon Cho, Youngseo Kim, Huiwen Chang, Jinkyu Kim, Sangpil Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06818)] [[Project](https://kuai-lab.github.io/soundini-gallery/)] \
13 Apr 2023



**Improving Diffusion Models for Scene Text Editing with Dual Encoders** \
*Jiabao Ji, Guanhua Zhang, Zhaowen Wang, Bairu Hou, Zhifei Zhang, Brian Price, Shiyu Chang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05568)] [[Github](https://github.com/UCSB-NLP-Chang/DiffSTE)] \
12 Apr 2023

**An Edit Friendly DDPM Noise Space: Inversion and Manipulations** \
*Inbar Huberman-Spiegelglas, Vladimir Kulikov, Tomer Michaeli* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06140)] \
12 Apr 2023

**Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA** \
*James Seale Smith, Yen-Chang Hsu, Lingyu Zhang, Ting Hua, Zsolt Kira, Yilin Shen, Hongxia Jin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06027)] [[Project](https://jamessealesmith.github.io/continual-diffusion/)] \
12 Apr 2023

**HRS-Bench: Holistic, Reliable and Scalable Benchmark for Text-to-Image Models** \
*Eslam Mohamed Bakr, Pengzhan Sun, Xiaoqian Shen, Faizan Farooq Khan, Li Erran Li, Mohamed Elhoseiny* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05390)] [[Project](https://eslambakr.github.io/hrsbench.github.io/)] \
11 Apr 2023

**Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond** \
*Mohammadreza Armandpour, Huangjie Zheng, Ali Sadeghian, Amir Sadeghian, Mingyuan Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04968)] \
11 Apr 2023

**Towards Real-time Text-driven Image Manipulation with Unconditional Diffusion Models** \
*Nikita Starodubcev, Dmitry Baranchuk, Valentin Khrulkov, Artem Babenko* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04344)] \
10 Apr 2023

**HumanSD: A Native Skeleton-Guided Diffusion Model for Human Image Generation** \
*Xuan Ju, Ailing Zeng, Chenchen Zhao, Jianan Wang, Lei Zhang, Qiang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04269)] [[Github](https://idea-research.github.io/HumanSD/)] \
9 Apr 2023

**Harnessing the Spatial-Temporal Attention of Diffusion Models for High-Fidelity Text-to-Image Synthesis** \
*Qiucheng Wu, Yujian Liu, Handong Zhao, Trung Bui, Zhe Lin, Yang Zhang, Shiyu Chang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03869)] [[Github](https://github.com/UCSB-NLP-Chang/Diffusion-SpaceTime-Attn)] \
7 Apr 2023

**Zero-shot Generative Model Adaptation via Image-specific Prompt Learning** \
*Jiayi Guo, Chaofei Wang, You Wu, Eric Zhang, Kai Wang, Xingqian Xu, Shiji Song, Humphrey Shi, Gao Huang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.03119)] [[Github](https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation)] \
6 Apr 2023

**Training-Free Layout Control with Cross-Attention Guidance** \
*Minghao Chen, Iro Laina, Andrea Vedaldi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03373)] [[Project](https://silent-chen.github.io/layout-guidance/)] [[Github](https://github.com/silent-chen/layout-guidance)] \
6 Apr 2023


**Benchmarking Robustness to Text-Guided Corruptions** \
*Mohammadreza Mofayezi, Yasamin Medghalchi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02963)] \
6 Apr 2023


**DITTO-NeRF: Diffusion-based Iterative Text To Omni-directional 3D Model** \
*Hoigi Seo, Hayeon Kim, Gwanghyun Kim, Se Young Chun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02827)] [[Project](https://janeyeon.github.io/ditto-nerf/)] \
6 Apr 2023



**Taming Encoder for Zero Fine-tuning Image Customization with Text-to-Image Diffusion Models** \
*Xuhui Jia, Yang Zhao, Kelvin C.K. Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, Yu-Chuan Su* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02642)] \
5 Apr 2023


**A Diffusion-based Method for Multi-turn Compositional Image Generation** \
*Chao Wang, Xiaoyu Yang, Jinmiao Huang, Kevin Ferreira* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02192)] \
5 Apr 2023

**viz2viz: Prompt-driven stylized visualization generation using a diffusion model** \
*Jiaqi Wu, John Joon Young Chung, Eytan Adar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01919)] \
4 Apr 2023

**Multimodal Garment Designer: Human-Centric Latent Diffusion Models for Fashion Image Editing** \
*Alberto Baldrati, Davide Morelli, Giuseppe Cartella, Marcella Cornia, Marco Bertini, Rita Cucchiara* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02051)] \
4 Apr 2023

**PODIA-3D: Domain Adaptation of 3D Generative Model Across Large Domain Gap Using Pose-Preserved Text-to-Image Diffusion** \
*Gwanghyun Kim, Ji Ha Jang, Se Young Chun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01900)] [[Project](https://gwang-kim.github.io/podia_3d/)] \
4 Apr 2023

**Text-Conditioned Sampling Framework for Text-to-Image Generation with Masked Generative Models** \
*Jaewoong Lee, Sangwon Jang, Jaehyeong Jo, Jaehong Yoon, Yunji Kim, Jin-Hwa Kim, Jung-Woo Ha, Sung Ju Hwang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01515)] \
4 Apr 2023


**ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model** \
*Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01116)] [[Project](https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html)] [[Github](https://github.com/mingyuan-zhang/ReMoDiffuse)] \
3 Apr 2023

**DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models** \
*Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, Kwan-Yee K. Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.00916)] \
3 Apr 2023

**DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance** \
*Longwen Zhang, Qiwei Qiu, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, Jingyi Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03117)] [[Project](https://sites.google.com/view/dreamface)] \
1 Apr 2023

**GlyphDraw: Learning to Draw Chinese Characters in Image Synthesis Models Coherently** \
*Jian Ma, Mingjun Zhao, Chen Chen, Ruichen Wang, Di Niu, Haonan Lu, Xiaodong Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17870)] [[Project](https://1073521013.github.io/glyph-draw.github.io/)] \
31 Mar 2023

**AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control** \
*Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17606)] [[Project](https://avatar-craft.github.io/)] [[Github](https://github.com/songrise/avatarcraft)] \
30 Mar 2023

**PAIR-Diffusion: Object-Level Image Editing with Structure-and-Appearance Paired Diffusion Models** \
*Vidit Goel, Elia Peruzzo, Yifan Jiang, Dejia Xu, Nicu Sebe, Trevor Darrell, Zhangyang Wang, Humphrey Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17546)] [[Github](https://github.com/Picsart-AI-Research/PAIR-Diffusion)] \
30 Mar 2023

**Social Biases through the Text-to-Image Generation Lens** \
*Ranjita Naik, Besmira Nushi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06034)] \
30 Mar 2023


**Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models** \
*Eric Zhang, Kai Wang, Xingqian Xu, Zhangyang Wang, Humphrey Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17591)] [[Github](https://github.com/SHI-Labs/Forget-Me-Not)] \
30 Mar 2023

**DiffCollage: Parallel Generation of Large Content with Diffusion Models** \
*Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, Ming-Yu Liu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.17076)] [[Project](https://research.nvidia.com/labs/dir/diffcollage/)] \
30 Mar 2023

**Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models** \
*Wen Wang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, Chunhua Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17599)] \
30 Mar 2023


**Discriminative Class Tokens for Text-to-Image Diffusion Models** \
*Idan Schwartz, Vésteinn Snæbjarnarson, Sagie Benaim, Hila Chefer, Ryan Cotterell, Lior Wolf, Serge Belongie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17155)] \
30 Mar 2023



**DAE-Talker: High Fidelity Speech-Driven Talking Face Generation with Diffusion Autoencoder** \
*Chenpng Du, Qi Chen, Tianyu He, Xu Tan, Xie Chen, Kai Yu, Sheng Zhao, Jiang Bian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17550)] \
30 Mar 2023

**LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation** \
*Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, Xi Li* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.17189)] [[Github](https://github.com/ZGCTroy/LayoutDiffusion)] \
30 Mar 2023


**4D Facial Expression Diffusion Model** \
*Kaifeng Zou, Sylvain Faisan, Boyang Yu, Sébastien Valette, Hyewon Seo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16611)] [[Github](https://github.com/ZOUKaifeng/4DFM)] \
29 Mar 2023

**MDP: A Generalized Framework for Text-Guided Image Editing by Manipulating the Diffusion Path** \
*Qian Wang, Biao Zhang, Michael Birsak, Peter Wonka* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16765)] [[Github](https://github.com/QianWangX/MDP-Diffusion)] \
29 Mar 2023



**Instruct 3D-to-3D: Text Instruction Guided 3D-to-3D conversion** \
*Hiromichi Kamata, Yuiko Sakuma, Akio Hayakawa, Masato Ishii, Takuya Narihira* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15780)] [[Github](https://sony.github.io/Instruct3Dto3D-doc/)] \
28 Mar 2023

**StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing** \
*Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, Jian Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15649)] \
28 Mar 2023

**Seer: Language Instructed Video Prediction with Latent Diffusion Models** \
*Xianfan Gu, Chuan Wen, Jiaming Song, Yang Gao* \
CVPR Workshop 2023. [[Paper](https://arxiv.org/abs/2303.14897)] \
27 Mar 2023


**Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation** \
*Susung Hong, Donghoon Ahn, Seungryong Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15413)] \
27 Mar 2023

**Anti-DreamBooth: Protecting users from personalized text-to-image synthesis** \
*Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc Tran, Anh Tran* \
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2303.15433)] [[Github](https://github.com/VinAIResearch/Anti-DreamBooth)] \
27 Mar 2023

**GestureDiffuCLIP: Gesture Diffusion Model with CLIP Latents** \
*Tenglong Ao, Zeyi Zhang, Libin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14613)] \
26 Mar 2023

**Better Aligning Text-to-Image Models with Human Preference** \
*Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, Hongsheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14420)] [[Github](https://tgxs002.github.io/align_sd_web/)] \
25 Mar 2023

**ISS++: Image as Stepping Stone for Text-Guided 3D Shape Generation** \
*Zhengzhe Liu, Peng Dai, Ruihui Li, Xiaojuan Qi, Chi-Wing Fu* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2303.15181)] \
24 Mar 2023

**DiffuScene: Scene Graph Denoising Diffusion Probabilistic Model for Generative Indoor Scene Synthesis** \
*Jiapeng Tang, Yinyu Nie, Lev Markhasin, Angela Dai, Justus Thies, Matthias Nießner* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14207)] [[Project](https://tangjiapeng.github.io/projects/DiffuScene/)] \
24 Mar 2023

**CompoNeRF: Text-guided Multi-object Compositional NeRF with Editable 3D Scene Layout** \
*Yiqi Lin, Haotian Bai, Sijia Li, Haonan Lu, Xiaodong Lin, Hui Xiong, Lin Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13843)] [[Project](https://fantasia3d.github.io/)] \
24 Mar 2023

**Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation** \
*Rui Chen, Yongwei Chen, Ningxin Jiao, Kui Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13873)] \
24 Mar 2023

**ReVersion: Diffusion-Based Relation Inversion from Images** \
*Ziqi Huang, Tianxing Wu, Yuming Jiang, Kelvin C.K. Chan, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13495)] [[Project](https://ziqihuangg.github.io/projects/reversion.html)] [[Github](https://github.com/ziqihuangg/ReVersion)]
23 Mar 2023

**Ablating Concepts in Text-to-Image Diffusion Models** \
*Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, Jun-Yan Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13516)] [[Project](https://www.cs.cmu.edu/~concept-ablation/)] [[Github](https://github.com/nupurkmr9/concept-ablation)] \
23 Mar 2023

**Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators** \
*Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, Humphrey Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13439)] [[Github](https://github.com/Picsart-AI-Research/Text2Video-Zero)] \
23 Mar 2023

**MagicFusion: Boosting Text-to-Image Generation Performance by Fusing Diffusion Models** \
*Jing Zhao, Heliang Zheng, Chaoyue Wang, Long Lan, Wenjing Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13126)] [[Project](https://magicfusion.github.io/)] [[Github](https://github.com/MagicFusion/MagicFusion.github.io)] \
23 Mar 2023

**Pix2Video: Video Editing using Image Diffusion** \
*Duygu Ceylan, Chun-Hao Paul Huang, Niloy J. Mitra* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12688)] [[Project](https://duyguceylan.github.io/pix2video.github.io/)] \
22 Mar 2023

**Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions** \
*Ayaan Haque, Matthew Tancik, Alexei A. Efros, Aleksander Holynski, Angjoo Kanazawa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12789)] [[Project](https://instruct-nerf2nerf.github.io/)] \
22 Mar 2023

**SALAD: Part-Level Latent Diffusion for 3D Shape Generation and Manipulation** \
*Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, Minhyuk Sung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12236)] [[Project](https://salad3d.github.io/)] \
21 Mar 2023

**Vox-E: Text-guided Voxel Editing of 3D Objects** \
*Etai Sella, Gal Fiebelman, Peter Hedman, Hadar Averbuch-Elor* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12048)] [[Project](https://tau-vailab.github.io/Vox-E/)] \
21 Mar 2023

**CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion** \
*Geonmo Gu, Sanghyuk Chun, Wonjae Kim, HeeJae Jun, Yoohoon Kang, Sangdoo Yun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11916)] \
21 Mar 2023


**3D-CLFusion: Fast Text-to-3D Rendering with Contrastive Latent Diffusion** \
*Yu-Jhe Li, Kris Kitani* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11938)] \
21 Mar 2023

**Text2Tex: Text-driven Texture Synthesis via Diffusion Models** \
*Dave Zhenyu Chen, Yawar Siddiqui, Hsin-Ying Lee, Sergey Tulyakov, Matthias Nießner* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11396)] [[Project](https://daveredrum.github.io/Text2Tex/)] \
20 Mar 2023

**Localizing Object-level Shape Variations with Text-to-Image Diffusion Models** \
*Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11306)] [[Project](https://orpatashnik.github.io/local-prompt-mixing/)] \
20 Mar 2023

**SVDiff: Compact Parameter Space for Diffusion Fine-Tuning** \
*Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, Feng Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11305)] \
20 Mar 2023

**Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models** \
*René Haas, Inbar Huberman-Spiegelglas, Rotem Mulayoff, Tomer Michaeli* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11073)] \
20 Mar 2023

**SKED: Sketch-guided Text-based 3D Editing** \
*Aryan Mikaeili, Or Perel, Daniel Cohen-Or, Ali Mahdavi-Amiri* \
arxiv 2023. [[Paper](https://arxiv.org/abs/2303.10735)] \
19 Mar 2023

**DialogPaint: A Dialog-based Image Editing Model** \
*Jingxuan Wei, Shiyu Wu, Xin Jiang, Yequan Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10073)] \
17 Mar 2023

**GlueGen: Plug and Play Multi-modal Encoders for X-to-image Generation** \
*Can Qin, Ning Yu, Chen Xing, Shu Zhang, Zeyuan Chen, Stefano Ermon, Yun Fu, Caiming Xiong, Ran Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10056)] \
17 Mar 2023

**DiffusionRet: Generative Text-Video Retrieval with Diffusion Model** \
*Peng Jin, Hao Li, Zesen Cheng, Kehan Li, Xiangyang Ji, Chang Liu, Li Yuan, Jie Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09867)] \
17 Mar 2023

**FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model** \
*Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, Jian Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09833)] [[Github](https://github.com/vvictoryuki/FreeDoM)] \
17 Mar 2023

**Unified Multi-Modal Latent Diffusion for Joint Subject and Text Conditional Image Generation** \
*Yiyang Ma, Huan Yang, Wenjing Wang, Jianlong Fu, Jiaying Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09319)] \
16 Mar 2023

**FateZero: Fusing Attentions for Zero-shot Text-based Video Editing** \
*Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, Qifeng Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09535)] [[Project](https://fate-zero-edit.github.io/)] [[Github](https://github.com/ChenyangQiQi/FateZero)] \
16 Mar 2023


**HIVE: Harnessing Human Feedback for Instructional Visual Editing** \
*Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, Caiming Xiong, Ran Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09618)] \
16 Mar 2023


**P+: Extended Textual Conditioning in Text-to-Image Generation** \
*Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, Kfir Aberman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09522)] [[Project](https://prompt-plus.github.io/)] \
16 Mar 2023

**Highly Personalized Text Embedding for Image Manipulation by Stable Diffusion** \
*Inhwa Han, Serin Yang, Taesung Kwon, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08767)] \
15 Mar 2023

**Aerial Diffusion: Text Guided Ground-to-Aerial View Translation from a Single Image using Diffusion Models** \
*Divya Kothandaraman, Tianyi Zhou, Ming Lin, Dinesh Manocha* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11444)] [[Github](https://github.com/divyakraman/AerialDiffusion)] \
15 Mar 2023


**Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer** \
*Serin Yang, Hyunmin Hwang, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08622)] \
15 Mar 2023

**Edit-A-Video: Single Video Editing with Object-Aware Consistency** \
*Chaehun Shin, Heeseung Kim, Che Hyun Lee, Sang-gil Lee, Sungroh Yoon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07945)] [[Project](https://edit-a-video.github.io/)] \
14 Mar 2023

**Editing Implicit Assumptions in Text-to-Image Diffusion Models** \
*Hadas Orgad, Bahjat Kawar, Yonatan Belinkov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08084)] [[Project](https://time-diffusion.github.io/)] [[Github](https://github.com/bahjat-kawar/time-diffusion)] \
14 Mar 2023



**Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation** \
*Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, Seungryong Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07937)] \
14 Mar 2023

**Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models** \
*Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, Nan Duan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04671)] [[Github](https://github.com/microsoft/visual-chatgpt)] \
8 Mar 2023

**Video-P2P: Video Editing with Cross-attention Control** \
*Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, Jiaya Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04761)] [[Project](https://video-p2p.github.io/)] \
8 Mar 2023

**Erasing Concepts from Diffusion Models** \
*Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, David Bau* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07345)] [[Project](https://erasing.baulab.info/)] [[Github](https://github.com/rohitgandikota/erasing)] \
13 Mar 2023


**One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale** \
*Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, Jun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06555)] [[Github](https://github.com/thu-ml/unidiffuser)] \
12 Mar 2023

**Cones: Concept Neurons in Diffusion Models for Customized Generation** \
*Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05125)] \
9 Mar 2023

**A Prompt Log Analysis of Text-to-Image Generation Systems** \
*Yutong Xie, Zhaoying Pan, Jinge Ma, Jie Luo, Qiaozhu Mei* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04587)] \
8 Mar 2023

**Zeroth-Order Optimization Meets Human Feedback: Provable Learning via Ranking Oracles** \
*Zhiwei Tang, Dmitry Rybin, Tsung-Hui Chang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03751)] [[Github](https://github.com/TZW1998/Taming-Stable-Diffusion-with-Human-Ranking-Feedback)] \
7 Mar 2023


**Unleashing Text-to-Image Diffusion Models for Visual Perception** \
*Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, Jiwen Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.02153)] [[Github](https://github.com/wl-zhao/VPD)] \
3 Mar 2023

**Collage Diffusion** \
*Vishnu Sarukkai, Linden Li, Arden Ma, Christopher Ré, Kayvon Fatahalian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.00262)] \
1 Mar 2023

**Towards Enhanced Controllability of Diffusion Models** \
*Wonwoong Cho, Hareesh Ravi, Midhun Harikumar, Vinh Khuc, Krishna Kumar Singh, Jingwan Lu, David I. Inouye, Ajinkya Kale* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.14368)] \
28 Feb 2023

**Directed Diffusion: Direct Control of Object Placement through Attention Guidance** \
*Wan-Duo Kurt Ma, J.P. Lewis, W. Bastiaan Kleijn, Thomas Leung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13153)] \
25 Feb 2023

**Modulating Pretrained Diffusion Models for Multimodal Image Synthesis** \
*Cusuh Ham, James Hays, Jingwan Lu, Krishna Kumar Singh, Zhifei Zhang, Tobias Hinz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12764)] \
24 Feb 2023

**Region-Aware Diffusion for Zero-shot Text-driven Image Editing** \
*Nisha Huang, Fan Tang, Weiming Dong, Tong-Yee Lee, Changsheng Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.11797)] [[Github](https://github.com/haha-lisa/RDM-Region-Aware-Diffusion-Model)] \
23 Feb 2023

**Controlled and Conditional Text to Image Generation with Diffusion Prior** \
*Pranav Aggarwal, Hareesh Ravi, Naveen Marri, Sachin Kelkar, Fengbin Chen, Vinh Khuc, Midhun Harikumar, Ritiz Tambi, Sudharshan Reddy Kakumanu, Purvak Lapsiya, Alvin Ghouas, Sarah Saber, Malavika Ramprasad, Baldo Faieta, Ajinkya Kale* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.11710)] \
23 Feb 2023

**Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC** \
*Yilun Du, Conor Durkan, Robin Strudel, Joshua B. Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, Will Grathwohl* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.11552)] [[Project](https://energy-based-model.github.io/reduce-reuse-recycle/)] \
22 Feb 2023

**Learning 3D Photography Videos via Self-supervised Diffusion on Single Images** \
*Xiaodong Wang, Chenfei Wu, Shengming Yin, Minheng Ni, Jianfeng Wang, Linjie Li, Zhengyuan Yang, Fan Yang, Lijuan Wang, Zicheng Liu, Yuejian Fang, Nan Duan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10781)] \
21 Feb 2023


**Exploring the Representation Manifolds of Stable Diffusion Through the Lens of Intrinsic Dimension** \
*Henry Kvinge, Davis Brown, Charles Godfrey* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.09301)] \
16 Feb 2023

**Text-driven Visual Synthesis with Latent Diffusion Prior** \
*Ting-Hsuan Liao, Songwei Ge, Yiran Xu, Yao-Chih Lee, Badour AlBahar, Jia-Bin Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08510)] [[Project](https://latent-diffusion-prior.github.io/)] \
16 Feb 2023

**T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models** \
*Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, Xiaohu Qie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08453)] [[Github](https://github.com/TencentARC/T2I-Adapter)] \
16 Feb 2023

**MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation** \
*Omer Bar-Tal, Lior Yariv, Yaron Lipman, Tali Dekel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08113)] [Project](https://multidiffusion.github.io/)] [[Github](https://github.com/omerbt/MultiDiffusion)] \
16 Feb 2023

**Boundary Guided Mixing Trajectory for Semantic Control with Diffusion Models** \
*Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, Yan Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08357)] \
16 Feb 2023

**Dataset Interfaces: Diagnosing Model Failures Using Controllable Counterfactual Generation** \
*Joshua Vendrow, Saachi Jain, Logan Engstrom, Aleksander Madry* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07865)] [[Github](https://github.com/MadryLab/dataset-interfaces)] \
15 Feb 2023

**PRedItOR: Text Guided Image Editing with Diffusion Prior**\
*Hareesh Ravi, Sachin Kelkar, Midhun Harikumar, Ajinkya Kale* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07979)] \
15 Feb 2023

**Text-Guided Scene Sketch-to-Photo Synthesis** \
*AprilPyone MaungMaung, Makoto Shing, Kentaro Mitsui, Kei Sawada, Fumio Okura* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.06883)] \
14 Feb 2023

**Universal Guidance for Diffusion Models** \
*Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, Tom Goldstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07121)] [[Github](https://github.com/arpitbansal297/Universal-Guided-Diffusion)] \
14 Feb 2023

**Adding Conditional Control to Text-to-Image Diffusion Models** \
*Lvmin Zhang, Maneesh Agrawala* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05543)] [[Github](https://github.com/lllyasviel/ControlNet)] \
10 Feb 2023


**Analyzing Multimodal Objectives Through the Lens of Generative Diffusion Guidance** \
*Chaerin Kong, Nojun Kwak* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10305)] \
10 Feb 2023

**Is This Loss Informative? Speeding Up Textual Inversion with Deterministic Objective Evaluation** \
*Anton Voronov, Mikhail Khoroshikh, Artem Babenko, Max Ryabinin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04841)] \
9 Feb 2023


**Q-Diffusion: Quantizing Diffusion Models** \
*Xiuyu Li, Long Lian, Yijiang Liu, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, Kurt Keutzer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04304)] [[Github](https://github.com/Xiuyu-Li/q-diffusion)] \
8 Feb 2023


**GLAZE: Protecting Artists from Style Mimicry by Text-to-Image Models** \
*Shawn Shan, Jenna Cryan, Emily Wenger, Haitao Zheng, Rana Hanocka, Ben Y. Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04222)] \
8 Feb 2023

**Zero-shot Generation of Coherent Storybook from Plain Text Story using Diffusion Models** \
*Hyeonho Jeong, Gihyun Kwon, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03900)] \
8 Feb 2023

**Fair Diffusion: Instructing Text-to-Image Generation Models on Fairness** \
*Felix Friedrich, Patrick Schramowski, Manuel Brack, Lukas Struppek, Dominik Hintersdorf, Sasha Luccioni, Kristian Kersting* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10893)] \
7 Feb 2023

**Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery** \
*Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, Tom Goldstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03668)] [[Github](https://github.com/YuxinWenRick/hard-prompts-made-easy)] \
7 Feb 2023

**Zero-shot Image-to-Image Translation** \
*Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, Jun-Yan Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03027)] \
6 Feb 2023

**Structure and Content-Guided Video Synthesis with Diffusion Models** \
*Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, Anastasis Germanidis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03011)] [[Project](https://research.runwayml.com/gen1)] \
6 Feb 2023

**Mixture of Diffusers for scene composition and high resolution image generation** \
*Álvaro Barbero Jiménez* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02412)] [[Github](https://github.com/albarji/mixture-of-diffusers)] \
5 Feb 2023

**ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval** \
*Kexun Zhang, Xianjun Yang, William Yang Wang, Lei Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02285)] \
5 Feb 2023

**Eliminating Prior Bias for Semantic Image Editing via Dual-Cycle Diffusion** \
*Zuopeng Yang, Tianshu Chu, Xin Lin, Erdun Gao, Daqing Liu, Jie Yang, Chaoyue Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02394)] \
5 Feb 2023

**Semantic-Guided Image Augmentation with Pre-trained Models** \
*Bohan Li, Xinghao Wang, Xiao Xu, Yutai Hou, Yunlong Feng, Feng Wang, Wanxiang Che* \
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2302.02070)] [[Project](https://texturepaper.github.io/TEXTurePaper/)] \
4 Feb 2023


**TEXTure: Text-Guided Texturing of 3D Shapes** \
*Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01721)] [[Project](https://texturepaper.github.io/TEXTurePaper/)] [[Github](https://github.com/TEXTurePaper/TEXTurePaper)] \
3 Feb 2023

**Dreamix: Video Diffusion Models are General Video Editors** \
*Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, Yedid Hoshen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01329)] [[Project](https://dreamix-video-editing.github.io/)] \
2 Feb 2023

**Trash to Treasure: Using text-to-image models to inform the design of physical artefacts** \
*Amy Smith, Hope Schroeder, Ziv Epstein, Michael Cook, Simon Colton, Andrew Lippman* \
AAAI 2023. [[Paper](https://arxiv.org/abs/2302.00561)] \
1 Feb 2023

**Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models** \
*Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, Daniel Cohen-Or* \
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2301.13826)] [[Project](https://attendandexcite.github.io/Attend-and-Excite/)] [[Github](https://github.com/AttendAndExcite/Attend-and-Excite)] \
31 Jan 2023

**Zero3D: Semantic-Driven Multi-Category 3D Shape Generation** \
*Bo Han, Yitong Liu, Yixuan Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13591)] \
31 Jan 2023

**Shape-aware Text-driven Layered Video Editing** \
*Yao-Chih Lee, Ji-Ze Genevieve Jang, Yi-Ting Chen, Elizabeth Qiu, Jia-Bin Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13173)] [[Project](https://text-video-edit.github.io/)] \
30 Jan 2023

**PromptMix: Text-to-image diffusion models enhance the performance of lightweight networks** \
*Arian Bakhtiarnia, Qi Zhang, Alexandros Iosifidis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12914)] [[Github](https://gitlab.au.dk/maleci/promptmix)] \
30 Jan 2023

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis** \
*Ming Tao, Bing-Kun Bao, Hao Tang, Changsheng Xu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2301.12959)] [[Github](https://github.com/tobran/GALIP)] \
30 Jan 2023


**SEGA: Instructing Diffusion using Semantic Dimensions** \
*Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, Kristian Kersting* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12247)] \
28 Jan 2023

**Towards Equitable Representation in Text-to-Image Synthesis Models with the Cross-Cultural Understanding Benchmark (CCUB) Dataset** \
*Zhixuan Liu, Youeun Shin, Beverley-Claire Okogwu, Youngsik Yun, Lia Coleman, Peter Schaldenbrand, Jihie Kim, Jean Oh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12073)] \
28 Jan 2023

**Text-To-4D Dynamic Scene Generation** \
*Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, Yaniv Taigman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11280)] \
26 Jan 2023

**Guiding Text-to-Image Diffusion Model Towards Grounded Generation** \
*Ziyi Li, Qinye Zhou, Xiaoyun Zhang, Ya Zhang, Yanfeng Wang, Weidi Xie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.05221)] [[Project](https://lipurple.github.io/Grounded_Diffusion/)] \
12 Jan 2023



**Speech Driven Video Editing via an Audio-Conditioned Diffusion Model** \
*Dan Bigioi, Shubhajit Basak, Hugh Jordan, Rachel McDonnell, Peter Corcoran* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.04474)] [[Project](https://danbigioi.github.io/DiffusionVideoEditing/)] [[Github](https://github.com/DanBigioi/DiffusionVideoEditing)] \
10 Jan 2023

**Visual Story Generation Based on Emotion and Keywords** \
*Yuetian Chen, Ruohua Li, Bowen Shi, Peiru Liu, Mei Si* \
AIIDE INT 2022. [[Paper](https://arxiv.org/abs/2301.02777)] \
7 Jan 2023


**DiffTalk: Crafting Diffusion Models for Generalized Talking Head Synthesis** \
*Shuai Shen, Wenliang Zhao, Zibin Meng, Wanhua Li, Zheng Zhu, Jie Zhou, Jiwen Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.03786)] \
10 Jan 2023

**Speech Driven Video Editing via an Audio-Conditioned Diffusion Model** \
*Dan Bigioi, Shubhajit Basak, Hugh Jordan, Rachel McDonnell, Peter Corcoran* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.04474)] \
10 Jan 2023

**Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation** \
*Michał Stypułkowski, Konstantinos Vougioukas, Sen He, Maciej Zięba, Stavros Petridis, Maja Pantic* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.03396)] [[Project](https://mstypulkowski.github.io/diffusedheads/)] \
6 Jan 2023


**Muse: Text-To-Image Generation via Masked Generative Transformers** \
*Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, Dilip Krishnan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.00704)] [[Project](https://muse-model.github.io/)] \
2 Jan 2023

**Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models** \
*Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, Shenghua Gao* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.14704)] [[Project](https://bluestyle97.github.io/dream3d/)] \
28 Dec 2022

**Exploring Vision Transformers as Diffusion Learners** \
*He Cao, Jianan Wang, Tianhe Ren, Xianbiao Qi, Yihao Chen, Yuan Yao, Lei Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.13771)] \
28 Dec 2022

**Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation** \
*Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, Mike Zheng Shou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.11565)] [[Project](https://tuneavideo.github.io/)] \
22 Dec 2022

**Contrastive Language-Vision AI Models Pretrained on Web-Scraped Multimodal Data Exhibit Sexual Objectification Bias** \
*Robert Wolfe, Yiwei Yang, Bill Howe, Aylin Caliskan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.11261)] \
21 Dec 2022

**Optimizing Prompts for Text-to-Image Generation** \
*Yaru Hao, Zewen Chi, Li Dong, Furu Wei* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09611)] [[Project](https://huggingface.co/spaces/microsoft/Promptist)] [[Github](https://github.com/microsoft/LMOps/tree/main/promptist)] \
19 Dec 2022

**Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models** \
*Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, Shiyu Chang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08698)] [[Github](https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement)] \
16 Dec 2022

**TeTIm-Eval: a novel curated evaluation data set for comparing text-to-image models** \
*Federico A. Galatolo, Mario G. C. A. Cimino, Edoardo Cogotti* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.07839)] \
15 Dec 2022

**The Infinite Index: Information Retrieval on Generative Text-To-Image Models** \
*Niklas Deckers, Maik Fröbe, Johannes Kiesel, Gianluca Pandolfo, Christopher Schröder, Benno Stein, Martin Potthast* \
CHIIR 2023. [[Paper](https://arxiv.org/abs/2212.07476)] \
14 Dec 2022


**LidarCLIP or: How I Learned to Talk to Point Clouds** \
*Georg Hess, Adam Tonderski, Christoffer Petersson, Lennart Svensson, Kalle Åström* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06858)] [[Github](https://github.com/atonderski/lidarclip)] \
13 Dec 2022

**Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting** \
*Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J. Fleet, Radu Soricut, Jason Baldridge, Mohammad Norouzi, Peter Anderson, William Chan* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.06909)] \
13 Dec 2022

**The Stable Artist: Steering Semantics in Diffusion Latent Space** \
*Manuel Brack, Patrick Schramowski, Felix Friedrich, Dominik Hintersdorf, Kristian Kersting* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06013)] \
12 Dec 2022

**SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model** \
*Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, Kun Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.05034)] \
9 Dec 2022

**Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis** \
*Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, William Yang Wang* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2212.05032)] [[Github](https://github.com/weixi-feng/Structured-Diffusion-Guidance)] \
9 Dec 2022

**MoFusion: A Framework for Denoising-Diffusion-based Motion Synthesis** \
*Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, Christian Theobalt* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04495)] [[Project](https://vcai.mpi-inf.mpg.de/projects/MoFusion/)] \
8 Dec 2022


**SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation** \
*Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander Schwing, Liangyan Gui* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04493)] [[Project](https://yccyenchicheng.github.io/SDFusion/)] \
8 Dec 2022


**SINE: SINgle Image Editing with Text-to-Image Diffusion Models** \
*Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris Metaxas, Jian Ren* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04489)] [[Project](https://zhang-zx.github.io/SINE/)] [[Github](https://github.com/zhang-zx/SINE)] \
8 Dec 2022

**Multi-Concept Customization of Text-to-Image Diffusion** \
*Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04488)] [[Project](https://www.cs.cmu.edu/~custom-diffusion/)] \
8 Dec 2022


**Diffusion Guided Domain Adaptation of Image Generators** \
*Kunpeng Song, Ligong Han, Bingchen Liu, Dimitris Metaxas, Ahmed Elgammal* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04473)] [[Project](https://styleganfusion.github.io/)] \
8 Dec 2022

**Executing your Commands via Motion Diffusion in Latent Space** \
*Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, Gang Yu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04048)] [[Project](https://chenxin.tech/mld/)] \
8 Dec 2022



**Talking Head Generation with Probabilistic Audio-to-Visual Diffusion Priors** \
*Zhentao Yu, Zixin Yin, Deyu Zhou, Duomin Wang, Finn Wong, Baoyuan Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04248)] [[Project](https://zxyin.github.io/TH-PAD/)] \
7 Dec 2022

**Magic: Multi Art Genre Intelligent Choreography Dataset and Network for 3D Dance Generation** \
*Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Xiu Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03741)] \
7 Dec 2022

**Judge, Localize, and Edit: Ensuring Visual Commonsense Morality for Text-to-Image Generation** \
*Seongbeom Park, Suhong Moon, Jinkyu Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03507)] \
7 Dec 2022


**NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors** \
*Congyue Deng, Chiyu "Max'' Jiang, Charles R. Qi, Xinchen Yan, Yin Zhou, Leonidas Guibas, Dragomir Anguelov* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03267)] \
6 Dec 2022


**Semantic-Conditional Diffusion Networks for Image Captioning** \
*Jianjie Luo, Yehao Li, Yingwei Pan, Ting Yao, Jianlin Feng, Hongyang Chao, Tao Mei* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.03099)] [[Github](https://github.com/YehLi/xmodaler/tree/master/configs/image_caption/scdnet)] \
6 Dec 2022

**Diffusion-SDF: Text-to-Shape via Voxelized Diffusion** \
*Muheng Li, Yueqi Duan, Jie Zhou, Jiwen Lu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.03293)] [[Project](https://ttlmh.github.io/DiffusionSDF/)] [[Github](https://github.com/ttlmh/Diffusion-SDF)] \
6 Dec 2022



**ADIR: Adaptive Diffusion for Image Reconstruction** \
*Shady Abu-Hussein, Tom Tirer, Raja Giryes* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03221)] [[Project](https://shadyabh.github.io/ADIR/)] \
6 Dec 2022

**M-VADER: A Model for Diffusion with Multimodal Context** \
*Samuel Weinbach, Marco Bellagente, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Björn Deiseroth, Koen Oostermeijer, Hannah Teufel, Andres Felipe Cruz-Salinas* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02936)] \
6 Dec 2022

**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding** \
*Gyeongman Kim, Hajin Shim, Hyunsu Kim, Yunjey Choi, Junho Kim, Eunho Yang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.02802)] [[Project](https://diff-video-ae.github.io/)] [[Github](https://github.com/man805/Diffusion-Video-Autoencoders)] \
6 Dec 2022


**Unite and Conquer: Cross Dataset Multimodal Synthesis using Diffusion Models** \
*Nithin Gopalakrishnan Nair, Wele Gedara Chaminda Bandara, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00793)] [[Project](https://nithin-gk.github.io/projectpages/Multidiff/index.html)] \
1 Dec 2022

**Shape-Guided Diffusion with Inside-Outside Attention** \
*Dong Huk Park, Grace Luo, Clayton Toste, Samaneh Azadi, Xihui Liu, Maka Karalashvili, Anna Rohrbach, Trevor Darrell* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00210)] [[Project](https://shape-guided-diffusion.github.io/)] \
1 Dec 2022


**SinDDM: A Single Image Denoising Diffusion Model** \
*Vladimir Kulikov, Shahar Yadin, Matan Kleiner, Tomer Michaeli* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16582)] [[Project](https://matankleiner.github.io/sinddm/)] \
29 Nov 2022

**DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model** \
*Gwanghyun Kim, Se Young Chun* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.16374)] [[Github](https://datid-3d.github.io/)] \
29 Nov 2022


**Refined Semantic Enhancement towards Frequency Diffusion for Video Captioning** \
*Xian Zhong, Zipeng Li, Shuqin Chen, Kui Jiang, Chen Chen, Mang Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.15076)] [[Github](https://github.com/lzp870/RSFD)] \
28 Nov 2022

**Unified Discrete Diffusion for Simultaneous Vision-Language Generation** \
*Minghui Hu, Chuanxia Zheng, Heliang Zheng, Tat-Jen Cham, Chaoyue Wang, Zuopeng Yang, Dacheng Tao, Ponnuthurai N. Suganthan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.14842)] \
27 Nov 2022


**3DDesigner: Towards Photorealistic 3D Object Generation and Editing with Text-guided Diffusion Models** \
*Gang Li, Heliang Zheng, Chaoyue Wang, Chang Li, Changwen Zheng, Dacheng Tao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.14108)] \
25 Nov 2022


**SpaText: Spatio-Textual Representation for Controllable Image Generation** \
*Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried, Xi Yin* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.14305)] [[Project](https://omriavrahami.com/spatext/)] \
25 Nov 2022

**Sketch-Guided Text-to-Image Diffusion Models** \
*Andrey Voynov, Kfir Aberman, Daniel Cohen-Or* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13752)] [[Project](https://sketch-guided-diffusion.github.io/)] \
24 Nov 2022

**Shifted Diffusion for Text-to-image Generation** \
*Yufan Zhou, Bingchen Liu, Yizhe Zhu, Xiao Yang, Changyou Chen, Jinhui Xu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.15388)] \
24 Nov 2022


**Make-A-Story: Visual Memory Conditioned Consistent Story Generation** \
*Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, Leonid Sigal* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.13319)] \
23 Nov 2022



**Schrödinger's Bat: Diffusion Models Sometimes Generate Polysemous Words in Superposition** \
*Jennifer C. White, Ryan Cotterell* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13095)] \
23 Nov 2022

**EDICT: Exact Diffusion Inversion via Coupled Transformations** \
*Bram Wallace, Akash Gokul, Nikhil Naik* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12446)] [[Github](https://github.com/salesforce/EDICT)] \
22 Nov 2022


**Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation** \
*Narek Tumanyan, Michal Geyer, Shai Bagon, Tali Dekel* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.12572)] [[Github](https://github.com/MichalGeyer/plug-and-play)] \
22 Nov 2022

**Human Evaluation of Text-to-Image Models on a Multi-Task Benchmark** \
*Vitali Petsiuk, Alexander E. Siemenn, Saisamrit Surbehera, Zad Chin, Keith Tyser, Gregory Hunter, Arvind Raghavan, Yann Hicke, Bryan A. Plummer, Ori Kerret, Tonio Buonassisi, Kate Saenko, Armando Solar-Lezama, Iddo Drori* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2211.12112)] \
22 Nov 2022

**SinDiffusion: Learning a Diffusion Model from a Single Natural Image** \
*Weilun Wang, Jianmin Bao, Wengang Zhou, Dongdong Chen, Dong Chen, Lu Yuan, Houqiang Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12445)] [[Github](https://github.com/WeilunWang/SinDiffusion)] \
22 Nov 2022

**SinFusion: Training Diffusion Models on a Single Image or Video** \
*Yaniv Nikankin, Niv Haim, Michal Irani* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11743)] [[Github](https://yanivnik.github.io/sinfusion/)] \
21 Nov 2022

**Exploring Discrete Diffusion Models for Image Captioning** \
*Zixin Zhu, Yixuan Wei, Jianfeng Wang, Zhe Gan, Zheng Zhang, Le Wang, Gang Hua, Lijuan Wang, Zicheng Liu, Han Hu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11694)] [[Github](https://github.com/buxiangzhiren/DDCap)] \
21 Nov 2022

**Investigating Prompt Engineering in Diffusion Models** \
*Sam Witteveen, Martin Andrews* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2211.15462)] \
21 Nov 2022

**VectorFusion: Text-to-SVG by Abstracting Pixel-Based Diffusion Models** \
*Ajay Jain, Amber Xie, Pieter Abbeel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11319)] [[Project](https://ajayj.com/vectorfusion)] \
21 Nov 2022



**Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models** \
*Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, Wenhu Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10950)] [[Github](https://github.com/xichenpan/ARLDM)] \
20 Nov 2022

**DiffStyler: Controllable Dual Diffusion for Text-Driven Image Stylization** \
*Nisha Huang, Yuxin Zhang, Fan Tang, Chongyang Ma, Haibin Huang, Yong Zhang, Weiming Dong, Changsheng Xu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10682)] \
19 Nov 2022

**Magic3D: High-Resolution Text-to-3D Content Creation** \
*Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, Tsung-Yi Lin* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.10440)] [[Project](https://deepimagination.cc/Magic3D/)] \
18 Nov 2022

**Invariant Learning via Diffusion Dreamed Distribution Shifts** \
*Priyatham Kattakinda, Alexander Levine, Soheil Feizi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10370)] \
18 Nov 2022

**Null-text Inversion for Editing Real Images using Guided Diffusion Models**\
*Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, Daniel Cohen-Or* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09794)] \
17 Nov 2022

**InstructPix2Pix: Learning to Follow Image Editing Instructions** \
*Tim Brooks, Aleksander Holynski, Alexei A. Efros* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.09800)] [[Project](https://www.timothybrooks.com/instruct-pix2pix)] [[Github](https://github.com/timothybrooks/instruct-pix2pix)] \
17 Nov 2022


**Versatile Diffusion: Text, Images and Variations All in One Diffusion Model** \
*Xingqian Xu, Zhangyang Wang, Eric Zhang, Kai Wang, Humphrey Shi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.08332)] [[Github](https://github.com/SHI-Labs/Versatile-Diffusion)] \
15 Nov 2022

**Direct Inversion: Optimization-Free Text-Driven Real Image Editing with Diffusion Models** \
*Adham Elarabawy, Harish Kamath, Samuel Denton* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.07825)] \
15 Nov 2022


**Arbitrary Style Guidance for Enhanced Diffusion-Based Text-to-Image Generation** \
*Zhihong Pan, Xin Zhou, Hao Tian* \
WACV 2023. [[Paper](https://arxiv.org/abs/2211.07751)] \
14 Nov 2022


**Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models** \
*Patrick Schramowski, Manuel Brack, Björn Deiseroth, Kristian Kersting* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.05105)] [[Github](https://github.com/ml-research/safe-latent-diffusion)] \
9 Nov 2022

**Rickrolling the Artist: Injecting Invisible Backdoors into Text-Guided Image Generation Models** \
*Lukas Struppek, Dominik Hintersdorf, Kristian Kersting* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.02408)] [[Github](https://github.com/LukasStruppek/Rickrolling-the-Artist)] \
4 Nov 2022

**eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers** \
*Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, Ming-Yu Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01324)] [[Github](https://deepimagination.cc/eDiffi/)] \
2 Nov 2022


**MagicMix: Semantic Mixing with Diffusion Models** \
*Jun Hao Liew, Hanshu Yan, Daquan Zhou, Jiashi Feng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.16056)] [[Project](https://magicmix.github.io/)] \
28 Oct 2022

**UPainting: Unified Text-to-Image Diffusion Generation with Cross-modal Guidance** \
*Wei Li, Xue Xu, Xinyan Xiao, Jiachen Liu, Hu Yang, Guohao Li, Zhanpeng Wang, Zhifan Feng, Qiaoqiao She, Yajuan Lyu, Hua Wu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.16031)] \
28 Oct 2022

**How well can Text-to-Image Generative Models understand Ethical Natural Language Interventions?** \
*Hritik Bansal, Da Yin, Masoud Monajatipoor, Kai-Wei Chang* \
EMNLP 2022. [[Paper](https://arxiv.org/abs/2210.15230)] [[Github](https://github.com/Hritikbansal/entigen_emnlp)] \
27 Oct 2022

**ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts** \
*Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2210.15257)] \
27 Oct 2022

**DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models** \
*Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, Duen Horng Chau* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.14896)] [[Project](https://poloclub.github.io/diffusiondb/)] [[Github](https://github.com/poloclub/diffusiondb)] \
26 Oct 2022

**Lafite2: Few-shot Text-to-Image Generation** \
*Yufan Zhou, Chunyuan Li, Changyou Chen, Jianfeng Gao, Jinhui Xu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.14124)] \
25 Oct 2022

**High-Resolution Image Editing via Multi-Stage Blended Diffusion** \
*Johannes Ackermann, Minjun Li* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2210.12965)] [[Github](https://github.com/pfnet-research/multi-stage-blended-diffusion)] \
24 Oct 2022

**Conditional Diffusion with Less Explicit Guidance via Model Predictive Control** \
*Max W. Shen, Ehsan Hajiramezanali, Gabriele Scalia, Alex Tseng, Nathaniel Diamant, Tommaso Biancalani, Andreas Loukas* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.12192)] \
21 Oct 2022

**A Visual Tour Of Current Challenges In Multimodal Language Models** \
*Shashank Sonkar, Naiming Liu, Richard G. Baraniuk* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.12565)] \
22 Oct 2022

**DiffEdit: Diffusion-based semantic image editing with mask guidance** \
*Guillaume Couairon, Jakob Verbeek, Holger Schwenk, Matthieu Cord* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.11427)] \
20 Oct 2022

**Diffusion Models already have a Semantic Latent Space** \
*Mingi Kwon, Jaeseok Jeong, Youngjung Uh* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.10960)] [[Project](https://kwonminki.github.io/Asyrp/)] \
20 Oct 2022


**UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image** \
*Dani Valevski, Matan Kalman, Yossi Matias, Yaniv Leviathan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.09477)] \
18 Oct 2022

**Swinv2-Imagen: Hierarchical Vision Transformer Diffusion Models for Text-to-Image Generation** \
*Ruijun Li, Weihua Li, Yi Yang, Hanyu Wei, Jianhua Jiang, Quan Bai* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.09549)] \
18 Oct 2022

**Imagic: Text-Based Real Image Editing with Diffusion Models** \
*Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, Michal Irani* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2210.09276)] [[Project](https://imagic-editing.github.io/)] \
17 Oct 2022

**Leveraging Off-the-shelf Diffusion Model for Multi-attribute Fashion Image Manipulation** \
*Chaerin Kong, DongHyeon Jeon, Ohjoon Kwon, Nojun Kwak* \
WACV 2022. [[Paper](https://arxiv.org/abs/2210.05872)] \
12 Oct 2022

**Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance** \
*Chen Henry Wu, Fernando De la Torre* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.05559)] [[Github-1](https://github.com/ChenWu98/cycle-diffusion)] [[Github-2](https://github.com/ChenWu98/unified-generative-zoo)] \
11 Oct 2022

**Imagen Video: High Definition Video Generation with Diffusion Models** \
*Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.02303)] \
5 Oct 2022


**DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics** \
*Ivan Kapelyukh, Vitalis Vosylius, Edward Johns* \
IEEE RA-L 2022. [[Paper](https://arxiv.org/abs/2210.02438)] \
5 Oct 2022


**LDEdit: Towards Generalized Text Guided Image Manipulation via Latent Diffusion Models** \
*Paramanand Chandramouli, Kanchana Vaishnavi Gandikota* \
BMVC 2022. [[Paper](https://arxiv.org/abs/2210.02249)] \
5 Oct 2022

**clip2latent: Text driven sampling of a pre-trained StyleGAN using denoising diffusion and CLIP** \
*Justin N. M. Pinkney, Chuan Li* \
BMVC 2022. [[Paper](https://arxiv.org/abs/2210.02347)] [[Github](https://github.com/justinpinkney/clip2latent)] \
5 Oct 2022

**Membership Inference Attacks Against Text-to-image Generation Models** \
*Yixin Wu, Ning Yu, Zheng Li, Michael Backes, Yang Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.00968)] \
3 Oct 2022

**Make-A-Video: Text-to-Video Generation without Text-Video Data** \
*Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, Yaniv Taigman* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14792)] \
29 Sep 2022

**DreamFusion: Text-to-3D using 2D Diffusion** \
*Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14988)] [[Github](https://dreamfusion3d.github.io/)] \
29 Sep 2022

**Re-Imagen: Retrieval-Augmented Text-to-Image Generator** \
*Wenhu Chen, Hexiang Hu, Chitwan Saharia, William W. Cohen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14491)] \
29 Sep 2022

**Creative Painting with Latent Diffusion Models** \
*Xianchao Wu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14697)] \
29 Sep 2022

**Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion** \
*Nisha Huang, Fan Tang, Weiming Dong, Changsheng Xu* \
ACM MM 2022. [[Paper](https://arxiv.org/abs/2209.13360)] [[Github](https://github.com/haha-lisa/MGAD-multimodal-guided-artwork-diffusion)] \
27 Sep 2022

**Personalizing Text-to-Image Generation via Aesthetic Gradients** \
*Victor Gallego* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2209.12330)] [[Github](https://github.com/vicgalle/stable-diffusion-aesthetic-gradients)] \
25 Sep 2022

**Best Prompts for Text-to-Image Models and How to Find Them** \
*Nikita Pavlichenko, Dmitry Ustalov* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2209.11711)] \
23 Sep 2022

**The Biased Artist: Exploiting Cultural Biases via Homoglyphs in Text-Guided Image Generation Models** \
*Lukas Struppek, Dominik Hintersdorf, Kristian Kersting* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.08891)]  [[Github](https://github.com/LukasStruppek/The-Biased-Artist)] \
19 Sep 2022

**Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models** \
*Chen Henry Wu, Saman Motamed, Shaunak Srivastava, Fernando De la Torre* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2209.06970)] [[Github](https://github.com/ChenWu98/Generative-Visual-Prompt)] \
14 Sep 2022



**ISS: Image as Stepping Stone for Text-Guided 3D Shape Generation** \
*Zhengzhe Liu, Peng Dai, Ruihui Li, Xiaojuan Qi, Chi-Wing Fu* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2209.04145)] [[Github](https://github.com/liuzhengzhe/ISS-Image-as-Stepping-Stone-for-Text-Guided-3D-Shape-Generation)] \
9 Sep 2022

**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation** \
*Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2208.12242)] [[Project](https://dreambooth.github.io/)] [[Github](https://github.com/Victarry/stable-dreambooth)] \
25 Aug 2022


**Text-Guided Synthesis of Artistic Images with Retrieval-Augmented Diffusion Models** \
*Robin Rombach, Andreas Blattmann, Björn Ommer* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.13038)] [[Github](https://github.com/CompVis/latent-diffusion)] \
26 Jul 2022

**Discrete Contrastive Diffusion for Cross-Modal and Conditional Generation** \
*Ye Zhu, Yu Wu, Kyle Olszewski, Jian Ren, Sergey Tulyakov, Yan Yan* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2206.07771)] [[Github](https://github.com/L-YeZhu/CDCD)] \
15 Jun 2022

**Blended Latent Diffusion** \
*Omri Avrahami, Ohad Fried, Dani Lischinski* \
ACM 2022. [[Paper](https://arxiv.org/abs/2206.02779)] [[Project](https://omriavrahami.com/blended-latent-diffusion-page/)] [[Github](https://github.com/omriav/blended-latent-diffusion)] \
6 Jun 2022

**Compositional Visual Generation with Composable Diffusion Models** \
*Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, Joshua B. Tenenbaum* \
ECCV 2022. [[Paper](https://arxiv.org/abs/2206.01714)] [[Project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)] [[Github](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch)] \
3 Jun 2022

**DiVAE: Photorealistic Images Synthesis with Denoising Diffusion Decoder** \
*Jie Shi, Chenfei Wu, Jian Liang, Xiang Liu, Nan Duan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00386)] \
1 Jun 2022


**Improved Vector Quantized Diffusion Models** \
*Zhicong Tang, Shuyang Gu, Jianmin Bao, Dong Chen, Fang Wen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.16007)] [[Github](https://github.com/microsoft/VQ-Diffusion)] \
31 May 2022

**Text2Human: Text-Driven Controllable Human Image Generation** \
*Yuming Jiang, Shuai Yang, Haonan Qiu, Wayne Wu, Chen Change Loy, Ziwei Liu* \
ACM 2022. [[Paper](https://arxiv.org/abs/2205.15996)] [[Github](https://github.com/yumingj/Text2Human)] \
31 May 2022

**Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding** \
*Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.11487)] [[Github](https://github.com/lucidrains/imagen-pytorch)]  \
23 May 2022


**Retrieval-Augmented Diffusion Models** \
*Andreas Blattmann, Robin Rombach, Kaan Oktay, Björn Ommer* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2204.11824)] [[Github](https://github.com/lucidrains/retrieval-augmented-ddpm)] \
25 Apr 2022


**Hierarchical Text-Conditional Image Generation with CLIP Latents** \
*Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.06125)] [[Github](https://github.com/lucidrains/DALLE2-pytorch)] \
13 Apr 2022


**KNN-Diffusion: Image Generation via Large-Scale Retrieval** \
*Oron Ashual, Shelly Sheynin, Adam Polyak, Uriel Singer, Oran Gafni, Eliya Nachmani, Yaniv Taigman* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2204.02849)] \
6 Apr 2022

**High-Resolution Image Synthesis with Latent Diffusion Models** \
*Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2112.10752)] [[Github](https://github.com/CompVis/latent-diffusion)] \
20 Dec 2021


**More Control for Free! Image Synthesis with Semantic Diffusion Guidance** \
*Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, Trevor Darrell* \
WACV 2021. [[Paper](https://arxiv.org/abs/2112.05744)] [[Project](https://xh-liu.github.io/sdg/)] \
10 Dec 2021

**Vector Quantized Diffusion Model for Text-to-Image Synthesis** \
*Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, Baining Guo* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.14822)] [[Github](https://github.com/microsoft/VQ-Diffusion)] \
29 Nov 2021

**Blended Diffusion for Text-driven Editing of Natural Images** \
*Omri Avrahami, Dani Lischinski, Ohad Fried* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.14818)] [[Project](https://omriavrahami.com/blended-diffusion-page/)] [[Github](https://github.com/omriav/blended-diffusion)] \
29 Nov 2021

**Tackling the Generative Learning Trilemma with Denoising Diffusion GANs** \
*Zhisheng Xiao, Karsten Kreis, Arash Vahdat* \
ICLR 2022 (Spotlight). [[Paper](https://arxiv.org/abs/2112.07804)] [[Project](https://nvlabs.github.io/denoising-diffusion-gan)] \
15 Dec 2021

**DiffusionCLIP: Text-guided Image Manipulation Using Diffusion Models** \
*Gwanghyun Kim, Jong Chul Ye* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2110.02711)] [[Github](https://github.com/gwang-kim/DiffusionCLIP)] \
6 Oct 2021


### 3D Vision

**Text-to-3D with Classifier Score Distillation** \
*Xin Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Song-Hai Zhang, Xiaojuan Qi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19415)] \
30 Oct 2023

**Controllable Group Choreography using Contrastive Diffusion** \
*Nhat Le, Tuong Do, Khoa Do, Hien Nguyen, Erman Tjiputra, Quang D. Tran, Anh Nguyen* \
ACM ToG 2023. [[Paper](https://arxiv.org/abs/2310.18986)] \
29 Oct 2023

**SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation** \
*Haobo Jiang, Mathieu Salzmann, Zheng Dang, Jin Xie, Jian Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17359)] \
26 Oct 2023

**6-DoF Stability Field via Diffusion Models** \
*Takuma Yoneda, Tianchong Jiang, Gregory Shakhnarovich, Matthew R. Walter* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.17649)] \
26 Oct 2023


**DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior** \
*Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, Yebin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16818)] \
25 Oct 2023


**DiffRef3D: A Diffusion-based Proposal Refinement Framework for 3D Object Detection** \
*Se-Ho Kim, Inyong Koo, Inyoung Lee, Byeongjun Park, Changick Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.16349)] \
25 Oct 2023

**iNVS: Repurposing Diffusion Inpainters for Novel View Synthesis** \
*Yash Kant, Aliaksandr Siarohin, Michael Vasilkovsky, Riza Alp Guler, Jian Ren, Sergey Tulyakov, Igor Gilitschenski* \
SIGGRAPH ASIA 2023. [[Paper](https://arxiv.org/abs/2310.16167)] [[Project](https://yashkant.github.io/invs/)] \
24 Oct 2023

**Wonder3D: Single Image to 3D using Cross-Domain Diffusion** \
*Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, Wenping Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.15008)] \
23 Oct 2023

**MAS: Multi-view Ancestral Sampling for 3D motion generation using 2D diffusion** \
*Roy Kapon, Guy Tevet, Daniel Cohen-Or, Amit H. Bermano* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.14729)] \
23 Oct 2023

**High-Quality 3D Face Reconstruction with Affine Convolutional Networks** \
*Zhiqian Lin, Jiangke Lin, Lincheng Li, Yi Yuan, Zhengxia Zou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.14237)] \
22 Oct 2023

**TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models** \
*Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, Kangxue Yin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.13772)] \
20 Oct 2023

**Conditional Generative Modeling for Images, 3D Animations, and Video** \
*Vikram Voleti* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.13157)] \
19 Oct 2023

**TapMo: Shape-aware Motion Generation of Skeleton-free Characters** \
*Jiaxu Zhang, Shaoli Huang, Zhigang Tu, Xin Chen, Xiaohang Zhan, Gang Yu, Ying Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12678)] \
19 Oct 2023


**Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping** \
*Zijie Pan, Jiachen Lu, Xiatian Zhu, Li Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12474)] \
19 Oct 2023

**Progressive3D: Progressively Local Editing for Text-to-3D Content Creation with Complex Semantic Prompts** \
*Xinhua Cheng, Tianyu Yang, Jianan Wang, Yu Li, Lei Zhang, Jian Zhang, Li Yuan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11784)] \
18 Oct 2023

**3D Structure-guided Network for Tooth Alignment in 2D Photograph** \
*Yulong Dou, Lanzhuju Mei, Dinggang Shen, Zhiming Cui* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.11106)] \
17 Oct 2023

**DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing** \
*Jia-Wei Liu, Yan-Pei Cao, Jay Zhangjie Wu, Weijia Mao, Yuchao Gu, Rui Zhao, Jussi Keppo, Ying Shan, Mike Zheng Shou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10624)] \
16 Oct 2023

**ConsistNet: Enforcing 3D Consistency for Multi-view Images Diffusion** \
*Jiayu Yang, Ziang Cheng, Yunfei Duan, Pan Ji, Hongdong Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10343)] \
16 Oct 2023

**PaintHuman: Towards High-fidelity Text-to-3D Human Texturing via Denoised Score Distillation** \
*Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, Wayne Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.09458)] \
14 Oct 2023

**OmniControl: Control Any Joint at Any Time for Human Motion Generation** \
*Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, Huaizu Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08580)] [[Project](https://neu-vi.github.io/omnicontrol/)] \
12 Oct 2023

**Consistent123: Improve Consistency for One Image to 3D Object Synthesis** \
*Haohan Weng, Tianyu Yang, Jianan Wang, Yu Li, Tong Zhang, C. L. Philip Chen, Lei Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.08092)] [[Project](https://consistent-123.github.io/)] \
12 Oct 2023

**What Does Stable Diffusion Know about the 3D Scene?** \
*Guanqi Zhan, Chuanxia Zheng, Weidi Xie, Andrew Zisserman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06836)] \
10 Oct 2023

**HiFi-123: Towards High-fidelity One Image to 3D Content Generation** \
*Wangbo Yu, Li Yuan, Yan-Pei Cao, Xiangjun Gao, Xiaoyu Li, Long Quan, Ying Shan, Yonghong Tian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06744)] \
10 Oct 2023

**IPDreamer: Appearance-Controllable 3D Object Generation with Image Prompts** \
*Bohan Zeng, Shanglin Li, Yutang Feng, Hong Li, Sicheng Gao, Jiaming Liu, Huaxia Li, Xu Tang, Jianzhuang Liu, Baochang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05375)] \
9 Oct 2023

**DragD3D: Vertex-based Editing for Realistic Mesh Deformations using 2D Diffusion Priors** \
*Tianhao Xie, Eugene Belilovsky, Sudhir Mudur, Tiberiu Popa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.04561)] \
6 Oct 2023

**Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints** \
*Chuan Fang, Xiaotao Hu, Kunming Luo, Ping Tan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03602)] \
5 Oct 2023

**FreeReg: Image-to-Point Cloud Registration Leveraging Pretrained Diffusion Models and Monocular Depth Estimators** \
*Haiping Wang, Yuan Liu, Bing Wang, Yujing Sun, Zhen Dong, Wenping Wang, Bisheng Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03420)] \
5 Oct 2023

**Consistent-1-to-3: Consistent Image to 3D View Synthesis via Geometry-aware Diffusion Models** \
*Jianglong Ye, Peng Wang, Kejie Li, Yichun Shi, Heng Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03020)] [[Project](https://jianglongye.com/consistent123/)] \
4 Oct 2023


**Efficient-3DiM: Learning a Generalizable Single-image Novel-view Synthesizer in One Day** \
*Yifan Jiang, Hao Tang, Jen-Hao Rick Chang, Liangchen Song, Zhangyang Wang, Liangliang Cao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.03015)] \
4 Oct 2023


**T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation** \
*Yuze He, Yushi Bai, Matthieu Lin, Wang Zhao, Yubin Hu, Jenny Sheng, Ran Yi, Juanzi Li, Yong-Jin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02977)] [[Project](https://t3bench.com/)] [[Github](https://github.com/THU-LYJ-Lab/T3Bench)] \
4 Oct 2023


**ED-NeRF: Efficient Text-Guided Editing of 3D Scene using Latent Space NeRF** \
*Jangho Park, Gihyun Kwon, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02712)] \
4 Oct 2023

**MagicDrive: Street View Generation with Diverse 3D Geometry Control** \
*Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, Qiang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02601)] [[Project](https://gaoruiyuan.com/magicdrive/)] \
4 Oct 2023


**SweetDreamer: Aligning Geometric Priors in 2D Diffusion for Consistent Text-to-3D** \
*Weiyu Li, Rui Chen, Xuelin Chen, Ping Tan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02596)] [[Project](https://sweetdreamer3d.github.io/)] \
4 Oct 2023

**Hierarchical Generation of Human-Object Interactions with Diffusion Probabilistic Models** \
*Huaijin Pi, Sida Peng, Minghui Yang, Xiaowei Zhou, Hujun Bao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.02242)] [[Project](https://zju3dv.github.io/hghoi/)] [[Github](https://github.com/zju3dv/hghoi)] \
3 Oct 2023


**HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation** \
*Xin Huang, Ruizhi Shao, Qi Zhang, Hongwen Zhang, Ying Feng, Yebin Liu, Qing Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.01406)] [[Project](https://humannorm.github.io/)] \
2 Oct 2023

**Diffusion Posterior Illumination for Ambiguity-aware Inverse Rendering** \
*Linjie Lyu, Ayush Tewari, Marc Habermann, Shunsuke Saito, Michael Zollhöfer, Thomas Leimkühler, Christian Theobalt* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00362)] \
30 Sep 2023

**EPiC-ly Fast Particle Cloud Generation with Flow-Matching and Diffusion** \
*Erik Buhmann, Cedric Ewen, Darius A. Faroughy, Tobias Golling, Gregor Kasieczka, Matthew Leigh, Guillaume Quétant, John Andrew Raine, Debajyoti Sengupta, David Shih* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.00049)] \
29 Sep 2023

**Consistent123: One Image to Highly Consistent 3D Asset Using Case-Aware Diffusion Priors** \
*Yukang Lin, Haonan Han, Chaoqun Gong, Zunnan Xu, Yachao Zhang, Xiu Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.17261)] \
29 Sep 2023

**Object Motion Guided Human Motion Synthesis** \
*Jiaman Li, Jiajun Wu, C. Karen Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.16237)] \
28 Sep 2023

**ITEM3D: Illumination-Aware Directional Texture Editing for 3D Models** \
*Shengqi Liu, Zhuo Chen, Jingnan Gao, Yichao Yan, Wenhan Zhu, Xiaobo Li, Ke Gao, Jiangjing Lyu, Xiaokang Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.14872)] \
26 Sep 2023




**Light Field Diffusion for Single-View Novel View Synthesis** \
*Yifeng Xiong, Haoyu Ma, Shanlin Sun, Kun Han, Xiaohui Xie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11525)] \
20 Sep 2023

**Latent Diffusion Models for Structural Component Design** \
*Ethan Herron, Jaydeep Rade, Anushrut Jignasu, Baskar Ganapathysubramanian, Aditya Balu, Soumik Sarkar, Adarsh Krishnamurthy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11601)] \
20 Sep 2023

**FaceDiffuser: Speech-Driven 3D Facial Animation Synthesis Using Diffusion** \
*Stefan Stan, Kazi Injamamul Haque, Zerrin Yumak* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.11306)] \
20 Sep 2023

**TwinTex: Geometry-aware Texture Generation for Abstracted 3D Architectural Models** \
*Weidan Xiong, Hongqian Zhang, Botao Peng, Ziyu Hu, Yongli Wu, Jianwei Guo, Hui Huang* \
SIGGRAPH ASIA 2023. [[Paper](https://arxiv.org/abs/2309.11258)] \
20 Sep 2023


**Language-Conditioned Affordance-Pose Detection in 3D Point Clouds** \
*Toan Nguyen, Minh Nhat Vu, Baoru Huang, Tuan Van Vo, Vy Truong, Ngan Le, Thieu Vo, Bac Le, Anh Nguyen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.10911)] \
19 Sep 2023

**Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation** \
*Kaouther Mouheb, Mobina Ghojogh Nejad, Lavsen Dahal, Ehsan Samei, W. Paul Segars, Joseph Y. Lo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.08289)] \
15 Sep 2023

**Unsupervised Disentangling of Facial Representations with 3D-aware Latent Diffusion Models** \
*Ruian He, Zhen Xing, Weimin Tan, Bo Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.08273)] \
15 Sep 2023

**M3Dsynth: A dataset of medical 3D images with AI-generated local manipulations** \
*Giada Zingarini, Davide Cozzolino, Riccardo Corvi, Giovanni Poggi, Luisa Verdoliva* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07973)] \
14 Sep 2023

**Large-Vocabulary 3D Diffusion Model with Transformer** \
*Ziang Cao, Fangzhou Hong, Tong Wu, Liang Pan, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07920)] [[Project](https://ziangcao0312.github.io/difftf_pages/)] [[Github](https://github.com/ziangcao0312/DiffTF)] \
14 Sep 2023

**UnifiedGesture: A Unified Gesture Synthesis Model for Multiple Skeletons** \
*Sicheng Yang, Zilin Wang, Zhiyong Wu, Minglei Li, Zhensong Zhang, Qiaochu Huang, Lei Hao, Songcen Xu, Xiaofei Wu, changpeng yang, Zonghong Dai* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2309.07051)] \
13 Sep 2023

**Fg-T2M: Fine-Grained Text-Driven Human Motion Generation via Diffusion Model** \
*Yin Wang, Zhiying Leng, Frederick W. B. Li, Shun-Cheng Wu, Xiaohui Liang* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2309.06284)] \
12 Sep 2023

**SyncDreamer: Generating Multiview-consistent Images from a Single-view Image** \
*Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, Wenping Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03453)] [[Project](https://liuyuan-pal.github.io/SyncDreamer/)] [[Github](https://github.com/liuyuan-pal/SyncDreamer)] \
7 Sep 2023

**SADIR: Shape-Aware Diffusion Models for 3D Image Reconstruction** \
*Nivetha Jayakumar, Tonmoy Hossain, Miaomiao Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03335)] \
6 Sep 2023

**MCM: Multi-condition Motion Synthesis Framework for Multi-scenario** \
*Zeyu Ling, Bo Han, Yongkang Wong, Mohan Kangkanhalli, Weidong Geng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03031)] \
6 Sep 2023

**DiverseMotion: Towards Diverse Human Motion Generation via Discrete Diffusion** \
*Yunhong Lou, Linchao Zhu, Yaxiong Wang, Xiaohan Wang, Yi Yang* \
AAAI 2024. [[Paper](https://arxiv.org/abs/2309.01372)] \
4 Sep 2023

**BuilDiff: 3D Building Shape Generation using Single-Image Conditional Point Cloud Diffusion Models** \
*Yao Wei, George Vosselman, Michael Ying Yang* \
ICCV Workshop 2023. [[Paper](https://arxiv.org/abs/2309.00158)] \
31 Aug 2023

**MVDream: Multi-view Diffusion for 3D Generation** \
*Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, Xiao Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16512)] \
31 Aug 2023

**Diffusion Inertial Poser: Human Motion Reconstruction from Arbitrary Sparse IMU Configurations** \
*Tom Van Wouwe, Seunghwan Lee, Antoine Falisse, Scott Delp, C. Karen Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.16682)] \
31 Aug 2023


**InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion** \
*Sirui Xu, Zhengyuan Li, Yu-Xiong Wang, Liang-Yan Gui* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.16905)] [[Project](https://sirui-xu.github.io/InterDiff/)] [[Github](https://github.com/Sirui-Xu/InterDiff)] \
31 Aug 2023



**Priority-Centric Human Motion Generation in Discrete Latent Space** \
*Hanyang Kong, Kehong Gong, Dongze Lian, Michael Bi Mi, Xinchao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.14480)] \
28 Aug 2023

**HoloFusion: Towards Photo-realistic 3D Generative Modeling** \
*Animesh Karnewar, Niloy J. Mitra, Andrea Vedaldi, David Novotny* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.14244)] [[Project](https://holodiffusion.github.io/holofusion/)] \
28 Aug 2023

**Unaligned 2D to 3D Translation with Conditional Vector-Quantized Code Diffusion using Transformers** \
*Abril Corona-Figueroa, Sam Bond-Taylor, Neelanjan Bhowmik, Yona Falinie A. Gaus, Toby P. Breckon, Hubert P. H. Shum, Chris G. Willcocks* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.14152)] \
27 Aug 2023

**Sparse3D: Distilling Multiview-Consistent Diffusion for Object Reconstruction from Sparse Views** \
*Zi-Xin Zou, Weihao Cheng, Yan-Pei Cao, Shi-Sheng Huang, Ying Shan, Song-Hai Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.14078)] \
27 Aug 2023

**Multi-plane denoising diffusion-based dimensionality expansion for 2D-to-3D reconstruction of microstructures with harmonized sampling** \
*Kang-Hyun Lee, Gun Jin Yun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.14035)] \
27 Aug 2023

**The DiffuseStyleGesture+ entry to the GENEA Challenge 2023** \
*Sicheng Yang, Haiwei Xue, Zhensong Zhang, Minglei Li, Zhiyong Wu, Xiaofei Wu, Songcen Xu, Zonghong Dai* \
ICMI 2023. [[Paper](https://arxiv.org/abs/2308.13879)] [[Github](https://github.com/YoungSeng/DiffuseStyleGesture/tree/DiffuseStyleGesturePlus/BEAT-TWH-main)] \
26 Aug 2023

**Distribution-Aligned Diffusion for Human Mesh Recovery** \
*Lin Geng Foo, Jia Gong, Hossein Rahmani, Jun Liu* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.13369)] [[Project](https://gongjia0208.github.io/HMDiff/)] \
25 Aug 2023

**EfficientDreamer: High-Fidelity and Robust 3D Creation via Orthogonal-view Diffusion Prior** \
*Minda Zhao, Chaoyi Zhao, Xinyue Liang, Lincheng Li, Zeng Zhao, Zhipeng Hu, Changjie Fan, Xin Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.13223)] \
25 Aug 2023



**DF-3DFace: One-to-Many Speech Synchronized 3D Face Animation with Diffusion** \
*Se Jin Park, Joanna Hong, Minsu Kim, Yong Man Ro* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.05934)] \
23 Aug 2023

**LongDanceDiff: Long-term Dance Generation with Conditional Diffusion Model** \
*Siqi Yang, Zejun Yang, Zhisheng Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.11945)] \
23 Aug 2023

**IT3D: Improved Text-to-3D Generation with Explicit View Synthesis** \
*Yiwen Chen, Chi Zhang, Xiaofeng Yang, Zhongang Cai, Gang Yu, Lei Yang, Guosheng Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.11473)] [[Github](https://github.com/buaacyw/IT3D-text-to-3D)] \
22 Aug 2023

**Texture Generation on 3D Meshes with Point-UV Diffusion** \
*Xin Yu, Peng Dai, Wenbo Li, Lan Ma, Zhengzhe Liu, Xiaojuan Qi* \
ICCV 2023. [[Paper](https://arxiv.org/abs/2308.10490)] \
21 Aug 2023

**Physics-Guided Human Motion Capture with Pose Probability Modeling** \
*Jingyi Ju, Buzhen Huang, Chen Zhu, Zhihao Li, Yangang Wang* \
IJCAI 2023. [[Paper](https://arxiv.org/abs/2308.09910)] [[Github](https://github.com/Me-Ditto/Physics-Guided-Mocap)] \
19 Aug 2023

**Unsupervised 3D Pose Estimation with Non-Rigid Structure-from-Motion Modeling** \
*Haorui Ji, Hui Deng, Yuchao Dai, Hongdong Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.10705)] \
18 Aug 2023

**MATLABER: Material-Aware Text-to-3D via LAtent BRDF auto-EncodeR** \
*Xudong Xu, Zhaoyang Lyu, Xingang Pan, Bo Dai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09278)] [[Project](https://sheldontsui.github.io/projects/Matlaber)] \
18 Aug 2023

**O^2-Recon: Completing 3D Reconstruction of Occluded Objects in the Scene with a Pre-trained 2D Diffusion Model** \
*Yubin Hu, Sheng Ye, Wang Zhao, Matthieu Lin, Yuze He, Yu-Hui Wen, Ying He, Yong-Jin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09591)] \
18 Aug 2023


**Denoising Diffusion for 3D Hand Pose Estimation from Images** \
*Maksym Ivashechkin, Oscar Mendez, Richard Bowden* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09523)] \
18 Aug 2023

**PoSynDA: Multi-Hypothesis Pose Synthesis Domain Adaptation for Robust 3D Human Pose Estimation** \
*Hanbing Liu, Jun-Yan He, Zhi-Qi Cheng, Wangmeng Xiang, Qize Yang, Wenhao Chai, Gaoang Wang, Xu Bao, Bin Luo, Yifeng Geng, Xuansong Xie* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.09678)] [[Github](https://github.com/hbing-l/PoSynDA)] \
18 Aug 2023

**Guide3D: Create 3D Avatars from Text and Image Guidance** \
*Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, Kwan-Yee K. Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09705)] \
18 Aug 2023



**DMCVR: Morphology-Guided Diffusion Model for 3D Cardiac Volume Reconstruction** \
*Xiaoxiao He, Chaowei Tan, Ligong Han, Bo Liu, Leon Axel, Kang Li, Dimitris N. Metaxas* \
MICCAI 2023. [[Paper](https://arxiv.org/abs/2308.09223)] [[Github](https://github.com/hexiaoxiao-cs/DMCVR)] \
18 Aug 2023

**HumanLiff: Layer-wise 3D Human Generation with Diffusion Model** \
*Shoukang Hu, Fangzhou Hong, Tao Hu, Liang Pan, Haiyi Mei, Weiye Xiao, Lei Yang, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09712)] [[Project](https://skhu101.github.io/HumanLiff/)] \
18 Aug 2023



**Watch Your Steps: Local Image and Scene Editing by Text Instructions** \
*Ashkan Mirzaei, Tristan Aumentado-Armstrong, Marcus A. Brubaker, Jonathan Kelly, Alex Levinshtein, Konstantinos G. Derpanis, Igor Gilitschenski* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.08947)] [[Project](https://ashmrz.github.io/WatchYourSteps/)] \
17 Aug 2023

**TeCH: Text-guided Reconstruction of Lifelike Clothed Humans** \
*Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, Justus Thies* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.08545)] [[Project](https://huangyangyi.github.io/TeCH/)] [[Github](https://github.com/huangyangyi/TeCH)] \
16 Aug 2023

**CCD-3DR: Consistent Conditioning in Diffusion for Single-Image 3D Reconstruction** \
*Yan Di, Chenyangguang Zhang, Pengyuan Wang, Guangyao Zhai, Ruida Zhang, Fabian Manhardt, Benjamin Busam, Xiangyang Ji, Federico Tombari* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07837)] \
15 Aug 2023

**Dancing Avatar: Pose and Text-Guided Human Motion Videos Synthesis with Image Diffusion Model** \
*Bosheng Qin, Wentao Ye, Qifan Yu, Siliang Tang, Yueting Zhuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.07749)] \
15 Aug 2023

**3D Scene Diffusion Guidance using Scene Graphs** \
*Mohammad Naanaa, Katharina Schmid, Yinyu Nie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.04468)] \
8 Aug 2023

**Cloth2Tex: A Customized Cloth Texture Generation Pipeline for 3D Virtual Try-On** \
*Daiheng Gao, Xu Chen, Xindi Zhang, Qi Wang, Ke Sun, Bang Zhang, Liefeng Bo, Qixing Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.04288)] \
8 Aug 2023



**AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose** \
*Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Kang Du, Min Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.03610)] [[Project](https://avatarverse3d.github.io/)] \
7 Aug 2023

**Generative Approach for Probabilistic Human Mesh Recovery using Diffusion Models** \
*Hanbyel Cho, Junmo Kim* \
ICCV Workshop 2023. [[Paper](https://arxiv.org/abs/2308.02963)] [[Github](https://github.com/hanbyel0105/Diff-HMR)] \
5 Aug 2023

**DiffDance: Cascaded Human Motion Diffusion Model for Dance Generation** \
*Qiaosong Qi, Le Zhuo, Aixi Zhang, Yue Liao, Fei Fang, Si Liu, Shuicheng Yan* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.02915)] \
5 Aug 2023

**Sketch and Text Guided Diffusion Model for Colored Point Cloud Generation** \
*Zijie Wu, Yaonan Wang, Mingtao Feng, He Xie, Ajmal Mian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02874)] \
5 Aug 2023

**On the Transition from Neural Representation to Symbolic Knowledge** \
*Junyan Cheng, Peter Chin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.02000)] \
3 Aug 2023

**Synthesizing Long-Term Human Motions with Diffusion Models via Coherent Sampling** \
*Zhao Yang, Bing Su, Ji-Rong Wen* \
ACM MM 2023. [[Paper](https://arxiv.org/abs/2308.01850)] [[Github](https://github.com/yangzhao1230/PCMDM)] \
3 Aug 2023



**HD-Fusion: Detailed Text-to-3D Generation Leveraging Multiple Noise Estimation** \
*Jinbo Wu, Xiaobo Gao, Xing Liu, Zhengyang Shen, Chen Zhao, Haocheng Feng, Jingtuo Liu, Errui Ding* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16183)] \
30 Jul 2023

**TransFusion: A Practical and Effective Transformer-based Diffusion Model for 3D Human Motion Prediction** \
*Sibo Tian, Minghui Zheng, Xiao Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16106)] \
30 Jul 2023

**TEDi: Temporally-Entangled Diffusion for Long-Term Motion Synthesis** \
*Zihan Zhang, Richard Liu, Kfir Aberman, Rana Hanocka* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.15042)] \
27 Jul 2023


**Points-to-3D: Bridging the Gap between Sparse Points and Shape-Controllable Text-to-3D Generation** \
*Chaohui Yu, Qiang Zhou, Jingliang Li, Zhe Zhang, Zhibin Wang, Fan Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.13908)] \
26 Jul 2023

**Fake It Without Making It: Conditioned Face Generation for Accurate 3D Face Shape Estimation** \
*Will Rowan, Patrik Huber, Nick Pears, Andrew Keeling* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.13639)] \
25 Jul 2023

**NIFTY: Neural Object Interaction Fields for Guided Human Motion Synthesis** \
*Nilesh Kulkarni, Davis Rempe, Kyle Genova, Abhijit Kundu, Justin Johnson, David Fouhey, Leonidas Guibas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.07511)] [[Project](https://nileshkulkarni.github.io/nifty/)] \
14 Jul 2023

**AvatarFusion: Zero-shot Generation of Clothing-Decoupled 3D Avatars Using 2D Diffusion** \
*Shuo Huang, Zongxin Yang, Liangting Li, Yi Yang, Jia Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.06526)] \
13 Jul 2023

**Articulated 3D Head Avatar Generation using Text-to-Image Diffusion Models** \
*Alexander W. Bergman, Wang Yifan, Gordon Wetzstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04859)] \
10 Jul 2023

**Back to Optimization: Diffusion-based Zero-Shot 3D Human Pose Estimation** \
*Zhongyu Jiang, Zhuoran Zhou, Lei Li, Wenhao Chai, Cheng-Yen Yang, Jenq-Neng Hwang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.03833)] \
7 Jul 2023

**AutoDecoding Latent 3D Diffusion Models** \
*Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, Chaoyang Wang, Luc Van Gool, Sergey Tulyakov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.05445)] \
7 Jul 2023

**SVDM: Single-View Diffusion Model for Pseudo-Stereo 3D Object Detection** \
*Yuguang Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.02270)] \
5 Jul 2023

**DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation** \
*Shentong Mo, Enze Xie, Ruihang Chu, Lewei Yao, Lanqing Hong, Matthias Nießner, Zhenguo Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.01831)] \
4 Jul 2023

**Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors** \
*Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, Bernard Ghanem* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17843)] [[Project](https://guochengqian.github.io/project/magic123/)] \
30 Jun 2023

**Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation** \
*Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, Shenghua Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17115)] \
29 Jun 2023

**DiffComplete: Diffusion-based Generative 3D Shape Completion** \
*Ruihang Chu, Enze Xie, Shentong Mo, Zhenguo Li, Matthias Nießner, Chi-Wing Fu, Jiaya Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.16329)] \
28 Jun 2023

**DreamTime: An Improved Optimization Strategy for Text-to-3D Content Creation** \
*Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, Zheng-Jun Zha, Lei Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12422)] \
21 Jun 2023

**EMoG: Synthesizing Emotive Co-speech 3D Gesture with Diffusion Model** \
*Lianying Yin, Yijun Wang, Tianyu He, Jinming Liu, Wei Zhao, Bohan Li, Xin Jin, Jianxin Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11496)] \
20 Jun 2023


**Point-Cloud Completion with Pretrained Text-to-image Diffusion Models** \
*Yoni Kasten, Ohad Rahamim, Gal Chechik* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.10533)] \
18 Jun 2023

**AvatarBooth: High-Quality and Customizable 3D Human Avatar Generation** \
*Yifei Zeng, Yuanxun Lu, Xinya Ji, Yao Yao, Hao Zhu, Xun Cao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09864)] \
16 Jun 2023

**Edit-DiffNeRF: Editing 3D Neural Radiance Fields using 2D Diffusion Model** \
*Lu Yu, Wei Xiang, Kang Han* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09551)] \
15 Jun 2023

**Adding 3D Geometry Control to Diffusion Models** \
*Wufei Ma, Qihao Liu, Jiahao Wang, Angtian Wang, Yaoyao Liu, Adam Kortylewski, Alan Yuille* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.08103)] \
13 Jun 2023

**Viewset Diffusion: (0-)Image-Conditioned 3D Generative Models from 2D Data** \
*Stanislaw Szymanowicz, Christian Rupprecht, Andrea Vedaldi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07881)] \
13 Jun 2023

**3D molecule generation by denoising voxel grids** \
*Pedro O. Pinheiro, Joshua Rackers, Joseph Kleinhenz, Michael Maser, Omar Mahmood, Andrew Martin Watkins, Stephen Ra, Vishnu Sresht, Saeed Saremi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07473)] \
13 Jun 2023

**InstructP2P: Learning to Edit 3D Point Clouds with Text Instructions** \
*Jiale Xu, Xintao Wang, Yan-Pei Cao, Weihao Cheng, Ying Shan, Shenghua Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.07154)] \
12 Jun 2023

**RePaint-NeRF: NeRF Editting via Semantic Masks and Diffusion Models** \
*Xingchen Zhou, Ying He, F. Richard Yu, Jianqiang Li, You Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05668)] \
9 Jun 2023

**Stochastic Multi-Person 3D Motion Forecasting** \
*Sirui Xu, Yu-Xiong Wang, Liang-Yan Gui* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05421)] \
8 Jun 2023

**ARTIC3D: Learning Robust Articulated 3D Shapes from Noisy Web Image Collections** \
*Chun-Han Yao, Amit Raj, Wei-Chih Hung, Yuanzhen Li, Michael Rubinstein, Ming-Hsuan Yang, Varun Jampani* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04619)] \
7 Jun 2023

**Synthesizing realistic sand assemblies with denoising diffusion in latent space** \
*Nikolaos N. Vlassis, WaiChing Sun, Khalid A. Alshibli, Richard A. Regueiro* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04411)] \
7 Jun 2023


**AvatarStudio: Text-driven Editing of 3D Dynamic Human Head Avatars** \
*Mohit Mendiratta, Xingang Pan, Mohamed Elgharib, Kartik Teotia, Mallikarjun B R, Ayush Tewari, Vladislav Golyanik, Adam Kortylewski, Christian Theobalt* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00547)] \
1 Jun 2023


**DiffRoom: Diffusion-based High-Quality 3D Room Reconstruction and Generation** \
*Xiaoliang Ju, Zhaoyang Huang, Yijin Li, Guofeng Zhang, Yu Qiao, Hongsheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00519)] \
1 Jun 2023

**Controllable Motion Diffusion Model** \
*Yi Shi, Jingbo Wang, Xuekun Jiang, Bo Dai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00416)] [Project](https://controllablemdm.github.io/)] \
1 Jun 2023

**FDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models** \
*Hao Zhang, Yanbo Xu, Tianyuan Dai, Yu-Wing, Tai Chi-Keung Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00783)] \
1 Jun 2023

**Learning Explicit Contact for Implicit Reconstruction of Hand-held Objects from Monocular Images** \
*Junxing Hu, Hongwen Zhang, Zerui Chen, Mengcheng Li, Yunlong Wang, Yebin Liu, Zhenan Sun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.20089)] [[Project](https://junxinghu.github.io/projects/hoi.html)] \
31 May 2023

**StyleAvatar3D: Leveraging Image-Text Diffusion Models for High-Fidelity 3D Avatar Generation** \
*Chi Zhang, Yiwen Chen, Yijun Fu, Zhenglin Zhou, Gang YU, Billzb Wang, Bin Fu, Tao Chen, Guosheng Lin, Chunhua Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19012)] \
30 May 2023

**HiFA: High-fidelity Text-to-3D with Advanced Diffusion Guidance** \
*Junzhe Zhu, Peiye Zhuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18766)] \
30 May 2023


**Conditional Diffusion Models for Semantic 3D Medical Image Synthesis** \
*Zolnamar Dorjsembe, Hsing-Kuo Pao, Sodtavilan Odonchimed, Furen Xiao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18453)] \
29 May 2023

**ZeroAvatar: Zero-shot 3D Avatar Generation from a Single Image** \
*Zhenzhen Weng, Zeyu Wang, Serena Yeung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16411)] \
25 May 2023

**NAP: Neural 3D Articulation Prior** \
*Jiahui Lei, Congyue Deng, Bokui Shen, Leonidas Guibas, Kostas Daniilidis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16315)] [[Project](https://www.cis.upenn.edu/~leijh/projects/nap/)] \
25 May 2023

**CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs** \
*Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, Benjamin Busam* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16283)] \
25 May 2023


**ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation** \
*Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, Jun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16213)] [[Project](https://ml.cs.tsinghua.edu.cn/prolificdreamer/)] \
25 May 2023

**DiffCLIP: Leveraging Stable Diffusion for Language Grounded 3D Classification** \
*Sitian Shen, Zilin Zhu, Linqian Fan, Harry Zhang, Xinxiao Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15957)] \
25 May 2023

**Confronting Ambiguity in 6D Object Pose Estimation via Score-Based Diffusion on SE(3)** \
*Tsu-Ching Hsiao, Hao-Wei Chen, Hsuan-Kung Yang, Chun-Yi Lee* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15873)] \
25 May 2023

**Deceptive-NeRF: Enhancing NeRF Reconstruction using Pseudo-Observations from Diffusion Models** \
*Xinhang Liu, Shiu-hong Kao, Jiaben Chen, Yu-Wing Tai, Chi-Keung Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15171)] \
24 May 2023

**Manifold Diffusion Fields** \
*Ahmed A. Elhag, Joshua M. Susskind, Miguel Angel Bautista* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15586)] \
24 May 2023

**Sin3DM: Learning a Diffusion Model from a Single 3D Textured Shape** \
*Rundi Wu, Ruoshi Liu, Carl Vondrick, Changxi Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15399)] [[Project](https://sin3dm.github.io/)] [[Github](https://github.com/Sin3DM/Sin3DM)] \
24 May 2023

**Understanding Text-driven Motion Synthesis with Keyframe Collaboration via Diffusion Models** \
*Dong Wei, Xiaoning Sun, Huaijiang Sun, Bin Li, Shengxiang Hu, Weiqing Li, Jianfeng Lu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13773)] \
23 May 2023

**DiffHand: End-to-End Hand Mesh Reconstruction via Diffusion Models** \
*Lijun Li, Li'an Zhuo, Bang Zhang, Liefeng Bo, Chen Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13705)] \
23 May 2023

**GMD: Controllable Human Motion Synthesis via Guided Diffusion Models** \
*Korrawe Karunratanakul, Konpat Preechakul, Supasorn Suwajanakorn, Siyu Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12577)] [[Project](https://korrawe.github.io/gmd-project/)] \
21 May 2023

**Towards Globally Consistent Stochastic Human Motion Prediction via Motion Diffusion** \
*Jiarui Sun, Girish Chowdhary* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12554)] \
21 May 2023

**Few-shot 3D Shape Generation** \
*Jingyuan Zhu, Huimin Ma, Jiansheng Chen, Jian Yuan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11664)] \
19 May 2023

**Chupa: Carving 3D Clothed Humans from Skinned Shape Priors using 2D Diffusion Probabilistic Models** \
*Byungjun Kim, Patrick Kwon, Kwangho Lee, Myunggi Lee, Sookwan Han, Daesik Kim, Hanbyul Joo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11870)] [[Project](https://snuvclab.github.io/chupa/)] \
19 May 2023

**Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields** \
*Jingbo Zhang, Xiaoyu Li, Ziyu Wan, Can Wang, Jing Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11588)] \
19 May 2023


**RoomDreamer: Text-Driven 3D Indoor Scene Synthesis with Coherent Geometry and Texture** \
*Liangchen Song, Liangliang Cao, Hongyu Xu, Kai Kang, Feng Tang, Junsong Yuan, Yang Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11337)] \
18 May 2023

**LDM3D: Latent Diffusion Model for 3D** \
*Gabriela Ben Melech Stan, Diana Wofk, Scottie Fox, Alex Redden, Will Saxton, Jean Yu, Estelle Aflalo, Shao-Yen Tseng, Fabio Nonato, Matthias Muller, Vasudev Lal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10853)] \
18 May 2023

**Make-An-Animation: Large-Scale Text-conditional 3D Human Motion Generation** \
*Samaneh Azadi, Akbar Shah, Thomas Hayes, Devi Parikh, Sonal Gupta* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09662)] [[Project](https://azadis.github.io/make-an-animation/)] \
16 May 2023

**FitMe: Deep Photorealistic 3D Morphable Model Avatars** \
*Alexandros Lattas, Stylianos Moschoglou, Stylianos Ploumpis, Baris Gecer, Jiankang Deng, Stefanos Zafeiriou* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2305.09641)] [[Project](https://alexlattas.com/fitme)] \
16 May 2023

**AMD: Autoregressive Motion Diffusion** \
*Bo Han, Hao Peng, Minjing Dong, Chang Xu, Yi Ren, Yixuan Shen, Yuheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09381)] \
16 May 2023


**Text-guided High-definition Consistency Texture Model** \
*Zhibin Tang, Tiantong He* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05901)] \
10 May 2023

**Relightify: Relightable 3D Faces from a Single Image via Diffusion Models** \
*Foivos Paraperas Papantoniou, Alexandros Lattas, Stylianos Moschoglou, Stefanos Zafeiriou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06077)] [[Project](https://foivospar.github.io/Relightify/)] \
10 May 2023

**CaloClouds: Fast Geometry-Independent Highly-Granular Calorimeter Simulation** \
*Erik Buhmann, Sascha Diefenbacher, Engin Eren, Frank Gaede, Gregor Kasieczka, Anatolii Korol, William Korcari, Katja Krüger, Peter McKeown* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04847)] \
8 May 2023

**Locally Attentional SDF Diffusion for Controllable 3D Shape Generation** \
*Xin-Yang Zheng, Hao Pan, Peng-Shuai Wang, Xin Tong, Yang Liu, Heung-Yeung Shum* \
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2305.04461)] \
8 May 2023

**DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion** \
*Kiyohiro Nakayama, Mikaela Angelina Uy, Jiahui Huang, Shi-Min Hu, Ke Li, Leonidas J Guibas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01921)] [[Github](https://difffacto.github.io/)] \
4 May 2023

**Shap-E: Generating Conditional 3D Implicit Functions** \
*Heewoo Jun, Alex Nichol* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.02463)] [[Github](https://github.com/openai/shap-e)]
3 May 2023

**ContactArt: Learning 3D Interaction Priors for Category-level Articulated Object and Hand Poses Estimation** \
*Zehao Zhu, Jiashun Wang, Yuzhe Qin, Deqing Sun, Varun Jampani, Xiaolong Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01618)] [[Project](https://zehaozhu.github.io/ContactArt/)] \
2 May 2023

**DreamPaint: Few-Shot Inpainting of E-Commerce Items for Virtual Try-On without 3D Modeling** \
*Mehmet Saygin Seyfioglu, Karim Bouyarmane, Suren Kumar, Amir Tavanaei, Ismail B. Tutar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01257)] \
2 May 2023

**Learning a Diffusion Prior for NeRFs** \
*Guandao Yang, Abhijit Kundu, Leonidas J. Guibas, Jonathan T. Barron, Ben Poole* \
ICLR Workshop 2023. [[Paper](https://arxiv.org/abs/2304.14473)] \
27 Apr 2023

**TextMesh: Generation of Realistic 3D Meshes From Text Prompts** \
*Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, Federico Tombari* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12439)] \
24 Apr 2023

**Nerfbusters: Removing Ghostly Artifacts from Casually Captured NeRFs** \
*Frederik Warburg, Ethan Weber, Matthew Tancik, Aleksander Holynski, Angjoo Kanazawa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.10532)] [[Project](https://ethanweber.me/nerfbusters/)] [[Github](https://github.com/ethanweber/nerfbusters)] \
20 Apr 2023

**Farm3D: Learning Articulated 3D Animals by Distilling 2D Diffusion** \
*Tomas Jakab, Ruining Li, Shangzhe Wu, Christian Rupprecht, Andrea Vedaldi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.10535)] [[Project](https://farm3d.github.io/)] \
20 Apr 2023

**Anything-3D: Towards Single-view Anything Reconstruction in the Wild** \
*Qiuhong Shen, Xingyi Yang, Xinchao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.10261)] \
19 Apr 2023


**Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model** \
*Yuming Du, Robin Kips, Albert Pumarola, Sebastian Starke, Ali Thabet, Artsiom Sanakoyeu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.08577)] [[Project](https://dulucas.github.io/agrol/)] [[Github]()] \
17 Apr 2023

**Towards Controllable Diffusion Models via Reward-Guided Exploration** \
*Hengtong Zhang, Tingyang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.07132)] \
14 Apr 2023

**Learning Controllable 3D Diffusion Models from Single-view Images** \
*Jiatao Gu, Qingzhe Gao, Shuangfei Zhai, Baoquan Chen, Lingjie Liu, Josh Susskind* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06700)] [[Project](https://jiataogu.me/control3diff/)] \
13 Apr 2023

**Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction** \
*Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, Hao Su* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06714)] [[Project](https://lakonik.github.io/ssdnerf/)] \
13 Apr 2023

**Probabilistic Human Mesh Recovery in 3D Scenes from Egocentric Views** \
*Siwei Zhang, Qianli Ma, Yan Zhang, Sadegh Aliakbarian, Darren Cosker, Siyu Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06024)] [[Project](https://sanweiliti.github.io/egohmr/egohmr.html)] \
12 Apr 2023

**InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions** \
*Han Liang, Wenqian Zhang, Wenxuan Li, Jingyi Yu, Lan Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05684)] [[Github](https://github.com/tr3e/InterGen)] \
12 Apr 2023

**Probabilistic Human Mesh Recovery in 3D Scenes from Egocentric Views** \
*Siwei Zhang, Qianli Ma, Yan Zhang, Sadegh Aliakbarian, Darren Cosker, Siyu Tang* \
arXiv 2023. [[Paper]()] [[Project](https://sanweiliti.github.io/egohmr/egohmr.html)] \
12 Apr 2023


**Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond** \
*Mohammadreza Armandpour, Huangjie Zheng, Ali Sadeghian, Amir Sadeghian, Mingyuan Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04968)] [[Project](https://perp-neg.github.io/)] \
11 Apr 2023



**NeRF applied to satellite imagery for surface reconstruction** \
*Federico Semeraro, Yi Zhang, Wenying Wu, Patrick Carroll* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04133)] [[Github](https://github.com/fsemerar/satnerf)] \
9 Apr 2023

**DITTO-NeRF: Diffusion-based Iterative Text To Omni-directional 3D Model** \
*Hoigi Seo, Hayeon Kim, Gwanghyun Kim, Se Young Chun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02827)] [[Project](https://janeyeon.github.io/ditto-nerf/)] \
6 Apr 2023


**Generative Novel View Synthesis with 3D-Aware Diffusion Models** \
*Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, Gordon Wetzstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02602)] [[Project](https://nvlabs.github.io/genvs/)] \
5 Apr 2023

**Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion** \
*Davis Rempe, Zhengyi Luo, Xue Bin Peng, Ye Yuan, Kris Kitani, Karsten Kreis, Sanja Fidler, Or Litany* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.01893)] [[Github](https://research.nvidia.com/labs/toronto-ai/trace-pace/)] \
4 Apr 2023

**PODIA-3D: Domain Adaptation of 3D Generative Model Across Large Domain Gap Using Pose-Preserved Text-to-Image Diffusion** \
*Gwanghyun Kim, Ji Ha Jang, Se Young Chun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01900)] [[Project](https://gwang-kim.github.io/podia_3d/)] \
4 Apr 2023

**ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model** \
*Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, Ziwei Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01116)] [[Project](https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html)] [[Github](https://github.com/mingyuan-zhang/ReMoDiffuse)] \
3 Apr 2023

**Controllable Motion Synthesis and Reconstruction with Autoregressive Diffusion Models** \
*Wenjie Yin, Ruibo Tu, Hang Yin, Danica Kragic, Hedvig Kjellström, Mårten Björkman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04681)] \
3 Apr 2023

**DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models** \
*Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, Kwan-Yee K. Wong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.00916)] \
3 Apr 2023

**DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance** \
*Longwen Zhang, Qiwei Qiu, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, Jingyi Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03117)] [[Project](https://sites.google.com/view/dreamface)] \
1 Apr 2023


**AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control** \
*Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17606)] [[Project](https://avatar-craft.github.io/)] [[Github](https://github.com/songrise/avatarcraft)] \
30 Mar 2023

**HOLODIFFUSION: Training a 3D Diffusion Model using 2D Images** \
*Animesh Karnewar, Andrea Vedaldi, David Novotny, Niloy Mitra* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.16509)] [[Project](https://holodiffusion.github.io/)] \
29 Mar 2023

**4D Facial Expression Diffusion Model** \
*Kaifeng Zou, Sylvain Faisan, Boyang Yu, Sébastien Valette, Hyewon Seo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16611)] [[Github](https://github.com/ZOUKaifeng/4DFM)] \
29 Mar 2023


**Instruct 3D-to-3D: Text Instruction Guided 3D-to-3D conversion** \
*Hiromichi Kamata, Yuiko Sakuma, Akio Hayakawa, Masato Ishii, Takuya Narihira* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15780)] [[Project](https://sony.github.io/Instruct3Dto3D-doc/)] [[Github](https://sony.github.io/Instruct3Dto3D-doc/)] \
28 Mar 2023

**Novel View Synthesis of Humans using Differentiable Rendering** \
*Guillaume Rochette, Chris Russell, Richard Bowden* \
IEEE T-BIOM 2023. [[Paper](https://arxiv.org/abs/2303.15880)] [[Github](https://github.com/GuillaumeRochette/HumanViewSynthesis)] \
28 Mar 2023


**Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation** \
*Susung Hong, Donghoon Ahn, Seungryong Kim* \
CVPR Workshop 2023. [[Paper](https://arxiv.org/abs/2303.15413)] \
27 Mar 2023

**Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior** \
*Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, Dong Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14184)] [[Project](https://make-it-3d.github.io/)] [[Github](https://make-it-3d.github.io/)] \
24 Mar 2023

**ISS++: Image as Stepping Stone for Text-Guided 3D Shape Generation** \
*Zhengzhe Liu, Peng Dai, Ruihui Li, Xiaojuan Qi, Chi-Wing Fu* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2303.15181)] \
24 Mar 2023

**CompoNeRF: Text-guided Multi-object Compositional NeRF with Editable 3D Scene Layout** \
*Yiqi Lin, Haotian Bai, Sijia Li, Haonan Lu, Xiaodong Lin, Hui Xiong, Lin Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13843)] [[Project](https://fantasia3d.github.io/)] \
24 Mar 2023

**Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation** \
*Rui Chen, Yongwei Chen, Ningxin Jiao, Kui Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13873)] [[Project](https://fantasia3d.github.io/)] [[Github](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)] \
24 Mar 2023

**DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video** \
*Ce Zheng, Guo-Jun Qi, Chen Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13397)] \
23 Mar 2023

**Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions** \
*Ayaan Haque, Matthew Tancik, Alexei A. Efros, Aleksander Holynski, Angjoo Kanazawa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12789)] [[Project](https://instruct-nerf2nerf.github.io/)] \
22 Mar 2023

**FeatureNeRF: Learning Generalizable NeRFs by Distilling Foundation Models** \
*Jianglong Ye, Naiyan Wang, Xiaolong Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12786)] [[Project](https://jianglongye.com/featurenerf/)] \
22 Mar 2023

**Vox-E: Text-guided Voxel Editing of 3D Objects** \
*Etai Sella, Gal Fiebelman, Peter Hedman, Hadar Averbuch-Elor* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12048)] [[Project](https://tau-vailab.github.io/Vox-E/)] \
21 Mar 2023

**Compositional 3D Scene Generation using Locally Conditioned Diffusion** \
*Ryan Po, Gordon Wetzstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12218)] [[Github](https://ryanpo.com/comp3d/)] \
21 Mar 2023

**Diffusion-Based 3D Human Pose Estimation with Multi-Hypothesis Aggregation** \
*Wenkang Shan, Zhenhua Liu, Xinfeng Zhang, Zhao Wang, Kai Han, Shanshe Wang, Siwei Ma, Wen Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11579)] [[Github](https://github.com/paTRICK-swk/D3DP)] \
21 Mar 2023

**3D-CLFusion: Fast Text-to-3D Rendering with Contrastive Latent Diffusion** \
*Yu-Jhe Li, Kris Kitani* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11938)] \
21 Mar 2023

**Affordance Diffusion: Synthesizing Hand-Object Interactions** \
*Yufei Ye, Xueting Li, Abhinav Gupta, Shalini De Mello, Stan Birchfield, Jiaming Song, Shubham Tulsiani, Sifei Liu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.12538)] [[Project](https://judyye.github.io/affordiffusion-www/)] \
21 Mar 2023



**SALAD: Part-Level Latent Diffusion for 3D Shape Generation and Manipulation** \
*Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, Minhyuk Sung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12236)] [[Project](https://salad3d.github.io/)] \
21 Mar 2023

**Learning a 3D Morphable Face Reflectance Model from Low-cost Data** \
*Yuxuan Han, Zhibo Wang, Feng Xu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.11686)] [[Project](https://yxuhan.github.io/ReflectanceMM/index.html)] \
21 Mar 2023

**Text2Tex: Text-driven Texture Synthesis via Diffusion Models** \
*Dave Zhenyu Chen, Yawar Siddiqui, Hsin-Ying Lee, Sergey Tulyakov, Matthias Nießner* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11396)] [[Project](https://daveredrum.github.io/Text2Tex/)] \
20 Mar 2023

**Zero-1-to-3: Zero-shot One Image to 3D Object** \
*Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, Carl Vondrick* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11328)] [[Project](https://zero123.cs.columbia.edu/)] [[Github](https://github.com/cvlab-columbia/zero123)] \
20 Mar 2023

**SKED: Sketch-guided Text-based 3D Editing** \
*Aryan Mikaeili, Or Perel, Daniel Cohen-Or, Ali Mahdavi-Amiri* \
arxiv 2023. [[Paper](https://arxiv.org/abs/2303.10735)] \
19 Mar 2023

**3DQD: Generalized Deep 3D Shape Prior via Part-Discretized Diffusion Process** \
*Yuhan Li, Yishun Dou, Xuanhong Chen, Bingbing Ni, Yilin Sun, Yutian Liu, Fuzhen Wang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.10406)] [[Github](https://github.com/colorful-liyu/3DQD)] \
18 Mar 2023

**Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation** \
*Lingting Zhu, Xian Liu, Xuanyu Liu, Rui Qian, Ziwei Liu, Lequan Yu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.09119)] [[Github](https://github.com/Advocate99/DiffGesture)] \
16 Mar 2023

**Diffusion-HPC: Generating Synthetic Images with Realistic Humans** \
*Zhenzhen Weng, Laura Bravo-Sánchez, Serena Yeung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09541)] [[Github](https://github.com/ZZWENG/Diffusion_HPC)] \
16 Mar 2023

**DINAR: Diffusion Inpainting of Neural Textures for One-Shot Human Avatars** \
*David Svitov, Dmitrii Gudkov, Renat Bashirov, Victor Lempitsky* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09375)] \
16 Mar 2023

**Improving 3D Imaging with Pre-Trained Perpendicular 2D Diffusion Models** \
*Suhyeon Lee, Hyungjin Chung, Minyoung Park, Jonghyuk Park, Wi-Sun Ryu, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08440)] \
15 Mar 2023

**Controllable Mesh Generation Through Sparse Latent Point Diffusion Models** \
*Zhaoyang Lyu, Jinyi Wang, Yuwei An, Ya Zhang, Dahua Lin, Bo Dai* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.07938)] [[Project](https://slide-3d.github.io/)] \
14 Mar 2023

**MeshDiffusion: Score-based Generative 3D Mesh Modeling** \
*Zhen Liu, Yao Feng, Michael J. Black, Derek Nowrouzezahrai, Liam Paull, Weiyang Liu* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2303.08133)] [[Project](https://meshdiffusion.github.io/)] [[Github](https://github.com/lzzcd001/MeshDiffusion/)] \
14 Mar 2023

**Point Cloud Diffusion Models for Automatic Implant Generation** \
*Paul Friedrich, Julia Wolleb, Florentin Bieder, Florian M. Thieringer, Philippe C. Cattin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08061)] \
14 Mar 2023

**Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation** \
*Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, Seungryong Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07937)] [[Github](https://github.com/KU-CVLAB/3DFuse)] \
14 Mar 2023

**GECCO: Geometrically-Conditioned Point Diffusion Models** \
*Michał J. Tyszkiewicz, Pascal Fua, Eduard Trulls* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05916)] \
10 Mar 2023



**3DGen: Triplane Latent Diffusion for Textured Mesh Generation** \
*Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, Barlas Oğuz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05371)] \
9 Mar 2023

**Human Motion Diffusion as a Generative Prior** \
*Yonatan Shafir, Guy Tevet, Roy Kapon, Amit H. Bermano* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.01418)] \
2 Mar 2023

**Can We Use Diffusion Probabilistic Models for 3D Motion Prediction?** \
*Hyemin Ahn, Esteve Valls Mascaro, Dongheui Lee* \
ICRA 2023. [[Paper](https://arxiv.org/abs/2302.14503)] [[Project](https://sites.google.com/view/diffusion-motion-prediction)] [[Github](https://github.com/cotton-ahn/diffusion-motion-prediction)] \
28 Feb 2023


**DiffusioNeRF: Regularizing Neural Radiance Fields with Denoising Diffusion Models** \
*Jamie Wynn, Daniyar Turmukhambetov* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2302.12231)] [[Github](https://github.com/nianticlabs/diffusionerf)] [[Github](https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion)] \
23 Feb 2023

**PC2: Projection-Conditioned Point Cloud Diffusion for Single-Image 3D Reconstruction** \
*Luke Melas-Kyriazi, Christian Rupprecht, Andrea Vedaldi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10668)] [Project](https://lukemelas.github.io/projection-conditioned-point-cloud-diffusion/)] \
23 Feb 2023

**NerfDiff: Single-image View Synthesis with NeRF-guided Distillation from 3D-aware Diffusion** \
*Jiatao Gu, Alex Trevithick, Kai-En Lin, Josh Susskind, Christian Theobalt, Lingjie Liu, Ravi Ramamoorthi* \
ICML 2023. [[Paper](https://arxiv.org/abs/2302.10109)] [[Github](https://jiataogu.me/nerfdiff/)] \
20 Feb 2023

**SinMDM: Single Motion Diffusion** \
*Sigal Raab, Inbal Leibovitch, Guy Tevet, Moab Arar, Amit H. Bermano, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05905)] [[Project](https://sinmdm.github.io/SinMDM-page/)] [[Github](https://github.com/SinMDM/SinMDM)] \
12 Feb 2023

**3D Colored Shape Reconstruction from a Single RGB Image through Diffusion** \
*Bo Li, Xiaolin Wei, Fengwei Chen, Bin Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05573)] \
11 Feb 2023

**HumanMAC: Masked Motion Completion for Human Motion Prediction** \
*Ling-Hao Chen, Jiawei Zhang, Yewen Li, Yiren Pang, Xiaobo Xia, Tongliang Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03665)] [[Project](https://lhchen.top/Human-MAC/)] [[Github](https://github.com/LinghaoChan/HumanMAC)] \
7 Feb 2023

**TEXTure: Text-Guided Texturing of 3D Shapes** \
*Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01721)] [[Project](https://texturepaper.github.io/TEXTurePaper/)] [[Github](https://github.com/TEXTurePaper/TEXTurePaper)] \
3 Feb 2023



**Zero3D: Semantic-Driven Multi-Category 3D Shape Generation** \
*Bo Han, Yitong Liu, Yixuan Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13591)] \
31 Jan 2023

**Neural Wavelet-domain Diffusion for 3D Shape Generation, Inversion, and Manipulation** \
*Jingyu Hu, Ka-Hei Hui, Zhengzhe Liu, Ruihui Li, Chi-Wing Fu* \
SIGGRAPH ASIA 2023. [[Paper](https://arxiv.org/abs/2302.00190)] [[Github](https://github.com/edward1997104/Wavelet-Generation)] \
1 Feb 2023

**3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models** \
*Biao Zhang, Jiapeng Tang, Matthias Niessner, Peter Wonka* \
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2301.11445)] [[Github](https://1zb.github.io/3DShape2VecSet/)] [[Github](https://github.com/1zb/3DShape2VecSet)] \
26 Jan 2023


**DiffMotion: Speech-Driven Gesture Synthesis Using Denoising Diffusion Model** \
*Fan Zhang, Naye Ji, Fuxing Gao, Yongping Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.10047)] \
24 Jan 2023

**Bipartite Graph Diffusion Model for Human Interaction Generation** \
*Baptiste Chopin, Hao Tang, Mohamed Daoudi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.10134)] \
24 Jan 2023


**Diffusion-based Generation, Optimization, and Planning in 3D Scenes** \
*Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, Song-Chun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.06015)] [[Project](https://scenediffuser.github.io/)] [[Github](https://github.com/scenediffuser/Scene-Diffuser)] \
15 Jan 2023

**Modiff: Action-Conditioned 3D Motion Generation with Denoising Diffusion Probabilistic Models** \
*Mengyi Zhao, Mengyuan Liu, Bin Ren, Shuling Dai, Nicu Sebe* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.03949)] \
10 Jan 2023


**Diffusion Probabilistic Models for Scene-Scale 3D Categorical Data** \
*Jumin Lee, Woobin Im, Sebin Lee, Sung-Eui Yoon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.00527)] [[Github](https://github.com/zoomin-lee/scene-scale-diffusion)] \
2 Jan 2023

**Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models** \
*Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, Shenghua Gao* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.14704)] [[Project](https://bluestyle97.github.io/dream3d/)] \
28 Dec 2022

**Point-E: A System for Generating 3D Point Clouds from Complex Prompts** \
*Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, Mark Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08751)] [[Github](https://github.com/openai/point-e)] \
16 Dec 2022

**Real-Time Rendering of Arbitrary Surface Geometries using Learnt Transfer** \
*Sirikonda Dhawal, Aakash KT, P.J. Narayanan* \
ICVGIP 2022. [[Paper](https://arxiv.org/abs/2212.09315)] \
19 Dec 2022

**Unifying Human Motion Synthesis and Style Transfer with Denoising Diffusion Probabilistic Models** \
*Ziyi Chang, Edmund J. C. Findlay, Haozheng Zhang, Hubert P. H. Shum* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08526)] \
16 Dec 2022

**Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion** \
*Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, Baining Guo* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06135)] [[Project](https://3d-avatar-diffusion.microsoft.com/#/)] \
12 Dec 2022

**Generative Scene Synthesis via Incremental View Inpainting using RGBD Diffusion Models** \
*Jiabao Lei, Jiapeng Tang, Kui Jia* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.05993)] [[Project](https://jblei.site/project-pages/rgbd-diffusion.html)] [[Github](https://github.com/Karbo123/RGBD-Diffusion)] \
12 Dec 2022

**Ego-Body Pose Estimation via Ego-Head Pose Estimation** \
*Jiaman Li, C. Karen Liu, Jiajun Wu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.04636)] \
9 Dec 2022


**MoFusion: A Framework for Denoising-Diffusion-based Motion Synthesis** \
*Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, Christian Theobalt* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.04495)] [[Project](https://vcai.mpi-inf.mpg.de/projects/MoFusion/)] \
8 Dec 2022


**SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation** \
*Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander Schwing, Liangyan Gui* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.04493)] [[Project](https://yccyenchicheng.github.io/SDFusion/)] \
8 Dec 2022


**Executing your Commands via Motion Diffusion in Latent Space** \
*Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, Gang Yu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.04048)] [[Project](https://chenxin.tech/mld/)] [[Github](https://github.com/ChenFengYe/motion-latent-diffusion)] \
8 Dec 2022

**Magic: Multi Art Genre Intelligent Choreography Dataset and Network for 3D Dance Generation** \
*Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Xiu Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03741)] \
7 Dec 2022


**NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors** \
*Congyue Deng, Chiyu "Max'' Jiang, Charles R. Qi, Xinchen Yan, Yin Zhou, Leonidas Guibas, Dragomir Anguelov* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.03267)] \
6 Dec 2022

**Diffusion-SDF: Text-to-Shape via Voxelized Diffusion** \
*Muheng Li, Yueqi Duan, Jie Zhou, Jiwen Lu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.03293)] [[Github](https://github.com/ttlmh/Diffusion-SDF)] \
6 Dec 2022



**Pretrained Diffusion Models for Unified Human Motion Synthesis** \
*Jianxin Ma, Shuai Bai, Chang Zhou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02837)] [[Project](https://ofa-sys.github.io/MoFusion/)] \
6 Dec 2022

**DiffuPose: Monocular 3D Human Pose Estimation via Denoising Diffusion Probabilistic Model** \
*Jeongjun Choi, Dongseok Shim, H. Jin Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02796)] \
6 Dec 2022

**PhysDiff: Physics-Guided Human Motion Diffusion Model** \
*Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, Jan Kautz* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.02500)] [[Project](https://nvlabs.github.io/PhysDiff/)] \
5 Dec 2022

**Fast Point Cloud Generation with Straight Flows** \
*Lemeng Wu, Dilin Wang, Chengyue Gong, Xingchao Liu, Yunyang Xiong, Rakesh Ranjan, Raghuraman Krishnamoorthi, Vikas Chandra, Qiang Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.01747)] \
4 Dec 2022


**DiffRF: Rendering-Guided 3D Radiance Field Diffusion** \
*Norman Müller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulò, Peter Kontschieder, Matthias Nießner* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.01206)] [[Project](https://sirwyver.github.io/DiffRF/)] \
2 Dec 2022

**3D-LDM: Neural Implicit 3D Shape Generation with Latent Diffusion Models** \
*Gimin Nam, Mariem Khlifi, Andrew Rodriguez, Alberto Tono, Linqi Zhou, Paul Guerrero* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00842)] \
1 Dec 2022


**Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation** \
*Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, Greg Shakhnarovich* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.00774)] [[Project](https://pals.ttic.edu/p/score-jacobian-chaining)] \
1 Dec 2022


**SparseFusion: Distilling View-conditioned Diffusion for 3D Reconstruction** \
*Zhizhuo Zhou, Shubham Tulsiani* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.00792)] [[Project](https://sparsefusion.github.io/)] [[Github](https://sparsefusion.github.io/)] \
1 Dec 2022

**3D Neural Field Generation using Triplane Diffusion** \
*J. Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, Gordon Wetzstein* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16677)] [[Project](https://jryanshue.com/nfd/)] \
30 Nov 2022


**DiffPose: Toward More Reliable 3D Pose Estimation** \
*Jia Gong, Lin Geng Foo, Zhipeng Fan, Qiuhong Ke, Hossein Rahmani, Jun Liu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.16940)] [[Github](https://github.com/GONGJIA0208/Diffpose)] \
30 Nov 2022

**DiffPose: Multi-hypothesis Human Pose Estimation using Diffusion models** \
*Karl Holmquist, Bastian Wandt* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16487)] [[Github](https://github.com/paTRICK-swk/D3DP)] \
29 Nov 2022

**DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model** \
*Gwanghyun Kim, Se Young Chun* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.16374)] [[Github](https://datid-3d.github.io/)] \
29 Nov 2022

**NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views** \
*Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, Zhangyang Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16431)] [[Project](https://vita-group.github.io/NeuralLift-360/)] [[Github](https://github.com/VITA-Group/NeuralLift-360)] \
29 Nov 2022

**Ada3Diff: Defending against 3D Adversarial Point Clouds via Adaptive Diffusion** \
*Kui Zhang, Hang Zhou, Jie Zhang, Qidong Huang, Weiming Zhang, Nenghai Yu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16247)] \
29 Nov 2022

**UDE: A Unified Driving Engine for Human Motion Generation** \
*Zixiang Zhou, Baoyuan Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.16016)] [[Project](https://zixiangzhou916.github.io/UDE/)] [[Github](https://github.com/zixiangzhou916/UDE/)] \
29 Nov 2022


**3DDesigner: Towards Photorealistic 3D Object Generation and Editing with Text-guided Diffusion Models** \
*Gang Li, Heliang Zheng, Chaoyue Wang, Chang Li, Changwen Zheng, Dacheng Tao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.14108)] \
25 Nov 2022

**DiffusionSDF: Conditional Generative Modeling of Signed Distance Functions** \
*Gene Chou, Yuval Bahat, Felix Heide* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13757)] [[Github](https://github.com/princeton-computational-imaging/Diffusion-SDF)] \
24 Nov 2022

**Tetrahedral Diffusion Models for 3D Shape Generation** \
*Nikolai Kalischek, Torben Peters, Jan D. Wegner, Konrad Schindler* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13220)] \
23 Nov 2022

**IC3D: Image-Conditioned 3D Diffusion for Shape Generation** \
*Cristian Sbrolli, Paolo Cudrano, Matteo Frosi, Matteo Matteucci* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10865)] \
20 Nov 2022


**Listen, denoise, action! Audio-driven motion synthesis with diffusion models** \
*Simon Alexanderson, Rajmund Nagy, Jonas Beskow, Gustav Eje Henter* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09707)] \
17 Nov 2022


**RenderDiffusion: Image Diffusion for 3D Reconstruction, Inpainting and Generation** \
*Titas Anciukevičius, Zexiang Xu, Matthew Fisher, Paul Henderson, Hakan Bilen, Niloy J. Mitra, Paul Guerrero* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.09869)] [[Github](https://github.com/Anciukevicius/RenderDiffusion)] \
17 Nov 2022


**Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures** \
*Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, Daniel Cohen-Or* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.07600)] [[Github](https://github.com/eladrich/latent-nerf)] \
14 Nov 2022

**ReFu: Refine and Fuse the Unobserved View for Detail-Preserving Single-Image 3D Human Reconstruction** \
*Gyumin Shim, Minsoo Lee, Jaegul Choo* \
ACM 2022. [[Paper](https://arxiv.org/abs/2211.04753)] \
9 Nov 2022

**StructDiffusion: Object-Centric Diffusion for Semantic Rearrangement of Novel Objects** \
*Weiyu Liu, Tucker Hermans, Sonia Chernova, Chris Paxton* \
RSS 2023. [[Paper](https://arxiv.org/abs/2211.04604)] \
8 Nov 2022

**Diffusion Motion: Generate Text-Guided 3D Human Motion by Diffusion Model** \
*Zhiyuan Ren, Zhihong Pan, Xin Zhou, Le Kang* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2210.12315)] \
22 Oct 2022

**LION: Latent Point Diffusion Models for 3D Shape Generation** \
*Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis* \
NeurIPS 2022. [[Paper](https://arxiv.org/pdf/2210.06978.pdf)] [[Project](https://nv-tlabs.github.io/LION/)] \
12 Oct 2022

**Human Joint Kinematics Diffusion-Refinement for Stochastic Motion Prediction** \
*Dong Wei, Huaijiang Sun, Bin Li, Jianfeng Lu, Weiqing Li, Xiaoning Sun, Shengxiang Hu* \
AAAI 2023. [[Paper](https://arxiv.org/abs/2210.05976)] \
12 Oct 2022


**A generic diffusion-based approach for 3D human pose prediction in the wild** \
*Saeed Saadatnejad, Ali Rasekh, Mohammadreza Mofayezi, Yasamin Medghalchi, Sara Rajabzadeh, Taylor Mordan, Alexandre Alahi* \
ICRA 2023. [[Paper](https://arxiv.org/abs/2210.05669)] \
11 Oct 2022


**Novel View Synthesis with Diffusion Models** \
*Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, Mohammad Norouzi* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.04628)] \
6 Oct 2022

**Neural Volumetric Mesh Generator** \
*Yan Zheng, Lemeng Wu, Xingchao Liu, Zhen Chen, Qiang Liu, Qixing Huang* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2210.03158)] \
6 Oct 2022


**Denoising Diffusion Probabilistic Models for Styled Walking Synthesis** \
*Edmund J. C. Findlay, Haozheng Zhang, Ziyi Chang, Hubert P. H. Shum* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2209.14828)] \
29 Sep 2022


**Human Motion Diffusion Model** \
*Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Amit H. Bermano, Daniel Cohen-Or* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14916)] [[Project](https://guytevet.github.io/mdm-page/)] \
29 Sep 2022


**ISS: Image as Stepping Stone for Text-Guided 3D Shape Generation** \
*Zhengzhe Liu, Peng Dai, Ruihui Li, Xiaojuan Qi, Chi-Wing Fu* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2209.04145)] [[Github](https://github.com/liuzhengzhe/ISS-Image-as-Stepping-Stone-for-Text-Guided-3D-Shape-Generation)] \
9 Sep 2022

**SE(3)-DiffusionFields: Learning cost functions for joint grasp and motion optimization through diffusion** \
*Julen Urain, Niklas Funk, Georgia Chalvatzaki, Jan Peters* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.03855)] [[Github](https://github.com/TheCamusean/grasp_diffusion)] \
8 Sep 2022

**First Hitting Diffusion Models for Generating Manifold, Graph and Categorical Data** \
*Mao Ye, Lemeng Wu, Qiang Liu* \
NeruIPS 2022. [[Paper](https://arxiv.org/abs/2209.01170)] \
2 Sep 2022

**FLAME: Free-form Language-based Motion Synthesis & Editing** \
*Jihoon Kim, Jiseob Kim, Sungjoon Choi* \
AAAI 2023. [[Paper](https://arxiv.org/abs/2209.00349)] \
1 Sep 2022

**Let us Build Bridges: Understanding and Extending Diffusion Generative Models** \
*Xingchao Liu, Lemeng Wu, Mao Ye, Qiang Liu* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2208.14699)] \
31 Aug 2022


**MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model** \
*Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, Ziwei Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.15001)] [[Project](https://mingyuan-zhang.github.io/projects/MotionDiffuse.html)] \
31 Aug 2022


**A Diffusion Model Predicts 3D Shapes from 2D Microscopy Images** \
*Dominik J. E. Waibel, Ernst Röell, Bastian Rieck, Raja Giryes, Carsten Marr* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.14125)] \
30 Aug 2022


**PointDP: Diffusion-driven Purification against Adversarial Attacks on 3D Point Cloud Recognition** \
*Jiachen Sun, Weili Nie, Zhiding Yu, Z. Morley Mao, Chaowei Xiao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.09801)] \
21 Aug 2022

**A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion** \
*Zhaoyang Lyu, Zhifeng Kong, Xudong Xu, Liang Pan, Dahua Lin* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2112.03530)] [[Github](https://github.com/zhaoyanglyu/point_diffusion_refinement)] \
7 Dec 2021

**Score-Based Point Cloud Denoising** \
*Shitong Luo, Wei Hu*\
ICCV 2021. [[Paper](https://arxiv.org/abs/2107.10981)] [[Github](https://github.com/luost26/score-denoise)] \
23 Jul 2021



**DiffuStereo: High Quality Human Reconstruction via Diffusion-based Stereo Using Sparse Cameras** \
*Ruizhi Shao, Zerong Zheng, Hongwen Zhang, Jingxiang Sun, Yebin Liu* \
ECCV 2022. [[Paper](https://arxiv.org/abs/2207.08000)] [[Project](http://liuyebin.com/diffustereo/diffustereo.html)] [[Github](https://github.com/DSaurus/DiffuStereo)] \
16 Jul 2022

**3D Shape Generation and Completion through Point-Voxel Diffusion** \
*Linqi Zhou, Yilun Du, Jiajun Wu* \
ICCV 2021. [[Paper](https://arxiv.org/abs/2104.03670)] [[Project](https://alexzhou907.github.io/pvd)] \
8 Apr 2021

**Diffusion Probabilistic Models for 3D Point Cloud Generation** \
*Shitong Luo, Wei Hu* \
CVPR 2021. [[Paper](https://arxiv.org/abs/2103.01458)] [[Github](https://github.com/luost26/diffusion-point-cloud)] \
2 Mar 2021 


### Adversarial Attack

**Generated Distributions Are All You Need for Membership Inference Attacks Against Generative Models** \
*Minxing Zhang, Ning Yu, Rui Wen, Michael Backes, Yang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.19410)] \
30 Oct 2023

**Adversarial Examples Are Not Real Features** \
*Ang Li, Yifei Wang, Yiwen Guo, Yisen Wang* \
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2310.18936)] \
29 Oct 2023

**Purify++: Improving Diffusion-Purification with Advanced Diffusion Models and Control of Randomness** \
*Boya Zhang, Weijian Luo, Zhihua Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.18762)] \
28 Oct 2023

**Energy-Based Models for Anomaly Detection: A Manifold Diffusion Recovery Approach** \
*Sangwoong Yoon, Young-Uk Jin, Yung-Kyun Noh, Frank C. Park* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.18677)] \
28 Oct 2023


**Model Selection of Anomaly Detectors in the Absence of Labeled Validation Data** \
*Clement Fung, Chen Qiu, Aodong Li, Maja Rudolph* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.10461)] \
16 Oct 2023

**Boosting Black-box Attack to Deep Neural Networks with Conditional Diffusion Models** \
*Renyang Liu, Wei Zhou, Tianwei Zhang, Kangjie Chen, Jun Zhao, Kwok-Yan Lam* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07492)] \
11 Oct 2023

**Investigating the Adversarial Robustness of Density Estimation Using the Probability Flow ODE** \
*Marius Arvinte, Cory Cornelius, Jason Martin, Nageen Himayat* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.07084)] \
10 Oct 2023

**Understanding and Improving Adversarial Attacks on Latent Diffusion Model** \
*Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2310.04687)] \
7 Oct 2023

**Semantic Adversarial Attacks via Diffusion Models** \
*Chenan Wang, Jinhao Duan, Chaowei Xiao, Edward Kim, Matthew Stamm, Kaidi Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.07398)] \
14 Sep 2023

**Catch You Everything Everywhere: Guarding Textual Inversion via Concept Watermarking** \
*Weitao Feng, Jiyan He, Jie Zhang, Tianwei Zhang, Wenbo Zhou, Weiming Zhang, Nenghai Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05940)] \
12 Sep 2023

**Diff-Privacy: Diffusion-based Face Privacy Protection** \
*Xiao He, Mingrui Zhu, Dongxin Chen, Nannan Wang, Xinbo Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.05330)] \
11 Sep 2023

**DiffDefense: Defending against Adversarial Attacks via Diffusion Models** \
*Hondamunige Prasanna Silva, Lorenzo Seidenari, Alberto Del Bimbo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03702)] [[Github](https://github.com/HondamunigePrasannaSilva/DiffDefence)] \
7 Sep 2023

**My Art My Choice: Adversarial Protection Against Unruly AI** \
*Anthony Rhodes, Ram Bhagat, Umur Aybars Ciftci, Ilke Demir* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.03198)] \
6 Sep 2023

**Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration** \
*Fengfan Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.01582)] \
4 Sep 2023


**Intriguing Properties of Diffusion Models: A Large-Scale Dataset for Evaluating Natural Attack Capability in Text-to-Image Generative Models** \
*Takami Sato, Justin Yue, Nanze Chen, Ningfei Wang, Qi Alfred Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.15692)] \
30 Aug 2023

**DiffSmooth: Certifiably Robust Learning via Diffusion Models and Local Smoothing** \
*Jiawei Zhang, Zhongzhu Chen, Huan Zhang, Chaowei Xiao, Bo Li* \
USENIX Security 2023. [[Paper](https://arxiv.org/abs/2308.14333)] \
28 Aug 2023

**A Probabilistic Fluctuation based Membership Inference Attack for Diffusion Models** \
*Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.12143)] \
23 Aug 2023



**White-box Membership Inference Attacks against Diffusion Models** \
*Yan Pang, Tianhao Wang, Xuhui Kang, Mengdi Huai, Yang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.06405)] \
11 Aug 2023

**BAGM: A Backdoor Attack for Manipulating Text-to-Image Generative Models** \
*Jordan Vice, Naveed Akhtar, Richard Hartley, Ajmal Mian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.16489)] [[Github](https://github.com/JJ-Vice/BAGM)] [[Dataset](https://ieee-datapo