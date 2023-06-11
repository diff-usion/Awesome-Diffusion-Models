[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/hee9joon/Awesome-Diffusion-Models) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

This repository contains a collection of resources and papers on ***Diffusion Models***.

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
*Mengchun Zhang<sup>1</sup>, Maryam Qamar<sup>1</sup>, Taegoo Kang, Yuna Jung, Chenshuang Zhang, Sung-Ho Bae, Chaoning Zhang* \
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
*Ling Yang<sup>1</sup>, Zhilong Zhang<sup>1</sup>, Shenda Hong, Wentao Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.00796)] \
2 Sep 2022

## Vision
### Generation

**Differential Diffusion: Giving Each Pixel Its Strength** \
*Eran Levin, Ohad Fried* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00950)] \
1 Jun 2023

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
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01900)] \
2 Jun 2023

**Conditioning Diffusion Models via Attributes and Semantic Masks for Face Generation** \
*Nico Giambi, Giuseppe Lisanti* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00914)] \
1 Jun 2023


**Addressing Discrepancies in Semantic and Visual Alignment in Neural Networks** \
*Natalie Abreu, Nathan Vaska, Victoria Helus* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01148)] \
1 Jun 2023


**Addressing Negative Transfer in Diffusion Models** \ 
*Hyojun Go<sup>1</sup>, JinYoung Kim<sup>1</sup>, Yunsung Lee<sup>1</sup>, Seunghyun Lee, Shinhyeok Oh, Hyeongdon Moon, Seungtaek Choi* \ 
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
*Paul S. Scotti<sup>1</sup>, Atmadeep Banerjee<sup>1</sup>, Jimmie Goode, Stepan Shabalin, Alex Nguyen, Ethan Cohen, Aidan J. Dempster, Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth A. Norman, Tanishq Mathew Abraham* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18274)] [[Github](https://medarc-ai.github.io/mindeye/)] \
29 May 2023

**Contrast, Attend and Diffuse to Decode High-Resolution Images from Brain Activities** \
*Jingyuan Sun<sup>1</sup>, Mingxiao Li<sup>1</sup>, Zijiao Chen, Yunhao Zhang, Shaonan Wang, Marie-Francine Moens* \
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
*Zijiao Chen<sup>1</sup>, Jiaxin Qing<sup>1</sup>, Juan Helen Zhou* \
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
*Changhao Shi<sup>1</sup>, Haomiao Ni<sup>1</sup>, Kai Li, Shaobo Han, Mingfu Liang, Martin Renqiang Min* \
CVPR Workshop 2023. [[Paper](https://arxiv.org/abs/2304.12536)] \
25 Apr 2023

**Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models** \
*Zhendong Wang, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang Wang, Weizhu Chen, Mingyuan Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12526)] \
25 Apr 2023


**Variational Diffusion Auto-encoder: Deep Latent Variable Model with Unconditional Diffusion Prior** \
*Georgios Batzolis<sup>1</sup>, Jan Stanczuk<sup>1</sup>, Carola-Bibiane Schönlieb* \
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
*Haomiao Ni<sup>1</sup>, Changhao Shi<sup>1</sup>, Kai Li, Sharon X. Huang, Martin Renqiang Min* \
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
*Yong-Hyun Park<sup>1</sup>, Mingi Kwon<sup>1</sup>, Junghyo Jo, Youngjung Uh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12469)] \
24 Feb 2023

**Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC** \
*Yilun Du, Conor Durkan, Robin Strudel, Joshua B. Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, Will Grathwohl* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.11552)] [[Project](https://energy-based-model.github.io/reduce-reuse-recycle/)] \
22 Feb 2023

**Learning 3D Photography Videos via Self-supervised Diffusion on Single Images** \
*Xiaodong Wang<sup>1</sup>, Chenfei Wu<sup>1</sup>, Shengming Yin, Minheng Ni, Jianfeng Wang, Linjie Li, Zhengyuan Yang, Fan Yang, Lijuan Wang, Zicheng Liu, Yuejian Fang, Nan Duan* \
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
*Jaemoo Choi<sup>1</sup>, Yesom Park<sup>1</sup>, Myungjoo Kang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05456)] \
20 Feb 2023



**Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent** \
*Giannis Daras<sup>1</sup>, Yuval Dagan<sup>1</sup>, Alexandros G. Dimakis, Constantinos Daskalakis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.09057)] [[Github](https://github.com/giannisdaras/cdm)] \
17 Feb 2023

**LayoutDiffuse: Adapting Foundational Diffusion Models for Layout-to-Image Generation** \
*Jiaxin Cheng<sup>1</sup>, Xiao Liang<sup>1</sup>, Xingjian Shi, Tong He, Tianjun Xiao, Mu Li* \
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
*Raghav Singhal<sup>1</sup>, Mark Goldstein<sup>1</sup>, Rajesh Ranganath* \
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
*Matthias Bauer<sup>1</sup>, Emilien Dupont, Andy Brock, Dan Rosenbaum, Jonathan Schwarz, Hyunjik Kim<sup>1</sup>* \
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
*Yilun Xu<sup>1</sup>, Shangyuan Tong<sup>1</sup>, Tommi Jaakkola* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2302.00670)] [[Github](https://github.com/Newbeeer/stf)] \
1 Feb 2023

**DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models** \
*Tao Yang, Yuwang Wang, Yan Lv, Nanning Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13721)] \
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
*Emiel Hoogeboom<sup>1</sup>, Jonathan Heek<sup>1</sup>, Tim Salimans* \
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
*Gyeongnyeon Kim<sup>1</sup>, Wooseok Jang<sup>1</sup>, Gyuseong Lee<sup>1</sup>, Susung Hong, Junyoung Seo, Seungryong Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08861)] [[Project](https://ku-cvlab.github.io/DAG/)] \
17 Dec 2022


**Towards Practical Plug-and-Play Diffusion Models** \
*Hyojun Go<sup>1</sup>, Yunsung Lee<sup>1</sup>, Jin-Young Kim<sup>1</sup>, Seunghyun Lee, Myeongho Jeong, Hyun Seung Lee, Seungtaek Choi* \
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
*Dongjun Kim<sup>1</sup>, Yeongmin Kim<sup>1</sup>, Wanmo Kang, Il-Chul Moon* \
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
*Ling Yang<sup>1</sup>, Zhilin Huang<sup>1</sup>, Yang Song, Shenda Hong, Guohao Li, Wentao Zhang, Bin Cui, Bernard Ghanem, Ming-Hsuan Yang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11138)] \
21 Nov 2022

**SinFusion: Training Diffusion Models on a Single Image or Video** \
*Yaniv Nikankin, Niv Haim, Michal Irani* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11743)] \
21 Nov 2022

**MagicVideo: Efficient Video Generation With Latent Diffusion Models** \
*Daquan Zhou<sup>1</sup>, Weimin Wang<sup>1</sup>, Hanshu Yan, Weiwei Lv, Yizhe Zhu, Jiashi Feng* \
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
*Vincent Tao Hu<sup>1</sup>, David W Zhang<sup>1</sup>, Yuki M. Asano, Gertjan J. Burghouts, Cees G. M. Snoek* \
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
*Shin-I Cheng<sup>1</sup>, Yu-Jie Chen<sup>1</sup>, Wei-Chen Chiu, Hsin-Ying Lee, Hung-Yu Tseng* \
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
*Xiulong Yang<sup>1</sup>, Sheng-Min Shih<sup>1</sup>, Yinlin Fu, Xiaoting Zhao, Shihao Ji* \
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
*Vikram Voleti<sup>1</sup>, Alexia Jolicoeur-Martineau<sup>1</sup>, Christopher Pal* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.09853)] [[Github](https://github.com/voletiv/mcvd-pytorch)] \
19 May 2022

**On Conditioning the Input Noise for Controlled Image Generation with Diffusion Models** \
*Vedant Singh<sup>1</sup>, Surgan Jandial<sup>1</sup>, Ayush Chopra, Siddharth Ramesh, Balaji Krishnamurthy, Vineeth N. Balasubramanian* \
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
*Andreas Blattmann<sup>1</sup>, Robin Rombach<sup>1</sup>, Kaan Oktay, Jonas Müller, Björn Ommer* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2204.11824)] \
25 Apr 2022


**Video Diffusion Models** \
*Jonathan Ho<sup>1</sup>, Tim Salimans<sup>1</sup>, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet* \
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
*Alex Nichol<sup>1</sup>, Prafulla Dhariwal<sup>1</sup>, Aditya Ramesh<sup>1</sup>, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen* \
ICML 2021. [[Paper](https://arxiv.org/abs/2112.10741)] [[Github](https://github.com/openai/glide-text2im)] \
20 Dec 2021

**High-Resolution Image Synthesis with Latent Diffusion Models** \
*Robin Rombach<sup>1</sup>, Andreas Blattmann<sup>1</sup>, Dominik Lorenz, Patrick Esser, Björn Ommer* \
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
*Sam Bond-Taylor<sup>1</sup>, Peter Hessey<sup>1</sup>, Hiroshi Sasaki, Toby P. Breckon, Chris G. Willcocks* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.12701)] [[Github](https://github.com/samb-t/unleashing-transformers)] \
24 Nov 2021

**Diffusion Normalizing Flow** \
*Qinsheng Zhang, Yongxin Chen* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2110.07579)] [[Github](https://github.com/qsh-zh/DiffFlow)] \
14 Oct 2021

**Denoising Diffusion Gamma Models** \
*Eliya Nachmani<sup>1</sup>, Robin San Roman<sup>1</sup>, Lior Wolf* \
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
*Patrick Esser<sup>1</sup>, Robin Rombach<sup>1</sup>, Andreas Blattmann<sup>1</sup>, Björn Ommer* \
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
*Jacob Austin<sup>1</sup>, Daniel D. Johnson<sup>1</sup>, Jonathan Ho, Daniel Tarlow, Rianne van den Berg* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2107.03006)] \
7 Jul 2021 

**Variational Diffusion Models** \
*Diederik P. Kingma<sup>1</sup>, Tim Salimans<sup>1</sup>, Ben Poole, Jonathan Ho* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.00630)] [[Github](https://github.com/google-research/vdm)] \
1 Jul 2021 

**Diffusion Priors In Variational Autoencoders** \
*Antoine Wehenkel<sup>1</sup>, Gilles Louppe<sup>1</sup>* \
ICML Workshop 2021. [[Paper](https://arxiv.org/abs/2106.15671)] \
29 Jun 2021

**Deep Generative Learning via Schrödinger Bridge** \
*Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, Can Yang* \
ICML 2021. [[Paper](https://arxiv.org/abs/2106.10410)] \
19 Jun 2021

**Non Gaussian Denoising Diffusion Models** \
*Eliya Nachmani<sup>1</sup>, Robin San Roman<sup>1</sup>, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.07582)] [[Project](https://enk100.github.io/Non-Gaussian-Denoising-Diffusion-Models/)] \
14 Jun 2021 

**D2C: Diffusion-Denoising Models for Few-shot Conditional Generation** \
*Abhishek Sinha<sup>1</sup>, Jiaming Song<sup>1</sup>, Chenlin Meng, Stefano Ermon* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2106.06819)] [[Project](https://d2c-model.github.io/)] [[Github](https://github.com/d2c-model/d2c-model.github.io)] \
12 Jun 2021

**Score-based Generative Modeling in Latent Space** \
*Arash Vahdat<sup>1</sup>, Karsten Kreis<sup>1</sup>, Jan Kautz* \
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
*Jonathan Ho<sup>1</sup>, Chitwan Saharia<sup>1</sup>, William Chan, David J. Fleet, Mohammad Norouzi, Tim Salimans* \
JMLR 2021. [[Paper](https://arxiv.org/abs/2106.15282)] [[Project](https://cascaded-diffusion.github.io/)] \
30 May 2021 

**Gotta Go Fast When Generating Data with Score-Based Models** \
*Alexia Jolicoeur-Martineau, Ke Li, Rémi Piché-Taillefer, Tal Kachman, Ioannis Mitliagkas* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.14080)] [[Github](https://github.com/AlexiaJM/score_sde_fast_sampling)] \
28 May 2021

**Diffusion Models Beat GANs on Image Synthesis** \
*Prafulla Dhariwal<sup>1</sup>, Alex Nichol<sup>1</sup>* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.05233)] [[Github](https://github.com/openai/guided-diffusion)] \
11 May 2021 

**Image Super-Resolution via Iterative Refinement** \
*Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.07636)] [[Project](https://iterative-refinement.github.io/)] [[Github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] \
15 Apr 2021 

**Noise Estimation for Generative Diffusion Models** \
*Robin San-Roman<sup>1</sup>, Eliya Nachmani<sup>1</sup>, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.02600)] \
6 Apr 2021 

**Improved Denoising Diffusion Probabilistic Models** \
*Alex Nichol<sup>1</sup>, Prafulla Dhariwal<sup>1</sup>* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2102.09672)] [[Github](https://github.com/openai/improved-diffusion)] \
18 Feb 2021 

**Maximum Likelihood Training of Score-Based Diffusion Models** \
*Yang Song<sup>1</sup>, Conor Durkan<sup>1</sup>, Iain Murray, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2101.09258)] \
22 Jan 2021 

**Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed** \
*Eric Luhman<sup>1</sup>, Troy Luhman<sup>1</sup>* \
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
*Alexia Jolicoeur-Martineau<sup>1</sup>, Rémi Piché-Taillefer<sup>1</sup>, Rémi Tachet des Combes, Ioannis Mitliagkas* \
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

**DiffCLIP: Leveraging Stable Diffusion for Language Grounded 3D Classification** \
*Sitian Shen, Zilin Zhu, Linqian Fan, Harry Zhang, Xinxiao Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15957)] \
25 May 2023


**Training on Thin Air: Improve Image Classification with Generated Data** \
*Yongchao Zhou, Hshmat Sahak, Jimmy Ba* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15316)] [[Project](https://sites.google.com/view/diffusion-inversion)] [[Github](https://github.com/yongchao97/diffusion_inversion)] \
24 May 2023

**Is Synthetic Data From Diffusion Models Ready for Knowledge Distillation?** \
*Zheng Li<sup>1</sup>, Yuxuan Li<sup>1</sup>, Penghai Zhao, Renjie Song, Xiang Li, Jian Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12954)] [[Github](https://github.com/zhengli97/DM-KD)] \
22 May 2023

**Boosting Human-Object Interaction Detection with Text-to-Image Diffusion Model** \
*Jie Yang<sup>1</sup>, Bingliang Li<sup>1</sup>, Fengyu Yang, Ailing Zeng, Lei Zhang, Ruimao Zhang* \
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
*Siyuan Li<sup>1</sup>, Tobias Fischer<sup>1</sup>, Lei Ke, Henghui Ding, Martin Danelljan, Fisher Yu* \
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
*Elvis Nava<sup>1</sup>, Seijin Kobayashi<sup>1</sup>, Yifei Yin, Robert K. Katzschmann, Benjamin F. Grewe* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.08942)] \
17 Oct 2022


### Segmentation

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

**Domain-agnostic segmentation of thalamic nuclei from joint structural and diffusion MRI** \
*Henry F. J. Tregidgo, Sonja Soskic, Mark D. Olchanyi, Juri Althonayan, Benjamin Billot, Chiara Maffei, Polina Golland, Anastasia Yendiki, Daniel C. Alexander, Martina Bocchetta, Jonathan D. Rohrer, Juan Eugenio Iglesias* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03413)] \
5 May 2023

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
*Chaofan Ma<sup>1</sup>, Yuhuan Yang<sup>1</sup>, Chen Ju, Fei Zhang, Jinxiang Liu, Yu Wang, Ya Zhang, Yanfeng Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09813)] \
17 Mar 2023

**Stochastic Segmentation with Conditional Categorical Diffusion Models** \
*Lukas Zbinden<sup>1</sup>, Lars Doorenbos<sup>1</sup>, Theodoros Pissas, Raphael Sznitman, Pablo Márquez-Neila* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08888)] [[Github](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)] \
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
*Boah Kim<sup>1</sup>, Yujin Oh<sup>1</sup>, Jong Chul Ye* \
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
*Julia Wolleb<sup>1</sup>, Robin Sandkühler<sup>1</sup>, Florentin Bieder, Philippe Valmaggia, Philippe C. Cattin* \
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

**Unpaired Image-to-Image Translation via Neural Schr\"odinger Bridge** \
*Beomsu Kim<sup>1</sup>, Gihyun Kwon<sup>1</sup>, Kwanyoung Kim, Jong Chul Ye* \
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
*Bohan Zeng<sup>1</sup>, Xuhui Liu<sup>1</sup>, Sicheng Gao<sup>1</sup>, Boyu Liu, Hong Li, Jianzhuang Liu, Baochang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03199)] \
6 Apr 2023



**Reference-based Image Composition with Sketch via Structure-aware Diffusion Model** \
*Kangyeol Kim, Sunghyun Park, Junsoo Lee, Jaegul Choo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.09748)] \
31 Mar 2023

**Training-free Style Transfer Emerges from h-space in Diffusion models** \
*Jaeseok Jeong<sup>1</sup>, Mingi Kwon<sup>1</sup>, Youngjung Uh* \
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
*Shidong Cao<sup>1</sup>, Wenhao Chai<sup>1</sup>, Shengyu Hao, Yanting Zhang, Hangyue Chen, Gaoang Wang* \
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
*Junyoung Seo<sup>1</sup>, Gyuseong Lee<sup>1</sup>, Seokju Cho, Jiyoung Lee, Seungryong Kim* \
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
*Julia Wolleb<sup>1</sup>, Robin Sandkühler<sup>1</sup>, Florentin Bieder, Philippe C. Cattin* \
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
*Hyungjin Chung<sup>1</sup>, Jeongsol Kim<sup>1</sup>, Jong Chul Ye* \
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
*Yi Huang<sup>1</sup>, Jiancheng Huang<sup>1</sup>, Jianzhuang Liu, Yu Dong, Jiaxi Lv, Shifeng Chen* \
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
*Weihang Mao<sup>1</sup>, Bo Han<sup>1</sup>, Zihao Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03174)] \
6 Apr 2023


**Inst-Inpaint: Instructing to Remove Objects with Diffusion Models** \
*Ahmet Burak Yildirim, Vedat Baday, Erkut Erdem, Aykut Erdem, Aysegul Dundar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03246)] [[Project](http://instinpaint.abyildirim.com/)] \
6 Apr 2023

**Towards Coherent Image Inpainting Using Denoising Diffusion Implicit Models** \
*Guanhua Zhang<sup>1</sup>, Jiabao Ji<sup>1</sup>, Yang Zhang, Mo Yu, Tommi Jaakkola, Shiyu Chang* \
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
*Ben Fei<sup>1</sup>, Zhaoyang Lyu<sup>1</sup>, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, Bo Dai* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.01247)] \
3 Apr 2023

**Implicit Diffusion Models for Continuous Super-Resolution** \
*Sicheng Gao<sup>1</sup>, Xuhui Liu<sup>1</sup>, Bohan Zeng<sup>1</sup>, Sheng Xu, Yanjing Li, Xiaoyan Luo, Jianzhuang Liu, Xiantong Zhen, Baochang Zhang* \
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
*Ye Mao<sup>1</sup>, Lan Jiang<sup>1</sup>, Xi Chen, Chao Li* \
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
*Zhenzhen Liu<sup>1</sup>, Jin Peng Zhou<sup>1</sup>, Yufan Wang, Kilian Q. Weinberger* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10326)] \
20 Feb 2023

**Restoration based Generative Models** \
*Jaemoo Choi<sup>1</sup>, Yesom Park<sup>1</sup>, Myungjoo Kang* \
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
*Tristan S.W. Stevens<sup>1</sup>, Jean-Luc Robert<sup>1</sup>, Faik C. Meral Jason Yu, Jun Seob Shin, Ruud J.G. van Sloun* \
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
*Yinhuai Wang<sup>1</sup>, Jiwen Yu<sup>1</sup>, Jian Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00490)] [[Github](https://github.com/wyhuai/DDNM)] \
1 Dec 2022


**FREDSR: Fourier Residual Efficient Diffusive GAN for Single Image Super Resolution** \
*Kyoungwan Woo<sup>1</sup>, Achyuta Rajaram<sup>1</sup>* \
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
*Hyungjin Chung<sup>1</sup>, Jeongsol Kim<sup>1</sup>, Sehui Kim, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10656)] \
19 Nov 2022


**Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models** \
*Hyungjin Chung<sup>1</sup>, Dohoon Ryu<sup>1</sup>, Michael T. McCann, Marc L. Klasky, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10655)] \
19 Nov 2022

**Patch-Based Denoising Diffusion Probabilistic Model for Sparse-View CT Reconstruction** \
*Wenjun Xia, Wenxiang Cong, Ge Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10388)] \
18 Nov 2022


**A Structure-Guided Diffusion Model for Large-Hole Diverse Image Completion** \
*Daichi Horita, Jiaolong Yang, Dong Chen, Yuki Koyama, Kiyoharu Aizawa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10437)] \
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
*Pouria Rouzrokh<sup>1</sup>, Bardia Khosravi<sup>1</sup>, Shahriar Faghani, Mana Moassefi, Sanaz Vahdati, Bradley J. Erickson* \
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
*Hyungjin Chung<sup>1</sup>, Jeongsol Kim<sup>1</sup>, Michael T. Mccann, Marc L. Klasky, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14687)] [[Github](https://github.com/DPS2022/diffusion-posterior-sampling)] \
29 Sep 2022

**Face Super-Resolution Using Stochastic Differential Equations** \
*Marcelo dos Santos<sup>1</sup>, Rayson Laroca<sup>1</sup>, Rafael O. Ribeiro, João Neves, Hugo Proença, David Menotti* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.12064)] [[Github](https://github.com/marcelowds/sr-sde)] \
24 Sep 2022


**JPEG Artifact Correction using Denoising Diffusion Restoration Models** \
*Bahjat Kawar<sup>1</sup>, Jiaming Song<sup>1</sup>, Stefano Ermon, Michael Elad* \
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
*Hyungjin Chung<sup>1</sup>, Byeongsu Sim<sup>1</sup>, Dohoon Ryu, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00941)] \
2 Jun 2022


**The Swiss Army Knife for Image-to-Image Translation: Multi-Task Diffusion Models** \
*Julia Wolleb<sup>1</sup>, Robin Sandkühler<sup>1</sup>, Florentin Bieder, Philippe C. Cattin* \
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
*Robin Rombach<sup>1</sup>, Andreas Blattmann<sup>1</sup>, Dominik Lorenz, Patrick Esser, Björn Ommer* \
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
*Yang Song<sup>1</sup>, Liyue Shen<sup>1</sup>, Lei Xing, Stefano Ermon* \
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
*Jonathan Ho<sup>1</sup>, Chitwan Saharia<sup>1</sup>, William Chan, David J. Fleet, Mohammad Norouzi, Tim Salimans* \
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

**Conditional Generation from Unconditional Diffusion Models using Denoiser Representations** \
*Alexandros Graikos, Srikar Yellapragada, Dimitris Samaras* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01900)] \
2 Jun 2023


**Unsupervised Anomaly Detection in Medical Images Using Masked Diffusion Model** \
*Hasan Iqbal<sup>1</sup>, Umar Khalid<sup>1</sup>, Jing Hua, Chen Chen* \
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
*Ibrahim Ethem Hamamci<sup>1</sup>, Sezgin Er<sup>1</sup>, Enis Simsar, Alperen Tezcan, Ayse Gulnihan Simsek, Furkan Almas, Sevval Nil Esirgun, Hadrien Reynaud, Sarthak Pati, Christian Bluethgen, Bjoern Menze* \
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

**Domain-agnostic segmentation of thalamic nuclei from joint structural and diffusion MRI** \
*Henry F. J. Tregidgo, Sonja Soskic, Mark D. Olchanyi, Juri Althonayan, Benjamin Billot, Chiara Maffei, Polina Golland, Anastasia Yendiki, Daniel C. Alexander, Martina Bocchetta, Jonathan D. Rohrer, Juan Eugenio Iglesias* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03413)] \
5 May 2023


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
*Roman Macháček<sup>1</sup>, Leila Mozaffari<sup>1</sup>, Zahra Sepasdar, Sravanthi Parasa, Pål Halvorsen, Michael A. Riegler, Vajira Thambawita* \
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
*Ye Mao<sup>1</sup>, Lan Jiang<sup>1</sup>, Xi Chen, Chao Li* \
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
*Suhyeon Lee<sup>1</sup>, Hyungjin Chung<sup>1</sup>, Minyoung Park, Jonghyuk Park, Wi-Sun Ryu, Jong Chul Ye* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08440)] \
15 Mar 2023

**Class-Guided Image-to-Image Diffusion: Cell Painting from Brightfield Images with Class Labels** \
*Jan Oscar Cross-Zamirski, Praveen Anand, Guy Williams, Elizabeth Mouchet, Yinhai Wang, Carola-Bibiane Schönlieb* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08863)] [[Github](https://github.com/crosszamirski/guided-I2I)] \
15 Mar 2023


**Stochastic Segmentation with Conditional Categorical Diffusion Models** \
*Lukas Zbinden<sup>1</sup>, Lars Doorenbos<sup>1</sup>, Theodoros Pissas, Raphael Sznitman, Pablo Márquez-Neila* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08888)] [[Github](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)] \
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
*Mohamed Akrout<sup>1</sup>, Bálint Gyepesi<sup>1</sup>, Péter Holló, Adrienn Poór, Blága Kincső, Stephen Solis, Katrina Cirone, Jeremy Kawahara, Dekker Slade, Latif Abid, Máté Kovács, István Fazekas* \
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
*Luke W. Sagers<sup>1</sup>, James A. Diao<sup>1</sup>, Matthew Groh, Pranav Rajpurkar, Adewole S. Adamson, Arjun K. Manrai* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2211.13352)] \
23 Nov 2022

**RoentGen: Vision-Language Foundation Model for Chest X-ray Generation** \
*Pierre Chambon<sup>1</sup>, Christian Bluethgen<sup>1</sup>, Jean-Benoit Delbrouck, Rogier Van der Sluijs, Małgorzata Połacin, Juan Manuel Zambrano Chaves, Tanishq Mathew Abraham, Shivanshu Purohit, Curtis P. Langlotz, Akshay Chaudhari* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12737)] \
23 Nov 2022

**DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction** \
*Jiaming Liu, Rushil Anirudh, Jayaraman J. Thiagarajan, Stewart He, K. Aditya Mohan, Ulugbek S. Kamilov, Hyojin Kim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12340)] \
22 Nov 2022



**Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models** \
*Hyungjin Chung<sup>1</sup>, Dohoon Ryu<sup>1</sup>, Michael T. McCann, Marc L. Klasky, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10655)] \
19 Nov 2022

**Patch-Based Denoising Diffusion Probabilistic Model for Sparse-View CT Reconstruction** \
*Wenjun Xia, Wenxiang Cong, Ge Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10388)] \
18 Nov 2022


**Brain PET Synthesis from MRI Using Joint Probability Distribution of Diffusion Model at Ultrahigh Fields** \
*Xie Taofeng<sup>1</sup>, Cao Chentao<sup>1</sup>, Cui Zhuoxu, Li Fanshi, Wei Zidong, Zhu Yanjie, Li Ye, Liang Dong, Jin Qiyu, Chen Guoqing, Wang Haifeng* \
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
*Pouria Rouzrokh<sup>1</sup>, Bardia Khosravi<sup>1</sup>, Shahriar Faghani, Mana Moassefi, Sanaz Vahdati, Bradley J. Erickson* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.12113)] [[Github](https://github.com/Mayo-Radiology-Informatics-Lab/MBTI)] \
21 Oct 2022


**Adapting Pretrained Vision-Language Foundational Models to Medical Imaging Domains** \
*Pierre Chambon<sup>1</sup>, Christian Bluethgen<sup>1</sup>, Curtis P. Langlotz, Akshay Chaudhari* \
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
*Boah Kim<sup>1</sup>, Yujin Oh<sup>1</sup>, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14566)] \
29 Sep 2022

**Conversion Between CT and MRI Images Using Diffusion and Score-Matching Models** \
*Qing Lyu, Ge Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.12104)] \
24 Sep 2022

**Brain Imaging Generation with Latent Diffusion Models** \
*Walter H. L. Pinaya<sup>1</sup>, Petru-Daniel Tudosiu<sup>1</sup>, Jessica Dafflon, Pedro F da Costa, Virginia Fernandez, Parashkev Nachev, Sebastien Ourselin, M. Jorge Cardoso* \
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
*Hyungjin Chung<sup>1</sup>, Byeongsu Sim<sup>1</sup>, Dohoon Ryu, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00941)] \
2 Jun 2022

**AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise** \
*Julian Wyatt, Adam Leach, Sebastian M. Schmon, Chris G. Willcocks* \
CVPR Workshop 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf)] [[Github](https://github.com/Julian-Wyatt/AnoDDPM)] \
1 Jun 2022

**The Swiss Army Knife for Image-to-Image Translation: Multi-Task Diffusion Models** \
*Julia Wolleb<sup>1</sup>, Robin Sandkühler<sup>1</sup>, Florentin Bieder, Philippe C. Cattin* \
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
*Yang Song<sup>1</sup>, Liyue Shen<sup>1</sup>, Lei Xing, Stefano Ermon* \
NeurIPS Workshop 2021. [[Paper](https://arxiv.org/abs/2111.08005)] [[Github](https://github.com/yang-song/score_inverse_problems)] \
15 Nov 2021

**Score-based diffusion models for accelerated MRI** \
*Hyungjin Chung, Jong chul Ye* \
MIA 2021. [[Paper](https://arxiv.org/abs/2110.05243)] [[Github](https://github.com/HJ-harry/score-MRI)] \
8 Oct 2021




### Multi-modal Learning

**Stable Diffusion is Unstable** \
*Chengbin Du, Yanxi Li, Zhongwei Qiu, Chang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02583)] \
5 Jun 2023

**LipVoicer: Generating Speech from Silent Videos Guided by Lip Reading** \
*Yochai Yemini, Aviv Shamsian, Lior Bracha, Sharon Gannot, Ethan Fetaya* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03258)] [[Project](https://lipvoicer.github.io/)] \
5 Jun 2023

**HeadSculpt: Crafting 3D Head Avatars with Text** \
*Xiao Han<sup>1</sup>, Yukang Cao<sup>1</sup>, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, Kwan-Yee K. Wong* \
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


**Efficient Text-Guided 3D-Aware Portrait Generation with Score Distillation Sampling on Distribution** \
*Yiji Cheng<sup>1</sup>, Fei Yin<sup>1</sup>, Xiaoke Huang, Xintong Yu, Jiaxiang Liu, Shikun Feng, Yujiu Yang, Yansong Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02083)] \
3 Jun 2023

**Probabilistic Adaptation of Text-to-Video Models** \
*Mengjiao Yang<sup>1</sup>, Yilun Du<sup>1</sup>, Bo Dai, Dale Schuurmans, Joshua B. Tenenbaum, Pieter Abbeel* \
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
*Virginia Fernandez<sup>1</sup>, Pedro Sanchez<sup>1</sup>, Walter Hugo Lopez Pinaya, Grzegorz Jacenków, Sotirios A. Tsaftaris, Jorge Cardoso* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01322)] \
2 Jun 2023

**StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners** \
*Yonglong Tian<sup>1</sup>, Lijie Fan<sup>1</sup>, Phillip Isola, Huiwen Chang, Dilip Krishnan* \
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
*Chang Liu<sup>1</sup>, Haoning Wu<sup>1</sup>, Yujie Zhong, Xiaoyun Zhang, Weidi Xie* \
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
*Pablo Pernias<sup>1</sup>, Dominic Rampas<sup>1</sup>, Marc Aubreville* \
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
*Zeyue Xue<sup>1</sup>, Guanglu Song<sup>1</sup>, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, Ping Luo* \
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
*Hyunsoo Lee<sup>1</sup>, Minsoo Kang<sup>1</sup>, Bohyung Han* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18007)] \
29 May 2023

**Text-to-image Editing by Image Information Removal** \
*Zhongping Zhang, Jian Zheng, Jacob Zhiyuan Fang, Bryan A. Plummer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17489)] \
27 May 2023

**Towards Consistent Video Editing with Text-to-Image Diffusion Models** \
*Zicheng Zhang<sup>1</sup>, Bonan Li<sup>1</sup>, Xuecheng Nie, Congying Han, Tiande Guo, Luoqi Liu* \
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
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16311)] [[Project](https://omriavrahami.com/break-a-scene/)] \
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
*Marco Bellagente<sup>1</sup>, Manuel Brack<sup>1</sup>, Hannah Teufel<sup>1</sup>, Felix Friedrich, Björn Deiseroth, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Koen Oostermeijer, Andres Felipe Cruz-Salinas, Patrick Schramowski, Kristian Kersting, Samuel Weinbach* \ 
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15296)] \ 
24 May 2023

**ChatFace: Chat-Guided Real Face Editing via Diffusion Latent Space Manipulation** \ 
*Dongxu Yue, Qin Guo, Munan Ning, Jiaxi Cui, Yuesheng Zhu, Li Yuan* \ 
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14742)] \ 
24 May 2023

**DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models** \
*Sungnyun Kim<sup>1</sup>, Junsoo Lee<sup>1</sup>, Kibeom Hong, Daesik Kim, Namhyuk Ahn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15194)] [[Github](https://github.com/sungnyun/diffblender)] \
24 May 2023

**I Spy a Metaphor: Large Language Models and Diffusion Models Co-Create Visual Metaphors** \
*Tuhin Chakrabarty<sup>1</sup>, Arkadiy Saakyan<sup>1</sup>, Olivia Winn<sup>1</sup>, Artemis Panagopoulou, Yue Yang, Marianna Apidianaki, Smaranda Muresan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14724)] \
24 May 2023

**BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing** \
*Dongxu Li, Junnan Li, Steven C. H. Hoi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14720)] \
24 May 2023

**Adversarial Nibbler: A Data-Centric Challenge for Improving the Safety of Text-to-Image Models** \
*Alicia Parrish<sup>1</sup>, Hannah Rose Kirk<sup>1</sup>, Jessica Quaye<sup>1</sup>, Charvi Rastogi<sup>1</sup>, Max Bartolo<sup>1</sup>, Oana Inel<sup>1</sup>, Juan Ciro, Rafael Mosquera, Addison Howard, Will Cukierski, D. Sculley, Vijay Janapa Reddi, Lora Aroyo* \
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

**Training Diffusion Models with Reinforcement Learning** \ 
*Kevin Black<sup>1</sup>, Michael Janner<sup>1</sup>, Yilun Du, Ilya Kostrikov, Sergey Levine* \ 
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
*Azade Farshad<sup>1</sup>, Yousef Yeganeh<sup>1</sup>, Yu Chi, Chengzhi Shen, Björn Ommer, Nassir Navab* \
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
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08870)] [[Github](https://github.com/soon-yau/upgpt)] \
18 Apr 2023

**TTIDA: Controllable Generative Data Augmentation via Text-to-Text and Text-to-Image Models** \
*Yuwei Yin, Jean Kaddour, Xiang Zhang, Yixin Nie, Zhenguang Liu, Lingpeng Kong, Qi Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08821)] \
18 Apr 2023

**Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models** \
*Andreas Blattmann<sup>1</sup>, Robin Rombach<sup>1</sup>, Huan Ling<sup>1</sup>, Tim Dockhorn<sup>1</sup>, Seung Wook Kim, Sanja Fidler, Karsten Kreis* \
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
*Samaneh Azadi<sup>1</sup>, Thomas Hayes<sup>1</sup>, Akbar Shah, Guan Pang, Devi Parikh, Sonal Gupta* \
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
*Qiucheng Wu<sup>1</sup>, Yujian Liu<sup>1</sup>, Handong Zhao, Trung Bui, Zhe Lin, Yang Zhang, Shiyu Chang* \
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
*Mohammadreza Mofayezi<sup>1</sup>, Yasamin Medghalchi<sup>1</sup>* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02963)] \
6 Apr 2023


**DITTO-NeRF: Diffusion-based Iterative Text To Omni-directional 3D Model** \
*Hoigi Seo<sup>1</sup>, Hayeon Kim<sup>1</sup>, Gwanghyun Kim, Se Young Chun* \
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
*Alberto Baldrati<sup>1</sup>, Davide Morelli<sup>1</sup>, Giuseppe Cartella, Marcella Cornia, Marco Bertini, Rita Cucchiara* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02051)] \
4 Apr 2023

**PODIA-3D: Domain Adaptation of 3D Generative Model Across Large Domain Gap Using Pose-Preserved Text-to-Image Diffusion** \
*Gwanghyun Kim, Ji Ha Jang, Se Young Chun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01900)] [[Project](https://gwang-kim.github.io/podia_3d/)] \
4 Apr 2023

**Text-Conditioned Sampling Framework for Text-to-Image Generation with Masked Generative Models** \
*Jaewoong Lee<sup>1</sup>, Sangwon Jang<sup>1</sup>, Jaehyeong Jo, Jaehong Yoon, Yunji Kim, Jin-Hwa Kim, Jung-Woo Ha, Sung Ju Hwang* \
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
*Longwen Zhang<sup>1</sup>, Qiwei Qiu<sup>1</sup>, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, Jingyi Yu* \
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
*Vidit Goel<sup>1</sup>, Elia Peruzzo<sup>1</sup>, Yifan Jiang, Dejia Xu, Nicu Sebe, Trevor Darrell, Zhangyang Wang, Humphrey Shi* \
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
*Wen Wang<sup>1</sup>, Kangyang Xie<sup>1</sup>, Zide Liu<sup>1</sup>, Hao Chen, Yue Cao, Xinlong Wang, Chunhua Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17599)] \
30 Mar 2023


**Discriminative Class Tokens for Text-to-Image Diffusion Models** \
*Idan Schwartz<sup>1</sup>, Vésteinn Snæbjarnarson<sup>1</sup>, Sagie Benaim, Hila Chefer, Ryan Cotterell, Lior Wolf, Serge Belongie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17155)] \
30 Mar 2023



**DAE-Talker: High Fidelity Speech-Driven Talking Face Generation with Diffusion Autoencoder** \
*Chenpng Du, Qi Chen, Tianyu He, Xu Tan, Xie Chen, Kai Yu, Sheng Zhao, Jiang Bian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17550)] \
30 Mar 2023

**LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation** \
*Guangcong Zheng<sup>1</sup>, Xianpan Zhou<sup>1</sup>, Xuewei Li, Zhongang Qi, Ying Shan, Xi Li* \
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
*Susung Hong<sup>1</sup>, Donghoon Ahn<sup>1</sup>, Seungryong Kim* \
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
*Yiqi Lin<sup>1</sup>, Haotian Bai<sup>1</sup>, Sijia Li, Haonan Lu, Xiaodong Lin, Hui Xiong, Lin Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13843)] [[Project](https://fantasia3d.github.io/)] \
24 Mar 2023

**Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation** \
*Rui Chen, Yongwei Chen, Ningxin Jiao, Kui Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13873)] \
24 Mar 2023

**ReVersion: Diffusion-Based Relation Inversion from Images** \
*Ziqi Huang<sup>1</sup>, Tianxing Wu<sup>1</sup>, Yuming Jiang, Kelvin C.K. Chan, Ziwei Liu* \
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
*Juil Koo<sup>1</sup>, Seungwoo Yoo<sup>1</sup>, Minh Hieu Nguyen<sup>1</sup>, Minhyuk Sung* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12236)] [[Project](https://salad3d.github.io/)] \
21 Mar 2023

**Vox-E: Text-guided Voxel Editing of 3D Objects** \
*Etai Sella, Gal Fiebelman, Peter Hedman, Hadar Averbuch-Elor* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12048)] [[Project](https://tau-vailab.github.io/Vox-E/)] \
21 Mar 2023

**CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion** \
*Geonmo Gu<sup>1</sup>, Sanghyuk Chun<sup>1</sup>, Wonjae Kim, HeeJae Jun, Yoohoon Kang, Sangdoo Yun* \
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
*Jingxuan Wei<sup>1</sup>, Shiyu Wu<sup>1</sup>, Xin Jiang, Yequan Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10073)] \
17 Mar 2023

**GlueGen: Plug and Play Multi-modal Encoders for X-to-image Generation** \
*Can Qin, Ning Yu, Chen Xing, Shu Zhang, Zeyuan Chen, Stefano Ermon, Yun Fu, Caiming Xiong, Ran Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10056)] \
17 Mar 2023

**DiffusionRet: Generative Text-Video Retrieval with Diffusion Model** \
*Peng Jin<sup>1</sup>, Hao Li<sup>1</sup>, Zesen Cheng, Kehan Li, Xiangyang Ji, Chang Liu, Li Yuan, Jie Chen* \
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
*Shu Zhang<sup>1</sup>, Xinyi Yang<sup>1</sup>, Yihao Feng<sup>1</sup>, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, Caiming Xiong, Ran Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09618)] \
16 Mar 2023


**P+: Extended Textual Conditioning in Text-to-Image Generation** \
*Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, Kfir Aberman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09522)] [[Project](https://prompt-plus.github.io/)] \
16 Mar 2023

**Highly Personalized Text Embedding for Image Manipulation by Stable Diffusion** \
*Inhwa Han<sup>1</sup>, Serin Yang<sup>1</sup>, Taesung Kwon, Jong Chul Ye* \
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
*Chaehun Shin<sup>1</sup>, Heeseung Kim<sup>1</sup>, Che Hyun Lee, Sang-gil Lee, Sungroh Yoon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07945)] [[Project](https://edit-a-video.github.io/)] \
14 Mar 2023

**Editing Implicit Assumptions in Text-to-Image Diffusion Models** \
*Hadas Orgad<sup>1</sup>, Bahjat Kawar<sup>1</sup>, Yonatan Belinkov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08084)] [[Project](https://time-diffusion.github.io/)] [[Github](https://github.com/bahjat-kawar/time-diffusion)] \
14 Mar 2023



**Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation** \
*Junyoung Seo<sup>1</sup>, Wooseok Jang<sup>1</sup>, Min-Seop Kwak<sup>1</sup>, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, Seungryong Kim* \
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
*Rohit Gandikota<sup>1</sup>, Joanna Materzynska<sup>1</sup>, Jaden Fiotto-Kaufman, David Bau* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07345)] [[Project](https://erasing.baulab.info/)] [[Github](https://github.com/rohitgandikota/erasing)] \
13 Mar 2023


**One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale** \
*Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, Jun Zhu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06555)] [[Github](https://github.com/thu-ml/unidiffuser)] \
12 Mar 2023

**Cones: Concept Neurons in Diffusion Models for Customized Generation** \
*Zhiheng Liu<sup>1</sup>, Ruili Feng<sup>1</sup>, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao* \
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
*Wenliang Zhao<sup>1</sup>, Yongming Rao<sup>1</sup>, Zuyan Liu<sup>1</sup>, Benlin Liu, Jie Zhou, Jiwen Lu* \
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
*Xiaodong Wang<sup>1</sup>, Chenfei Wu<sup>1</sup>, Shengming Yin, Minheng Ni, Jianfeng Wang, Linjie Li, Zhengyuan Yang, Fan Yang, Lijuan Wang, Zicheng Liu, Yuejian Fang, Nan Duan* \
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
*Omer Bar-Tal<sup>1</sup>, Lior Yariv<sup>1</sup>, Yaron Lipman, Tali Dekel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08113)] [Project](https://multidiffusion.github.io/)] [[Github](https://github.com/omerbt/MultiDiffusion)] \
16 Feb 2023

**Boundary Guided Mixing Trajectory for Semantic Control with Diffusion Models** \
*Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, Yan Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08357)] \
16 Feb 2023

**Dataset Interfaces: Diagnosing Model Failures Using Controllable Counterfactual Generation** \
*Joshua Vendrow<sup>1</sup>, Saachi Jain<sup>1</sup>, Logan Engstrom, Aleksander Madry* \
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
*Arpit Bansal<sup>1</sup>, Hong-Min Chu<sup>1</sup>, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, Tom Goldstein* \
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
*Anton Voronov<sup>1</sup>, Mikhail Khoroshikh<sup>1</sup>, Artem Babenko, Max Ryabinin* \
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
*Yuxin Wen<sup>1</sup>, Neel Jain<sup>1</sup>, John Kirchenbauer, Micah Goldblum, Jonas Geiping, Tom Goldstein* \
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
*Elad Richardson<sup>1</sup>, Gal Metzer<sup>1</sup>, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01721)] [[Project](https://texturepaper.github.io/TEXTurePaper/)] [[Github](https://github.com/TEXTurePaper/TEXTurePaper)] \
3 Feb 2023

**Dreamix: Video Diffusion Models are General Video Editors** \
*Eyal Molad<sup>1</sup>, Eliahu Horwitz<sup>1</sup>, Dani Valevski<sup>1</sup>, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, Yedid Hoshen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01329)] [[Project](https://dreamix-video-editing.github.io/)] \
2 Feb 2023

**Trash to Treasure: Using text-to-image models to inform the design of physical artefacts** \
*Amy Smith<sup>1</sup>, Hope Schroeder<sup>1</sup>, Ziv Epstein, Michael Cook, Simon Colton, Andrew Lippman* \
AAAI 2023. [[Paper](https://arxiv.org/abs/2302.00561)] \
1 Feb 2023

**Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models** \
*Hila Chefer<sup>1</sup>, Yuval Alaluf<sup>1</sup>, Yael Vinker, Lior Wolf, Daniel Cohen-Or* \
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
*Uriel Singer<sup>1</sup>, Shelly Sheynin<sup>1</sup>, Adam Polyak<sup>1</sup>, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, Yaniv Taigman* \
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
*Huiwen Chang<sup>1</sup>, Han Zhang<sup>1</sup>, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, Dilip Krishnan* \
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
*Yaru Hao<sup>1</sup>, Zewen Chi<sup>1</sup>, Li Dong, Furu Wei* \
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
*Su Wang<sup>1</sup>, Chitwan Saharia<sup>1</sup>, Ceslee Montgomery<sup>1</sup>, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J. Fleet, Radu Soricut, Jason Baldridge, Mohammad Norouzi, Peter Anderson, William Chan* \
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
*Samuel Weinbach<sup>1</sup>, Marco Bellagente<sup>1</sup>, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Björn Deiseroth, Koen Oostermeijer, Hannah Teufel, Andres Felipe Cruz-Salinas* \
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
*Dong Huk Park<sup>1</sup>, Grace Luo<sup>1</sup>, Clayton Toste, Samaneh Azadi, Xihui Liu, Maka Karalashvili, Anna Rohrbach, Trevor Darrell* \
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
*Narek Tumanyan<sup>1</sup>, Michal Geyer<sup>1</sup>, Shai Bagon, Tali Dekel* \
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
*Ajay Jain<sup>1</sup>, Amber Xie<sup>1</sup>, Pieter Abbeel* \
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
*Chen-Hsuan Lin<sup>1</sup>, Jun Gao<sup>1</sup>, Luming Tang<sup>1</sup>, Towaki Takikawa<sup>1</sup>, Xiaohui Zeng<sup>1</sup>, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, Tsung-Yi Lin* \
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
*Hritik Bansal<sup>1</sup>, Da Yin<sup>1</sup>, Masoud Monajatipoor, Kai-Wei Chang* \
EMNLP 2022. [[Paper](https://arxiv.org/abs/2210.15230)] [[Github](https://github.com/Hritikbansal/entigen_emnlp)] \
27 Oct 2022

**ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts** \
*Zhida Feng<sup>1</sup>, Zhenyu Zhang<sup>1</sup>, Xintong Yu<sup>1</sup>, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang* \
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
*Bahjat Kawar<sup>1</sup>, Shiran Zada<sup>1</sup>, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, Michal Irani* \
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
*Jonathan Ho<sup>1</sup>, William Chan<sup>1</sup>, Chitwan Saharia<sup>1</sup>, Jay Whang<sup>1</sup>, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans* \
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
*Robin Rombach<sup>1</sup>, Andreas Blattmann<sup>1</sup>, Björn Ommer* \
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
*Nan Liu<sup>1</sup>, Shuang Li<sup>1</sup>, Yilun Du<sup>1</sup>, Antonio Torralba, Joshua B. Tenenbaum* \
ECCV 2022. [[Paper](https://arxiv.org/abs/2206.01714)] [[Project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)] [[Github](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch)] \
3 Jun 2022

**DiVAE: Photorealistic Images Synthesis with Denoising Diffusion Decoder** \
*Jie Shi<sup>1</sup>, Chenfei Wu<sup>1</sup>, Jian Liang, Xiang Liu, Nan Duan* \
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
*Chitwan Saharia<sup>1</sup>, William Chan<sup>1</sup>, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.11487)] [[Github](https://github.com/lucidrains/imagen-pytorch)]  \
23 May 2022


**Retrieval-Augmented Diffusion Models** \
*Andreas Blattmann<sup>1</sup>, Robin Rombach<sup>1</sup>, Kaan Oktay, Björn Ommer* \
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
*Robin Rombach<sup>1</sup>, Andreas Blattmann<sup>1</sup>, Dominik Lorenz, Patrick Esser, Björn Ommer* \
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
*Hao Zhang<sup>1</sup>, Yanbo Xu<sup>1</sup>, Tianyuan Dai<sup>1</sup>, Yu-Wing, Tai Chi-Keung Tang* \
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
*Guangyao Zhai<sup>1</sup>, Evin Pınar Örnek<sup>1</sup>, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, Benjamin Busam* \
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
*Byungjun Kim<sup>1</sup>, Patrick Kwon<sup>1</sup>, Kwangho Lee, Myunggi Lee, Sookwan Han, Daesik Kim, Hanbyul Joo* \ 
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
*Zehao Zhu<sup>1</sup>, Jiashun Wang<sup>1</sup>, Yuzhe Qin, Deqing Sun, Varun Jampani, Xiaolong Wang* \
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
*Tomas Jakab<sup>1</sup>, Ruining Li<sup>1</sup>, Shangzhe Wu, Christian Rupprecht, Andrea Vedaldi* \
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
*Federico Semeraro<sup>1</sup>, Yi Zhang<sup>1</sup>, Wenying Wu<sup>1</sup>, Patrick Carroll<sup>1</sup>* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04133)] [[Github](https://github.com/fsemerar/satnerf)] \
9 Apr 2023

**DITTO-NeRF: Diffusion-based Iterative Text To Omni-directional 3D Model** \
*Hoigi Seo<sup>1</sup>, Hayeon Kim<sup>1</sup>, Gwanghyun Kim, Se Young Chun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02827)] [[Project](https://janeyeon.github.io/ditto-nerf/)] \
6 Apr 2023


**Generative Novel View Synthesis with 3D-Aware Diffusion Models** \
*Eric R. Chan<sup>1</sup>, Koki Nagano<sup>1</sup>, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, Gordon Wetzstein* \
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
*Longwen Zhang<sup>1</sup>, Qiwei Qiu<sup>1</sup>, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, Jingyi Yu* \
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
*Susung Hong<sup>1</sup>, Donghoon Ahn<sup>1</sup>, Seungryong Kim* \
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
*Yiqi Lin<sup>1</sup>, Haotian Bai<sup>1</sup>, Sijia Li, Haonan Lu, Xiaodong Lin, Hui Xiong, Lin Wang* \
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
*Juil Koo<sup>1</sup>, Seungwoo Yoo<sup>1</sup>, Minh Hieu Nguyen<sup>1</sup>, Minhyuk Sung* \
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
*Lingting Zhu<sup>1</sup>, Xian Liu<sup>1</sup>, Xuanyu Liu, Rui Qian, Ziwei Liu, Lequan Yu* \
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
*Suhyeon Lee<sup>1</sup>, Hyungjin Chung<sup>1</sup>, Minyoung Park, Jonghyuk Park, Wi-Sun Ryu, Jong Chul Ye* \
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
*Junyoung Seo<sup>1</sup>, Wooseok Jang<sup>1</sup>, Min-Seop Kwak<sup>1</sup>, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, Seungryong Kim* \
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
*Yonatan Shafir<sup>1</sup>, Guy Tevet<sup>1</sup>, Roy Kapon, Amit H. Bermano* \
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
*Sigal Raab<sup>1</sup>, Inbal Leibovitch<sup>1</sup>, Guy Tevet, Moab Arar, Amit H. Bermano, Daniel Cohen-Or* \
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
*Elad Richardson<sup>1</sup>, Gal Metzer<sup>1</sup>, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01721)] [[Project](https://texturepaper.github.io/TEXTurePaper/)] [[Github](https://github.com/TEXTurePaper/TEXTurePaper)] \
3 Feb 2023



**Zero3D: Semantic-Driven Multi-Category 3D Shape Generation** \
*Bo Han, Yitong Liu, Yixuan Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13591)] \
31 Jan 2023

**Neural Wavelet-domain Diffusion for 3D Shape Generation, Inversion, and Manipulation** \
*Jingyu Hu<sup>1</sup>, Ka-Hei Hui<sup>1</sup>, Zhengzhe Liu, Ruihui Li, Chi-Wing Fu* \
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
*Siyuan Huang<sup>1</sup>, Zan Wang<sup>1</sup>, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, Song-Chun Zhu* \
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
*Alex Nichol<sup>1</sup>, Heewoo Jun<sup>1</sup>, Prafulla Dhariwal, Pamela Mishkin, Mark Chen* \
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
*Tengfei Wang<sup>1</sup>, Bo Zhang<sup>1</sup>, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, Baining Guo* \
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
*Haochen Wang<sup>1</sup>, Xiaodan Du<sup>1</sup>, Jiahao Li<sup>1</sup>, Raymond A. Yeh, Greg Shakhnarovich* \
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
*Jia Gong<sup>1</sup>, Lin Geng Foo<sup>1</sup>, Zhipeng Fan, Qiuhong Ke, Hossein Rahmani, Jun Liu* \
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
*Gal Metzer<sup>1</sup>, Elad Richardson<sup>1</sup>, Or Patashnik, Raja Giryes, Daniel Cohen-Or* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.07600)] [[Github](https://github.com/eladrich/latent-nerf)] \
14 Nov 2022

**ReFu: Refine and Fuse the Unobserved View for Detail-Preserving Single-Image 3D Human Reconstruction** \
*Gyumin Shim<sup>1</sup>, Minsoo Lee<sup>1</sup>, Jaegul Choo* \
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

**An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization** \ 
*Fei Kong, Jinhao Duan, RuiPeng Ma, Hengtao Shen, Xiaofeng Zhu, Xiaoshuang Shi, Kaidi Xu* \ 
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18355)] \ 
26 May 2023


**Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability** \
*Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16494)] [[Github](https://github.com/xavihart/Diff-PGD)] \
25 May 2023


**Differentially Private Latent Diffusion Models** \
*Saiyue Lyu<sup>1</sup>, Margarita Vinaroz<sup>1</sup>, Michael F. Liu<sup>1</sup>, Mijung Park* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15759)] \
25 May 2023

**Latent Magic: An Investigation into Adversarial Examples Crafted in the Semantic Latent Space** \
*BoYang Zheng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12906)] \
22 May 2023


**Mist: Towards Improved Adversarial Examples for Diffusion Models** \
*Chumeng Liang, Xiaoyu Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12683)] \
22 May 2023

**Content-based Unrestricted Adversarial Attack** \
*Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10665)] \
18 May 2023

**Zero-Day Backdoor Attack against Text-to-Image Diffusion Models via Personalization** \
*Yihao Huang, Qing Guo, Felix Juefei-Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10701)] \
18 May 2023


**Raising the Bar for Certified Adversarial Robustness with Diffusion Models** \
*Thomas Altstidl, David Dobre, Björn Eskofier, Gauthier Gidel, Leo Schwinn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10388)] \
17 May 2023

**Diffusion Models for Imperceptible and Transferable Adversarial Attack** \
*Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08192)] [[Github](https://github.com/WindVChen/DiffAttack)] \
14 May 2023

**On enhancing the robustness of Vision Transformers: Defensive Diffusion** \
*Raza Imam<sup>1</sup>, Muhammad Huzaifa<sup>1</sup>, Mohammed El-Amine Azz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08031)] [[Github](https://github.com/Muhammad-Huzaifaa/Defensive_Diffusion)] \
14 May 2023


**Generative Steganography Diffusion** \
*Ping Wei, Qing Zhou, Zichi Wang, Zhenxing Qian, Xinpeng Zhang, Sheng Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03472)] \
5 May 2023


**A Pilot Study of Query-Free Adversarial Attack against Stable Diffusion** \
*Haomin Zhuang, Yihua Zhang, Sijia Liu* \
CVPR Workshop 2023. [[Paper](https://arxiv.org/abs/2303.16378)] \
3 Apr 2023

**Black-box Backdoor Defense via Zero-shot Image Purification** \
*Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Ninghao Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12175)] \
21 Mar 2023

**Adversarial Counterfactual Visual Explanations** \
*Guillaume Jeanneret, Loïc Simon, Frédéric Jurie* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.09962)] [[Github](https://github.com/guillaumejs2403/ACE)] \
17 Mar 2023

**Robust Evaluation of Diffusion-Based Adversarial Purification** \
*Minjong Lee, Dongwoo Kim* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2303.09051)] \
16 Mar 2023

**The Devil's Advocate: Shattering the Illusion of Unexploitable Data using Diffusion Models** \
*Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08500)] \
15 Mar 2023

**TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets** \
*Weixin Chen, Dawn Song, Bo Li* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.05762)] [[Github](https://github.com/chenweixin107/TrojDiff)] \
10 Mar 2023

**Generative Model-Based Attack on Learnable Image Encryption for Privacy-Preserving Deep Learning** \
*AprilPyone MaungMaung, Hitoshi Kiya* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05036)] \
9 Mar 2023

**Differentially Private Diffusion Models Generate Useful Synthetic Images** \
*Sahra Ghalebikesabi, Leonard Berrada, Sven Gowal, Ira Ktena, Robert Stanforth, Jamie Hayes, Soham De, Samuel L. Smith, Olivia Wiles, Borja Balle* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13861)] \
27 Feb 2023

**Data Forensics in Diffusion Models: A Systematic Analysis of Membership Privacy** \
*Derui Zhu<sup>1</sup>, Dingfan Chen<sup>1</sup>, Jens Grossklags, Mario Fritz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07801)] \
15 Feb 2023

**Raising the Cost of Malicious AI-Powered Image Editing** \
*Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, Aleksander Madry* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.06588)] [[Github](https://github.com/MadryLab/photoguard)] \
13 Feb 2023

**Adversarial Example Does Good: Preventing Painting Imitation from Diffusion Models via Adversarial Examples** \
*Chumeng Liang<sup>1</sup>, Xiaoyu Wu<sup>1</sup>, Yang Hua, Jiaru Zhang, Yiming Xue, Tao Song, Zhengui Xue, Ruhui Ma, Haibing Guan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04578)] \
9 Feb 2023

**Better Diffusion Models Further Improve Adversarial Training** \
*Zekai Wang<sup>1</sup>, Tianyu Pang<sup>1</sup>, Chao Du, Min Lin, Weiwei Liu, Shuicheng Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04638)] [[Github](https://github.com/wzekai99/DM-Improves-AT)] \
9 Feb 2023


**Membership Inference Attacks against Diffusion Models** \
*Tomoya Matsumoto, Takayuki Miura, Naoto Yanai* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03262)] \
7 Feb 2023

**MorDIFF: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Diffusion Autoencoders** \
*Naser Damer, Meiling Fang, Patrick Siebke, Jan Niklas Kolf, Marco Huber, Fadi Boutros* \
IWBF 2023. [[Paper](https://arxiv.org/abs/2302.01843)] [[Github](https://github.com/naserdamer/mordiff)] \
3 Feb 2023

**Extracting Training Data from Diffusion Models** \
*Nicholas Carlini<sup>1</sup>, Jamie Hayes<sup>1</sup>, Milad Nasr<sup>1</sup>, Matthew Jagielski, Vikash Sehwag, Florian Tramèr, Borja Balle, Daphne Ippolito, Eric Wallace* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.00860)] \
2 Feb 2023

**Are Diffusion Models Vulnerable to Membership Inference Attacks?** \
*Jinhao Duan, Fei Kong, Shiqi Wang, Xiaoshuang Shi, Kaidi Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01316)] \
2 Feb 2023

**Salient Conditional Diffusion for Defending Against Backdoor Attacks** \
*Brandon B. May<sup>1</sup>, N. Joseph Tatro<sup>1</sup>, Piyush Kumar, Nathan Shnidman* \
ICLR Workshop 2023. [[Paper](https://arxiv.org/abs/2301.13721)] \
31 Jan 2023


**Extracting Training Data from Diffusion Models** \
*Nicholas Carlini<sup>1</sup>, Jamie Hayes<sup>1</sup>, Milad Nasr<sup>1</sup>, Matthew Jagielski, Vikash Sehwag, Florian Tramèr, Borja Balle, Daphne Ippolito, Eric Wallace* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13188)] \
30 Jan 2023

**Membership Inference of Diffusion Models** \
*Hailong Hu, Jun Pang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.09956)] \
24 Jan 2023

**Denoising Diffusion Probabilistic Models as a Defense against Adversarial Attacks** \
*Lars Lien Ankile, Anna Midgley, Sebastian Weisshaar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.06871)] [[Github](https://github.com/ankile/Adversarial-Diffusion)] \
17 Jan 2023

**Fight Fire With Fire: Reversing Skin Adversarial Examples by Multiscale Diffusive and Denoising Aggregation Mechanism** \
*Yongwei Wang<sup>1</sup>, Yuan Li<sup>1</sup>, Zhiqi Shen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.10373)] \
22 Aug 2022

**DensePure: Understanding Diffusion Models towards Adversarial Robustness** \
*Chaowei Xiao<sup>1</sup>, Zhongzhu Chen<sup>1</sup>, Kun Jin<sup>1</sup>, Jiongxiao Wang<sup>1</sup>, Weili Nie, Mingyan Liu, Anima Anandkumar, Bo Li, Dawn Song* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2211.00322)] \
1 Nov 2022

**Improving Adversarial Robustness by Contrastive Guided Diffusion Process** \
*Yidong Ouyang, Liyan Xie, Guang Cheng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.09643)] \
18 Oct 2022

**Differentially Private Diffusion Models** \
*Tim Dockhorn, Tianshi Cao, Arash Vahdat, Karsten Kreis* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.09929)] [[Project](https://nv-tlabs.github.io/DPDM/)] \
18 Oct 2022

**PointDP: Diffusion-driven Purification against Adversarial Attacks on 3D Point Cloud Recognition** \
*Jiachen Sun, Weili Nie, Zhiding Yu, Z. Morley Mao, Chaowei Xiao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.09801)] \
21 Aug 2022


**Threat Model-Agnostic Adversarial Defense using Diffusion Models** \
*Tsachi Blau, Roy Ganz, Bahjat Kawar, Alex Bronstein, Michael Elad* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.08089)] [[Github](https://github.com/tsachiblau/Threat-Model-Agnostic-Adversarial-Defense-using-Diffusion-Models)] \
17 Jul 2022

**Back to the Source: Diffusion-Driven Test-Time Adaptation** \
*Jin Gao<sup>1</sup>, Jialing Zhang<sup>1</sup>, Xihui Liu, Trevor Darrell, Evan Shelhamer, Dequan Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.03442)] [[Github](https://github.com/shiyegao/DDA)] \
7 Jul 2022

**Guided Diffusion Model for Adversarial Purification from Random Noise** \
*Quanlin Wu, Hang Ye, Yuntian Gu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.10875)] \
17 Jun 2022

**(Certified!!) Adversarial Robustness for Free!** \
*Nicholas Carlini, Florian Tramer, Krishnamurthy (Dj)Dvijotham, J. Zico Kolter* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2206.10550)] \
21 Jun 2022

**Guided Diffusion Model for Adversarial Purification** \
*Jinyi Wang<sup>1</sup>, Zhaoyang Lyu<sup>1</sup>, Dahua Lin, Bo Dai, Hongfei Fu* \
ICML 2022. [[Paper](https://arxiv.org/abs/2205.14969)] [[Github](https://github.com/jinyiw/guideddiffusionpur)] \
30 May 2022

**Diffusion Models for Adversarial Purification** \
*Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar* \
ICML 2022. [[Paper](https://arxiv.org/abs/2205.07460)] [[Project](https://diffpure.github.io/)] [[Github](https://github.com/NVlabs/DiffPure)] \
16 May 2022

**TFDPM: Attack detection for cyber-physical systems with diffusion probabilistic models** \
*Tijin Yan, Tong Zhou, Yufeng Zhan, Yuanqing Xia* \
Elsveier Knowledge-Based Systems 2021. [[Paper](https://arxiv.org/abs/2112.10774)] \
20 Dec 2021

**Adversarial purification with Score-based generative models** \
*Jongmin Yoon, Sung Ju Hwang, Juho Lee* \
ICML 2021. [[Paper](https://arxiv.org/abs/2106.06041)] [[Github](https://github.com/jmyoon1/adp)] \
11 Jun 2021




### Miscellany

**Phoenix: A Federated Generative Diffusion Model** \
*Fiona Victoria Stanley Jothiraj, Afra Mashhadi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04098)] \
7 Jun 2023

**Change Diffusion: Change Detection Map Generation Based on Difference-Feature Guided DDPM** \
*Yihan Wen, Jialu Sui, Xianping Ma, Wendi Liang, Xiaokang Zhang, Man-On Pun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03424)] \
6 Jun 2023

**Towards Visual Foundational Models of Physical Scenes** \
*Chethan Parameshwara<sup>1</sup>, Alessandro Achille<sup>1</sup>, Matthew Trager, Xiaolong Li, Jiawei Mo, Matthew Trager, Ashwin Swaminathan, CJ Taylor, Dheera Venkatraman, Xiaohan Fei, Stefano Soatto* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03727)] \
6 Jun 2023



**Protecting the Intellectual Property of Diffusion Models by the Watermark Diffusion Process** \
*Sen Peng, Yufei Chen, Cong Wang, Xiaohua Jia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03436)] \
6 Jun 2023


**Emergent Correspondence from Image Diffusion** \
*Luming Tang<sup>1</sup>, Menglin Jia<sup>1</sup>, Qianqian Wang<sup>1</sup>, Cheng Perng Phoo, Bharath Hariharan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03881)] [[Project](https://diffusionfeatures.github.io/)] \
6 Jun 2023

**Enhance Diffusion to Improve Robust Generalization** \
*Jianhui Sun, Sanchit Sinha, Aidong Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02618)] \
5 Jun 2023

**Training Data Attribution for Diffusion Models** \
*Zheng Dai, David K Gifford* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02174)] \
3 Jun 2023

**Deep Classifier Mimicry without Data Access** \
*Steven Braun, Martin Mundt, Kristian Kersting* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02090)] \
3 Jun 2023

**Quantifying Sample Anonymity in Score-Based Generative Models with Adversarial Fingerprinting** \
*Mischa Dombrowski, Bernhard Kainz* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01363)] \
2 Jun 2023

**Generative Autoencoders as Watermark Attackers: Analyses of Vulnerabilities and Threats** \
*Xuandong Zhao, Kexun Zhang, Yu-Xiang Wang, Lei Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01953)] \
2 Jun 2023

**PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models** \
*Jiacheng Chen, Ruizhi Deng, Yasutaka Furukawa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01461)] \
2 Jun 2023

**Unlearnable Examples for Diffusion Models: Protect Data from Unauthorized Exploitation** \
*Zhengyue Zhao, Jinhao Duan, Xing Hu, Kaidi Xu, Chenan Wang, Rui Zhang, Zidong Du, Qi Guo, Yunji Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01902)] \
2 Jun 2023

**Robust Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers** \
*Ruotong Wang<sup>1</sup>, Hongrui Chen<sup>1</sup>, Zihao Zhu, Li Liu, Yong Zhang, Yanbo Fan, Baoyuan Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00816)] \
1 Jun 2023


**Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust** \
*Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.20030)] [[Github](https://github.com/YuxinWenRick/tree-ring-watermark)] \
31 May 2023

**GANDiffFace: Controllable Generation of Synthetic Datasets for Face Recognition with Realistic Variations** \
*Pietro Melzi, Christian Rathgeb, Ruben Tolosana, Ruben Vera-Rodriguez, Dominik Lawatsch, Florian Domin, Maxim Schaubert* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19962)] \
31 May 2023

**Improving Handwritten OCR with Training Samples Generated by Glyph Conditional Denoising Diffusion Probabilistic Model** \
*Haisong Ding, Bozhi Luan, Dongnan Gui, Kai Chen, Qiang Huo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19543)] \
31 May 2023

**Label-Retrieval-Augmented Diffusion Models for Learning from Noisy Labels** \
*Jian Chen, Ruiyi Zhang, Tong Yu, Rohan Sharma, Zhiqiang Xu, Tong Sun, Changyou Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19518)] [[Github](https://github.com/puar-playground/LRA-diffusion)] \
31 May 2023

**DiffMatch: Diffusion Model for Dense Matching** \
*Jisu Nam, Gyuseong Lee, Sunwoo Kim, Hyeonsu Kim, Hyoungwon Cho, Seyeon Kim, Seungryong Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19094)] [[Project](https://ku-cvlab.github.io/DiffMatch/)] \
30 May 2023


**Calliffusion: Chinese Calligraphy Generation and Style Transfer with Diffusion Modeling** \
*Qisheng Liao, Gus Xia, Zhinuo Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19124)] \
30 May 2023

**Diffusion-Stego: Training-free Diffusion Generative Steganography via Message Projection** \
*Daegyu Kim, Chaehun Shin, Jooyoung Choi, Dahuin Jung, Sungroh Yoon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18726)] \
30 May 2023

**On Diffusion Modeling for Anomaly Detection** \
*Victor Livernoche, Vineet Jain, Yashar Hezaveh, Siamak Ravanbakhsh* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18593)] \
29 May 2023

**Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation** \
*Giorgio Giannone, Akash Srivastava, Ole Winther, Faez Ahmed* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18470)] \
29 May 2023

**Generating Driving Scenes with Diffusion** \
*Ethan Pronovost, Kai Wang, Nick Roy* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18452)] \
29 May 2023

**Generative Diffusion for 3D Turbulent Flows** \
*Marten Lienen, Jan Hansen-Palmus, David Lüdke, Stephan Günnemann* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01776)] \
29 May 2023

**GlyphControl: Glyph Conditional Control for Visual Text Generation** \
*Yukang Yang, Dongnan Gui, Yuhui Yuan, Haisong Ding, Han Hu, Kai Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18259)] [[Github](https://github.com/AIGText/GlyphControl-release)] \
29 May 2023

**High-Fidelity Image Compression with Score-based Generative Models** \
*Emiel Hoogeboom, Eirikur Agustsson, Fabian Mentzer, Luca Versari, George Toderici, Lucas Theis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18231)] \
26 May 2023

**CamoDiffusion: Camouflaged Object Detection via Conditional Diffusion Models** \
*Zhongxi Chen, Ke Sun, Xianming Lin, Rongrong Ji* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17932)] \
29 May 2023

**DiffusionNAG: Task-guided Neural Architecture Generation with Diffusion Models** \
*Sohyun An, Hayeon Lee, Jaehyeong Jo, Seanie Lee, Sung Ju Hwang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16943)] \
26 May 2023

**CRoSS: Diffusion Model Makes Controllable, Robust and Secure Image Steganography** \
*Jiwen Yu, Xuanyu Zhang, Youmin Xu, Jian Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16936)] [[Github](https://github.com/vvictoryuki/CRoSS)] \
26 May 2023

**Realistic Noise Synthesis with Diffusion Models** \
*Qi Wu, Mingyan Han, Ting Jiang, Haoqiang Fan, Bing Zeng, Shuaicheng Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14022)] \
23 May 2023

**Anomaly Detection with Conditioned Denoising Diffusion Models** \
*Arian Mousakhan, Thomas Brox, Jawad Tayyub* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15956)] \
25 May 2023


**Knowledge Diffusion for Distillation** \
*Tao Huang, Yuan Zhang, Mingkai Zheng, Shan You, Fei Wang, Chen Qian, Chang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15712)] [[Github](https://github.com/hunto/DiffKD)] \
25 May 2023


**Zero-shot Generation of Training Data with Denoising Diffusion Probabilistic Model for Handwritten Chinese Character Recognition** \
*Dongnan Gui, Kai Chen, Haisong Ding, Qiang Huo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15660)] \
25 May 2023

**Unsupervised Semantic Correspondence Using Stable Diffusion** \
*Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15581)] \
24 May 2023

**A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence** \
*Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, Ming-Hsuan Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15347)] [[Project](https://sd-complements-dino.github.io/)] \
24 May 2023

**Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence** \
*Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, Trevor Darrell* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14334)] [[Project](https://diffusion-hyperfeatures.github.io/)] \
23 May 2023

**DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection** \
*Jiang Liu, Chun Pong Lau, Rama Chellappa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13625)] \
23 May 2023

**GSURE-Based Diffusion Model Training with Corrupted Data** \
*Bahjat Kawar, Noam Elata, Tomer Michaeli, Michael Elad* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13128)] [[Github](https://github.com/bahjat-kawar/gsure-diffusion/)] \
22 May 2023

**Watermarking Diffusion Model** \
*Yugeng Liu, Zheng Li, Michael Backes, Yun Shen, Yang Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12502)] \
21 May 2023

**DiffUCD:Unsupervised Hyperspectral Image Change Detection with Semantic Correlation Diffusion Model** \
*Xiangrong Zhang, Shunli Tian, Guanchun Wang, Huiyu Zhou, Licheng Jiao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12410)] \
21 May 2023

**Incomplete Multi-view Clustering via Diffusion Completion** \
*Sifan Fang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11489)] \
19 May 2023


**SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models** \
*Ziyi Wu, Jingyu Hu, Wuyue Lu, Igor Gilitschenski, Animesh Garg* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11281)] \
18 May 2023


**Selective Guidance: Are All the Denoising Steps of Guided Diffusion Important?** \
*Pareesa Ameneh Golnari, Zhewei Yao, Yuxiong He* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09847)] \
16 May 2023

**A Method for Training-free Person Image Picture Generation** \
*Tianyu Chen* \
ICOAI 2023. [[Paper](https://arxiv.org/abs/2305.09817)] \
16 May 2023

**Constructing a personalized AI assistant for shear wall layout using Stable Diffusion** \
*Lufeng Wang, Jiepeng Liu, Guozhong Cheng, En Liu, Wei Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10830)] \
18 May 2023

**DiffUTE: Universal Text Editing Diffusion Model** \
*Chen, Haoxing, Xu, Zhuoer, Gu, Zhangxuan, Lan, Jun, Zheng, Xing, Li, Yaohui, Meng, Changhua, Zhu, Huijia, Wang, Weiqiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10825)] [[Github](https://github.com/chenhaoxing/DiffUTE)] \
18 May 2023

**CDDM: Channel Denoising Diffusion Models for Wireless Communications** \
*Tong Wu, Zhiyong Chen, Dazhi He, Liang Qian, Yin Xu, Meixia Tao, Wenjun Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09161)] \
16 May 2023

**Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples** \
*Wan Jiang<sup>1</sup>, Yunfeng Diao<sup>1</sup>, He Wang, Jianxin Sun, Meng Wang, Richang Hong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09241)] \
16 May 2023

**Diffusion Dataset Generation: Towards Closing the Sim2Real Gap for Pedestrian Detection** \
*Andrew Farley, Mohsen Zand, Michael Greenspan* \
CRV 2023. [[Paper](https://arxiv.org/abs/2305.09401)] \
16 May 2023

**A Reproducible Extraction of Training Images from Diffusion Models** \
*Ryan Webster* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08694)] [[Github](https://github.com/ryanwebster90/onestep-extraction)] \
15 May 2023

**Laughing Matters: Introducing Laughing-Face Generation using Diffusion Models** \
*Antoni Bigata Casademunt, Rodrigo Mira, Nikita Drobyshev, Konstantinos Vougioukas, Stavros Petridis, Maja Pantic* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08854)] \
15 May 2023

**Undercover Deepfakes: Detecting Fake Segments in Videos** \
*Sanjay Saha, Rashindrie Perera, Sachith Seneviratne, Tamasha Malepathirana, Sanka Rasnayaka, Deshani Geethika, Terence Sim, Saman Halgamuge* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06564)] [[Github](https://github.com/sanjaysaha1311/temporal-deepfake-segmentation)] \
11 May 2023

**Comprehensive Dataset of Synthetic and Manipulated Overhead Imagery for Development and Evaluation of Forensic Tools** \
*Brandon B. May, Kirill Trapeznikov, Shengbang Fang, Matthew C. Stamm* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05784)] \
9 May 2023



**DifFIQA: Face Image Quality Assessment Using Denoising Diffusion Probabilistic Models** \
*Žiga Babnik, Peter Peer, Vitomir Štruc* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05768)] \
9 May 2023

**Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning** \
*Shengfang Zhai, Yinpeng Dong, Qingni Shen, Shi Pu, Yuejian Fang, Hang Su* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04175)] \
7 May 2023

**Exploring One-shot Semi-supervised Federated Learning with A Pre-trained Diffusion Model** \
*Mingzhao Yang, Shangchao Su, Bin Li, Xiangyang Xue* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04063)] \
6 May 2023

**Towards Prompt-robust Face Privacy Protection via Adversarial Decoupling Augmentation Framework** \
*Ruijia Wu, Yuhang Wang, Huafeng Shi, Zhipeng Yu, Yichao Wu, Ding Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03980)]
6 May 2023


**Conditional Diffusion Feature Refinement for Continuous Sign Language Recognition** \
*Leming Guo, Wanli Xue, Qing Guo, Yuxi Zhou, Tiantian Yuan, Shengyong Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03614)] \
5 May 2023

**LayoutDM: Transformer-based Diffusion Model for Layout Generation** \
*Shang Chai, Liansheng Zhuang, Fengying Yan* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2305.02567)] \
4 May 2023

**Long-Term Rhythmic Video Soundtracker** \
*Jiashuo Yu, Yaohui Wang, Xinyuan Chen, Xiao Sun, Yu Qiao* \
ICML 2023. [[Paper](https://arxiv.org/abs/2305.01319)] [[Github](https://github.com/OpenGVLab/LORIS)] \
2 May 2023

**Putting People in Their Place: Affordance-Aware Human Insertion into Scenes** \
*Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A. Efros, Krishna Kumar Singh* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2304.14406)] [[Project](https://sumith1896.github.io/affordance-insertion/)] [[Github](https://github.com/adobe-research/affordance-insertion)] \
27 Apr 2023

**Single-View Height Estimation with Conditional Diffusion Probabilistic Models** \
*Isaac Corley, Peyman Najafirad* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13214)] \
26 Apr 2023

**Diffusion Probabilistic Model Based Accurate and High-Degree-of-Freedom Metasurface Inverse Design** \
*Zezhou Zhang, Chuanchuan Yang, Yifeng Qin, Hao Feng, Jiqiang Feng, Hongbin Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13038)] \
25 Apr 2023

**Improving Synthetically Generated Image Detection in Cross-Concept Settings** \
*Pantelis Dogoulis, Giorgos Kordopatis-Zilos, Ioannis Kompatsiaris, Symeon Papadopoulos* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12053)] \
24 Apr 2023

**Fast Diffusion Probabilistic Model Sampling through the lens of Backward Error Analysis** \
*Yansong Gao, Zhihong Pan, Xin Zhou, Le Kang, Pratik Chaudhari* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11446)] \
22 Apr 2023

**Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations** \
*Yu-Hui Chen, Raman Sarokin, Juhyun Lee, Jiuqiang Tang, Chuo-Ling Chang, Andrei Kulik, Matthias Grundmann* \
CVPR 2023 Workshop. [[Paper](https://arxiv.org/abs/2304.11267)] \
21 Apr 2023

**A data augmentation perspective on diffusion models and retrieval** \
*Max F. Burg, Florian Wenzel, Dominik Zietlow, Max Horn, Osama Makansi, Francesco Locatello, Chris Russell* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.10253)] \
20 Apr 2023

**Diffusion models with location-scale noise** \
*Alexia Jolicoeur-Martineau, Kilian Fatras, Ke Li, Tal Kachman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05907)] \
12 Apr 2023

**Exploring Diffusion Models for Unsupervised Video Anomaly Detection** \
*Anil Osman Tur, Nicola Dall'Asen, Cigdem Beyan, Elisa Ricci* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05841)] \
12 Apr 2023

**CamDiff: Camouflage Image Augmentation via Diffusion Model** \
*Xue-Jing Luo, Shuo Wang, Zongwei Wu, Christos Sakaridis, Yun Cheng, Deng-Ping Fan, Luc Van Gool* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05469)] [[Github](https://github.com/drlxj/CamDiff)] \
11 Apr 2023

**DDRF: Denoising Diffusion Model for Remote Sensing Image Fusion** \
*ZiHan Cao, ShiQi Cao, Xiao Wu, JunMing Hou, Ran Ran, Liang-Jian Deng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04774)] \
10 Apr 2023

**CCLAP: Controllable Chinese Landscape Painting Generation via Latent Diffusion Model** \
*Zhongqi Wang, Jie Zhang, Zhilong Ji, Jinfeng Bai, Shiguang Shan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04156)] \
9 Apr 2023

**ChiroDiff: Modelling chirographic data with Diffusion Models** \
*Ayan Das, Yongxin Yang, Timothy Hospedales, Tao Xiang, Yi-Zhe Song* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2304.03785)] [[Project](https://ayandas.me/chirodiff)] \
7 Apr 2023

**RoSteALS: Robust Steganography using Autoencoder Latent Space** \
*Tu Bui, Shruti Agarwal, Ning Yu, John Collomosse* \
CVPR Workshop 2023. [[Paper](https://arxiv.org/abs/2304.03400)] [[Github](https://github.com/TuBui/RoSteALS)] \
6 Apr 2023

**JPEG Compressed Images Can Bypass Protections Against AI Editing** \
*Pedro Sandoval-Segura, Jonas Geiping, Tom Goldstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02234)] \
5 Apr 2023

**Learning to Read Braille: Bridging the Tactile Reality Gap with Diffusion Models** \
*Carolina Higuera, Byron Boots, Mustafa Mukadam* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01182)] \
3 Apr 2023

**Textile Pattern Generation Using Diffusion Models** \
*Halil Faruk Karagoz, Gulcin Baykal, Irem Arikan Eksi, Gozde Unal* \
ITFC 2023. [[Paper](https://arxiv.org/abs/2304.00520)] \
2 Apr 2023

**Parents and Children: Distinguishing Multimodal DeepFakes from Natural Images** \
*Roberto Amoroso<sup>1</sup>, Davide Morelli<sup>1</sup>, Marcella Cornia, Lorenzo Baraldi, Alberto Del Bimbo, Rita Cucchiara* \
ACM 2023. [[Paper](https://arxiv.org/abs/2304.00500)] \
2 Apr 2023

**NeuroDAVIS: A neural network model for data visualization** \
*Chayan Maitra, Dibyendu B. Seal, Rajat K. De* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01222)] \
1 Apr 2023

**Diffusion Action Segmentation** \
*Daochang Liu, Qiyue Li, AnhDung Dinh, Tingting Jiang, Mubarak Shah, Chang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17959)] \
31 Mar 2023

**One-shot Unsupervised Domain Adaptation with Personalized Diffusion Models** \
*Yasser Benigmim, Subhankar Roy, Slim Essid, Vicky Kalogeiton, Stéphane Lathuilière* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.18080)] \
31 Mar 2023

**DDP: Diffusion Model for Dense Visual Prediction** \
*Yuanfeng Ji, Zhe Chen, Enze Xie, Lanqing Hong, Xihui Liu, Zhaoqiang Liu, Tong Lu, Zhenguo Li, Ping Luo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17559)] \
30 Mar 2023

**WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models** \
*Konstantina Nikolaidou, George Retsinas, Vincent Christlein, Mathias Seuret, Giorgos Sfikas, Elisa Barney Smith, Hamam Mokayed, Marcus Liwicki* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16576)] \
29 Mar 2023

**Visual Chain-of-Thought Diffusion Models** \
*William Harvey, Frank Wood* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16187)] \
28 Mar 2023


**DiffTAD: Temporal Action Detection with Proposal Denoising Diffusion** \
*Sauradip Nag, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.14863)] \
27 Mar 2023

**The Stable Signature: Rooting Watermarks in Latent Diffusion Models** \
*Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, Teddy Furon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15435)] [[Project](https://pierrefdz.github.io/publications/stablesignature/)] \
27 Mar 2023

**Exploring Continual Learning of Diffusion Models** \
*Michał Zając, Kamil Deja, Anna Kuzina, Jakub M. Tomczak, Tomasz Trzciński, Florian Shkurti, Piotr Miłoś* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15342)] \
27 Mar 2023


**Freestyle Layout-to-Image Synthesis** \
*Han Xue, Zhiwu Huang, Qianru Sun, Li Song, Wenjun Zhang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.14412)] \
25 Mar 2023

**Controllable Inversion of Black-Box Face-Recognition Models via Diffusion** \
*Manuel Kansy, Anton Raël, Graziana Mignone, Jacek Naruniec, Christopher Schroers, Markus Gross, Romann M. Weber* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13006)] \
23 Mar 2023

**End-to-End Diffusion Latent Optimization Improves Classifier Guidance** \
*Bram Wallace, Akash Gokul, Stefano Ermon, Nikhil Naik* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13703)] \
23 Mar 2023


**DiffPattern: Layout Pattern Generation via Discrete Diffusion** \
*Zixiao Wang, Yunheng Shen, Wenqian Zhao, Yang Bai, Guojin Chen, Farzan Farnia, Bei Yu* \
DAC 2023. [[Paper](https://arxiv.org/abs/2303.13060)] \
23 Mar 2023

**Diffuse-Denoise-Count: Accurate Crowd-Counting with Diffusion Models** \
*Yasiru Ranasinghe, Nithin Gopalakrishnan Nair, Wele Gedara Chaminda Bandara, Vishal M. Patel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12790)] \
22 Mar 2023

**LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models** \
*Junyi Zhang<sup>1</sup>, Jiaqi Guo<sup>1</sup>, Shizhao Sun, Jian-Guang Lou, Dongmei Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11589)] \
21 Mar 2023

**Positional Diffusion: Ordering Unordered Sets with Diffusion Probabilistic Models** \
*Francesco Giuliari, Gianluca Scarpellini, Stuart James, Yiming Wang, Alessio Del Bue* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11120)] [[Project](https://iit-pavis.github.io/Positional_Diffusion/)] \
20 Mar 2023

**Leapfrog Diffusion Model for Stochastic Trajectory Prediction** \
*Weibo Mao, Chenxin Xu, Qi Zhu, Siheng Chen, Yanfeng Wang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.10895)] [[Github](https://github.com/MediaBrain-SJTU/LED)] \
20 Mar 2023

**Pluralistic Aging Diffusion Autoencoder** \
*Peipei Li, Rui Wang, Huaibo Huang, Ran He, Zhaofeng He* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11086)] \
20 Mar 2023

**AnimeDiffusion: Anime Face Line Drawing Colorization via Diffusion Models** \
*Yu Cao, Xiangqiao Meng, P.Y. Mok, Xueting Liu, Tong-Yee Lee, Ping Li* \
arxiv 2023. [[Paper](https://arxiv.org/abs/2303.11137)] \
20 Mar 2023

**Diffusion-based Document Layout Generation** \
*Liu He, Yijuan Lu, John Corring, Dinei Florencio, Cha Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10787)] \
19 Mar 2023

**On the De-duplication of LAION-2B** \
*Ryan Webster, Julien Rabin, Loic Simon, Frederic Jurie* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12733)] \
17 Mar 2023

**A Recipe for Watermarking Diffusion Models** \
*Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, Min Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10137)] [[Github](https://github.com/yunqing-me/WatermarkDM)] \
17 Mar 2023

**DIRE for Diffusion-Generated Image Detection** \
*Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, Houqiang Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09295)] [[Github](https://github.com/ZhendongWang6/DIRE)] \
16 Mar 2023

**DS-Fusion: Artistic Typography via Discriminated and Stylized Diffusion** \
*Maham Tanveer, Yizhi Wang, Ali Mahdavi-Amiri, Hao Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09604)] [[Project](https://ds-fusion.github.io/)] \
16 Mar 2023

**DiffusionAD: Denoising Diffusion for Anomaly Detection** \
*Hui Zhang, Zheng Wang, Zuxuan Wu, Yu-Gang Jiang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08730)] \
15 Mar 2023

**LayoutDM: Discrete Diffusion Model for Controllable Layout Generation** \
*Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, Kota Yamaguchi* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.08137)] [[Project](https://cyberagentailab.github.io/layout-dm/)] [[Github](https://github.com/CyberAgentAILab/layout-dm)] \
14 Mar 2023

**Parallel Vertex Diffusion for Unified Visual Grounding** \
*Zesen Cheng, Kehan Li, Peng Jin, Xiangyang Ji, Li Yuan, Chang Liu, Jie Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07216)] \
13 Mar 2023

**DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion** \
*Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, Luc Van Gool* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06840)] \
13 Mar 2023

**Detecting Images Generated by Diffusers** \
*Davide Alessandro Coccomini, Andrea Esuli, Fabrizio Falchi, Claudio Gennaro, Giuseppe Amato* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05275)] \
9 Mar 2023

**Unifying Layout Generation with a Decoupled Diffusion Model** \
*Mude Hui, Zhizheng Zhang, Xiaoyi Zhang, Wenxuan Xie, Yuwang Wang, Yan Lu* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.05049)] \
9 Mar 2023

**DLT: Conditioned layout generation with Joint Discrete-Continuous Diffusion Layout Transformer** \
*Elad Levi, Eli Brosh, Mykola Mykhailych, Meir Perez* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03755)] \
7 Mar 2023

**Diffusion in the Dark: A Diffusion Model for Low-Light Text Recognition** \
*Cindy M. Nguyen, Eric R. Chan, Alexander W. Bergman, Gordon Wetzstein* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04291)] [[Project](https://ccnguyen.github.io/diffusion-in-the-dark/)] \
7 Mar 2023

**Word-As-Image for Semantic Typography** \
*Shir Iluz<sup>1</sup>, Yael Vinker<sup>1</sup>, Amir Hertz, Daniel Berio, Daniel Cohen-Or, Ariel Shamir* \
SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2303.01818)] [[Project](https://wordasimage.github.io/Word-As-Image-Page/)] \
3 Mar 2023

**Makeup Extraction of 3D Representation via Illumination-Aware Image Decomposition** \
*Xingchao Yang, Takafumi Taketomi, Yoshihiro Kanamori* \
Eurographics 2023. [[Paper](https://arxiv.org/abs/2302.13279)] \
26 Feb 2023

**Monocular Depth Estimation using Diffusion Models** \
*Saurabh Saxena, Abhishek Kar, Mohammad Norouzi, David J. Fleet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.14816)] [[Github](https://depth-gen.github.io/)] \
28 Feb 2023

**Spatial-temporal Transformer-guided Diffusion based Data Augmentation for Efficient Skeleton-based Action Recognition** \
*Yifan Jiang, Han Chen, Hanseok Ko* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13434)] \
26 Feb 2023

**LDFA: Latent Diffusion Face Anonymization for Self-driving Applications** \
*Marvin Klemp, Kevin Rösch, Royden Wagner, Jannik Quehl, Martin Lauer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08931)] \
17 Feb 2023

**Road Redesign Technique Achieving Enhanced Road Safety by Inpainting with a Diffusion Model** \
*Sumit Mishra, Medhavi Mishra, Taeyoung Kim, Dongsoo Har* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07440)] \
15 Feb 2023

**Effective Data Augmentation With Diffusion Models** \
*Brandon Trabucco, Kyle Doherty, Max Gurinas, Ruslan Salakhutdinov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07944)] [[Project](https://btrabuc.co/da-fusion/)] \
7 Feb 2023

**Learning End-to-End Channel Coding with Diffusion Models** \
*Muah Kim, Rick Fritschek, Rafael F. Schaefer* \
WSA/SCC 2023. [[Paper](https://arxiv.org/abs/2302.01714)] \
3 Feb 2023

**Extracting Training Data from Diffusion Models** \
*Nicholas Carlini<sup>1</sup>, Jamie Hayes<sup>1</sup>, Milad Nasr<sup>1</sup>, Matthew Jagielski, Vikash Sehwag, Florian Tramèr, Borja Balle, Daphne Ippolito, Eric Wallace* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13188)] \
2 Feb 2023

**Diffusion Models for High-Resolution Solar Forecasts** \
*Yusuke Hatanaka, Yannik Glaser, Geoff Galgon, Giuseppe Torri, Peter Sadowski* \
arxiv 2023. [[Paper](https://arxiv.org/abs/2302.00170)] \
1 Feb 2023

**A Denoising Diffusion Model for Fluid Field Prediction** \
*Gefan Yang<sup>1</sup>, Stefan Sommer<sup>1</sup>* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11661)] \
27 Jan 2023

**Diffusion Models as Artists: Are we Closing the Gap between Humans and Machines?** \
*Victor Boutin, Thomas Fel, Lakshya Singhal, Rishav Mukherji, Akash Nagaraj, Julien Colin, Thomas Serre* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11722)] \
27 Jan 2023

**PLay: Parametrically Conditioned Layout Generation using Latent Diffusion** \
*Chin-Yi Cheng, Forrest Huang, Gang Li, Yang Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11529)] \
27 Jan 2023

**LEGO-Net: Learning Regular Rearrangements of Objects in Rooms** \
*Qiuhong Anna Wei, Sijie Ding, Jeong Joon Park, Rahul Sajnani, Adrien Poulenard, Srinath Sridhar, Leonidas Guibas* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2301.09629)] [[Project](https://ivl.cs.brown.edu/#/projects/lego-net)] \
23 Jan 2023

**Dif-Fusion: Towards High Color Fidelity in Infrared and Visible Image Fusion with Diffusion Models** \
*Jun Yue, Leyuan Fang, Shaobo Xia, Yue Deng, Jiayi Ma* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.08072)] \
19 Jan 2023

**Neural Image Compression with a Diffusion-Based Decoder** \
*Noor Fathima Goose<sup>1</sup>, Jens Petersen<sup>1</sup>, Auke Wiggers<sup>1</sup>, Tianlin Xu, Guillaume Sautière* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.05489)] \
13 Jan 2023

**Diffusion Models For Stronger Face Morphing Attacks** \
*Zander Blasingame, Chen Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.04218)] \
10 Jan 2023

**AI Art in Architecture** \
*Joern Ploennigs, Markus Berger* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09399)] \
19 Dec 2022

**Diffusing Surrogate Dreams of Video Scenes to Predict Video Memorability** \
*Lorin Sweeney, Graham Healy, Alan F. Smeaton* \
MediaEval Workshop 2022. [[Paper](https://arxiv.org/abs/2212.09308)] \
19 Dec 2022

**Diff-Font: Diffusion Model for Robust One-Shot Font Generation** \
*Haibin He, Xinyuan Chen, Chaoyue Wang, Juhua Liu, Bo Du, Dacheng Tao, Yu Qiao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.05895)] \
12 Dec 2022

**How to Backdoor Diffusion Models?** \
*Sheng-Yen Chou, Pin-Yu Chen, Tsung-Yi Ho* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.05400)] \
11 Dec 2022

**Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models** \
*Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.03860)] [[Github](https://github.com/somepago/DCR)] \
7 Dec 2022

**ObjectStitch: Generative Object Compositing** \
*Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, Daniel Aliaga* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00932)] \
2 Dec 2022

**Post-training Quantization on Diffusion Models** \
*Yuzhang Shang<sup>1</sup>, Zhihang Yuan<sup>1</sup>, Bin Xie<sup>1</sup>, Bingzhe Wu, Yan Yan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.15736)] [[Github](https://github.com/42Shawn/PTQ4DM)] \
28 Nov 2022

**Diffusion Probabilistic Model Made Slim** \
*Xingyi Yang, Daquan Zhou, Jiashi Feng, Xinchao Wang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2211.17106)]\
27 Nov 2022

**BeLFusion: Latent Diffusion for Behavior-Driven Human Motion Prediction** \
*German Barquero, Sergio Escalera, Cristina Palmero* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.14304)] [[Project](https://barquerogerman.github.io/BeLFusion/)] [[Github](https://github.com/BarqueroGerman/BeLFusion)]  \
25 Nov 2022

**JigsawPlan: Room Layout Jigsaw Puzzle Extreme Structure from Motion using Diffusion Models** \
*Sepidehsadat Hosseini, Mohammad Amin Shabani, Saghar Irandoust, Yasutaka Furukawa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13785)] [[Project](https://sepidsh.github.io/JigsawPlan/index.html)] \
24 Nov 2022

**HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising** \
*Mohammad Amin Shabani, Sepidehsadat Hosseini, Yasutaka Furukawa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.13287)] [[Project](https://aminshabani.github.io/housediffusion/)] \
23 Nov 2022

**Can denoising diffusion probabilistic models generate realistic astrophysical fields?** \
*Nayantara Mudur, Douglas P. Finkbeiner* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2211.12444)] \
22 Nov 2022

**DiffDreamer: Consistent Single-view Perpetual View Generation with Conditional Diffusion Models** \
*Shengqu Cai, Eric Ryan Chan, Songyou Peng, Mohamad Shahbazi, Anton Obukhov, Luc Van Gool, Gordon Wetzstein* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.12131)] [[Project](https://primecai.github.io/diffdreamer)] \
22 Nov 2022

**CaDM: Codec-aware Diffusion Modeling for Neural-enhanced Video Streaming** \
*Qihua Zhou, Ruibin Li, Song Guo, Yi Liu, Jingcai Guo, Zhenda Xu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.08428)] \
15 Nov 2022

**Extreme Generative Image Compression by Learning Text Embedding from Diffusion Models** \
*Zhihong Pan, Xin Zhou, Hao Tian* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.07793)] \
14 Nov 2022

**Evaluating a Synthetic Image Dataset Generated with Stable Diffusion** \
*Andreas Stöckl* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01777)] \
3 Nov 2022

**On the detection of synthetic images generated by diffusion models** \
*Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, Luisa Verdoliva* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.00680)] [[Github](https://github.com/grip-unina/DMimageDetection)] \
1 Nov 2022

**DOLPH: Diffusion Models for Phase Retrieval** \
*Shirin Shoushtari<sup>1</sup>, Jiaming Liu<sup>1</sup>, Ulugbek S. Kamilov* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.00529)] \
1 Nov 2022

**Towards the Detection of Diffusion Model Deepfakes** \
*Jonas Ricker, Simon Damm, Thorsten Holz, Asja Fischer* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.14571)] \
26 Oct 2022

**Deep Data Augmentation for Weed Recognition Enhancement: A Diffusion Probabilistic Model and Transfer Learning Based Approach** \
*Dong Chen, Xinda Qi, Yu Zheng, Yuzhen Lu, Zhaojian Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.09509)] [[Github](https://github.com/DongChen06/DMWeeds)] \
18 Oct 2022

**DE-FAKE: Detection and Attribution of Fake Images Generated by Text-to-Image Diffusion Models** \
*Zeyang Sha, Zheng Li, Ning Yu, Yang Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.06998)] \
13 Oct 2022

**Markup-to-Image Diffusion Models with Scheduled Sampling** \
*Yuntian Deng, Noriyuki Kojima, Alexander M. Rush* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.05147)] \
11 Oct 2022

**What the DAAM: Interpreting Stable Diffusion Using Cross Attention** \
*Raphael Tang, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Jimmy Lin, Ferhan Ture* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.04885)] [[Github](https://github.com/castorini/daam)] \
10 Oct 2022

**CLIP-Diffusion-LM: Apply Diffusion Model on Image Captioning** \
*Shitong Xu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.04559)] [[Github](https://github.com/xu-shitong/diffusion-image-captioning)] \
10 Oct 2022

**Diffusion Models Beat GANs on Topology Optimization** \
*François Mazé, Faez Ahmed* \
AAAI 2022. [[Paper](https://arxiv.org/abs/2208.09591)] [[Project](https://decode.mit.edu/projects/topodiff/)] [[Github](https://github.com/francoismaze/topodiff)] \
20 Aug 2022

**Vector Quantized Diffusion Model with CodeUnet for Text-to-Sign Pose Sequences Generation** \
*Pan Xie, Qipeng Zhang, Zexian Li, Hao Tang, Yao Du, Xiaohui Hu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.09141)] \
19 Aug 2022

**Deep Diffusion Models for Seismic Processing** \
*Ricard Durall, Ammar Ghanim, Mario Fernandez, Norman Ettrich, Janis Keuper* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.10451)] \
21 Jul 2022

**Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion** \
*Tianpei Gu<sup>1</sup>, Guangyi Chen<sup>1</sup>, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou, Jiwen Lu*\
CVPR 2022. [[Paper](https://arxiv.org/abs/2203.13777)] [[Github](https://github.com/gutianpei/MID)] \
25 Mar 2022



## Audio
### Generation

**EmoMix: Emotion Mixing via Diffusion Models for Emotional Speech Synthesis** \
*Haobin Tang<sup>1</sup>, Xulong Zhang<sup>1</sup>, Jianzong Wang, Ning Cheng, Jing Xiao* \
InterSpeech 2023. [[Paper](https://arxiv.org/abs/2306.00648)] \
1 Jun 2023

**Efficient Neural Music Generation** \ 
*Max W. Y. Lam, Qiao Tian, Tang Li, Zongyu Yin, Siyuan Feng, Ming Tu, Yuliang Ji, Rui Xia, Mingbo Ma, Xuchen Song, Jitong Chen, Yuping Wang, Yuxuan Wang* \ 
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15719)] [[Github](https://efficient-melody.github.io/)] \ 
25 May 2023

**Generating symbolic music using diffusion models** \
*Lilac Atassi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08385)] \
15 Mar 2023

**DiffuseRoll: Multi-track multi-category music generation based on diffusion model** \
*Hongfei Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07794)] \
14 Mar 2023


**Multi-Source Diffusion Models for Simultaneous Music Generation and Separation** \
*Giorgio Mariani<sup>1</sup>, Irene Tallini<sup>1</sup>, Emilian Postolache<sup>1</sup>, Michele Mancusi<sup>1</sup>, Luca Cosmo, Emanuele Rodolà* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02257)] [[Project](https://gladia-research-group.github.io/multi-source-diffusion-models/)] \
4 Feb 2023

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation** \
*Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, Baining Guo* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2212.09478)] [[Github](https://github.com/researchmm/MM-Diffusion)] \
19 Dec 2022

**SDMuse: Stochastic Differential Music Editing and Generation via Hybrid Representation** \
*Chen Zhang, Yi Ren, Kejun Zhang, Shuicheng Yan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.00222)] [[Project](https://sdmuse.github.io/posts/sdmuse/)] \
1 Nov 2022

**Full-band General Audio Synthesis with Score-based Diffusion** \
*Santiago Pascual, Gautam Bhattacharya, Chunghsin Yeh, Jordi Pons, Joan Serrà* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.14661)] \
26 Oct 2022

**Hierarchical Diffusion Models for Singing Voice Neural Vocoder** \
*Naoya Takahashi, Mayank Kumar, Singh, Yuki Mitsufuji* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.07508)] \
14 Oct 2022



**Mandarin Singing Voice Synthesis with Denoising Diffusion Probabilistic Wasserstein GAN** \
*Yin-Ping Cho, Yu Tsao, Hsin-Min Wang, Yi-Wen Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.10446)] [[Project](https://yinping-cho.github.io/diffwgansvs.github.io/)] \
21 Sep 2022


**DDSP-based Singing Vocoders: A New Subtractive-based Synthesizer and A Comprehensive Evaluation** \
*Da-Yi Wu<sup>1</sup>, Wen-Yi Hsiao<sup>1</sup>, Fu-Rong Yang, Oscar Friedman, Warren Jackson, Scott Bruzenak, Yi-Wen Liu, Yi-Hsuan Yang* \
ISMIR 2022. [[Paper](https://arxiv.org/abs/2208.04756)] [[Github](https://github.com/YatingMusic/ddsp-singing-vocoders/)] \
9 Aug 2022

**ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech** \
*Rongjie Huang<sup>1</sup>, Zhou Zhao, Huadai Liu<sup>1</sup>, Jinglin Liu, Chenye Cui, Yi Ren* \
ACM Multimedia 2022. [[Paper](https://arxiv.org/abs/2207.06389)] [[Project](https://prodiff.github.io/)] \
13 Jul 2022


**CARD: Classification and Regression Diffusion Models** \
*Xizewen Han<sup>1</sup>, Huangjie Zheng<sup>1</sup>, Mingyuan Zhou* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.07275)]  \
15 Jun 2022

**Adversarial Audio Synthesis with Complex-valued Polynomial Networks** \
*Yongtao Wu, Grigorios G Chrysos, Volkan Cevher* \
ICML workshop 2022. [[Paper](https://arxiv.org/abs/2206.06811)] \
14 Jun 2022

**Multi-instrument Music Synthesis with Spectrogram Diffusion** \
*Curtis Hawthorne, Ian Simon, Adam Roberts, Neil Zeghidour, Josh Gardner, Ethan Manilow, Jesse Engel* \
ISMIR 2022. [[Paper](https://arxiv.org/abs/2206.05408)] \
11 Jun 2022

**BinauralGrad: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis** \
*Yichong Leng, Zehua Chen, Junliang Guo, Haohe Liu, Jiawei Chen, Xu Tan, Danilo Mandic, Lei He, Xiang-Yang Li, Tao Qin, Sheng Zhao, Tie-Yan Liu* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.14807)] [[Github](https://speechresearch.github.io/binauralgrad/)] \
30 May 2022

**FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis** \
*Rongjie Huang<sup>1</sup>, Max W. Y. Lam<sup>1</sup>, Jun Wang, Dan Su, Dong Yu, Yi Ren, Zhou Zhao* \
IJCAI 2022. [[Paper](https://arxiv.org/abs/2204.09934)] [[Project](https://fastdiff.github.io/)] [[Github](https://github.com/Rongjiehuang/FastDiff)] \
21 Apr 2022

**SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping** \
*Yuma Koizumi, Heiga Zen, Kohei Yatabe, Nanxin Chen, Michiel Bacchiani* \
Interspeech 2022. [[Paper](https://arxiv.org/abs/2203.16749)] \
31 Mar 2022

**BDDM: Bilateral Denoising Diffusion Models for Fast and High-Quality Speech Synthesis** \
*Max W. Y. Lam, Jun Wang, Dan Su, Dong Yu* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2203.13508)] [[Github](https://github.com/tencent-ailab/bddm)] \
25 Mar 2022

**ItôWave: Itô Stochastic Differential Equation Is All You Need For Wave Generation** \
*Shoule Wu<sup>1</sup>, Ziqiang Shi<sup>1</sup>* \
CoRR 2022. [[Paper](https://arxiv.org/abs/2201.12519)] [[Project](https://wushoule.github.io/ItoAudio/)] \
29 Jan 2022

**Itô-Taylor Sampling Scheme for Denoising Diffusion Probabilistic Models using Ideal Derivatives** \
*Hideyuki Tachibana, Mocho Go, Muneyoshi Inahara, Yotaro Katayama, Yotaro Watanabe* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.13339)] \
26 Dec 2021

**Denoising Diffusion Gamma Models** \
*Eliya Nachmani<sup>1</sup>, Robin San Roman<sup>1</sup>, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2110.05948)] \
10 Oct 2021

**Variational Diffusion Models** \
*Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2107.00630)] [[Github](https://github.com/revsic/jax-variational-diffwave)] \
1 Jul 2021 

**CRASH: Raw Audio Score-based Generative Modeling for Controllable High-resolution Drum Sound Synthesis** \
*Simon Rouard<sup>1</sup>, Gaëtan Hadjeres<sup>1</sup>* \
ISMIR 2021. [[Paper](https://arxiv.org/abs/2106.07431)] [[Project](https://crash-diffusion.github.io/crash/)] \
14 Jun 2021

**PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Driven Adaptive Prior** \
*Sang-gil Lee, Heeseung Kim, Chaehun Shin, Xu Tan, Chang Liu, Qi Meng, Tao Qin, Wei Chen, Sungroh Yoon, Tie-Yan Liu* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2106.06406)] [[Project](https://speechresearch.github.io/priorgrad/)] \
11 Jun 2021 

**ItôTTS and ItôWave: Linear Stochastic Differential Equation Is All You Need For Audio Generation** \
*Shoule Wu<sup>1</sup>, Ziqiang Shi<sup>1</sup>* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2105.07583)] [[Project](https://wushoule.github.io/ItoAudio/)] \
17 May 2021

**DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism** \
*Jinglin Liu<sup>1</sup>, Chengxi Li<sup>1</sup>, Yi Ren<sup>1</sup>, Feiyang Chen, Peng Liu, Zhou Zhao* \
AAAI 2022. [[Paper](https://arxiv.org/abs/2105.02446)] [[Project](https://diffsinger.github.io/)] [[Github](https://github.com/keonlee9420/DiffSinger)] \
6 May 2021

**Symbolic Music Generation with Diffusion Models** \
*Gautam Mittal, Jesse Engel, Curtis Hawthorne, Ian Simon* \
ISMIR 2021. [[Paper](https://arxiv.org/abs/2103.16091)] [[Github](https://github.com/magenta/symbolic-music-diffusion)] \
30 Mar 2021 

**DiffWave: A Versatile Diffusion Model for Audio Synthesis** \
*Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, Bryan Catanzaro* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2009.09761)] [[Github](https://diffwave-demo.github.io/)] \
21 Sep 2020 

**WaveGrad: Estimating Gradients for Waveform Generation** \
*Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, William Chan*\
ICLR 2021. [[Paper](https://arxiv.org/abs/2009.00713)] [[Project](https://wavegrad.github.io/)] [[Github](https://github.com/ivanvovk/WaveGrad)] \
2 Sep 2020 


### Conversion

**DDDM-VC: Decoupled Denoising Diffusion Models with Disentangled Representation and Prior Mixup for Verified Robust Voice Conversion** \ 
*Ha-Yeong Choi<sup>1</sup>, Sang-Hoon Lee<sup>1</sup>, Seong-Whan Lee* \ 
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15816)] [[Project](https://hayeong0.github.io/DDDM-VC-demo/)] \ 
25 May 2023

**Duplex Diffusion Models Improve Speech-to-Speech Translation** \
*Xianchao Wu* \
ACL 2023. [[Paper](https://arxiv.org/abs/2305.12628)] \
22 May 2023

**DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion**  \
*Songxiang Liu<sup>1</sup>, Yuewen Cao<sup>1</sup>, Dan Su, Helen Meng* \
IEEE 2021. [[Paper](https://arxiv.org/abs/2105.13871)] [[Github](https://github.com/liusongxiang/diffsvc)] \
28 May 2021

**Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme** \
*Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov, Jiansheng Wei* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2109.13821)] [[Project](https://diffvc-fast-ml-solver.github.io/)] \
28 Sep 2021

### Enhancement

**UnDiff: Unsupervised Voice Restoration with Unconditional Diffusion Model** \
*Anastasiia Iashchenko<sup>1</sup>, Pavel Andreev<sup>1</sup>, Ivan Shchekotov, Nicholas Babaev, Dmitry Vetrov* \
Interspeech 2023. [[Paper](https://arxiv.org/abs/2306.00721)] \
1 Jun 2023

**SE-Bridge: Speech Enhancement with Consistent Brownian Bridge** \
*Zhibin Qiu, Mengfan Fu, Fuchun Sun, Gulila Altenbek, Hao Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13796)] \
23 May 2023

**Diffusion-Based Speech Enhancement with Joint Generative and Predictive Decoders** \
*Hao Shi, Kazuki Shimada, Masato Hirano, Takashi Shibuya, Yuichiro Koyama, Zhi Zhong, Shusuke Takahashi, Tatsuya Kawahara, Yuki Mitsufuji* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10734)] \
18 May 2023

**Speech Signal Improvement Using Causal Generative Diffusion Models** \
*Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Tal Peer, Timo Gerkmann* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2303.08674)] \
15 Mar 2023

**Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement** \
*Bunlong Lay, Simon Welker, Julius Richter, Timo Gerkmann* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.14748)] \
28 Feb 2023

**Metric-oriented Speech Enhancement using Diffusion Probabilistic Model** \
*Chen Chen, Yuchen Hu, Weiwei Weng, Eng Siong Chng* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.11989)] \
23 Feb 2023

**StoRM: A Diffusion-based Stochastic Regeneration Model for Speech Enhancement and Dereverberation** \
*Jean-Marie Lemercier, Julius Richter, Simon Welker, Timo Gerkmann* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2212.11851)] \
22 Dec 2022


**Unsupervised vocal dereverberation with diffusion-based generative models** \
*Koichi Saito, Naoki Murata, Toshimitsu Uesaka, Chieh-Hsin Lai, Yuhta Takida, Takao Fukui, Yuki Mitsufuji* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2211.04124)] \
8 Nov 2022

**DiffPhase: Generative Diffusion-based STFT Phase Retrieval** \
*Tal Peer, Simon Welker, Timo Gerkmann* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2211.04332)] \
8 Nov 2022


**Cold Diffusion for Speech Enhancement** \
*Hao Yen, François G. Germain, Gordon Wichern, Jonathan Le Roux* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2211.02527)] \
4 Nov 2022


**Analysing Diffusion-based Generative Approaches versus Discriminative Approaches for Speech Restoration** \
*Jean-Marie Lemercier, Julius Richter, Simon Welker, Timo Gerkmann* \
Interspeech 2022. [[Paper](https://arxiv.org/abs/2211.02397)] [[Project](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse-multitask.html)] [[Github](https://github.com/sp-uhh/sgmse)] \
4 Nov 2022

**SRTNet: Time Domain Speech Enhancement Via Stochastic Refinement** \
*Zhibin Qiu, Mengfan Fu, Yinfeng Yu, LiLi Yin, Fuchun Sun, Hao Huang* \
ICASSP 2022. [[Paper](https://arxiv.org/abs/2210.16805)] [[Github](https://github.com/zhibinQiu/SRTNet)] \
30 Oct 2022

**A Versatile Diffusion-based Generative Refiner for Speech Enhancement** \
*Ryosuke Sawata, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Takashi Shibuya, Shusuke Takahashi, Yuki Mitsufuji* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2210.17287)] \
27 Oct 2022

**Conditioning and Sampling in Variational Diffusion Models for Speech Super-resolution** \
*Chin-Yun Yu, Sung-Lin Yeh, György Fazekas, Hao Tang* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2210.15793)] [[Project](https://yoyololicon.github.io/diffwave-sr/)] [[Github](https://github.com/yoyololicon/diffwave-sr)] \
27 Oct 2022

**Solving Audio Inverse Problems with a Diffusion Model** \
*Eloi Moliner, Jaakko Lehtinen, Vesa Välimäki* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2210.15228)] \
27 Oct 2022

**Speech Enhancement and Dereverberation with Diffusion-based Generative Models** \
*Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.05830)] [[Project](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse)] [[Github](https://github.com/sp-uhh/sgmse)] \
11 Aug 2022

**NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates** \
*Seungu Han, Junhyeok Lee* \
Interspeech 2022. [[Paper](https://arxiv.org/abs/2206.08545)] [[Project](https://mindslab-ai.github.io/nuwave2/)] \
17 Jun 2022

**Universal Speech Enhancement with Score-based Diffusion** \
*Joan Serrà, Santiago Pascual, Jordi Pons, R. Oguz Araz, Davide Scaini* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.03065)] \
7 Jun 2022

**Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain** \
*Simon Welker, Julius Richter, Timo Gerkmann* \
InterSpeech 2022. [[Paper](https://arxiv.org/abs/2203.17004)]  [[Github](https://github.com/sp-uhh/sgmse)] \
31 Mar 2022

**Conditional Diffusion Probabilistic Model for Speech Enhancement** \
*Yen-Ju Lu, Zhong-Qiu Wang, Shinji Watanabe, Alexander Richard, Cheng Yu, Yu Tsao* \
IEEE 2022. [[Paper](https://arxiv.org/abs/2202.05256)] [[Github](https://github.com/neillu23/cdiffuse)] \
10 Feb 2022

**A Study on Speech Enhancement Based on Diffusion Probabilistic Model** \
*Yen-Ju Lu<sup>1</sup>, Yu Tsao<sup>1</sup>, Shinji Watanabe* \
APSIPA 2021. [[Paper](https://arxiv.org/abs/2107.11876)] \
25 Jul 2021


**Restoring degraded speech via a modified diffusion model** \
*Jianwei Zhang, Suren Jayasuriya, Visar Berisha* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.11347)] \
22 Apr 2021

**NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling**  \
*Junhyeok Lee, Seungu Han* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.02321)] [[Project](https://mindslab-ai.github.io/nuwave/)] [[Github](https://github.com/mindslab-ai/nuwave)] \
6 Apr 2021




### Separation

**Diffusion-based Signal Refiner for Speech Separation** \
*Masato Hirano, Kazuki Shimada, Yuichiro Koyama, Shusuke Takahashi, Yuki Mitsufuji* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05857)] \
10 May 2023

**Multi-Source Diffusion Models for Simultaneous Music Generation and Separation** \
*Giorgio Mariani<sup>1</sup>, Irene Tallini<sup>1</sup>, Emilian Postolache<sup>1</sup>, Michele Mancusi<sup>1</sup>, Luca Cosmo, Emanuele Rodolà* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02257)] [[Project](https://gladia-research-group.github.io/multi-source-diffusion-models/)] \
4 Feb 2023

**Separate And Diffuse: Using a Pretrained Diffusion Model for Improving Source Separation** \
*Shahar Lutati, Eliya Nachmani, Lior Wolf* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.10752)] \
25 Jan 2023

**StoRM: A Diffusion-based Stochastic Regeneration Model for Speech Enhancement and Dereverberation** \
*Jean-Marie Lemercier, Julius Richter, Simon Welker, Timo Gerkmann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.11851)] [[Github](https://github.com/sp-uhh/storm)] \
22 Dec 2022

**Diffusion-based Generative Speech Source Separation** \
*Robin Scheibler, Youna Ji, Soo-Whan Chung, Jaeuk Byun, Soyeon Choe, Min-Seok Choi* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2210.17327)] \
31 Oct 2022

**Instrument Separation of Symbolic Music by Explicitly Guided Diffusion Model** \
*Sangjun Han, Hyeongrae Ihm, DaeHan Ahn, Woohyung Lim* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2209.02696)] \
5 Sep 2022



### Text-to-Speech

**Mega-TTS: Zero-Shot Text-to-Speech at Scale with Intrinsic Inductive Bias** \
*Ziyue Jiang<sup>1</sup>, Yi Ren<sup>1</sup>, Zhenhui Ye<sup>1</sup>, Jinglin Liu, Chen Zhang, Qian Yang, Shengpeng Ji, Rongjie Huang, Chunfeng Wang, Xiang Yin, Zejun Ma, Zhou Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03509)] [[Github](https://mega-tts.github.io/demo-page/)] \
6 Jun 2023

**Make-An-Audio 2: Temporal-Enhanced Text-to-Audio Generation** \
*Jiawei Huang<sup>1</sup>, Yi Ren<sup>1</sup>, Rongjie Huang, Dongchao Yang, Zhenhui Ye, Chen Zhang, Jinglin Liu, Xiang Yin, Zejun Ma, Zhou Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18474)] \
29 May 2023

**ZET-Speech: Zero-shot adaptive Emotion-controllable Text-to-Speech Synthesis with Diffusion and Style-based Models** \
*Minki Kang<sup>1</sup>, Wooseok Han<sup>1</sup>, Sung Ju Hwang, Eunho Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13831)] \
23 May 2023

**U-DiT TTS: U-Diffusion Vision Transformer for Text-to-Speech** \
*Xin Jing, Yi Chang, Zijiang Yang, Jiangjian Xie, Andreas Triantafyllopoulos, Bjoern W. Schuller* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13195)] [[Project](https://eihw.github.io/u-dit-tts/)] \
22 May 2023

**DiffAVA: Personalized Text-to-Audio Generation with Visual Alignment** \
*Shentong Mo, Jing Shi, Yapeng Tian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12903)] \
22 May 2023

**ViT-TTS: Visual Text-to-Speech with Scalable Diffusion Transformer** \
*Huadai Liu<sup>1</sup>, Rongjie Huang<sup>1</sup>, Xuan Lin, Wenqiang Xu, Maozong Zheng, Hong Chen, Jinzheng He, Zhou Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12708)] [[Project](https://vit-tts.github.io/)] \
22 May 2023

**RMSSinger: Realistic-Music-Score based Singing Voice Synthesis** \
*Jinzheng He, Jinglin Liu, Zhenhui Ye, Rongjie Huang, Chenye Cui, Huadai Liu, Zhou Zhao* \
ACL 2023. [[Paper](https://arxiv.org/abs/2305.10686)] [[Project](https://rmssinger.github.io/)] \
18 May 2023

**CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model** \
*Zhen Ye, Wei Xue, Xu Tan, Jie Chen, Qifeng Liu, Yike Guo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.06908)] [[Github](https://comospeech.github.io/)] \
11 May 2023


**Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model** \
*Deepanway Ghosal, Navonil Majumder, Ambuj Mehrish, Soujanya Poria* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13731)] [[Project](https://tango-web.github.io/)] [[Github](https://github.com/declare-lab/tango)] \
24 Apr 2023

**DiffVoice: Text-to-Speech with Latent Diffusion** \
*Zhijun Liu, Yiwei Guo, Kai Yu* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2304.11750)] \
23 Apr 2023

**An investigation into the adaptability of a diffusion-based TTS model** \
*Haolin Chen, Philip N. Garner* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.01849)] \
3 Mar 2023

**Imaginary Voice: Face-styled Diffusion Model for Text-to-Speech** \
*Jiyoung Lee, Joon Son Chung, Soo-Whan Chung* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2302.13700)] \
27 Feb 2023

**ERNIE-Music: Text-to-Waveform Music Generation with Diffusion Models** \
*Pengfei Zhu, Chao Pang, Shuohuan Wang, Yekun Chai, Yu Sun, Hao Tian, Hua Wu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04456)] \
9 Feb 2023

**Noise2Music: Text-conditioned Music Generation with Diffusion Models** \
*Qingqing Huang<sup>1</sup>, Daniel S. Park<sup>1</sup>, Tao Wang, Timo I. Denk, Andy Ly, Nanxin Chen, Zhengdong Zhang, Zhishuai Zhang, Jiahui Yu, Christian Frank, Jesse Engel, Quoc V. Le, William Chan, Wei Han* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03917)] [[Project](https://google-research.github.io/noise2music/)] \
8 Feb 2023

**Moûsai: Text-to-Music Generation with Long-Context Latent Diffusion** \
*Flavio Schneider, Zhijing Jin, Bernhard Schölkopf* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11757)] [[Project](https://anonymous0.notion.site/anonymous0/Mo-sai-Text-to-Audio-with-Long-Context-Latent-Diffusion-b43dbc71caf94b5898f9e8de714ab5dc)] [[Github](https://github.com/archinetai/audio-diffusion-pytorch)] \
27 Jan 2023

**InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt** \
*Dongchao Yang<sup>1</sup>, Songxiang Liu<sup>1</sup>, Rongjie Huang, Guangzhi Lei, Chao Weng, Helen Meng, Dong Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13662)] [[Project](http://dongchaoyang.top/InstructTTS/)] \
31 Jan 2023


**Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models** \
*Rongjie Huang<sup>1</sup>, Jiawei Huang<sup>1</sup>, Dongchao Yang<sup>1</sup>, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, Zhou Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12661)] [[Project](https://text-to-audio.github.io/)] \
30 Jan 2023

**AudioLDM: Text-to-Audio Generation with Latent Diffusion Models** \
*Haohe Liu<sup>1</sup>, Zehua Chen<sup>1</sup>, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, Mark D. Plumbley* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12503)] [[Project](https://audioldm.github.io/)] [[Github](https://github.com/haoheliu/AudioLDM)] \
29 Jan 2023

**ResGrad: Residual Denoising Diffusion Probabilistic Models for Text to Speech** \
*Zehua Chen, Yihan Wu, Yichong Leng, Jiawei Chen, Haohe Liu, Xu Tan, Yang Cui, Ke Wang, Lei He, Sheng Zhao, Jiang Bian, Danilo Mandic* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.14518)] [[Project](https://resgrad1.github.io/)] \
30 Dec 2022


**Text-to-speech synthesis based on latent variable conversion using diffusion probabilistic model and variational autoencoder** \
*Yusuke Yasuda, Tomoki Toda* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2212.08329)] \
16 Dec 2022

**Any-speaker Adaptive Text-To-Speech Synthesis with Diffusion Models** \
*Minki Kang, Dongchan Min, Sung Ju Hwang* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2211.09383)] [[Project](https://nardien.github.io/grad-stylespeech-demo/)] \
17 Nov 2022

**EmoDiff: Intensity Controllable Emotional Text-to-Speech with Soft-Label Guidance** \
*Yiwei Guo, Chenpeng Du, Xie Chen, Kai Yu* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2211.09496)] [[Project](https://cantabile-kwok.github.io/EmoDiff-intensity-ctrl/)] \
17 Nov 2022

**NoreSpeech: Knowledge Distillation based Conditional Diffusion Model for Noise-robust Expressive TTS** \
*Dongchao Yang, Songxiang Liu, Jianwei Yu, Helin Wang, Chao Weng, Yuexian Zou* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2211.02448)] \
4 Nov 2022

**WaveFit: An Iterative and Non-autoregressive Neural Vocoder based on Fixed-Point Iteration** \
*Yuma Koizumi, Kohei Yatabe, Heiga Zen, Michiel Bacchiani* \
IEEE SLT 2023. [[Paper](https://arxiv.org/abs/2210.01029)] [[Project](https://google.github.io/df-conformer/wavefit/)] \
3 Oct 2022

**Diffsound: Discrete Diffusion Model for Text-to-sound Generation** \
*Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, Dong Yu* \
TASLP 2022. [[Paper](https://arxiv.org/abs/2207.09983)] [[Project](http://dongchaoyang.top/text-to-sound-synthesis-demo/)] \
20 Jul 2022

**Zero-Shot Voice Conditioning for Denoising Diffusion TTS Models** \
*Alon Levkovitch, Eliya Nachmani, Lior Wolf* \
Interspeech 2022. [[Paper](https://arxiv.org/abs/2206.02246)] [[Project](https://alonlevko.github.io/ilvr-tts-diff)] \
5 Jun 2022

**Guided-TTS 2: A Diffusion Model for High-quality Adaptive Text-to-Speech with Untranscribed Data** \
*Sungwon Kim<sup>1</sup>, Heeseung Kim<sup>1</sup>, Sungroh Yoon* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.15370)] [[Project](https://ksw0306.github.io/guided-tts2-demo/)] \
30 May 2022

**InferGrad: Improving Diffusion Models for Vocoder by Considering Inference in Training** \
*Zehua Chen, Xu Tan, Ke Wang, Shifeng Pan, Danilo Mandic, Lei He, Sheng Zhao* \
ICASSP 2022. [[Paper](https://arxiv.org/abs/2202.03751)] \
8 Feb 2022

**DiffGAN-TTS: High-Fidelity and Efficient Text-to-Speech with Denoising Diffusion GANs** \
*Songxiang Liu, Dan Su, Dong Yu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11972)] [[Github](https://github.com/keonlee9420/DiffGAN-TTS)] \
28 Jan 2022

**Guided-TTS:Text-to-Speech with Untranscribed Speech** \
*Heeseung Kim, Sungwon Kim, Sungroh Yoon* \
ICML 2021. [[Paper](https://arxiv.org/abs/2111.11755)] \
30 Nov 2021



**EdiTTS: Score-based Editing for Controllable Text-to-Speech** \
*Jaesung Tae<sup>1</sup>, Hyeongju Kim<sup>1</sup>, Taesu Kim* \
Interspeech 2022. [[Paper](https://arxiv.org/abs/2110.02584)] [[Project](https://editts.github.io/)] [[Github](https://github.com/neosapience/EdiTTS)] \
6 Oct 2021

**WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis** \
*Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, Najim Dehak, William Chan* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2106.09660)] [[Project](https://mindslab-ai.github.io/wavegrad2/)] [[Github](https://github.com/keonlee9420/WaveGrad2)] [[Github2](https://github.com/mindslab-ai/wavegrad2)] \
17 Jun 2021 

**Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech** \
*Vadim Popov<sup>1</sup>, Ivan Vovk<sup>1</sup>, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov* \
ICML 2021. [[Paper](https://arxiv.org/abs/2105.06337)] [[Project](https://grad-tts.github.io/)] [[Github](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)] \
13 May 2021

**DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism** \
*Jinglin Liu<sup>1</sup>, Chengxi Li<sup>1</sup>, Yi Ren<sup>1</sup>, Feiyang Chen, Peng Liu, Zhou Zhao* \
AAAI 2022. [[Paper](https://arxiv.org/abs/2105.02446)] [[Project](https://diffsinger.github.io/)] [[Github](https://github.com/keonlee9420/DiffSinger)] \
6 May 2021

**Diff-TTS: A Denoising Diffusion Model for Text-to-Speech**  \
*Myeonghun Jeong, Hyeongju Kim, Sung Jun Cheon, Byoung Jin Choi, Nam Soo Kim* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.01409)] \
3 Apr 2021

### Miscellany

**Zero-Shot Blind Audio Bandwidth Extension** \
*Eloi Moliner, Filip Elvander, Vesa Välimäki* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01433)] \
2 Jun 2023


**Diverse and Expressive Speech Prosody Prediction with Denoising Diffusion Probabilistic Model** \
*Xiang Li, Songxiang Liu, Max W. Y. Lam, Zhiyong Wu, Chao Weng, Helen Meng* \
Interspeech 2023. [[Paper](https://arxiv.org/abs/2305.16749)] \
26 May 2023


**Diffusion-Based Audio Inpainting** \
*Eloi Moliner, Vesa Välimäki* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15266)] \
24 May 2023

**FluentSpeech: Stutter-Oriented Automatic Speech Editing with Context-Aware Diffusion Models** \ 
*Ziyue Jiang<sup>1</sup>, Qian Yang<sup>1</sup>, Jialong Zuo, Zhenhui Ye, Rongjie Huang, Yi Ren, Zhou Zhao* \ 
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13612)] [[Github](https://github.com/Zain-Jiang/Speech-Editing-Toolkit)] \ 
23 May 2023

**A Preliminary Study on Augmenting Speech Emotion Recognition using a Diffusion Model** \
*Ibrahim Malik<sup>1</sup>, Siddique Latif<sup>1</sup>, Raja Jurdak, Björn Schuller* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11413)] \
19 May 2023

**AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models** \
*Yuancheng Wang, Zeqian Ju, Xu Tan, Lei He, Zhizheng Wu, Jiang Bian, Sheng Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.00830)] [[Project](https://audit-demo.github.io/)] \
3 Apr 2023

**Data Augmentation for Environmental Sound Classification Using Diffusion Probabilistic Model with Top-k Selection Discriminator** \
*Yunhao Chen, Yunjie Zhu, Zihui Yan, Jianlu Shen, Zhen Ren, Yifan Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15161)] [[Github](https://github.com/JNAIC/DPMs-for-Audio-Data-Augmentation)] \
27 Mar 2023

**Enhancing Unsupervised Speech Recognition with Diffusion GANs** \
*Xianchao Wu* \
ICASSP 2023. [[Paper](https://arxiv.org/abs/2303.13559)] \
23 Mar 2023

**Defending against Adversarial Audio via Diffusion Model** \
*Shutong Wu, Jiongxiao Wang, Wei Ping, Weili Nie, Chaowei Xiao* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2303.01507)] [[Github](https://github.com/cychomatica/AudioPure)] \
2 Mar 2023

**TransFusion: Transcribing Speech with Multinomial Diffusion** \
*Matthew Baas, Kevin Eloff, Herman Kamper* \
SACAIR 2022. [[Paper](https://arxiv.org/abs/2210.07677)] [[Github](https://github.com/RF5/transfusion-asr)] \
14 Oct 2022

## Natural Language

**PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model** \
*Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Josh Susskind, Navdeep Jaitly* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02531)] \
5 Jun 2023

**DiffusEmp: A Diffusion Model-Based Framework with Multi-Grained Control for Empathetic Response Generation** \
*Guanqun Bi, Lei Shen, Yanan Cao, Meng Chen, Yuqiang Xie, Zheng Lin, Xiaodong He* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01657)] \
2 Jun 2023

**Perturbation-Assisted Sample Synthesis: A Novel Approach for Uncertainty Quantification** \
*Yifei Liu, Rex Shen, Xiaotong Shen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18671)] \
30 May 2023

**Likelihood-Based Diffusion Language Models** \
*Ishaan Gulrajani, Tatsunori B. Hashimoto* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18619)] [[Github](https://github.com/igul222/plaid)] \
30 May 2023

**Learning to Imagine: Visually-Augmented Natural Language Generation** \
*Tianyi Tang, Yushuo Chen, Yifan Du, Junyi Li, Wayne Xin Zhao, Ji-Rong Wen* \
ACL 2023. [[Paper](https://arxiv.org/abs/2305.16944)] \
26 May 2023

**Decomposing the Enigma: Subgoal-based Demonstration Learning for Formal Theorem Proving** \
*Xueliang Zhao, Wenda Li, Lingpeng Kong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16366)] \
25 May 2023


**Dior-CVAE: Diffusion Priors in Variational Dialog Generation** \
*Tianyu Yang, Thy Thy Tran, Iryna Gurevych* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15025)] \
24 May 2023


**SSD-2: Scaling and Inference-time Fusion of Diffusion Language Models** \
*Xiaochuang Han, Sachin Kumar, Yulia Tsvetkov, Marjan Ghazvininejad* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14771)] \
24 May 2023

**DiffusionNER: Boundary Diffusion for Named Entity Recognition** \
*Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang* \
ACL 2023. [[Paper](https://arxiv.org/abs/2305.13298)] \
22 May 2023

**DiffCap: Exploring Continuous Diffusion on Image Captioning** \
*Yufeng He<sup>1</sup>, Zefan Cai<sup>1</sup>, Xu Gan, Baobao Chang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12144)] \
20 May 2023

**DiffuSIA: A Spiral Interaction Architecture for Encoder-Decoder Text Diffusion** \
*Chao-Hong Tan, Jia-Chen Gu, Zhen-Hua Ling* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11517)] \
19 May 2023

**Democratized Diffusion Language Model** \
*Nikita Balagansky, Daniil Gavrilov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10818)] \
18 May 2023

**AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation** \
*Tong Wu<sup>1</sup>, Zhihao Fan<sup>1</sup>, Xiao Liu, Yeyun Gong, Yelong Shen, Jian Jiao, Hai-Tao Zheng, Juntao Li, Zhongyu Wei, Jian Guo, Nan Duan, Weizhu Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09515)] \
16 May 2023

**TESS: Text-to-Text Self-Conditioned Simplex Diffusion** \
*Rabeeh Karimi Mahabadi, Jaesung Tae, Hamish Ivison, James Henderson, Iz Beltagy, Matthew E. Peters, Arman Cohan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08379)] \
15 May 2023

**Large Language Models Need Holistically Thought in Medical Conversational QA** \
*Yixuan Weng, Bin Li, Fei Xia, Minjun Zhu, Bin Sun, Shizhu He, Kang Liu, Jun Zhao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.05410)] [[Github](https://github.com/WENGSYX/HoT)] \
9 May 2023

**Diffusion Theory as a Scalpel: Detecting and Purifying Poisonous Dimensions in Pre-trained Language Models Caused by Backdoor or Bias** \
*Zhiyuan Zhang, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, Xu Sun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04547)] \
8 May 2023

**Can Diffusion Model Achieve Better Performance in Text Generation? Bridging the Gap between Training and Inference!** \
*Zecheng Tang<sup>1</sup>, Pinzheng Wang<sup>1</sup>, Keyan Zhou, Juntao Li, Ziqiang Cao, Min Zhang* \
ACL 2023. [[Paper](https://arxiv.org/abs/2305.04465)] [[Github](https://github.com/CODINNLG/Bridge_Gap_Diffusion)] \
8 May 2023


**Diffusion-NAT: Self-Prompting Discrete Diffusion for Non-Autoregressive Text Generation** \
*Kun Zhou, Yifan Li, Wayne Xin Zhao, Ji-Rong Wen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04044)] \
6 May 2023

**DiffuSum: Generation Enhanced Extractive Summarization with Diffusion** \
*Haopeng Zhang<sup>1</sup>, Xiao Liu<sup>1</sup>, Jiawei Zhang* \
ACL 2023. [[Paper](https://arxiv.org/abs/2305.01735)] \
2 May 2023

**RenderDiffusion: Text Generation as Image Generation** \
*Junyi Li, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12519)] \
25 Apr 2023


**TTIDA: Controllable Generative Data Augmentation via Text-to-Text and Text-to-Image Models** \
*Yuwei Yin, Jean Kaddour, Xiang Zhang, Yixin Nie, Zhenguang Liu, Lingpeng Kong, Qi Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08821)] \
18 Apr 2023

**A Cheaper and Better Diffusion Language Model with Soft-Masked Noise** \
*Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, Diyi Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04746)] [[Github](https://github.com/amazon-science/masked-diffusion-lm)] \
10 Apr 2023

**DINOISER: Diffused Conditional Sequence Learning by Manipulating Noises** \
*Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, Mingxuan Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10025)] \
20 Feb 2023

**A Reparameterized Discrete Diffusion Model for Text Generation** \
*Lin Zheng, Jianbo Yuan, Lei Yu, Lingpeng Kong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05737)] [[Github](https://github.com/HKUNLP/reparam-discrete-diffusion)] \
11 Feb 2023

**Long Horizon Temperature Scaling** \
*Andy Shih, Dorsa Sadigh, Stefano Ermon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03686)] [[Github](https://github.com/AndyShih12/LongHorizonTemperatureScaling)] \
7 Feb 2023

**DiffusER: Diffusion via Edit-based Reconstruction** \
*Machel Reid, Vincent Josua Hellendoorn, Graham Neubig* \
ICLR 2023. [[Paper](https://openreview.net/forum?id=nG9RF9z1yy3)] \
2 Feb 2023

**GENIE: Large Scale Pre-training for Text Generation with Diffusion Model** \
*Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, Weizhu Chen, Nan Duan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.11685)] \
22 Dec 2022

**Diff-Glat: Diffusion Glancing Transformer for Parallel Sequence to Sequence Learning** \
*Lihua Qian, Mingxuan Wang, Yang Liu, Hao Zhou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10240)] \
20 Dec 2022

**SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers** \
*Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Fei Huang, Songfang Huang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10325)] \
20 Dec 2022

**Latent Diffusion for Language Generation** \
*Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, Kilian Weinberger* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09462)] \
19 Dec 2022

**Difformer: Empowering Diffusion Model on Embedding Space for Text Generation** \
*Zhujin Gao<sup>1</sup>, Junliang Guo<sup>1</sup>, Xu Tan, Yongxin Zhu, Fang Zhang, Jiang Bian, Linli Xu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09412)] \
19 Dec 2022

**DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models** \
*Zhengfu He<sup>1</sup>, Tianxiang Sun<sup>1</sup>, Kuanning Wang, Xuanjing Huang, Xipeng Qiu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.15029)] [[Github](https://github.com/Hzfinfdu/Diffusion-BERT)] \
28 Nov 2022

**Continuous diffusion for categorical data** \
*Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H. Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Rémi Leblond, Will Grathwohl, Jonas Adler* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.15089)] \
28 Nov 2022


**Self-conditioned Embedding Diffusion for Text Generation** \
*Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, Rémi Leblond* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.04236)] \
8 Nov 2022

**DiffusER: Discrete Diffusion via Edit-based Reconstruction** \
*Machel Reid, Vincent J. Hellendoorn, Graham Neubig* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.16886)] \
30 Oct 2022

**SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control** \
*Xiaochuang Han, Sachin Kumar, Yulia Tsvetkov* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.17432)] [[Github](https://github.com/xhan77/ssd-lm)] \
31 Oct 2022


**DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models** \
*Shansan Gong<sup>1</sup>, Mukai Li<sup>1</sup>, Jiangtao Feng, Zhiyong Wu, LingPeng Kong* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.08933)] \
17 Oct 2022

**Composable Text Controls in Latent Space with ODEs** \
*Guangyi Liu, Zeyu Feng, Yuan Gao, Zichao Yang, Xiaodan Liang, Junwei Bao, Xiaodong He, Shuguang Cui, Zhen Li, Zhiting Hu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2208.00638)] [[Github](https://github.com/guangyliu/LatentOps)] \
1 Aug 2022


**Structured Denoising Diffusion Models in Discrete State-Spaces** \
*Jacob Austin<sup>1</sup>, Daniel D. Johnson<sup>1</sup>, Jonathan Ho, Daniel Tarlow, Rianne van den Berg* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2107.03006)] \
7 Jul 2021 

**Latent Diffusion Energy-Based Model for Interpretable Text Modeling** \
*Peiyu Yu, Sirui Xie, Xiaojian Ma, Baoxiong Jia, Bo Pang, Ruigi Gao, Yixin Zhu, Song-Chun Zhu, Ying Nian Wu* \
ICML 2022. [[Paper](https://arxiv.org/abs/2206.05895)] [[Github](https://github.com/yuPeiyu98/LDEBM)] \
13 Jun 2022

**Diffusion-LM Improves Controllable Text Generation** \
*Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, Tatsunori B. Hashimoto* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.14217)] [[Github](https://github.com/XiangLi1999/Diffusion-LM)] \
27 May 2022


**Step-unrolled Denoising Autoencoders for Text Generation** \
*Nikolay Savinov, Junyoung Chung, Mikolaj Binkowski, Erich Elsen, Aaron van den Oord* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2112.06749)] [[Github](https://github.com/vvvm23/sundae)] \
13 Dec 2021


**Zero-Shot Translation using Diffusion Models** \
*Eliya Nachmani<sup>1</sup>, Shaked Dovrat<sup>1</sup>* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.01471)] \
2 Nov 2021


**Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions** \
*Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, Max Welling* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2102.05379)] \
10 Feb 2021

## Tabular and Time Series

### Generation


**DiffECG: A Generalized Probabilistic Diffusion Model for ECG Signals Synthesis** \
*Nour Neifar, Achraf Ben-Hamadou, Afef Mdhaffar, Mohamed Jmaiel* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01875)] \
2 Jun 2023

**CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis** \
*Chaejeong Lee, Jayoung Kim, Noseong Park* \
ICML 2023. [[Paper](https://arxiv.org/abs/2304.12654)] \
25 Apr 2023

**Customized Load Profiles Synthesis for Electricity Customers Based on Conditional Diffusion Models** \
*Zhenyi Wang, Hongcai Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12076)] \
24 Apr 2023

**Synthetic Health-related Longitudinal Data with Mixed-type Variables Generated using Diffusion Models** \
*Nicholas I-Hsien Kuo, Louisa Jorm, Sebastiano Barbieri* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12281)] \
22 Mar 2023

**Diffusing Gaussian Mixtures for Generating Categorical Data** \
*Florence Regol, Mark Coates* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04635)] \
8 Mar 2023

**EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models** \
*Hongyi Yuan<sup>1</sup>, Songchi Zhou<sup>1</sup>, Sheng Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05656)] [[Github](https://github.com/sczzz3/ehrdiff)] \
10 Mar 2023

**Synthesizing Mixed-type Electronic Health Records using Diffusion Models** \
*Taha Ceritli, Ghadeer O. Ghosheh, Vinod Kumar Chauhan, Tingting Zhu, Andrew P. Creagh, David A. Clifton* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.14679)] \
28 Feb 2023

**MedDiff: Generating Electronic Health Records using Accelerated Denoising Diffusion Model** \
*Huan He, Shifan Zhao, Yuanzhe Xi, Joyce C Ho* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04355)] \
8 Feb 2023

**Diffusion-based Conditional ECG Generation with Structured State Space Models** \
*Juan Miguel Lopez Alcaraz, Nils Strodthoff* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.08227)] \
19 Jan 2023

**TabDDPM: Modelling Tabular Data with Diffusion Models** \
*Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, Artem Babenko* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.15421)] [[Github](https://github.com/rotot0/tab-ddpm)] \
30 Sep 2022



### Forecasting

**DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting** \
*Salva Rühling Cachay, Bo Zhao, Hailey James, Rose Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01984)] \
3 Jun 2023

**DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model** \
*Zhixian Wang, Qingsong Wen, Chaoli Zhang, Liang Sun, Yi Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01001)] \
31 May 2023


**TDSTF: Transformer-based Diffusion probabilistic model for Sparse Time series Forecasting** \
*Ping Chang, Huayu Li, Stuart F. Quan, Janet Roveda, Ao Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.06625)] [[Github](https://github.com/PingChang818/TDSTF)] \
16 Jan 2023

**Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement** \
*Yan Li, Xinjiang Lu, Yaqing Wang, Dejing Dou* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2301.03028)] [[Github](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE)] \
8 Jan 2023

**Denoising diffusion probabilistic models for probabilistic energy forecasting** \
*Esteban Hernandez, Jonathan Dumas* \
Powertech 2022. [[Paper](https://arxiv.org/abs/2212.02977)] \
6 Dec 2022

**Modeling Temporal Data as Continuous Functions with Process Diffusion** \
*Marin Biloš, Kashif Rasul, Anderson Schneider, Yuriy Nevmyvaka, Stephan Günnemann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.02590)] \
4 Nov 2022

**Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models** \
*Juan Miguel Lopez Alcaraz, Nils Strodthoff* \
TMLR 2022. [[Paper](https://arxiv.org/abs/2208.09399)] [[Github](https://github.com/AI4HealthUOL/SSSD)] \
19 Aug 2022

**CARD: Classification and Regression Diffusion Models** \
*Xizewen Han<sup>1</sup>, Huangjie Zheng<sup>1</sup>, Mingyuan Zhou* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.07275)]  \
15 Jun 2022

**ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models** \
*Tijin Yan, Hongwei Zhang, Tong Zhou, Yufeng Zhan, Yuanqing Xia* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.10121)] [[Github](https://github.com/yantijin/ScoreGradPred)] \
18 Jun 2021

**Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting** \
*Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2101.12072)] [[Github](https://github.com/zalandoresearch/pytorch-ts)] \
2 Feb 2021 




### Imputation

**PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation** \
*Mingzhe Liu<sup>1</sup>, Han Huang<sup>1</sup>, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu* \
ICDE 2023. [[Paper](https://arxiv.org/abs/2302.09746)] [[Github](https://github.com/LMZZML/PriSTI)] \
20 Feb 2023


**Modeling Temporal Data as Continuous Functions with Process Diffusion** \
*Marin Biloš, Kashif Rasul, Anderson Schneider, Yuriy Nevmyvaka, Stephan Günnemann* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.02590)] \
4 Nov 2022

**Diffusion models for missing value imputation in tabular data** \
*Shuhan Zheng, Nontawat Charoenphakdee* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2210.17128)] \
31 Oct 2022

**Neural Markov Controlled SDE: Stochastic Optimization for Continuous-Time Data** \
*Sung Woo Park, Kyungjae Lee, Junseok Kwon* \
ICLR 2022. [[Paper](https://openreview.net/forum?id=7DI6op61AY)] \
29 Sep 2021

**Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models** \
*Juan Miguel Lopez Alcaraz, Nils Strodthoff* \
TMLR 2022. [[Paper](https://arxiv.org/abs/2208.09399)] [[Github](https://github.com/AI4HealthUOL/SSSD)] \
19 Aug 2022

**CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation** \
*Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2107.03502)] [[Github](https://github.com/ermongroup/csdi)]\
7 Jul 2021 

### Miscellany

**Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures** \
*Wei Yuan, Shilong Yuan, Chaoqun Yang, Quoc Viet Hung Nguyen, Hongzhi Yin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08183)] \
14 May 2023

**Domain-Specific Denoising Diffusion Probabilistic Models for Brain Dynamics** \
*Yiqun Duan, Jinzhao Zhou, Zhen Wang, Yu-Cheng Chang, Yu-Kai Wang, Chin-Teng Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04200)] \
7 May 2023

**Conditional Denoising Diffusion for Sequential Recommendation** \
*Yu Wang, Zhiwei Liu, Liangwei Yang, Philip S. Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11433)] \
22 Apr 2023

**Diffusion Recommender Model** \
*Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, Tat-Seng Chua* \
SIGIR 2023. [[Paper](https://arxiv.org/abs/2304.04971)] \
11 Apr 2023

**Sequential Recommendation with Diffusion Models** \
*Hanwen Du, Huanhuan Yuan, Zhen Huang, Pengpeng Zhao, Xiaofang Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04541)] \
10 Apr 2023
 
**DiffuRec: A Diffusion Model for Sequential Recommendation** \
*Zihao Li, Aixin Sun, Chenliang Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.00686)] \
3 Apr 2023

**EEG Synthetic Data Generation Using Probabilistic Diffusion Models** \
*Giulio Tosato, Cesare M. Dalbagno, Francesco Fumagalli* \
Synapsium 2023. [[Paper](https://arxiv.org/abs/2303.06068)] \
6 Mar 2023

**DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal** \
*Huayu Li, Gregory Ditzler, Janet Roveda, Ao Li* \
IEEE JBHI 2023. [[Paper](https://arxiv.org/abs/2208.00542)] \
31 Jul 2022

**Recommendation via Collaborative Diffusion Generative Model** \
*Joojo Walker, Ting Zhong, Fengli Zhang, Qiang Gao, Fan Zhou* \
KSEM 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-10989-8_47)] \
19 Jul 2022

## Graph

### Generation

**A Diffusion Model for Event Skeleton Generation** \
*Fangqi Zhu, Lin Zhang, Jun Gao, Bing Qin, Ruifeng Xu, Haiqin Yang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17458)] [[Github](https://github.com/zhufq00/EventSkeletonGeneration)] \
27 May 2023


**Confidence-Based Feature Imputation for Graphs with Partially Known Features** \
*Daeho Um, Jiwoong Park, Seulki Park, Jin Young Choi* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2305.16618)] [[Github](https://github.com/daehoum1/pcfi)] \
26 May 2023

**Spatio-temporal Diffusion Point Processes** \
*Yuan Yuan, Jingtao Ding, Chenyang Shao, Depeng Jin, Yong Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12403)] [[Github](https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes)] \
21 May 2023

**Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling** \
*Xiaohui Chen, Jiaxing He, Xu Han, Li-Ping Liu* \
ICML 2023. [[Paper](https://arxiv.org/abs/2305.04111)] \
6 May 2023



**A 2D Graph-Based Generative Approach For Exploring Transition States Using Diffusion Model** \
*Seonghwan Kim, Jeheon Woo, Woo Youn Kim* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12233)] \
20 Apr 2023

**Two-stage Denoising Diffusion Model for Source Localization in Graph Inverse Problems** \
*Bosong Huang, Weihao Yu, Ruzhong Xie, Jing Xiao, Jin Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08841)] \
18 Apr 2023

**Diffusion Probabilistic Models for Graph-Structured Prediction** \
*Hyosoon Jang, Sangwoo Mo, Sungsoo Ahn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10506)] \
21 Feb 2023

**GraphGUIDE: interpretable and controllable conditional graph generation with discrete Bernoulli diffusion** \
*Alex M. Tseng, Nathaniel Diamant, Tommaso Biancalani, Gabriele Scalia* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03790)] \
7 Feb 2023

**Graph Generation with Destination-Driven Diffusion Mixture** \
*Jaehyeong Jo<sup>1</sup>, Dongki Kim<sup>1</sup>, Sung Ju Hwang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03596)] \
7 Feb 2023

**DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models** \
*Haomin Wen, Youfang Lin, Yutong Xia, Huaiyu Wan, Roger Zimmermann, Yuxuan Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13629)] \
31 Jan 2023

**DIFFormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion** \
*Qitian Wu, Chenxiao Yang, Wentao Zhao, Yixuan He, David Wipf, Junchi Yan* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2301.09474)] \
23 Jan 2023

**GraphGDP: Generative Diffusion Processes for Permutation Invariant Graph Generation** \
*Han Huang, Leilei Sun, Bowen Du, Yanjie Fu, Weifeng Lv* \
IEEE ICDM 2022. [[Paper](https://arxiv.org/abs/2212.01842)] [[Github](https://github.com/GRAPH-0/GraphGDP)] \
4 Dec 2022

**NVDiff: Graph Generation through the Diffusion of Node Vectors** \
*Cristian Sbrolli, Paolo Cudrano, Matteo Frosi, Matteo Matteucci* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10794)] \
20 Nov 2022

**Fast Graph Generative Model via Spectral Diffusion** \
*Tianze Luo, Zhanfeng Mo, Sinno Jialin Pan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.08892)] \
16 Nov 2022

**Diffusion Models for Graphs Benefit From Discrete State Spaces** \
*Kilian Konstantin Haefeli, Karolis Martinkus, Nathanaël Perraudin, Roger Wattenhofer* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2210.01549)] \
4 Oct 2022

**DiGress: Discrete Denoising diffusion for graph generation** \
*Clement Vignac<sup>1</sup>, Igor Krawczuk<sup>1</sup>, Antoine Siraudin, Bohan Wang, Volkan Cevher, Pascal Frossard* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2209.14734)] \
29 Sep 2022

**Permutation Invariant Graph Generation via Score-Based Generative Modeling** \
*Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, Stefano Ermon* \
AISTATS 2021. [[Paper](https://arxiv.org/abs/2003.00638)] [[Github](https://github.com/ermongroup/GraphScoreMatching)] \
2 Mar 2020

### Molecular and Material Generation

**DiffPack: A Torsional Diffusion Model for Autoregressive Protein Side-Chain Packing** \
*Yangtian Zhan<sup>1</sup>, Zuobai Zhang<sup>1</sup>, Bozitao Zhong, Sanchit Misra, Jian Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01794)] \
1 Jun 2023

**Protein Design with Guided Discrete Diffusion** \
*Nate Gruver<sup>1</sup>, Samuel Stanton<sup>1</sup>, Nathan C. Frey, Tim G. J. Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, Andrew Gordon Wilson* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.20009)] \
31 May 2023


**RINGER: Rapid Conformer Generation for Macrocycles with Sequence-Conditioned Internal Coordinate Diffusion** \
*Colin A. Grambow, Hayley Weir, Nathaniel L. Diamant, Alex M. Tseng, Tommaso Biancalani, Gabriele Scalia, Kangway V. Chuang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19800)] \
30 May 2023

**Trans-Dimensional Generative Modeling via Jump Diffusion Models** \
*Andrew Campbell, William Harvey, Christian Weilbach, Valentin De Bortoli, Tom Rainforth, Arnaud Doucet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16261)] \
25 May 2023

**Learning Joint 2D & 3D Diffusion Models for Complete Molecule Generation** \
*Han Huang, Leilei Sun, Bowen Du, Weifeng Lv* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12347)] [[Github](https://github.com/GRAPH-0/JODO)] \
21 May 2023

**MolDiff: Addressing the Atom-Bond Inconsistency Problem in 3D Molecule Diffusion Generation** \
*Xingang Peng, Jiaqi Guan, Qiang Liu, Jianzhu Ma* \
ICML 2023. [[Paper](https://arxiv.org/abs/2305.07508)] \
11 May 2023

**Coarse-to-Fine: a Hierarchical Diffusion Model for Molecule Generation in 3D** \
*Bo Qiang<sup>1</sup>, Yuxuan Song<sup>1</sup>, Minkai Xu, Jingjing Gong, Bowen Gao, Hao Zhou, Weiying Ma, Yanyan Lan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13266)] \
5 May 2023

**A Latent Diffusion Model for Protein Structure Generation** \
*Cong Fu<sup>1</sup>, Keqiang Yan<sup>1</sup>, Limei Wang, Wing Yee Au, Michael McThrow, Tao Komikado, Koji Maruhashi, Kanji Uchino, Xiaoning Qian, Shuiwang Ji* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.04120)] \
6 May 2023

**Geometric Latent Diffusion Models for 3D Molecule Generation** \
*Minkai Xu, Alexander Powers, Ron Dror, Stefano Ermon, Jure Leskovec* \
ICML 2023. [[Paper](https://arxiv.org/abs/2305.01140)] \
2 May 2023

**MUDiff: Unified Diffusion for Complete Molecule Generation** \
*Chenqing Hua, Sitao Luan, Minkai Xu, Rex Ying, Jie Fu, Stefano Ermon, Doina Precup* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.14621)] \
28 Apr 2023


**Towards Controllable Diffusion Models via Reward-Guided Exploration** \
*Hengtong Zhang, Tingyang Xu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.07132)] \
14 Apr 2023

**Accurate transition state generation with an object-aware equivariant elementary reaction diffusion model** \
*Chenru Duan, Yuanqi Du, Haojun Jia, Heather J. Kulik* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06174)] \
12 Apr 2023

**DiffDock-PP: Rigid Protein-Protein Docking with Diffusion Models** \
*Mohamed Amine Ketata, Cedrik Laue, Ruslan Mammadov, Hannes Stärk, Menghua Wu, Gabriele Corso, Céline Marquet, Regina Barzilay, Tommi S. Jaakkola* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2304.03889)] \
8 Apr 2023

**3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction** \
*Jiaqi Guan, Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, Jianzhu Ma* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2303.03543)] \
6 Mar 2023

**EigenFold: Generative Protein Structure Prediction with Diffusion Models** \
*Bowen Jing, Ezra Erives, Peter Pao-Huang, Gabriele Corso, Bonnie Berger, Tommi Jaakkola* \
ICLR Workshop 2023. [[Paper](https://arxiv.org/abs/2304.02198)] [[Github](https://github.com/bjing2016/EigenFold)] \
5 Apr 2023

**Denoising diffusion algorithm for inverse design of microstructures with fine-tuned nonlinear material properties** \
*Nikolaos N. Vlassis, WaiChing Sun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12881)] \
24 Feb 2023

**Aligned Diffusion Schrödinger Bridges** \
*Vignesh Ram Somnath<sup>1</sup>, Matteo Pariset<sup>1</sup>, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas Krause, Charlotte Bunne* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.11419)] \
22 Feb 2023


**MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation** \
*Clement Vignac, Nagham Osman, Laura Toni, Pascal Frossard* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.09048)] \
17 Feb 2023

**Geometry-Complete Diffusion for 3D Molecule Generation** \
*Alex Morehead, Jianlin Cheng* \
ICML Workshop 2023. [[Paper](https://arxiv.org/abs/2302.04313)] [[Github](https://github.com/BioinfoMachineLearning/bio-diffusion)] \
8 Feb 2023

**SE(3) diffusion model with application to protein backbone generation** \
*Jason Yim<sup>1</sup>, Brian L. Trippe<sup>1</sup>, Valentin De Bortoli<sup>1</sup>, Emile Mathieu<sup>1</sup>, Arnaud Doucet, Regina Barzilay, Tommi Jaakkola* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02277)] \
5 Feb 2023

**Data-Efficient Protein 3D Geometric Pretraining via Refinement of Diffused Protein Structure Decoy** \
*Yufei Huang, Lirong Wu, Haitao Lin, Jiangbin Zheng, Ge Wang, Stan Z. Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10888)] \
5 Feb 2023


**Two for One: Diffusion Models and Force Fields for Coarse-Grained Molecular Dynamics** \
*Marloes Arts, Victor Garcia Satorras, Chin-Wei Huang, Daniel Zuegner, Marco Federici, Cecilia Clementi, Frank Noé, Robert Pinsler, Rianne van den Berg* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.00600)] \
1 Feb 2023

**Generating Novel, Designable, and Diverse Protein Structures by Equivariantly Diffusing Oriented Residue Clouds** \
*Yeqing Lin, Mohammed AlQuraishi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12485)] \
29 Jan 2023

**Physics-Inspired Protein Encoder Pre-Training via Siamese Sequence-Structure Diffusion Trajectory Prediction** \
*Zuobai Zhang<sup>1</sup>, Minghao Xu<sup>1</sup>, Aurélie Lozano, Vijil Chenthamarakshan, Payel Das, Jian Tang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12068)] \
28 Jan 2023

**DiffSDS: A language diffusion model for protein backbone inpainting under geometric conditions and constraints** \
*Zhangyang Gao, Cheng Tan, Stan Z. Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.09642)] \
22 Jan 2023

**DiffBP: Generative Diffusion of 3D Molecules for Target Protein Binding** \
*Haitao Lin<sup>1</sup>, Yufei Huang<sup>1</sup>, Meng Liu, Xuanjing Li, Shuiwang Ji, Stan Z. Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.11214)] \
21 Nov 2022

**ParticleGrid: Enabling Deep Learning using 3D Representation of Materials** \
*Shehtab Zaman, Ethan Ferguson, Cecile Pereira, Denis Akhiyarov, Mauricio Araya-Polo, Kenneth Chiu* \
IEEE eScience 2022. [[Paper](https://arxiv.org/abs/2211.08506)] \
15 Nov 2022

**Structure-based Drug Design with Equivariant Diffusion Models** \
*Arne Schneuing<sup>1</sup>, Yuanqi Du<sup>1</sup>, Charles Harris, Arian Jamasb, Ilia Igashov, Weitao Du, Tom Blundell, Pietro Lió, Carla Gomes, Max Welling, Michael Bronstein, Bruno Correia* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.13695)] \
24 Oct 2022

**Protein Sequence and Structure Co-Design with Equivariant Translation** \
*Chence Shi, Chuanrui Wang, Jiarui Lu, Bozitao Zhong, Jian Tang* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.08761)] \
17 Oct 2022

**Equivariant 3D-Conditional Diffusion Models for Molecular Linker Design** \
*Ilia Igashov, Hannes Stärk, Clément Vignac, Victor Garcia Satorras, Pascal Frossard, Max Welling, Michael Bronstein, Bruno Correia* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.05274)] \
11 Oct 2022

**State-specific protein-ligand complex structure prediction with a multi-scale deep generative model** \
*Zhuoran Qiao, Weili Nie, Arash Vahdat, Thomas F. Miller III, Anima Anandkumar* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2209.15171)] \
30 Sep 2022

**Equivariant Energy-Guided SDE for Inverse Molecular Design** \
*Fan Bao<sup>1</sup>, Min Zhao<sup>1</sup>, Zhongkai Hao, Peiyao Li, Chongxuan Li, Jun Zhu* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2209.15408)] \
30 Sep 2022

**Protein structure generation via folding diffusion** \
*Kevin E. Wu, Kevin K. Yang, Rianne van den Berg, James Y. Zou, Alex X. Lu, Ava P. Amini* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.15611)] \
30 Sep 2022

**MDM: Molecular Diffusion Model for 3D Molecule Generation** \
*Lei Huang, Hengtong Zhang, Tingyang Xu, Ka-Chun Wong* \
AAAI 2023. [[Paper](https://arxiv.org/abs/2209.05710)] \
13 Sep 2022

**Diffusion-based Molecule Generation with Informative Prior Bridges** \
*Lemeng Wu<sup>1</sup>, Chengyue Gong<sup>1</sup>, Xingchao Liu, Mao Ye, Qiang Liu* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2209.00865)] \
2 Sep 2022


**Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models** \
*Shitong Luo<sup>1</sup>, Yufeng Su<sup>1</sup>, Xingang Peng, Sheng Wang, Jian Peng, Jianzhu Ma* \
BioRXiv 2022. [[Paper](https://www.biorxiv.org/content/10.1101/2022.07.10.499510v1)] \
11 Jul 2022

**Data-driven discovery of novel 2D materials by deep generative models** \
*Peder Lyngby, Kristian Sommer Thygesen* \
NPJ 2022. [[Paper](https://arxiv.org/abs/2206.12159)] \
24 Jun 2022

**Score-based Generative Models for Calorimeter Shower Simulation** \
*Vinicius Mikuni, Benjamin Nachman* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.11898)] \
17 Jun 2022

**Diffusion probabilistic modeling of protein backbones in 3D for the motif-scaffolding problem** \
*Brian L. Trippe<sup>1</sup>, Jason Yim<sup>1</sup>, Doug Tischer, Tamara Broderick, David Baker, Regina Barzilay, Tommi Jaakkola* \
CoRR 2022. [[Paper](https://arxiv.org/abs/2206.04119)] \
8 Jun 2022

**Torsional Diffusion for Molecular Conformer Generation** \
*Bowen Jing, Gabriele Corso, Regina Barzilay, Tommi S. Jaakkola* \
ICLR Workshop 2022. [[Paper](https://arxiv.org/abs/2206.01729)] [[Github](https://github.com/gcorso/torsional-diffusion)] \
1 Jun 2022

**Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models** \
*Namrata Anand, Tudor Achim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.15019)] [[Project](https://nanand2.github.io/proteins/)] [[Github](https://github.com/lucidrains/ddpm-ipa-protein-generation)] \
26 May 2022

**A Score-based Geometric Model for Molecular Dynamics Simulations** \
*Fang Wu<sup>1</sup>, Qiang Zhang<sup>1</sup>, Xurui Jin, Yinghui Jiang, Stan Z. Li* \
CoRR 2022. [[Paper](https://arxiv.org/abs/2204.08672)] \
19 Apr 2022

**Equivariant Diffusion for Molecule Generation in 3D** \
*Emiel Hoogeboom<sup>1</sup>, Victor Garcia Satorras<sup>1</sup>, Clément Vignac, Max Welling* \
ICML 2022. [[Paper](https://arxiv.org/abs/2203.17003)] [[Github](https://github.com/ehoogeboom/e3_diffusion_for_molecules)] \
31 Mar 2022

**GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation** \
*Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, Jian Tang* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2203.02923)] [[Github](https://github.com/MinkaiXu/GeoDiff)] \
6 Mar 2022


**Crystal Diffusion Variational Autoencoder for Periodic Material Generation** \
*Tian Xie<sup>1</sup>, Xiang Fu<sup>1</sup>, Octavian-Eugen Ganea<sup>1</sup>, Regina Barzilay, Tommi Jaakkola*\
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2110.06197)] [[Github](https://github.com/txie-93/cdvae)] \
12 Oct 2021


**Predicting Molecular Conformation via Dynamic Graph Score Matching** \
*Shitong Luo, Chence Shi, Minkai Xu, Jian Tang* \
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/hash/a45a1d12ee0fb7f1f872ab91da18f899-Abstract.html)] \
22 May 2021

## Reinforcement Learning

**Professional Basketball Player Behavior Synthesis via Planning with Diffusion** \
*Xiusi Chen, Wei-Yao Wang, Ziniu Hu, Curtis Chou, Lam Hoang, Kun Jin, Mingyan Liu, P. Jeffrey Brantingham, Wei Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.04090)] \
7 Jun 2023

**MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion** \
*Chiyu Max Jiang, Andre Cornman, Cheolho Park, Ben Sapp, Yin Zhou, Dragomir Anguelov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03083)] \
5 Jun 2023

**Extracting Reward Functions from Diffusion Models** \
*Felipe Nuti<sup>1</sup>, Tim Franzmeyer<sup>1</sup>, João F. Henriques* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.01804)] \
1 Jun 2023


**Reconstructing Graph Diffusion History from a Single Snapshot** \
*Ruizhong Qiu, Dingsu Wang, Lei Ying, H. Vincent Poor, Yifang Zhang, Hanghang Tong* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00488)] \
1 Jun 2023

**SafeDiffuser: Safe Planning with Diffusion Probabilistic Models** \
*Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Daniela Rus* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00148)] [Project](https://safediffuser.github.io/safediffuser/)] \
31 May 2023

**Efficient Diffusion Policies for Offline Reinforcement Learning** \
*Bingyi Kang<sup>1</sup>, Xiao Ma<sup>1</sup>, Chao Du, Tianyu Pang, Shuicheng Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.20081)] [[Github](https://github.com/sail-sg/edp)] \
31 May 2023

**MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL** \
*Fei Ni, Jianye Hao, Yao Mu, Yifu Yuan, Yan Zheng, Bin Wang, Zhixuan Liang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19923)] \
31 May 2023


**Generating Behaviorally Diverse Policies with Latent Diffusion Models** \
*Shashank Hegde<sup>1</sup>, Sumeet Batra<sup>1</sup>, K. R. Zentner, Gaurav S. Sukhatme* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18738)] [[Project](https://sites.google.com/view/policydiffusion/home)] \
30 May 2023

**Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning** \
*Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, Xuelong Li* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18459)] \
29 May 2023


**MADiff: Offline Multi-agent Learning with Diffusion Models** \
*Zhengbang Zhu<sup>1</sup>, Minghuan Liu<sup>1</sup>, Liyuan Mao<sup>1</sup>, Bingyi Kang, Minkai Xu, Yong Yu, Stefano Ermon, Weinan Zhang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17330)] \
27 May 2023


**Policy Representation via Diffusion Probability Model for Reinforcement Learning** \
*Long Yang, Zhixiong Huang, Fenghao Lei, Yucun Zhong, Yiming Yang, Cong Fang, Shiting Wen, Binbin Zhou, Zhouchen Lin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13122)] [[Github](https://github.com/BellmanTimeHut/DIPO)] \
22 May 2023

**Diffusion Co-Policy for Synergistic Human-Robot Collaborative Tasks** \
*Eley Ng, Ziang Liu, Monroe Kennedy III* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12171)] \
20 May 2023

**Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning** \
*Cheng Lu<sup>1</sup>, Huayu Chen<sup>1</sup>, Jianfei Chen, Hang Su, Chongxuan Li, Jun Zhu* \
ICML 2023. [[Paper](https://arxiv.org/abs/2304.12824)] [[Github](https://github.com/ChenDRAG/CEP-energy-guided-diffusion)] \
25 Apr 2023


**Goal-Conditioned Imitation Learning using Score-based Diffusion Policies** \
*Moritz Reuss, Maximilian Li, Xiaogang Jia, Rudolf Lioutikov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.02532)] \
5 Apr 2023


**PDPP:Projected Diffusion for Procedure Planning in Instructional Videos** \
*Hanlin Wang, Yilu Wu, Sheng Guo, Limin Wang* \
CVPR 2023. [[Paper](https://arxiv.org/abs/2303.14676)] \
26 Mar 2023

**EDGI: Equivariant Diffusion for Planning with Embodied Agents** \
*Johann Brehmer, Joey Bose, Pim de Haan, Taco Cohen* \
ICLR Workshop 2023. [[Paper](https://arxiv.org/abs/2303.12410)] \
22 Mar 2023

**Multi-Agent Adversarial Training Using Diffusion Learning** \
*Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.01936)] \
3 Mar 2023


**Diffusion Model-Augmented Behavioral Cloning** \
*Hsiang-Chun Wang, Shang-Fu Chen, Shao-Hua Sun* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13335)] \
26 Feb 2023

**To the Noise and Back: Diffusion for Shared Autonomy** \
*Takuma Yoneda, Luzhe Sun, Bradly Stadie, Ge Yang, Matthew Walter* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12244)] [[Github](https://diffusion-for-shared-autonomy.github.io/)] \
23 Feb 2023

**AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners** \
*Zhixuan Liang, Yao Mu, Mingyu Ding, Fei Ni, Masayoshi Tomizuka, Ping Luo* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01877)] \
3 Feb 2023

**Imitating Human Behaviour with Diffusion Models** \
*Tim Pearce, Tabish Rashid, Anssi Kanervisto, Dave Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, Sam Devlin* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2301.10677)] \
25 Jan 2023

**Is Conditional Generative Modeling all you need for Decision-Making?** \
*Anurag Ajay<sup>1</sup>, Yilun Du<sup>1</sup>, Abhi Gupta<sup>1</sup>, Joshua Tenenbaum, Tommi Jaakkola, Pulkit Agrawal* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2211.15657)] \
28 Nov 2022

**LAD: Language Augmented Diffusion for Reinforcement Learning** \
*Edwin Zhang, Yujie Lu, William Wang, Amy Zhang* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2210.15629)] \
27 Oct 2022

**Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning** \
*Zhendong Wang, Jonathan J Hunt, Mingyuan Zhou* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2208.06193)] [[Github](https://github.com/zhendong-wang/diffusion-policies-for-offline-rl)] \
12 Oct 2022

**Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling** \
*Huayu Chen, Cheng Lu, Chengyang Ying, Hang Su, Jun Zhu* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2209.14548)] \
29 Sep 2022


**Planning with Diffusion for Flexible Behavior Synthesis** \
*Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine* \
ICML 2022. [[Paper](https://arxiv.org/abs/2205.09991)] [[Github](https://github.com/jannerm/diffuser)] \
20 May 2022


## Theory

**Variational Gaussian Process Diffusion Processes** \
*Prakhar Verma, Vincent Adam, Arno Solin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02066)] \
3 Jun 2023

**Exploring the Optimal Choice for Generative Processes in Diffusion Models: Ordinary vs Stochastic Differential Equations** \
*Yu Cao, Jingrun Chen, Yixin Luo, Xiang Zhou* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.02063)] \
3 Jun 2023


**On the Equivalence of Consistency-Type Models: Consistency Models, Consistent Diffusion Models, and Fokker-Planck Regularization** \
*Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Naoki Murata, Yuki Mitsufuji, Stefano Ermon* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00367)] \
1 Jun 2023

**Deep Stochastic Mechanics** \
*Elena Orlova<sup>1</sup>, Aleksei Ustimenko<sup>1</sup>, Ruoxi Jiang, Peter Y. Lu, Rebecca Willett* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19685)] \
31 May 2023

**Conditional score-based diffusion models for Bayesian inference in infinite dimensions** \
*Lorenzo Baldassari, Ali Siahkoohi, Josselin Garnier, Knut Solna, Maarten V. de Hoop* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19147)] \
28 May 2023

**Error Bounds for Flow Matching Methods** \
*Joe Benton, George Deligiannidis, Arnaud Doucet* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16860)] \
26 May 2023

**Tree-Based Diffusion Schr\"odinger Bridge with Applications to Wasserstein Barycenters** \
*Maxence Noble, Valentin De Bortoli, Arnaud Doucet, Alain Durmus* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16557)] \
26 May 2023

**Debias Coarsely, Sample Conditionally: Statistical Downscaling through Optimal Transport and Probabilistic Diffusion Models** \
*Zhong Yi Wan, Ricardo Baptista, Yi-fan Chen, John Anderson, Anudhyan Boral, Fei Sha, Leonardo Zepeda-Núñez* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15618)] \
24 May 2023

**Optimal Linear Subspace Search: Learning to Construct Fast and High-Quality Schedulers for Diffusion Models** \
*Zhongjie Duan, Chengyu Wang, Cen Chen, Jun Huang, Weining Qian* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14677)] \
24 May 2023

**SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models** \
*Martin Gonzalez, Nelson Fernandez, Thuy Tran, Elies Gherbi, Hatem Hajri, Nader Masmoudi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14267)] \
23 May 2023

**Improved Convergence of Score-Based Diffusion Models via Prediction-Correction** \
*Francesco Pedrotti, Jan Maas, Marco Mondelli* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14164)] \
23 May 2023

**Expressiveness Remarks for Denoising Diffusion Models and Samplers** \
*Francisco Vargas, Teodora Reu, Anna Kerekes* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09605)] \
16 May 2023

**The Score-Difference Flow for Implicit Generative Modeling** \
*Romann M. Weber* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12906)] \
25 Apr 2023

**Diffusion Models for Constrained Domains** \
*Nic Fishman, Leo Klarner, Valentin De Bortoli, Emile Mathieu, Michael Hutchinson* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05364)] \
11 Apr 2023

**Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling** \
*Stefano Peluchetti* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.00917)]
3 Apr 2023


**Efficient Sampling of Stochastic Differential Equations with Positive Semi-Definite Models** \ 
*Anant Raj, Umut Şimşekli, Alessandro Rudi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17109)] \
30 Mar 2023

**Diffusion Schrödinger Bridge Matching** \
*Yuyang Shi, Valentin De Bortoli, Andrew Campbell, Arnaud Doucet* \
arXiv 2023. [[Paper(https://arxiv.org/abs/2303.16852)] \
29 Mar 2023

**Restoration-Degradation Beyond Linear Diffusions: A Non-Asymptotic Analysis For DDIM-Type Samplers** \
*Sitan Chen, Giannis Daras, Alexandros G. Dimakis* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03384)] \
6 Mar 2023

**Diffusion Models are Minimax Optimal Distribution Estimators** \
*Kazusato Oko, Shunta Akiyama, Taiji Suzuki* \
ICLR Workshop 2023. [[Paper](https://arxiv.org/abs/2303.01861)] \
3 Mar 2023

**Understanding the Diffusion Objective as a Weighted Integral of ELBOs** \
*Diederik P. Kingma, Ruiqi Gao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.00848)] \
1 Mar 2023

**Continuous-Time Functional Diffusion Processes** \
*Giulio Franzese, Simone Rossi, Dario Rossi, Markus Heinonen, Maurizio Filippone, Pietro Michiardi* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.00800)] \
1 Mar 2023

**Denoising Diffusion Samplers** \
*Francisco Vargas, Will Grathwohl, Arnaud Doucet* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2302.13834)] \
27 Feb 2023

**Infinite-Dimensional Diffusion Models for Function Spaces** \
*Jakiw Pidstrigach, Youssef Marzouk, Sebastian Reich, Sven Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10130)] \
20 Feb 2023

**Score-based Diffusion Models in Function Space** \
*Jae Hyun Lim<sup>1</sup>, Nikola B. Kovachki<sup>1</sup>, Ricardo Baptista<sup>1</sup>, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, Anima Anandkumar* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07400)] \
14 Feb 2023

**Score Approximation, Estimation and Distribution Recovery of Diffusion Models on Low-Dimensional Data** \
*Minshuo Chen<sup>1</sup>, Kaixuan Huang<sup>1</sup>, Tuo Zhao, Mengdi Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07194)] \
14 Feb 2023

**Stochastic Modified Flows, Mean-Field Limits and Dynamics of Stochastic Gradient Descent** \
*Benjamin Gess<sup>1</sup>, Sebastian Kassing<sup>1</sup>, Vitalii Konarovskyi* \
JMLR 2023. [[Paper](https://arxiv.org/abs/2302.07125)] \
14 Feb 2023

**Monte Carlo Neural Operator for Learning PDEs via Probabilistic Representation** \
*Rui Zhang, Qi Meng, Rongchan Zhu, Yue Wang, Wenlei Shi, Shihua Zhang, Zhi-Ming Ma, Tie-Yan Liu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05104)] \
10 Feb 2023

**Example-Based Sampling with Diffusion Models** \
*Bastien Doignies, Nicolas Bonneel, David Coeurjolly, Julie Digne, Loïs Paulin, Jean-Claude Iehl, Victor Ostromoukhov* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05116)] \
10 Feb 2023

**Information-Theoretic Diffusion** \
*Xianghao Kong, Rob Brekelmans, Greg Ver Steeg* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2302.03792)] [[Github](https://github.com/gregversteeg/InfoDiffusionSimple)] \
7 Feb 2023

**Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport** \
*Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Kilian Fatras, Guy Wolf, Yoshua Bengio* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.00482)] [[Github](https://github.com/atong01/conditional-flow-matching)] \
1 Feb 2023

**Transport with Support: Data-Conditional Diffusion Bridges** \
*Ella Tamir, Martin Trapp, Arno Solin* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13636)] \
31 Jan 2023

**Understanding and contextualising diffusion models** \
*Stefano Scotta, Alberto Messina* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01394)] \
26 Jan 2023

**On the Mathematics of Diffusion Models** \
*David McAllester* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11108)] \
25 Jan 2023

**Understanding the diffusion models by conditional expectations** \
*Yubin Lu, Zhongjian Wang, Guillaume Bal* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.07882)] \
19 Jan 2023

**Thompson Sampling with Diffusion Generative Prior** \
*Yu-Guan Hsieh, Shiva Prasad Kasiviswanathan, Branislav Kveton, Patrick Blöbaum* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.05182)] \
12 Jan 2023

**Your diffusion model secretly knows the dimension of the data manifold** \
*Georgios Batzolis<sup>1</sup>, Jan Stanczuk<sup>1</sup>, Carola-Bibiane Schönlieb* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.12611)] \
23 Dec 2022

**Score-based Generative Modeling Secretly Minimizes the Wasserstein Distance** \
*Dohyun Kwon, Ying Fan, Kangwook Lee* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2212.06359)] [[Github](https://github.com/UW-Madison-Lee-Lab/score-wasserstein)] \
13 Dec 2022

**Nonlinear controllability and function representation by neural stochastic differential equations** \
*Tanya Veeravalli, Maxim Raginsky* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2212.00896)] \
1 Dec 2022

**Diffusion Generative Models in Infinite Dimensions** \
*Gavin Kerrigan, Justin Ley, Padhraic Smyth* \
AISTATS 2023. [[Paper](https://arxiv.org/abs/2212.00886)] \
1 Dec 2022

**Neural Langevin Dynamics: towards interpretable Neural Stochastic Differential Equations** \
*Simon M. Koop, Mark A. Peletier, Jacobus W. Portegies, Vlado Menkovski* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09537)] \
17 Nov 2022

**Improved Analysis of Score-based Generative Modeling: User-Friendly Bounds under Minimal Smoothness Assumptions** \
*Hongrui Chen, Holden Lee, Jianfeng Lu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01916)] \
3 Nov 2022

**Categorical SDEs with Simplex Diffusion** \
*Pierre H. Richemond, Sander Dieleman, Arnaud Doucet* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.14784)] \
26 Oct 2022

**Diffusion Models for Causal Discovery via Topological Ordering** \
*Pedro Sanchez, Xiao Liu, Alison Q O'Neil, Sotirios A. Tsaftaris* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2210.06201)] [[Github](https://github.com/vios-s/DiffAN)] \
12 Oct 2022

**Regularizing Score-based Models with Score Fokker-Planck Equations** \
*Chieh-Hsin Lai, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki Mitsufuji, Stefano Ermon* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2210.04296)] \
9 Oct 2022

**Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models** \
*Louis Sharrock, Jack Simons, Song Liu, Mark Beaumont* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.04872)] \
10 Oct 2022

**Analyzing Diffusion as Serial Reproduction** \
*Raja Marjieh, Ilia Sucholutsky, Thomas A. Langlois, Nori Jacoby, Thomas L. Griffiths* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14821)] \
29 Sep 2022

**Convergence of score-based generative modeling for general data distributions** \
*Holden Lee, Jianfeng Lu, Yixin Tan* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2209.12381)] \
26 Sep 2022

**Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions** \
*Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, Anru R. Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.11215)] \
22 Sep 2022

**Riemannian Diffusion Models** \
*Chin-Wei Huang, Milad Aghajohari, Avishek Joey Bose, Prakash Panangaden, Aaron Courville* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2208.07949)] \
16 Aug 2022

**Convergence of denoising diffusion models under the manifold hypothesis** \
*Valentin De Bortoli* \
TMLR 2022. [[Paper](https://arxiv.org/abs/2208.05314)] \
10 Aug 2022


**Neural Diffusion Processes** \
*Vincent Dutordoir, Alan Saul, Zoubin Ghahramani, Fergus Simpson* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.03992)] \
8 Jun 2022

**Theory and Algorithms for Diffusion Processes on Riemannian Manifolds** \
*Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, Tommi Jaakkola* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.01729)] [[Github](https://github.com/gcorso/torsional-diffusion)] \
1 Jun 2022

**Riemannian Score-Based Generative Modeling** \
*Valentin De Bortoli<sup>1</sup>, Emile Mathieu<sup>1</sup>, Michael Hutchinson, James Thornton, Yee Whye Teh, Arnaud Doucet* \
NeurIPS 2022. [[Paper](https://arxiv.org/abs/2202.02763)] \
6 Feb 2022

**Interpreting diffusion score matching using normalizing flow** \
*Wenbo Gong<sup>1</sup>, Yingzhen Li<sup>1</sup>* \
ICML Workshop 2021. [[Paper](https://arxiv.org/abs/2107.10072)] \
21 Jul 2021

**A Connection Between Score Matching and Denoising Autoencoders** \
*Pascal Vincent* \
Neural Computation 2011. [[Paper](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)] \
7 Jul 2011

**Bayesian Learning via Stochastic Gradient Langevin Dynamics** \
*Max Welling, Yee Whye Teh* \
ICML 2011. [[Paper](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)] [[Github](https://github.com/JavierAntoran/Bayesian-Neural-Networks#stochastic-gradient-langevin-dynamics-sgld)] \
20 Apr 2011

## Applications

**High-dimensional and Permutation Invariant Anomaly Detection** \
*Vinicius Mikuni, Benjamin Nachman* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03933)] \
6 Jun 2023

**SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting** \
*Lei Chen, Fei Du, Yuan Hu, Fan Wang, Zhibin Wang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03110)] \
5 Jun 2023


**Inverse-design of nonlinear mechanical metamaterials via video denoising diffusion models** \
*Jan-Hendrik Bastek, Dennis M. Kochmann* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19836)] \
31 May 2023

**A Score-Based Model for Learning Neural Wavefunctions** \
*Xuan Zhang, Shenglong Xu, Shuiwang Ji* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16540)] \
25 May 2023

**Dirichlet Diffusion Score Model for Biological Sequence Generation** \
*Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, Jian Zhou* \
ICML 2023 [[Paper](https://arxiv.org/abs/2305.10699)] [[Github](https://github.com/jzhoulab/ddsm)] \
18 May 2023

**GETMusic: Generating Any Music Tracks with a Unified Representation and Diffusion Framework** \
*Ang Lv, Xu Tan, Peiling Lu, Wei Ye, Shikun Zhang, Jiang Bian, Rui Yan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10841)] [[Github](https://github.com/microsoft/muzic)] \
18 May 2023

**End-To-End Latent Variational Diffusion Models for Inverse Problems in High Energy Physics** \
*Alexander Shmakov, Kevin Greif, Michael Fenton, Aishik Ghosh, Pierre Baldi, Daniel Whiteson* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10399)] \
17 May 2023

**Discrete Diffusion Probabilistic Models for Symbolic Music Generation** \
*Matthias Plasser, Silvan Peter, Gerhard Widmer* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09489)] [[Github](https://github.com/plassma/symbolic-music-discrete-diffusion)] \
16 May 2023

**AWFSD: Accelerated Wirtinger Flow with Score-based Diffusion Image Prior for Poisson-Gaussian Holographic Phase Retrieval** \
*Zongyu Li<sup>1</sup>, Jason Hu<sup>1</sup>, Xiaojian Xu, Liyue Shen, Jeffrey A. Fessler* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.07712)] \
12 May 2023

**LatentPINNs: Generative physics-informed neural networks via a latent representation learning** \
*Mohammad H. Taufik, Tariq Alkhalifah* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.07671)] \
11 May 2023

**Latent diffusion models for generative precipitation nowcasting with accurate uncertainty quantification** \
*Jussi Leinonen, Ulrich Hamann, Daniele Nerini, Urs Germann, Gabriele Franch* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.12891)] \
25 Apr 2023

**DiffESM: Conditional Emulation of Earth System Models with Diffusion Models** \
*Seth Bassetti, Brian Hutchinson, Claudia Tebaldi, Ben Kravitz* \
ICLR 2023. [[Paper](https://arxiv.org/abs/2304.11699)] \
23 Apr 2023

**Diffusion Model for GPS Trajectory Generation** \
*Yuanshao Zhu, Yongchao Ye, Xiangyu Zhao, James J.Q. Yu* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11582)] \
23 Apr 2023

**On Accelerating Diffusion-Based Sampling Process via Improved Integration Approximation** \
*Guoqiang Zhang, Niwa Kenta, W. Bastiaan Kleijn* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11328)] \
22 Apr 2023

**iPINNs: Incremental learning for Physics-informed neural networks** \
*Aleksandr Dekhovich, Marcel H.F. Sluiter, David M.J. Tax, Miguel A. Bessa* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.04854)] \
10 Apr 2023


**Denoising Diffusion Probabilistic Models to Predict the Density of Molecular Clouds** \
*Duo Xu, Jonathan C. Tan, Chia-Jung Hsu, Ye Zhu* \
ApJ 2023. [[Paper](https://arxiv.org/abs/2304.01670)] \
4 Apr 2023

**Deep Generative Model and Its Applications in Efficient Wireless Network Management: A Tutorial and Case Study** \
*Yinqiu Liu, Hongyang Du, Dusit Niyato, Jiawen Kang, Zehui Xiong, Dong In Kim, Abbas Jamalipour* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17114)] \
30 Mar 2023

**AI-Generated 6G Internet Design: A Diffusion Model-based Learning Approach** \
*Yudong Huang, Minrui Xu, Xinyuan Zhang, Dusit Niyato, Zehui Xiong, Shuo Wang, Tao Huang* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13869)] \
24 Mar 2023

**Generative AI-aided Optimization for AI-Generated Content (AIGC) Services in Edge Networks** \
*Hongyang Du, Zonghang Li, Dusit Niyato, Jiawen Kang, Zehui Xiong, Huawei Huang, Shiwen Mao* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13052)] [[Github](https://github.com/Lizonghang/AGOD)] \
23 Mar 2023

**Stable Bias: Analyzing Societal Representations in Diffusion Models** \
*Alexandra Sasha Luccioni, Christopher Akiki, Margaret Mitchell, Yacine Jernite* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11408)] \
20 Mar 2023

**PC-JeDi: Diffusion for Particle Cloud Generation in High Energy Physics** \
*Matthew Leigh, Debajyoti Sengupta, Guillaume Quétant, John Andrew Raine, Knut Zoch, Tobias Golling* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.05376)] \
9 Mar 2023

**Generating Initial Conditions for Ensemble Data Assimilation of Large-Eddy Simulations with Latent Diffusion Models** \
*Alex Rybchuk, Malik Hassanaly, Nicholas Hamilton, Paula Doubrawa, Mitchell J. Fulton, Luis A. Martínez-Tossas* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.00836)] \
1 Mar 2023


**ReorientDiff: Diffusion Model based Reorientation for Object Manipulation** \
*Utkarsh A. Mishra, Yongxin Chen* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12700)] [[Project](http://umishra.me/ReorientDiff/)] \
28 Feb 2023

**Interventional and Counterfactual Inference with Diffusion Models** \
*Patrick Chao, Patrick Blöbaum, Shiva Prasad Kasiviswanathan* \
arXiv 2023. [[Paper](https://arxiv.org/abs/2302.00860)] \
2 Feb 2023

**Denoising Diffusion for Sampling SAT Solutions** \
*Karlis Freivalds, Sergejs Kozlovics* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2212.00121)] \
30 Nov 2022


**Generating astronomical spectra from photometry with conditional diffusion models** \
*Lars Doorenbos, Stefano Cavuoti, Giuseppe Longo, Massimo Brescia, Raphael Sznitman, Pablo Márquez-Neila* \
NeurIPS Workshop 2022. [[Paper](https://arxiv.org/abs/2211.05556)] [[Github](https://github.com/LarsDoorenbos/generate-spectra)] \
10 Nov 2022


**Graphically Structured Diffusion Models** \
*Christian Weilbach, William Harvey, Frank Wood* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11633)] \
20 Oct 2022

**Denoising Diffusion Error Correction Codes** \
*Yoni Choukroun, Lior Wolf* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.13533)] \
16 Sep 2022

**Deep Diffusion Models for Robust Channel Estimation** \
*Marius Arvinte, Jonathan I Tamir* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.08177)] [[Github](https://github.com/utcsilab/diffusion-channels)] \
16 Nov 2021

**Diffusion models for Handwriting Generation** \
*Troy Luhman<sup>1</sup>, Eric Luhman<sup>1</sup>* \
arXiv 2020. [[Paper](https://arxiv.org/abs/2011.06704)] [[Github](https://github.com/tcl9876/Diffusion-Handwriting-Generation)] \
13 Nov 2020 
