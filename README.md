[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/hee9joon/Awesome-Diffusion-Models) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

This repository contains a collection of resources and papers on ***Diffusion Models and Score-matching Models***.

## Contents
- [Resources](#resources)
  - [Introductory Posts](#introductory-posts)
  - [Introductory Papers](#introductory-papers)
  - [Introductory Videos](#introductory-videos)
  - [Introductory Lectures](#introductory-lectures)
- [Papers](#papers)
  - [Vision](#vision)
    - [Image Generation](#image-generation)
    - [Segmentation](#segmentation)
    - [Image-to-Image Translation](#image-to-image-translation)
    - [Super Resolution](#super-resolution)
    - [Image Editing](#image-editing)
    - [Text-to-Image](#text-to-image)
    - [Medical Imaging](#medical-imaging)
    - [Video Generation](#video-generation)
    - [Point Cloud](#point-cloud)
  - [Audio](#audio)
    - [Audio Generation](#audio-generation)
    - [Audio Conversion](#audio-conversion)
    - [Audio Enhancement](#audio-enhancement)
    - [Text-to-Speech](#text-to-speech)
  - [Miscellaneous](#miscellaneous)
    - [Adversarial Attack and Defense](#adversarial-attack-and-defense)
    - [Natural Language](#natural-language)
    - [Time-Series](#time-series)
    - [Molecule Generation](#molecule-generation)
    - [Applications](#applications)

# Resources
## Introductory Posts

**Recent rise of diffusion-based models** \
*Maciej Domagała* \
[[Website](https://maciejdomagala.github.io/generative_models/2022/06/06/recent-rise-of-diffusion-based-models.html)] \
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
29 June 2021

**Generative Modeling by Estimating Gradients of the Data Distribution** \
*Yang Song* \
[[Website](https://yang-song.github.io/blog/2021/score/)] \
5 May 2021

## Introductory Papers

**A Connection Between Score Matching and Denoising Autoencoders** \
*Pascal Vincent* \
Neural Computation 2011. [[Paper](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)] \
7 Jul 2011

**Bayesian Learning via Stochastic Gradient Langevin Dynamics** \
*Max Welling, Yee Whye Teh* \
ICML 2011. [[Paper](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)] [[Github](https://github.com/JavierAntoran/Bayesian-Neural-Networks#stochastic-gradient-langevin-dynamics-sgld)] \
20 Apr 2022

## Introductory Videos

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
19 June 2022

**Diffusion Probabilistic Models** \
*Jascha Sohl-Dickstein, MIT 6.S192 - Lecture 22* \
[[Video](https://www.youtube.com/watch?v=XCUlnHP1TNM)] \
19 April 2022

# Papers

## Vision
### Image Generation

**Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling** \
*Hengyuan Ma, Li Zhang, Xiatian Zhu, Jianfeng Feng* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.02196)] \
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
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.09012)] \
17 June 2022

**A Flexible Diffusion Model** \
*Weitao Du, Tao Yang, He Zhang, Yuanqi Du* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.10365)] \
17 Jun 2022

**Lossy Compression with Gaussian Diffusion** \
*Lucas Theis, Tim Salimans, Matthew D. Hoffman, Fabian Mentzer* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.08889)] \
17 Jun 2022

**Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching** \
*Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu* \
ICML 2022. [[Paper](https://arxiv.org/abs/2206.08265)] \
16 Jun 2022

**Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models** \
*Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, Bo Zhang* \
ICML 2022. [[Paper](https://arxiv.org/abs/2206.07309)] [[Github](https://github.com/baofff/Extended-Analytic-DPM)] \
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
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00927)] \
2 Jun 2022

**Elucidating the Design Space of Diffusion-Based Generative Models** \
*Tero Karras, Miika Aittala, Timo Aila, Samuli Laine* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00364)] \
1 Jun 2022

**On Analyzing Generative and Denoising Capabilities of Diffusion-based Deep Generative Models** \
*Kamil Deja, Anna Kuzina, Tomasz Trzciński, Jakub M. Tomczak* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00070)] \
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
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.13699)] \
27 May 2022

**Accelerating Diffusion Models via Early Stop of the Diffusion Process** \
*Zhaoyang Lyu, Xudong XU, Ceyuan Yang, Dahua Lin, Bo Dai* \
ICML 2022. [[Paper](https://arxiv.org/abs/2205.12524)] \
25 May 2022

**On Conditioning the Input Noise for Controlled Image Generation with Diffusion Models** \
*Vedant Singh<sup>1</sup>, Surgan Jandial<sup>1</sup>, Ayush Chopra, Siddharth Ramesh, Balaji Krishnamurthy, Vineeth N. Balasubramanian* \
arxiv 2022. [[Paper](https://arxiv.org/abs/2205.03859)] \
8 May 2022

**Subspace Diffusion Generative Models** \
*Bowen Jing, Gabriele Corso, Renato Berlinghieri, Tommi Jaakkola* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.01490)] [[Github](https://github.com/bjing2016/subspace-diffusion)] \
3 May 2022

**Fast Sampling of Diffusion Models with Exponential Integrator** \
*Qinsheng Zhang, Yongxin Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.13902)] \
29 Apr 2022

**Retrieval-Augmented Diffusion Models** \
*Andreas Blattmann<sup>1</sup>, Robin Rombach<sup>1</sup>, Kaan Oktay, Björn Ommer* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.11824)] \
25 Apr 2022

**Perception Prioritized Training of Diffusion Models** \
*Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, Sungroh Yoon* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.00227)] [[Github](https://github.com/jychoi118/P2-weighting)] \
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

**Dynamic Dual-Output Diffusion Models** \
*Yaniv Benny, Lior Wolf* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.04304)] \
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

**Progressive Distillation for Fast Sampling of Diffusion Models** \
*Tim Salimans, Jonathan Ho* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2202.00512)] \
1 Feb 2022

**Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models** \
*Fan Bao, Chongxuan Li, Jun Zhu, Bo Zhang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.06503)] \
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
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.10741)] \
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
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.07068)] [[Project](https://nv-tlabs.github.io/CLD-SGM/)] \
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
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.15640)] [[Project](https://diff-ae.github.io/)] \
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
NeurIPS Workshop 2021. [[Paper](https://openreview.net/pdf?id=qw8AKxfYbI)] \
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

**SDEdit: Image Synthesis and Editing with Stochastic Differential Equations** \
*Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2108.01073)] [[Project](https://sde-image-editing.github.io/)] [[Github](https://github.com/ermongroup/SDEdit)] \
2 Aug 2021

**Structured Denoising Diffusion Models in Discrete State-Spaces** \
*Jacob Austin<sup>1</sup>, Daniel D. Johnson<sup>1</sup>, Jonathan Ho, Daniel Tarlow, Rianne van den Berg* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.03006)] \
7 Jul 2021 

**Variational Diffusion Models** \
*Diederik P. Kingma<sup>1</sup>, Tim Salimans<sup>1</sup>, Ben Poole, Jonathan Ho* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.00630)] [[Github](https://github.com/revsic/jax-variational-diffwave)] \
1 Jul 2021 

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
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.06819)] [[Project](https://d2c-model.github.io/)] [[Github](https://github.com/d2c-model/d2c-model.github.io)] \
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
ICML Workshop 2021. [[Paper](https://arxiv.org/abs/2106.02808)] [[Github](https://github.com/CW-Huang/sdeflow-light)] \
5 Jun 2021 

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
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.15282)] [[Project](https://cascaded-diffusion.github.io/)] \
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

### Segmentation

**Semantic Image Synthesis via Diffusion Models** \
*Weilun Wang, Jianmin Bao, Wengang Zhou, Dongdong Chen, Dong Chen, Lu Yuan, Houqiang Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.00050)] \
30 Jun 2022

**Diffusion models as plug-and-play priors** \
*Alexandros Graikos, Nikolay Malkin, Nebojsa Jojic, Dimitris Samaras* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.09012)] \
17 June 2022

**Diffusion Models for Implicit Image Segmentation Ensembles** \
*Julia Wolleb<sup>1</sup>, Robin Sandkühler<sup>1</sup>, Florentin Bieder, Philippe Valmaggia, Philippe C. Cattin* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.03145)] \
6 Dec 2021

**Label-Efficient Semantic Segmentation with Diffusion Models** \
*Dmitry Baranchuk, Ivan Rubachev, Andrey Voynov, Valentin Khrulkov, Artem Babenko* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.03126)] [[Github](https://github.com/yandex-research/ddpm-segmentation)] \
6 Dec 2021

**SegDiff: Image Segmentation with Diffusion Probabilistic Models** \
*Tomer Amit, Eliya Nachmani, Tal Shaharbany, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.00390)] \
1 Dec 2021


### Image-to-Image Translation

**Discrete Contrastive Diffusion for Cross-Modal and Conditional Generation** \
*Ye Zhu, Yu Wu, Kyle Olszewski, Jian Ren, Sergey Tulyakov, Yan Yan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.07771)] [[Github](https://github.com/L-YeZhu/CDCD)] \
15 Jun 2022


**SAR Despeckling using a Denoising Diffusion Probabilistic Model** \
*Malsha V. Perera, Nithin Gopalakrishnan Nair, Wele Gedara Chaminda Bandara, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.04514)] \
9 Jun 2022

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
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11793)] \
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

### Super Resolution

**Denoising Diffusion Restoration Models** \
*Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11793)] \
27 Jan 2022

**DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents** \
*Kushagra Pandey, Avideep Mukherjee, Piyush Rai, Abhishek Kumar* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.00308)] [[Github](https://github.com/kpandey008/DiffuseVAE)] \
2 Jan 2022

**High-Resolution Image Synthesis with Latent Diffusion Models** \
*Robin Rombach<sup>1</sup>, Andreas Blattmann<sup>1</sup>, Dominik Lorenz, Patrick Esser, Björn Ommer* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.10752)] [[Github](https://github.com/CompVis/latent-diffusion)] \
20 Dec 2021

**Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction** \
*Hyungjin Chung, Byeongsu Sim, Jong Chul Ye* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.05146)] \
9 Dec 2021

**Conditional Image Generation with Score-Based Diffusion Models** \
*Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, Christian Etmann* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.13606)] \
26 Nov 2021

**S3RP: Self-Supervised Super-Resolution and Prediction for Advection-Diffusion Process** \
*Chulin Wang, Kyongmin Yeo, Xiao Jin, Andres Codas, Levente J. Klein, Bruce Elmegreen* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.04639)] \
8 Nov 2021

**Autoregressive Diffusion Models** \
*Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, Tim Salimans* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2110.02037)] \
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
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.14951)] \
30 Apr 2021

**Image Super-Resolution via Iterative Refinement**  \
*Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.07636)] [[Project](https://iterative-refinement.github.io/)] [[Github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] \
15 Apr 2021

### Image Editing

**Blended Latent Diffusion** \
*Omri Avrahami, Ohad Fried, Dani Lischinski* \
ACM 2022. [[Paper](https://arxiv.org/abs/2206.02779)] [[Project](https://omriavrahami.com/blended-latent-diffusion-page/)] [[Github](https://github.com/omriav/blended-latent-diffusion)] \
6 Jun 2022

**Improving Diffusion Models for Inverse Problems using Manifold Constraints** \
*Hyungjin Chung<sup>1</sup>, Byeongsu Sim<sup>1</sup>, Dohoon Ryu, Jong Chul Ye* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00941)] \
2 Jun 2022


**DiVAE: Photorealistic Images Synthesis with Denoising Diffusion Decoder** \
*Jie Shi<sup>1</sup>, Chenfei Wu<sup>1</sup>, Jian Liang, Xiang Liu, Nan Duan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.00386)] \
1 Jun 2022

**Denoising Diffusion Restoration Models** \
*Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11793)] \
27 Jan 2022

**RePaint: Inpainting using Denoising Diffusion Probabilistic Models** \
*Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.09865)] [[Github](https://github.com/andreas128/RePaint)] \
24 Jan 2022

**High-Resolution Image Synthesis with Latent Diffusion Models** \
*Robin Rombach<sup>1</sup>, Andreas Blattmann<sup>1</sup>, Dominik Lorenz, Patrick Esser, Björn Ommer* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.10752)] [[Github](https://github.com/CompVis/latent-diffusion)] \
20 Dec 2021

**Tackling the Generative Learning Trilemma with Denoising Diffusion GANs** \
*Zhisheng Xiao, Karsten Kreis, Arash Vahdat* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.07804)] [[Project](https://nvlabs.github.io/denoising-diffusion-gan)] \
15 Dec 2021

**Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction** \
*Hyungjin Chung, Byeongsu Sim, Jong Chul Ye* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.05146)] \
9 Dec 2021

**Conditional Image Generation with Score-Based Diffusion Models** \
*Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, Christian Etmann* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.13606)] \
26 Nov 2021

**Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes** \
*Sam Bond-Taylor<sup>1</sup>, Peter Hessey<sup>1</sup>, Hiroshi Sasaki, Toby P. Breckon, Chris G. Willcocks* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.12701)] [[Github](https://github.com/samb-t/unleashing-transformers)] \
24 Nov 2021

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021

**SDEdit: Image Synthesis and Editing with Stochastic Differential Equations**  \
*Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2108.01073)] [[Project](https://sde-image-editing.github.io/)] [[Github](https://github.com/ermongroup/SDEdit)] \
2 Aug 2021

### Text-to-Image

**Discrete Contrastive Diffusion for Cross-Modal and Conditional Generation** \
*Ye Zhu, Yu Wu, Kyle Olszewski, Jian Ren, Sergey Tulyakov, Yan Yan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.07771)] [[Github](https://github.com/L-YeZhu/CDCD)] \
15 Jun 2022

**Blended Latent Diffusion** \
*Omri Avrahami, Ohad Fried, Dani Lischinski* \
ACM 2022. [[Paper](https://arxiv.org/abs/2206.02779)] [[Project](https://omriavrahami.com/blended-latent-diffusion-page/)] [[Github](https://github.com/omriav/blended-latent-diffusion)] \
6 Jun 2022

**Compositional Visual Generation with Composable Diffusion Models** \
*Nan Liu<sup>1</sup>, Shuang Li<sup>1</sup>, Yilun Du<sup>1</sup>, Antonio Torralba, Joshua B. Tenenbaum* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.01714)] [[Project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)] \
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
ACM 2022. [[Paper](https://arxiv.org/abs/2205.15996)] \
31 May 2022

**Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding** \
*Chitwan Saharia<sup>1</sup>, William Chan<sup>1</sup>, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.11487)]  \
23 May 2022

**Retrieval-Augmented Diffusion Models** \
*Andreas Blattmann<sup>1</sup>, Robin Rombach<sup>1</sup>, Kaan Oktay, Björn Ommer* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.11824)] \
25 Apr 2022

**Hierarchical Text-Conditional Image Generation with CLIP Latents** \
*Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.06125)] \
13 Apr 2022


**KNN-Diffusion: Image Generation via Large-Scale Retrieval** \
*Oron Ashual, Shelly Sheynin, Adam Polyak, Uriel Singer, Oran Gafni, Eliya Nachmani, Yaniv Taigman* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.02849)] \
6 Apr 2022

**More Control for Free! Image Synthesis with Semantic Diffusion Guidance** \
*Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, Trevor Darrell* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.05744)] \
10 Dec 2021

**Vector Quantized Diffusion Model for Text-to-Image Synthesis** \
*Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, Baining Guo* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.14822)] [[Github](https://github.com/microsoft/VQ-Diffusion)] \
29 Nov 2021

**Blended Diffusion for Text-driven Editing of Natural Images** \
*Omri Avrahami, Dani Lischinski, Ohad Fried* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2111.14818)] [[Project](https://omriavrahami.com/blended-diffusion-page/)] [[Github](https://github.com/omriav/blended-diffusion)] \
29 Nov 2021

**DiffusionCLIP: Text-guided Image Manipulation Using Diffusion Models** \
*Gwanghyun Kim, Jong Chul Ye* \
CVPR 2022. [[Paper](https://arxiv.org/abs/2110.02711)] \
6 Oct 2021

### Medical Imaging

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
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.13295)] \
27 Jun 2022


**Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models** \
*Walter H. L. Pinaya, Mark S. Graham, Robert Gray, Pedro F Da Costa, Petru-Daniel Tudosiu, Paul Wright, Yee H. Mah, Andrew D. MacKinnon, James T. Teo, Rolf Jager, David Werring, Geraint Rees, Parashkev Nachev, Sebastien Ourselin, M. Jorge Cardos* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.03461)] \
7 Jun 2022

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

**Diffusion Models for Medical Anomaly Detection** \
*Julia Wolleb, Florentin Bieder, Robin Sandkühler, Philippe C. Cattin* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.04306)] \
8 Mar 2022

**Towards performant and reliable undersampled MR reconstruction via diffusion model sampling** \
*Cheng Peng, Pengfei Guo, S. Kevin Zhou, Vishal Patel, Rama Chellappa* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.04292)] \
8 Mar 2022

**Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction** \
*Yutong Xie, Quanzheng Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.03623)] [[Github](https://github.com/Theodore-PKU/MC-DDPM)] \
5 Mar 2022

**MRI Reconstruction via Data Driven Markov Chain with Joint Uncertainty Estimation** \
*Guanxiong Luo, Martin Heide, Martin Uecker* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.01479)] \
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
NeurIPS Workshop 2021. [[Paper](https://arxiv.org/abs/2111.08005)] \
15 Nov 2021

**Score-based diffusion models for accelerated MRI** \
*Hyungjin Chung, Jong chul Ye* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2110.05243)] \
8 Oct 2021


### Video Generation


**Diffusion Models for Video Prediction and Infilling** \
*Tobias Höppe, Arash Mehrjou, Stefan Bauer, Didrik Nielsen, Andrea Dittadi* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.07696)] \
15 Jun 2022

**Flexible Diffusion Modeling of Long Videos** \
*William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, Frank Wood* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.11495)] \
23 May 2022

**Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation** \
*Vikram Voleti<sup>1</sup>, Alexia Jolicoeur-Martineau<sup>1</sup>, Christopher Pal* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.09853)] \
19 May 2022

**Video Diffusion Models** \
*Jonathan Ho<sup>1</sup>, Tim Salimans<sup>1</sup>, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.03458)] \
7 Apr 2022

**Diffusion Probabilistic Modeling for Video Generation** \
*Ruihan Yang, Prakhar Srivastava, Stephan Mandt* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.09481)] \
16 Mar 2022


### Point Cloud

**A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion** \
*Zhaoyang Lyu, Zhifeng Kong, Xudong Xu, Liang Pan, Dahua Lin* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.03530)] \
7 Dec 2021

**Score-Based Point Cloud Denoising** \
*Shitong Luo, Wei Hu*\
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.10981)] [[Github](https://github.com/luost26/score-denoise)] \
23 Jul 2021

**3D Shape Generation and Completion through Point-Voxel Diffusion** \
*Linqi Zhou, Yilun Du, Jiajun Wu* \
ICCV 2021. [[Paper](https://arxiv.org/abs/2104.03670)] [[Project](https://alexzhou907.github.io/pvd)] \
8 Apr 2021

**Diffusion Probabilistic Models for 3D Point Cloud Generation** \
*Shitong Luo, Wei Hu* \
CVPR 2021. [[Paper](https://arxiv.org/abs/2103.01458)] [[Github](https://github.com/luost26/diffusion-point-cloud)] \
2 Mar 2021 

## Audio
### Audio Generation

**Adversarial Audio Synthesis with Complex-valued Polynomial Networks** \
*Yongtao Wu, Grigorios G Chrysos, Volkan Cevher* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.06811)] \
14 Jun 2022

**BinauralGrad: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis** \
*Yichong Leng, Zehua Chen, Junliang Guo, Haohe Liu, Jiawei Chen, Xu Tan, Danilo Mandic, Lei He, Xiang-Yang Li, Tao Qin, Sheng Zhao, Tie-Yan Liu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.14807)] \
30 May 2022

**FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis** \
*Rongjie Huang<sup>1</sup>, Max W. Y. Lam<sup>1</sup>, Jun Wang, Dan Su, Dong Yu, Yi Ren, Zhou Zhao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.09934)] [[Project](https://fastdiff.github.io/)] \
21 Apr 2022

**SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping** \
*Yuma Koizumi, Heiga Zen, Kohei Yatabe, Nanxin Chen, Michiel Bacchiani* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.16749)] \
31 Mar 2022

**BDDM: Bilateral Denoising Diffusion Models for Fast and High-Quality Speech Synthesis** \
*Max W. Y. Lam, Jun Wang, Dan Su, Dong Yu* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2203.13508)] [[Github](https://github.com/tencent-ailab/bddm)] \
25 Mar 2022

**ItôWave: Itô Stochastic Differential Equation Is All You Need For Wave Generation** \
*Shoule Wu<sup>1</sup>, Ziqiang Shi<sup>1</sup>* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.12519)] [[Project](https://wushoule.github.io/ItoAudio/)] \
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
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.00630)] [[Github](https://github.com/revsic/jax-variational-diffwave)] \
1 Jul 2021 

**CRASH: Raw Audio Score-based Generative Modeling for Controllable High-resolution Drum Sound Synthesis** \
*Simon Rouard<sup>1</sup>, Gaëtan Hadjeres<sup>1</sup>* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.07431)] [[Project](https://crash-diffusion.github.io/crash/)] \
14 Jun 2021

**PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Driven Adaptive Prior** \
*Sang-gil Lee, Heeseung Kim, Chaehun Shin, Xu Tan, Chang Liu, Qi Meng, Tao Qin, Wei Chen, Sungroh Yoon, Tie-Yan Liu* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.06406)] [[Project](https://speechresearch.github.io/priorgrad/)] \
11 Jun 2021 

**DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism** \
*Jinglin Liu<sup>1</sup>, Chengxi Li<sup>1</sup>, Yi Ren<sup>1</sup>, Feiyang Chen, Peng Liu, Zhou Zhao* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.02446)] [[Project](https://diffsinger.github.io/)] [[Github](https://github.com/keonlee9420/DiffSinger)] \
6 May 2021

**Symbolic Music Generation with Diffusion Models** \
*Gautam Mittal, Jesse Engel, Curtis Hawthorne, Ian Simon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2103.16091)] [[Code](https://github.com/magenta/symbolic-music-diffusion)] \
30 Mar 2021 

**DiffWave: A Versatile Diffusion Model for Audio Synthesis** \
*Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, Bryan Catanzaro* \
ICLR 2021 [[Paper](https://arxiv.org/abs/2009.09761)] [[Github](https://diffwave-demo.github.io/)] \
21 Sep 2020 

**WaveGrad: Estimating Gradients for Waveform Generation** \
*Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, William Chan*\
ICLR 2021. [[Paper](https://arxiv.org/abs/2009.00713)] [[Project](https://wavegrad.github.io/)] [[Github](https://github.com/ivanvovk/WaveGrad)] \
2 Sep 2020 


### Audio Conversion

**DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion**  \
*Songxiang Liu<sup>1</sup>, Yuewen Cao<sup>1</sup>, Dan Su, Helen Meng* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.13871)] [[Github](https://github.com/liusongxiang/diffsvc)] \
28 May 2021


### Audio Enhancement

**NU-Wave 2: A General Neural Audio Upsampling Model for Various Sampling Rates** \
*Seungu Han, Junhyeok Lee* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.08545)] [[Project](https://mindslab-ai.github.io/nuwave2/)] \
17 Jun 2022

**Universal Speech Enhancement with Score-based Diffusion** \
*Joan Serrà, Santiago Pascual, Jordi Pons, R. Oguz Araz, Davide Scaini* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.03065)] \
7 Jun 2022

**Conditional Diffusion Probabilistic Model for Speech Enhancement** \
*Yen-Ju Lu, Zhong-Qiu Wang, Shinji Watanabe, Alexander Richard, Cheng Yu, Yu Tsao* \
IEEE 2022. [[Paper](https://arxiv.org/abs/2202.05256)] \
10 Feb 2022


**A Study on Speech Enhancement Based on Diffusion Probabilistic Model** \
*Yen-Ju Lu<sup>1</sup>, Yu Tsao<sup>1</sup>, Shinji Watanabe* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.11876)] \
25 Jul 2021

**Restoring degraded speech via a modified diffusion model** \
*Jianwei Zhang, Suren Jayasuriya, Visar Berisha* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.11347)] \
22 Apr 2021

**NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling**  \
*Junhyeok Lee, Seungu Han* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.02321)] [[Project](https://mindslab-ai.github.io/nuwave/)] [[Github](https://github.com/mindslab-ai/nuwave)] \
6 Apr 2021


### Text-to-Speech

**Guided-TTS 2: A Diffusion Model for High-quality Adaptive Text-to-Speech with Untranscribed Data** \
*Sungwon Kim<sup>1</sup>, Heeseung Kim<sup>1</sup>, Sungroh Yoon* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.15370)] \
30 May 2022

**InferGrad: Improving Diffusion Models for Vocoder by Considering Inference in Training** \
*Zehua Chen, Xu Tan, Ke Wang, Shifeng Pan, Danilo Mandic, Lei He, Sheng Zhao* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.03751)] \
8 Feb 2022

**DiffGAN-TTS: High-Fidelity and Efficient Text-to-Speech with Denoising Diffusion GANs** \
*Songxiang Liu, Dan Su, Dong Yu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11972)] \
28 Jan 2022

**Guided-TTS:Text-to-Speech with Untranscribed Speech** \
*Heeseung Kim, Sungwon Kim, Sungroh Yoon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.11755)] \
32 Nov 2021

**EdiTTS: Score-based Editing for Controllable Text-to-Speech** \
*Jaesung Tae<sup>1</sup>, Hyeongju Kim<sup>1</sup>, Taesu Kim* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2110.02584)] \
6 Oct 2021

**WaveGrad 2: Iterative Refinement for Text-to-Speech Synthesis** \
*Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, Najim Dehak, William Chan* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.09660)] [[Project](https://mindslab-ai.github.io/wavegrad2/)] [[Github](https://github.com/keonlee9420/WaveGrad2)] [[Github2](https://github.com/mindslab-ai/wavegrad2)] \
17 Jun 2021 

**Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech** \
*Vadim Popov<sup>1</sup>, Ivan Vovk<sup>1</sup>, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov* \
ICML 2021. [[Paper](https://arxiv.org/abs/2105.06337)] [[Project](https://grad-tts.github.io/)] [[Github](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)] \
13 May 2021

**DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism** \
*Jinglin Liu<sup>1</sup>, Chengxi Li<sup>1</sup>, Yi Ren<sup>1</sup>, Feiyang Chen, Peng Liu, Zhou Zhao* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.02446)] [[Project](https://diffsinger.github.io/)] [[Github](https://github.com/keonlee9420/DiffSinger)] \
6 May 2021

**Diff-TTS: A Denoising Diffusion Model for Text-to-Speech**  \
*Myeonghun Jeong, Hyeongju Kim, Sung Jun Cheon, Byoung Jin Choi, Nam Soo Kim* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.01409)] \
3 Apr 2021

## Miscellaneous

### Adversarial Attack and Defense

**Back to the Source: Diffusion-Driven Test-Time Adaptation** \
*Jin Gao<sup>1</sup>, Jialing Zhang<sup>1</sup>, Xihui Liu, Trevor Darrell, Evan Shelhamer, Dequan Wang* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.03442)] \
7 Jul 2022

**Guided Diffusion Model for Adversarial Purification from Random Noise** \
*Quanlin Wu, Hang Ye, Yuntian Gu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.10875)] \
17 Jun 2022

**(Certified!!) Adversarial Robustness for Free!** \
*Nicholas Carlini, Florian Tramer, Krishnamurthy (Dj)Dvijotham, J. Zico Kolter* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.10550)] \
21 Jun 2022

**Guided Diffusion Model for Adversarial Purification** \
*Jinyi Wang<sup>1</sup>, Zhaoyang Lyu<sup>1</sup>, Dahua Lin, Bo Dai, Hongfei Fu* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.14969)] \
30 May 2022

**Diffusion Models for Adversarial Purification** \
*Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar* \
ICML 2022. [[Paper](https://arxiv.org/abs/2205.07460)] [[Project](https://diffpure.github.io/)] \
16 May 2022

**TFDPM: Attack detection for cyber-physical systems with diffusion probabilistic models** \
*Tijin Yan, Tong Zhou, Yufeng Zhan, Yuanqing Xia* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2112.10774)] \
20 Dec 2021

**Adversarial purification with Score-based generative models** \
*Jongmin Yoon, Sung Ju Hwang, Juho Lee* \
ICML 2021. [[Paper](https://arxiv.org/abs/2106.06041)] [[Github](https://github.com/jmyoon1/adp)] \
11 Jun 2021


### Natural Language

**Latent Diffusion Energy-Based Model for Interpretable Text Modeling** \
*Peiyu Yu, Sirui Xie, Xiaojian Ma, Baoxiong Jia, Bo Pang, Ruigi Gao, Yixin Zhu, Song-Chun Zhu, Ying Nian Wu* \
ICML 2022. [[Paper](https://arxiv.org/abs/2206.05895)] \
13 Jun 2022


**Diffusion-LM Improves Controllable Text Generation** \
*Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, Tatsunori B. Hashimoto* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.14217)] \
27 May 2022

**Zero-Shot Translation using Diffusion Models** \
*Eliya Nachmani<sup>1</sup>, Shaked Dovrat<sup>1</sup>* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.01471)] \
2 Nov 2021

### Time-Series

**CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation** \
*Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon* \
NeurIPS 2021. [[Paper](https://arxiv.org/abs/2107.03502)] [[Github](https://github.com/ermongroup/csdi)]\
7 Jul 2021 

**ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models** \
*Tijin Yan, Hongwei Zhang, Tong Zhou, Yufeng Zhan, Yuanqing Xia* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.10121)] [[Github](https://github.com/yantijin/ScoreGradPred)] \
18 Jun 2021

**Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting** \
*Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2101.12072)] [[Github](https://github.com/zalandoresearch/pytorch-ts)] \
2 Feb 2021 

### Molecule Generation

**Data-driven discovery of novel 2D materials by deep generative models** \
*Peder Lyngby, Kristian Sommer Thygesen* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.12159)] \
24 Jun 2022

**Score-based Generative Models for Calorimeter Shower Simulation** \
*Vinicius Mikuni, Benjamin Nachman* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.11898)] \
17 Jun 2022

**Diffusion probabilistic modeling of protein backbones in 3D for the motif-scaffolding problem** \
*Brian L. Trippe<sup>1</sup>, Jason Yim<sup>1</sup>, Doug Tischer, Tamara Broderick, David Baker, Regina Barzilay, Tommi Jaakkola* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.04119)] \
8 Jun 2022

**Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models** \
*Namrata Anand, Tudor Achim* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.15019)] [[Project](https://nanand2.github.io/proteins/)] \
26 May 2022

**A Score-based Geometric Model for Molecular Dynamics Simulations** \
*Fang Wu<sup>1</sup>, Qiang Zhang<sup>1</sup>, Xurui Jin, Yinghui Jiang, Stan Z. Li* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.08672)] \
19 Apr 2022

**Equivariant Diffusion for Molecule Generation in 3D** \
*Emiel Hoogeboom<sup>1</sup>, Victor Garcia Satorras<sup>1</sup>, Clément Vignac, Max Welling* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.17003)] \
31 Mar 2022

**GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation** \
*Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, Jian Tang* \
ICLR 2022. [[Paper](https://arxiv.org/abs/2203.02923)] \
6 Mar 2022

### Applications

**Remote Sensing Change Detection (Segmentation) using Denoising Diffusion Probabilistic Models** \
*Wele Gedara Chaminda Bandara, Nithin Gopalakrishnan Nair, Vishal M. Patel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.11892)] [[Github](https://github.com/wgcban/ddpm-cd)] \
23 Jun 2022

**Discrete Contrastive Diffusion for Cross-Modal and Conditional Generation** \
*Ye Zhu, Yu Wu, Kyle Olszewski, Jian Ren, Sergey Tulyakov, Yan Yan* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.07771)] [[Github](https://github.com/L-YeZhu/CDCD)] \
15 Jun 2022

**CARD: Classification and Regression Diffusion Models** \
*Xizewen Han<sup>1</sup>, Huangjie Zheng<sup>1</sup>, Mingyuan Zhou* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.07275)] \
15 Jun 2022

**Multi-instrument Music Synthesis with Spectrogram Diffusion** \
*Curtis Hawthorne, Ian Simon, Adam Roberts, Neil Zeghidour, Josh Gardner, Ethan Manilow, Jesse Engel* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.05408)] \
11 Jun 2022

**Neural Diffusion Processes** \
*Vincent Dutordoir, Alan Saul, Zoubin Ghahramani, Fergus Simpson* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.03992)] \
8 Jun 2022

**Theory and Algorithms for Diffusion Processes on Riemannian Manifolds** \
*Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, Tommi Jaakkola* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2206.01729)] [[Github](https://github.com/gcorso/torsional-diffusion)] \
1 Jun 2022

**Planning with Diffusion for Flexible Behavior Synthesis** \
*Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine* \
arxiv 2022. [[Paper](https://arxiv.org/abs/2205.09991)] \
20 May 2022

**Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion** \
*Tianpei Gu<sup>1</sup>, Guangyi Chen<sup>1</sup>, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou, Jiwen Lu*\
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.13777)] \
25 Mar 2022

**Riemannian Score-Based Generative Modeling** \
*Valentin De Bortoli<sup>1</sup>, Emile Mathieu<sup>1</sup>, Michael Hutchinson, James Thornton, Yee Whye Teh, Arnaud Doucet* \
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.02763)] \
6 Feb 2022

**Deep Diffusion Models for Robust Channel Estimation** \
*Marius Arvinte, Jonathan I Tamir* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.08177)] [[Github](https://github.com/utcsilab/diffusion-channels)] \
16 Nov 2021

**Deep diffusion-based forecasting of COVID-19 by incorporating network-level mobility information** \
*Padmaksha Roy, Shailik Sarkar, Subhodip Biswas, Fanglan Chen, Zhiqian Chen, Naren Ramakrishnan, Chang-Tien Lu* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2111.05199)] \
9 Nov 2021

**Realistic galaxy image simulation via score-based generative models** \
*Michael J. Smith (Hertfordshire), James E. Geach, Ryan A. Jackson, Nikhil Arora, Connor Stone, Stéphane Courteau* \
MNRAS 2022. [[Paper](https://arxiv.org/abs/2111.01713)] \
2 Nov 2021

**Crystal Diffusion Variational Autoencoder for Periodic Material Generation** \
*Tian Xie<sup>1</sup>, Xiang Fu<sup>1</sup>, Octavian-Eugen Ganea<sup>1</sup>, Regina Barzilay, Tommi Jaakkola*\
arXiv 2021. [[Paper](https://arxiv.org/abs/2110.06197)] \
12 Oct 2021

**Diffusion models for Handwriting Generation** \
*Troy Luhman<sup>1</sup>, Eric Luhman<sup>1</sup>* \
arXiv 2020. [[Paper](https://arxiv.org/abs/2011.06704)] [[Github](https://github.com/tcl9876/Diffusion-Handwriting-Generation)] \
13 Nov 2020 

**Permutation Invariant Graph Generation via Score-Based Generative Modeling** \
*Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, Stefano Ermon* \
AISTATS 2021. [[Paper](https://arxiv.org/abs/2003.00638)] [[Github](https://github.com/ermongroup/GraphScoreMatching)] \
2 Mar 2020
