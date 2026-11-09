# 5. Results

In this section, we present experiments that investigate the generalization abilities induced by post-training with RL and SFT. We adopt Llama-3.2-Vision-11B (Dubey et al., 2024) as the backbone model. Following the standard pipelines of RLHF (Ouyang et al., 2022) and RL4VLM (Zhai et al., 2024a), we initialize the model with SFT before running RL. We specifically study the follow- ing questions. Section 5.1: how does SFT or RL affect the model's generalization to different rules? Section 5.2: when the model contains a visual component, how does RL/SFT affect its generalization to different visual variants? Section 5.3: how does RL/SFT affect visual recognition capability in a VLM? Section 5.4: what role does SFT play in RL training? Section 5.5: how does the number of verification iterations affect generalization?

## 5.1. Generalization across Rules

We evaluate the performance of different post-training methods on GeneralPoints and V-IRL, each of which has a pure language (-L) and a vision-language (-VL) variant, and each encompassing rule variations. For each task, we separately scale the training compute for RL and SFT on a single rule. We consider the results on the trained rule as in-distribution (ID) performance, whereas results on the unseen rules measures out-of-distribution (OOD) generalization. In GeneralPoints, the ID case treats all 'J', 'Q', 'K' as 10, and the OOD cases interprets them as 11, 12, and 13. As for V-IRL, the ID case adopts the absolute orienta-

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-06_810_1527_240_274.jpg)

**Figure 5: Success rate (\%) - GFLOPs trendlines for RL and SFT on GeneralPoints and V-IRL. The top row shows in-distribution performance, while the bottom row shows out-of-distribution performance. Results are presented for both pure language (-L) and vision-language (-VL) variants of each task. For GeneralPoints, we report the episode success rate, while for V-IRL, we report per-step accuracy with overall success rate in Figures 1 and 18. Detailed evaluation setups (and curve smoothing) are provided in Appendix C.3.**

tion coordinate system and the OOD case uses the relative orientation action space. Other details and additional experimental setup can be found in Appendix C.

RL generalizes, SFT memorizes. As illustrated in Figure 5, RL consistently improves OOD performance on all tasks, including both unimodal (LLM) and multimodal (VLM). Specifically, Figure 6 demonstrates that RL achieves an increase of +3.5\% on GP-L (11.5\% → $15.0 \%$ ) and $+\mathbf{1 1 . 0 \%}$ on V-IRL-L \((80.8 \% \rightarrow 91.8 \%)\). Even with the additional challenge of visual recognition in the VLM, RL maintains consistent performance improvements of \(\mathbf{+ 3 . 0 \%}(11.2 \% \rightarrow 14.2 \%)\) on GP-VL and $\mathbf{+ 9 . 3 \%}$ (35.7\% → 45.0\%) on V-IRL-VL, respectively. In contrast, SFT consistently exhibits performance degradation across all OOD evaluations on all tasks: -8.1\% on GP-L ( \(11.5 \% \rightarrow 3.4 \%),-\mathbf{7 9 . 5 \%}\) on V-IRL-L \((80.8 \% \rightarrow 1.3 \%)\), -5.6\% (11.2\% → 5.6\%) on GP-VL, and -33.2\% (35.7\% → 2.5\%) on V-IRL-VL.

## 5.2. Generalization in Visual Out-of-Distribution Tasks

Section 5.1 demonstrates that RL yields generalization across rule variations, whereas SFT exhibits the opposite trend. Since VLMs also incorporate a visual modality, we next study the effects of visual variation in OOD generalization. For GeneralPoints, we train the VLM using the black suits ( ) and test out-of-distribution performance on the red suits ( $\boldsymbol{\psi}$, ). For V-IRL, we train the model on routes collected in New York City and evaluate it on the original V-IRL VLN mini benchmark (Yang et al., 2024a) containing routes from various cities worldwide (see Appendix B. 1 for details). Note that the rules remain consistent across experiments in this section.

RL generalizes in visual OOD tasks. As shown in Figure 7, we observe that RL still generalizes in visual OOD tasks, while SFT continues to suffer. Specifically, in GP-VL and VIRL-VL, RL achieves performance improvements of \(\mathbf{+ 1 7 . 6 \%}(23.6 \% \rightarrow 41.2 \%),+\mathbf{6 1 . 1 \%}(16.7 \% \rightarrow\) 77.8\%), whereas SFT suffers from performance decreases of \(\mathbf{- 9 . 9 \%}(23.6 \% \rightarrow 13.7 \%)\) and \(\mathbf{- 5 . 6 \%}(16.7 \% \rightarrow 11.1 \%)\). As a byproduct of this visual OOD study, we also show that our multi-turn RL formulation improves the state-ofthe-art results (see Table 5 of Yang et al. (2024a)) on the V-IRL mini benchmark by \(\mathbf{+ 3 3 . 8 \%}(44.0 \% \rightarrow 77.8 \%)\). Notably, unlike the previous state-of-the-art approach reported in V-IRL, which relies on a two stage VLM-LLM collaboration technique and tailored prompt engineering on closed-sourced model (OpenAI, 2023a), our end-to-end RL approach enables an open-sourced model (Dubey et al., 2024) to reach superior performance.

## 5.3. RL Improves Visual Capabilities

Building upon the above observation that VLMs trained with RL generalize to visual OOD tasks (Section 5.2), we

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-07_394_1516_229_276.jpg)

**Figure 6: Comparison of out-of-distribution performance under rule variants. We report the success rate for GeneralPoints and per-step-accuracy for V-IRL. For each subplot, RL and SFT are trained with equal computation, and their shared initial checkpoint (marked as Init) is set as baseline. Detailed setups are provided in Appendix C.3.**

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-07_486_1430_776_314.jpg)

**Figure 7: Comparison of out-of-distribution performance under visual variants. Similar to Figures 5 and 6, we present both the performance dynamics (shown as lines) and final performance (shown as bars) for visual out-of-distribution evaluations. The previous state-of-the-art on V-IRL VLN mini benchmark (Yang et al., 2024a) is marked in orange. Detailed evaluation setups (and curve smoothing) are provided in Appendix C.3.**

consider a natural follow-up question: How does RL affect VLMs' visual capabilities? To study this question, we conducted additional ablation studies in the GP-VL environment to investigate the OOD performance of RL and SFT, along with the model's visual recognition accuracy, in terms of recognizing the 4 cards from the input image. In particular, we study how scaling post-training compute via RL/SFT both affects generalization in rule-based OOD (Figure 8 left), and visual recognition accuracy and visual OOD (Figure 8 right).

Scaling RL improves visual recognition accuracy in VLM training. As shown in Figure 8, we observe that the VLM's visual recognition accuracy largely affects the overall performance, which was similarly observed in Zhong et al. (2024). In addition, scaling up RL compute also improves visual recognition accuracy, as a byproduct of its generalization capability, while scaling SFT deteriorates both visual recognition accuracy and overall performance. Additional experimental results are provided in Figures 16 and 17 of Appendix D.1.

## 5.4. The Role of SFT for RL Training

Despite the superiority of RL in generalizing the model's reasoning and visual capabilities, as discussed previously, the experimental pipeline still instantiates RL after SFT. In this subsection, we focus on another key question: Is SFT necessary for RL training? To answer this question, we conduct additional experiments that directly apply endto-end RL to post-train the base model Llama3.2 using GeneralPoints in the purely language case (Figure 9).

SFT is necessary for RL training when the backbone model does not follow instructions. Figure 9 shows that without SFT, all end-to-end RL runs fail to improve. More specifically, we observe that without SFT, the base model suffers from poor instruction following capability. A detailed failure case is provided in Figure 20 (in Appendix D.3), revealing that the base Llama-3.2-Vision-11B model tends to generate long, tangential, and unstructured responses. This issue makes it impossible to retrieve taskrelated information and rewards for RL training. Note that due to the difference in backbone model, our results do not contradict with DeepSeekAI et al. (2025), which suggests that SFT is unnecessary for downstream RL training.

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-08_500_1518_233_276.jpg)

**Figure 8: Recognition vs. success rate for RL and SFT under different variants in GP-VL. We report both indistribution (red) and OOD (blue) performance of recognition (y-axis) and episode success rate (x-axis). We denote the training compute of each data point via transparency (color bar) while connected (★-○) pairs are evaluated using same checkpoints. As scaling up post-training compute, RL improves both recognition and overall accuracy, while SFT shows opposite effect.**

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-08_362_640_1027_278.jpg)

**Figure 9: RL experiments on GP-L without SFT initialization. All trials fail due to poor instruction following capability of the base model.**

## 5.5. Role of Verification Iterations

Verification serves as another crucial component in our multi-step training and evaluation pipeline (see Figures 2 and 3). To validate its necessity and better understand its effect, we conduct RL experiments with different verification iterations \{1, 3, 5, 10\} using GP-L (Figure 10).

Scaling up verification improves generalization. In Figure 10, we observe that RL generalizes better with more verification steps. More specifically, under the same computational budget across all experiments, we observe improvements of + 2.15\% (3 steps), +2.99\% (5 steps), +5.99\% (10 steps). In contrast, in the case with one verification step, we only observe a marginal improvement of $\mathbf{+ 0 . 4 8} \boldsymbol{\%}$ in OOD performance improvement.

