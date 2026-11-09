# C. Experimental Setup

This section details the experimental setup used in Section 5. We first describe our data collection setup for supervised fine-tuning (Appendix C.1). Then, we present the training pipeline (Appendix C.2). Finally, we describe our evaluation metrics and the statistical tools used for generating plots (Appendix C.3).

# C.1. Data

SFT data collection. As illustrated in Figures 11 to 14, GeneralPoints and V-IRL environments naturally align with prompt-response dialogue structures. We create training samples by pairing each system prompt with its corresponding expert response. All SFT experiments in the main body use optimal single-turn prompt-response pairs, without any verification or revision steps.

SFT on sub-optimal trajectories To examine how more diverse SFT data affects the out-of-distribution performance of SFT, we conduct an ablation study on GP-L using sub-optimal trajectories as training data. Unlike expert prompt-response pairs, these sub-optimal trajectories include errors and verification messages in their prompts. This format aligns with evaluation scenarios where multiple verification iterations are allowed, similar to the data being used for the downstream RL training. In Figure 15, we observe that SFT still merely memorizes the training data with degraded out-of-distribution performance. This evidence suggests that memorization occurs due to the fundamental nature of SFT training rather than the SFT data.

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-18_468_606_1616_296.jpg)

**Figure 15: SFT experiments on GP-L with suboptimal trajectories. Similar to results in Figure 5, SFT overfits the training data even we increase the trajectory diversity.**

# C.2. Training Pipeline

As illustrated in Section 5, we follow the training pipeline by RL4VLM (Zhai et al., 2024a), where we first initialize the model with SFT, then separately scale up the compute for SFT and RL (Schulman et al., 2017), starting from this initialized model. For all experiments of SFT and RL in the main body, we tune all components using a shared learning rate per experiment. All training experiments are conducted on an 8 H800 machine (80GB).

# C.3. Evaluation Metric

Per-step accuracy. We report the per-step accuracy for V-IRL-VL task in Figures 5 and 6. An individual step is considered correct when the model's chosen action matches the expert trajectory at that position. Note that intermediate verification steps are counted as independent samples here.

Success rate. We report the success rate (\%) of GP-L, GP-VL, V-IRL-L and V-IRL-VL in Figures 5 and 6. In the GeneralPoints task, success is defined as succeeding at least once during the inference time verification. In the V-IRL task, a sample is recorded as success when the model takes correct action at each movable point on the route.

Computation estimation. We estimate the FLOPs for training $X$ following the similar manner of (Snell et al., 2024; Hoffmann et al., 2023), where $X_{\text {train }}=6 N D_{\text {train }}$ and $X_{\text {inference }}=2 N D_{\text {inference }}$. Here, $N$ represents the model parameters and $D_{\text {train }}$ represents the number of tokens during training. Suppose our SFT and RL experients starts from a checkpoint trained on $D_{\text {init }}$ tokens, we can estimate the training computation of SFT and RL via the following equations:
$$
\begin{aligned}
X_{S F T} & =6 N\left(D_{\text {init }}+D_{S F T}\right) \\
X_{R L} & =6 N\left(D_{\text {init }}+D_{R L}\right)+2 N D_{\text {buffer }}
\end{aligned}
$$
Note that the used on-policy RL algorithm PPO (Schulman et al., 2017) contains iterative stages of replay buffer collection and optimization, hence requiring additional inference computation. For simplicity, we approximate the term via:
$$
\begin{aligned}
D_{\text {buffer }} & \approx \frac{E \bar{d}_{i} \bar{d}_{o}}{D_{R L}} \cdot D_{R L} \\
& =\lambda D_{R L}
\end{aligned}
$$
where $E \in \mathbb{N}$ denotes the number of auto-regressive generation processes, $\bar{d}_{i}, \bar{d}_{o}$ denote average input tokens and output tokens. We estimate the $\lambda$ for GeneralPoints and V-IRL as 6 and 5.1 respectively after calculation.

Line smoothing and error bar. All line plots in our paper adopt Savitzky-Golay filter with polynomial order 3 as smoothing function. We assume each evaluated data point

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-19_489_1432_233_319.jpg)

**Figure 16: Ablation studies on GeneralPoints-VL SFT. We ablate the learning rate and report the in-distribution episode success rate (\%) of all experiments. None of the experiments shows an increasing trend beyond 30\% success rate.**

follows a binomial distribution and approximate the standard error using \(\sqrt{\frac{P(1-P)}{N}}\), where $P$ is the demical success rate and $N$ is the number of samples.

