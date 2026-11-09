# 5 Empirical Experiments

We evaluate the downstream task performance of LoRA on RoBERTa (Liu et al., 2019), DeBERTa (He et al., 2021), and GPT-2 (Radford et al., b), before scaling up to GPT-3 175B (Brown et al., 2020). Our experiments cover a wide range of tasks, from natural language understanding (NLU) to generation (NLG). Specifically, we evaluate on the GLUE (Wang et al., 2019) benchmark for RoBERTa and DeBERTa. We follow the setup of Li \& Liang (2021) on GPT-2 for a direct comparison and add WikiSQL (Zhong et al., 2017) (NL to SQL queries) and SAMSum (Gliwa et al., 2019) (conversation summarization) for large-scale experiments on GPT-3. See Appendix C for more details on the datasets we use. We use NVIDIA Tesla V100 for all experiments.

## 5.1 Baselines

To compare with other baselines broadly, we replicate the setups used by prior work and reuse their reported numbers whenever possible. This, however, means that some baselines might only appear in certain experiments.

Fine-Tuning (FT) is a common approach for adaptation. During fine-tuning, the model is initialized to the pre-trained weights and biases, and all model parameters undergo gradient updates.A simple variant is to update only some layers while freezing others. We include one such baseline reported in prior work (Li \& Liang, 2021) on GPT-2, which adapts just the last two layers ( $\mathbf{F T}^{\text {Top2 }}$ ).

\footnotetext{
${ }^{4}$ We still need the 350GB model during deployment; however, storing 100 adapted models only requires $350 \mathrm{~GB}+35 \mathrm{MB} * 100 \approx 354 \mathrm{~GB}$ as opposed to $100 * 350 \mathrm{~GB} \approx 35 \mathrm{~TB}$.
${ }^{5}$ For GPT-3 175B, the training throughput for full fine-tuning is 32.5 tokens/s per V100 GPU; with the same number of weight shards for model parallelism, the throughput is 43.1 tokens/s per V100 GPU for LoRA.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|}
\hline Model \& Method & \# Trainable Parameters & MNLI & SST-2 & MRPC & CoLA & QNLI & QQP & RTE & STS-B & Avg. \\
\hline \(\operatorname{RoB}_{\text {base }}(\mathrm{FT})^{*}\) & 125.0M & 87.6 & 94.8 & 90.2 & 63.6 & 92.8 & 91.9 & 78.7 & 91.2 & 86.4 \\
\hline $\operatorname{RoB}_{\text {base }}$ (BitFit)* & 0.1M & 84.7 & 93.7 & 92.7 & 62.0 & 91.8 & 84.0 & 81.5 & 90.8 & 85.2 \\
\hline \(\operatorname{RoB}_{\text {base }}\left(\text { Adpt }^{\mathrm{D}}\right)^{*}\) & 0.3M & $87.1_{ \pm .0}$ & $94.2_{ \pm .1}$ & $88.5_{ \pm 1.1}$ & $60.8_{ \pm .4}$ & $93.1_{ \pm .1}$ & $90.2_{ \pm .0}$ & $71.5_{ \pm 2.7}$ & $89.7_{ \pm .3}$ & 84.4 \\
\hline \(\operatorname{RoB}_{\text {base }}\left(\text { Adpt }^{\mathrm{D}}\right)^{*}\) & 0.9M & $87.3_{ \pm .1}$ & $94.7_{ \pm .3}$ & $88.4_{ \pm .1}$ & $62.6_{ \pm .9}$ & $93.0_{ \pm .2}$ & $90.6_{ \pm .0}$ & $75.9_{ \pm 2.2}$ & $90.3_{ \pm .1}$ & 85.4 \\
\hline \(\mathrm{RoB}_{\text {base }}(\mathrm{LoRA})\) & 0.3M & $87.5_{ \pm .3}$ & $\mathbf{9 5 . 1}_{+. \mathbf{2}}$ & $89.7_{ \pm .7}$ & $63.4_{ \pm 1.2}$ & $\mathbf{9 3 . 3}_{+3}$ & $90.8_{ \pm .1}$ & $\mathbf{8 6 . 6}_{ \pm .7}$ & $\mathbf{9 1 . 5}_{ \pm . \mathbf{2}}$ & 87.2 \\
\hline \(\mathrm{RoB}_{\text {large }}(\mathrm{FT})^{*}\) & 355.0M & 90.2 & 96.4 & 90.9 & 68.0 & 94.7 & 92.2 & 86.6 & 92.4 & 88.9 \\
\hline $\mathrm{RoB}_{\text {large }}($ LoRA \()\) & 0.8M & $\mathbf{9 0 . 6}_{ \pm .2}$ & $96.2_{ \pm .5}$ & $\mathbf{9 0 . 9}_{ \pm 1.2}$ & $\mathbf{6 8 . 2}_{\mathbf{\pm 1 . 9}}$ & $\mathbf{9 4 . 9}_{+.3}$ & $91.6_{ \pm .1}$ & $\mathbf{8 7 . 4}_{ \pm 2.5}$ & $\mathbf{9 2 . 6}_{ \pm .2}$ & 89.0 \\
\hline $\operatorname{RoB}_{\text {large }}\left(\right.$ Adpt \(\left.^{\text {P }}\right) \dagger\) & 3.0M & $90.2_{ \pm .3}$ & $96.1_{ \pm .3}$ & $90.2_{ \pm .7}$ & $\mathbf{6 8 . 3}_{ \pm 1.0}$ & $\mathbf{9 4 . 8}_{\text {±. } 2}$ & $\mathbf{9 1 . 9}_{ \pm .1}$ & $83.8_{ \pm 2.9}$ & $92.1_{ \pm .7}$ & 88.4 \\
\hline $\operatorname{RoB}_{\text {large }}\left(\right.$ Adpt \(\left.^{\text {P }}\right) \dagger\) & 0.8M & $\mathbf{9 0 . 5}_{ \pm .3}$ & $\mathbf{9 6 . 6}_{ \pm .2}$ & $89.7_{ \pm 1.2}$ & $67.8_{ \pm 2.5}$ & $\mathbf{9 4 . 8}_{+.3}$ & $91.7_{ \pm .2}$ & $80.1_{ \pm 2.9}$ & $91.9_{ \pm .4}$ & 87.9 \\
\hline $\operatorname{RoB}_{\text {large }}\left(\right.$ Adpt \(\left.^{\text {H }}\right) \dagger\) & 6.0M & $89.9_{ \pm .5}$ & $96.2_{ \pm .3}$ & $88.7_{ \pm 2.9}$ & $66.5_{ \pm 4.4}$ & $94.7_{ \pm .2}$ & $92.1_{ \pm .1}$ & $83.4_{ \pm 1.1}$ & $91.0_{ \pm 1.7}$ & 87.8 \\
\hline $\operatorname{RoB}_{\text {large }}\left(\right.$ Adpt \(\left.^{\text {H }}\right) \dagger\) & 0.8M & $90.3_{ \pm .3}$ & $96.3_{ \pm .5}$ & $87.7_{ \pm 1.7}$ & $66.3_{ \pm 2.0}$ & $94.7_{ \pm .2}$ & $91.5_{ \pm .1}$ & $72.9_{ \pm 2.9}$ & $91.5_{ \pm .5}$ & 86.4 \\
\hline \(\operatorname{RoB}_{\text {large }}(\operatorname{LoRA}) \dagger\) & 0.8M & $\mathbf{9 0 . 6}_{ \pm .2}$ & $96.2_{ \pm .5}$ & $\mathbf{9 0 . 2}_{+1.0}$ & $68.2_{ \pm 1.9}$ & $\mathbf{9 4 . 8}_{+.3}$ & $91.6_{ \pm .2}$ & $\mathbf{8 5 . 2}_{\mathbf{\pm 1 . 1}}$ & $\mathbf{9 2 . 3}_{+.5}$ & 88.6 \\
\hline \(\mathrm{DeB}_{\mathrm{XXL}}(\mathrm{FT})^{*}\) & 1500.0M & 91.8 & 97.2 & 92.0 & 72.0 & 96.0 & 92.7 & 93.9 & 92.9 & 91.1 \\
\hline $\mathrm{DeB}_{\text {XXL }}$ (LoRA) & 4.7M & $\mathbf{9 1 . 9}_{+.2}$ & $96.9_{ \pm .2}$ & $\mathbf{9 2 . 6}_{ \pm .6}$ & $\mathbf{7 2 . 4}_{\mathbf{\pm 1 . 1}}$ & $\mathbf{9 6 . 0}_{\text {±. } 1}$ & $\mathbf{9 2 . 9}_{+.1}$ & $\mathbf{9 4 . 9}_{+.4}$ & $\mathbf{9 3 . 0}_{ \pm .2}$ & 91.3 \\
\hline
\end{tabular}

**Table 2: RoBERTa ${ **_{\text {base }}$, RoBERTa $_{\text {large }}$, and DeBERTa ${ }_{\text {XXL }}$ with different adaptation methods on the GLUE benchmark. We report the overall (matched and mismatched) accuracy for MNLI, Matthew's correlation for CoLA, Pearson correlation for STS-B, and accuracy for other tasks. Higher is better for all metrics. * indicates numbers published in prior works. † indicates runs configured in a setup similar to Houlsby et al. (2019) for a fair comparison.}
\end{table}

Bias-only or BitFit is a baseline where we only train the bias vectors while freezing everything else. Contemporarily, this baseline has also been studied by BitFit (Zaken et al., 2021).

Prefix-embedding tuning (PreEmbed) inserts special tokens among the input tokens. These special tokens have trainable word embeddings and are generally not in the model's vocabulary. Where to place such tokens can have an impact on performance. We focus on "prefixing", which prepends such tokens to the prompt, and "infixing", which appends to the prompt; both are discussed in Li \& Liang (2021). We use $l_{p}$ (resp. $l_{i}$ ) denote the number of prefix (resp. infix) tokens. The number of trainable parameters is \(|\Theta|=d_{\text {model }} \times\left(l_{p}+l_{i}\right)\).

Prefix-layer tuning (PreLayer) is an extension to prefix-embedding tuning. Instead of just learning the word embeddings (or equivalently, the activations after the embedding layer) for some special tokens, we learn the activations after every Transformer layer. The activations computed from previous layers are simply replaced by trainable ones. The resulting number of trainable parameters is \(|\Theta|=L \times d_{\text {model }} \times\left(l_{p}+l_{i}\right)\), where $L$ is the number of Transformer layers.

Adapter tuning as proposed in Houlsby et al. (2019) inserts adapter layers between the selfattention module (and the MLP module) and the subsequent residual connection. There are two fully connected layers with biases in an adapter layer with a nonlinearity in between. We call this original design Adapter ${ }^{\text {H }}$. Recently, Lin et al. (2020) proposed a more efficient design with the adapter layer applied only after the MLP module and after a LayerNorm. We call it Adapter ${ }^{\mathbf{L}}$. This is very similar to another deign proposed in Pfeiffer et al. (2021), which we call Adapter ${ }^{\mathbf{P}}$. We also include another baseline call AdapterDrop (Rücklé et al., 2020) which drops some adapter layers for greater efficiency (Adapter ${ }^{\mathbf{D}}$ ). We cite numbers from prior works whenever possible to maximize the number of baselines we compare with; they are in rows with an asterisk (*) in the first column. In all cases, we have \(|\Theta|=\hat{L}_{\text {Adpt }} \times\left(2 \times d_{\text {model }} \times r+r+d_{\text {model }}\right)+2 \times \hat{L}_{L N} \times d_{\text {model }}\) where $\hat{L}_{\text {Adpt }}$ is the number of adapter layers and $\hat{L}_{L N}$ the number of trainable LayerNorms (e.g., in Adapter ${ }^{\mathrm{L}}$ ).

LoRA adds trainable pairs of rank decomposition matrices in parallel to existing weight matrices. As mentioned in Section 4.2, we only apply LoRA to $W_{q}$ and $W_{v}$ in most experiments for simplicity. The number of trainable parameters is determined by the rank $r$ and the shape of the original weights: $|\Theta|=2 \times \hat{L}_{\text {LoRA }} \times d_{\text {model }} \times r$, where $\hat{L}_{\text {LoRA }}$ is the number of weight matrices we apply LoRA to.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Model \& Method} & \multirow{2}{*}{\# Trainable Parameters} & \multicolumn{5}{|c|}{E2E NLG Challenge} \\
\hline & & BLEU & NIST & MET & ROUGE-L & CIDEr \\
\hline GPT-2 M (FT)* & 354.92M & 68.2 & 8.62 & 46.2 & 71.0 & 2.47 \\
\hline GPT-2 M (Adapter ${ }^{\text {L }}$ )* & 0.37M & 66.3 & 8.41 & 45.0 & 69.8 & 2.40 \\
\hline GPT-2 M (Adapter ${ }^{\text {L }}$ )* & 11.09M & 68.9 & 8.71 & 46.1 & 71.3 & 2.47 \\
\hline GPT-2 M (Adapter ${ }^{\mathrm{H}}$ ) & 11.09M & $67.3_{ \pm .6}$ & $8.50_{ \pm .07}$ & $46.0_{ \pm .2}$ & $70.7_{ \pm .2}$ & $2.44_{ \pm .01}$ \\
\hline GPT-2 M (FTTop2)* & 25.19M & 68.1 & 8.59 & 46.0 & 70.8 & 2.41 \\
\hline GPT-2 M (PreLayer)* & 0.35M & 69.7 & 8.81 & 46.1 & 71.4 & 2.49 \\
\hline GPT-2 M (LoRA) & 0.35M & $\mathbf{7 0 . 4}_{\text {±. } 1}$ & $\mathbf{8 . 8 5}_{\mathbf{\pm . 0 2}}$ & $\mathbf{4 6 . 8}_{\mathbf{\pm . 2}}$ & $\mathbf{7 1 . 8}_{\boldsymbol{\pm} \boldsymbol{.} \mathbf{1}}$ & $\mathbf{2 . 5 3}_{\mathbf{\pm . 0 2}}$ \\
\hline GPT-2 L (FT)* & 774.03M & 68.5 & 8.78 & 46.0 & 69.9 & 2.45 \\
\hline GPT-2 L (Adapter ${ }^{\text {L }}$ ) & 0.88M & $69.1_{ \pm .1}$ & $8.68_{ \pm .03}$ & $46.3_{ \pm .0}$ & $71.4_{ \pm .2}$ & $\mathbf{2 . 4 9}_{ \pm . \mathbf{0}}$ \\
\hline GPT-2 L (Adapter ${ }^{\text {L }}$ ) & 23.00M & $68.9_{ \pm .3}$ & $8.70_{ \pm .04}$ & $46.1_{ \pm .1}$ & $71.3_{ \pm .2}$ & $2.45_{ \pm .02}$ \\
\hline GPT-2 L (PreLayer)* & 0.77M & 70.3 & 8.85 & 46.2 & 71.7 & 2.47 \\
\hline GPT-2 L (LoRA) & 0.77M & $\mathbf{7 0 . 4}_{\text {±. } 1}$ & $\mathbf{8 . 8 9}_{ \pm . \mathbf{0 2}}$ & $\mathbf{4 6 . 8}_{\mathbf{\pm . 2}}$ & $\mathbf{7 2 . 0}_{\mathbf{\pm . 2}}$ & $2.47_{ \pm .02}$ \\
\hline
\end{tabular}

**Table 3: GPT-2 medium (M) and large (L) with different adaptation methods on the E2E NLG Challenge. For all metrics, higher is better. LoRA outperforms several baselines with comparable or fewer trainable parameters. Confidence intervals are shown for experiments we ran. * indicates numbers published in prior works.**
\end{table}

## 5.2 RoBERTA BASE/LARGE

RoBERTa (Liu et al., 2019) optimized the pre-training recipe originally proposed in BERT (Devlin et al., 2019a) and boosted the latter's task performance without introducing many more trainable parameters. While RoBERTa has been overtaken by much larger models on NLP leaderboards such as the GLUE benchmark (Wang et al., 2019) in recent years, it remains a competitive and popular pre-trained model for its size among practitioners. We take the pre-trained RoBERTa base (125M) and RoBERTa large (355M) from the HuggingFace Transformers library (Wolf et al., 2020) and evaluate the performance of different efficient adaptation approaches on tasks from the GLUE benchmark. We also replicate Houlsby et al. (2019) and Pfeiffer et al. (2021) according to their setup. To ensure a fair comparison, we make two crucial changes to how we evaluate LoRA when comparing with adapters. First, we use the same batch size for all tasks and use a sequence length of 128 to match the adapter baselines. Second, we initialize the model to the pre-trained model for MRPC, RTE, and STS-B, not a model already adapted to MNLI like the fine-tuning baseline. Runs following this more restricted setup from Houlsby et al. (2019) are labeled with † . The result is presented in Table 2 (Top Three Sections). See Section D. 1 for details on the hyperparameters used.

## 5.3 DeBERTA XXL

DeBERTa (He et al., 2021) is a more recent variant of BERT that is trained on a much larger scale and performs very competitively on benchmarks such as GLUE (Wang et al., 2019) and SuperGLUE (Wang et al., 2020). We evaluate if LoRA can still match the performance of a fully fine-tuned DeBERTa XXL (1.5B) on GLUE. The result is presented in Table 2 (Bottom Section). See Section D. 2 for details on the hyperparameters used.

## 5.4 GPT-2 MEDIUM/LARGE

Having shown that LoRA can be a competitive alternative to full fine-tuning on NLU, we hope to answer if LoRA still prevails on NLG models, such as GPT-2 medium and large (Radford et al., b). We keep our setup as close as possible to Li \& Liang (2021) for a direct comparison. Due to space constraint, we only present our result on E2E NLG Challenge (Table 3) in this section. See Section F. 1 for results on WebNLG (Gardent et al., 2017) and DART (Nan et al., 2020). We include a list of the hyperparameters used in Section D. 3

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline \multirow{2}{*}{Model\&Method} & \multirow{2}{*}{\# Trainable Parameters} & WikiSQL & MNLI-m & SAMSum \\
\hline & & Acc. (\%) & Acc. (\%) & R1/R2/RL \\
\hline GPT-3 (FT) & $175,255.8 \mathrm{M}$ & 73.8 & 89.5 & 52.0/28.0/44.5 \\
\hline GPT-3 (BitFit) & 14.2M & 71.3 & 91.0 & 51.3/27.4/43.5 \\
\hline GPT-3 (PreEmbed) & 3.2M & 63.1 & 88.6 & 48.3/24.2/40.5 \\
\hline GPT-3 (PreLayer) & 20.2M & 70.1 & 89.5 & 50.8/27.3/43.5 \\
\hline GPT-3 (Adapter ${ }^{\mathrm{H}}$ ) & 7.1M & 71.9 & 89.8 & 53.0/28.9/44.8 \\
\hline GPT-3 (Adapter ${ }^{\mathrm{H}}$ ) & 40.1M & 73.2 & 91.5 & 53.2/29.0/45.1 \\
\hline GPT-3 (LoRA) & 4.7M & 73.4 & 91.7 & 53.8/29.8/45.9 \\
\hline GPT-3 (LoRA) & 37.7M & 74.0 & 91.6 & 53.4/29.2/45.1 \\
\hline
\end{tabular}

**Table 4: Performance of different adaptation methods on GPT-3 175B. We report the logical form validation accuracy on WikiSQL, validation accuracy on MultiNLI-matched, and Rouge-1/2/L on SAMSum. LoRA performs better than prior approaches, including full fine-tuning. The results on WikiSQL have a fluctuation around ±0.5\%, MNLI-m around ±0.1\%, and SAMSum around $\pm 0.2 / \pm 0.2 / \pm 0.1$ for the three metrics.**
\end{table}

## 5.5 Scaling up to GPT-3 175B

As a final stress test for LoRA, we scale up to GPT-3 with 175 billion parameters. Due to the high training cost, we only report the typical standard deviation for a given task over random seeds, as opposed to providing one for every entry. See Section D. 4 for details on the hyperparameters used.

As shown in Table 4. LoRA matches or exceeds the fine-tuning baseline on all three datasets. Note that not all methods benefit monotonically from having more trainable parameters, as shown in Fig-ure 2. We observe a significant performance drop when we use more than 256 special tokens for prefix-embedding tuning or more than 32 special tokens for prefix-layer tuning. This corroborates similar observations in Li \& Liang (2021). While a thorough investigation into this phenomenon is out-of-scope for this work, we suspect that having more special tokens causes the input distribution to shift further away from the pre-training data distribution. Separately, we investigate the performance of different adaptation approaches in the low-data regime in Section F. 3

![Figure](./huLoRALowRankAdaptation2021_images/2f8c4be1-43fa-4600-99ef-0945e0605540-08_430_1346_1579_387.jpg)

**Figure 2: GPT-3 175B validation accuracy vs. number of trainable parameters of several adaptation methods on WikiSQL and MNLI-matched. LoRA exhibits better scalability and task performance. See Section F. 2 for more details on the plotted data points.**

