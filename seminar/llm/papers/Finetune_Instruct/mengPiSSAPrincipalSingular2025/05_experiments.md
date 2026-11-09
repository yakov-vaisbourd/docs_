# 5 Experiments

The experiments were conducted on the NVIDIA A800-SXM4(80G) GPU. In our experiments, we adopt the Alpaca [49] implementation strategy, using the AdamW optimizer with a batch size of 128, a learning rate of 2e-5, cosine annealing schedules, and a warmup ratio of 0.03, without any weight decay. As discussed in Section B. 3 of QLoRA [10], we compute the loss using only the responses from the instruction-following datasets. We ensure lora_alpha is always equal to lora_r, set lora_dropout to 0, and incorporate the adapters into all linear layers of the base model. We utilize the Float32 computation type for both the base model and the adapter in LoRA and PiSSA. For QLoRA, LoftQ, and QPiSSA, we use 4-bit NormalFloat [10] for the base model and Float32 for the adapter. BFloat16 [50] is used for full parameter fine-tuning to save the resources (see Appendix D).

## 5.1 Evaluating the Performance of PiSSA on both NLG and NLU Tasks

We begin by comparing PiSSA, LoRA, and full-parameter fine-tuning on natural language generation (NLG) tasks. We fine-tuned LLaMA 2-7B [51], Mistral-7B-v0.1 [52], and Gemma-7B [53] on the MetaMathQA dataset [2] to assess their mathematical problem-solving capabilities on the GSM8K [54] and MATH [55] validation sets. Additionally, the models were fine-tuned on the Code-Feedback dataset [56] and evaluated for coding proficiency using the HumanEval [57] and MBPP [58] datasets. Furthermore, the models were trained on the WizardLM-Evol-Instruct dataset [7] and tested for conversational abilities on the MT-Bench dataset [6]. All experiments were conducted using subsets containing 100K data points and were trained for only one epoch to reduce training overhead.

As shown in Table 2, across all models and tasks, fine-tuning with PiSSA consistently surpasses the performance of fine-tuning with LoRA. Further experiments demonstrated that this improvement is robust across various amounts of training data and epochs (Section 5.2), including both 4-bit and full precision (Section 5.3), different model sizes and types (Section 5.4), and varying proportions of trainable parameters (Section 5.5).

We also evaluate PiSSA's natural language understanding (NLU) capability on the GLUE benchmark [60] with DeBERTa-v3-base [61]. Table 3 presents the results across 8 tasks. PiSSA outperforms LoRA in 7 out of 8 NLU tasks, achieving an overall average improvement of $1.21 \%$. Upon reviewing the training loss on the exceptional dataset, MNLI, we observed that PiSSA's average loss of 0.17 was lower than LoRA's 0.24 in the final epoch. This indicates that the fitting ability of PiSSA remains stronger than that of LoRA.

\begin{table}

**Table 2: Comparison of PiSSA and LoRA on NLG tasks, with results averaged over three runs and reported with standard deviations.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline Model & Strategy & GSM8K & MATH & HumanEval & MBPP & MT-Bench \\
\hline \multirow{4}{*}{LLaMA 2-7B} & Full FT & $49.13 \pm 0.21$ & $7.29 \pm 0.22$ & $21.20 \pm 0.30$ & $35.59 \pm 0.25$ & 4.91±0.01 \\
\hline & LoRA(gaussian) & $42.85 \pm 0.12$ & $5.50 \pm 0.33$ & $18.35 \pm 0.31$ & $35.50 \pm 0.14$ & $4.59 \pm 0.07$ \\
\hline & LoRA(kaiming) & $43.23 \pm 0.64$ & $5.90 \pm 0.16$ & $18.21 \pm 0.45$ & 35.47±0.37 & $4.56 \pm 0.04$ \\
\hline & PiSSA & 53.22±0.55 & 7.47±0.34 & 21.92±0.38 & 37.24±0.63 & $4.88 \pm 0.03$ \\
\hline \multirow{4}{*}{Mistral-7B} & Full FT & $69.91 \pm 0.25$ & $18.64 \pm 0.35$ & $45.31 \pm 0.14$ & $51.46 \pm 0.13$ & $4.95 \pm 0.05$ \\
\hline & LoRA(gaussian) & $69.50 \pm 0.42$ & 20.08±0.20 & $43.78 \pm 0.34$ & $58.46 \pm 0.37$ & $4.90 \pm 0.05$ \\
\hline & LoRA(kaiming) & $69.40 \pm 0.25$ & 19.99±0.44 & 43.74±0.14 & 58.39±0.02 & $4.93 \pm 0.05$ \\
\hline & PiSSA & 73.31±0.23 & 23.12±0.52 & 46.88±0.25 & 62.55±0.58 & 5.34±0.04 \\
\hline \multirow{4}{*}{Gemma-7B} & Full FT & $72.09 \pm 0.32$ & $22.71 \pm 0.34$ & $47.02 \pm 0.27$ & $55.67 \pm 0.50$ & $5.40 \pm 0.12$ \\
\hline & LoRA(gaussian) & 75.11±0.64 & 30.41±0.48 & $53.70 \pm 0.23$ & 65.58±0.29 & $4.98 \pm 0.02$ \\
\hline & LoRA(kaiming) & $74.53 \pm 0.47$ & $29.90 \pm 0.16$ & $53.57 \pm 0.27$ & 65.25±0.29 & $4.97 \pm 0.09$ \\
\hline & PiSSA & 77.78±0.32 & 31.33±0.33 & 54.31±0.28 & 66.17±0.43 & 5.64±0.10 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 3: Comparison of PiSSA and LoRA on NLU tasks. LoRA ${ **^{G}$ and LoRA ${ }^{K}$ denote LoRA with Gaussian and Kaiming initialization for $B$, respectively. Results for full fine-tuning, BitFit [15], HAdapter [30], PAdapter [36], LoRA ${ }^{G}$ [11] and AdaLoRA are from AdaLoRA [59], averaged over five runs. Remaining methods are averaged over three runs, with details in Appendix L.}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|}
\hline Method & Params & MNLI & SST2 & MRPC & CoLA & QNLI & QQP & RTE & STSB & ALL \\
\hline Full FT & 184M & 89.90 & 95.63 & 89.46 & 69.19 & 94.03 & 92.40 & 83.75 & 91.60 & 88.25 \\
\hline BitFit & 0.1M & 89.37 & 94.84 & 87.75 & 66.96 & 92.24 & 88.41 & 78.70 & 91.35 & 86.20 \\
\hline HAdapter & 1.22M & 90.13 & 95.53 & 89.95 & 68.64 & 94.11 & 91.91 & 84.48 & 91.48 & 88.28 \\
\hline PAdapter & 1.18M & 90.33 & 95.61 & 89.46 & 68.77 & 94.29 & 92.04 & 85.20 & 91.54 & 88.41 \\
\hline LoRA ${ }^{G}$ & 1.33M & 90.65 & 94.95 & 89.95 & 69.82 & 93.87 & 91.99 & 85.20 & 91.60 & 88.50 \\
\hline LoRA ${ }^{K}$ & 1.33M & 89.96 & 95.64 & 90.28 & 70.69 & 93.84 & 92.03 & 84.84 & 91.68 & 88.62 \\
\hline DoRA & 1.27M & 90.29 & 95.79 & 90.93 & 70.85 & 94.10 & 92.07 & 86.04 & 91.79 & 88.98 \\
\hline AdaLoRA & 1.27M & 90.76 & 96.10 & 90.69 & 71.45 & 94.55 & 92.23 & 88.09 & 91.84 & 89.46 \\
\hline PiSSA & 1.33M & 90.37 & 96.22 & 91.50 & 73.12 & 94.43 & 92.33 & 88.69 & 92.00 & 89.83 \\
\hline
\end{tabular}
\end{table}

## 5.2 Experiments using Full Data and More Epochs

In this section, we finetune LLaMA 2-7B model on the complete MetaMathQA-395K dataset for 3 epochs to ensure thorough saturation. The training loss and gradient norms is visualized to demonstrate quicker convergence and evaluated on the GSM8K dataset every 1000 steps to demonstrate superior performance of PiSSA compared to LoRA. The results are depicted in Figure 4 Additionally, similar comparisons on Mistral-7B and Gemma-7B are detailed in Appendix J

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-07_385_1390_1856_367.jpg)

**Figure 4: The loss, grad norm, and evaluation accuracy over the training steps of LoRA (indicated in blue), PiSSA (in orange), and full parameter fine-tuning (in red).**

According to Figure 4a, the loss of PiSSA reduces rapidly during the first 100 steps, and the grad norm (shown in Figure 4b) of PiSSA is significantly higher than that of LoRA, with a trend similar to full fine-tuning. Throughout the process, the loss of PiSSA remains lower than that of LoRA, indicating
that PiSSA converges to a better local optimum. As shown in Figure 4c, PiSSA consistently achieves higher accuracy compared to LoRA, and in most cases also surpasses full parameters fine-tuning. We hypothesize that this is because PiSSA is a denoised version of full fine-tuning. Comparing the grad norm and loss curves of PiSSA and full fine-tuning, we can see that the larger grad norm of full fine-tuning does not bring lower loss, indicating that a portion of the grad norm is spent on noisy directions not beneficial for loss reduction. This phenomenon is consistent with Figure 2a

## 5.3 Conducting 4-bit Quantization Experiments

In this section, we first compare the initial quantization error reduction ratio of PiSSA, QLoRA, and LoftQ. This ratio is defined as \(\left(1-\frac{\left\|W-\left(n f 4\left(W^{\prime}\right)+A B\right)\right\|_{*}}{\|W-n f 4(W)\|_{*}}\right) \times 100 \%\), measuring the relative error decrease achieved by each mehod compared to directly quantizing the base model. The partial results are presented in Table 4, and the complete results can be found in Table 8 in Appendix E.

\begin{table}

**Table 4: The quantization error reduction ratio of QLoRA, LoftQ, and PiSSA across different layers.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|}
\hline & Method & Rank & Q & K & V & O & Gate & Up & Down & AVG \\
\hline \multirow{3}{*}{LLaMA 2-7B} & QLoRA & - & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline & loftQ & 128 & 16.5 & 16.5 & 15.9 & 16.0 & 12.4 & 12.4 & 12.3 & 14.6 \\
\hline & PiSSA & 128 & 27.9 & 27.2 & 18.7 & 18.6 & 15.8 & 13.6 & 13.6 & 19.4 \\
\hline \multirow{3}{*}{LLaMA 3-8B} & QLoRA & - & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline & loftQ & 128 & 16.4 & 29.8 & 28.8 & 16.1 & 11.9 & 11.7 & 11.7 & 18.1 \\
\hline & PiSSA & 128 & 26.3 & 41.7 & 32.3 & 20.1 & 14.4 & 12.5 & 12.9 & 22.9 \\
\hline \multirow{4}{*}{LLaMA 3-70B} & QLoRA & - & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline & LoftQ & 64 & 6.1 & 17.8 & 17.0 & 6.0 & 4.3 & 4.4 & 4.2 & 8.5 \\
\hline & PiSSA & 64 & 15.7 & 34.2 & 18.9 & 7.5 & 6.7 & 5.7 & 4.7 & 13.4 \\
\hline & PiSSA & 128 & 23.2 & 49.0 & 30.5 & 12.5 & 10.1 & 8.8 & 8.2 & 20.3 \\
\hline
\end{tabular}
\end{table}
In Table 4, PiSSA reduces the quantization error by about 20\% compared to directly quantizing the base model. The reduction is more significant for lower-rank matrices. For instance, in the LLaMA-3-70B [62], all "Key" projection layers see a reduction of $49 \%$. The results in Table 4 validate that QLoRA, discussed in Section 4, does not reduce quantization error. In contrast, PiSSA significantly outperforms LoftQ in reducing quantization error, as further discussed in Appendix H.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-08_715_1382_1692_371.jpg)

**Figure 5: The loss, grad norm, and evaluation accuracy over the training steps of (Q)LoRA, (Q)PiSSA, LoftQ and full parameter fine-tuning.**

The difference between QPiSSA and PiSSA is the quantization of the residual model to 4 bits. As introduced in Section 4, the residual model has less influence on the optimal direction compared with the principal adapter, which is the same in both QPiSSA and PiSSA. Therefore, besides reducing the quantization error, we expect QPiSSA to also converge faster than QLoRA and LoftQ. We train LLaMA 3-8B using LoRA/QLoRA, PiSSA/QPiSSA, LoftQ, and full fine-tuning on MetaMathQA-395K for 3 epochs, recording the loss, gradient norm, and accuracy on GSM8K, as shown in Figure 5

According to Figure 5. QPiSSA's loss reduction speed in the first 100 steps is even faster than PiSSA and full fine-tuning. Although LoftQ can reduce the quantization error, its loss convergence speed is not faster than LoRA and QLoRA, indicating that QPiSSA's ability to reduce the quantization error and its fast convergence might also be orthogonal capabilities. After sufficient training, QPiSSA's loss is also much lower than that of LoRA/QLoRA and LoftQ. The grad norm is significantly larger than those of LoRA/QLoRA and LoftQ. In terms of fine-tuning performance, QPiSSA's accuracy is higher than that of QLoRA and LoftQ and even better than that of full-precision LoRA.

## 5.4 Experiments Across Various Sizes and Types of Models

In this section, we compare (Q)PiSSA and (Q)LoRA across 9 models, ranging from 7-70B parameters, including LLaMA 2-7/13B [51], LLaMA-3-8/70B [62], Mistral-7B [52], Gemma-7B [53], and Qwen1.5-7B [63], Yi-1.5-34B [64] and MoE models: DeepSeek-MoE-16B [65] and Mixtral-8x7B [66]. These models were fine-tuned on the MetaMathQA-100K and CodeFeedback-100K dataset and evaluated on the GSM8K and HumanEval. DeepSeek-MoE-16B, Mixtral-8x7B, Yi-1.5-34B, and LLaMA-3-70B were fine-tuned with QPiSSA and QLoRA, while the other models were using PiSSA and LoRA. From Figure 6. (Q)PiSSA, compared to (Q)LoRA, shows improved accuracy across various sizes and types of models, demonstrating its consistent advantage over (Q)LoRA.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-09_405_1382_1290_369.jpg)

**Figure 6: Comparison of (Q)PiSSA and (Q)LoRA across models from 7B to 70B.**

## 5.5 Experiments on Various Ranks

This section explores the impact of incrementally increasing the rank of PiSSA/QPiSSA and LoRA/QLoRA from 1 to 128, aiming to determine whether PiSSA/QPiSSA consistently outperforms LoRA/QLoRA under different ranks. The training is conducted using the MetaMathQA-100K dataset for 1 epoch, while the validation is performed on the GSM8K and MATH datasets. The outcomes of these experiments are depicted in Figure 7, with additional results presented in Appendix K.

Figure 7a illustrates the quantization error reduction ratio across various ranks. In this figure, QLoRA shows no reduction in quantization error, while QPiSSA consistently outperforms LoftQ in reducing quantization error across all ranks, with a particularly notable advantage at lower ranks. In Figure 7b, the final loss on the training set is shown for models trained with ranks ranging from 1 to 128. The results indicate that PiSSA and QPiSSA achieve a better fit to the training data compared to LoRA, QLoRA, and LoftQ. In Figures 7c and Figures 7d, we compare the accuracy of the fine-tuned models on the GSM8K and MATH validation sets under various ranks, finding that PiSSA consistently outperforms LoRA with the same amount of trainable parameters. Furthermore, as the rank increases, PiSSA will reach and surpass the performance of full-parameter fine-tuning.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-10_595_1339_247_391.jpg)

**Figure 7: The comparison among (Q)LoRA, (Q)PiSSA, LoftQ, and full fine-tuning across ranks.**

