# D Comparison of Fine-Tuning in BF16 and FP32 Precision

In this section, we compare the effects of training with BFloat16 and Float32 precision. The comparing include four models: LLaMA-2-7B, Mistral-7B, Gemma-7B, and LLaMA-3-8B, each fine-tuned with all parameters in both BFloat16 and Float32 precision on the MetaMathQA-395K dataset. The validation results conducted on the GSM8K dataset are shown in Figure 7

\begin{table}

**Table 7: Comparison of fine-tuning results of LLaMA-2-7B, Mistral-7B, Gemma-7B, and LLaMA-3-8B in BF16 and FP32 precision on MetaMathQA-395K dataset for 3 epochs.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Model} & \multicolumn{2}{|l|}{Training Loss} & \multicolumn{2}{|c|}{GSM8K ACC (\%)} & \multicolumn{2}{|c|}{MATH ACC (\%)} \\
\hline & BF16 & FP32 & BF16 & FP32 & BF16 & FP32 \\
\hline LLaMA-2-7B & 0.1532 & 0.1316 & 63.15 & 68.31 & 13.14 & 20.38 \\
\hline Mistral-7B & 0.1145 & 0.1306 & 73.09 & 65.88 & 26.44 & 23.66 \\
\hline Gemma-7B & 0.1331 & 0.1382 & 75.21 & 75.97 & 29.18 & 28.64 \\
\hline LLaMA-3-8B & 0.1271 & 0.1317 & 81.96 & 75.44 & 33.16 & 28.72 \\
\hline
\end{tabular}
\end{table}
From Table 7, it is evident that the choice of precision greatly affects the experimental results. For example, the LLaMA-2-7B model shows a 5.16\% higher performance on the GSM8K dataset when using FP32 compared to BF16. Conversely, the Mistral-7B and LLaMA-3-8B on GSM8K are 7.21\% and 6.52\% lower with FP32 than with BF16 separately. The Gemma-7B model shows similar performance with both precisions. Unfortunately, the experiments did not prove which precision is better. To reduce training costs, we use BF16 precision when fine-tuning all parameters. For methods with lower training costs, such as LoRA, PiSSA, we use FP32 precision. For QLoRA, QPiSSA and LoftQ, the base model was used NF4 precision, while the adapter layers used FP32 precision.

