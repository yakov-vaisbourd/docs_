# I Combining QPiSSA with Various Quantization Methods

In addition to NF4 quantization, QPiSSA can also be combined with GPTQ and INT8 quantization. We posit that PiSSA effectively reduces quantization error for several reasons:
\begin{itemize}
\item[-] It reduces outlier values;
\item[-] It makes the value distribution more Gaussian-like;
\item[-] It preserves larger values in full precision, thereby narrowing the weight distribution in the quantized portion.
\end{itemize}

While INT8 also targets the reduction of outlier values (point 1), PiSSA has the potential to enhance this effect. The second point aligns well with NF4, and the third point is crucial as PiSSA uses an adaptor to retain a significant portion of weights in full precision, maintaining the integrity of critical values.

\begin{table}

**Table 9: Quantization Error and Accuracy for PiSSA Combined with Various Quantization Methods. GPTQ quantizes each row $w$ independently, adjusting one weight at a time while updating all remaining, non-quantized weights. Therefore, the nuclear norm method used for calculating quantization error in the main paper is not applicable to GPTQ. Instead, we measure Perplexity on WikiText-2, where a lower Perplexity indicates reduced quantization error.**
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline \multirow{2}{*}{Model} & \multirow{2}{*}{Dataset} & \multicolumn{2}{|c|}{Quantization Error} & \multicolumn{2}{|c|}{GSM8K Accuracy} \\
\hline & & QLoRA & PiSSA & QLoRA & PiSSA \\
\hline \multirow{3}{*}{LLaMA-3-8B} & NF4 & 324.8 (nuclear norm) & 265.8 (nuclear norm) & 70.79±0.42 & 73.76±0.20 \\
\hline & INT8 & 34.47 (nuclear norm) & 28.21 (nuclear norm) & 71.68±0.14 & 76.54±0.32 \\
\hline & GPTQ & 20.79 (perplexity) & 6.23 (perplexity) & $70.18 \pm 0.42$ & 74.58±0.22 \\
\hline
\end{tabular}
\end{table}
As shown in Table 9, QPiSSA combined with INT8 reduces quantization error by $\mathbf{1 8 . 1 6 \%}$ on LLaMA-3-8B, and significantly outperforms QLoRA using INT8. Furthermore, in row 3 of the table, the perplexity of LLaMA-3-8B increases to 20.79 after quantization with GPTQ-4bit on the C4 dataset. However, when PiSSA is applied, the perplexity is reduced to 6.23. These results confirm the effectiveness of PiSSA in reducing quantization error, as discussed in the main paper.

Overall, QPiSSA demonstrates a clear advantage over QLoRA when combined with various quantization methods, retaining the fast convergence and superior performance characteristics of PiSSA while minimizing quantization error.

