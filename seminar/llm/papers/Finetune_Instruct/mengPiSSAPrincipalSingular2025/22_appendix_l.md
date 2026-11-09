# L Experimental Settings on NLU

Datasets We evaluate the performance of PiSSA on GLUE benchmark, including 2 single-sentence classification tasks (CoLA, SST), 5 pairwise text classification tasks (MNLI, RTE, QQP, MRPC and QNLI) and 1 text similarity prediction task (STS-B). We report overall matched and mismatched accuracy on MNLI, Matthew's correlation on CoLA, Pearson correlation on STS-B, and accuracy on the other datasets.

Implementation Details To evaluate the performance of PiSSA intuitively, we compared PiSSA to LoRA with the same number of trainable parameters. DeBERTa-v3-base has $184 M$ trainable parameters. PiSSA and LoRA were applied to $W_{Q}, W_{K}$ and $W_{V}$ respectively, resulting in a total of $1.33 M$ trainable parameters.

The results for full fine-tune, BitFit [15], HAdapter [30], PAdapter [36], LoRA with Gassian initialization [11] and AdaLoRA are sourced from AdaLoRA [59], based on five runs. The remaining results use the publicly available LoftQ [14] codebase and are averaged over three runs. In LoRA, the B matrix is initialized to zero, while the A matrix can be initialized using various methods, such as Gaussian initialization and Kaiming initialization [68]. The selection of the initialization method can influence the final results. In this paper, we report the different results of LoRA based on Gaussian initialization and Kaiming initialization in the experiments, as shown in Table 2 and Table 3. For DoRA, we used the code from the PEFT package for deployment and conducted a search on key hyperparameters. We set the rank of PiSSA in this experiment as 8 and selecte lora alpha in 8, 16. We utilize AdamW with linear learning rate schedule to optimize and tune learning rate (LR) from 1e-4,2e-4,3e-4,4e-4,5e-4, 6e-4, 5e-5, 3e-5. Batch sizes (BS) are selected from 6, 8, 16, 32. The hyperparameter configurations of PiSSA, DoRA and LoRA with Kaiming Initialization are shown in Table 10. LoRA ${ }^{K}$ denotes LoRA with Kaiming initialization, and $\alpha$ denotes LoRA alpha.

\begin{table}

**Table 10: Hyperparameters of PiSSA, DoRA and LoRA with Kaiming Initialization on GLUE.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Dataset} & \multicolumn{4}{|c|}{PiSSA} & \multicolumn{4}{|c|}{DoRA} & \multicolumn{4}{|c|}{LoRA ${ }^{K}$} \\
\hline & Epoch & BS & LR & $\alpha$ & Epoch & BS & LR & $\alpha$ & Epoch & BS & LR & $\alpha$ \\
\hline MNLI & 5 & 16 & 5e-4 & 8 & 10 & 32 & 2e-4 & 16 & 10 & 32 & 3e-4 & 8 \\
\hline SST-2 & 20 & 16 & $3 \mathrm{e}-5$ & 8 & 10 & 16 & 4e-4 & 16 & 10 & 32 & 1e-4 & 8 \\
\hline MRPC & 20 & 32 & 2e-4 & 8 & 10 & 32 & 4e-4 & 16 & 10 & 32 & 4e-4 & 8 \\
\hline CoLA & 20 & 16 & 1e-4 & 8 & 20 & 8 & $1 \mathrm{e}-4$ & 6 & 30 & 32 & 4e-4 & 8 \\
\hline QNLI & 10 & 32 & 1e-4 & 16 & 10 & 16 & 2e-4 & 16 & 25 & 32 & 3e-4 & 8 \\
\hline QQP & 10 & 16 & 1e-4 & 8 & 10 & 16 & $1 \mathrm{e}-4$ & 6 & 10 & 16 & 3e-4 & 8 \\
\hline RTE & 50 & 16 & 1e-4 & 8 & 50 & 8 & 2e-4 & 6 & 50 & 32 & 4e-4 & 8 \\
\hline STS-B & 20 & 8 & 3e-4 & 8 & 20 & 16 & $3 \mathrm{e}-4$ & 6 & 30 & 16 & 4e-4 & 8 \\
\hline
\end{tabular}
\end{table}

