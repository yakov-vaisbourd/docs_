# D Hyperparameters Used in Experiments

# D. 1 RoBERTA

We train using AdamW with a linear learning rate decay schedule. We sweep learning rate, number of training epochs, and batch size for LoRA. Following Liu et al. (2019), we initialize the LoRA modules to our best MNLI checkpoint when adapting to MRPC, RTE, and STS-B, instead of the usual initialization; the pre-trained model stays frozen for all tasks. We report the median over 5 random seeds; the result for each run is taken from the best epoch. For a fair comparison with the setup in Houlsby et al. (2019) and Pfeiffer et al. (2021), we restrict the model sequence length to 128 and used a fixed batch size for all tasks. Importantly, we start with the pre-trained RoBERTa large model when adapting to MRPC, RTE, and STS-B, instead of a model already adapted to MNLI. The runs with this restricted setup are marked with † . See the hyperparameters used in our runs in Table 9

# D. 2 DeBERTA

We again train using AdamW with a linear learning rate decay schedule. Following He et al. (2021), we tune learning rate, dropout probability, warm-up steps, and batch size. We use the same model sequence length used by (He et al., 2021) to keep our comparison fair. Following He et al. (2021), we initialize the LoRA modules to our best MNLI checkpoint when adapting to MRPC, RTE, and STS-B, instead of the usual initialization; the pre-trained model stays frozen for all tasks. We report the median over 5 random seeds; the result for each run is taken from the best epoch. See the hyperparameters used in our runs in Table 10

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline Method & Dataset & MNLI & SST-2 & MRPC & CoLA & QNLI & QQP & RTE & STS-B \\
\hline & Optimizer Warmup Ratio LR Schedule & \multicolumn{8}{|c|}{AdamW 0.06 Linear} \\
\hline \multirow{6}{*}{RoBERTa base LoRA} & Batch Size & 16 & 16 & 16 & 32 & 32 & 16 & 32 & 16 \\
\hline & \# Epochs & 30 & 60 & 30 & 80 & 25 & 25 & 80 & 40 \\
\hline & Learning Rate & 5E-04 & 5E-04 & 4E-04 & 4E-04 & 4E-04 & 5E-04 & 5E-04 & 4E-04 \\
\hline & LoRA Config. & & & & $r_{q}=r_{v}=8$ & & & & \\
\hline & LoRA $\alpha$ & & & & & & & & \\
\hline & Max Seq. Len. & \multicolumn{8}{|c|}{512} \\
\hline \multirow{6}{*}{RoBERTa large LoRA} & Batch Size & 4 & 4 & 4 & 4 & 4 & 4 & 8 & 8 \\
\hline & \# Epochs & 10 & 10 & 20 & 20 & 10 & 20 & 20 & 30 \\
\hline & Learning Rate & 3E-04 & 4E-04 & 3E-04 & 2E-04 & 2E-04 & 3E-04 & 4E-04 & 2E-04 \\
\hline & & & & & \multicolumn{2}{|l|}{$r_{q}=r_{v}$} & & & \\
\hline & LoRA $\alpha$ & & & & 16 & 16 & & & \\
\hline & Max Seq. Len. & 128 & 128 & 512 & 128 & 512 & 512 & 512 & 512 \\
\hline \multirow{4}{*}{RoBERTa large LoRA†} & Batch Size & & & & & & & & \\
\hline & Learning Rate & 3E-04 & 4E-04 & 3E-04 & 2E-04 & 2E-04 & 3E-04 & 4E-04 & 2E-04 \\
\hline & & & & & $r_{q}=r_{v}$ & \multicolumn{4}{|c|}{} \\
\hline & Max Seq. Len. & \multicolumn{8}{|c|}{128} \\
\hline \multirow{4}{*}{RoBERTa large Adpt ${ }^{\text {P }}$ (3M)†} & Batch Size & & & & & & & & \\
\hline & Learning Rate & 3E-05 & 3E-05 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 \\
\hline & & & & & 64 & & & & \\
\hline & Max Seq. Len. & \multicolumn{8}{|c|}{128} \\
\hline \multirow{3}{*}{RoBERTa large Adpt \({ }^{\mathrm{P}}(0.8 \mathrm{M}) \dagger\)} & Batch Size & & & & & & & & \\
\hline & Learning Rate & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 \\
\hline & & & & & 16 128 & & & & \\
\hline \multirow{4}{*}{RoBERTa large Adpt ${ }^{\mathrm{H}}$ (6M) †} & Batch Size & & & & & & & & \\
\hline & Learning Rate & 3E-05 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 \\
\hline & Bottleneck $r$ & & & & 64 & & & & \\
\hline & Max Seq. Len. & \multicolumn{8}{|c|}{128} \\
\hline \multirow{4}{*}{RoBERTa large Adpt ${ }^{\mathrm{H}}$ (0.8M)†} & Batch Size & & & & & & & & \\
\hline & Learning Rate & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 & 3E-04 \\
\hline & Max Seq. Len. & \multicolumn{8}{|c|}{\multirow[b]{2}{*}{128}} \\
\hline & & & & & & & & & \\
\hline
\end{tabular}

**Table 9: The hyperparameters we used for RoBERTa on the GLUE benchmark.**
\end{table}

# D. 3 GPT-2

We train all of our GPT-2 models using AdamW (Loshchilov \& Hutter, 2017) with a linear learning rate schedule for 5 epochs. We use the batch size, learning rate, and beam search beam size described in Li \& Liang (2021). Accordingly, we also tune the above hyperparameters for LoRA. We report the mean over 3 random seeds; the result for each run is taken from the best epoch. The hyperparameters used for LoRA in GPT-2 are listed in Table 11. For those used for other baselines, see Li \& Liang (2021).

# D. 4 GPT-3

For all GPT-3 experiments, we train using AdamW (Loshchilov \& Hutter, 2017) for 2 epochs with a batch size of 128 samples and a weight decay factor of 0.1 . We use a sequence length of 384 for

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline Method & Dataset & MNLI & SST-2 & MRPC & CoLA & QNLI & QQP & RTE & STS-B \\
\hline & Optimizer Warmup Ratio LR Schedule & \multicolumn{8}{|c|}{AdamW 0.1 Linear} \\
\hline \multirow{7}{*}{DeBERTa XXL LoRA} & Batch Size & 8 & 8 & 32 & 4 & 6 & 8 & 4 & 4 \\
\hline & \# Epochs & 5 & 16 & 30 & 10 & 8 & 11 & 11 & 10 \\
\hline & Learning Rate & 1E-04 & 6E-05 & 2E-04 & 1E-04 & 1E-04 & 1E-04 & 2E-04 & 2E-04 \\
\hline & Weight Decay & 0 & 0.01 & 0.01 & 0 & 0.01 & 0.01 & 0.01 & 0.1 \\
\hline & CLS Dropout & 0.15 & 0 & 0 & 0.1 & 0.1 & 0.2 & 0.2 & 0.2 \\
\hline & LoRA Config. LoRA $\alpha$ & \multicolumn{8}{|c|}{$r_{q}=r_{v}=8$} \\
\hline & Max Seq. Len. & 256 & 128 & 128 & 64 & 512 & 320 & 320 & 128 \\
\hline
\end{tabular}

**Table 10: The hyperparameters for DeBERTa XXL on tasks included in the GLUE benchmark.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Dataset & E2E & WebNLG & DART \\
\hline & \multicolumn{3}{|c|}{Training} \\
\hline Optimizer & \multicolumn{3}{|c|}{AdamW} \\
\hline Weight Decay & 0.01 & 0.01 & 0.0 \\
\hline Dropout Prob & 0.1 & 0.1 & 0.0 \\
\hline Batch Size & \multicolumn{3}{|c|}{8} \\
\hline \# Epoch & \multicolumn{3}{|c|}{5} \\
\hline Warmup Steps & \multicolumn{3}{|c|}{500} \\
\hline Learning Rate Schedule & \multicolumn{3}{|c|}{Linear} \\
\hline Label Smooth & \multirow[t]{4}{*}{0.1} & 0.1 & \multirow[t]{4}{*}{0.0} \\
\hline Learning Rate & \multicolumn{3}{|c|}{0.0002} \\
\hline Adaptation & \multicolumn{3}{|c|}{$r_{q}=r_{v}=4$} \\
\hline LoRA $\alpha$ & \multicolumn{3}{|c|}{32} \\
\hline & \multicolumn{3}{|c|}{Inference} \\
\hline Beam Size & \multicolumn{3}{|c|}{10} \\
\hline Length Penalty & \multirow[t]{2}{*}{0.9} & 0.8 & \multirow[t]{2}{*}{0.8} \\
\hline no repeat ngram size & \multicolumn{3}{|c|}{4} \\
\hline
\end{tabular}

**Table 11: The hyperparameters for GPT-2 LoRA on E2E, WebNLG and DART.**
\end{table}

WikiSQL (Zhong et al., 2017), 768 for MNLI (Williams et al., 2018), and 2048 for SAMSum (Gliwa et al., 2019). We tune learning rate for all method-dataset combinations. See Section D. 4 for more details on the hyperparameters used. For prefix-embedding tuning, we find the optimal $l_{p}$ and $l_{i}$ to be 256 and 8, respectively, totalling $3.2 M$ trainable parameters. We use $l_{p}=8$ and $l_{i}=8$ for prefix-layer tuning with $20.2 M$ trainable parameters to obtain the overall best performance. We present two parameter budgets for LoRA: $4.7 \mathrm{M}\left(r_{q}=r_{v}=1\right.$ or \(\left.r_{v}=2\right)\) and $37.7 \mathrm{M}\left(r_{q}=r_{v}=8\right.$ or $r_{q}=r_{k}=r_{v}=r_{o}=2$ ). We report the best validation performance from each run. The training hyperparameters used in our GPT-3 experiments are listed in Table 12

