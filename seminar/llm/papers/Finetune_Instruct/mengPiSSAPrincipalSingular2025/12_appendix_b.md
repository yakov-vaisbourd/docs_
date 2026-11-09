# B Fast Singular Value Decomposition

In order to speed up the decomposition of the pre-trained matrix $W$, we adopted the algorithm proposed by Halko et.al [47] (denoted as Fast SVD), which introduces randomness to achieve an approximate matrix decomposition. We compare the initialization time, error, and training loss between SVD and Fast SVD, with the results shown in Table 6. Initialization time refers to the computation time taken to decompose the pre-trained parameter matrix $W$, measured in seconds. Initialization error indicates the magnitude of the discrepancy introduced by Fast SVD compared to SVD after decomposing the matrix. Specifically, the error is the sum of the absolute differences between the matrices decomposed by original SVD and Fast SVD. For the error, we report the results of the self-attention module in the table. Loss refers to the loss value at the end of training. In Fast SVD, the parameter niter refers to the number of subspace iterations to conduct. A larger niter leads to increased decomposition time but results in smaller decomposition error. The symbol $\infty$ represents the experimental results with the SVD method.

\begin{table}

**Table 6: Comparation between SVD and Fast SVD in terms of initialization time, error and training loss.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{Metric} & \multirow{2}{*}{Niter} & \multicolumn{8}{|c|}{Rank} \\
\hline & & 1 & 2 & 4 & 8 & 16 & 32 & 64 & 128 \\
\hline \multirow{6}{*}{Initialize Time} & 1 & 5.05 & 8.75 & 5.07 & 8.42 & 5.55 & 8.47 & 6.80 & 11.89 \\
\hline & 2 & 4.38 & 4.71 & 4.79 & 4.84 & 5.06 & 5.79 & 7.70 & 16.75 \\
\hline & 4 & 5.16 & 4.73 & 5.09 & 5.16 & 5.60 & 7.01 & 7.90 & 11.41 \\
\hline & 8 & 4.72 & 5.11 & 5.14 & 5.40 & 5.94 & 7.80 & 10.09 & 14.81 \\
\hline & 16 & 6.24 & 6.57 & 6.80 & 7.04 & 7.66 & 9.99 & 14.59 & 22.67 \\
\hline & $\infty$ & 434.92 & 434.15 & 434.30 & 435.42 & 435.25 & 437.22 & 434.48 & 435.84 \\
\hline \multirow{6}{*}{Initialize Error} & 1 & 1.30E-3 & $1.33 \mathrm{E}-3$ & $1.55 \mathrm{E}-3$ & $1.9 \mathrm{E}-3$ & $1.98 \mathrm{E}-3$ & $1.97 \mathrm{E}-3$ & $2.00 \mathrm{E}-3$ & $1.93 \mathrm{E}-3$ \\
\hline & 2 & 5.84E-4 & 1.25E-3 & $1.45 \mathrm{E}-3$ & $1.43 \mathrm{E}-3$ & $1.48 \mathrm{E}-3$ & 1.55E-3 & $1.48 \mathrm{E}-3$ & $1.33 \mathrm{E}-3$ \\
\hline & 4 & 6.01E-4 & 8.75E-4 & 6.75E-4 & 1.10E-3 & $1.05 \mathrm{E}-3$ & 1.03E-3 & 1.08E-3 & 9.75E-4 \\
\hline & 8 & 1.26E-4 & 2.34E-4 & 5.25E-4 & 7.25E-4 & 5.75E-4 & 8.25E-4 & 8.25E-4 & 7.75E-4 \\
\hline & 16 & 7.93E-5 & 2.25E-4 & 1.28E-4 & 6.50E-4 & 4.25E-4 & 6.50E-4 & 6.00E-4 & 4.75E-4 \\
\hline & $\infty$ & - & - & - & - & - & - & - & - \\
\hline \multirow{6}{*}{Training Loss} & 1 & 0.3629 & 0.3420 & 0.3237 & 0.3044 & 0.2855 & 0.2657 & 0.2468 & 0.2301 \\
\hline & 2 & 0.3467 & 0.3337 & 0.3172 & 0.2984 & 0.2795 & 0.2610 & 0.2435 & 0.2282 \\
\hline & 4 & 0.3445 & 0.3294 & 0.3134 & 0.2958 & 0.2761 & 0.2581 & 0.2414 & 0.2271 \\
\hline & 8 & 0.3425 & 0.3279 & 0.3122 & 0.2950 & 0.2753 & 0.2571 & 0.2406 & 0.2267 \\
\hline & 16 & 0.3413 & 0.3275 & 0.3116 & 0.2946 & 0.2762 & 0.2565 & 0.2405 & 0.2266 \\
\hline & $\infty$ & 0.3412 & 0.3269 & 0.3116 & 0.2945 & 0.2762 & 0.2564 & 0.2403 & 0.2264 \\
\hline
\end{tabular}
\end{table}
It can be observed that the computation time of the SVD is tens of times that of Fast SVD. In addition, SVD exhibits consistently high time consumption with minimal variation as the rank increases, while Fast SVD, although experiencing a slight increase in computation time with higher ranks, remains significantly lower compared to SVD throughout. As the rank increases, the initialization error initially rises gradually, with a slight decrease observed when the rank reaches 128. And at the same rank, increasing the niter in Fast SVD leads to a gradual reduction in error. For training loss, we observed that as the rank increases, the training loss decreases gradually. At the same rank, with the increase of niter, the training loss of models initialized based on Fast SVD approaches that of models initialized based on SVD.

