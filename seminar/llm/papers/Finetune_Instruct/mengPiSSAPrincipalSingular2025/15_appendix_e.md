# E Reducing Quantization Error through Multiple Iteration of SVD

Table 8 provides a supplementary explanation of the results in Table 4. When number of iterations $T>1$, LoftQ uses an $N$-bit quantized weight $Q \in \mathbb{R}_{N}^{m \times n}$ and low-rank approximations $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{n \times r}$ to minimize the following objective by alternating between quantization and singular value decomposition:
$$
\min _{Q, A, B}\left\|W-\left(Q+A B^{\top}\right)\right\|_{F},
$$
where $\|\cdot\|_{F}$ denotes the Frobenius norm, $A$ and $B$ are set to zero. Inspired by LoftQ, our QPiSSA $T$-iter alternately minimize the following objective:
$$
\min _{W_{r e s}, A, B}\left\|W-\left(n f 4\left(W_{r e s}\right)+A B^{\top}\right)\right\|_{F},
$$
where $A$ and $B$ are initialized by the principal singular values and singular vectors. The process is summarized in Algorithm 1 :
\begin{lstlisting}[mathescape=true]
Algorithm 1 QPiSSA- $T$-iters, $T \geq 2$
input Pre-trained weight $W$, target rank $r$, 4-bit quantization function \(n f 4(\cdot)\), alternating step $T$
Initialize \(A_{0}, B_{0} \leftarrow \operatorname{SVD}(W)\) by (2) and (3)
Initialize residual weight $W_{\text {res }} \leftarrow W-A_{0} B_{0}^{\top}$
for $\mathrm{t}=2$ to $T$ do
Update \(A_{t}, B_{t} \leftarrow \operatorname{SVD}\left(W-n f 4\left(W_{\text {res }}\right)\right)\) by (2) and (3)
Update residual weight $W_{\text {res }} \leftarrow W-A_{t-1} B_{t-1}^{\dagger}$
end for
output \(n f 4\left(W_{\text {res }}\right), A_{T}, B_{T}\)
\end{lstlisting}

\begin{table}

**Table 8: PiSSA reduces more quantization error on various ranks and number of iterations.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline & Method & Rank & niter & Q & K & V & O & Gate & Up & Down & AVG \\
\hline \multirow{5}{*}{LLaMA -2-7B} & QLoRA & - & - & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline & loftQ & 128 & 1 & 8.1 & 8.1 & 7.2 & 7.3 & 5.3 & 5.1 & 5.1 & 6.6 \\
\hline & PiSSA & 128 & 1 & 19.0 & 18.1 & 8.9 & 8.9 & 8.2 & 5.9 & 6.0 & 10.7 \\
\hline & loftQ & 128 & 5 & 16.5 & 16.5 & 15.9 & 16.0 & 12.4 & 12.4 & 12.3 & 14.6 \\
\hline & PiSSA & 128 & 5 & 27.9 & 27.2 & 18.7 & 18.6 & 15.8 & 13.6 & 13.6 & 19.4 \\
\hline \multirow{9}{*}{LLaMA -3-8B} & QLoRA & - & - & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline & LoftQ & 64 & 1 & 4.3 & 11.0 & 9.9 & 3.9 & 2.7 & 2.5 & 2.6 & 5.3 \\
\hline & PiSSA & 64 & 1 & 11.3 & 16.4 & 8.8 & 6.3 & 4.5 & 2.9 & 3.3 & 7.7 \\
\hline & loftQ & 64 & 5 & 10.1 & 18.8 & 18.2 & 9.9 & 7.1 & 7.1 & 7.1 & 11.2 \\
\hline & PiSSA & 64 & 5 & 17.1 & 27.3 & 19.5 & 12.1 & 8.9 & 7.2 & 7.6 & 14.3 \\
\hline & loftQ & 128 & 1 & 8.2 & 20.7 & 18.8 & 7.5 & 5.2 & 4.8 & 4.9 & 10.0 \\
\hline & PiSSA & 128 & 1 & 17.1 & 26.5 & 10.7 & 10.7 & 7.0 & 5.0 & 5.6 & 11.8 \\
\hline & loftQ & 128 & 5 & 16.4 & 29.8 & 28.8 & 16.1 & 11.9 & 11.7 & 11.7 & 18.1 \\
\hline & PiSSA & 128 & 5 & 26.3 & 41.7 & 32.3 & 20.1 & 14.4 & 12.5 & 12.9 & 22.9 \\
\hline \multirow{7}{*}{LLaMA -3-70B} & QLoRA & - & - & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\hline & LoftQ & 64 & 1 & 2.4 & 11.6 & 9.2 & 1.9 & 1.8 & 1.7 & 1.3 & 4.3 \\
\hline & PiSSA & 64 & 1 & 12.3 & 25.0 & 9.0 & 4.1 & 4.2 & 3.2 & 2.2 & 8.6 \\
\hline & LoftQ & 64 & 5 & 6.1 & 17.8 & 17.0 & 6.0 & 4.3 & 4.4 & 4.2 & 8.5 \\
\hline & PiSSA & 64 & 5 & 15.7 & 34.2 & 18.9 & 7.5 & 6.7 & 5.7 & 4.7 & 13.4 \\
\hline & PiSSA & 128 & 1 & 17.7 & 36.6 & 15.7 & 6.7 & 5.8 & 4.5 & 3.8 & 13.0 \\
\hline & PiSSA & 128 & 5 & 23.2 & 49.0 & 30.5 & 12.5 & 10.1 & 8.8 & 8.2 & 20.3 \\
\hline
\end{tabular}
\end{table}
According to Table 8, multiple iterations can significantly reduce quantization error. For instance, using QPiSSA-r64 with 5-iter on LLaMA-3-8B reduces the quantization error nearly twice as much as with 1-iter. In the main paper, we used 5 iterations in Section 5.3 and Section 5.4, while 1 iteration was used in Section 5.5.

