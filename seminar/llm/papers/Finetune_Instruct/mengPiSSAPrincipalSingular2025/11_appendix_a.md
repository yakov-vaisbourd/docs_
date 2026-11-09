# A Enhancing PiSSA with LoRA Improvement Methods

AdaLoRA introduces three improvements over LoRA:
\begin{itemize}
\item[-] Trainable parameters in AdaLoRA are changed to $A, B$, and $E . A$ and $B$ are Gaussianinitialized, and $E$ is a zero-initialized $r$-dimensional vector, making \(\operatorname{Adiag}(E) B=\Delta W\), similar to singular value decomposition.
\item[-] A regularization loss $\left|A A^{T}-I\right|+\left|B^{T} B-I\right|$ is used to make $A$ and $B$ gradually orthogonal during training, resembling the SVD of $\Delta W$.
\item[-] An initial large rank is set, and less important E values are gradually masked during training, resulting in different final ranks for each layer, achieving better performance with the same number of parameters.
\end{itemize}

Despite the extensive use of SVD terms, AdaLoRA does not perform actual SVD on any matrix. In the PEFT domain, terms like low-rank decomposition, and singular value decomposition often appear. They generally refer to products of low-dimensional matrices approximating an ideal $\Delta W$ without actual matrix decomposition. To our knowledge, PiSSA is the first to perform SVD on the original model, fine-tuning the principal component while keeping the residual model frozen.

PiSSA and AdaLoRA represent different improvements to LoRA, making them combinable. Therefore, we additionally improved PiSSA based on AdaLoRA's three innovations:
\begin{itemize}
\item[-] After extracting the principal singular values and vectors of $W$, we use $S$ as an independent trainable vector instead of multiplying it into $U$ and $V$.
\item[-] Since PiSSA's $U$ and $V$ are orthogonal at the beginning, maintaining their orthogonality through orthogonal regularization is very easy.
\item[-] Although AdaLoRA claims to dynamically reduce the number of trainable parameters, the initially large number of parameters is not truly pruned, resulting in more parameters being updated during actual training. Therefore, we did not use this improvement.
\end{itemize}

DoRA adds a learnable magnitude module to LoRA, normalizing $W+A B$ at each update step and multiplying its by the magnitude module. This allows $A, B$ to learn the direction and the magnitude module to learn the magnitude of $\Delta W$. While this approach can improve fine-tuning performance, normalizing $W+A B$ at each step results in slower fine-tuning speeds. In contrast, PiSSA only changes LoRA's initialization method, matching LoRA in training speed and converging faster, thereby reducing training costs.

\begin{table}

**Table 5: GSM8K accuracy for LoRA and PiSSA when combined with LoRA improvement methods.**
\begin{tabular}[t]{|l|l|l|l|}
\hline Model & Method & LoRA+ & PiSSA+ \\
\hline \multirow{3}{*}{LLaMA-3-8B} & Vanilla & $71.01 \pm 0.199$ & 76.75±0.036 \\
\hline & DoRA & $72.38 \pm 0.189$ & 77.51±0.257 \\
\hline & AdaLoRA & $72.31 \pm 0.202$ & 78.59±0.199 \\
\hline
\end{tabular}
\end{table}
PiSSA, with its intrinsic principal singular values and orthogonal singular vectors, is very suitable for combination with AdaLoRA. According to Table 5 The performance of the improved PiSSA surpasses all the other methods including PiSSA. From lines 1, and 2 of the table, it is evident that the performance of PiSSA combined with DoRA significantly surpasses that of DoRA alone and also exceeds the performance of PiSSA alone. Taking into account the combination experiments of PiSSA with AdaLoRA, it can be inferred that PiSSA benefits from the enhancement techniques of LoRA, demonstrating the potential of PiSSA when integrated with other methods.

