# 1 Introduction

Fine-tuning large language models (LLMs) is a highly effective technique for boosting their capabilities in various tasks [1, 2, 3, 4], ensuring models to follow instructions [5, 6, 7], and instilling models with desirable behaviors while eliminating undesirable ones [8, 9]. However, the fine-tuning process for very large models is accompanied by prohibitive costs. For example, regular 16-bit fine-tuning of a LLaMA 65B parameter model requires over 780 GB of GPU memory [10], and the VRAM consumption for training GPT-3 175B reaches 1.2TB [11]. Consequently, various parameter-efficient fine-tuning (PEFT) [12, 13] methods have been proposed to reduce the number of parameters and memory usage required for fine-tuning. Due to the ability to maintain the performance of full finetuning without adding additional inference latency, Low-Rank Adaptation (LoRA) [11] has emerged as a popular PEFT method.

\footnotetext{
*Correspondence to: Muhan Zhang <muhan@pku.edu.cn>
}

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-02_548_1376_249_376.jpg)

**Figure 1: The comparison among Full Fine-tuning, training with LoRA, and PiSSA. In this visualization, blue modules represent parts of the model whose parameters are frozen during training, while orange modules indicate components that require updates. QLoRA quantizes the pretrained matrix in LoRA to 4-bit, whereas QPiSSA quantizes the residual matrix in PiSSA.**

\begin{table}

**Table 1: Comparison of similarities and differences between PiSSA and LoRA. In this table, bold highlights the model's primary component, while underline denotes the residual component.**
\begin{tabular}[t]{|l|l|l|}
\hline & LoRA & PiSSA \\
\hline Forward & \(Y=X(\mathbf{W}+\underline{\Delta W})=X(\mathbf{W}+\underline{A B})\) & \(Y=X\left(\underline{W^{\text {res }}}+\mathbf{W}^{\text {pri }}\right)=X\left(\underline{W^{\text {res }}}+\mathbf{A B}\right)\) \\
\hline Initialization & $$
\begin{gathered}
\underline{A} \sim \mathcal{N}\left(0, \sigma^{2}\right) \in \mathbb{R}^{m \times r} \\
\underline{B}=0 \in \mathbb{R}^{r \times n}
\end{gathered}
$$ & $$
\begin{gathered}
\mathbf{A}=U_{[:,: \mathbf{r}]} S_{[: \mathbf{r} ; \mathbf{r}]}^{1 / 2} \in \mathbb{R}^{m \times r} \\
\mathbf{B}=S_{[: \mathbf{r} ; \mathbf{r}]}^{1 / 2} V_{[:, \mathbf{r}]}^{\top} \in \mathbb{R}^{r \times n} \\
\frac{W^{\mathrm{res}}}{=}=U_{[:, r: \mathbf{r}]} S_{[\mathbf{r}:, r: \mathbf{r}]} V_{[::, r:]}^{\top} \in \mathbb{R}^{m \times n}
\end{gathered}
$$ \\
\hline Gradient & $$
\begin{gathered}
\frac{\partial L}{\partial A}=X^{\top}\left(\frac{\partial L}{\partial Y}\right) \underline{B}^{\top} \rightarrow \underline{0} \\
\frac{\partial L}{\partial B}=\underline{A^{\top}} X^{\top}\left(\frac{\partial L}{\partial Y}\right) \rightarrow \underline{\text { Random Direction }}
\end{gathered}
$$ & $$
\begin{aligned}
& \frac{\partial L}{\partial A}=X^{\top}\left(\frac{\partial L}{\partial Y}\right) \mathbf{B}^{\top} \rightarrow \text { Principal } \\
& \frac{\partial L}{\partial B}=\mathbf{A}^{\top} X^{\top}\left(\frac{\partial L}{\partial Y}\right) \rightarrow \text { Principal }
\end{aligned}
$$ \\
\hline Comparison & Fine-tunes noise while freezing $\mathbf{W}$. Slow convergence and underperformance. QLoRA cannot reduce quantization error. & Fine-tunes principal parts freezing $\underline{W^{\text {res }}}$. Fast convergence and better performance. QPiSSA can reduce quantization error. \\
\hline
\end{tabular}
\end{table}
LoRA [11] hypothesizes that the modifications to parameter matrices during fine-tuning exhibit low-rank properties. As depicted in Figure 1b, for a pre-trained weight matrix $W \in \mathbb{R}^{m \times n}$, LoRA substitutes the updates with a low-rank decomposition $\Delta W=A B$, where $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$, and the rank \(r \ll \min (m, n)\). For $Y=X W$, the modified forward pass is as follows:
$$
Y=X(W+\Delta W)=X(W+A B),
$$
A random Gaussian initialization is used for $A$ and zero for $B$, making $A B=0$ at the beginning of training, thereby the injection of adapters does not affect the model's output initially. LoRA avoids the need to compute gradients or maintain the optimizer states for the original matrix $W$, instead optimizing the injected, significantly smaller low-rank matrices $A, B$. Thus, it could reduce the number of trainable parameters by 10,000× and the GPU memory requirement by 3 × [11]. LoRA is capable of achieving comparable performance to full parameter fine-tuning. By integrating the quantization of pre-trained matrices $W$, LoRA also enables reducing the average memory requirements by 16× [10]. Meanwhile, the adapters can still utilize higher precision weights, thus, the quantization usually does not significantly degrade the performance of LoRA.
According to Equation 1 the gradients of A and B are \(\frac{\partial L}{\partial A}=X^{\top}\left(\frac{\partial L}{\partial Y}\right) B^{\top}\) and \(\frac{\partial L}{\partial B}=A^{\top} X^{\top}\left(\frac{\partial L}{\partial Y}\right)\). Compared to full fine-tuning, using LoRA initially does not change the output $Y$ for the same input $X$, so the magnitude and direction of gradient are primarily determined by the values of $A$ and $B$. Since $A$ and $B$ are initialized with Gaussian noise and zeros in LoRA, the gradients could be small
and uninformative for a long time, leading to slow convergence in the fine-tuning process. We also observe this phenomenon empirically, as LoRA often wastes much time around the initial point.

Our Principal Singular values and Singular vectors Adapter (PiSSA) diverges from LoRA and its successors by focusing not on approximating $\Delta W$, but $W$. We apply singular value decomposition (SVD) to matrix $W$. Based on the magnitude of the singular values, we partition $W$ into two parts: the principal low-rank matrix $W^{p r i}$, comprising a few largest singular values, and the residual matrix $W^{\text {res }}$, which possesses the remaining smaller singular values (with a larger quantity, representing a possible long-tail distribution). The principal matrix $W^{\text {pri }}$ can be represented by the product of $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$, where \(r \ll \min (m, n)\). As depicted in Figure $1 \mathrm{c}, A$ and $B$ are initialized based on the principal singular values and singular vectors and are trainable. Conversely, $W^{\text {res }}$ is initialized with the product of the residual singular values and singular vectors and remains frozen during fine-tuning. Since the principal singular vectors represent the directions in which the matrix $W$ has the most significant stretching or impact, by directly tuning these principal components, PiSSA is able to fit the training data faster and better (as demonstrated in Figure 2a). Moreover, the loss and gradient norm curves of PiSSA often demonstrate a similar trend to those of full parameter finetuning in our experiments (Figure 4), indicating that fine-tuning the principal components matches the behavior of fine-tuning the full matrix to some degree.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-03_520_1378_959_378.jpg)

**Figure 2: We illustrate the two key advantages of PiSSA: converging faster and better, and reducing quantization error. In the left figure, we use a toy example to show PiSSA's faster convergence, where we first train a two-layer MLP classifying odd numbers of MNIST, and then fine-tune the model on even numbers. PiSSA finds the right direction more quickly and achieves a lower loss with the same number of steps. In the right figure, PiSSA reduces quantization error more effectively than LoftQ [14], with an optional 5-iteration SVD for further error reduction, as detailed in Appendix E.**

Because the principal components $W^{p r i}$ are preserved in the adapter at full precision, an additional benefit of PiSSA is that when applying quantization to the frozen part $W^{\text {res }}$, we can significantly reduce the quantization error compared to QLoRA (which quantizes the entire $W$ ), as illustrated in Figure 2b. Therefore, PiSSA is even more compatible with quantization than LoRA, making it a superior plug-and-play substitution for LoRA.

Our paper makes several significant contributions:
\begin{itemize}
\item[-] We analyze the initial gradient magnitude and direction in LoRA, demonstrating that $A$ initially has a zero gradient and $B$ has a random gradient, which slows down convergence and may lead to convergence at suboptimal local minima.
\item[-] We propose PiSSA initialization, a novel method that approximates the optimization direction of full-parameter fine-tuning by adapting a model's principal components. To our knowledge, PiSSA is the first to apply SVD to the original model, using principal singular values and vectors to initialize the adapter for fine-tuning, while keeping the residual components frozen. Experiments show that PiSSA converges faster and outperforms LoRA.
\item[-] We combine PiSSA with NF4 quantization to propose QPiSSA, which reduces quantization error by about 20\% compared to QLoRA, while maintaining the fast convergence and high performance of PiSSA.
\end{itemize}

