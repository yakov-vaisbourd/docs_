# 3 PiSSA: Principal Singular Values and Singular Vectors Adaptation

This section formally presents our Principal Singular values and Singular vectors Adaptation method. PiSSA computes the singular value decomposition (SVD) of matrices $W$ within the self-attention and multilayer perceptron (MLP) layers. The (economy size) SVD of a matrix $W \in \mathbb{R}^{m \times n}$ is given by $W=U S V^{\top}$, where \(U \in \mathbb{R}^{m \times \min (m, n)}, V \in \mathbb{R}^{n \times \min (m, n)}\) are the singular vectors with orthonormal columns, and $V^{\top}$ is the transpose of \(V . S=\operatorname{diag}(\mathbf{s}) \in \mathbb{R}^{\min (m, n) \times \min (m, n)}\), where the operation diag \((\mathbf{s})\) transforms $\mathbf{s}$ to a diagonal matrix $S$, and \(\mathbf{s} \in \mathbb{R}_{\geq 0}^{\min (m, n)}\) represents the singular values arranged in descending order. When the top $r$ singular values $\mathbf{s}_{[r, r]}$ are significantly larger than the remaining singular values $\mathbf{s}_{[r:]}$, we denote the intrinsic rank of $W$ as $r$. Consequently, $S$, along with $U$ and $V$, can be divided into two groups: the principal singular values and vectors- $\left\{U_{[:,: r]}, S_{[: r,: r]}, V_{[:,: r]}\right\}$, and the residual singular values and vectors- $\left\{U_{[:, r:]}, S_{[r:, r:]}, V_{[:, r:]}\right\}$, where the matrix slicing notations are the same as those in PyTorch and $[: r]$ denotes the first $r$ dimensions. The principal singular values and vectors are utilized to initialize the injected adapter consisting of $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$ :
$$
\begin{aligned}
& A=U_{[:,: r]} S_{[: r,: r]}^{1 / 2} \in \mathbb{R}^{m \times r}, \\
& B=S_{[: r,: r]}^{1 / 2} V_{[:,: r]}^{\top} \in \mathbb{R}^{r \times n} .
\end{aligned}
$$
The residual singular values and vectors are used to build the residual matrix which is frozen during fine-tuning:
$$
W^{r e s}=U_{[:, r:]} S_{[r:, r:]} V_{[:, r:]}^{\top} \in \mathbb{R}^{m \times n} .
$$
As indicated by Equation 5, the integration of $A B$ with the residual matrix also preserves the full capability of the pre-trained model in the beginning of fine-tuning:
$$
Y=X W=X\left(W^{r e s}+W^{p r i}\right)=X\left(W^{r e s}+A B\right) .
$$

Similar to LoRA, the gradients of $A$ and $B$ are also given by \(\frac{\partial L}{\partial A}=X^{\top}\left(\frac{\partial L}{\partial Y}\right) B^{\top}\) and $\frac{\partial L}{\partial B}=$ \(A^{\top} X^{\top}\left(\frac{\partial L}{\partial Y}\right)\). Since elements of $\mathbf{s}_{[: r]} \gg$ elements of $\mathbf{s}_{[r:]}$, the trainable adapter $W^{p r i}=A B$ contains the most essential directions of $W$. In the ideal case, training $A B$ mirrors the process of fine-tuning the entire model despite using fewer parameters. The ability to directly fine-tune the most essential part of a model enables PiSSA to converge faster and better. In contrast, LoRA initializes the adapters $A$ and $B$ with Gaussian noise and zeros while keeping $W$ frozen. Consequently, the gradients are small or in random directions during the early stages of fine-tuning, possibly introducing much waste of gradient descent steps. Moreover, an inferior initialization might lead to suboptimal local minimum, resulting in worse generalization performance.

Since PiSSA shares the identical architecture with LoRA, it inherits most of LoRA's benefits. These include but are not limited to the capability of fine-tuning a model with a reduced number of trainable parameters, quantizing the residual model to decrease memory consumption during forward propagation in training, and easy deployment. The adapter's straightforward linear structure facilitates the integration of trainable matrices with the pre-trained weights upon deployment, thereby maintaining the original inference speed of a fully fine-tuned model. Employing the Fast SVD technique [47] allowed PiSSA to finish initialization in several seconds (Appendix B), which is a negligible cost.

For storage efficiency, we can choose not to store the dense parameter matrix $\Delta W$, but to store the low-rank matrices, $\Delta A$ and $\Delta B$ instead. As shown in Appendix C, leveraging solely the $\Delta A$ and $\Delta B$ facilitates their seamless integration with the original pre-trained models. Finally, one pre-trained model can accommodate multiple $\Delta A, \Delta B$, fine-tuned by diverse PiSSA or LoRA procedures, which enables fast adaptation of the pre-trained model to different downstream applications.

