# C Equivalently Converting PiSSA into LoRA

The advantage of PiSSA lies in its ability to significantly enhance training outcomes during the finetuning phase. After training, it allows for the direct sharing of the trained matrices $A$ and $B$. However, if we directly save $A, B$, users need to perform singular value decomposition on the original model to get $W^{\text {res }}$, which requires additional time. When employing fast singular value decomposition, there can be slight inaccuracies too. More importantly, such a way necessitates altering the parameters of the original model, which can be inconvenient when using multiple adapters, especially when some adapters might be disabled or activated. Therefore, we recommend converting the trained PiSSA module equivalently into a LoRA module, thereby eliminating the need to modify the original model's parameters during sharing and usage. In the initialization phase, PiSSA decomposes the original matrix into principal components and a residual matrix: $W=W^{\text {res }}+A B$. Upon completion of training, the model adjusts the weights as follows: $W+\Delta W=W^{\text {res }}+A^{\prime} B^{\prime}$. Thus, the modification of the model weights by PiSSA is given by:
$$
\begin{aligned}
\Delta W & =A^{\prime} B^{\prime}-A B \\
& =\underbrace{\left[\begin{array}{ll}
A^{\prime} & A
\end{array}\right]}_{\Delta A} \underbrace{\left[\begin{array}{c}
B^{\prime} \\
-B
\end{array}\right]}_{\Delta B}
\end{aligned}
$$
where $\Delta A \in \mathbb{R}^{m \times 2 r}$ and $\Delta B \in \mathbb{R}^{2 r \times n}$. Therefore, we can store and share the new adaptor $\Delta A$ and $\Delta B$ instead of $A^{\prime}, B^{\prime}$, which allows directly inserting the adaptor to the original matrix and avoids breaking $W$. Since $r$ is typically small, the twice storage overhead is still acceptable. This modification allows for plug-and-play usage without the need for singular value decomposition, saving time and avoiding computational errors associated with the SVD, without necessitating changes to the original model parameters.

