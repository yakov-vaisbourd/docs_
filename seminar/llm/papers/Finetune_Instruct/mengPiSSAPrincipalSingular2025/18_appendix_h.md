# H Comparing the Quantization Error of QLoRA, LoftQ and QPiSSA

This section extends the discussion in Section 4 by providing a comprehensive comparison of the quantization errors associated with QLoRA, LoftQ, and QPiSSA. Using the "layers[0].self_attn.q_proj" of LLaMA 2-7B as an example, we illustrate the singular values of critical matrices during the quantization process with QLoRA, LoftQ, and PiSSA in Figure 10. A larger sum of the singular values (nuclear norm) of the error matrix indicates a greater quantization error.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-21_581_1387_577_369.jpg)

**Figure 10: Several important singular values for calculation the quantization error of QLoRA, LoftQ and PiSSA.**

The quantization error of QLoRA, which quantizes the base model to Normal Float 4-bit (NF4) and initializes $A$ and $B$ with Gaussian-Zero initialization, is:
$$
\text { Quantization Error of QLoRA }=\|W-(n f 4(W)+A B)\|_{*}=\|W-n f 4(W)\|_{*},
$$
As shown in Equation 13, QLoRA decomposes the original matrix in Figure 10a into the sum of a quantized matrix (Figure 10b) and an error matrix (Figure 10d). By comparing Figure 10a and Figure 10d, we can see that the magnitude of the error matrix is much smaller than that of the original matrix. Therefore, the benefit of preserving the principal components of the $W$ matrix with the adapter is greater than that of preserving the principal components of the error matrix with the adapter.

LoftQ [14], designed to preserve the principal components of the error matrix using the adapter, first performs singular value decomposition on the quantization error matrix of QLoRA:
$$
U^{e r r} S^{e r r} V^{e r r}=W-n f 4(W),
$$
then uses the larger singular values to initialize $A$ and $B$, thereby reducing the quantization error to:
$$
\text { Loft } Q^{\text {err }}=\|W-(n f 4(W)+A B)\| *=\left\|U_{[r:]}^{\text {err }} S_{[r:, r:]}^{\text {err }} V_{[r:]}^{\text {err }}\right\| *=\sum_{i=r}^{\min (m, n)} S_{[i, i]}^{\text {err }} .
$$
LoftQ eliminates only the largest $r$ singular values $S_{[: r]}^{\text {err }}$ (see Figure 10e) from the QLoRA error matrix (Figure 10d).

Our PiSSA, however, does not quantify the base model but the residual model:
$$
\text { Quantization Error of PiSSA }=\left\|W-\left(n f 4\left(W^{\text {res }}\right)+A B\right)\right\|_{*}=\left\|W^{\text {res }}-n f 4\left(W^{\text {res }}\right)\right\|_{*},
$$
where $A$ and $B$ are initialized following Equation 2 and 3. Since the residual model has removed the large-singular-value components, the value distribution of $W^{\text {res }}$ can be better fitted by a Student's t-distribution with higher degrees of freedom compared to $W$ (as can be seen in Figure 11) and thus quantizing $W^{\text {res }}$ results in lower error using 4-bit NormalFloat (shown in Figure 10f).

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-22_394_1382_244_369.jpg)

**Figure 11: Fitting the original matrix and the residual matrix using Student's t-distribution.**

