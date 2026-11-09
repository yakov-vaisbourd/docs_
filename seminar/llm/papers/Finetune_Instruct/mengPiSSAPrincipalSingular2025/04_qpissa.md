# 4 QPiSSA: An Extension Method with Lower Quantization Error

Quantization divides the value range of a matrix into several continuous regions, and maps all values falling inside a region into the same "quantized" value. It is an effective technique to reduce the memory consumption of forward propagation. At the same time, LoRA greatly reduces the backward memory requirement, making it highly suitable to use LoRA and quantization together, where the base model is quantized for memory-efficient forward propagation, and the LoRA adaptors are kept in full precision for accurate backward parameter updates. One representative previous work, QLoRA, quantizes the base model to Normal Float 4-bit (NF4) and initializes the full-precision $A$ and $B$ with Gaussian-Zero initialization. Therefore, the overall error is given by:
$$
\text { Quantization Error of QLoRA }=\|W-(n f 4(W)+A B)\|_{*}=\|W-n f 4(W)\|_{*},
$$
where $\|M\|_{*}$ denotes the nuclear norm (also known as the trace norm [48]), defined as:
$$
\|M\|_{*}=\operatorname{trace}\left(\sqrt{M^{*} M}\right)=\sum_{i=1}^{\min \{m, n\}} \sigma_{i}(M),
$$
where \(\sigma_{i}(M)\) is the $i^{\text {th }}$ singular value of $M$. As we can see, the quantization error of QLoRA is the same as that of directly quantizing the base model. Our QPiSSA, however, does not quantize the base model but the residual model. Therefore, its error is given by:
$$
\text { Quantization Error of QPiSSA }=\left\|W-\left(n f 4\left(W^{r e s}\right)+A B\right)\right\|_{*}=\left\|W^{r e s}-n f 4\left(W^{r e s}\right)\right\|_{*} .
$$
Since the residual model has removed the large-singular-value components, $W^{r e s}$ has a narrower distribution than that of $W$, as can be seen in Figures 3a and 3b (comparing the singular value distributions of $W$ and $W^{\text {res }}$ ), as well as Figures 3c and 3f (comparing the value distributions of $W$ and $W^{\text {res }}$ ), which is beneficial for reducing the quantization error. Moreover, given that NF4 is optimized for data with a normal distribution, as discussed by Dettmers et al. [10], we fit the values of $W$ and $W^{\text {res }}$ to a Gaussian distribution respectively. As illustrated in Figures 3c and 3f, $W^{\text {res }}$ aligns more closely with a Gaussian distribution and exhibits a smaller standard deviation, making it more suitable for applying NF4 than $W$. Both the above lead QPiSSA to achieve a significantly lower quantization error than QLoRA, shown in Figures 3d and 3e
Besides the advantage of reducing quantization error, QPiSSA's gradient direction is similar to that of PiSSA, resulting in significantly better fine-tuning performance compared to QLoRA.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-06_579_1387_247_369.jpg)

**Figure 3: Visualizations of LLaMA 2-7B's "layers[0].self_attn.q_proj" matrix, with distributions for the full model shown in Appendix G. Figures (a), (b), (d), and (e) display the singular values of \(W, W^{\text {res **}, W-n f 4(W)\), and \(W^{\text {res }}-n f 4\left(W^{\text {res }}\right)\), respectively. Figures (c) and (f) show the data distributions of $W$ and $W^{\text {res }}$.}

