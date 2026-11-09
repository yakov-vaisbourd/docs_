# G The Residual Matrices having a Narrower Distribution

To intuitively compare the distribution differences between quantized original and residual models, in Figure 3. we took LLaMA 2-7B's first Query layer as an example to illustrate the distribution of $W$ and $W_{\text {res }}$. However, using only one layer of one model is not statistically significant. In this section, we applied PiSSA initialization to LLaMA-2-7B, Mistral-7B, Gemma-7B, and LLaMA-3-8B, and fit the values in every linear layer with Gaussian distribution and calculated their mu and sigma.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-20_283_1387_1624_369.jpg)

**Figure 9: Comparison of Loss and Ratio to the target A and target B for LoRA and PiSSA across the initial 5 steps.**

The results in Figure 9 show that the residual models' means are closer to 0, and the standard deviations are smaller after PiSSA initialization. Thus, $W^{\text {res }}$ indeed has a narrower distribution than W in a statistical sense. Nevertheless, the difference is not as large as that in the first layer after averaging all layers, which we suspect is because middle layers in a model tend to have more even eigenvalue distributions due to redundancy and insufficient training.

