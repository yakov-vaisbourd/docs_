# K Supplementary Experiments on Various Ranks

# K. 1 Quantization Error for More Type of Layers

Figure 7a only shows the reduction ratio of quantization error for "q_proj" layers. In Figure 14, we present the error reduction ratios for the remaining types of linear layers under different ranks.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-24_631_1378_520_373.jpg)

**Figure 14: Comparison of quantization errors in QLoRA, LoftQ, and PiSSA across k_proj, v_proj, o_proj and gate_proj, up_proj, down_proj layers.**

From Figure 14 it can be observed that under different ranks, the reduction ratio of quantization error for various linear layers in LLaMA-2-7B, including "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", and "down_proj" layers, is consistently lower with PiSSA compared to LotfQ.

# K. 2 Evaluation Performance for More Model on Various Ranks

Section 5.5 only validated the effectiveness of LLaMA-2-7B. In Figure 15, we also present the comparative results of Mistral-7B-v0.1, and Gemma-7B under different ranks.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-25_1414_1385_443_371.jpg)

**Figure 15: Fine-tuning LLaMA 2-7B, Mistral-7B-v0.1, and Gemma-7B on the MetaMathQA dataset: A comparison of full parameter fine-tuning (indicated by a dashed line), LoRA (in blue), and PiSSA (in orange).**

From Figure 15. PiSSA uses fewer trainable parameters compared to LoRA while achieving or even surpassing full-parameter fine-tuning on LLaMA-2-7B and Mistral-7B. Remarkably, on Gemma-7B, PiSSA exceeds full-parameter fine-tuning performance even at rank $=1$. However, as the rank increases to 128, the performance of PiSSA begins to decline, indicating that PiSSA over-parameterizes earlier than LoRA. This over-parameterization phenomenon does not occur on LLaMA-2-7B, suggesting that increasing the rank further might enable PiSSA to achieve even higher performance on LLaMA-2-7B.

# K. 3 More Training Loss and Grad Norm under Various Ranks

In Figure 16 and 17, we examining the loss and gradient norm during the training process of PiSSA and LoRA on LLaMA 2-7B, Mistral-7B-v0.1, and Gemma-7B using different ranks.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-26_683_1378_434_373.jpg)

**Figure 16: Comparison of training loss for LLaMA-2-7B, Mistral-7B, and Gemma-7B, organized into three rows, using LoRA and PISSA across ranks $2^{i**, i \in[0,7]$, organized into eight columns.}

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-26_681_1376_1292_376.jpg)

**Figure 17: Comparison of grad norm for LLaMA-2-7B, Mistral-7B, and Gemma-7B, organized into three rows, using LoRA and PISSA across ranks $2^{i**, i \in[0,7]$, organized into eight columns.}

From Figure 16. PiSSA consistently shows a faster initial loss reduction compared to LoRA across various ranks. Additionally, the final loss remains lower than that of LoRA. This advantage is particularly pronounced when the rank is smaller. From Figure 17, the gradient norm of PiSSA remains consistently higher than that of LoRA throughout the training process, indicating its efficient fitting of the training data. A closer look at the first few steps of LoRA's gradient norm reveals a trend of rising and then falling. According to Section 3 LoRA's gradients are initially close to zero, leading to very slow model updates. This requires several steps to elevate LoRA's weights to a higher level before subsequent updates. This phenomenon validates our assertion that LoRA wastes some training steps and therefore converges more slowly. It demonstrates the robustness of the faster convergence property of PiSSA across various ranks.

