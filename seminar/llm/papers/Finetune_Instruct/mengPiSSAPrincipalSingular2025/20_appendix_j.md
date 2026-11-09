# J Evaluating PiSSA on Mixtral and Gemma with More Training Steps

This is the supplement for Section 5.2. We applied PiSSA, LoRA, and full parameter fine-tuning on the full MetaMathQA-395K dataset using Mistral-7B and Gemma-7B models, training for 3 epochs. Figures 12 and 13 display the training loss, gradient norm, and evaluation accuracy on GSM8K.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-23_387_1391_484_364.jpg)

**Figure 12: Fine-tuning Mistral-7B-v0.1 on the MetaMathQA-395K dataset for 3 epochs: A comparison of full parameter fine-tuning (indicated by a dashed line), LoRA (in blue), and PiSSA (in orange).**

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-23_380_1391_1023_364.jpg)

**Figure 13: Fine-tuning Gemma-7B on the MetaMathQA-395K dataset for 3 epochs: A comparison of full parameter fine-tuning (indicated by a dashed line), LoRA (in blue), and PiSSA (in orange).**

As shown in Figure 12a and 13a, the loss for full parameter fine-tuning decreases sharply with each epoch, indicating overfitting to the training data. Notably, during the entire first epoch, the loss for full parameter fine-tuning on Mistral and Gemma is significantly higher than for LoRA and PiSSA, suggesting that full parameter fine-tuning has weaker generalization capabilities compared to LoRA and PiSSA on Mistral-7B and Gemma-7B models. The gradient norm for the first epoch in Figure 13b fluctuates dramatically with each step, further indicating instability in the training process for full parameter fine-tuning. Consequently, as illustrated in Figures 12c and 13c, the performance of full parameter fine-tuning is markedly inferior to that of LoRA and PiSSA. These experiments demonstrate that using parameter-efficient fine-tuning can prevent the over-fitting issue caused by over-parameters.

