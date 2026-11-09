# M Comparison of Initial Gradient Subspaces

To compare the gradient subspaces of PiSSA and LoRA, we conducted two additional experiments to validate our analysis.

First, we trained LLaMA-3-8B on the MetaMath dataset five times, initializing LoRA with different random seeds while using the same batch of 128 training examples to compute LoRA's gradients. After performing dimensionality reduction to two dimensions, the results are presented in Table 11.

\begin{table}

**Table 11: Ablation results for LoRA and PiSSA across different seeds.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline & Method & Seed 0 & Seed 1 & Seed 2 & Seed 3 & Seed 4 \\
\hline \multirow[b]{2}{*}{grad_A} & LoRA & [0,0] & [0,0] & [0,0] & [0,0] & [0,0] \\
\hline & PiSSA & [0,1] & [0,1] & [0,1] & [0,1] & [0,1] \\
\hline \multirow[b]{2}{*}{grad_B} & LoRA & [-0.99, 0.12] & [0.95, 0.31] & [0.46, -0.89] & [0.24, 0.97] & [0.04, -0.99] \\
\hline & PiSSA & [1,0] & [1,0] & [1,0] & [1,0] & [1,0] \\
\hline
\end{tabular}
\end{table}
We observe that the gradient of matrix $A$ remains consistently zero, while the gradient direction of matrix $B$ varies across initializations. This behavior arises because matrix $A$ 's gradient depends on matrix $B$, which in LoRA is initialized to zero, resulting in a zero gradient for $A$. In contrast, matrix $B$ is initialized from a Gaussian distribution, leading to variation in its gradient direction across different seeds. In comparison, PiSSA's gradient direction remains consistent across all five training runs, as it solely depends on the original model and the training data. This experiment highlights the stability of PiSSA's optimization trajectory relative to LoRA's more variable directionality.

Next, we quantitatively compared the effect of updating along the principal singular value direction versus a "random" direction during the early stages of fine-tuning. We trained LLaMA-3-8B on the MetaMathQA dataset using both PiSSA and LoRA, saving the parameters and gradients from the first 50 iterations. At the 50th step, the loss values for LoRA and PiSSA were 0.3677 and 0.2899, respectively. Using the parameters from the 50th step as the target point, we evaluated the movement in the first five steps relative to the target, computing how much progress was made towards the final point. We then divided this progress by the total target distance to obtain a ratio. These ratios are shown in Figure 18.

![Figure](./mengPiSSAPrincipalSingular2025_images/8fb8a5f5-fa7a-4a4b-9a29-5fcafda4b529-28_462_1367_1588_362.jpg)

**Figure 18: Comparison of Loss and Ratio to the target A and target B for LoRA and PiSSA across the initial 5 steps.**

The results reveal that after just five updates, PiSSA reduced the loss from 0.8884 to 0.3346 , while LoRA's loss reduction was more modest, dropping to only 0.5538. This demonstrates the advantage of updating along the principal singular value direction, which PiSSA leverages, leading to faster convergence. Further, in the first step, matrix $A$ in LoRA exhibited a zero gradient and therefore did not update. Over the next four steps, it moved only $15.94 \%$ towards the target direction. Similarly, matrix $B$ in LoRA consistently moved less towards the target endpoint compared to PiSSA.

