# 2 Task Vectors

For our purposes, a task is instantiated by a dataset and a loss function used for fine-tuning. Let $\theta_{\text {pre }} \in \mathbb{R}^{d}$ be the weights of a pre-trained model, and $\theta_{\mathrm{ft}}^{t} \in \mathbb{R}^{d}$ the corresponding weights after fine-tuning on task $t$. The task vector $\tau_{t} \in \mathbb{R}^{d}$ is given by the element-wise difference between $\theta_{\mathrm{ft}}^{t}$ and $\theta_{\text {pre }}$, i.e., $\tau_{t}=\theta_{\mathrm{ft}}^{t}-\theta_{\text {pre }}$. When the task is clear from context, we omit the identifier $t$, referring to the task vector simply as $\tau$.

Task vectors can be applied to any model parameters $\theta$ from the same architecture, via element-wise addition, with an optional scaling term $\lambda$, such that the resulting model has weights $\theta_{\text {new }}=\theta+\lambda \tau$. In our experiments, the scaling term is determined using held-out validation sets. Note that adding a single task vector to a pre-trained model with $\lambda=1$ results in the model fine-tuned on that task.

\footnotetext{
${ }^{2}$ Code available at https://github.com/mlfoundations/task_vectors.
}

Following Ilharco et al. [39], we focus on open-ended models, where it is possible to fine-tune on a downstream task without introducing new parameters (e.g., open-vocabulary image classifiers [78; 42; 76; 3] and text-to-text models [79; 77; 9; 38]). In cases where fine-tuning introduces new parameters (e.g., a new classification head), we could follow Matena \& Raffel [63] and merge only the shared weights, but this exploration is left for future work.

Editing models with task arithmetic. We focus on three arithmetic expressions over task vectors, as illustrated in Figure 1: negating a task vector, adding task vectors together, and combining task vectors to form analogies. All operations are applied element-wise to the weight vectors.

When negating a task vector $\tau$, applying the resulting vector $\tau_{\text {new }}=-\tau$ corresponds to extrapolating between the fine-tuned model and the pre-trained model. The resulting model is worse at the target task, with little change in performance on control tasks (Section 3). Adding two or more task vectors $\tau_{i}$ yields $\tau_{\text {new }}=\sum_{i} \tau_{i}$, and results in a multi-task model proficient in all tasks, sometimes even with gains over models fine-tuned on individual tasks (Section 4). Finally, when tasks $A, B, C$ and $D$ form an analogy in the form " $A$ is to $B$ as $C$ is to $D$ ", the task vector \(\tau_{\text {new }}=\tau_{C}+\left(\tau_{B}-\tau_{A}\right)\) improves performance on task $D$, even if there is little or no data for that task (Section 5).

For all operations, the model weights obtained by applying $\tau_{\text {new }}$ are given by $\theta_{\text {new }}=\theta+\lambda \tau_{\text {new }}$, where the scaling term $\lambda$ is determined using held-out validation sets.

