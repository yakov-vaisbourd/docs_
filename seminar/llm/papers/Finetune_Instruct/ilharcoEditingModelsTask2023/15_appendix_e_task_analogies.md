# E Task analogies

Similarly in the experiments where multiple models are added together, we use a single scaling coefficient for the vector resulting from the task arithmetic, $\lambda \in\{0,0.1, \cdots, 1.0\}$. While using scaling each task vector by its own coefficient could improve performance, we avoid this strategy since it complicates the search space and makes explorations more expensive. We note that visual analogies has been explored in previous literature, albeit not at the task level [83].

\begin{table}

**Table 11: Learning via analogy. By leveraging vectors from related tasks, we can improve accuracy on four new target tasks without any training data, and with little change on control settings. Results are shown for the CLIP models [77], additional details are provided in Appendix E.2.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Method} & \multicolumn{2}{|c|}{Queens} & \multicolumn{2}{|c|}{Kings} & \multicolumn{2}{|c|}{Woman} & \multicolumn{2}{|c|}{Men} \\
\hline & Target & Control & Target & Control & Target & Control & Target & Control \\
\hline ViT-B/32 & 0.00 & 63.4 & 0.00 & 63.4 & 0.00 & 63.4 & 0.00 & 63.4 \\
\hline + task vectors & 42.0 & 62.4 & 30.0 & 62.4 & 69.4 & 62.5 & 58.0 & 62.6 \\
\hline ViT-B/16 & 0.00 & 68.3 & 0.00 & 68.3 & 0.00 & 68.3 & 0.00 & 68.3 \\
\hline + task vectors & 66.0 & 67.5 & 94.0 & 67.4 & 87.8 & 67.5 & 62.0 & 67.6 \\
\hline ViT-L/14 & 0.00 & 75.5 & 0.00 & 75.5 & 0.00 & 75.5 & 0.00 & 75.5 \\
\hline + task vectors & 100 & 74.7 & 100 & 74.5 & 100 & 74.6 & 96.0 & 74.6 \\
\hline
\end{tabular}
\end{table}

# E. 1 Domain generalization

Here, we use task analogies to improve performance on tasks where no labeled data is available. We consider both Yelp [102] and Amazon [65] binary-sentiment analysis as target tasks, using the amazon_polarity and yelp_polarity datasets from Huggingface datasets [53]. As detailed in 5, given target and auxiliary tasks, we construct task vectors using the relationship \(\hat{\tau}_{\text {target; sent }}=\tau_{\text {target; } \text { lm }}+\left(\tau_{\text {auxiliary; sent }}-\tau_{\text {auxiliary; } \text { lm }}\right)\). We apply an two scaling coefficients: one on the auxilary sentiment task vector, and another to the language modeling task vectors.

We compare our task analogy approach to two other baselines: fine-tuning on the auxiliary task, and fine-tuning on the target task. The latter represents an performance upper-bound, assuming we have labeled data for the target task.

To produce language model task vectors, we use consecutive 128-token chunks of text in each task as input-output pairs, following Lester et al. [52]. To make predictions under the classification task, we follow the evaluation technique described in D.7.

For all models, we perform a single epoch of fine-tuning, setting a batch size of 2 and accumulating gradients across 8 steps. We use AdamW and a linear learning rate schedule. We set the maximum input and output sequence length to be 128. For each model scale, we perform a grid search over learning rates in $\{1 \mathrm{e}-5,3 \mathrm{e}-5,5 \mathrm{e}-5,8 \mathrm{e}-4\}$, choosing the fastest learning rate that avoids divergence.

To construct a task vector using the task analogy, we perform a grid over the values $\lambda \in$ \{0.0, 0.1, ..., 1.0\} for each scaling coefficient. Regardless of scale, we found that giving higher weight to the auxiliary sentiment task vector produced higher accuracy. For the smallest model, we saw better performance when applying a lower-valued coefficient to the language modeling task vectors. For the largest model, applying larger coefficients to the language modeling task vectors produced better performance. This trend may be reflective of the finding in 3.2 that task forgetting is more effective with larger models.

# E. 2 Kings and Queens

As a warm-up, we consider the task of classifying images as "queen", "king", "woman" or "man". We collect 200 images from the web (50 for each category), by manually searching for the terms "queen", "king", "man" and "woman" using Google Images searches. We present samples in Figure 19.

Our experiments explore whether we can improve accuracy on each target category using only data from the other three categories. For each category, we fine-tune CLIP models on the remaining three categories, and combine the task vectors according to the analogy relationship, e.g., $\hat{\tau}_{\text {king }}=$ \(\tau_{\text {queen }}+\left(\tau_{\text {man }}-\tau_{\text {woman }}\right)\). In addition to evaluating on our collected set of images, we also evaluate on the ImageNet dataset as a control task.

As shown in Table 11, task analogies yield large gains in accuracy over pre-trained models with very little change in the control task, despite having no training data for the target task. Similar to Ilharco et al. [39]; Ramasesh et al. [81], we find that results improve with model scale.

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-30_657_1081_279_520.jpg)

**Figure 19: Samples from the dataset we collect for classifying queens, kings, women and men, as described in Section E.2.**

Fine-tuning CLIP models is done as described in Section B.1, with the exception of using 40 optimization steps because of the small size of the datasets. When fine-tuning, we use only the images from one category (e.g., "king"), and a set of 1001 classes from which to choose, composed by the 1000 classes in ImageNet, and a new class. Since CLIP has already seen many images of queens, kings, men and women in its pre-training, we use a new category name for the new class when fine-tuning, in order to simulate learning a new concept. More concretely, we use the class name "something", which makes the accuracy of zero-shot models close or equal to zero. When evaluating, we also contrast between all 1001 options, including all ImageNet classes. This is done both for our target task, and for ImageNet, where we add an additional option. Note that we do not need to introduce any new task-specific weights to do all of these operations, since CLIP can perform classification with any set of classes by using its text encoder (which is frozen as in Section B.1).

# E. 3 Subpopulations

We fine-tune CLIP models on each of the subpopulations with the same hyper-parameters as described in Section B.1, using 500 optimization steps regardless of the number of samples. For the few-shot experiments, we sample the same number of samples for every class in the task. For convenience, let ImageNet-A ${ }^{11}$ and ImageNet-B represent the two subpopulations from ImageNet, and Sketches-A and Sketches-B represent the two subpopulations from the sketches dataset from Eitz et al. [22]. Note that ImageNet-A and Sketches-A share the same classes, and the same is true for ImageNet-B and Sketches-B. We present samples in Figure 20.

Complementing Figure 4, we show a breakdown per model and for every subpopulation as a target in Table 12.

Independent scaling coefficients. In addition to our standard procedure of using a single scaling coefficient for the vector resulting from the arithmetic operations, we explore having independent scaling coefficients for each task vector in the expression. In other words, we explore the models $\theta_{\text {new }}=\theta+\lambda_{C} \tau_{C}+\lambda_{B} \tau_{B}-\lambda_{A} \tau_{A}$ for various scaling coefficients $\lambda_{A}, \lambda_{B}, \lambda_{C} \in\{0,0.1, \cdots, 1.0\}$. On average, the optimal scaling coefficients were $\lambda_{B}^{\star}=\lambda_{C}^{\star}=0.32$ and $\lambda_{A}^{\star}=0.28$. Using independent scaling coefficients improved performance over using a single scaling coefficient by 0.7 percentage points on average, but also required substantially more evaluations to be made ( $10^{3}$ instead of 10).

\footnotetext{
${ }^{11}$ Not to be confused with the adversarial dataset from Hendrycks et al. [37].
}

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-31_629_1100_406_496.jpg)

**Figure 20: Samples from the datasets used for the analogies with subpopulations experiments, as described in Section E.3.**

\begin{table}

**Table 12: Learning by analogy on subpopulations. Results are shown for multiple CLIP models, as detailed in Section E.3.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline Model & Samples per class & Task vectors & Sketches-A & Sketches-B & Accuracy ImageNet-A & ImageNet-B & Average \\
\hline \multirow{8}{*}{ViT-B/32} & 0 & × & 0.712 & 0.677 & 0.861 & 0.923 & 0.793 \\
\hline & 0 & ✓ & 0.782 & 0.758 & 0.861 & 0.926 & 0.832 \\
\hline & 1 & × & 0.754 & 0.758 & 0.868 & 0.919 & 0.825 \\
\hline & 1 & ✓ & 0.782 & 0.766 & 0.866 & 0.922 & 0.834 \\
\hline & 2 & × & 0.768 & 0.778 & 0.868 & 0.919 & 0.833 \\
\hline & 2 & ✓ & 0.786 & 0.800 & 0.867 & 0.922 & 0.844 \\
\hline & 4 & × & 0.810 & 0.780 & 0.871 & 0.926 & 0.847 \\
\hline & 4 & ✓ & 0.802 & 0.796 & 0.871 & 0.927 & 0.849 \\
\hline \multirow{8}{*}{ViT-B/16} & 0 & × & 0.716 & 0.732 & 0.885 & 0.946 & 0.820 \\
\hline & 0 & ✓ & 0.794 & 0.794 & 0.889 & 0.953 & 0.858 \\
\hline & 1 & × & 0.758 & 0.812 & 0.894 & 0.948 & 0.853 \\
\hline & 1 & ✓ & 0.796 & 0.804 & 0.897 & 0.957 & 0.863 \\
\hline & 2 & × & 0.792 & 0.817 & 0.897 & 0.951 & 0.865 \\
\hline & 2 & ✓ & 0.804 & 0.829 & 0.899 & 0.956 & 0.872 \\
\hline & 4 & × & 0.815 & 0.812 & 0.904 & 0.952 & 0.871 \\
\hline & 4 & ✓ & 0.831 & 0.825 & 0.904 & 0.953 & 0.878 \\
\hline \multirow{8}{*}{ViT-L/14} & 0 & × & 0.823 & 0.831 & 0.913 & 0.962 & 0.882 \\
\hline & 0 & ✓ & 0.879 & 0.861 & 0.922 & 0.968 & 0.908 \\
\hline & 1 & × & 0.845 & 0.863 & 0.923 & 0.971 & 0.900 \\
\hline & 1 & ✓ & 0.879 & 0.863 & 0.930 & 0.973 & 0.911 \\
\hline & 2 & × & 0.865 & 0.881 & 0.925 & 0.973 & 0.911 \\
\hline & 2 & ✓ & 0.875 & 0.881 & 0.932 & 0.975 & 0.916 \\
\hline & 4 & × & 0.875 & 0.883 & 0.934 & 0.973 & 0.916 \\
\hline & 4 & ✓ & 0.903 & 0.887 & 0.941 & 0.975 & 0.927 \\
\hline
\end{tabular}
\end{table}
