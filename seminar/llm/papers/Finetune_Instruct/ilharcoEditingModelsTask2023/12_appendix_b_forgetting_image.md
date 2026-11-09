# B Forgetting image classification tasks

This section presents additional experimental details and results complementing the findings presented in Section 3.1, showcasing the effect of negating task vectors from image classification tasks.

# B. 1 Experimental details

We follow the same procedure from [39] when fine-tune CLIP models [78]. Namely, we fine-tune for 2000 iterations with a batch size of 128, learning rate 1e-5 and a cosine annealing learning rate schedule with 200 warm-up steps and the AdamW optimizer [58; 75], with weight decay 0.1. When fine-tuning, we freeze the weights of the classification layer output by CLIP's text encoder, so that we do not introduce additional learnable parameters, as in [39]. As shown by [39], freezing the classification layer does not harm accuracy. After fine-tuning, we evaluate scaling coefficients $\lambda \in\{0.0,0.05,0.1, \cdots, 1.0\}$, choosing the highest value such that the resulting model still retains at least 95\% of the accuracy of the pre-trained model on the control task.

# B. 2 Baselines

We contrast our results with two baselines, fine-tuning with gradient ascent as in Golatkar et al. [34]; Tarun et al. [90], and against using a random vector of the same magnitude as the task vector on a layer-by-layer basis.

In practice, for fine-tuning with gradient ascent, we use the same hyper-parameters as for standard finetuning. However, instead of optimizing to minimize the cross-entropy loss \(\ell=\mathbb{E}_{x, y \in \mathcal{D}}\left[-\log f(x)_{y}\right]\), we optimize to minimize its negative value, \(\ell_{\text {neg }}=-\ell=\mathbb{E}_{x, y \in \mathcal{D}}\left[\log f(x)_{y}\right]\), where $x, y$ are samples in the dataset $\mathcal{D}$ and \(f(x)_{y}\) is the probability assigned by the model $f$ that the inputs $x$ belong to label $y$. This is equivalent to performing gradient ascent on $\ell$.

For the random vector baseline, we first compute the different between the parameters of the pretrained and fine-tuned models for each layer \(L, \tau^{(L)}=\theta_{\mathrm{ft}}^{(L)}-\theta_{\mathrm{pre}}^{(L)}\). Then, we draw a new vector \(\tau_{\text {rand }}^{(L)} \sim \mathcal{N}(0, I)\) where each element is drawn from a normal distribution with mean 0 and variance 1. We then scale this vector so it has the same magnitude as \(\tau^{(L)}\), resulting in \(\tau_{\text {scaled }}^{(L)}=\tau_{\text {rand }}^{(L)} \frac{\left\|\tau^{(L)}\right\|}{\left\|\tau_{\text {rand }}^{(L)}\right\|}\). Finally, we concatenate all the vectors \(\tau_{\text {scaled }}^{(L)}\) for all layers to form a new vector withe the same dimensionality as the model parameters $\theta$, which is used in the same way as task vectors.

# B. 3 Breakdown per task

Tables 5, 6 and 7 show a breakdown of accuracy for the eight tasks and the three CLIP models we examine.

\footnotetext{
\({ }^{5} \theta_{\text {pre }}+0.5\left(\tau_{1}+\tau_{2}\right)=\theta_{\text {pre }}+0.5\left(\left(\theta_{1}-\theta_{\text {pre }}\right)+\left(\theta_{2}-\theta_{\text {pre }}\right)\right)=0.5\left(\theta_{1}+\theta_{2}\right)\).
}

\begin{table}

**Table 5: Forgetting via negation on image classification tasks. Results are shown for a CLIP ViT-L/14 model [78], reporting accuracy on both the target (T) and control (C) tasks.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Method} & \multicolumn{2}{|c|}{Cars} & \multicolumn{2}{|c|}{DTD} & \multicolumn{2}{|c|}{EuroSAT} & \multicolumn{2}{|c|}{GTSRB} & \multicolumn{2}{|c|}{MNIST} & \multicolumn{2}{|c|}{RESISC45} & \multicolumn{2}{|c|}{SUN397} & \multicolumn{2}{|c|}{SVHN} \\
\hline & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ \\
\hline Pre-trained & 77.8 & 75.5 & 55.4 & 75.5 & 60.2 & 75.5 & 50.6 & 75.5 & 76.4 & 75.5 & 71.0 & 75.5 & 68.3 & 75.5 & 58.6 & 75.5 \\
\hline Fine-tuned & 92.8 & 73.1 & 83.7 & 72.3 & 99.2 & 70.5 & 99.3 & 73.1 & 99.8 & 72.9 & 96.9 & 73.8 & 82.4 & 72.7 & 98.0 & 72.6 \\
\hline Neg. gradients & 0.00 & 4.82 & 2.13 & 0.10 & 9.26 & 1.07 & 1.19 & 0.07 & 9.80 & 67.0 & 2.14 & 0.07 & 0.25 & 0.00 & 6.70 & 57.2 \\
\hline Random vector & 72.0 & 73.3 & 52.1 & 72.2 & 59.7 & 73.5 & 43.4 & 72.5 & 74.8 & 72.8 & 70.8 & 73.0 & 66.9 & 72.7 & 47.1 & 72.9 \\
\hline Neg. task vector & 32.0 & 72.4 & 26.7 & 72.2 & 7.33 & 73.3 & 6.45 & 72.2 & 2.69 & 74.9 & 19.7 & 72.9 & 50.8 & 72.6 & 6.71 & 72.7 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 6: Forgetting via negation on image classification tasks. Results are shown for a CLIP ViT-B/16 model [78], reporting accuracy on both the target (T) and control (C) tasks.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{Method} & \multicolumn{2}{|c|}{Cars} & \multicolumn{2}{|c|}{DTD} & \multicolumn{2}{|c|}{EuroSAT} & \multicolumn{2}{|c|}{GTSRB} & \multicolumn{2}{|c|}{MNIST} & \multicolumn{2}{|c|}{RESISC45} & \multicolumn{2}{|c|}{SUN397} & \multicolumn{2}{|c|}{SVHN} \\
\hline & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ \\
\hline Pre-trained & 64.6 & 68.3 & 44.9 & 68.3 & 53.9 & 68.3 & 43.4 & 68.3 & 51.6 & 68.3 & 65.8 & 68.3 & 65.5 & 68.3 & 52.0 & 68.3 \\
\hline Fine-tuned & 87.0 & 61.9 & 82.3 & 57.5 & 99.1 & 56.0 & 99.0 & 54.7 & 99.7 & 55.2 & 96.4 & 62.2 & 79.0 & 61.7 & 97.7 & 56.8 \\
\hline Neg. gradients & 0.36 & 0.11 & 2.13 & 0.09 & 9.26 & 0.14 & 0.71 & 0.10 & 0.04 & 1.20 & 2.60 & 0.10 & 0.25 & 0.00 & 0.08 & 3.69 \\
\hline Rand. task vector & 61.0 & 65.6 & 43.9 & 66.3 & 51.7 & 66.2 & 43.1 & 65.0 & 51.6 & 68.3 & 63.6 & 65.6 & 63.7 & 65.2 & 46.2 & 65.5 \\
\hline Neg. task vector & 30.8 & 65.4 & 26.5 & 65.6 & 12.3 & 65.8 & 9.53 & 65.8 & 9.55 & 65.4 & 26.5 & 65.1 & 48.6 & 65.1 & 6.43 & 65.4 \\
\hline
\end{tabular}
\end{table}

We observe qualitatively similar results in all cases. Similarly to what is observed in [39], we also see that results improve with scale: on average, the largest model, ViT-L/14, achieves lower accuracy on the target tasks, compared to the smaller models.

# B. 4 Additional visualizations

In Figure 9, we show how accuracy on the target and control tasks vary as we change the scaling coefficients $\lambda$, both for the task vector obtained by fine-tuning on the target task and for a random vector of the same magnitude.

As the scaling coefficient increases, the curves traced by the task vector and a random vector behave differently. For task vectors, performance on the target tasks ( $y$-axis) initially decreases faster than performance on the control task ( $x$-axis), so there exists models with high accuracy on the control task but low accuracy on the target task. In contrast, such points do not exist in the curves traced by random vectors, which move more linearly towards the origin. In practice, this means forgetting is effective for task vectors obtained by fine-tuning, but not for random vectors.

\begin{table}

**Table 7: Forgetting via negation on image classification tasks. Results are shown for a CLIP ViT-B/32 model [78], reporting accuracy on both the target (T) and control (C) tasks.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{Method} & \multicolumn{2}{|c|}{Cars} & \multicolumn{2}{|c|}{DTD} & \multicolumn{2}{|c|}{EuroSAT} & \multicolumn{2}{|c|}{GTSRB} & \multicolumn{2}{|c|}{MNIST} & \multicolumn{2}{|c|}{RESISC45} & \multicolumn{2}{|c|}{SUN397} & \multicolumn{2}{|c|}{SVHN} \\
\hline & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ & T↓ & C↑ \\
\hline Pre-trained & 59.6 & 63.4 & 44.1 & 63.4 & 45.9 & 63.4 & 32.5 & 63.4 & 48.7 & 63.4 & 60.7 & 63.4 & 63.2 & 63.4 & 31.5 & 63.4 \\
\hline Fine-tuned & 79.2 & 55.2 & 78.7 & 49.3 & 98.6 & 47.2 & 98.5 & 39.1 & 99.6 & 42.5 & 95.0 & 53.2 & 75.1 & 54.6 & 97.2 & 44.7 \\
\hline Neg. gradients & 0.01 & 0.11 & 2.13 & 0.10 & 9.26 & 0.10 & 1.19 & 0.07 & 0.00 & 1.22 & 2.60 & 0.10 & 0.25 & 0.01 & 6.38 & 0.29 \\
\hline Rand. task vector & 54.1 & 60.9 & 39.9 & 61.5 & 45.8 & 63.4 & 27.9 & 60.7 & 48.3 & 63.4 & 57.1 & 60.9 & 61.3 & 60.5 & 31.2 & 60.7 \\
\hline Neg. task vector & 36.0 & 61.1 & 27.8 & 60.2 & 13.6 & 61.3 & 8.13 & 61.4 & 16.7 & 60.7 & 31.7 & 61.0 & 50.7 & 60.5 & 7.65 & 61.0 \\
\hline
\end{tabular}
\end{table}

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-21_475_1375_286_374.jpg)

**Figure 9: Comparison between task vectors and random vectors for forgetting image classification tasks.**

\begin{table}

**Table 8: The effect of semantic overlap with the control task in forgetting experiments on image classification tasks. Results are shown for a CLIP ViT-L/14 model, reporting accuracy both on the target task and control task (Ctrl, ImageNet).**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Method} & \multicolumn{4}{|c|}{Without filtering} & \multicolumn{4}{|c|}{With filtering} \\
\hline & Cars (↓) & Ctrl (↑) & SUN397 (↓) & Ctrl (↑) & Cars (↓) & Ctrl (↑) & SUN397 (↓) & Ctrl (↑) \\
\hline Pre-trained & 77.8 & 75.5 & 68.3 & 75.5 & 77.8 & 75.5 & 68.3 & 76.1 \\
\hline Fine-tuned & 92.8 & 73.1 & 82.4 & 72.7 & 92.8 & 73.3 & 82.4 & 73.1 \\
\hline Neg. task vector & 32.0 & 72.4 & 50.8 & 72.6 & 32.0 & 72.5 & 48.1 & 72.4 \\
\hline
\end{tabular}
\end{table}

# B. 5 The effect of class overlap

In Tables 5, 7, 6, we observe that the tasks where forgetting via task vectors is least effective are tasks where the distribution of images is closer to ImageNet, SUN397 [101], a scene understanding dataset with classes such as "church" and "tower", and Stanford Cars [47], a dataset with with many car categories such as "2012 Tesla Model S" or "2012 BMW M3 coupe". One reasonable hypothesis is that forgetting is less effective for those tasks due to the overlap with the images from the control tasks.

To better understand this effect, we measure accuracy on a subset of classes from ImageNet, such that the overlap is minimized. Concretely, we exclude nodes from the WordNet hierarchy from which the ImageNet classes are based. ${ }^{6}$ For the Cars dataset, we exclude the all subnodes under the node "wheeled vehicle" (e.g., "minivan", "jeep", "limousine"). For SUN397, we exclude all subnodes under the nodes "structure" and "geological formation". As shown in Table 8, we do not observe large differences after filtering.

# B. 6 Interpolating with a model fine-tuned with gradient ascent

One baseline explored in the experiments is fine-tuning with gradient ascent, as explored in Golatkar et al. [34]; Tarun et al. [90]. Our results show that this strategy is effective at reducing the accuracy on treatment tasks, but also substantially deteriorates accuracy on the control task, which is undesirable.

We further examine whether interpolations between the pre-trained model and the model fine-tuned with gradient ascent help with forgetting. Our results, shown in Figure 10, indicate that interpolations greatly mitigate the low accuracy on the control task of the fine-tuned model, leading to even better accuracy trade-offs than the solutions obtained by extrapolation with standard fine-tuning.

\footnotetext{
${ }^{6}$ A visualization is available at https://observablehq.com/@mbostock/ imagenet-hierarchy
}

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-22_471_1371_287_376.jpg)

**Figure 10: Comparison with interpolations between the pre-trained model and models fine-tuned with gradient ascent.**

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-22_405_1369_946_376.jpg)

**Figure 11: Correlation between the gain in accuracy from fine-tuning and the drop in accuracy when subtracting the corresponding task vector for image classification tasks.**

# B. 7 When negating task vectors works best

We observe a positive correlation between the gain in accuracy from fine-tuning and the drop in accuracy when subtracting the corresponding task vector, both in comparison with the pre-trained model (Figure 11). We speculate that the reason for this correlation is that when the gains from fine-tuning are small, the task vector provides a less clear direction of improvement, and the opposite direction thus provides a less clear direction of performance deterioration. In the extreme case where fine-tuning does not improve accuracy, it would be surprising if the corresponding task vector is useful.

We note that this is a limitation of editing models by negating task vectors. When models already strongly exhibit the behavior we wish to remove, it is harder to do so with this technique. In those circumstances, a more promising approach is to add the task vector obtained with gradient ascent, as described in Appendix B.6.

# B. 8 Additional tasks

In addition to the tasks explored in Section 4.1, we study two other tasks, OCR and person identification.

For OCR, we use the synthetic dataset from Ilharco et al. [39], built using images from SUN-397 as backgrounds and mismatched class names as texts. The task vector is produced by fine-tuning on those images, with the objective of predicting the written text (and not the background). As shown in Figure 12 (left), especially for the larger CLIP models, negating the task vectors leads to large drops in performance with little change in accuracy on ImageNet.

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-23_393_1317_307_410.jpg)

**Figure 12: Forgetting by negating task vectors on additional vision tasks, OCR and person identification.**

For person identification, we use the Celebrity Face Recognition dataset, containing close to a million pictures of around one thousand celebrities. ${ }^{7}$ We split the data into a training, validation and test set with proportions 0.8, 0.1 and 0.1. Results are shown in Figure 12 (right). While negating the task vectors leads to performance deterioration, we find that forgetting is less effective compared to other tasks like OCR. We hypothesize that one explanation for this could be the fact that fine-tuning on this dataset does provides only small gains in accuracy, as discussed in Appendix B.7.

