# C RLOO's Connection to Contrastive Loss

Multiple other works in iterative fine-tuning (Zhao et al., 2023; Yuan et al., 2023),utilize a contrastive-style loss by up weighing the log-probabilities of positive samples and down-weighing the probabilities of negative samples, as determined by the reward model
$$
\mathcal{L}_{c}^{k=2}=-\log \pi\left(y_{+} \mid x\right)+\log \pi\left(y_{-} \mid x\right)
$$

We also have the corresponding loss with $k=2$ to Equation 2.3 as:
$$
\mathcal{L}_{\mathrm{RLOO}}^{k=2}=\frac{\left(R\left(y_{+}, x\right)-R\left(y_{-}, x\right)\right)}{2}\left(-\log \pi\left(y_{+} \mid x\right)+\log \pi\left(y_{-} \mid x\right)\right)
$$
It's clear that $\mathrm{RLOO}_{k=2}$ loss is exactly the contrastive loss but weighted by the difference between the absolute cores (the $\frac{1}{k}$ factor is merged into the learning-rate).

