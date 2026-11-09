# 5 Why does In-Context Learning work?

Section 4 shows that the ground truth input-label mapping in the demonstrations has little impact to performance gains from in-context learning. This section further examines what other aspects of the demonstrations lead to good performance of incontext learning.

We identify four aspects of the demonstrations \(\left(x_{1}, y_{1}\right) \ldots\left(x_{k}, y_{k}\right)\) that potentially provide learning signal (depicted in Figure 7).
\begin{itemize}
\item[1.] The input-label mapping, i.e., whether each input $x_{i}$ is paired with a correct label $y_{i}$.
\item[2.] The distribution of the input text, i.e., the underlying distribution that $x_{1} \ldots x_{k}$ are from.
\item[3.] The label space, i.e., the space covered by $y_{1} \ldots y_{k}$.
\item[4.] The format-specifically, the use of inputlabel pairing as the format.
\end{itemize}

As Section 4 does for the input-label mapping, we design a series of variants of the demonstrations that quantify the impact of each aspect in isolation (Section 5.1-5.3). We then additionally discuss the trend of the models meta-trained with an in-context learning objective (Section 5.4). For all experiments, models are evaluated on five classification

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-06_475_1434_229_351.jpg)

**Figure 8: Impact of the distribution of the inputs. Evaluated in classification (top) and multi-choice (bottom). The impact of the distribution of the input text can be measured by comparing □ and ■. The gap is substantial, with an exception in Direct MetaICL (discussion in Section 5.1).**

and four multi-choice datasets as in Section 4.2. See Appendix B and Table 4 for implementation details and example demonstrations, respectively.

## 5.1 Impact of the distribution of the input text

We experiment with OOD demonstrations which include out-of-distribution (OOD) text instead of the inputs from unlabeled training data. Specifically, a set of $k$ sentences $\left\{x_{i, \text { rand }}\right\}_{i=1}^{k}$ are randomly sampled from an external corpus, and replace $x_{1} \ldots x_{k}$ in the demonstrations. This variant assesses the impact of the distribution of the input text, while keeping the label space and the format of the demonstrations.

Results. Figure 8 shows that using out-ofdistribution inputs instead of the inputs from the training data significantly drops the performance when Channel MetaICL, Direct GPT-J or Channel GPT-J are used, both in classification and multichoice, by 3-16\% in absolute. In the case of Direct GPT-J in multi-choice, it is even significantly worse than no demonstrations. Direct MetaICL is an exception, which we think is the effect of meta-training (discussion in Section 5.4).

This suggests that in-distribution inputs in the demonstrations substantially contribute to performance gains. This is likely because conditioning on the in-distribution text makes the task closer to language modeling, since the LM always conditioned on the in-distribution text during training.

## 5.2 Impact of the label space

We also experiment with demonstrations w/ random English words that use random English words as labels for all $k$ pairs. Specifically, we sample a random subset of English words $\mathcal{C}_{\text {rand }}$ where $\left|\mathcal{C}_{\text {rand }}\right|=|\mathcal{C}|$, and randomly pair $\tilde{y}_{i} \in \mathcal{C}_{\text {rand }}$ with $x_{i}$. This variant assesses the impact of the label space, while keeping the distribution of the input text and the format of the demonstrations.

Results. Based on Figure 9, direct models and channel models exhibit different patterns. With direct models, the performance gap between using random labels within the label space and using random English words is significant, ranging between 5-16\% absolute. This indicates that conditioning on the label space significantly contributes to performance gains. This is true even for multi-choice tasks where there is no fixed set of labels-we hypothesize that multi-choice tasks still do have a particular distribution of the choices (e.g., objects like "Bolts" or "Screws" in the OpenBookQA dataset) that the model uses.

On the other hand, removing the output space does not lead to significant drop in the channel models: there is 0-2\% drop in absolute, or sometimes even an increase. We hypothesize that this is because the channel models only condition on the labels, and thus are not benefiting from knowing the label space. This is in contrast to direct models which must generate the correct labels.

## 5.3 Impact of input-label pairing

Section 5.1 and 5.2 focus on variants which keep the format of the demonstrations as much as possible. This section explores variants that change the format. While there are many aspects of the format, we make minimal modifications to remove the pairings of inputs to labels. Specifically, we evaluate demonstrations with no labels where the LM is

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-07_475_1432_229_351.jpg)

**Figure 9: Impact of the label space. Evaluated in classification (top) and multi-choice (bottom). The impact of the label space can be measured by comparing ■ and ■ . The gap is significant in the direct models but not in the channel models (discussion in Section 5.2).**

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-07_471_1428_891_353.jpg)

**Figure 10: Impact of the format, i.e., the use of the input-label pairs. Evaluated in classification (top) and multichoice (bottom). Variants of demonstrations without keeping the format (! and ) are overall not better than no demonstrations (■). Keeping the format is especially significant when it is possible to achieve substantial gains with the label space but without the inputs (■ vs. ■ in Direct MetaICL), or with the input distribution but without the labels (■ vs. ■ in Channel MetaICL and Channel GPT-J). More discussion in Section 5.3.**

conditioned on the concatenation of $x_{1} \ldots x_{k}$, and demonstrations with labels only where the LM is conditioned on the concatenation of $y_{1} \ldots y_{k}$. These ablations provide the no-format counterparts of the 'demonstrations with random English words' and 'demonstrations with OOD inputs', respectively.

Results. Based on Figure 10, removing the format is close to or worse than no demonstrations, indicating the importance of the format. This is likely because conditioning on a sequence of inputlabel pairs triggers the model to mimic the overall format and complete the new example as expected when the test input is given.

More interestingly, keeping the format plays a significant role in retaining a large portion of performance gains by only using the inputs or only using the labels. For instance, with Direct MetaICL, it is possible to retain 95\% and 82\% of improvements from in-context learning (demonstrations with gold labels) by simply sampling random sentences from a corpus and randomly pairing them with the label set ( □ in Figure 10) in classification and multi-choice, respectively. Similarly, with the channel models, it is possible to retain 82\%, 87\%, 86\% and 75\% of improvements from in-context learning by simply pairing each input from the unlabeled training data with a random English word ( in Figure 10) in MetaICL classification, GPT-J classification, MetaICL multi-choice and GPT-J multi-choice, respectively. For all of these cases, removing inputs instead of using OOD inputs, or removing labels instead of using random English words is significantly worse, indicating that keeping the format of the input-label pairs is key.

## 5.4 Impact of meta-training

Different from other models, MetaICL is trained with an in-context learning objective, in line with
recent work that uses multi-task training on a large collection of supervised datasets (called metatraining) for generalization to new tasks (Aghajanyan et al., 2021; Khashabi et al., 2020; Wei et al., 2022a; Sanh et al., 2022). We aim to better understand the role of this meta-training in relation with our findings by closely examining the result of MetaICL. In particular, we observe that the patterns we see so far are significantly more evident with MetaICL than with other models. For instance, the ground truth input-label mapping matters even less, and keeping the format of the demonstrations matters even more. There is nearly zero influence of the input-label mapping and the input distribution in Direct MetaICL, and the input-label mapping and the output space in Channel MetaICL.

Based on this observation, we hypothesize that meta-training encourages the model to exclusively exploit simpler aspects of the demonstrations and to ignore others. This is based on our intuition that (1) the input-label mapping is likely harder to exploit, (2) the format is likely easier to exploit, and (3) the space of the text that the model is trained to generate is likely easier to exploit than the space of the text that the model conditions on. ${ }^{8}$

