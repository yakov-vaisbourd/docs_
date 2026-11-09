# A Frequently asked questions

# A. 1 Are instruction-finetuned models better for single-task finetuning?

In this paper we showed that instruction-finetuned models are better for unseen tasks in a few-shot prompted setting. We did not evaluate how instruction-finetuned models are at single-task finetuning in this paper, but several related work may provide useful hints.
\begin{itemize}
\item[-] Aghajanyan et al. (2021) showed that a stage of massively-multitask finetuning (without instructions) improves performance on a range of downstream finetuning tasks for models such as RoBERTa (Liu et al., 2019) and BART (Lewis et al., 2019).
\item[-] Wei et al. (2021) showed that instruction-finetuned models are better at prompt tuning (Lester et al., 2021) for unseen tasks.
\end{itemize}

We leave this investigation for our specific models as future work. We also note that models such as PaLM and U-PaLM 540B should ideally retain multi-task abilities, as it would be inefficient for such a large model to only perform a single task.

# A. 2 Does using CoT prompting for evaluation always improve performance?

Whether CoT improves performance is highly task-dependent. As noted in Wei et al. (2021); Kojima et al. (2022), CoT helps over direct answer-only prompting the most when tasks require multiple steps of reasoning and when used with a large-enough language model. Such tasks include challenging math problems or symbolic manipulation problems that require multiple steps.

In Table 5, CoT does not improve performance on MMLU, compared with direct prompting. This is likely because MMLU consists mostly of knowledge recall tasks that do not require reasoning. However, when we combine CoT prompting with self-consistency, this can improve performance over no CoT, as shown in Table 4. Moreover, the effect of model scale on CoT can also be observed in Table 5. For BBH, models that are not instruction finetuned only benefit from CoT if they are 62B or larger. This is consistent with prior observations that CoT is an emergent ability of scaling language models (Wei et al., 2022a). For instruction-finetuned models on BBH, using CoT only helps for the Flan-PaLM 540B, Flan-cont-PaLM 62B, and U-PaLM 540B.

# A. 3 Does instruction finetuning improve performance more or less for larger models?

In this paper, we showed that, across a range of model sizes and architectures, instruction finetuning improves performance compared with no instruction finetuning. However, whether the effect of finetuning increases with scale is confounded by the fact that the baseline of no instruction finetuning also gets better with scale. For example, one could compare the 8B and 540B models as shown in Table 3. Instruction finetuning improves the normalized average score across four datasets for 8B model by 15.5\% and the 540B model by $9.4 \%$. However, the baseline is also higher for the 540B model, so it could also be relevant to compute the relative reduction in error rate, which was $18.4 \%$ for the 540 B model and $16.6 \%$ for the 8 B model.

# A. 4 How many examples were used as part of the CoT mixture in finetuning?

There are a total of 74,730 examples in the CoT mixture. Here are the number of training examples per dataset: AQuA (2,715), CREAK (6,910), ECQA (7,110), ESNLI (36,170), GSM8K (7,470), QASC (1,080), QED (5,145), Sensemaking (6,070), StrategyQA (2,060).

