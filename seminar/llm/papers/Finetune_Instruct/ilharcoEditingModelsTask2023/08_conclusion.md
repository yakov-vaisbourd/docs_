# 8 Conclusion

In this paper we introduce a new paradigm for editing models based on arithmetic operations over task vectors. For various vision and NLP models, adding multiple specialized task vectors results in a single model that performs well on all target tasks, or even improves performance on a single task. Negating task vectors allows users to remove undesirable behaviors, e.g., toxic generations, or even forget specific tasks altogether, while retaining performance everywhere else. Finally, task analogies leverage existing data to improve performance on domains or subpopulations where data is scarce.

Arithmetic operations over task vectors only involve adding or subtracting model weights, and thus are efficient to compute, especially when compared to alternatives that involve additional finetuning. Thus, users can easily experiment with various model edits, recycling and transferring knowledge from large collections of publicly available fine-tuned models. Since these operations result in a single model of the same size, they incur no extra inference cost. Our code is available at https://github.com/mlfoundations/task_vectors.

