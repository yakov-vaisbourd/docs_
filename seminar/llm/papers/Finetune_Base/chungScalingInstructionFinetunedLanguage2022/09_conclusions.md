# 9 Conclusions

In this paper we have extended instruction finetuning and trained Flan-PaLM by (1) scaling to a 540B-parameter language model, (2) scaling to 1.8K finetuning tasks, and (3) including chain-of-thought (CoT) data in finetuning. Experiments show that model performance substantially improved with both larger model size and more finetuning tasks. Moreover, whereas prior instruction finetuning methods degraded performance on CoT tasks, jointly finetuning with CoT data improved performance on all evaluations.

Flan-PaLM achieves state-of-the-art performance on several benchmarks, such as 75.2\% on five-shot MMLU. Flan-PaLM also has improved usability-for example, it can perform zero-shot reasoning without prompt engineering or few-shot exemplars. Additionally, we show that instruction finetuning is compatible with a range of model sizes, architectures, and pre-training objectives. To this end, we publicly release Flan-T5 models that outperform baseline T5 models by a large margin.

