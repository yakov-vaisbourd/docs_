# 2 Background: Pretraining

Learning for most NLP research systems since 2018 consists of training in two stages. First, a neural language model (LM), often with millions of parameters, is trained on large unlabeled cor-

\footnotetext{
${ }^{1}$ https://github.com/allenai/ dont-stop-pretraining
}
pora. The word (or wordpiece; Wu et al. 2016) representations learned in the pretrained model are then reused in supervised training for a downstream task, with optional updates (fine-tuning) of the representations and network from the first stage.

One such pretrained LM is RoBERTA (Liu et al., 2019), which uses the same transformerbased architecture (Vaswani et al., 2017) as its predecessor, BERT (Devlin et al., 2019). It is trained with a masked language modeling objective (i.e., cross-entropy loss on predicting randomly masked tokens). The unlabeled pretraining corpus for ROBERTA contains over 160 GB of uncompressed raw text from different English-language corpora (see Appendix §A.1). RoBERTA attains better performance on an assortment of tasks than its predecessors, making it our baseline of choice.

Although RoBERTA's pretraining corpus is derived from multiple sources, it has not yet been established if these sources are diverse enough to generalize to most of the variation in the English language. In other words, we would like to understand what is out of RoBERTA's domain. Towards this end, we explore further adaptation by continued pretraining of this large LM into two categories of unlabeled data: (i) large corpora of domain-specific text (§3), and (ii) available unlabeled data associated with a given task (§4).

