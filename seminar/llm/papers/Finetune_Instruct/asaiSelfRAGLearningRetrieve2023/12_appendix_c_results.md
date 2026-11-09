# C Results

# C. 1 Analysis

Reliance on parametric- and non-parametric memories. We conduct analysis on how frequently model answers come from retrieved passages (non-parametric memories) or their own parametric memories. On two open-domain QA datasets, TriviaQA and PopQA, we conduct the following analysis: 1) sample query models successfully answer correctly, 2) for each query in this group, check whether the matched ground-truth answer is a sub-string of the retrieved passage or not. We evaluate Self-Rag 7B, Alpaca 7B, Alpaca 13B, and Llama2-Chat-13B. We found that Self-Rag significantly less frequently generates answers that are not included in the provided evidence; in particular, in Alpaca 30B, 20\% of the correct predictions are not included in the provided passages, followed by Llama2-chat 13B (18\%) and Alpaca (15\%), while it is only 2\% in Self-Rag. When retrieved passages are not relevant, Self-Rag generates Isrel =Irrelevant, indicating that the following answers may not be factually grounded, while those instruction-tuned models continue to generate plausible answers.

\footnotetext{
${ }^{8}$ https://github.com/facebookresearch/atlas
${ }^{9}$ https://programmablesearchengine.google.com/about/
}

# C. 2 Human Evaluation Examples

Table 6 shows examples with human evaluations on S\&P and correctness of IsREL and IsSUP reflection tokens.

# C. 3 Qualitative Examples

Table 7 shows several examples predicted by our Self-Rag (13B). The first example is the model output to an ASQA question. The first reference states that Emperor Constantine made Sunday a day of rest from labor, and further the second citation supports the fact that the official adoption of Sunday as a day of rest by Constantine in AD 321. In the second example, the model predicts Contradictory to the first output as the output says the person has served as the CEO since 2010, while the passage says he stepped down as CEO in 2015. Indicating those factual contradictions as reflection tokens enables to enforcement of hard control and also verification of model outputs easily. In the third example, while the generation is mostly correct, Self-Rag predicts Partially Support to the statement listing the name of the songs, as they were not explicitly mentioned.

