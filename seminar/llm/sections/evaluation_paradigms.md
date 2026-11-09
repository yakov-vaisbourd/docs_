# Evaluation Paradigms

## Benchmarks and Leaderboards

Standard LLM benchmarks test different capability dimensions: MMLU (massive multitask language understanding, 57-subject multiple choice), HellaSwag (commonsense reasoning), HumanEval (code generation), GSM8K (grade-school math), TruthfulQA (factual accuracy), and MATH (competition mathematics). Aggregate leaderboards (e.g.\ LMSYS Chatbot Arena, Open LLM Leaderboard) rank models across many tasks. Key caveat: benchmarks become contaminated once their test sets appear in pre-training data, inflating reported scores. SOTA on a narrow benchmark is a narrow claim - it says nothing about other tasks, safety, or deployment reliability.

## Perplexity

The exponentiated average negative log-likelihood per token on a held-out corpus:

$$
\begin{aligned}
\mathrm{PPL} = \exp\!\left(-\frac{1}{L}\sum_{l=1}^{L} \log P_\theta(x_l \mid x_{1:l-1})\right)
\end{aligned}
$$

Lower perplexity means the model assigns higher probability to the true tokens. Used to compare base models on language modelling quality; less meaningful for instruction-tuned models where the goal is task performance rather than generic text likelihood.

## Human Evaluation

Human raters compare model outputs directly, either by absolute scoring (Likert scales on helpfulness, harmlessness, honesty) or by pairwise preference (which of two responses is better?). LMSYS Chatbot Arena uses ELO ratings from crowdsourced pairwise battles. Human evaluation is expensive, slow, and subject to position bias (preferring the first or longer response) and annotator disagreement.

## LLM-as-Judge

A strong LLM (e.g.\ GPT-4) is used as an automatic evaluator. Given a question, a model response, and optionally a reference answer, the judge produces a score or a preference ranking. Structured rubrics (True/False questions, aspect-level scoring) are more reliable than open-ended scoring. Key failure modes: positional bias, self-preference (models favour their own style), and sensitivity to prompt wording. MT-Bench and AlpacaEval use this paradigm.

## Task-Specific Evaluation

For code: pass@k (fraction of problems solved by at least one of $k$ sampled completions). For reasoning: exact-match accuracy against a verifiable answer (GSM8K, MATH). For retrieval: recall@k and NDCG. For summarisation: factual consistency measured by NLI models (FactScore). For safety: red-teaming attack success rate. Task-specific metrics are more discriminative than surface-level text overlap metrics for the respective capability.

