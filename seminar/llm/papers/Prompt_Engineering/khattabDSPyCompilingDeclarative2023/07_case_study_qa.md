# 7 Case Study: Complex Question Answering

In this case study, we explore the multi-hop question answering task with the HotPotQA (Yang et al., 2018) dataset in the open-domain "fullwiki" setting. For retrieval, we use a search index of the official Wikipedia 2017 "abstracts" dump of HotPotQA. Search is conducted by a ColBERTv2 (Santhanam et al., 2021) retriever. The HotPotQA test set is hidden, so we reserve the official validation set for our testing, and sample 1000 examples for that. We sub-divide the training set into 70\%/30\% train/validation splits. In the training (and thus validation) split, we keep only examples marked as "hard" in the original dataset, which matches the designation of the official validation and test sets. For training and for reporting development results, we sample 200 and 300 examples respectively.

Programs Considered Our simplest baseline is the vanilla program used in the previous case study on GSM8K (Sec 6); the "question -> answer" signature is universal enough that it will work for this task (and many others) when compiled appropriately.

Our baseline RAG program is the one given in Section 3.2 as a simple example of RAG with a dspy. Chain0fThought layer. We will see that this program does not excel at HotPotQA, and this motivates us to evaluate two multi-hop programs.

To that end, we first test ReAct (Yao et al., 2022), a multi-step agent for tool use, which is implemented as a built-in module in DSPy. In the simplest case, a ReAct module for a particular signature can be declared as follows in DSPy:
```
react = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)], max_iters=5)
```

We also test the following custom program, which simulates the information flow in Baleen (Khattab et al., 2021a) and IRRR (Qi et al., 2020) and has similarities to IRCoT (Trivedi et al., 2022).
```
class BasicMultiHop(dspy.Module):
    def __init__(self, passages_per_hop):
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    def forward(self, question):
        context = []
        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages
        return self.generate_answer(context=context, question=question)
multihop = BasicMultiHop(passages_per_hop=3)
```

Compiling For compilers, we continue to use the ones that we used for GSM8K (see Sec 6). We also consider two compositions of our teleprompters. For ReAct, we consider bootstrapping with BootstrapFewShotWithRandomSearch starting from an earlier bootstrap of the ReAct program. For the simple multihop program, we also consider fine-tuning with T5-Large starting from the earlier bootstrap of that program.
```
l multihop_t5 = dspy.BootstrapFinetune(metric=answer_exact_match).compile(program,
    teacher=bootstrap, trainset=trainset, target='t5-large')
```

Results Table 2 summarizes our results. Compared with the vanilla few-shot prompting, a chainof-thought and retrieval-augmented generation (CoT_RAG) program can self-bootstrap in DSPy to increase answer EM substantially. However, this relies entirely on the ColBERTv2 retriever to find relevant passages directly from the original questions, limiting its passage recall. This is tackled in the react and multihop programs, which will generate queries for the retriever in multiple iterative "hops". Indeed, overall, a simple multihop program performs the best, and in general bootstrap again proves to be very effective at raising its quality relative to its fewshot variant for both LMs.

In particular, we can see that bootstrap (and/or bootstrap×2) can outperform both fewshot prompting (for multihop) and expert human reasoning (for react; adapted slightly from Yao et al. (2022) to our retrieval setting). Perhaps most importantly, we can make llama2-13b-chat competitive with GPT-3.5 by simply compiling our programs.

To assess the finetuning capacity of DSPy, we also evaluated the compiler multihop_t5 defined above which produces a T5-Large (770M parameter) model. This program scores 39.3\% answer EM and $46.0 \%$ passage accuracy on the dev set, using only 200 labeled inputs and 800 unlabeled

\begin{table}

**Table 2: Results with in-context learning on HotPotQA multi-hop retrieval question answering. We report answer exact match (Ans) and pair-retrieval accuracy (Psg). Each row represents a separate pipeline: the module in the Program column is compiled against the examples in the Training set. The programs, compilers, and (small) training sets are defined in the main text. For HotPotQA, we use the training set (and not dev) directly for cross-validation. *The marked result is evaluated on 50\% of our test set due to cost.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow{3}{*}{Program} & \multirow{3}{*}{Compiler} & \multicolumn{4}{|c|}{GPT-3.5} & \multicolumn{4}{|c|}{Llama2-13b-chat} \\
\hline & & \multicolumn{2}{|c|}{Dev} & \multicolumn{2}{|c|}{Test} & \multicolumn{2}{|c|}{Dev} & \multicolumn{2}{|c|}{Test} \\
\hline & & Ans & Psg & Ans & Psg & Ans & Psg & Ans & Psg \\
\hline vanilla & fewshot & 34.3 & n/a & 31.5 & n/a & 27.5 & n/a & 21.8 & n/a \\
\hline \multirow{2}{*}{CoT_RAG} & fewshot & 36.4 & 36.0 & 29.8 & 34.4 & 34.5 & 36.0 & 28.0 & 34.4 \\
\hline & bootstrap & 42.3 & 36.0 & - & - & 38.3 & 36.0 & 32.9 & 34.4 \\
\hline \multirow{4}{*}{react} & none & 20.3 & - & - & - & 20.0 & - & - & - \\
\hline & +human_r & 33.0 & - & - & - & 28.3 & - & - & - \\
\hline & bootstrap & 31.0 & - & - & - & 24.7 & - & - & - \\
\hline & bootstrap × 2 & 39.0 & - & - & - & 40.0 & - & - & - \\
\hline \multirow{3}{*}{multihop} & fewshot & 36.9 & 38.3 & 31.2 & 40.8 & 34.7 & 32.0 & 31.3 & 30.8 \\
\hline & bootstrap & 48.7 & 47.0 & 39.6 & 43.8 & 42.0 & 48.3 & 36.4 & 43.5 \\
\hline & ensemble & 54.7 & - & 45.6* & - & 50.0 & - & 41.0 & - \\
\hline
\end{tabular}
\end{table}
questions. For compiling, we use a teacher program consisting of an ensemble (union) of two multihop with llama2-13b-chat. Considering its extremely small size and local availability, this compiled program with T5-Large would impose orders of magnitude lower costs for inference than a proprietary LM like GPT-3.5.

Our results may be pegged against the evaluation on HotPotQA in a number of recent papers, though there is significant variation in evaluation methodology and test set samples across studies in this space. Using CoT prompting, Si et al. (2022) achieve 25.2\% EM. With a "recite-and-answer" technique that uses PaLM-62B (Chowdhery et al., 2022) to recite evidence passages, Sun et al. (2022) achieve 26.5\% EM. Wang et al. (2022a) achieve 33.8\% EM and 44.6\% F1 when applying selfconsistency for PaLM-540B. Yao et al. (2022) achieve 27.4\% EM using ReAct with PaLM-540B and 30.8 with text-davinci-002, with a tool giving it the ability for search using a Wikipedia API. They push their PaLM results to 35.1\% EM by applying an additional CoT step with selfconsistency, which may resemble our ensemble approach in the sense of aggregating multiple answers. Trivedi et al. (2022) reports 49\% using a pipeline with code-davinci-002 LM on a sample of 500 HotPotQA questions.

