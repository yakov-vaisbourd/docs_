# 5 Goals of Evaluation

Programming frameworks can be evaluated along many dimensions: computational efficiency, developer efficiency, intuitiveness of the code and concepts, and so forth. In this paper, we focus on perhaps the most pressing issue for current LM pipelines: the role of hand-written, task-specific prompts in achieving performant systems. Our evaluations seek to test the following hypotheses:

H1 With DSPy, we can replace hand-crafted prompt strings with concise and well-defined modules, without reducing quality or expressive power.

H2 Parameterizing the modules and treating prompting as an optimization problem makes DSPy better at adapting to different LMs, and it may outperform expert-written prompts.

H3 The resulting modularity makes it possible to more thoroughly explore complex pipelines that have useful performance characteristics or that fit nuanced metrics.

Our evaluation will explore these hypotheses using diverse task-program pairs. We hope this begins a shift from underspecified questions like "how do different LMs compare on GSM8K" toward "how they compare on GSM8K with program P when compiled with strategy S", which is a well-defined and reproducible run. Ultimately, our goal is to reduce the role of artful prompt construction in modern AI in favor of the development of new modular, composable programs and optimizers.

\begin{table}

**Table 1: Results with in-context learning on GSM8K math word problems. Each row represents a separate pipeline: the module in the Program column is compiled against the examples in the Training set. The programs, compilers, and (small) training sets are defined in Section 6. Rows with ensemble build on the immediately preceding row. Notably, all programs in this table are expressed by composing two to four DSPy modules and teleprompters. Compiling the correct modules, instead of string prompts, improves different LMs from 4-20\% accuracy to 49-88\% accuracy.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Program} & \multirow[b]{2}{*}{Compilation} & \multirow[b]{2}{*}{Training} & \multicolumn{2}{|c|}{GPT-3.5} & \multicolumn{2}{|c|}{Llama2-13b-chat} \\
\hline & & & Dev & Test & Dev & Test \\
\hline \multirow{5}{*}{vanilla} & none & n/a & 24.0 & 25.2 & 7.0 & 9.4 \\
\hline & fewshot & trainset & 33.1 & - & 4.3 & - \\
\hline & bootstrap & trainset & 44.0 & - & 28.0 & - \\
\hline & bootstrap×2 & trainset & 64.7 & 61.7 & 37.3 & 36.5 \\
\hline & +ensemble & trainset & 62.7 & 61.9 & 39.0 & 34.6 \\
\hline \multirow{5}{*}{CoT} & none & n/a & 50.0 & - & 26.7 & - \\
\hline & fewshot & trainset & 63.0 & - & 27.3 & - \\
\hline & fewshot & +human_CoT & 78.6 & 72.4 & 34.3 & 33.7 \\
\hline & bootstrap & trainset & 80.3 & 72.9 & 43.3 & - \\
\hline & +ensemble & trainset & 88.3 & 81.6 & 43.7 & - \\
\hline \multirow{4}{*}{reflection} & none & n/a & 65.0 & - & 36.7 & - \\
\hline & fewshot & trainset & 71.7 & - & 36.3 & - \\
\hline & bootstrap & trainset & 83.0 & 76.0 & 44.3 & 40.2 \\
\hline & +ensemble & trainset & 86.7 & - & 49.0 & 46.9 \\
\hline
\end{tabular}
\end{table}

