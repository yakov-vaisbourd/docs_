\title{
Large Language Models as Optimizers
}

\author{
Chengrun Yang* Xuezhi Wang Yifeng Lu Hanxiao Liu Quoc V. Le Denny Zhou Xinyun Chen* \\ \{chengrun, xuezhiw, yifenglu, hanxiaol\}@google.com \{qvl, dennyzhou, xinyunchen\}@google.com
}

Google DeepMind * Equal contribution

\begin{abstract}
Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step. We first showcase OPRO on linear regression and traveling salesman problems, then move on to our main application in prompt optimization, where the goal is to find instructions that maximize the task accuracy. With a variety of LLMs, we demonstrate that the best prompts optimized by OPRO outperform human-designed prompts by up to 8\% on GSM8K, and by up to 50\% on Big-Bench Hard tasks. Code at https://github.com/google-deepmind/opro
\end{abstract}

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-01_354_932_1341_589.jpg)

**Figure 1: Prompt optimization on GSM8K (Cobbe et al., 2021) and BBH (Suzgun et al., 2022) movie_recommendation. The optimization on GSM8K has pre-trained PaLM 2-L as the scorer and the instruction-tuned PaLM 2-L (denoted PaLM 2-L-IT) as the optimizer; the optimization on BBH movie_recommendation has text-bison as the scorer and PaLM 2-L-IT as the optimizer. Each dot is the average accuracy across all (up to 8) generated instructions in the single step, and the shaded region represents standard deviation. See Section 5 for more details on experimental setup.**

\begin{table}

**Table 1: Top instructions with the highest GSM8K zero-shot test accuracies from prompt optimization with different optimizer LLMs. All results use the pre-trained PaLM 2-L as the scorer.**
\begin{tabular}[t]{|l|l|l|}
\hline Source & Instruction & Acc \\
\hline \multicolumn{3}{|l|}{Baselines} \\
\hline Kojima et al., 2022 & Let's think step by step. & 71.8 \\
\hline Zhou et al., 2022b) & Let's work this out in a step by step way to be sure we have the right answer. & 58.8 \\
\hline & (empty string) & 34.0 \\
\hline \multicolumn{3}{|l|}{Ours} \\
\hline PaLM 2-L-IT & Take a deep breath and work on this problem step-by-step. & 80.2 \\
\hline PaLM 2-L & Break this down. & 79.9 \\
\hline & gpt-3.5-turbo A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem. & 78.5 \\
\hline gpt-4 & Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer. & 74.5 \\
\hline
\end{tabular}
\end{table}

