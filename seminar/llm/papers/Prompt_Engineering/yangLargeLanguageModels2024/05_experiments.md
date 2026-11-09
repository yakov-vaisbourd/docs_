# 5 Prompt Optimization Experiments

We present the evaluation results for prompt optimization in this section. Our experiments demonstrate that OPRO brings a significant performance gain across the board, with different combinations of LLMs as the optimizer and the scorer.

Section 5.1 describes the experiment setup. Section 5.2 shows main results on reasoning tasks like GSM8K and BBH. Section 5.3 shows ablation studies. Section 5.4 analyzes overfitting in prompt optimization. Section 5.5 compares the prompt optimization performance of meta-prompts in OPRO and EvoPrompt (Guo et al., 2023).

## 5.1 Evaluation Setup

Models. The LLMs we use as the optimizer and the scorer are:
\begin{itemize}
\item[-] Optimizer LLM: Pre-trained PaLM 2-L (Anil et al., 2023), instruction-tuned PaLM 2-L (denoted PaLM 2-L-IT), text-bison, gpt-3.5-turbo, and gpt-4.
\item[-] Scorer LLM: Pre-trained PaLM 2-L and text-bison.
\end{itemize}

With pre-trained PaLM 2-L as the scorer, the optimizer LLM generates A_begin instructions. Since text-bison has been instruction-tuned, the optimizer LLM generates Q_begin and Q_end instructions when text-bison is used as the scorer.

Benchmarks. Our primary evaluation benchmarks are GSM8K (Cobbe et al., 2021) and Big-Bench Hard (BBH) (Suzgun et al., 2022). GSM8K is a benchmark of grade school math word problems with 7,473 training samples and 1,319 test samples, where chain-of-thought prompting (Wei et al., 2022) and the zero-shot instruction "Let's think step by step." (Kojima et al., 2022) have drastically improved the performance over the standard prompting. BBH is a suite of 23 challenging BIG-Bench tasks (Srivastava et al., 2022) that covers a wide range of topics beyond arithmetic reasoning, including symbolic manipulation and commonsense reasoning. Each task contains up to 250 examples in total.

To examine the transferability of the optimized instructions, we also evaluate the instructions optimized for GSM8K on two other mathematical reasoning datasets, i.e., MultiArith (Roy \& Roth, 2016) and AQuA (Ling et al., 2017).

Implementation details. We set the temperature to be 0 when evaluating the performance of generated instructions, in which case the scorer LLM greedily decodes. Unless otherwise specified, we set the default temperature to be 1.0 for optimizer LLMs to generate diverse and creative instructions. At each optimization step, we prompt the optimizer LLM with the meta-prompt 8 times to generate 8 instructions, then we add these instructions with their training scores to the optimization trajectory in the meta-prompt. Our meta-prompt at each step contains the best 20 instructions so far and 3 randomly picked exemplars from the training set. We study the effect of different hyperparameters in ablation studies (Section 5.3). Appendix C.2 presents the full meta-prompts for different optimizer LLMs.

\begin{table}

**Table 4: Test accuracies on GSM8K. We show the instruction with the highest test accuracy for each scorer-optimizer pair.**
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Scorer & Optimizer / Source & Instruction position & Top instruction & Acc \\
\hline \multicolumn{5}{|l|}{Baselines} \\
\hline PaLM 2-L & Kojima et al., 2022) & A_begin & Let's think step by step. & 71.8 \\
\hline PaLM 2-L & (Zhou et al., 2022b & A_begin & Let's work this out in a step by step way to be sure we have the right answer. & 58.8 \\
\hline PaLM 2-L & & A_begin & Let's solve the problem. & 60.8 \\
\hline PaLM 2-L & & A_begin & (empty string) & 34.0 \\
\hline text-bison & Kojima et al., 2022) & Q_begin & Let's think step by step. & 64.4 \\
\hline text-bison & Zhou et al., 2022b & Q_begin & Let's work this out in a step by step way to be sure we have the right answer. & 65.6 \\
\hline text-bison & & Q_begin & Let's solve the problem. & 59.1 \\
\hline text-bison & & Q_begin & (empty string) & 56.8 \\
\hline \multicolumn{5}{|l|}{Ours} \\
\hline PaLM 2-L & PaLM 2-L-IT & A_begin & Take a deep breath and work on this problem step-by-step. Break this down. & 80.2 \\
\hline PaLM 2-L & PaLM 2-L & A_begin & Break this down. & 79.9 \\
\hline PaLM 2-L & gpt-3.5-turbo & A_begin & A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem. & 78.5 \\
\hline PaLM 2-L & gpt-4 & A_begin & Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer. & 74.5 \\
\hline text-bison & PaLM 2-L-IT & Q_begin & Let's work together to solve math word problems! First, we will read and discuss the problem together to make sure we understand it. Then, we will work together to find the solution. I will give you hints and help you work through the problem if you get stuck. & 64.4 \\
\hline text-bison & text-bison & Q_end & Let's work through this problem step-by-step: & 68.5 \\
\hline text-bison & gpt-3.5-turbo & Q_end & Analyze the given information, break down the problem into manageable steps, apply suitable mathematical operations, and provide a clear, accurate, and concise solution, ensuring precise rounding if necessary. Consider all variables and carefully consider the problem's context for an efficient solution. & 66.5 \\
\hline text-bison & gpt-4 & Q_begin & Start by dissecting the problem to highlight important numbers and their relations. Decide on the necessary mathematical operations like addition, subtraction, multiplication, or division, required for resolution. Implement these operations, keeping in mind any units or conditions. Round off by ensuring your solution fits the context of the problem to ensure accuracy. & 62.7 \\
\hline
\end{tabular}
\end{table}

## 5.2 Main Results

We show prompt optimization curves on GSM8K and two BBH tasks in this section. The curves on other BBH tasks are deferred to Appendix D, and the tables containing all accuracy numbers are in Appendix E.

## 5.2.1 GSM8K

For prompt optimization, we randomly sample 3.5\% examples from the GSM8K training set. The same subset is used throughout optimization, so that the task accuracies computed at intermediate optimization steps are approximations of the training accuracy on all 7,473 training examples. This balances the evaluation cost with the generalization performance. After the optimization procedure finishes, we evaluate the found instructions on the entire GSM8K test set.

Figure 1(a) in Section 1 shows prompt optimization curves with pre-trained PaLM 2-L as scorer and PaLM 2-L-IT as optimizer, and the initial instruction is "Let's solve the problem" with a (approximated, and same below) training accuracy of 60.5. We observe that the optimization curve shows an overall upward trend with several leaps throughout the optimization process, for example:
\begin{itemize}
\item[-] "Let's think carefully about the problem and solve it together." at Step 2 with the training accuracy 63.2;
\item[-] "Let's break it down!" at Step 4 with training accuracy 71.3;
\item[-] "Let's calculate our way to the solution!" at Step 5 with training accuracy 73.9;
\item[-] "Let's do the math!" at Step 6 with training accuracy 78.2.
\end{itemize}

The optimization curves also generally show a decrease of the variance among the accuracies of instructions generated at each step, indicating that the optimizer LLM generates distributionally better instructions throughout the optimization.

Next, we present the results of generating Q_begin instructions with the text-bison scorer and the PaLM 2-L-IT optimizer, starting from an empty instruction with a 57.1 training accuracy. The optimization curve in Figure 4(a) shows a similar upward trend, during which a few leaps in the training accuracy include:
\begin{itemize}
\item[-] "Solve the following problems using the given information." at Step 2 with training accuracy 59.8;
\item[-] "Solve the following problems by applying the given information and using the appropriate mathematical operations." at Step 3 with training accuracy 64.0;
\item[-] "Let's read the problem carefully and identify the given information. Then, we can create an equation and solve for the unknown variable." at Step 4 with training accuracy 67.0;
\item[-] "I'm always down for solving a math word problem together. Just give me a moment to read and understand the problem. Then, I'll create an equation that models the problem, which I'll solve for the unknown variable. I also may or may not use some helpful diagrams or visuals to understand the problem. Lastly, be sure to allow me some time to carefully check my work before submitting any responses!" at Step 29 with training accuracy 70.1.
\end{itemize}

Note that although our default setting is to run OPRO for 200 steps in prompt optimization, we need much fewer steps if the goal is to find some outstanding instructions. An example is that the Figure 1(a) experiment found "Let's do the math!" at Step 6 with training accuracy 78.2, almost matching the "Take a deep breath and work on this problem step-by-step." found at the 107th step with training accuracy 80.2, at a point where the optimization curve is still trending upwards. This is because a leap in our optimization curve does not always correspond to a much better instruction being discovered; instead, it can be due to a large qualitative improvement of all 8 generated instructions in this step. The latter usually happens several steps after the former: after a much better instruction is discovered in one step, the meta-prompt gradually gets rid of worse instructions in the latter steps by generating instructions similar to the much-better one. The top instructions kept in the meta-prompt gradually improves in this procedure. At a point when the meta-prompt only triggers higher quality instructions, the leap happens.

Finally, Figure 4(b) shows that the pre-trained PaLM 2-L can also serve as the optimizer LLM and improve its own prediction performance. Different from other optimizer LLMs that are instructiontuned, the pre-trained PaLM 2-L performs better when the prompt is formatted in a few-shot manner. Therefore, we include two initial instructions to start the optimization: the empty instruction (with a training accuracy 32.2) and "The answer is" (with a training accuracy 33.3). See Figure 21 in Appendix C for the meta-prompt format. The generated instructions follow the same style as "The answer is": most instructions are also phrases suitable as the prefix of a sentence, like "Here you go:" (generated at Step 11 with training accuracy 61.3) and "Let's do it:" (generated at Step 13 with training accuracy 75.1).

Table 4 summarizes top instructions found on GSM8K with different scorer and optimizer LLMs. We observe that:
\begin{itemize}
\item[-] The styles of instructions found by different optimizer LLMs vary a lot: PaLM 2-L-IT and text-bison ones are concise, while GPT ones are long and detailed.
\item[-] Although some top instructions contain the "step-by-step" phrase, most others achieve a comparable or better accuracy with different semantic meanings.
\end{itemize}

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-11_385_1007_296_559.jpg)

**Figure 4: Prompt optimization on GSM8K with (a) the text-bison scorer and the PaLM 2-L-IT optimizer, and (b) pre-trained PaLM 2-L as both scorer and optimizer.**

## 5.2.2 BBH

On BBH, the optimization starts from an empty string as the initial instruction by default. The instructions are placed at A_begin when the scorer is PaLM 2-L, and at Q_begin when the scorer is text-bison. For each task, we utilize a subset of 20\% examples for prompt optimization, and the rest examples are for testing. We show experimental results on more variants of the instruction position and initialization in Appendix E.

Figure 5 visualizes the per-task accuracy difference on all 23 BBH tasks compared to the instruction "Let's think step by step." (Kojima et al., 2022) and the empty instruction, and we present the concrete accuracies in Table 7 of Appendix E. We show that the instructions found by OPRO outperform "Let's think step by step." on almost all tasks by a large margin: our instructions outperform by over 5\% on 19/23 tasks with the PaLM 2-L scorer, and on 15/23 tasks with the text-bison scorer. Our prompt optimization algorithm also improves instructions from the empty starting point by over 5\% on most tasks: 20/23 with the PaLM 2-L scorer and 15/23 with the text-bison scorer.

Similar to GSM8K, we observe upward trends in optimization curves on almost all BBH tasks, as shown in Figure 6. See Figure 23 and 24 in Appendix Dfor more curves on other BBH tasks.

We next show some examples of instructions found through the course of optimization. On the task ruin_names, starting from the empty instruction (with 64.0 training accuracy), with the text-bison scorer and the PaLM 2-L-IT optimizer, the following instructions are generated:
\begin{itemize}
\item[-] "Consider the following when editing artist or movie names humorously:" at Step 1 with training accuracy 72.0;
\item[-] "When making humorous edits of artist or movie names, you can change one or more letters or even create puns by adding new words that sound similar." at Step 18 with training accuracy 80.0;
\item[-] "We can make humorous edits of artist/movie names by changing letters to create new words that are similar in sound but have different meanings. For example, The Police can be changed to The Polite, The Abyss can be changed to Toe Abyss, and Schindler's List can be changed to Schindler's Lost." at Step 38 with training accuracy 82.0.
\end{itemize}

Although the above instructions are semantically similar, a paraphrase by the optimizer LLM offers a notable accuracy improvement. We further highlight this observation in Section 5.2.3

Below are some instructions generated when performing prompt optimization on temporal_sequences, starting from the empty instruction (with the training accuracy of 64.0):
\begin{itemize}
\item[-] "To solve this problem, we need to first identify the time period when the person was not seen doing anything else. Then, we need to check if the place they went to was open during that time period. If it was, then that is the time period when they could have gone to that place." at Step 2 with training accuracy 42.0;
\item[-] "To find the time period when a person could have gone to a place, identify the time periods when they were not seen doing anything else and the place was open. If there are multiple time periods that match these criteria, then the person could have gone to the place during any of these time periods." at Step 18 with training accuracy 54.0;
\end{itemize}

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-12_1679_1360_299_382.jpg)

**Figure 5: On 23 BBH tasks, the accuracy differences among instructions found by prompt optimization (with the PaLM 2-L-IT optimizer), "Let's think step by step.", and the empty string (optimization starting point).**

\begin{itemize}
\item[-] "To determine the possible time period when a person went to a place, first identify all the time periods when the person was not seen doing anything else and the place was open. Then, rule out any time periods during which the person was seen doing something else. The remaining time periods are the possible times when the person could have gone to the place." at Step 41 with training accuracy 72.0.
\end{itemize}
Table 5 presents the best instructions generated on movie_recommendation, ruin_names, and temporal_sequences tasks with different combinations of the optimizer and the scorer LLMs. Again,

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-13_385_1003_288_559.jpg)

**Figure 6: Training accuracy curves of prompt optimization on BBH ruin_names and temporal_sequences with the text-bison scorer and the PaLM 2-L-IT optimizer. The optimizations start from the empty string.**

different optimizer LLMs produce instructions of different styles. See Appendix E for results on more BBH tasks.

## 5.2.3 Semantically similar instructions may achieve drastically different ACCURACIES

One challenge of prompt optimization is the sensitivity of model performance to subtle changes in the instruction. For example, with the PaLM 2-L scorer on the GSM8K test set, "Let's think step by step." achieves accuracy 71.8, "Let's solve the problem together." has accuracy 60.5, while the accuracy of "Let's work together to solve this problem step by step." is only 49.4, although it is the semantic combination of the two upper instructions. This behavior increases both the variance across single-step instructions and the oscillation during optimization, and motivates us to generate multiple instructions at each step to improve the optimization stability.

## 5.2.4 Transferability of found instructions

We assess the transferability of found prompts to different datasets of the same domain, where we evaluate the top instructions found for GSM8K on two more math reasoning benchmarks Multi-Arith (Roy \& Roth, 2016) and AQuA (Ling et al., 2017). Table 6 shows that our optimized prompts also outperform baseline prompts with different scorer LLMs on these two benchmarks.

## 5.3 Ablation Studies

We use text-bison as the scorer and PaLM 2-L as the optimizer for all ablation studies. The tasks we evaluate are GSM8K (math reasoning) and BBH sports_understanding (non-math reasoning).

Meta-prompt design. The meta-prompt design is crucial in achieving good prompt optimization performance. We investigate the following core design choices:
\begin{itemize}
\item[-] The order of the previous instructions. We compare the following options: (1) from lowest to highest (our default setting); (2) from highest to lowest; (3) random. Figures 7(a) and 7(b) show that the default setting achieves better final accuracies and converges faster. One hypothesis is that the optimizer LLM output is affected more by the past instructions closer to the end of the meta-prompt. This is consistent with the recency bias observed in Zhao et al. (2021), which states that LLMs are more likely to generate tokens similar to the end of the prompt.
\item[-] The effect of instruction scores. In terms of how to present the accuracy scores, we compare three options: (1) rounding the accuracies to integers, which is equivalent to bucketizing the accuracy scores to 100 buckets (our default setting); (2) bucketizing the accuracies to 20 buckets; (3) not showing the accuracies, only showing the instructions in the ascending order. Figures 7(c) and 7(d) show that the accuracy scores assists the optimizer LLM in better understanding the quality difference among previous instructions, and thus the optimizer LLM proposes better new instructions that are similar to the best ones in the input optimization trajectory.
\item[-] The effect of exemplars. We compare three options: (1) showing 3 exemplars from the task (default); (2) showing 10 exemplars from the task; (3) no exemplars. Figures 7(e) and 7(f) show
\end{itemize}

\begin{table}

**Table 5: Top instructions with the highest accuracies found in prompt optimization on BBH movie_recommendation, ruin_names, and temporal_sequences.**
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Scorer & Optimizer & Instruction position & Instruction & Acc \\
\hline \multicolumn{5}{|l|}{movie_recommendation} \\
\hline PaLM 2-L & PaLM 2-L-IT & A_begin & Based on your input, I have analyzed the given movies in terms of genre, plot, tone, audience rating, year of release, director, cast, and reviews. I have also taken into account the given options. The movie that is most similar to the given movies in terms of all these factors is: & 90.8 \\
\hline PaLM 2-L & PaLM 2-L & A_begin & The best film: & 88.4 \\
\hline PaLM 2-L & gpt-3.5-turbo & A_begin & Let's uncover the perfect movie recommendation from the options provided, ensuring an exceptional cinematic experience together as we select the most captivating and satisfying choice that will keep us thoroughly engaged and immersed until the very end. & 88.0 \\
\hline text-bison & PaLM 2-L-IT & Q_begin & What is the highest-rated movie similar to the given movies, with a similar IMDb rating and released in the same year? & 91.6 \\
\hline text-bison & gpt-3.5-turbo & Q_begin & Based on the movie list provided, carefully consider your preferences and make a well-informed decision. & 70.8 \\
\hline \multicolumn{5}{|l|}{ruin_names} \\
\hline PaLM 2-L & PaLM 2-L-IT & A_begin & Which is the funniest pun on the artist or movie name? & 88.0 \\
\hline PaLM 2-L & PaLM 2-L & A_begin & Answer for ruin: & 83.6 \\
\hline PaLM 2-L & gpt-3.5-turbo & A_begin & Prepare to have a side-splittingly funny time as we uncover the most clever and hilarious alternatives for these artist or movie names, challenging your wit to guess the correct one with a burst of creativity, humor, and imaginative twists! & 86.8 \\
\hline text-bison & PaLM 2-L-IT & Q_begin & A humorous edit of an artist or movie name can be created by replacing one or more letters to form a new word or phrase that sounds similar but has a different meaning. The new word or phrase should be relevant to the original word, but it should also be a surprise, which makes the edit funny. For example, the artist or movie name "Rocky" can be changed to "Ricky," and "Schindler's List" can be changed to "Schindler's Lift." Be creative and have fun! & 83.6 \\
\hline text-bison & gpt-3.5-turbo & Q_begin & Choose the option that offers the most clever and humorous alteration of the given artist or movie name. Let your creativity shine and select the answer that will undoubtedly bring a smile to your face! Make sure to think outside the box! & 75.2 \\
\hline \multicolumn{5}{|l|}{temporal_sequences (no PaLM 2-L as scorer results because its training accuracy on empty string is 100.0)} \\
\hline text-bison & PaLM 2-L-IT & Q_begin & To determine the time period when a person went to a place, first identify all the time periods when the person's whereabouts are unknown. Then, rule out any time periods during which the person was seen doing something else or the place was closed. The remaining time periods are the possible times when the person could have gone to the place. & 80.4 \\
\hline text-bison & gpt-3.5-turbo & Q_begin & Identify the optimal time slot for the individual to engage in the mentioned location/activity considering the given sightings and waking up time, taking into account the opening and closing times of the location and the duration of each event. & 53.6 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 6: Transferability across datasets: accuracies of top instructions found for GSM8K on MultiArith and AQuA.**
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline \multirow{2}{*}{Scorer} & \multirow{2}{*}{Source} & \multirow[t]{2}{*}{Instruction position} & \multirow{2}{*}{Instruction} & \multicolumn{2}{|c|}{Accuracy} \\
\hline & & & & MultiArith & AQuA \\
\hline \multicolumn{6}{|l|}{Baselines} \\
\hline PaLM 2-L & Kojima et al., 2022) & A_begin & Let's think step by step. & 85.7 & 44.9 \\
\hline PaLM 2-L & (Zhou et al., 2022b) & A_begin & Let's work this out in a step by step way to be sure we have the right answer. & 72.8 & 48.4 \\
\hline PaLM 2-L & & A_begin & Let's solve the problem. & 87.5 & 44.1 \\
\hline PaLM 2-L & & A_begin & (empty string) & 69.3 & 37.8 \\
\hline text-bison & Kojima et al., 2022) & Q_begin & Let's think step by step. & 92.5 & 31.9 \\
\hline text-bison & Zhou et al., 2022b) & Q_begin & Let's work this out in a step by step way to be sure we have the right answer. & 93.7 & 32.3 \\
\hline text-bison & & Q_begin & Let's solve the problem. & 85.5 & 29.9 \\
\hline text-bison & & Q_begin & (empty string) & 82.2 & 33.5 \\
\hline \multicolumn{6}{|l|}{Ours} \\
\hline PaLM 2-L & PaLM 2-L-IT on GSM8K & A_begin & Take a deep breath and work on this problem step-by-step. & 95.3 & 54.3 \\
\hline text-bison & PaLM 2-L-IT on GSM8K & Q_begin & Let's work together to solve math word problems! First, we will read and discuss the problem together to make sure we understand it. Then, we will work together to find the solution. I will give you hints and help you work through the problem if you get stuck. & 96.8 & 37.8 \\
\hline
\end{tabular}
\end{table}
that presenting exemplars in the meta-prompt is critical, as it provides information on what the task looks like and helps the optimizer model phrase new instructions better. However, more exemplars do not necessarily improve the performance, as a few exemplars are usually sufficient to describe the task. In addition, including more exemplars results in a longer meta-prompt with a dominating exemplar part, which may distract the optimizer LLM from other important components like the optimization trajectory.

The number of generated instructions per step. Computing a mini-batch of gradients reduces the variance of a stochastic gradient descent procedure. Similarly, generating multiple instructions in each step improves the optimization stability with LLMs. On the other hand, to achieve better performance with a fixed budget for the number of instructions to evaluate, the number of per-step instructions should not be too large, so as to allow more optimization steps to incorporate richer information of past instructions with their accuracies. Taking both aspects into consideration, Figure 8 compares the optimization performance of sampling 1 / 2 / 4 / 8 (default) / 16 instructions in each step, showing that sampling 8 instructions at each step overall achieves the best performance.

Starting point. We study the effect of different initial instructions for prompt optimization. Our default setting is to start from an empty string when the scorer LLM is (instruction-tuned) text-bison, and to start from either the empty string (on BBH tasks) or "Let's solve the problem." (on GSM8K) with instruction position A_begin when the scorer LLM is the (pre-trained) PaLM 2-L. Figure 9(a) shows the performance of text-bison as the scorer LLM with 3 options of initial instructions: (1) the empty string; (2) "Solve the following problem."; or (3) "Solve the following problem." and "Let's solve the problem.". We observe that the accuracies do not differ much with different starting points. Interestingly, the styles of the generated instructions are also similar. For example, most of the generated instructions starting from (1) and (2) contain the phrase "solve this problem", like "Let's work together to solve this problem." in Step 4 with training accuracy 64.8 from (1), and "Let's solve the following problems using the given information." in Step 3 with training accuracy 62.8 from (2).

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-16_1975_1326_369_407.jpg)

**Figure 7: Ablation studies: how each part of the meta-prompt matters. The dots are the average values across 3 optimization repetitions, and the shaded regions represent standard deviations.**

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-17_633_1301_281_407.jpg)

**Figure 8: Ablation studies: the number of generated instructions in each step. The dots are the average values across 3 optimization repetitions, and the shaded regions represent standard deviations. The x-axis represents the total number of evaluated instructions through the optimization; e.g., we run 200 optimization steps when sampling 8 instructions in each step, run 400 steps when sampling 4 instructions in each step, etc.**

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-17_624_1310_1235_400.jpg)

**Figure 9: Ablation studies: the initial instructions for prompt optimization. The dots are the average values across 3 optimization repetitions, and the shaded regions represent standard deviations.**

Figure 9(b) presents the results of of PaLM 2-L as the scorer LLM with the following options of initial instructions: (1) "Let's solve the problem."; (2) the empty string; or (3) "Let's think step by step.". We notice that the performance differs much more with different initial instructions, especially at the beginning of the optimization. Specifically, starting from (1) leads to better generated instructions than (2) in the first 30 steps, while the instructions optimized from both (1) and (2) are worse than (3) throughout. A similar observation holds when using PaLM 2-L as scorer and gpt-3.5-turbo as optimizer for BBH tasks, by comparing the results starting from the empty string (Appendix E.2) and from "Let's solve the problem." (Appendix E.3). Taking a closer look into the optimization process of (2), we find that although both "solve the problem" and "step by step" show up in generated instructions at Step 5, it takes the optimizer LLM more steps to get rid of worse instructions presented in the meta-prompt when starting from instructions with lower accuracies. Therefore, one direction for future work is to accelerate convergence from weaker starting points.

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-18_568_1138_283_489.jpg)

**Figure 10: Ablation studies: temperature of the optimizer model. The dots are the average values across 3 optimization repetitions, and the shaded regions represent standard deviations.**

Diversity per step. We evaluate the following temperatures of the optimizer LLM: \{0.0, 0.5, 1.0 (default), 1.5, 2.0\}. Figure 10 shows the default temperature 1.0 achieves the best performance. Specifically, optimizations with smaller temperatures (0.0 and 0.5) lack exploration and thus creativity, and the optimizer LLM often gets stuck at the same instruction for tens of steps, resulting in flat optimization curves. On the other hand, with larger temperatures (1.5 and 2.0), the optimizer LLM more often ignores the trajectory of previous instructions presented in the meta-prompt and thus lacks exploitation, therefore the optimization curve does not have a steady upward trend.

Comparison with one-step instruction generation. Our current iterative procedure runs for multiple steps and generates a new batch of solutions in each step. To validate the importance of leveraging the optimization trajectory for generating new prompts, we compare to a baseline that generates all instructions in a single step without entering into the optimization procedure. We compare these two approaches on GSM8K and BBH sports_understanding with the PaLM 2-L-IT optimizer. For GSM8K the scorer LLM is pre-trained PaLM 2-L and the initial instruction is "Let's solve the problem", and for BBH sports_understanding the scorer LLM is text-bison and the initial instruction is the empty string. The baseline generates 50 instructions in a single step, thus its meta-prompt only includes task exemplars, the initial instruction with its accuracy, and the same meta-instructions as our full meta-prompt for performing optimization. All the other hyperparameters remain the same.

Our results show that this one-step instruction generation performs much worse than our optimization approach. Specifically: (1) On GSM8K, the best instruction among all 50 is still "Let's solve the problem", with a 64.4 training accuracy and a 60.8 test accuracy. On the other hand, our approach (corresponding to Figure 1(a) in the main paper) found "Let's do the math!" with a 78.2 training accuracy and a 76.3 test accuracy at the 5th step by generating 8 instructions at each step. (2) Similarly, on BBH sports_understanding, the best instruction among all 50 achieved a 84.0 training accuracy and 80.0 test accuracy. This is again worse than the instruction found by our approach at Step 4, which achieved a 88.0 training accuracy and a 84.5 test accuracy.

## 5.4 Overfitting Analysis in Prompt Optimization

For simplicity, we do not set aside a validation set in our default setting of prompt optimization. We made this decision based on the experiments when a validation set is present.

Overfitting may result in training accuracy being much higher than the validation/test accuracy. It is difficult to avoid overfitting, but overfitting is less harmful when each candidate solution (natural language instruction in the prompt optimization context) overfits to a similar extent. In this case, a higher training accuracy solution still achieves a higher validation/test accuracy, and one can adopt solutions with the highest training accuracies as the final result. Figure 11 shows this is the case for OPRO in prompt optimization: when setting aside a validation set with the same size as the training

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-19_534_1299_296_407.jpg)

**Figure 11: Overfitting analysis. The exemplars are splitted to 1/3 training, 1/3 validation and 1/3 test. We compute the validation accuracy every 3 steps. The training/validation dots are the average training/validation accuracies across 3 optimization repetitions, respectively, and the shaded regions represent standard deviations.**

set, the validation accuracy curves trend up and down alongside the training curves in both prompt optimization settings.

Of course, overfitting still occurs in the instructions found by our prompt optimization: in Table 7 and 10 our training accuracies are often $5 \%-20 \%$ higher than our test accuracies, despite that our test and overall accuracies are still mostly higher than human-written counterparts. Setting aside a larger training set and optimizing for fewer steps (early stopping) may help reduce overfitting.

## 5.5 Comparison with EvoPrompt

Some concurrent works on prompt optimization propose meta-prompts that explicitly ask the LLM to perform mutation and crossovers of existing prompts (Fernando et al., 2023; Guo et al., 2023). In our evaluation, we compare our approach to the Genetic Algorithm (GA) and Differential Evolution (DE) versions of EvoPrompt (Guo et al., 2023). Specifically, in the GA meta-prompt, given two prompts, the meta-prompt instructs the LLM to cross over the two prompts and generates a new one, then mutates the newly generated prompt to produce the final prompt. DE extends the GA meta-prompt to include more detailed instructions, e.g., asking the LLM to identify different parts between the two given prompts before performing the mutation. This is in contrast with OPRO, which leverages the optimization trajectory including multiple past prompts, instead of only 2 previous prompts. Meanwhile, OPRO also provides the LLM with richer information to facilitate the understanding of the optimization problem, including exemplars and task accuracies of different prompts.

Figure 12 presents the results on GSM8K and BBH sports_understanding benchmarks, where we use gpt-3.5-turbo as the optimizer. On GSM8K, the initial instructions of all approaches are "Let's solve the problem." and "Here is the answer.", which are simple and generic. Again, we observe that OPRO performance steadily improves with more optimization steps. On the other hand, both versions of EvoPrompt even degrade the performance on GSM8K. The main reason is because EvoPrompt does not utilize exemplars for prompt optimization, thus it lacks the understanding of the task to optimize for. In this way, EvoPrompt relies on good-quality and task-specific initial prompts to optimize from.

Given this observation, we provide more task-specific initial instructions for experiments on BBH sports_understanding, which are "Solve the sports understanding problem." and "Give me the answer to sports understanding." In this case, EvoPrompt (DE) is able to find better prompts than the initial ones, but the optimization curve is less stable than OPRO. This indicates that leveraging the optimization trajectory helps the LLM to identify promising directions to improve existing prompts.

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-20_645_1226_296_448.jpg)

**Figure 12: Comparison with EvoPrompt in prompt optimization. We use the gpt-3.5-turbo optimizer for both experiments. "EvoPrompt (GA)" uses the meta-prompt from Guo et al. (2023), Figure 1; "EvoPrompt (DE)" uses the meta-prompt from Guo et al. (2023), Figure 2. All optimizations in (a) use the pre-trained PaLM 2-L scorer and start from two simple instructions "Let's solve the problem." and "Here is the answer."; all optimizations in (b) use the text-bison scorer and start from two richer (task-specific) instructions "Solve the sports understanding problem." and "Give me the answer to sports understanding.". The dots are the average values across 3 optimization repetitions, and the shaded regions represent standard deviations. We use temperature 1.0 for OPRO and temperature 0.5 for EvoPrompt, same as the default settings in respective works.**

