# C Additional Results

# C. 1 Instruction Induction

Few-shot In-context Learning We evaluated APE-generated instructions in the few-shot incontext learning, where we insert the instruction before the in-context demonstrations. Those instructions are selected based on zero-shot execution accuracy, and we denote this setting as "Instruction + In-context" in Figure 8. As shown in Figure 8, adding an instruction achieves a comparable or better test performance than the standard in-context learning performance on 21 of 24 tasks. Counter-intuitively, adding in-context examples for Rhymes, Large Animal, and Second Letters hurts model performance. We conjecture that it may be because the selected instructions overfit the zero-shot learning scenario and thus do not perform well on the few-shot case. Therefore, we experiment using few-shot execution accuracy as the selection metric. Figure 14 shows that the few-shot metric achieves comparable or slightly better than the zero-shot metric except for Rhymes. To have an intuitive understanding of what is happening, we provide a qualitative analysis below.

Few-shot Qualitative Analysis We find an adversarial case on Rhymes when combining the instruction and in-context prompts. Table 8 shows that 4 of 5 filtered instructions ask to echo the input word. These proposals effectively hack the evaluation with near-perfect test accuracy, as every word rhymes with itself. However, adding in-context examples for these instructions creates a misalignment between instruction (induces trivial rhymes) and context (induces non-trivial rhymes), resulting in a significant drop in performance. If we instead score the instructions based on the few-shot metric, this performance drop can be alleviated since the model can choose a more aligned instruction.

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-20_574_1296_1177_415.jpg)

**Figure 8: Few-shot in-context test accuracy on 24 Instruction Induction tasks. APE improves the few-shot in-context learning performance on 21 out of 24 tasks.**

# C. 2 BIG-Bench Instruction Induction

We use APE to generate new prompts for the tasks in BIG-Bench Instruction Induction (BBII). When compared to human prompts, APE-generated prompts improve or match zero-shot performance on 17 out of 21 tasks. We report the normalized preferred metric defined in Srivastava et al. (2022). Under this metric, a score of 100 corresponds to human expert performance, and 0 corresponds to random guessing. Note that a model can achieve a score less than 0 if it performs worse than random guessing on a multiple-choice task.

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-21_696_1296_636_415.jpg)

**Figure 9: APE improves or matches normalized zero-shot performance on 17 out of 21 BIG-Bench Instruction Induction tasks.**

Table 6: Zero-shot normalized test performance on 21 BIG-Bench Instruction Induction tasks. APE improves or matches performance on 17 out of 21 tasks.

\begin{tabular}[t]{|l|l|l|}
\hline \multirow[b]{2}{*}{Task} & \multicolumn{2}{|l|}{Normalized Performance} \\
\hline & Human & APE \\
\hline causal judgment & 18.0 & 18.0 \\
\hline disambiguation qa & -0.4 & 5.6 \\
\hline dyck languages & 3.0 & 18.0 \\
\hline epistemic reasoning & 36.0 & 38.0 \\
\hline gender inclusive sentences german & 13.0 & 22.0 \\
\hline implicatures & 60.0 & 60.0 \\
\hline linguistics puzzles & 0.0 & 0.0 \\
\hline logical fallacy detection & 24.0 & 12.0 \\
\hline movie recommendation & -2.7 & 12.0 \\
\hline navigate & -8.0 & 12.0 \\
\hline object counting & 2.0 & 44.0 \\
\hline operators & 48.0 & 47.0 \\
\hline presuppositions as nli & 13.0 & 5.5 \\
\hline question selection & -2.6 & -0.9 \\
\hline ruin names & 1.3 & -14.7 \\
\hline snarks & 2.0 & 4.0 \\
\hline sports understanding & 36.0 & 36.0 \\
\hline tense & 84.0 & 85.0 \\
\hline winowhy & -12.0 & 12.0 \\
\hline word sorting & 11.0 & 30.0 \\
\hline word unscrambling & 10.0 & 15.0 \\
\hline
\end{tabular}

# C. 3 Zero-shot Chain of Thought Reasoning

We use APE to discover a better chain of thought (CoT) prompt than "Let's think step by step." from Kojima et al. (2022). APE finds a general prompt "Let's work this out in a step by step way to be sure we have the right answer." which is able to improve text-davinci-002's zero-shot-CoT performance on MultiArith Roy \& Roth (2016) from 78.7 to 82.0 and GSM8K Cobbe et al. (2021) 40.7 to 43.0 compared to the original CoT prompt. We include full results on 12 tasks with this new APE CoT prompt in Figure 10

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-22_500_1303_629_412.jpg)

**Figure 10: The performance of APE discovered prompt "Let's work this out in a step by step way to be sure we have the right answer." on the 12 tasks from Kojima et al. (2022). We collect a CoT dataset from the original paper and filter out incorrect answers. We then use APE to optimize the CoT prompt. We improve performance on $6 / 12$ tasks and nearly match human performance on $4 / 12$ tasks. We hypothesize Shuffled Objects and Last Letter are hard to optimize on with a general prompt.**

\begin{table}

**Table 7: Zero-shot chain of thoughts performance on the MultiArith (Roy \& Roth, 2016) dataset using InstructGPT (text-davinci-002). Template (*1) was proposed in Kojima et al. (2022) to enable the zero-shot chain of thoughts reasoning of large language models, while template (*2) and (*3) were used in Ahn et al. (2022) and Reynolds \& McDonell (2021), respectively.**
\begin{tabular}[t]{|l|l|l|l|}
\hline No. & Category & Zero-shot CoT Trigger Prompt & Accuracy \\
\hline 1 & APE & Let's work this out in a step by step way to be sure we have the right answer. & 82.0 \\
\hline 2 & Human-Designed & Let's think step by step. (*1) & 78.7 \\
\hline 3 & & First, (*2) & 77.3 \\
\hline 4 & & Let's think about this logically. & 74.5 \\
\hline 5 & & Let's solve this problem by splitting it into steps. (*3) & 72.2 \\
\hline 6 & & Let's be realistic and think step by step. & 70.8 \\
\hline 7 & & Let's think like a detective step by step. & 70.3 \\
\hline 8 & & Let's think & 57.5 \\
\hline 9 & & Before we dive into the answer, & 55.7 \\
\hline 10 & & The answer is after the proof. & 45.7 \\
\hline - & & (Zero-shot) & 17.7 \\
\hline
\end{tabular}
\end{table}

# C. 4 Quantitative Analysis

Can we use other LLMs for instruction proposal? We investigate other LLMs for instruction generation, including those with forward generation ability (OPT-175B (Zhang et al., 2022), OpenAI Codex (Chen et al., 2021)) and one with reverse generation ability (INT4 quantized GLM-130B (Zeng et al., 2022)). We evaluate their performance on six tasks selected from instruction induction on both zero-shot and few-shot settings ${ }^{6}$ Figures 15 and 16 show that InstructGPT achieves the best performance except for passivization, where it underperforms compared to the two other forwardgeneration models. Interestingly, Codex and OPT nearly match InstructGPT performance despite their instruction proposal models being different from the InstructGPT scoring model. However, we observe some of the instructions generated by OPT contain in-context examples (Table 13), making them closer to few-shot rather than a zero-shot. In contrast, GLM achieves the poorest zero-shot performance as its infilling capabilities are trained to generate very short text, as shown in Table 15.

How important is the meta prompt? In our experiments, we observe that the meta prompt for instruction generation can substantially influences the distribution of proposed instructions. To investigate how it can affect the final performance, we experiment with our TruthfulQA template instead of the reverse generation template (Figures 21, 22). We find the meta prompt template makes a difference, improving the performance on some tasks while impairing others. Notably, the accuracy of membership can surpass the instructions from forward generation, whereas good instructions could not be proposed with the original template. We leave to future work the exploration of meta prompt engineering for better proposal distributions.

How transferable are the generated instructions? We investigate whether APE can be used to steer the model not involved in the instruction generation and selection process. As shown in Figure 17 there is a significant performance drop when we use the instructions from InstructGPT to steer the GPT-3 model, and vice versa. This performance drop can be mitigated by a human written instruction. It suggests that the alignment between the scoring model and execution model is crucial, and the instructions generated by InstructGPT work best for the InstructGPT itself but do not transfer well to a different model like GPT-3. In contrast, GPT-3-generated instructions can steer GPT-3 exceptionally well, outperforming the InstructGPT instructions and human instructions by a large margin. Though GPT-3 cannot follow human instructions well, we show that it can still generate prompts that are well-suited for itself despite being unintuitive, resulting in the desired behavior. We provide the generated prompts in Table 16.

\footnotetext{
${ }^{6}$ These six tasks are chosen such that two of them are worse than humans, and the other four are human-level. They cover six categories (spelling, morphosyntax, lexical semantics, semantics, multi-lingual, and GLUE).
}

