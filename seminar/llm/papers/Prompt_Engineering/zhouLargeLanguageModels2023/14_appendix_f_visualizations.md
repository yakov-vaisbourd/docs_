# F Additional Visualizations

Visualization Hyperparameters As we tuned the hyperparameters of APE including the number of proposals generated per demonstration and the number of demonstrations per random seed, we discovered better ones for instruction induction. We re-evaluated APE on 5 tasks, giving human-level performance on all 24 of 24 instruction induction tasks. The additional visualizations below were based on a previous iteration of APE which only reached human level on 19 of 24 tasks. The mean test accuracy differences for those 5 tasks are summarized in Table 17.

\begin{table}

**Table 17: APE hyperparameter tuning improvements on instruction induction.**
\begin{tabular}[t]{|l|l|l|l|}
\hline Task Name & APE (Old) Accuracy, Mean & APE (New) Accuracy, Mean & APE (New) - Human \\
\hline Second Letter & 0.596 & 0.8 & 0.034 \\
\hline Pluralization & 0.984 & 0.996 & -0.004 \\
\hline Passivization & 0.622 & 1 & 0.001 \\
\hline Sentence Similarity & 0.186 & 0.256 & -0.01 \\
\hline Membership & 0.126 & 0.612 & -0.001 \\
\hline
\end{tabular}
\end{table}

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-34_574_1296_1220_415.jpg)

**Figure 14: Few-shot in-context test accuracy of best performing instructions selected using few-shot execution accuracy on 24 Instruction Induction tasks.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-34_322_1296_2093_415.jpg)

**Figure 15: Zero-shot test accuracy on 6 Instruction Induction tasks. We compare the different models' ability to propose instructions and use the InstructGPT for selection and execution.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-35_322_1296_294_415.jpg)

**Figure 16: Few-shot test accuracy on 6 Instruction Induction tasks. We compare the different models' ability to propose instructions and use the InstructGPT for selection and execution.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-36_692_1296_430_410.jpg)

**Figure 17: Zero-shot test accuracy on 6 Instruction Induction tasks. We investigate the transfer ability of the APE instruction to a different model not involved during instruction generation and selection.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-36_700_1308_1547_408.jpg)

**Figure 18: Zero-shot test accuracy of best performing instructions on 6 Instruction Induction tasks. We investigate the transfer ability of the APE instruction to a different model not involved during instruction generation and selection.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-37_692_1300_434_410.jpg)

**Figure 19: Few-shot test accuracy on 6 Instruction Induction tasks. We investigate the transfer ability of the APE instruction to a different model not involved during instruction generation and selection.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-37_696_1306_1555_408.jpg)

**Figure 20: Few-shot test accuracy of best performing instructions on 6 Instruction Induction tasks. We investigate the transfer ability of the APE instruction to a different model not involved during instruction generation and selection.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-38_483_1354_290_387.jpg)

**Figure 23: Zero-shot test accuracy on 24 Instruction Induction tasks using eight different LLMs.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-38_322_1296_932_415.jpg)

**Figure 21: Zero-shot test accuracy on 6 Instruction Induction tasks. We compare the performance of different templates used to propose instruction. Insert Template 1 is adapted from instruction induction, while Insert Template 2 is from TruthfulQA.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-38_322_1291_1508_417.jpg)

**Figure 22: Few-shot test accuracy on 6 Instruction Induction tasks. We compare the performance of different templates used to propose instruction. Insert Template 1 is adpted from instruction induction, while Insert Template 2 is from TruthfulQA.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-39_571_1296_309_415.jpg)

**Figure 24: Zero-shot test accuracy on 24 Instruction Induction tasks using two different metrics and two different LLM models.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-39_574_1293_1068_417.jpg)

**Figure 25: In-Context learning without instruction on 24 Instruction Induction tasks using two different metrics and two different LLM models.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-39_571_1293_1828_417.jpg)

**Figure 26: Test accuracy of in-Context learning with instruction on 24 Instruction Induction tasks using two different metrics and two different LLM models.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-40_618_1285_495_416.jpg)

**Figure 27: Survival function and the histogram of test accuracy on a simple task (i.e. Pluralization)**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-40_615_1285_1636_416.jpg)

**Figure 28: Survival function and the histogram of test accuracy on a challenging task (i.e. Start With)**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-41_609_1342_482_387.jpg)

**Figure 29: Iterative Monte Carlo search improves the quality of the instruction candidates at each round. Task: Antonyms.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-41_609_1339_1620_389.jpg)

**Figure 30: Iterative Monte Carlo search improves the quality of the instruction candidates at each round. Task: Cause Selection.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-42_609_1342_482_387.jpg)

**Figure 31: Iterative Monte Carlo search improves the quality of the instruction candidates at each round. Task: Passivization.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-42_609_1339_1620_389.jpg)

**Figure 32: Iterative Monte Carlo search improves the quality of the instruction candidates at each round. Task: Second Letter.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-43_609_1342_482_387.jpg)

**Figure 33: Iterative Monte Carlo search improves the quality of the instruction candidates at each round. Task: Sentiment.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-43_609_1339_1620_389.jpg)

**Figure 34: Iterative Monte Carlo search improves the quality of the instruction candidates at each round. Task: Translation en-fr.**

