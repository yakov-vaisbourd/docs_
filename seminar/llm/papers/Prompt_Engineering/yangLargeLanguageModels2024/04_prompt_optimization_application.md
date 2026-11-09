# 4 Application: Prompt Optimization

Next, we demonstrate the effectiveness of OPRO on prompt optimization, where the objective is to find the prompt that maximizes task accuracy. We first introduce the problem setup, then illustrate the meta-prompt design.

## 4.1 Problem Setup

We focus on prompt optimization for natural language tasks, where both the input and output are in the text format. The task is represented as a dataset with training and test splits, where the training set is used to calculate the training accuracy as the objective value during the optimization process, and we compute the test accuracy on the test set after the optimization finishes. While traditional optimization often requires a decently large training set, our experiment shows that a small number or fraction of training samples (e.g., 3.5\% of the training set for GSM8K (Cobbe et al., 2021), 20\% for Big-Bench Hard (Suzgun et al., 2022)) is sufficient. The objective function evaluator is an LLM
```
I have some texts along with their corresponding scores. The texts are arranged in ascending order
based on their scores, where higher scores indicate better quality.
text:
Let’s figure it out!
score:
61
text:
Let’s solve the problem.
score:
63
(... more instructions and scores ...)
```

The following exemplars show how to apply your text: you replace <INS> in each input with your text, then read the input and give an output. We say your output is wrong if your output is different from the given output, and we say your output is correct if they are the same.
input:
Q: Alannah, Beatrix, and Queen are preparing for the new school year and have been given books by their parents. Alannah has 20 more books than Beatrix. Queen has 1/5 times more books than Alannah. If Beatrix has 30 books, how many books do the three have together?
A: <INS>
output:
140
(... more exemplars ...)
Write your new text that is different from the old ones and has a score as high as possible. Write the text in square brackets.

Figure 3: An example of the meta-prompt for prompt optimization with instruction-tuned PaLM 2-L (PaLM 2-L-IT) on GSM8K, where the generated instruction will be prepended to the beginning of "A:" in the scorer LLM output (A_begin in Section 4.1). <INS> denotes the position where the generated instruction will be added. The blue text contains solution-score pairs; the purple text describes the optimization task and output format; the orange text are meta-instructions.
to which the optimized prompt will be applied, and it can be the same or different from the LLM for optimization. We denote the LLM for objective function evaluation as the scorer LLM, and the LLM for optimization as the optimizer LLM.

The output of the optimizer LLM is an instruction, which is concatenated to the question part of every exemplar and prompts the scorer LLM. We consider the following positions to insert the instruction:
\begin{itemize}
\item[-] Q_begin: the instruction is added before the original question.
\item[-] Q_end: the instruction is added after the original question.
\item[-] A_begin: the instruction is added to the beginning of the scorer LLM output. This is applicable to pretrained LLMs without instruction tuning, where the prompt is formatted as a sequence of QA pairs.
\end{itemize}

We exemplify these prompting formats in Appendix B

## 4.2 Meta-Prompt Design

Figure 3 shows an example of the meta-prompt for prompt optimization on GSM8K (Cobbe et al., 2021). More details are as follows.

Optimization problem examples. The problem description includes a few examples taken from the training set to demonstrate the task for the generated instructions. For example, from the input-output pair in Figure 3, we can infer this is a math word problem. The input-output pair also demonstrates the position where the generated instruction will be added to, and this is essential for the optimizer LLM to generate instructions of the same style. In each optimization step, we add several (three for example) training examples to the meta-prompt by random sampling the training set or choose the ones the previous instructions fall short of.

Optimization trajectory. The optimization trajectory includes instructions generated from the past optimization steps, along with their scores. The old instructions and scores are sorted by the score in ascending order. The score is the training accuracy in prompt optimization. We only keep instructions with the highest scores in the meta-prompt in consideration of the LLM context length limit.

Meta-instructions. We also add meta-instructions: the instructions to the optimizer LLM that explain the optimization goal and instruct the model how to use the above information. The meta-instructions may also specify the desired generated instruction format for easier parsing.

