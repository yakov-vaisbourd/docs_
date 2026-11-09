# 3 Self-Instruct Data from GPT3

In this section, we apply our method for inducing instruction data to GPT3 as a case study. We use the largest GPT3 LM ("davinci" engine) accessed through the OpenAI API. ${ }^{6}$ The parameters for making queries are described in Appendix A.2. Here we present an overview of the generated data.

## 3.1 Statistics

Table 1 describes the basic statistics of the generated data. We generate a total of over 52 K instructions and more than 82 K instances corresponding to these instructions after filtering.

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline \multicolumn{2}{|l|}{statistic} \\
\hline \# of instructions & 52,445 \\
\hline - \# of classification instructions & 11,584 \\
\hline - \# of non-classification instructions & 40,861 \\
\hline \# of instances & 82,439 \\
\hline - \# of instances with empty input & 35,878 \\
\hline ave. instruction length (in words) & 15.9 \\
\hline ave. non-empty input length (in words) & 12.7 \\
\hline ave. output length (in words) & 18.9 \\
\hline
\end{tabular}

**Table 1: Statistics of the generated data by applying Self-Instruct to GPT3.**
\end{table}

\footnotetext{
${ }^{6}$ https://openai.com/api/
}

## 3.2 Diversity

To study what types of instructions are generated and how diverse they are, we identify the verb-noun structure in the generated instructions. We use the Berkeley Neural Parser ${ }^{7}$ (Kitaev and Klein, 2018; Kitaev et al., 2019) to parse the instructions and then extract the verb that is closest to the root as well as its first direct noun object. 26,559 out of the 52,445 instructions contain such structure; other instructions usually contain more complex clauses (e.g., "Classify whether this tweet contains political content or not.") or are framed as questions (e.g., "Which of these statements are true?"). We plot the top 20 most common root verbs and their top 4 direct noun objects in Figure 3, which account for $14 \%$ of the entire set. Overall, we see quite diverse intents and textual formats in these instructions.

We further study how the generated instructions differ from the seed instructions used to prompt the generation. For each generated instruction, we compute its highest ROUGE-L overlap with the 175 seed instructions. We plot the distribution of these ROUGE-L scores in Figure 4. The results indicate a decent number of new instructions were generated, which do not have much overlap with the seeds. We also demonstrate diversity in the length of the instructions, instance inputs, and instance outputs in Figure 5.

## 3.3 Quality

So far, we have shown the quantity and diversity of the generated data, but its quality remains uncertain. To investigate this, we randomly sample 200 instructions and randomly select 1 instance per instruction. We asked an expert annotator (author of this work) to label whether each instance is correct or not, in terms of the instruction, the instance input, and the instance output. Evaluation results in Table 2 show that most of the generated instructions are meaningful, while the generated instances may contain more noise (to a reasonable extent). However, we found that even though the generations may contain errors, most of them are still in the correct format or partially correct, which can provide useful guidance for training models to follow instructions. We listed a number of good examples and bad examples in Table 10 and 11, respectively.

\footnotetext{
${ }^{7}$ https://parser.kitaev.io/
}

![Figure](./wangSelfInstructAligningLanguage2023_images/c0364e9b-aeb3-41c0-b14a-38c8a8857c92-05_877_884_192_240.jpg)

**Figure 3: The top 20 most common root verbs (inner circle) and their top 4 direct noun objects (outer circle) in the generated instructions. Despite their diversity, the instructions shown here only account for 14\% of all the generated instructions because many instructions (e.g., "Classify whether the user is satisfied with the service.") do not contain such a verb-noun structure.**

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Quality Review Question & Yes \% \\
\hline Does the instruction describe a valid task? & 92\% \\
\hline Is the input appropriate for the instruction? & 79\% \\
\hline Is the output a correct and acceptable response to the instruction and input? & 58\% \\
\hline All fields are valid & 54\% \\
\hline
\end{tabular}

**Table 2: Data quality review for the instruction, input, and output of the generated data. See Table 10 and Table 11 for representative valid and invalid examples.**
\end{table}

