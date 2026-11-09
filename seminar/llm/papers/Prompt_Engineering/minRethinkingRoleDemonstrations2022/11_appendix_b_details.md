# B Experimental Details

Example template We follow Ye et al. (2021); Min et al. (2021b); Logan IV et al. (2021) in using the minimal format to transform the input to a sequence (e.g. a concatenation of multiple inputs) and using the label words from each dataset as it is. We also explore manual templates taken from prior work (Holtzman et al., 2021; Zhao et al., 2021) as reported in Section 4.2, although we find that using these templates is not consistently better than using minimal templates. We thus run main experiments with minimal templates. Example templates are provided in Table 3.

Format of the demonstrations We follow the standard of each model for formatting the demonstrations, either from exploration in prior work or the example code provided in the official tutorial. For GPT-2, we separate the input and the label,

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Dataset & \# Train & \# Eval \\
\hline \multicolumn{3}{|l|}{Task category: Sentiment analysis} \\
\hline financial_phrasebank & 1,811 & 453 \\
\hline poem_sentiment & 892 & 105 \\
\hline \multicolumn{3}{|l|}{Task category: Paraphrase detection} \\
\hline medical_questions_pairs & 2,438 & 610 \\
\hline glue-mrpc & 3,668 & 408 \\
\hline \multicolumn{3}{|l|}{Task category: Natural language inference} \\
\hline glue-wnli & 635 & 71 \\
\hline climate_fever & 1,228 & 307 \\
\hline glue-rte & 2,490 & 277 \\
\hline superglue-cb & 250 & 56 \\
\hline sick & 4,439 & 495 \\
\hline \multicolumn{3}{|l|}{Task category: Hate speech detection} \\
\hline hate_speech18 & 8,562 & 2,141 \\
\hline ethos-national_origin & 346 & 87 \\
\hline ethos-race & 346 & 87 \\
\hline ethos-religion & 346 & 87 \\
\hline tweet_eval-hate & 8,993 & 999 \\
\hline tweet_eval-stance_atheism & 461 & 52 \\
\hline tweet_eval-stance_feminist & 597 & 67 \\
\hline \multicolumn{3}{|l|}{Task category: Question answering} \\
\hline quarel & 1,941 & 278 \\
\hline openbookqa & 4,957 & 500 \\
\hline qasc & 8,134 & 926 \\
\hline commonsense_qa & 9,741 & 1,221 \\
\hline ai2_arc & 1,119 & 299 \\
\hline \multicolumn{3}{|l|}{Task category: Sentence completion} \\
\hline codah & 1665 & 556 \\
\hline superglue-copa & 400 & 100 \\
\hline dream & 6116 & 2040 \\
\hline quartz-with_knowledge & 2696 & 384 \\
\hline quartz-no_knowledge & 2696 & 384 \\
\hline
\end{tabular}

**Table 2: 26 datasets used for experiments, classified into 6 task categories. \# Train and \# Test indicate the number of training and test examples of the dataset. Note that \# train is based on the original training dataset but we use $k$ random samples for $k$-shot evaluation.**
\end{table}
and each demonstration example with a space. For MetaICL, GPT-J and GPT-3, we separate the input and the label with a newline (\n), and each demonstration example with three newlines. For fairseq models, we use a newline to separate the input and the label as well as each demonstration example.

Details in variants of the demonstrations For "demonstrations w/ $a \%$ accurate labels" $(0 \leq$ \(a \leq 100)\), we use $k \times a / 100$ correct pairs and \(k \times(1-a / 100)\) incorrect pairs in a random order, as described in Algorithm 1. For "OOD demonstrations", we use CC-News (Nagel, 2016) as an external corpus. We consider the length of the text during sampling, so that sampled sentences have similar length to the test input. For "demonstrations with random English words", we use pypi.org/ project/english-words for the set of En-

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-14_580_1491_228_322.jpg)

**Figure 11: Results of No-demonstration, Gold demonstration and Random demonstration on 3 classification datasets (top) and 3 multi-choice datasets (bottom). Details in Section 4.1. This figure is for providing numbers that are comparable across models-full results with more datasets are reported in Figure 3.**

\begin{lstlisting}[mathescape=true]
Algorithm 1 Forming the demonstrations with an
accuracy of $a \%$.
procedure FormDemons \(\left(\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{k}, a\right)\)
$D \leftarrow[]$ // demonstration to be formed
$n \leftarrow k \times a / 100$ // number of correct pairs
\(\mathcal{G} \leftarrow \operatorname{Sample}(\operatorname{Range}(1, k), n)\)
for \(i \in \operatorname{Range}(1, k)\) do
if $i \in \mathcal{G}$ then $/ /$ add correct pair
\(D . \operatorname{append}\left(\left(x_{i}, y_{i}\right)\right)\)
else // add incorrect pair
D.append $\left(\left(x_{i}\right.\right.$, Sample \(\left.\left.\left(\mathcal{C}-y_{i}\right)\right)\right)\)
return $D$
\end{lstlisting}

glish words, which consists of 61,569 words.
Table 4 provides a list of example demonstrations for each method used in Section 5.

