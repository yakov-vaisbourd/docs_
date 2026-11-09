# A Additional Results

This section shows the full results for all datasets we evaluate. Results for translation and struct to text are shown in Table 1, and the results for eight NLU task clusters are shown in Table 2.

We show FLAN's performance using the best of up to ten instruction templates as well as the template with the best performance on the dev set. For LaMDA-PT, we use the templates from Brown et al. (2020), which were optimized for GPT-3, without performing any prompt engineering to optimize them on our model. For simplicity, we use greedy search for all generative tasks (compared with beam search used in Brown et al. (2020)). Unlike GPT-3, which chooses the number of few-shot exemplars $k$ via best dev set performance, for few-shot LaMDA-PT we choose the highest $k$ that fits in the context length of 1024 tokens, from $k \in\{1,3,5,10\}$.

For DROP (Dua et al., 2019) and SQuADv2 (Rajpurkar et al., 2018), based on email correspondence with Brown et al. (2020), their definition of zero-shot differs from ours in that they actually use exemplars, but only from the same passage as the inference question (each passage has more than one question). Hence, GPT-3 zero-shot results are not directly comparable with ours for DROP and SQuADv2. We mark these results using the ${ }^{\dagger}$ symbol. Moreover, it is unclear how to parse the end of an answer for these two datasets, and so we use curly bracket delimiters \{ and \}, where we expect \} to indicate the end of the answer.

For struct to text, reported T5/mT5 results are from the GEM benchmark paper (Gehrmann et al., 2021), though we do not report their results for DART (through correspondence with authors, we confirmed that their results for DART were incorrect). Though we use a summarization task cluster during instruction tuning, we leave evaluation of summarization for future work, as the mean input of most summarization datasets exceeds FLAN's input length of 1024 tokens.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow{3}{*}{} & \multirow[b]{3}{*}{Metric} & \multirow{3}{*}{Supervised} & \multicolumn{2}{|c|}{\multirow[b]{2}{*}{LaMDA-PT}} & \multicolumn{2}{|c|}{\multirow[b]{2}{*}{GPT-3 175B}} & \multicolumn{6}{|c|}{FLAN 137B} \\
\hline & & & & & & & \multicolumn{2}{|c|}{zero-shot} & \multicolumn{4}{|c|}{few-shot} \\
\hline & & & zero- & fewshot & zeroshot & fewshot $[k]$ & average template & best dev template & average template & best dev template & $[k]$ & \# $t$ \\
\hline \multicolumn{13}{|l|}{Translation} \\
\hline WMT '14 En→Fr BLEU & & $35.0^{d}$ & 11.2 & 31.5 [5] & 25.2 & 32.6 [64] & $32.9_{ \pm 1.1}$ & 33.9 & $33.9 \pm 0.2$ & 33.8 & [9] & 5 \\
\hline WMT '14 Fr→En BLEU & & $45.6^{c}$ & 7.2 & 34.7 [5] & 21.2 & 39.2 [64] & $35.5 \pm 1.3$ & 35.9 & 38.0±0.1 & 37.9 & [9] & 3 \\
\hline WMT '16 En→De BLEU & & $38.6^{f}$ & 7.7 & 26.7 [5] & 24.6 & 29.7 [64] & 25.4±1.8 & 27.0 & 26.8±0.4 & 26.1 & [11] & 5 \\
\hline WMT '16 De→En BLEU & & $41.2^{e}$ & 20.8 & 36.8 [5] & 27.2 & 40.6 [64] & $38.9 \pm 0.3$ & 38.9 & 40.6±0.1 & 40.7 & [11] & 3 \\
\hline WMT '16 En→Ro BLEU & & $39.9^{g}$ & 3.5 & 22.9 [5] & 14.1 & $21.0{ }_{[64]}$ & 16.7±1.6 & 18.9 & 20.5±0.1 & 20.5 & [9] & 5 \\
\hline WMT '16 Ro→En BLEU & & $38.5^{g}$ & 9.7 & 37.5 [5] & 19.9 & 39.5 [64] & $36.8 \pm 0.5$ & 37.3 & $38.2 \pm 0.1$ & 38.1 & [9] & 3 \\
\hline \multicolumn{13}{|l|}{Struct to Text} \\
\hline CommonGen & Rouge-1 & $64.0^{a}$ & 3.9 & 56.7 [3] & - & - & $54.6 \pm 2.3$ & 56.3 & 56.6±0.3 & 56.4 & [16] & 6 \\
\hline & Rouge-2 & $29.4{ }^{a}$ & 1.5 & 29.6 [3] & - & - & $28.8 \pm 2.4$ & 27.6 & 30.9±0.7 & 29.9 & [16] & 6 \\
\hline & Rouge-L & $54.5^{a}$ & 3.2 & 48.5 [3] & - & - & 48.4±1.9 & 48.7 & 50.7±0.2 & 51.0 & [16] & 6 \\
\hline DART & Rouge-1 & - & 11.3 & 56.0 [3] & - & - & $45.5 \pm 4.2$ & 48.9 & 57.9±1.6 & 59.2 & [11] & 7 \\
\hline & Rouge-2 & - & 1.5 & 29.6 [3] & - & - & 25.0±3.7 & 30.0 & 35.8±1.0 & 36.2 & [11] & 7 \\
\hline & Rouge-L & - & 3.2 & 48.5 [3] & - & - & 38.4±3.8 & 43.4 & 48.5±0.9 & 48.2 & [11] & 7 \\
\hline E2ENLG & Rouge-1 & $72.6^{a}$ & 6.2 & 56.7 [3] & - & - & $44.8 \pm 3.9$ & 51.4 & 59.1±1.3 & 59.7 & [12] & 9 \\
\hline & Rouge-2 & $47.5^{a}$ & 2.5 & 31.4 [3] & - & - & $24.2 \pm 3.6$ & 30.1 & 33.2±1.1 & 33.6 & [12] & 9 \\
\hline & Rouge-L & $56.4^{a}$ & 4.9 & 41.1 [3] & - & - & $37.0 \pm 3.5$ & 42.4 & 44.9±0.8 & 45.1 & [12] & 9 \\
\hline WebNLG & Rouge-1 & $83.5^{a}$ & 13.9 & 68.3 [3] & - & - & $50.6 \pm 4.7$ & 57.7 & 68.5±2.2 & 71.2 & [10] & 8 \\
\hline & Rouge-2 & $63.6^{a}$ & 6.9 & $46.0{ }_{[3]}$ & - & - & $29.8 \pm 4.2$ & 35.4 & 48.0±1.5 & 49.8 & [10] & 8 \\
\hline & Rouge-L & $71.0^{a}$ & 11.8 & 56.5 [3] & - & - & 43.4±4.5 & 49.7 & 58.8±1.1 & 60.2 & [10] & 8 \\
\hline
\end{tabular}

**Table 1: Results for translation and struct-to-text tasks. $[k]$ indicates the number of few-shot exemplars. \#t indicates the number of templates that FLAN is evaluated on. ${ **^{a}$ T5-11B, ${ }^{6}$ Edunov et al. (2018), ${ }^{d}$ Durrani et al. (2014), ${ }^{e}$ Wang et al. (2019b), ${ }^{f}$ Sennrich et al. (2016), ${ }^{9}$ Liu et al. (2020).}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow{3}{*}{} & & \multirow{3}{*}{} & \multicolumn{2}{|r|}{\multirow[b]{2}{*}{GLaM}} & \multicolumn{2}{|c|}{\multirow[b]{2}{*}{LaMDA-PT}} & \multicolumn{2}{|c|}{\multirow[b]{2}{*}{GPT-3 175B}} & \multicolumn{5}{|c|}{FLAN 137B} \\
\hline & & & & & & & & & \multicolumn{2}{|c|}{zero-shot} & \multicolumn{3}{|c|}{few-shot} \\
\hline & & & zeroshot & & zeroshot & fewshot $[k]$ & zeroshot & fewshot $[k]$ & average template & best dev template & average template & best dev template & [ $k$ ] \# t \\
\hline \multicolumn{14}{|l|}{NLI} \\
\hline ANLI R1 & 33.3 & $57.4^{b}$ & 40.9 & 42.4 & 39.6 & 39.0 [5] & 34.6 & 36.8 [50] & $47.7 \pm 1.4$ & 46.4 & $44.2 \pm 2.3$ & 47.9 & [6] \\
\hline ANLI R2 & 33.3 & $48.3^{b}$ & 38.2 & 40.0 & 39.9 & 37.5 [5] & 35.4 & 34.0 [50] & $43.9_{ \pm 1.3}$ & 44.0 & 41.6±1.4 & 41.1 & [6] \\
\hline ANLI R3 & 33.3 & $43.5^{b}$ & 40.9 & 40.8 & 39.3 & 40.7 [5] & 34.5 & 40.2 [50] & 47.0±1.3 & 48.5 & $42.8 \pm 2.2$ & 46.8 & [6] 8 \\
\hline CB & 33.3 & $93.6^{a}$ & 33.9 & 73.2 & 42.9 & 34.4 [5] & 46.4 & 82.1 [32] & 64.1±14.7 & 83.9 & 82.6±4.4 & 82.1 & [7] 10 \\
\hline MNLI-m & 33.3 & $92.2^{a}$ & - & - & 35.7 & 43.7 [5] & - & - & $51.1 \pm 6.2$ & 61.2 & 60.8±3.7 & 63.5 & [10] 10 \\
\hline MNLI-mm & 33.3 & $91.9^{a}$ & - & - & 37.0 & 43.8 [5] & - & - & 51.0±6.5 & 62.4 & 61.0±3.5 & 63.5 & [10] 10 \\
\hline QNLI & 50.0 & $96.9^{a}$ & - & - & 50.6 & 55.7 [5] & - & - & 59.6±4.9 & 66.4 & 62.0±1.7 & 63.3 & [12] 9 \\
\hline RTE & 50.0 & $92.5^{a}$ & 68.8 & 71.5 & 73.3 & 70.8 [5] & 63.5 & 72.9 [32] & $78.3 \pm 7.9$ & 84.1 & 79.9±6.9 & 84.5 & [8] 10 \\
\hline SNLI & 33.3 & $91.3^{b}$ & - & - & 33.3 & 54.7 [5] & - & - & $43.0 \pm 7.4$ & 53.4 & $62.3 \pm 2.4$ & 65.6 & [15] 9 \\
\hline WNLI & 50.0 & $94.5^{a}$ & - & - & 56.3 & 64.8 [5] & - & - & $61.0 \pm 10.6$ & 74.6 & $55.4 \pm 11.0$ & 70.4 & [14] 10 \\
\hline \multicolumn{14}{|l|}{Reading Comp.} \\
\hline BoolQ & 50.0 & $91.2^{a}$ & 83.0 & 82.8 & 81.0 & $80.0{ }_{[1]}^{[1]}$ & 60.5 & 77.5 [32] & $80.2 \pm 3.1$ & 82.9 & $83.6 \pm 0.8$ & 84.6 & [4] \\
\hline DROP & - & $80.5^{b}$ & 54.9 & 55.2 & 3.8 & $10.3{ }_{[1]}$ & $23.6^{\dagger}$ & 36.5 [20] & 21.9±0.9 & 22.7 & $22.3 \pm 1.1$ & 23.9 & [2] \\
\hline MultiRC & - & $88.1^{a}$ & 45.1 & 62.0 & 60.0 & 59.6 [5] & 72.9 & 74.8 [32] & 74.5±3.7 & 77.5 & $69.2 \pm 3.2$ & 72.1 & [1] \\
\hline OBQA & 25.0 & $85.4^{a}$ & 53.0 & 55.2 & 41.8 & 50.6 [10] & 57.6 & 65.4 [100] & $77.4 \pm 1.3$ & 78.4 & 77.2±1.3 & 78.2 & [16] \\
\hline SQuADv1 & - & $96.2^{a}$ & - & - & 22.7 & 50.2 [3] & - & - & $79.5 \pm 1.6$ & 80.1 & 82.1±0.5 & 82.7 & [4] \\
\hline SQuADv2 & - & $83.4{ }^{b}$ & 68.3 & 70.0 & 11.1 & 34.9 [3] & $59.5^{\dagger}$ & $69.8{ }_{\text {[16] }}$ & $40.9_{ \pm 1.8}$ & 44.2 & $40.8 \pm 0.9$ & 43.1 & [3] 10 \\
\hline \multicolumn{14}{|l|}{Closed-Book QA} \\
\hline ARC-c & 25.0 & $81.1^{a}$ & 48.2 & 50.3 & 42.0 & 49.4 [10] & 51.4 & 51.5 [50] & 61.7±1.4 & 63.1 & $63.7_{ \pm 0.6}$ & 63.8 & [13] \\
\hline ARC-e & 25.0 & $92.6^{a}$ & 71.9 & 76.6 & 76.4 & 80.9 [10] & 68.8 & 70.1 [50] & 79.5±0.8 & 79.6 & 80.5±0.5 & 80.7 & [14] 7 \\
\hline NQ & - & $36.6^{a}$ & 21.5 & 23.9 & 3.2 & 22.1 [5] & 14.6 & 29.9 [64] & 18.6±2.7 & 20.7 & 27.2±0.5 & 27.6 & [16] 10 \\
\hline TQA (wiki) & - & $60.5^{a}$ & 68.8 & 71.5 & 21.9 & 63.3 [10] & 64.3 & 71.2 [64] & $66.5 \pm 2.6$ & 68.1 & $66.5 \pm 1.0$ & 67.3 & [16] 10 \\
\hline TQA (tfds-dev) & - & $51.0^{a}$ & - & - & 18.4 & 55.1 [10] & & - & $55.0 \pm 2.3$ & 56.7 & $57.2 \pm 0.6$ & 57.8 & [16] 10 \\
\hline \multicolumn{14}{|l|}{Commonsense} \\
\hline COPA & 50.0 & $94.8^{a}$ & 90.0 & 92.0 & 90.0 & $89.0{ }_{[10]}$ & 91.0 & 92.0 [32] & $90.6 \pm 2.0$ & 91.0 & $88.5 \pm 3.8$ & 87.0 & [16] \\
\hline HellaSwag & 25.0 & $47.3^{b}$ & 77.1 & 76.8 & 57.0 & 58.8 [10] & 78.9 & 79.3 [20] & $56.4 \pm 0.5$ & 56.7 & $59.4 \pm 0.2$ & 59.2 & [3] \\
\hline PIQA & 50.0 & $66.8^{b}$ & 80.4 & 81.4 & 80.3* & 80.2* [10] & 81.0 & 82.3 [50] & $80.9^{*} \pm 0.8$ & 80.5* & $82.1^{*} \pm 0.3$ & 81.7* & [10] \\
\hline StoryCloze & 50.0 & $89.2^{b}$ & 82.5 & 84.0 & 79.5 & 83.7 [10] & 83.2 & 87.7 [70] & $92.2 \pm 1.3$ & 93.4 & $93.3 \pm 0.9$ & 94.7 & [10] 8 \\
\hline \multicolumn{14}{|l|}{Sentiment} \\
\hline IMDB & 50.0 & $95.5^{b}$ & - & - & 76.9 & 83.3 [1] & - & - & $94.1 \pm 0.4$ & 94.3 & $94.8 \pm 0.3$ & 95.0 & [2] \\
\hline Sent140 & 50.0 & $87.0^{b}$ & - & - & 41.4 & 63.3 [5] & - & - & $69.9 \pm 2.5$ & 73.5 & $68.7 \pm 1.2$ & 69.3 & [16] \\
\hline SST-2 & 50.0 & $97.5^{a}$ & - & - & 51.0 & 92.3 [5] & 71.6 & 95.6 [8] & $92.6 \pm 1.7$ & 94.6 & $94.4_{ \pm 0.8}$ & 94.6 & [16] \\
\hline Yelp & 50.0 & $98.1^{b}$ & - & - & 84.7 & 89.6 [3] & - & - & $97.8 \pm 0.2$ & 98.1 & 97.9±0.1 & 98.0 & [4] 7 \\
\hline \multicolumn{14}{|l|}{Paraphrase} \\
\hline MRPC & 50.0 & $90.4^{a}$ & - & - & 53.7 & 64.0 [5] & - & - & $69.1 \pm 1.3$ & 69.1 & $67.5 \pm 1.7$ & 67.2 & [10] 10 \\
\hline QQP & 50.0 & $90.6^{a}$ & - & - & 34.9 & 58.9 [3] & - & - & $72.1 \pm 6.8$ & 75.9 & $73.5 \pm 2.9$ & 75.9 & [16] 7 \\
\hline PAWS Wiki & 50.0 & $91.9^{a}$ & - & - & 45.5 & 53.5 [5] & - & - & $61.5 \pm 6.5$ & 69.4 & $66.2 \pm 0.9$ & 70.2 & [10] 10 \\
\hline \multicolumn{14}{|l|}{Coreference} \\
\hline DPR & 50.0 & $84.8^{b}$ & - & - & 54.6 & 57.3 [5] & - & - & $60.3_{ \pm 3.5}$ & 66.8 & $62.4 \pm 1.6$ & 63.3 & [16] 10 \\
\hline Winogrande & 50.0 & $65.8^{b}$ & 73.4 & 73.0 & 68.3 & 68.4 [10] & 70.2 & 77.7 [50] & $67.3 \pm 2.5$ & 71.2 & $72.3_{ \pm 0.9}$ & 72.8 & [16] 10 \\
\hline WSC273 & 50.0 & $70.0^{b}$ & 86.8 & 83.9 & 81.0 & 61.5 [5] & 88.3 & 88.5 [32] & $80.8 \pm 3.7$ & - & - ± - & - & [-] 10 \\
\hline \multicolumn{14}{|l|}{Read. Comp. w/ Commonsense} \\
\hline CosmosQA & 25.0 & $67.1^{b}$ & - & - & 34.1 & 33.8 [5] & - & - & $58.4 \pm 1.3$ & 60.6 & $56.7 \pm 1.3$ & 56.0 & [5] \\
\hline ReCoRD & - & $93.4^{a}$ & 90.3 & 90.3 & 87.8* & 87.6* [1] & 90.2 & 89.0 [32] & $67.8^{*} \pm 3.0$ & 72.5* & $77.0^{*} \pm 2.0$ & 79.0* & [1] 10 \\
\hline
\end{tabular}

**Table 2: Results for eight NLU task clusters. All values shown are for accuracy (or exact match) except DROP, MultiRC, and SQuAD v1 and v2, which are F1. $[k]$ indicates the number of few-shot exemplars. \# $t$ indicates the number of templates that FLAN is evaluated on. ${ **^{a}$ T5-11B, ${ }^{b}$ BERT-large. *see data contamination (Appendix C). WSC273 (Levesque et al., 2012) does not have training or validation sets, and so we do not compute few-shot results for FLAN. For Trivia QA (TQA), we report exact match (EM) on both the wikipedia subset of the dev set to compare with GPT-3, as well as the full TFDS dev set.}
\end{table}

