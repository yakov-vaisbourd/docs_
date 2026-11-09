# Changes from V4 to V5
\begin{itemize}
\item[-] Replaced the tables in the main figure with a figure, which takes up less space and focuses on zero-shot performance.
\item[-] Added GLaM 64B/64E as a baseline.
\item[-] Moved the ablation about the role of instructions, as well as prompt tuning, into the main paper (and condensed the figures).
\end{itemize}

# Changes to V4 from V3
\begin{itemize}
\item[-] We added a Frequently Asked Questions section (Appendix E).
\item[-] We added a section with qualitative examples (Appendix F).
\item[-] We added an additional ablation study on the role of instructions during finetuning (Appendix B.2).
\item[-] We updated the related work (Appendix D) with manuscripts posted on arxiv since our initial preprint.
\end{itemize}

# Changes to V3 from V2
\begin{itemize}
\item[-] The number of tokens used in pretraining was corrected from 2.81T to 2.49T tokens.
\end{itemize}

# Changes to V2 from V1
\begin{itemize}
\item[-] We updated the terminology to "datasets" and "task clusters."
\item[-] We renamed the previous "open-domain QA" task cluster to "closed-book QA."
\item[-] We extended the related work section and moved it to the Appendix D, using a shorter version in the main body.
\item[-] We added FLAN and LaMDA-PT results for additional datasets for which GPT-3 results were not reported.
\item[-] For TriviaQA, v1 reported results on the tfds dev set of 11,313 examples. GPT-3 actually evaluates on the wikipedia dev set of 7,993 examples, so we ran an additional evaluation on that dev set in order to compare with GPT-3's performance. Zero-shot FLAN now beats zero-shot GPT-3 on that task (and therefore on 20 of 25 tasks). We still show the original result in Table 2, though there is no GPT-3 result to compare with.
\item[-] We moved commonsense reasoning and coreference resolution from the main body to the Appendix.
\item[-] We moved prompt tuning from the main body to §4.5
\item[-] We added data contamination analysis (Appendix C).
\item[-] We added few-shot instruction tuning (§4.4).
\item[-] We cited additional datasets in Appendix G.
\item[-] The number of tokens used in pretraining was corrected from 2.81T to 2.49T tokens.
\end{itemize}

