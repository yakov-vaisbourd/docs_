# H Ethical Considerations

All considerations in Chowdhery et al. (2022) apply to instruction finetuned models. Additionally, we note that while instruction finetuning improves many zero-shot and few-shot capabilities, downstream developers should consider the full range of potential risk (Weidinger et al., 2022), and anticipate risks specific to their application context (Hasan et al., 2022; Brey, 2012). Design decisions related to prompting, sampling and additional mitigations (eg, safety filters) may lead to new or different forms of bias and harm that can not be evaluated in the instruction finetuned model alone. We recommend appropriate measures be taken to assess risks and potential harms in the application context before deployment.

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline \multicolumn{2}{|c|}{Model Summary} \\
\hline Model architecture & Dense encoder-decoder models of 5 different sizes. See Table 2. \\
\hline Input(s) & The model takes text as input. See https://github.com/google-research/t5x/blob/main/docs/models.md \\
\hline Output(s) & \begin{tabular}[t]{l}
The model generates text as output. See https://github.com/google-research/t5x/blob/main/docs/models.md \\
Usage
\end{tabular} \\
\hline Application & The primary use is research on language models, including: research on zero-shot NLP tasks and in-context few-shot learning NLP tasks, such as reasoning, and question answering; advancing fairness and safety research, and understanding limitations of current large language models. \\
\hline Known Caveats & \begin{tabular}[t]{l}
Language models, including Flan-T5, can potentially be used for language generation in a harmful way, according to Rae et al. (2021a). Flan-T5 should not be used directly in any application, without a prior assessment of safety and fairness concerns specific to the application. \\
System Type
\end{tabular} \\
\hline System Description & This is a standalone model. \\
\hline Upstream Dependencies & None. \\
\hline Upstream Dependencies & None. \\
\hline & Implementation Frameworks \\
\hline Hardware \& Software for Training & Hardware: TPU v3 or TPU v4 (Jouppi et al., 2020). Software: T5X (Roberts et al., 2022), JAX (Bradbury et al., 2018). \\
\hline Hardware \& Software for Deployment & Hardware: TPU v3 or TPU v4 (Jouppi et al., 2020). Software: T5X (Roberts et al., 2022). \\
\hline Compute Requirements & Number of chips $\geq 4$. \\
\hline & Model Characteristics \\
\hline Model Initialization & These models are based on pretrained T5 (Raffel et al., 2020) and fine-tuned with instructions for better zero-shot and few-shot performance. There is one fine-tuned Flan model per T5 model size. \\
\hline Model Status & This is a static model trained on an offline dataset. \\
\hline Model Stats & \begin{tabular}[t]{l}
Flan-T5-small has 77 million weights. Flan-T5-base has 250 million weights. Flan-T5-large has 780 million weights. Flan-T5-XL has 3 billion weights. Flan-T5-XXL has 11 billion weights. See Table 2 for details. \\
Data Overview
\end{tabular} \\
\hline Fine-tuning Dataset & See Section 2.1. \\
\hline Evaluation Dataset & See Section 2.3. \\
\hline & Evaluation Results \\
\hline Evaluation Results & See Table 5. \\
\hline & Model Usage \& Limitations \\
\hline Sensitive Use & Flan-T5 should not be applied for any unacceptable use cases, e.g., generation of abusive speech. \\
\hline Known Limitations & Flan-T5 has not been tested in real world applications. \\
\hline Ethical Considerations \& Risks & Flan-T5 is fine-tuned on a large corpus of text data that was not filtered for explicit content or assessed for existing biases. As a result the model itself is potentially vulnerable to generating equivalently inappropriate content or replicating inherent biases in the underlying data. \\
\hline
\end{tabular}

**Table 26: Flan-T5 model card.**
\end{table}

