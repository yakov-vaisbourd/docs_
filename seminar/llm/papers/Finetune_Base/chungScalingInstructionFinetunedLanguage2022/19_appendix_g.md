# G Model cards

# G. 1 Flan-PaLM

Flan-PaLM shares the same model architecture, implementation frameworks, usage and limitations, as the PaLM model (Chowdhery et al., 2022). We show parts of the model card (Mitchell et al., 2019) that are specific to Flan-PaLM in Table 25.

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline & Model Characteristics \\
\hline Model Initialization & The model is initialized from PaLM (Chowdhery et al., 2022). We also have a variation initialized from U-PaLM (Tay et al., 2022b). \\
\hline Model Stats & \begin{tabular}[t]{l}
For each size PaLM model, we fine tuned an equivalent version of Flan. The largest models, Flan-PaLM and Flan-U-PaLM, have 540 billion parameters. We also have 62-billion and 6-billion parameter models. See Table 2 for details. \\
Usage
\end{tabular} \\
\hline Application & \begin{tabular}[t]{l}
The primary use is research on language models, including: zero-shot and incontext few-shot learning NLP tasks, such as reasoning, and question answering; advancing fairness and safety research, and understanding limitations of current LLMs. \\
Data Overview
\end{tabular} \\
\hline Fine-tuning Dataset & See Section 2.1. \\
\hline Evaluation Dataset & See Section 2.3. \\
\hline & Evaluation Results \\
\hline Evaluation Results & See Table 5. \\
\hline
\end{tabular}

**Table 25: Flan-PaLM model card. The model summary, system type, implementation frameworks, and model usage \& limitations are the same as the original PaLM (Chowdhery et al., 2022). See the model card of PaLM for details.**
\end{table}

# G. 2 Flan-T5

We show the model card of Flan-T5 in Table 26.

