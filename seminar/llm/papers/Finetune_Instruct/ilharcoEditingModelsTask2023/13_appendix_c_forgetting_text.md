# C Forgetting with text generation

This section presents additional experimental details and results complementing the findings presented in Section 3.2, showcasing the effect of negating task vectors from text generation.

# C. 1 Experimental details

To obtain task vectors, we fine-tune on data Civil Comments [8] where the toxicity score is larger than 0.8. We then fine-tune GPT-2 models [77] from Hugging Face transformers library [97]. We use a learning rate of 1e-5, and fine-tune with a causal language modeling objective with the AdamW optimizer for 5 epochs using a global batch size of 32. After fine-tuning, we evaluate models obtained by adding task vectors with scaling coefficients $\lambda \in\{0.0,0.1, \cdots, 1.0\}$. In Table 2, we report results for the largest scaling coefficient such that perplexity is still within 0.5 points of the perplexity of the pre-trained model. To evaluate toxicity, we generate 1000 samples from the models. To encourage a higher chance of toxic generations, we condition the generations using the prefix "I don't care if this is controversial". In early experiments, we also tried other prompts, which lead to similar qualitative results. We evaluate other prompts in Appendix C.3. To evaluate fluency, we measure the perplexity of the models on WikiText-103 with a striding window of size 1024 and a stride of 512 tokens.

# C. 2 Additional models

In addition to the GPT-2 Large models showed in Table 2, we present results for GPT-2 Medium and GPT-2 Small models in Tables 9 and 10. We observe the same qualitative trends for the additional models. As in image classification, we also find that applying task vectors is more effective for larger models.

# C. 3 RealToxicityPrompts

We present additional experiments using RealToxicityPrompts [29], a dataset of natural language prompts used for measuring toxicity in language models. As in Gehman et al. [29], we evaluate language models using 25 generations per prompt, using the Perspective API. ${ }^{8}$

\footnotetext{
${ }^{7}$ https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset.
${ }^{8}$ https://github.com/conversationai/perspectiveapi
}

\begin{table}

**Table 9: Making language models less toxic with negative task vectors. Results are shown for the GPT-2 Medium model.**
\begin{tabular}[t]{|l|l|l|l|}
\hline Method & \% toxic generations (↓) & Avg. toxicity score (↓) & WikiText-103 perplexity (↓) \\
\hline Pre-trained & 4.3 & 0.06 & 18.5 \\
\hline Fine-tuned & 54.5 & 0.54 & 20.2 \\
\hline Gradient ascent & 0.0 & 0.00 & $>10^{10}$ \\
\hline Random task vector & 4.2 & 0.05 & 18.5 \\
\hline Negative task vector & 1.8 & 0.02 & 18.9 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 10: Making language models less toxic with negative task vectors. Results are shown for the GPT-2 Small model.**
\begin{tabular}[t]{|l|l|l|l|}
\hline Method & \% toxic generations (↓) & Avg. toxicity score (↓) & WikiText-103 perplexity (↓) \\
\hline Pre-trained & 3.7 & 0.04 & 25.2 \\
\hline Fine-tuned & 62.9 & 0.61 & 28.1 \\
\hline Gradient ascent & 0.0 & 0.00 & $>10^{10}$ \\
\hline Random task vector & 3.2 & 0.04 & 25.3 \\
\hline Negative task vector & 2.5 & 0.03 & 25.3 \\
\hline
\end{tabular}
\end{table}

In Figure 13, we present results showing the expected maximum toxicity across the 25 generations and the perplexity on WikiText-103 as we vary the scaling coefficients. We show results both for the challenging subset of the dataset, containing 1.2k prompts, and for a random subset of the full dataset with one thousand prompts. In both cases, we see qualitatively similar trends: negating task vectors produced by fine-tuning on toxic data reduces the amount toxicity of the generations. For GPT-2 large, we see close to vertical movement as the scaling coefficient increases, showing large decreases in accuracy with little change in perplexity on WikiText-103. However, especially for the challenging set of the benchmark, there is still significant headroom for improvement.

