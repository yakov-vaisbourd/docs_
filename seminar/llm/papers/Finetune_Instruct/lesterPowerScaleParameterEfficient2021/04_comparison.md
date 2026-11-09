# 4 Comparison to Similar Approaches

In this section, we review recent work on learning continuous prompts, and draw comparisons with our method. One important axis of comparison is the number of task-specific parameters each method requires, as shown in Figure 4. Among methods with learnable parameters, prompt tuning is the most parameter efficient, requiring less than 0.01\% task-specific parameters for models over a billion parameters. ${ }^{9}$

Li and Liang (2021) propose "prefix tuning": learning a sequence of prefixes that are prepended

\footnotetext{
${ }^{8}$ https://github.com/google-research/ text-to-text-transfer-transformer/ blob/main/released_checkpoints.md\# lm-adapted-t511lm100k
${ }^{9}$ To compare with prompt design, we count each token ID in the prompt as a parameter, and assume a prompt of between 500-2000 tokens to match the GPT-3 setting. While this technique is by far the most parameter efficient, it comes at the cost of task quality.
}
at every transformer layer. This is akin to learning transformer activations that are fixed across examples at every network layer. In contrast, prompt tuning uses a single prompt representation that is prepended to the embedded input. Beyond requiring fewer parameters, our approach allows the transformer to update the intermediate-layer task representations, as contextualized by an input example. Their work builds on GPT-2 (Radford et al., 2019) and BART (Lewis et al., 2020), while ours focuses on T5 and examines changes in performance and robustness to design choices as model size increases. When using BART, prefix tuning includes prefixes on both the encoder and decoder network, while prompt tuning only requires prompts on the encoder. Li and Liang (2021) also rely on a reparameterization of the prefix to stabilize learning, which adds a large number of parameters during training, whereas our configuration does not require this reparameterization and is robust across SuperGLUE tasks and model sizes.

Hambardzumyan et al. (2021) propose "WARP", where prompt parameters are added to the input layer. This method works with masked language models, relying on a [MASK] token and a learnable output layer to project the mask to class logits. This formulation restricts the model to producing a single output, limiting it to classification. Prompt tuning does not require any changes to the input or a task-specific head. The performance of prompt tuning is also considerably closer to the strong performance of model tuning.

Liu et al. (2021) propose "P-tuning" where learnable continuous prompts are interleaved throughout the embedded input, using patterns based on human design. Our approach removes this complication by simply prepending the prompt to the input. To achieve strong SuperGLUE results, P-tuning has to be used in conjunction with model tuning, that is, models jointly update both the prompt and the main model parameters, whereas our approach keeps the original language model frozen. ${ }^{10}$

Qin and Eisner (2021) use "soft words" to learn prompts to extract knowledge from pre-trained LMs. Prompts are positioned in relation to the input based on hand-designed prompt prototypes, and a learned $\Delta_{i}^{\ell}$ parameter is included for each layer, so parameter cost scales with model depth.

\footnotetext{
${ }^{10}$ As another difference, P-tuning requires the addition of "anchor" tokens in the input (e.g. a question mark following the hypothesis in the RTE task) to achieve strong performance, while prompt tuning leaves inputs untouched.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Dataset & Domain & Model & Prompt & $\Delta$ \\
\hline SQuAD & Wiki & $94.9 \pm 0.2$ & $94.8 \pm 0.1$ & -0.1 \\
\hline TextbookQA & Book & $54.3 \pm 3.7$ & $66.8 \pm 2.9$ & +12.5 \\
\hline BioASQ & Bio & 77.9 ±0.4 & 79.1 ±0.3 & +1.2 \\
\hline RACE & Exam & $59.8 \pm 0.6$ & $\mathbf{6 0 . 7} \pm 0.5$ & +0.9 \\
\hline RE & Wiki & 88.4 ±0.1 & $88.8 \pm 0.2$ & +0.4 \\
\hline DuoRC & Movie & 68.9 ±0.7 & $67.7 \pm 1.1$ & -1.2 \\
\hline DROP & Wiki & 68.9 ±1.7 & $67.1 \pm 1.9$ & -1.8 \\
\hline
\end{tabular}

**Table 1: F1 mean and stddev for models trained on SQuAD and evaluated on out-of-domain datasets from the MRQA 2019 shared task. Prompt tuning tends to give stronger zero-shot performance than model tuning, especially on datasets with large domain shifts like TextbookQA.**
\end{table}

Logeswaran et al. (2020) use a learnable prepended token to adapt transformer models to various tasks, but focus on small synthetic datasets designed to accommodate a compositional task representation, as opposed to larger real-world datasets. Their base models are small transformers trained from scratch jointly with the task representations, whereas we keep the base model frozen and investigate scaling laws using larger transformers.

More generally, work on task prompts is closely aligned with work on "adapters" (Rebuffi et al., 2017; Houlsby et al., 2019), small bottleneck layers inserted between frozen pre-trained network layers. Adapters offer another means of reducing task-specific parameters, with Houlsby et al. (2019) achieving GLUE performance close to full model tuning when freezing BERT-Large and only adding 2-4\% additional parameters. Pfeiffer et al. (2020) use multiple adapters in a multilingual context to explicitly separate language understanding from task specification, similar to our approach. A core difference between adapters and prompt tuning is how the approaches change model behavior. Adapters modify the actual function that acts on the input representation, parameterized by the neural network, by allowing the rewriting of activations at any given layer. Prompt tuning modifies behavior by leaving the function fixed and adding new input representations that can affect how subsequent input is processed.

