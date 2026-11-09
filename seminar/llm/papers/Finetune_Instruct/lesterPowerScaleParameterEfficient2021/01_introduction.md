# 1 Introduction

With the wide success of pre-trained large language models, a range of techniques has arisen to adapt these general-purpose models to downstream tasks. ELMo (Peters et al., 2018) proposed freezing the pre-trained model and learning a task-specific weighting of its per-layer representations. However, since GPT (Radford et al., 2018) and BERT (Devlin et al., 2019), the dominant adaptation technique has been model tuning (or "fine-tuning"), where all model parameters are tuned during adaptation, as proposed by Howard and Ruder (2018).

\footnotetext{
*Work done as a Google AI Resident.
}

![Figure](./lesterPowerScaleParameterEfficient2021_images/cdaa5bee-4b30-4a3a-a4df-bdd3721f9de9-01_580_676_745_1087.jpg)

**Figure 1: Standard model tuning of T5 achieves strong performance, but requires storing separate copies of the model for each end task. Our prompt tuning of T5 matches the quality of model tuning as size increases, while enabling the reuse of a single frozen model for all tasks. Our approach significantly outperforms fewshot prompt design using GPT-3. We show mean and standard deviation across 3 runs for tuning methods.**

More recently, Brown et al. (2020) showed that prompt design (or "priming") is surprisingly effective at modulating a frozen GPT-3 model's behavior through text prompts. Prompts are typically composed of a task description and/or several canonical examples. This return to "freezing" pre-trained models is appealing, especially as model size continues to increase. Rather than requiring a separate copy of the model for each downstream task, a single generalist model can simultaneously serve many different tasks.

Unfortunately, prompt-based adaptation has several key drawbacks. Task description is error-prone and requires human involvement, and the effectiveness of a prompt is limited by how much conditioning text can fit into the model's input. As a result, downstream task quality still lags far behind that of tuned models. For instance, GPT-3 175B fewshot performance on SuperGLUE is 17.5 points be-

![Figure](./lesterPowerScaleParameterEfficient2021_images/cdaa5bee-4b30-4a3a-a4df-bdd3721f9de9-02_385_770_245_241.jpg)

**Figure 2: Model tuning requires making a taskspecific copy of the entire pre-trained model for each downstream task and inference must be performed in separate batches. Prompt tuning only requires storing a small task-specific prompt for each task, and enables mixed-task inference using the original pretrained model. With a T5 "XXL" model, each copy of the tuned model requires 11 billion parameters. By contrast, our tuned prompts would only require 20,480 parameters per task-a reduction of over five orders of magnitude-assuming a prompt length of 5 tokens.**

low fine-tuned T5-XXL (Raffel et al., 2020) (71.8 vs. 89.3) despite using 16 times more parameters.

Several efforts to automate prompt design have been recently proposed. Shin et al. (2020) propose a search algorithm over the discrete space of words, guided by the downstream application training data. While this technique outperforms manual prompt design, there is still a gap relative to model tuning.

Li and Liang (2021) propose "prefix tuning" and show strong results on generative tasks. This method freezes the model parameters and backpropagates the error during tuning to prefix activations prepended to each layer in the encoder stack, including the input layer. Hambardzumyan et al. (2021) simplify this recipe by restricting the trainable parameters to the input and output subnetworks of a masked language model, and show reasonable results on classifications tasks.

In this paper, we propose prompt tuning as a further simplification for adapting language models. We freeze the entire pre-trained model and only allow an additional $k$ tunable tokens per downstream task to be prepended to the input text. This "soft prompt" is trained end-to-end and can condense the signal from a full labeled dataset, allowing our method to outperform few-shot prompts and close the quality gap with model tuning (Figure 1). At the same time, since a single pre-trained model is recycled for all downstream tasks, we retain the efficient serving benefits of frozen models (Figure 2).

While we developed our method concurrently with Li and Liang (2021) and Hambardzumyan et al. (2021), we are the first to show that prompt tuning alone (with no intermediate-layer prefixes or task-specific output layers) is sufficient to be competitive with model tuning. Through detailed experiments in sections 2-3, we demonstrate that language model capacity is a key ingredient for these approaches to succeed. As Figure 1 shows, prompt tuning becomes more competitive with scale.

We compare with similar approaches in Section 4. Explicitly separating task-specific parameters from the "generalist" parameters needed for general language-understanding has a range of additional benefits. We show in Section 5 that by capturing the task definition in the prompt while keeping the generalist parameters fixed, we are able to achieve better resilience to domain shifts. In Section 6, we show that "prompt ensembling", learning multiple prompts for the same task, can boost quality and is more efficient than classic model ensembling. Finally, in Section 7, we investigate the interpretability of our learned soft prompts. In sum, our key contributions are:
\begin{itemize}
\item[1.] Proposing prompt tuning and showing its competitiveness with model tuning in the regime of large language models.
\item[2.] Ablating many design choices, and showing quality and robustness improve with scale.
\item[3.] Showing prompt tuning outperforms model tuning on domain shift problems.
\item[4.] Proposing "prompt ensembling" and showing its effectiveness.
\end{itemize}

