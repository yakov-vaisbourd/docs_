# 2 Problem Statement

While our proposal is agnostic to training objective, we focus on language modeling as our motivating use case. Below is a brief description of the language modeling problem and, in particular, the maximization of conditional probabilities given a task-specific prompt.

Suppose we are given a pre-trained autoregressive language model \(P_{\Phi}(y \mid x)\) parametrized by $\Phi$. For instance, \(P_{\Phi}(y \mid x)\) can be a generic multi-task learner such as GPT (Radford et al., by Brown et al., 2020) based on the Transformer architecture (Vaswani et al., 2017). Consider adapting this pre-trained model to downstream conditional text generation tasks, such as summarization, machine reading comprehension (MRC), and natural language to SQL (NL2SQL). Each downstream task is represented by a training dataset of context-target pairs: \(\mathcal{Z}=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1, . ., N}\), where both $x_{i}$ and $y_{i}$ are sequences of tokens. For example, in NL2SQL, $x_{i}$ is a natural language query and $y_{i}$ its corresponding SQL command; for summarization, $x_{i}$ is the content of an article and $y_{i}$ its summary.

During full fine-tuning, the model is initialized to pre-trained weights $\Phi_{0}$ and updated to $\Phi_{0}+\Delta \Phi$ by repeatedly following the gradient to maximize the conditional language modeling objective:
$$
\max _{\Phi} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left(P_{\Phi}\left(y_{t} \mid x, y_{<t}\right)\right)
$$
One of the main drawbacks for full fine-tuning is that for each downstream task, we learn a different set of parameters $\Delta \Phi$ whose dimension $|\Delta \Phi|$ equals $\left|\Phi_{0}\right|$. Thus, if the pre-trained model is large (such as GPT-3 with $\left|\Phi_{0}\right| \approx 175$ Billion), storing and deploying many independent instances of fine-tuned models can be challenging, if at all feasible.

In this paper, we adopt a more parameter-efficient approach, where the task-specific parameter increment \(\Delta \Phi=\Delta \Phi(\Theta)\) is further encoded by a much smaller-sized set of parameters $\Theta$ with $|\Theta| \ll\left|\Phi_{0}\right|$. The task of finding $\Delta \Phi$ thus becomes optimizing over $\Theta$ :
$$
\max _{\Theta} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left(p_{\Phi_{0}+\Delta \Phi(\Theta)}\left(y_{t} \mid x, y_{<t}\right)\right)
$$
In the subsequent sections, we propose to use a low-rank representation to encode $\Delta \Phi$ that is both compute- and memory-efficient. When the pre-trained model is GPT-3 175B, the number of trainable parameters $|\Theta|$ can be as small as $0.01 \%$ of $\left|\Phi_{0}\right|$.

