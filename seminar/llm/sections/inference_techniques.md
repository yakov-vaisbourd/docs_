# Inference Techniques

## In-Context Learning (ICL)

A test query is prefixed with a few demonstration examples inside the prompt. The model generates an answer by analogy with those examples, with no gradient update to its weights. A theoretical interpretation: for softmax-free linear attention, prepending ICL examples produces an output of the form

$$
\begin{aligned}
\mathcal{F}_\mathrm{ICL}(\mathbf{z}_{1:n}) = (\Delta W_{ICL} + W_{ZSL})\,W_Q\,\mathbf{z}_{1:n}
\end{aligned}
$$

where $\Delta W_{ICL} = \sum_i W_V \mathbf{z}_{icl,i}(W_K \mathbf{z}_{icl,i})^T$ has the same form as a gradient descent weight update - the model is meta-optimised toward the demonstrations without backpropagation.

## Retrieval-Augmented Generation (RAG)

At inference, relevant documents or past experiences are fetched from an external database and prepended to the prompt as additional context. The LLM then conditions its output on both the query and the retrieved material. RAG avoids retraining: new knowledge is simply inserted into the database. Its ceiling is database coverage - scenarios absent from the database fall back to zero-shot behaviour.

## Zero-Shot Generalisation

The model is applied to a domain it has never been trained on, relying entirely on its pre-trained priors (or, in the RAG case, on retrieved examples from a different-domain database). Distinguished from few-shot: zero-shot means no target-domain labels are used during training, but ICL examples may still be provided at inference time.

