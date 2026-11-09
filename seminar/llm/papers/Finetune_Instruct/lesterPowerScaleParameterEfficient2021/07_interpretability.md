# 7 Interpretability

An ideally interpretable prompt would consist of natural language that clearly describes the task at hand, explicitly asks the model for some result or action, and makes it easy to understand why the prompt elicited such behavior from the model.

As prompt tuning works in the continuous embedding space rather than the discrete token space, interpreting prompts becomes more difficult. To test the interpretability of our learned soft prompts, we compute the nearest neighbors to each prompt token from the frozen model's vocabulary. We use cosine distance between the vocabulary embedding vector and the prompt token representation as the similarity metric.

We observe that for a given learned prompt token, the top-5 nearest neighbors form tight semantic clusters. For example, we see lexically similar clusters such as \{ Technology / technology / Technologies / technological / technologies \}, as well as more diverse but still strongly related clusters such as \{ entirely / completely / totally / altogether /100\% \}. The nature of these clusters suggests that the prompts are in fact learning "word-like" representations. We found that random vectors drawn from the embedding space do not show this sort of semantic clustering.

When initializing the prompts using the "classlabel" strategy, we often find that the class labels persist through training. Specifically, if a prompt token is initialized to a given label, that label is often among the learned token's nearest neighbors after tuning. When initializing with the "Random Uniform" or "Sampled Vocab" methods, the class labels can also be found in the nearest neighbors of the prompts; however they tend to appear as neighbors to multiple prompt tokens. This suggests that the model is learning to store the expected output classes in the prompts as reference, and initializing the prompt to outputs classes makes this easier and more centralized.

When examining longer prompts (e.g. size 100), we often find several prompt tokens with the same nearest neighbors. This suggests there is either excess capacity in the prompt, or that the lack of sequential structure in the prompt representation makes it difficult for the model to localize information to a specific position.

While the learned prompts taken as sequences show little interpretability, we do observe a high frequency of words like science, technology and engineering as the nearest neighbors for prompts trained on the BoolQ dataset and approximately 20\% of the questions are in the "Nature/Science" category. While more investigation is needed, this suggests that one role of the prompt may be to prime the model to interpret inputs in a specific domain or context (e.g. "scientific").

