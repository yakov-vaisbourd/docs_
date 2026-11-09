# 4 Multi-time Retrieval Baselines

Existing passive multi-time retrieval augmented LMs can also be formulated using our framework (subsection 2.3). In this section, we formally introduce three baseline categories based on when and what to retrieve. These baselines are not exact reproductions of the corresponding paper because many design choices differ which makes direct comparisons impossible. We implemented them using the same settings, with the only variation being when and what to retrieve.

Previous-window approaches trigger retrieval every $l$ tokens, where $l$ represents the window size. Generated tokens from the previous window are used as the query:
$$
\begin{aligned}
\boldsymbol{q}_{t} & =\boldsymbol{y}_{t-1} \quad(t \geq 2), \\
\boldsymbol{y}_{t} & =\left[w_{(t-1) l+1}, \ldots, w_{t l}\right] .
\end{aligned}
$$
Some existing methods in this category are RETRO (Borgeaud et al., 2022), IC-RALM (Ram et al.,

\footnotetext{
${ }^{2}$ https://api.openai.com/v1/completions April 23.
${ }^{3}$ https://www.microsoft.com/en-us/bing/apis/ bing-web-search-api
}
2023), which retrieve every few tokens, and KNN-LM (Khandelwal et al., 2020), which retrieves every token. ${ }^{4}$ We follow Ram et al. (2023) to use a window size of $l=16$.

Previous-sentence approaches trigger retrieval every sentence and use the previous sentence as the query, and IRCoT (Trivedi et al., 2022) belongs to this category:
$$
\begin{aligned}
& \boldsymbol{q}_{t}=\boldsymbol{y}_{t-1} \quad(t \geq 2), \\
& \boldsymbol{y}_{t}=\boldsymbol{s}_{t} .
\end{aligned}
$$

Question decomposition approaches manually annotated task-specific exemplars to guide LMs to generate decomposed sub-questions while producing outputs. For example, self-ask (Press et al., 2022), a method in this category, manually inserts sub-questions in exemplars using Prompt D.2. For the test case, retrieval is triggered dynamically whenever the model generates a sub-question.

The aforementioned approaches can retrieve additional information while generating. However, they have notable drawbacks: (1) Using previously generated tokens as queries might not reflect what LMs intend to generate in the future. (2) Retrieving information at a fixed interval can be inefficient because it might occur at inappropriate points. (3) Question decomposition approaches require taskspecific prompt engineering, which restricts their generalizability in new tasks.

