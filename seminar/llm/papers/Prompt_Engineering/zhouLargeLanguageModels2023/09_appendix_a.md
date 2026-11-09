# A Prompt Engineering in the Wild

Large models with natural language interfaces, including models for text generation and image synthesis, have seen an increasing amount of public usage in recent years. As finding the right prompt can be difficult for humans, a number of guides on prompt engineering as well as tools to aid in prompt discovery have been developed. Among others, see, for example:
\begin{itemize}
\item[-] https://blog.andrewcantino.com/blog/2021/04/21/prompt-engineering-tips-and-tricks/
\item[-] https://techcrunch.com/2022/07/29/a-startup-is-charging-1-99-for-strings-of-text-to-feed-to-dall-e-2/
\item[-] https://news.ycombinator.com/item?id=32943224
\item[-] https://promptomania.com/stable-diffusion-prompt-builder/
\item[-] https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion
\end{itemize}

In this paper we apply APE to generate effective instructions for steering LLMs, but the general framework Algorithm 1 could be applied to steer other models with natural language interfaces so long as an appropriate proposal method and scoring function can be designed.

