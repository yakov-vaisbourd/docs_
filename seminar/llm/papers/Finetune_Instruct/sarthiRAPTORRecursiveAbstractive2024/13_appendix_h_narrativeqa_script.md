# H NarrativeQA Evaluation Script

We made several modifications to AllenNLP's evaluation script ${ }^{3}$ to better fit our evaluation needs:
\begin{itemize}
\item[-] Added Smoothing: Smoothing was incorporated to handle cases where BLEU score is zero, due to no n-gram matches occurring in the reference text. A BLEU score of zero skews the results, leading to an overly harsh evaluation for rare or novel phrases. By adding
\end{itemize}

\footnotetext{
${ }^{3}$ docs.allennlp.org/models/main/models/rc/tools/narrativeqa/
}
\begin{itemize}
\item[a] smoothing function, we prevent the BLEU scores from dropping to zero, providing a more fair evaluation.
\item[-] Modified BLEU-4 Weighting: The original script applied a weight of 1 to the highest order n-gram (4-gram) and 0 to the rest in its BLEU-4 calculation (i.e., weights=(0, 0, 0, 1)). This approach may overly focus on 4-gram matches while neglecting lower-order matches. To provide a more balanced evaluation, we evenly distributed the weight across all n-gram levels, changing the weights for the BLEU-4 calculation to (0.25, 0.25, 0.25, 0.25).
\item[-] Tokenization before Mapping in METEOR Calculation: The original script utilized a simple split and map method for METEOR calculation. We fixed this by first tokenizing the text and then mapping the tokens. This amendment improves the accuracy of the METEOR calculation by taking into account the correct linguistic boundaries of words.
\end{itemize}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline \multicolumn{2}{|c|}{Question: What is the central theme of the story?} \\
\hline RAPTOR & Fairy Godmother helps Cinderella attend a ball by transforming her rags... Cinderella impresses the Prince at the ball... she loses track of time and has to run home alone in the darkness. The Prince is unable to find Cinderella and goes in search of her ... She forgave her sisters, and treated them always very kindly, and the Prince had great cause to be glad that he had found the glass slipper. \\
\hline DPR & Two mice were turned into footmen; four grasshoppers into white horses. Next, the Fairy touched Cinderella's rags, and they became rich satin robes, trimmed with point lace. ... Alas! the clock had struck Eleven. She found no coach, only a pumpkin, and the rat and mice ran quickly away when they saw her; while all her fine dress turned to rags, and she had to run home alone... They told her a beautiful Princess had been at the ball, with whom the Prince was delighted. They did not know it was Cinderella. \\
\hline \multicolumn{2}{|c|}{Question: How does Cinderella find a happy ending?} \\
\hline RAPTOR & The story of Cinderella involves her godmother, a fairy, who transforms a pumpkin into a grand coach with her wand and allows Cinderella to attend the ball. However, Cinderella must return home before the clock strikes eleven or her dress will turn back into rags... Cinderella impresses the Prince at the ball but leaves before he can find out who she is… The Prince searched for the owner of a lost glass slipper and found it belonged to Cinderella. She forgave her sisters and the Prince was glad to have found her. \\
\hline DPR & the clock had struck Eleven... The Prince was very much surprised when he missed Cinderella again, and leaving the ball, went in search of her. . . Fairy touched Cinderella's rags, and they became rich satin robes, trimmed with point lace... Her old shoes became a charming pair of glass slippers, which shone like diamonds. "Now go to the ball, my love," she said, "and enjoy yourself. But remember, you must leave the room before the clock strikes eleven. If you do not your dress will return to its original rags." \\
\hline
\end{tabular}

**Table 13: Relevant excerpts from text retrieved by RAPTOR and DPR for the questions on the fairytale Cinderella.**
\end{table}

