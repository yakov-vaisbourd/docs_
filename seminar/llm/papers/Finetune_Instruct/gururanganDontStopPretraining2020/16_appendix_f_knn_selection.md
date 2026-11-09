# F $k$-Nearest Neighbors Data Selection

In Table 21, we display nearest neighbor documents in the BioMed domain identified by our selection method, on the RCT dataset.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline \multirow{2}{*}{} & \multicolumn{5}{|c|}{Data Sample Unseen During Dapt} \\
\hline & PT & BioMed & CS & News & Reviews \\
\hline \multirow{5}{*}{\begin{tabular}[t]{l}
RoBERTA \\
Reviews
\end{tabular}} & 1.19 & 1.32 & 1.63 & 1.08 & 2.10 \\
\hline & 1.63 & 0.99 & 1.63 & 1.69 & 2.59 \\
\hline & 1.82 & 1.43 & 1.34 & 1.92 & 2.78 \\
\hline & 1.33 & 1.50 & 1.82 & 1.16 & 2.16 \\
\hline & 2.07 & 2.23 & 2.44 & 2.27 & 1.93 \\
\hline
\end{tabular}

**Table 12: RoBERTA's (row 1) and domain-adapted RoBERTA's (rows 2-5) masked LM loss on randomly sampled held-out documents from each domain (lower implies a better fit). PT denotes a sample from sources similar to ROBERTA's pretraining corpus. The lowest masked LM for each domain sample is boldfaced.**
\end{table}

\begin{tabular}[t]{|l|l|}
\hline Computing Infrastructure & Google Cloud v3-8 TPU \\
\hline Model implementations & https://github.com/allenai/tpu_pretrain \\
\hline
\end{tabular}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Hyperparameter & Assignment \\
\hline number of steps & 100 epochs (TAPT) or 12.5 K steps (DAPT) \\
\hline batch size & 256 or 2058 \\
\hline maximum learning rate & 0.0001 or 0.0005 \\
\hline learning rate optimizer & Adam \\
\hline Adam epsilon & 1e-6 \\
\hline Adam beta weights & 0.9, 0.98 \\
\hline learning rate scheduler & None or warmup linear \\
\hline Weight decay & 0.01 \\
\hline Warmup proportion & 0.06 \\
\hline learning rate decay & linear \\
\hline
\end{tabular}

**Table 13: Hyperparameters for domain- and task- adaptive pretraining.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Computing Infrastructure & \multicolumn{2}{|c|}{Quadro RTX 8000 GPU} \\
\hline Model implementation & \multicolumn{2}{|l|}{https://github.com/allenai/dont-stop-pretraining} \\
\hline \multirow{9}{*}{} & Hyperparameter & Assignment \\
\hline & number of epochs & 3 or 10 \\
\hline & patience & 3 \\
\hline & batch size & 16 \\
\hline & learning rate & 2e-5 \\
\hline & dropout & 0.1 \\
\hline & feedforward layer & 1 \\
\hline & feedforward nonlinearity & tanh \\
\hline & classification layer & 1 \\
\hline
\end{tabular}

**Table 14: Hyperparameters for RoBERTA text classifier.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Domain} & \multirow[b]{2}{*}{Task} & \multirow[b]{2}{*}{RoBERTA} & \multicolumn{3}{|c|}{Additional Pretraining Phases} \\
\hline & & & DAPT & TAPT & DAPT + TAPT \\
\hline \multirow{2}{*}{BioMed} & ChemProt & $83.2_{1.4}$ & $\mathbf{8 4 . 1}_{0.5}$ & $83.0_{0.6}$ & $\mathbf{8 4 . 1}_{0.5}$ \\
\hline & ${ }^{\dagger}$ RCT & $88.1_{0.05}$ & $\mathbf{8 8 . 5}_{0.1}$ & $88.3_{0.1}$ & $\mathbf{8 8 . 5}_{0.1}$ \\
\hline \multirow{2}{*}{CS} & ACL-ARC & $71.3_{2.8}$ & $73.2_{1.5}$ & $73.2_{3.6}$ & $\mathbf{7 8 . 6}_{2.9}$ \\
\hline & Scierc & $83.8_{1.1}$ & $\mathbf{8 8 . 4}_{1.7}$ & $85.9_{0.8}$ & $88.0_{1.3}$ \\
\hline \multirow{2}{*}{News} & HyperPartisan & $84.0_{1.5}$ & $79.1_{3.5}$ & $\mathbf{8 2 . 7}_{3.3}$ & $80.8_{2.3}$ \\
\hline & ${ }^{\dagger}$ AGNews & $94.3_{0.1}$ & $94.3_{0.1}$ & $94.7_{0.1}$ & $\mathbf{9 4 . 9}_{0.1}$ \\
\hline \multirow{2}{*}{Reviews} & ${ }^{\dagger}$ Helpfulness & $65.5_{3.4}$ & $66.5_{1.4}$ & $69.2_{2.4}$ & $\mathbf{6 9 . 4}_{2.1}$ \\
\hline & ${ }^{\dagger}$ IMDB & $94.8_{0.1}$ & $95.3_{0.1}$ & $95.4_{0.1}$ & $\mathbf{9 5 . 7}_{0.2}$ \\
\hline
\end{tabular}

**Table 15: Results on different phases of adaptive pretraining compared to the baseline RoBERTA (col. 1). Our approaches are dapt (col. 2, §3), tapt (col. 3, §4), and a combination of both (col. 4). Reported results are development macro- $F_{1**$, except for ChemProt and RCT, for which we report micro- $F_{1}$, following Beltagy et al. (2019). We report averages across five random seeds, with standard deviations as subscripts. † indicates high-resource settings. Best task performance is boldfaced. State-of-the-art results we can compare to: ChemProt (84.6), RCT (92.9), ACL-ARC (71.0), Scierc (81.8), HyperPartisan (94.8), AGNews (95.5), IMDB (96.2); references in §A.2.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Dom. & Task & RoB. & DAPT & ¬DAPT \\
\hline \multirow{2}{*}{BM} & ChemProt & $83.2_{1.4}$ & $\mathbf{8 4 . 1}_{0.5}$ & $80.9_{0.5}$ \\
\hline & ${ }^{\dagger}$ RCT & $88.1_{0.0}$ & $\mathbf{8 8 . 5}_{0.1}$ & $87.9_{0.1}$ \\
\hline \multirow{2}{*}{CS} & ACL-ARC & $71.3_{2.8}$ & $\mathbf{7 3 . 2}_{1.5}$ & $68.1_{5.4}$ \\
\hline & SciERC & $83.8_{1.1}$ & $\mathbf{8 8 . 4}_{1.7}$ & $83.9_{0.9}$ \\
\hline \multirow{2}{*}{News} & HyP. & $\mathbf{8 4 . 0}_{1.5}$ & $79.1_{3.5}$ & $71.6_{4.6}$ \\
\hline & ${ }^{\dagger}$ AGNews & $\mathbf{9 4 . 3}_{0.1}$ & $\mathbf{9 4 . 3}_{0.1}$ & $94.0_{0.1}$ \\
\hline \multirow{2}{*}{Rev.} & ${ }^{\dagger}$ Helpful. & $65.5_{3.4}$ & $\mathbf{6 6 . 5}_{1.4}$ & $65.5_{3.0}$ \\
\hline & ${ }^{\dagger}$ IMDB & $94.8_{0.1}$ & $\mathbf{9 5 . 3}_{0.1}$ & $93.8_{0.2}$ \\
\hline
\end{tabular}

**Table 16: Development comparison of RoBERTA (RoBa.) and DAPT to adaptation to an irrelevant domain (¬ dapt). See §3.3 for our choice of irrelevant domains. Reported results follow the same format as Table 5.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline BioMed & RCT & ChemProt \\
\hline TAPT & $88.3_{0.1}$ & $83.0_{0.6}$ \\
\hline Transfer-TAPT & \(88.0_{0.1}(\downarrow 0.3)\) & \(81.1_{0.5}(\downarrow 1.9)\) \\
\hline News & HyperPartisan & AGNews \\
\hline TAPT & $82.7_{3.3}$ & 94.70.1 \\
\hline Transfer-TAPT & \(77.6_{3.6}(\downarrow 5.1)\) & \(94.4_{0.1}(\downarrow 0.4)\) \\
\hline
\end{tabular}

**Table 17: Development results for TAPT transferability.**
\end{table}

\begin{tabular}[t]{|l|l|l|}
\hline CS & ACL-ARC & Scierc \\
\hline TAPT & 73.23.6 & $85.9_{0.8}$ \\
\hline Transfer-TAPT & \(74.0_{4.5}(\uparrow 1.2)\) & \(85.5_{1.1}(\downarrow 0.4)\) \\
\hline Amazon reviews & Helpfulness & IMDB \\
\hline TAPT & $69.2_{2.4}$ & 95.40.1 \\
\hline Transfer-TAPT & $65.4_{2.7}$ (↓ 3.8) & \(94.9_{0.1}(\downarrow 0.5)\) \\
\hline
\end{tabular}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline \multirow[t]{2}{*}{Pretraining} & BioMed & News & Reviews \\
\hline & RCT-500 & HyperPartisan & ${ }^{\dagger}$ IMDB \\
\hline TAPT & $80.5_{1.3}$ & $82.7_{3.3}$ & $95.4_{0.1}$ \\
\hline DAPT + TAPT & $83.9_{0.3}$ & $80.8_{2.3}$ & $95.7_{0.2}$ \\
\hline Curated-TAPT & $84.4_{0.3}$ & $\mathbf{8 4 . 9}_{\mathbf{1 . 9}}$ & $95.8_{0.1}$ \\
\hline DAPT + Curated-TAPT & $\mathbf{8 4 . 5}_{\mathbf{0 . 3}}$ & $83.1_{3.7}$ & $\mathbf{9 6 . 0}_{\mathbf{0 . 1}}$ \\
\hline
\end{tabular}

**Table 18: Mean development set macro- $F_{1**$ (for HyperPartisan and IMDB) and micro- $F_{1}$ (for RCT-500), with Curated-tapt across five random seeds, with standard deviations as subscripts. † indicates high-resource settings.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline \multirow[t]{2}{*}{Pretraining} & \multicolumn{2}{|c|}{BioMed} & CS \\
\hline & ChemProt & RCT-500 & ACL-ARC \\
\hline ROBERTA & $83.2_{1.4}$ & $80.3_{0.5}$ & $71.3_{2.8}$ \\
\hline TAPT & $83.0_{0.6}$ & $80.5_{1.3}$ & $73.2_{3.6}$ \\
\hline RAND-TAPT & $83.3_{0.5}$ & $81.6_{0.6}$ & $78.7_{4.0}$ \\
\hline 50nn-tapt & $83.3_{0.8}$ & $81.7_{0.5}$ & $70.1_{3.5}$ \\
\hline 150NN-TAPT & $83.3_{0.9}$ & $81.9_{0.8}$ & $\mathbf{7 8 . 5}_{2.2}$ \\
\hline 500nn-tapt & $\mathbf{8 4 . 5}_{0.4}$ & $82.6_{0.4}$ & $77.4_{2.3}$ \\
\hline DAPT & $84.1_{0.5}$ & $\mathbf{8 3 . 5}_{0.8}$ & $73.2_{1.5}$ \\
\hline
\end{tabular}

**Table 19: Mean development set macro- $F_{1**$ (for HYP. and IMDB) and micro- $F_{1}$ (for RCT), across five random seeds, with standard deviations as subscripts, comparing rand-tapt (with 50 candidates) and $k$ nn-tapt selection. Neighbors of the task data are selected from the domain data.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline IMDB review & RealNews article \\
\hline Spooks is enjoyable trash, featuring some well directed sequences, ridiculous plots and dialogue, and some third rate acting. Many have described this is a UK version of 24, and one can see the similarities. The American version shares the weak silly plots, but the execution is so much slicker, sexier and I suspect, expensive. Some people describe weak comedy as gentle comedy. This is gentle spy story hour, the exact opposite of anything created by John Le Carre. Give me Smiley any day. & [...] Remember poor Helen Flynn from Spooks? In 2002, the headlong BBC spy caper was in such a hurry to establish the high-wire stakes of its morally compromised world that Lisa Faulkners keen-as-mustard MI5 rookie turned out to be a lot more expendable than her prominent billing suggested. [...] Functioning as both a shocking twist and rather callous statement that No-One Is Safe, it gave the slick drama an instant patina of edginess while generating a record-breaking number of complaints. [...] \\
\hline The Sopranos is perhaps the most mind-opening series you could possibly ever want to watch. It's smart, it's quirky, it's funny - and it carries the mafia genre so well that most people can't resist watching. The best aspect of this show is the overwhelming realism of the characters, set in the subterranean world of the New York crime families. For most of the time, you really don't know whether the wise guys will stab someone in the back, or buy them lunch. Further adding to the realistic approach of the characters in this show is the depth of their personalities - These are dangerous men, most of them murderers, but by God if you don't love them too. I've laughed at their wisecracks, been torn when they've made err in judgement, and felt scared at the sheer ruthlessness of a serious criminal. [...] & The drumbeat regarding the Breaking Bad finale has led to the inevitable speculation on whether the final chapter in this serialized gem will live up to the hype or disappoint (thank you, Dexter, for setting that bar pretty low), with debate, second-guessing and graduate-thesis-length analysis sure to follow. The Most Memorable TV Series Finales of All-Time [...] No ending in recent years has been more divisive than The Sopranos for some, a brilliant flash (literally, in a way) of genius; for others (including yours truly), a too-cute copout, cryptically leaving its characters in perpetual limbo. The precedent to that would be St. Elsewhere, which irked many with its provocative, surreal notion that the whole series was, in fact, conjured in the mind of an autistic child. [...] \\
\hline The Wicker Man, starring Nicolas Cage, is by no means a good movie, but I can't really say it's one I regret watching. I could go on and on about the negative aspects of the movie, like the terrible acting and the lengthy scenes where Cage is looking for the girl, has a hallucination, followed by another hallucination, followed by a dream sequence- with a hallucination, etc., but it's just not worth dwelling on when it comes to a movie like this. Instead, here's five reasons why you SHOULD watch The Wicker Man, even though it's bad: 5 . It's hard to deny that it has some genuinely creepy ideas to it, the only problem is in its cheesy, unintentionally funny execution. If nothing else, this is a movie that may inspire you to see the original 1973 film, or even read the short story on which it is based. 4. For a cheesy horror/thriller, it is really aesthetically pleasing. [...] NOTE: The Unrated version of the movie is the best to watch, and it's better to watch the Theatrical version just for its little added on epilogue, which features a cameo from James Franco. & [...] What did you ultimately feel about "The Wicker Man" movie when all was said and done? [...] Im a fan of the original and Im glad that I made the movie because they dont make movies like that anymore and probably the result of what "Wicker Man" did is the reason why they dont make movies like that anymore. Again, its kind of that 70s sensibility, but Im trying to do things that are outside the box. Sometimes that means itll work and other times it wont. Again though Im going to try and learn from anything that I do. I think that it was a great cast, and Neil La Bute is one of the easiest directors that Ive ever worked with. He really loves actors and he really gives you a relaxed feeling on the set, that you can achieve whatever it is that youre trying to put together, but at the end of the day the frustration that I had with The Wicker Man, which I think has been remedied on the DVD because I believe the DVD has the directors original cut, is that they cut the horror out of the horror film to try and get a PG-13 rating. I mean, I dont know how to stop something like that. So Im not happy with the way that the picture ended, but Im happy with the spirit with which it was made. [...] \\
\hline Dr. Seuss would sure be mad right now if he was alive. Cat in the Hat proves to show how movie productions can take a classic story and turn it into a mindless pile of goop. We have Mike Myers as the infamous Cat in the Hat, big mistake! Myers proves he can't act in this film. He acts like a prissy show girl with a thousand tricks up his sleeve. The kids in this movie are all right, somewhere in between the lines of dull and annoying. The story is just like the original with a couple of tweaks and like most movies based on other stories, never tweak with the original story! Bringing in the evil neighbor Quin was a bad idea. He is a stupid villain that would never get anywhere in life. [...] & The Cat in the Hat, [...] Based on the book by Dr. Seuss [...] From the moment his tall, red-and-white-striped hat appears at their door, Sally and her brother know that the Cat in the Hat is the most mischievous cat they will ever meet. Suddenly the rainy afternoon is transformed by the Cat and his antics. Will their house ever be the same? Can the kids clean up before mom comes home? With some tricks (and a fish) and Thing Two and Thing One, with the Cat in The Hat, the fun's never done!Dr. Seuss is known worldwide as the imaginative master of children's literature. His books include a wonderful blend of invented and actual words, and his rhymes have helped many children and adults learn and better their understanding of the English language. [...] \\
\hline
\end{tabular}

**Table 20: Additional examples that highlight the overlap between IMDB reviews and RealNews articles.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Source & During median follow-up of 905 days ( IQR 773-1050 ) , 49 people died and 987 unplanned admissions were recorded ( totalling 5530 days in hospital ) . \\
\hline Neighbor 0 & Of this group, 26\% died after discharge from hospital, and the median time to death was 11 days (interquartile range, 4.0-15.0 days) after discharge. \\
\hline Neighbor 1 & The median hospital stay was 17 days (range 8-26 days), and all the patients were discharged within 1 month. \\
\hline Neighbor 2 & The median hospital stay was 17 days (range 8-26 days). \\
\hline Neighbor 3 & The median time between discharge and death was 25 days (mean, 59.1 days) and no patient was alive after 193 days. \\
\hline Neighbor 4 & The length of hospital stay after colostomy formation ranged from 3 days to 14 days with a median duration of 6 days (+IQR of 4 to 8 days). \\
\hline Source & Randomized , controlled , parallel clinical trial . \\
\hline Neighbor 0 & Design: Unblinded, randomised clinical controlled trial. \\
\hline Neighbor 1 & These studies and others led to the phase III randomized trial RTOG 0617/NCCTG 0628/ CALGB 30609. \\
\hline Neighbor 2 & -Definitive randomized controlled clinical trial (RCT): \\
\hline Neighbor 3 & $\mathrm{RCT} \frac{1}{4}$ randomized controlled trial. \\
\hline Neighbor 4 & randomized controlled trial [ Fig. 3(A)]. \\
\hline Source & Forty primary molar teeth in 40 healthy children aged 5-9 years were treated by direct pulp capping. \\
\hline Neighbor 0 & In our study, we specifically determined the usefulness of the Er:YAG laser in caries removal and cavity preparation of primary and young permanent teeth in children ages 4 to 18 years. \\
\hline Neighbor 1 & Males watched more TV than females, although it was only in primary school-aged children and on weekdays. \\
\hline Neighbor 2 & Assent was obtained from children and adolescents aged 7-17 years. \\
\hline Neighbor 3 & Cardiopulmonary resuscitation was not applied to children aged ¡5 years (Table 2). \\
\hline Neighbor 4 & It measures HRQoL in children and adolescents aged 2 to 25 years. \\
\hline
\end{tabular}

**Table 21: 5 nearest neighbors of sentences from the RCT dataset (Source) in the BioMed domain (Neighbors 0-4).**
\end{table}
