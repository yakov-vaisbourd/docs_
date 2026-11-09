# B. Prompt Formats Used

Tables 5 and 6 show the default prompt format used for all tasks. Table 7 shows the 15 different formats used when studying the effect of prompt format for SST-2.

![Figure](./zhaoCalibrateUseImproving2021_images/bfa78706-72f5-4d26-85c6-dd246cd5745f-12_518_1710_397_180.jpg)

**Figure 9. Contextual calibration improves GPT-3's accuracy across various prompt formats for LAMA. We plot GPT-2 2.7B's mean accuracy over 15 different formats for the LAMA "place of death" relation (P20), "Headquarter Location" relation (P159), and "place of birth" relation (P19).**

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Dataset} & \multirow[t]{2}{*}{LM} & \multicolumn{2}{|c|}{0-shot} & \multicolumn{2}{|c|}{1-shot} & \multicolumn{2}{|c|}{4-shot} & \multicolumn{2}{|c|}{8-shot} \\
\hline & & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours \\
\hline \multicolumn{10}{|l|}{Text Classification} \\
\hline AGNews & GPT-2 & $44.0_{0.0}$ & $\mathbf{6 0 . 0}_{0.0}$ & $45.4_{8.4}$ & $\mathbf{6 7 . 9}_{5.7}$ & $44.6_{12.2}$ & $\mathbf{5 8 . 0}_{13.6}$ & $57.1_{11.6}$ & $\mathbf{6 3 . 1}_{7.3}$ \\
\hline TREC & GPT-2 & $24.0_{0.0}$ & $\mathbf{3 7 . 3}_{0.0}$ & $21.5_{5.2}$ & $\mathbf{4 1 . 1}_{2.6}$ & $23.1_{5.9}$ & $\mathbf{4 4 . 2}_{2.2}$ & $32.7_{7.5}$ & $\mathbf{4 4 . 1}_{3.6}$ \\
\hline CB & GPT-2 & $\mathbf{4 4 . 6}_{0.0}$ & $17.9_{0.0}$ & $\mathbf{4 9 . 6}_{10.0}$ & $47.1_{12.2}$ & $40.0_{8.3}$ & $\mathbf{5 5 . 4}_{7.3}$ & $48.9_{5.7}$ & $\mathbf{6 3 . 2}_{1.4}$ \\
\hline RTE & GPT-2 & $\mathbf{5 1 . 0}_{0.0}$ & $48.5_{0.0}$ & $\mathbf{5 7 . 6}_{2.1}$ & $56.3_{2.4}$ & $53.2_{6.0}$ & $\mathbf{5 7 . 5}_{1.8}$ & $54.9_{3.0}$ & $\mathbf{5 7 . 7}_{1.29}$ \\
\hline SST-2 & GPT-2 & $60.0_{0.0}$ & $\mathbf{8 2 . 0}_{0.0}$ & $66.7_{17.9}$ & $\mathbf{7 3 . 0}_{11.4}$ & $64.9_{8.4}$ & $\mathbf{7 3 . 8}_{10.9}$ & $54.5_{4.6}$ & $\mathbf{6 4 . 6}_{8.8}$ \\
\hline DBPedia & GPT-2 & $\mathbf{6 4 . 3}_{0.0}$ & $58.3_{0.0}$ & $33.6_{18.9}$ & $\mathbf{6 9 . 5}_{9.4}$ & $53.0_{14.8}$ & $\mathbf{7 5 . 3}_{8.1}$ & $66.0_{3.6}$ & $\mathbf{7 4 . 3}_{8.7}$ \\
\hline \multicolumn{10}{|l|}{Fact Retrieval} \\
\hline LAMA & GPT-2 & $14.0_{0.0}$ & $\mathbf{2 2 . 7}_{0.0}$ & $29.7_{1.8}$ & $\mathbf{3 1 . 6}_{1.3}$ & $35.8_{3.8}$ & $\mathbf{3 7 . 4}_{3.4}$ & $\mathbf{4 2 . 5}_{1.3}$ & $\mathbf{4 2 . 5}_{1.4}$ \\
\hline \multicolumn{10}{|l|}{Information Extraction} \\
\hline MIT-G & GPT-2 & $7.7_{0.0}$ & $\mathbf{1 0 . 0}_{0.0}$ & $32.9_{10.0}$ & $\mathbf{4 1 . 2}_{4.1}$ & $44.3_{6.5}$ & $\mathbf{4 7 . 7}_{\text {5.8 }}$ & $56.9_{2.5}$ & $\mathbf{5 9 . 5}_{2.5}$ \\
\hline MIT-D & GPT-2 & $29.3_{0.0}$ & $\mathbf{4 1 . 7}_{0.0}$ & $26.2_{10.5}$ & $\mathbf{5 8 . 8}_{4.8}$ & $70.5_{2.5}$ & $\mathbf{7 5 . 4}_{1.8}$ & $77.1_{4.4}$ & $\mathbf{7 8 . 1}_{3.9}$ \\
\hline ATIS-A & GPT-2 & $15.1_{0.0}$ & $\mathbf{3 5 . 5}_{0.0}$ & $41.5_{11.7}$ & $\mathbf{5 1 . 4}_{7.5}$ & $55.1_{18.9}$ & $\mathbf{6 5 . 8}_{11.7}$ & $63.4_{10.6}$ & $\mathbf{6 9 . 9}_{10.4}$ \\
\hline ATIS-D & GPT-2 & $1.0_{0.0}$ & $\mathbf{2 . 5}_{0.0}$ & $62.3_{9.2}$ & $\mathbf{6 8 . 7}_{4.3}$ & $81.1_{3.6}$ & $\mathbf{8 3 . 2}_{7.2}$ & $81.8_{4.5}$ & $\mathbf{8 3 . 9}_{5.0}$ \\
\hline
\end{tabular}

**Table 4. Contextual calibration improves accuracy for GPT-2. This table is analogous to Table 1 but shows results for GPT-2 XL.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Task & Prompt & Label Names \\
\hline SST-2 & \begin{tabular}[t]{l}
Review: This movie is amazing! Sentiment: Positive \\
Review: Horrific movie, don't see it. Sentiment:
\end{tabular} & Positive, Negative \\
\hline AGNews & \begin{tabular}[t]{l}
Article: USATODAY.com - Retail sales bounced back a bit in July, and new claims for jobless benefits fell last week, the government said Thursday, indicating the economy is improving from a midsummer slump. Answer: Business \\
Article: New hard-drive based devices feature color screens, support for WMP 10. Answer:
\end{tabular} & World, Sports, Business, Technology \\
\hline TREC & \begin{tabular}[t]{l}
Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation. \\
Question: How did serfdom develop in and then leave Russia? Answer Type: Description \\
Question: When was Ozzy Osbourne born? Answer Type:
\end{tabular} & Number, Location, Person, Description, Entity, Abbreviation \\
\hline DBPedia & \begin{tabular}[t]{l}
Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book. \\
Article: Geoffrey D. Falksen (born July 31 1982) is an American steampunk writer. Answer: Artist \\
Article: The Perrin River is a 1.3-mile-long (2.1 km) tidal river in the U.S. state of Virginia. It is a small inlet on the north shore of the York River near that river's mouth at Chesapeake Bay. Answer:
\end{tabular} & Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, Book \\
\hline CB & \begin{tabular}[t]{l}
But he ended up eating it himself. I was reluctant to kiss my mother, afraid that somehow her weakness and unhappiness would infect me. Naturally I didn't think for a minute that my life and spirit could stimulate her. question: her life and spirit could stimulate her mother. True, False, or Neither? answer: Neither \\
Valence the void-brain, Valence the virtuous valet. Why couldn't the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping? question: Valence was helping. True, False, or Neither? answer:
\end{tabular} & True, False, Neither \\
\hline RTE & \begin{tabular}[t]{l}
Others argue that Mr. Sharon should have negotiated the Gaza pullout - both to obtain at least some written promises of better Palestinian behavior, and to provide Mr. Abbas with a prime prize to show his people that diplomacy, not violence, delivered Gaza. question: Mr. Abbas is a member of the Palestinian family. True or False? answer: False \\
The program will include Falla's "Night in the Gardens of Spain," Ravel's Piano Concerto in G, Berlioz's Overture to "Beatrice and Benedict," and Roy Harris' Symphony No. 3. question: Beatrice and Benedict is an overture by Berlioz. True or False? answer:
\end{tabular} & True, False \\
\hline
\end{tabular}

**Table 5. The prompts used for text classification. We show one training example per task for illustration purposes. The right column shows the label names (to make predictions, we check the LM's probability for these tokens).**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Task & Prompt \\
\hline LAMA & \begin{tabular}[t]{l}
Alexander Berntsson was born in Sweden \\
Khalid Karami was born in
\end{tabular} \\
\hline ATIS (Airline) & \begin{tabular}[t]{l}
Sentence: what are the two american airlines flights that leave from dallas to san francisco in the evening Airline name: american airlines \\
Sentence: list a flight on american airlines from toronto to san diego Airline name:
\end{tabular} \\
\hline ATIS (Depart Date) & \begin{tabular}[t]{l}
Sentence: please list any flight available leaving oakland california tuesday arriving philadelphia wednesday Depart date - Day name: tuesday \\
Sentence: show me all all flights from pittsburgh to atlanta on wednesday which leave before noon and serve breakfast Depart date - Day name:
\end{tabular} \\
\hline MIT Movies (Genre) & \begin{tabular}[t]{l}
Sentence: last to a famous series of animated movies about a big green ogre and his donkey and cat friends Genre: animated \\
Sentence: what is a great comedy featuring the talents of steve carell as a loser looking for a friend Genre:
\end{tabular} \\
\hline MIT Movies (Director) & \begin{tabular}[t]{l}
Sentence: in 2005 director christopher nolan rebooted a legendary dc comics superhero with a darker grittier edge in which movie Director: christopher nolan \\
Sentence: what 1967 mike nichols film features dustin hoffman in romantic interludes with anne bancroft as mrs robinson Director:
\end{tabular} \\
\hline
\end{tabular}

**Table 6. The prompts used for generation tasks. We show one training example per task for illustration purposes.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Format ID & Prompt & Label Names \\
\hline 1 & \begin{tabular}[t]{l}
Review: This movie is amazing! Answer: Positive \\
Review: Horrific movie, don't see it. Answer:
\end{tabular} & Positive, Negative \\
\hline 2 & \begin{tabular}[t]{l}
Review: This movie is amazing! Answer: good \\
Review: Horrific movie, don't see it. Answer:
\end{tabular} & good, bad \\
\hline 3 & My review for last night's film: This movie is amazing! The critics agreed that this movie was good My review for last night's film: Horrific movie, don't see it. The critics agreed that this movie was & good, bad \\
\hline 4 & \begin{tabular}[t]{l}
Here is what our critics think for this month's films. \\
One of our critics wrote "This movie is amazing!". Her sentiment towards the film was positive. One of our critics wrote "Horrific movie, don't see it". Her sentiment towards the film was
\end{tabular} & positive, negative \\
\hline 5 & \begin{tabular}[t]{l}
Critical reception [ edit ] \\
In a contemporary review, Roger Ebert wrote "This movie is amazing!". Entertainment Weekly agreed, and the overall critical reception of the film was good. \\
In a contemporary review, Roger Ebert wrote "Horrific movie, don't see it". Entertainment Weekly agreed, and the overall critical reception of the film was
\end{tabular} & good, bad \\
\hline 6 & \begin{tabular}[t]{l}
Review: This movie is amazing! Positive Review? Yes \\
Review: Horrific movie, don't see it. Positive Review?
\end{tabular} & Yes, No \\
\hline 7 & \begin{tabular}[t]{l}
Review: This movie is amazing! Question: Is the sentiment of the above review Positive or Negative? Answer: Positive \\
Review: This movie is amazing! Question: Is the sentiment of the above review Positive or Negative? Answer:
\end{tabular} & Positive, Negative \\
\hline 8 & \begin{tabular}[t]{l}
Review: This movie is amazing! Question: Did the author think that the movie was good or bad? Answer: good \\
Review: This movie is amazing! Question: Did the author think that the movie was good or bad? Answer:
\end{tabular} & good, bad \\
\hline 9 & \begin{tabular}[t]{l}
Question: Did the author of the following tweet think that the movie was good or bad? Tweet: This movie is amazing! Answer: good \\
Question: Did the author of the following tweet think that the movie was good or bad? Tweet: Horrific movie, don't see it Answer:
\end{tabular} & good, bad \\
\hline 10 & This movie is amazing! My overall feeling was that the movie was good Horrific movie, don't see it. My overall feeling was that the movie was & good, bad \\
\hline 11 & This movie is amazing! I liked the movie. Horrific movie, don't see it. I & liked, hated \\
\hline 12 & This movie is amazing! My friend asked me if I would give the movie 0 or 5 stars, I said 5 Horrific movie, don't see it. My friend asked me if I would give the movie 0 or 5 stars, I said & 0, 5 \\
\hline 13 & \begin{tabular}[t]{l}
Input: This movie is amazing! Sentiment: Positive \\
Input: Horrific movie, don't see it. Sentiment:
\end{tabular} & Positive, Negative \\
\hline 14 & \begin{tabular}[t]{l}
Review: This movie is amazing! Positive: True \\
Review: Horrific movie, don't see it. Positive:
\end{tabular} & True, False \\
\hline 15 & \begin{tabular}[t]{l}
Review: This movie is amazing! Stars: 5 \\
Review: Horrific movie, don't see it. Stars:
\end{tabular} & 5, 0 \\
\hline
\end{tabular}

**Table 7. The different prompt formats used when studying the effect of format for SST-2. We show one training example for illustration.**
\end{table}
