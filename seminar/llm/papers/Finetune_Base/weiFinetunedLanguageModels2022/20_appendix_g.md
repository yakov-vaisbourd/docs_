# G Tasks and Datasets

This appendix further details the datasets that we use in this paper. We group datasets into one of the following task clusters:
\begin{itemize}
\item[-] Natural language inference concerns how two sentences relate, typically asking, given a first sentence, whether a second sentence is true, false, or possibly true. We use the following datasets:
\begin{itemize}
\item[1.] ANLI (Nie et al., 2020)
\item[2.] CB (De Marneffe et al., 2019)
\item[3.] MNLI (Williams et al., 2018)
\item[4.] QNLI (Rajpurkar et al., 2018)
\item[5.] SNLI (Bowman et al., 2015)
\item[6.] WNLI (Levesque et al., 2012)
\item[7.] RTE (Dagan et al., 2005; Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009)
\end{itemize}
\item[-] Reading comprehension tests the ability to answer a question when given a passage that contains the answer. We use the following datasets:
\begin{itemize}
\item[1.] BoolQ Clark et al. (2019a)
\item[2.] DROP (Dua et al., 2019)
\item[3.] MultiRC (Khashabi et al., 2018)
\item[4.] OBQA (Mihaylov et al., 2018)
\item[5.] SQuADv1 (Rajpurkar et al., 2016)
\item[6.] SQuADv2 (Rajpurkar et al., 2018)
\end{itemize}
\item[-] Commonsense reasoning evaluates the ability to perform physical or scientific reasoning with an element of common sense. We use the following datasets:
\begin{itemize}
\item[1.] COPA (Roemmele et al., 2011)
\item[2.] HellaSwag (Zellers et al., 2019)
\item[3.] PiQA (Bisk et al., 2020)
\item[4.] StoryCloze (Mostafazadeh et al., 2016)
\end{itemize}
\item[-] Sentiment analysis is a classic NLP task aims to understand whether a piece of text is positive or negative. We use the following datasets:
\begin{itemize}
\item[1.] IMDB (Maas et al., 2011)
\item[2.] Sentiment140 (Go et al., 2009)
\item[3.] SST-2 (Socher et al., 2013)
\item[4.] Yelp (Fast.AI)
\end{itemize}
\item[-] Closed-book QA asks models to answer questions about the world without specific access to information that contains the answer. We use the following datasets:
\begin{itemize}
\item[1.] ARC (Clark et al., 2018)
\item[2.] NQ (Lee et al., 2019; Kwiatkowski et al., 2019)
\item[3.] TriviaQA Joshi et al. (2017)
\end{itemize}
\item[-] Paraphrase detection asks a model to determine whether two sentences are semantically equivalent. ${ }^{4}$ We use the following datasets:
\begin{itemize}
\item[1.] MRPC (Dolan \& Brockett 2005)
\item[2.] QQP (Wang et al., 2018, see)
\item[3.] Paws Wiki (Zhang et al., 2019)
\end{itemize}
\item[-] Coreference resolution tests the ability to identify expressions of the same entity in some given text. We use the following datasets:
\begin{itemize}
\item[1.] DPR (Rahman \& Ng, 2012)
\item[2.] Winogrande (Sakaguchi et al., 2020)
\end{itemize}
\end{itemize}

\footnotetext{
${ }^{4}$ Although paraphrasing can be seen as positive entailment in both directions, it has been distinct from NLI in the academic literature.
}
\begin{itemize}
\begin{itemize}
\item[3.] WSC273 (Levesque et al., 2012)
\end{itemize}
\item[-] Reading comprehension with commonsense combines elements of both reading comprehension with commonsense. We use the following datasets:
\begin{itemize}
\item[1.] CosmosQA (Huang et al., 2019)
\item[2.] ReCoRD (Zhang et al., 2018)
\end{itemize}
\item[-] Struct to text tests the ability to describe some structured data using natural language. We use the following datasets:
\begin{itemize}
\item[1.] CommonGen (Lin et al., 2020)
\item[2.] DART (Nan et al., 2021)
\item[3.] E2ENLG (Dušek et al., 2019)
\item[4.] WebNLG (Gardent et al., 2017)
\end{itemize}
\item[-] Translation is the task of translating text from one language into a different language. We use the following datasets:
\begin{itemize}
\item[1.] En-Fr from WMT'14 (Bojar et al., 2014)
\item[2.] En-De, En-Tr, En-Cs, En-Fi, En-Ro, and En-Ru from WMT'16 (Bojar et al., 2016)
\item[3.] En-Es from Paracrawl (Bañón et al., 2020)
\end{itemize}
\item[-] Summarization asks models to read a piece of text and generate an abbreviated summary of it. We use the following datasets:
\begin{itemize}
\item[1.] AESLC (Zhang \& Tetreault, 2019)
\item[2.] CNN-DM (See et al., 2017)
\item[3.] Gigaword (Napoles et al., 2012)
\item[4.] MultiNews (Fabbri et al., 2019)
\item[5.] Newsroom (Grusky et al., 2018)
\item[6.] Samsum (Gliwa et al., 2019)
\item[7.] XSum (Narayan et al., 2018)
\item[8.] AG News (Zhang et al., 2015)
\item[9.] Opinion Abstracts - Rotten Tomatoes (Wang \& Ling, 2016)
\item[10.] Opinion Abstracts - iDebate (Wang \& Ling, 2016)
\item[11.] Wiki Lingua English (Ladhak et al., 2020)
\end{itemize}
\item[-] Additional datasets that we assign to a miscellaneous task cluster include:
\begin{itemize}
\item[1.] Conversational question-answering: QuAC (Choi et al., 2018) and CoQA (Reddy et al., 2019)
\item[2.] Evaluating context-sentence word meanings: WiC (Pilehvar \& Camacho-Collados, 2019)
\item[3.] Question classification: TREC (Li \& Roth, 2002; Hovy et al., 2001)
\item[4.] Linguistic acceptability: CoLA (Warstadt et al., 2019)
\item[5.] Math questions (Saxton et al., 2019)
\end{itemize}
\end{itemize}

For all tasks, our finetuning and evaluation code uses tensorflow datasets (TFDS) to load and process datasets. Regarding the number of training examples per dataset, we limited the training set size per dataset to 30,000 so that no dataset dominated the finetuning distribution. When a test set with labels was available in TFDS, we used it; otherwise, we used the TFDS validation set as our test set, splitting the training set into a train and dev set.

On the following pages, we show inputs and outputs for evaluation tasks where we compared with GPT-3. See the attached supplementary material for the templates for all other datasets.

# G. 1 Natural Language Inference

Input
Joey Heindle (born 14 May 1993 in Munich) is a German singer. He is best known for winning the seventh season of the game show Ich bin ein Star - Holt mich hier raus! and finishing in 5th place in season 9 of Deutschland sucht den Superstar, despite universally negative reviews from the jury each week.

Based on the paragraph above can we conclude that "Joey Heindle was highly disliked by people on television."?

OPTIONS:
\begin{itemize}
\item[-] Yes
\item[-] It's impossible to say
\item[-] No
\end{itemize}

Target
\begin{itemize}
\item[] Yes
\end{itemize}

Table 6: Example input and target for Adversarial NLI (ANLI). ANLI (Nie et al., 2020) is a large-scale NLI benchmark with adversarial examples collected iteratively with a human and model in the loop. The task is to determine whether a hypothesis is entailed by a premise (entailment, not entailment, or impossible to say). There are three rounds, R1-R3. Of the three training sets with 16,946, 45,460, and 100,459 examples, we use 16,946, 30,000, and 30,000 for train and 200 from each of the three TFDS validation sets for dev. We use the TFDS "test" sets of 1,000, 1,000, and 1,200 examples as our test set for reporting numbers.

Input
A: so I watch the fish, you know. Whatever I can do to keep myself occupied. I like to have the TV on, because that usually keeps me, um, more occupied. It kind of takes the time away and I don't realize, that's really the only time I ever watch TV, is when I'm on the bike. and then usually after I'm done riding the bike, just to cool myself down, I usually take a walk, you know, and that just kind of uh, gets me, you know, to where I'm not quite as tired I guess. But it's definitely a task. B: You think so? A: I can't say that I really enjoy it.

Based on the paragraph above can we conclude that "she really enjoys it"?
OPTIONS:
\begin{itemize}
\item[-] Yes
\item[-] No
\item[-] It's impossible to say
\end{itemize}

Target
\begin{itemize}
\item[] No
\end{itemize}

Table 7: Example input and target for Commitment Bank (CB). CB (De Marneffe et al., 2019) is a corpus of texts in which a hypothesis is extracted from a premise, and the task is to determine whether the hypothesis is entailed by the premise (entailment, not entailment, or impossible to say). Of the training set with 250 examples, we use 200 for train and 50 for dev. We use the TFDS validation set of 56 examples as our test set for reporting numbers.

Input
After years of study, the Vatican's doctrinal congregation has sent church leaders a confidential document concluding that "sex-change" procedures do not change a person's gender in the eyes of the church.

Based on the paragraph above can we conclude that "Sex-change operations become more common."?

OPTIONS:
\begin{itemize}
\item[-] yes
\item[-] no
\end{itemize}

Target
\begin{itemize}
\item[] no
\end{itemize}

Table 8: Example input and target for Recognizing Textual Entailment (RTE). RTE (Dagan et al. 2005; Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009) asks whether a second sentence is entailed by a first (binary, either entailed or not entailed). Of the training set with 2490 examples, we use 2,290 for train and 200 for dev. We use the TFDS validation set of 277 examples as our test set for reporting numbers.

# G. 2 Reading Comprehension

Input
There are four ways an individual can acquire Canadian citizenship: by birth on Canadian soil; by descent (being born to a Canadian parent); by grant (naturalization); and by adoption. Among them, only citizenship by birth is granted automatically with limited exceptions, while citizenship by descent or adoption is acquired automatically if the specified conditions have been met. Citizenship by grant, on the other hand, must be approved by the Minister of Immigration, Refugees and Citizenship.

Can we conclude that can i get canadian citizenship if my grandfather was canadian?
OPTIONS:
\begin{itemize}
\item[-] no
\item[-] yes
\end{itemize}

Target
\begin{itemize}
\item[] no
\end{itemize}

Table 9: Example input and target for Boolean Questions (BoolQ). BoolQ Clark et al. (2019a) asks a yes/no question based on a passage and a question. Of the training set with 9,427 examples, we use 9,227 for train and 200 for dev. We use the TFDS validation set of 3,270 examples as our test set for reporting numbers.

Input
Imagine you are standing in a farm field in central Illinois. The land is so flat you can see for miles and miles. On a clear day, you might see a grain silo 20 miles away. You might think to yourself, it sure is flat around here. If you drive one hundred miles to the south, the landscape changes. In southern Illinois, there are rolling hills. Why do you think this is? What could have caused these features? There are no big rivers that may have eroded and deposited this material. The ground is capable of supporting grass and trees, so wind erosion would not explain it. To answer the question, you need to go back 12,000 years. Around 12,000 years ago, a giant ice sheet covered much of the Midwest United States. Springfield, Illinois, was covered by over a mile of ice. Its hard to imagine a mile thick sheet of ice. The massive ice sheet, called a glacier, caused the features on the land you see today. Where did glaciers go? Where can you see them today? Glaciers are masses of flowing ice.

Question: "How big were the glaciers?"
Response: "One mile"
Does the response correctly answer the question?
OPTIONS:
\begin{itemize}
\item[-] no
\item[-] yes
\end{itemize}

Target
\begin{itemize}
\item[] yes
\end{itemize}

Table 10: Example input and target for Multi-Sentence Reading Comprehension (MultiRC). MultiRC Khashabi et al. (2018) asks an open-ended question given a paragraph that contains the answer. Of the training set with 27,243 examples, we use 27,043 for train and 200 for dev. We use the TFDS validation set of 4,848 examples as our test set for reporting numbers.

Input
soil is a renewable resource for growing plants
A plant that needs to expand will be able to have an endless resource in
OPTIONS:
\begin{itemize}
\item[-] dirt
\item[-] pesticides
\item[-] pay
\item[-] beans
\end{itemize}

Target
dirt

Table 11: Example input and target for Openbook Question Answering (OBQA). OBQA (Mihaylov et al., 2018) asks 4-way multiple choice questions based facts. Of the training set with 4,957 examples, we use all for train and 200 in the TFDS validation set of 500 examples for dev. We use the TFDS test set of 500 examples as our test set for reporting numbers.

# G. 3 Commonsense Reasoning

Input
I packed up my belongings. What is the cause?
OPTIONS:
\begin{itemize}
\item[-] I was hunting for a new apartment.
\item[-] I was moving out of my apartment.
\end{itemize}

Target
I was moving out of my apartment.

Table 12: Example input and target for Choice of Plausible Alternatives (COPA). COPA (Roemmele et al., 2011) is a causal reasoning task that asks to infer either a cause of effect of a premise from two choices. Of the training set with 400 examples, we use 350 for train and 50 for dev. We use the TFDS validation set of 100 examples as our test set for reporting numbers.

Input
What happens next in this paragraph?
Once the rope is inside the hook, he begins moving up the wall but shortly after he stops and begins talking. The male then begins talking about the clip again and goes back up the wall. as he
OPTIONS:
\begin{itemize}
\item[-] progresses, there are hooks everywhere on the wall and when he gets near them, he puts his rope inside of it for support and safety.
\item[-] changes time, an instant replay of his initial move is shown a second time.
\item[-] continues to talk, another male speaks about the move and shows another closeup of the plex by the male.
\item[-] continues, other people start to arrive and begin to hang out with him as he makes a few parts of the rope.
\end{itemize}

Target
progresses, there are hooks everywhere on the wall and when he gets near them, he puts his rope inside of it for support and safety.

Table 13: Example input and target for Commonsense Sentence Completion (HellaSwag). HellaSwag (Zellers et al., 2019) tests for sentence completion that requires common sense, asking for the most probable ending given four contexts. Of the training set with 39,905 examples, we use 30,000 for train and 200 for dev. We use the TFDS validation set of 10,042 examples as our test set for reporting numbers.

Input
Here is a goal: Remove smell from garbage disposal.
How would you accomplish this goal?
OPTIONS:
\begin{itemize}
\item[-] Create soda ice cubes and grind through disposal.
\item[-] Create vinegar ice cubes and grind through disposal.
\end{itemize}

Target
Create vinegar ice cubes and grind through disposal.

Table 14: Example input and target for Physical Question Answering (PiQA). PiQA (Bisk et al. 2020) is a commonsense QA benchmark for naive physics reasoning, where a solution to a goal must be selected from two choices. Of the training set with 16,113 examples, we use 16,013 for train and 100 for dev. We use the TFDS validation set of 1,838 examples as our test set for reporting numbers.

Input
Caroline never drinks carbonated beverages. Her friends pick on her because of it. One day they challenged her to drink a soda. Caroline wanted to win the challenge.

Predict the next sentence.
OPTIONS:
\begin{itemize}
\item[-] Caroline refused to open the soda.
\item[-] Caroline opened the soda and drank it all in one gulp!
\end{itemize}

Target
\begin{itemize}
\item[] Caroline opened the soda and drank it all in one gulp!
\end{itemize}

Table 15: Example input and target for The Story Cloze Test (StoryCloze). StoryCloze (Mostafazadeh et al., 2016) is a commonsense reasoning framework for story generation, where a system chooses the correct ending to a four-sentence story. We use the 2016 version on TFDS. Of the validation set with 1,871 examples (no training set is available), we use 1,671 for train and 200 for dev. We use the TFDS test set of 1,871 examples as our test set for reporting numbers.

# G. 4 Closed-Book QA

Input
What season is the Northern Hemisphere experiencing when it is tilted directly toward the Sun?
OPTIONS:
\begin{itemize}
\item[-] fall
\item[-] winter
\item[-] spring
\item[-] summer
\end{itemize}

Target
\begin{itemize}
\item[] summer
\end{itemize}

\begin{table}

**Table 16: Example input and target for The AI2 Reasoning Challenge (ARC). ARC (Clark et al. 2018) asks grade-school level 4-way multiple choice science questions. There is a challenge set and an easy set, where the challenge set questions were answered incorrectly by both a retrieval-based algorithm and a co-occurrence algorithm. Of the training sets with 1,119 examples (challenge) and 2,251 (easy), we use we use 919 and 2,051 respectively for train and 200 each for dev. We use the TFDS test sets of 1,172 and 2,376 examples respectively as our test set for reporting numbers.**
\begin{tabular}[t]{|l|}
\hline Input \\
\hline Question: who is the girl in more than you know?? \\
\hline Answer: \\
\hline Target \\
\hline Romi Van Renterghem. \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 17: Example input and target for Natural Questions (Open) (NQ). NQ (Lee et al., 2019; Kwiatkowski et al., 2019) asks for an open-ended answer given a question, where all questions can be answered using the contents of Wikipedia. Of the training set of 87,925 examples, we use 30,000 for train and 200 for dev. We use the TFDS validation set of 3,610 examples as our test set for reporting numbers.**
\begin{tabular}[t]{|l|}
\hline Input Please answer this question: Henry Croft, an orphan street sweeper who collected money for charity, is associated with what organised charitable tradition of working class culture in London, England? \\
\hline Target pearly kings and queens \\
\hline
\end{tabular}
\end{table}

Table 18: Example input and target for Trivia Question Answering (TriviaQA). TriviaQA Joshi et al. (2017) includes question-answer pairs authored by trivia enthusiasts. Of the training set of 87,622 examples, we use 30,000 for train and 200 for dev. We use 7,993 examples from Wikipedia of the 11,313 examples in the TFDS validation set, which is the same validation set used in (Brown et al. 2020). as our test set for reporting numbers.

# G. 5 Coreference Resolution

Input
How does the sentence end?
Elena wanted to move out of her parents fast but Victoria wanted to stay for a while,
OPTIONS:
\begin{itemize}
\item[-] Elena went to school.
\item[-] Victoria went to school.
\end{itemize}

Target
Victoria went to school.

Table 19: Example input and target for Adversarial Winograd Schema Challenge (Winogrande). Winogrande (Sakaguchi et al., 2020) tests for coreference resolution by asking a model to fill in a masked token in a sentence by choosing an entity from two options. Of the 40.4k examples in the XL training set, we use 30,000 for train and 200 for dev. We use the TFDS validation set of 1,267 as our test set for reporting numbers.

Input
Jane knocked on Susan's door, but there was no answer.
OPTIONS:
\begin{itemize}
\item[-] Jane was out.
\item[-] Susan was out.
\end{itemize}

Target
Susan was out.

Table 20: Example input and target for Winograd Schema Challenge (WSC273). WSC273 (Levesque et al., 2012) tests for coreference resolution by asking a model to complete the sentence in a fashion that requires understanding the entities in the sentence. Of the 0 examples in the training set (WSC273 is test-set only), we use none for train and none for dev. We use the TFDS test set as our test set for reporting numbers.

# G. 6 Reading Comprehension with Commonsense

Input
Complete the passage.
(CNN) - At first glance, "The Flat" might seem like an episode of "Hoarders," Israeli-style. The documentary film opens after an elderly woman dies in Tel Aviv. Her grandchildren assemble to clean out her apartment, packed with dusty books, vintage clothing (dozens of pairs of fancy gloves, for instance), enough purses to stock a department store, jewelry, mementoes and closets full of knickknacks. But buried among the detritus they chance upon something remarkable -mysterious papers linking the grandparents to an important Nazi figure. How could such ardent Zionists, who left their native Germany in the early 1930s, have been involved with an SS official like Leopold von Mildenstein?

What I found out was this journey, the Nazi (
OPTIONS:
\begin{itemize}
\item[-] Arnon Goldfinger) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] CNN) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] Germany) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] Israeli) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] Leopold von Mildenstein) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] Nazi) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] SS) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] Tel Aviv) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] The Flat) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\item[-] Zionists) and his wife were accompanied by my grandparents," Goldfinger told CNN.
\end{itemize}

Target
Leopold von Mildenstein) and his wife were accompanied by my grandparents," Goldfinger told CNN.

Table 21: Example input and target for Reading Comprehension with Commonsense Reasoning (ReCoRD). ReCoRD (Zhang et al., 2018) asks for the answer to a cloze-style question where an entity is masked out. Of the the training set of 100,730 examples, we use 30,000 for train and 200 for dev. We use the TFDS validation set of 10,000 examples as our test set for reporting numbers.

# G. 7 Translation (7 languages)

Input
Here the largest town of the district is located: Nordenham , lying opposite to Bremerhaven at the Weser mouth.

Translate to German
Target
An der B 211 befindet sich in Loyermoor der so genannte "Geest-Abbruch", der eine Höhendifferenz von gut 30 Meter überbrückt.

Table 22: Example input and output for translation. This example is from WMT' 16 English-German; all languages use the same translation templates.
