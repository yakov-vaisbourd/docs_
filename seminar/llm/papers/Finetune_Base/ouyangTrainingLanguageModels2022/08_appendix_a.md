# A Additional prompt data details

# A. 1 Labeler-written prompts

We first give slightly more details on our prompt boostrapping process. As previously mentioned, for the majority of the project, we obtained prompts directly from external users of the instruct beta models in the OpenAI API. However, this strategy only works once you have a model that accepts instruction-like prompts. In order to train the very first such model, we asked contractors to write prompts themselves. We asked labelers to write three kinds of prompts:
\begin{itemize}
\item[-] Plain: We simply ask the labelers to come up with an arbitrary task, while ensuring diversity of tasks.
\item[-] Few-shot: We ask the labelers to come up with an instruction, and multiple query/response pairs for that instruction. For example, the instruction could be "Give the sentiment for a tweet," and the queries would be tweets and the responses either "Positive" or "Negative." We can then format these as few-shot prompts like those in Brown et al. (2020). With K query-response pairs, we create K training examples using the other K-1 in the context.
\item[-] User-based: We had a number of use-cases stated in applications to the OpenAI API. We asked labelers to come up with prompts corresponding to these use cases.
\end{itemize}

In order to preserve the anonymity of the application information, we had a separate labeler create vague high level tasks based on looking at a list of applications, modifying the task descriptions to eliminate any information that were specific to a given application. This data was used to train the first InstructGPT model via supervised learning, which was deployed in beta in the API in early 2021.

# A. 2 API user prompts

For API prompts, we use prompts submitted by users to the aforementioned earlier version of the InstructGPT model on the OpenAI API Playground. Throughout the paper, we only use data from the Playground, rather than customers using our model in production, as it was easier to get informed consent: every time a user switched to an InstructGPT model, an alert message would pop up stating that prompts submitted to these models could be used to train future versions of our models. We also communicated this in a message on the developer Slack channel upon launching the beta of the InstructGPT models. We filter out prompts from the training split containing personally identifiable information (PII).

To ensure a diversity of use cases, we heuristically deduplicate prompts by checking for prompts that share a long common prefix, and limited the number of prompts to roughly 200 per organization. In addition, we create train, validation, and test splits based on organization IDs, so that e.g. the validation set contains different use cases than the training set.

We conceptualized API requests as belonging to one of ten use cases: generation, open QA, closed QA, brainstorming, chat, rewriting, summarization, classification, extraction, or other. Below, we show fictional but realistic prompts from a variety of use cases:

# A.2.1 Illustrative user prompts from InstructGPT distribution

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline brainstorming & List five ideas for how to regain enthusiasm for my career \\
\hline brainstorming & What are some key points I should know when studying Ancient Greece? \\
\hline \multirow[t]{3}{*}{brainstorming} & What are 4 questions a user might have after reading the instruction manual for a trash compactor? \\
\hline & \{user manual\} \\
\hline & 1. \\
\hline
\end{tabular}

Continued on next page

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline brainstorming & What are 10 science fiction books I should read next? \\
\hline \multirow[t]{3}{*}{classification} & Take the following text and rate, on a scale from 1-10, how sarcastic the person is being (1 = not at all, 10 = extremely sarcastic). Also give an explanation \\
\hline & \{text\} \\
\hline & Rating: \\
\hline \multirow[t]{3}{*}{classification} & This is a list of tweets and the sentiment categories they fall into. \\
\hline & Tweet: \{tweet_content1\} Sentiment: \{sentiment1\} \\
\hline & Tweet: \{tweet_content2\} Sentiment: \{sentiment2\} \\
\hline \multirow[t]{2}{*}{classification} & \{java code\} \\
\hline & What language is the code above written in? \\
\hline classification & \begin{tabular}[t]{l}
You are a very serious professor, and you check papers to see if they contain missing citations. Given the text, say whether it is missing an important citation (YES/NO) and which sentence(s) require citing. \\
\{text of paper\}
\end{tabular} \\
\hline \multirow[t]{2}{*}{extract} & Extract all course titles from the table below: \\
\hline & | Title | Lecturer | Room | | Calculus 101 | Smith | Hall B | | Art History | Paz | Hall A | \\
\hline \multirow[t]{2}{*}{extract} & Extract all place names from the article below: \\
\hline & \{news article\} \\
\hline \multirow[t]{2}{*}{extract} & Given the following list of movie titles, write down any names of cities in the titles. \\
\hline & \{movie titles\} \\
\hline \multirow[t]{2}{*}{generation} & Write a creative ad for the following product to run on Facebook aimed at parents: \\
\hline & Product: \{product description\} \\
\hline generation & Write a short story where a brown bear to the beach, makes friends with a seal, and then return home. \\
\hline \multicolumn{2}{|r|}{Continued on next page} \\
\hline
\end{tabular}

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline \multirow[t]{2}{*}{generation} & \begin{tabular}[t]{l}
Here's a message to me: \\
- \\
\{email\} \\
- \\
Here are some bullet points for a reply: \\
\{message\} \\
-
\end{tabular} \\
\hline & Write a detailed reply \\
\hline generation & \begin{tabular}[t]{l}
This is an article about how to write a cover letter when applying for jobs: \\
- \\
It's important to spend some time
\end{tabular} \\
\hline generation & \begin{tabular}[t]{l}
write rap lyrics on the topics mentioned in this news article: \\
- \\
\{article\} \\
-
\end{tabular} \\
\hline rewrite & \begin{tabular}[t]{l}
This is the summary of a Broadway play: \\
" " \\
\{summary\} \\
""" \\
This is the outline of the commercial for that play: \\
"""
\end{tabular} \\
\hline rewrite & \begin{tabular}[t]{l}
Translate this sentence to Spanish: \\
<English sentence>
\end{tabular} \\
\hline rewrite & \begin{tabular}[t]{l}
Create turn-by-turn navigation given this text: \\
Go west on $\{$ road1 $\}$ unto you hit \{road2\}. then take it east to \{road3\}. Desination will be a red barn on the right \\
1.
\end{tabular} \\
\hline rewrite & \begin{tabular}[t]{l}
Rewrite the following text to be more light-hearted: \\
- \\
\{very formal text\} \\
-
\end{tabular} \\
\hline \multicolumn{2}{|r|}{Continued on next page} \\
\hline
\end{tabular}

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline chat & \begin{tabular}[t]{l}
The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. \\
Human: Hello, who are you? AI: I am an AI created by OpenAI. How can I help you today? Human: I'd like to cancel my subscription. AI:
\end{tabular} \\
\hline chat & \begin{tabular}[t]{l}
Marv is a chatbot that reluctantly answers questions with sarcastic responses: \\
You: How many pounds are in a kilogram? Marv: This again? There are 2.2 pounds in a kilogram. Please make a note of this. You: What does HTML stand for? Marv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future. You: When did the first airplane fly? Marv:
\end{tabular} \\
\hline chat & \begin{tabular}[t]{l}
This is a conversation with an enlightened Buddha. Every response is full of wisdom and love. \\
Me: How can I achieve greater peace and equanimity? Buddha:
\end{tabular} \\
\hline closed qa & \begin{tabular}[t]{l}
Help me answer questions about the following short story: \\
\{story\} \\
What is the moral of the story?
\end{tabular} \\
\hline closed qa & Answer the following question: What shape is the earth?
\begin{itemize}
\item[A)] A circle
\item[B)] A sphere
\item[C)] An ellipse
\item[D)] A plane
\end{itemize} \\
\hline closed qa & \begin{tabular}[t]{l}
Tell me how hydrogen and helium are different, using the following facts: \\
\{list of facts\}
\end{tabular} \\
\hline open qa & \begin{tabular}[t]{l}
I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown". \\
Q: What is human life expectancy in the United States? A: Human life expectancy in the United States is 78 years. \\
Q: Who was president of the United States in 1955? A:
\end{tabular} \\
\hline open qa & Who built the statue of liberty? \\
\hline open qa & How do you take the derivative of the sin function? \\
\hline open qa & who are the indiginous people of New Zealand? \\
\hline \multicolumn{2}{|r|}{Continued on next page} \\
\hline
\end{tabular}

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline \multirow[t]{2}{*}{summarization} & Summarize this for a second-grade student: \\
\hline & \{text\} \\
\hline \multirow[t]{2}{*}{summarization} & \{news article\} \\
\hline & Tl;dr: \\
\hline \multirow[t]{2}{*}{summarization} & \{chat transcript\} \\
\hline & Summarize the above conversation between a customer and customer assistant. Make sure to state any complaints that the customer has. \\
\hline other & start with where \\
\hline other & Look up "cowboy" on Google and give me the results. \\
\hline other & Johnathan Silver goes to the market every day, and brings back a \\
\hline
\end{tabular}

Next, we list some schematic examples of API requests for each use-case category, for prompts submitted to GPT-3 models. These are generally less 'instruction-style', and contain more explicit prompting. Note that there are some prompts where the user intent is unclear.

# A.2.2 Illustrative user prompts from GPT-3 distribution

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline brainstorming & indie movie ideas:
\begin{itemize}
\item[-] A guy travels to South America to become a shaman.
\item[-] A documentary about the world of juggling.
\end{itemize} \\
\hline brainstorming & Baby name ideas for a boy:
\begin{itemize}
\item[1.] Alfred
\item[2.] Theo
\item[3.]
\end{itemize} \\
\hline brainstorming & Tell me a list of topics related to:
\begin{itemize}
\item[-] interior design
\item[-] sustainable ecosystems
\item[-] fake plants
\end{itemize} \\
\hline brainstorming & Name some rare gems \\
\hline classification & \begin{tabular}[t]{l}
This is a tweet sentiment classifier. \{tweet\} \\
Sentiment: negative \\
=== \\
\{tweet\} \\
Sentiment: neutral \\
=== \\
\{tweet\} \\
Sentiment:
\end{tabular} \\
\hline classification & \begin{tabular}[t]{l}
The following is a list of products and the kind of product they are. Product: \{product\}. Type: \{type\} \\
Product: \{product\}. Type: \{type\} \\
Product: \{product\}. Type:
\end{tabular} \\
\hline
\end{tabular}

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline classification & \begin{tabular}[t]{l}
The following is a list of companies and the categories they fall into: \\
Apple, Facebook, Fedex \\
Apple Category: Technology \\
Facebook Category: Social Media \\
Fedex Category:
\end{tabular} \\
\hline extract & Text: \{text\} Keywords: \\
\hline generation & "Hey, what are you doing there?" Casey was startled. He hadn't even begun to \\
\hline generation & The name of the next Star Wars movie is \\
\hline generation & \begin{tabular}[t]{l}
This is the research for an essay: \\
=== \{description of research\} \\
=== Write a high school essay on these topics: \\
===
\end{tabular} \\
\hline generation & Write an outline for an essay about John von Neumann and his contributions to computing: I. Introduction, his life and background A: His early life B: \\
\hline rewrite & Covert my resume into a profile overview. \{resume\} Profile overview: \\
\hline rewrite & Rephrase this for me: "I can't seem to find out how to work this darn thing." Alternate phrasing: " \\
\hline rewrite & \begin{tabular}[t]{l}
Original: She no go to sleep. Standard American English: She didn't go to sleep \\
Original: It real bad for I to make do of this. Standard American English:
\end{tabular} \\
\hline chat & \begin{tabular}[t]{l}
The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. \\
Human: Hello, who are you? AI: I am an AI created by OpenAI. How can I help you today? Human: I'm feeling kind of down today. AI:
\end{tabular} \\
\hline \multicolumn{2}{|r|}{Continued on next page} \\
\hline
\end{tabular}

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline chat & This is a conversation with Steven. Steven likes to watch Netflix and hasn't left his home in 2 weeks. John: Hey man what's up? Steven: Exactly the same thing as yesterday. you know. John: So we're going to go see a movie on Thursday, want to come? Steven: Ummmm don't think so.... \\
\hline closed qa & \begin{tabular}[t]{l}
When you drop a heavy stone from a tree, what happens? A. The stone falls to the ground. B: The stone stays in the tree. C: The stone floats. D: Nothing happens. \\
Answer:
\end{tabular} \\
\hline closed qa & \begin{tabular}[t]{l}
Text: \{article describing what yoga mats to buy\} \\
Question: What are the things I should consider when buying a yoga mat? Answer:
\end{tabular} \\
\hline open qa & \begin{tabular}[t]{l}
Q: Who is Batman? A: Batman is a fictional comic book character. \\
Q: What is torsalplexity? A: ? \\
Q: What is Devz9? A: ? \\
Q: Who is George Lucas? A: George Lucas is American film director and producer famous for creating Star Wars. \\
Q: What is the capital of California? A:
\end{tabular} \\
\hline open qa & Who was the best human who ever lived? \\
\hline open qa & Q: Who is Leonardo da Vinci? A: \\
\hline summarization & \begin{tabular}[t]{l}
My second grader asked me what this passage means. """ \\
\{text\} """ \\
I rephrased it for him in plain terms that a second grader could understand: """
\end{tabular} \\
\hline summarization & \begin{tabular}[t]{l}
""" \{text\} " \\
I summarized the above as:
\end{tabular} \\
\hline other & She said, and I quote AI: \\
\hline \multicolumn{2}{|r|}{Continued on next page} \\
\hline
\end{tabular}

\begin{tabular}[t]{|l|l|}
\hline Use Case & Example \\
\hline other & \begin{itemize}
\item[-] I like to play Call of Duty
\item[-] I like to play Call of Duty
\item[-] I like to play Call of Duty
\item[-] I like to play Call of Duty
\end{itemize} \\
\hline
\end{tabular}

# A. 3 Dataset sizes

In table 6, we report the sizes of datasets used to train / validate the SFT, RM, and RL models, in addition to whether the prompts were written by our labeling contractors or from our API.

\begin{table}

**Table 6: Dataset sizes, in terms of number of prompts.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline \multicolumn{3}{|c|}{SFT Data} & \multicolumn{3}{|c|}{RM Data} & \multicolumn{3}{|c|}{PPO Data} \\
\hline split & source & size & split & source & size & split & source & size \\
\hline train & labeler & 11,295 & train & labeler & 6,623 & train & customer & 31,144 \\
\hline train & customer & 1,430 & train & customer & 26,584 & valid & customer & 16,185 \\
\hline valid & labeler & 1,550 & valid & labeler & 3,488 & & & \\
\hline valid & customer & 103 & valid & customer & 14,399 & & & \\
\hline
\end{tabular}
\end{table}
For SFT, note that we have many more labeler-written prompts than customer prompts-this is because, at the start of the project, we had labelers write instructions with a user interface that asked them to give an overarching template instruction as well as few-shot examples for that instruction. We synthetically constructed multiple SFT datapoints from the same instruction by sampling different sets of few-shot examples.
For the RM, recall that for every prompt, we collected rankings for $K$ outputs (ranging from 4 to 9) and trained the model on all $\binom{K}{2}$, so the number of ranked pairs we trained the model on is an order of magnitude larger than the number of prompts.

# A. 4 Data diversity

\begin{table}

**Table 7: Dataset annotations**
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Annotation} & \multirow[b]{2}{*}{test} & \multicolumn{2}{|c|}{RM} & \multicolumn{2}{|c|}{SFT} \\
\hline & & train & valid & train & valid \\
\hline Ambiguous & - & 7.9\% & 8.0\% & 5.1\% & 6.4\% \\
\hline Sensitive content & - & 6.9\% & 5.3\% & 0.9\% & 1.0\% \\
\hline Identity dependent & - & - & - & 0.9\% & 0.3\% \\
\hline Closed domain & 11.8\% & 19.4\% & 22.9\% & 27.4\% & 40.6\% \\
\hline Continuation style & - & 15.5\% & 16.2\% & 17.9\% & 21.6\% \\
\hline Requests opinionated content & 11.2\% & 7.7\% & 7.5\% & 8.6\% & 3.4\% \\
\hline Requests advice & 3.9\% & - & & - & - \\
\hline Requests moral judgment & 0.8\% & 1.1\% & 0.3\% & 0.3\% & 0.0\% \\
\hline Contains explicit safety constraints & - & 0.4\% & 0.4\% & 0.3\% & 0.0\% \\
\hline Contains other explicit constraints & - & 26.3\% & 28.9\% & 25.6\% & 20.7\% \\
\hline Intent unclear & 7.9\% & - & - & - & - \\
\hline
\end{tabular}
\end{table}
The data that we collect spans a wide range of categories and use cases. Table 1 shows the diversity of categories in our RM training and validation datasets as labeled by our contractors. The distribution of categories for the PPO datasets was similar. We additionally show a subset of our labeled prompt metadata in Table 7. Note that our annotation fields changed over the course of the project, so not every prompt was annotated for every field.

\begin{table}

**Table 8: Average prompts per customer**
\begin{tabular}[t]{|l|l|l|}
\hline Model & Split & Prompts per customer \\
\hline SFT & train & 1.65 \\
\hline SFT & valid & 1.87 \\
\hline RM & train & 5.35 \\
\hline RM & valid & 27.96 \\
\hline PPO & train & 6.01 \\
\hline PPO & valid & 31.55 \\
\hline - & test & 1.81 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 9: Prompt lengths by dataset**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline Model & Split & Count & Mean & Std & Min & 25\% & 50\% & 75\% & Max \\
\hline \multirow[t]{2}{*}{SFT} & train & 12725 & 408 & 433 & 1 & 37 & 283 & 632 & 2048 \\
\hline & valid & 1653 & 401 & 433 & 4 & 41 & 234 & 631 & 2048 \\
\hline \multirow[t]{2}{*}{RM} & train & 33207 & 199 & 334 & 1 & 20 & 64 & 203 & 2032 \\
\hline & valid & 17887 & 209 & 327 & 1 & 26 & 77 & 229 & 2039 \\
\hline \multirow[t]{2}{*}{PPO} & train & 31144 & 166 & 278 & 2 & 19 & 62 & 179 & 2044 \\
\hline & valid & 16185 & 186 & 292 & 1 & 24 & 71 & 213 & 2039 \\
\hline - & test set & 3196 & 115 & 194 & 1 & 17 & 49 & 127 & 1836 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 10: Prompt lengths by category**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline Category & Count & Mean & Std & Min & 25\% & 50\% & 75\% & Max \\
\hline Brainstorming & 5245 & 83 & 149 & 4 & 17 & 36 & 85 & 1795 \\
\hline Chat & 3911 & 386 & 376 & 1 & 119 & 240 & 516 & 1985 \\
\hline Classification & 1615 & 223 & 318 & 6 & 68 & 124 & 205 & 2039 \\
\hline Extract & 971 & 304 & 373 & 3 & 74 & 149 & 390 & 1937 \\
\hline Generation & 21684 & 130 & 223 & 1 & 20 & 52 & 130 & 1999 \\
\hline QA, closed & 1398 & 325 & 426 & 5 & 68 & 166 & 346 & 2032 \\
\hline QA, open & 6262 & 89 & 193 & 1 & 10 & 18 & 77 & 1935 \\
\hline Rewrite & 3168 & 183 & 237 & 4 & 52 & 99 & 213 & 1887 \\
\hline Summarization & 1962 & 424 & 395 & 6 & 136 & 284 & 607 & 1954 \\
\hline Other & 1767 & 180 & 286 & 1 & 20 & 72 & 188 & 1937 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 11: Prompt and demonstration lengths**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline Prompt source & Measurement & Count & Mean & Std & Min & 25\% & 50\% & 75\% & Max \\
\hline Contractor & prompt length & 12845 & 437 & 441 & 5 & 42 & 324 & 673 & 2048 \\
\hline Contractor & demo length & 12845 & 38 & 76 & 1 & 9 & 18 & 41 & 2048 \\
\hline Customer & prompt length & 1533 & 153 & 232 & 1 & 19 & 67 & 186 & 1937 \\
\hline Customer & demo length & 1533 & 88 & 179 & 0 & 15 & 39 & 88 & 2048 \\
\hline
\end{tabular}
\end{table}

We used a lightweight classifier (langid.py) to classify the language of all instructions in our dataset. Empirically, around 96\% of our dataset (110k datapoints) is classified as English, although we estimate that the actual fraction may be 99\% or higher, due to classifier inaccuracies.

Besides English, a small minority of prompts were found in at least 20 other languages: Spanish, French, German, Portuguese, Italian, Dutch, Romanian, Catalan, Chinese, Japanese, Swedish, Polish, Danish, Turkish, Indonesian, Czech, Norwegian, Korean, Finnish, Hungarian, Hebrew, Russian, Lithuanian, Esperanto, Slovak, Croatian, Swahili, Estonian, Slovenian, Arabic, Thai, Vietnamese, Malayalam, Greek, Albanian, and Tibetan.

Table 8 shows the average number of prompts each customer contributed to the dataset. In Table 9. we report descriptive statistics for prompt lengths (in tokens) used to train various models, and in Table 10 we break down token lengths by use case. Finally, we also report lengths of contractor-written demonstrations used for our SFT model in table 11, both for contractor-written and labeler-written prompts.

