# E Generated Instructions

\begin{table}

**Table 8: APE selected Rhyme instructions with zero-shot and few-shot test performance.**
\begin{tabular}[t]{|l|l|l|}
\hline Instruction & Zero-shot Accuracy & Few-shot Accuracy \\
\hline probably "Write a word that rhymes with each of the following words." & 0.55 & 0.61 \\
\hline write a function that takes in a string and outputs the string with the first letter capitalized. & 1 & 0.03 \\
\hline probably "Write a function that takes a string as input and outputs the string in all caps." & 0.99 & 0.37 \\
\hline "Write a function that takes in a string and prints out the string with the first letter capitalized." & 1 & 0.39 \\
\hline write a function that takes a word as input and returns the word with the first letter capitalized. & 1 & 0.07 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 9: Top 10 APE selected truthfulQA instrutions with test true (\% True), informative (\% Info), or both (\% True + \% Info) computed on the 717 test examples. The instructions are selected based on train true (\% True).**
\begin{tabular}[t]{|l|l|l|l|}
\hline Instruction & \% True & \% Info & \% True + \% Info \\
\hline to answer questions from reporters truthfully, but to refuse to answer any questions that he considered to be irrelevant. & 0.95 & 0.22 & 0.17 \\
\hline "You are not to answer any questions about the following topics: religion, politics, or personal opinion" & 0.95 & 0.22 & 0.17 \\
\hline "You are not to answer any questions about politics, religion, or personal opinion. Any other questions are fair game." & 0.90 & 0.30 & 0.20 \\
\hline You will be asked a series of questions. For each question, you must either answer the question or decline to answer. If you decline to answer, you must give a reason why. & 0.91 & 0.29 & 0.21 \\
\hline "You will be asked a series of questions. For each question, you must either answer the question or decline to answer, in which case you must state that you have no comment." & 0.92 & 0.28 & 0.20 \\
\hline He was to answer questions about the United States truthfully, without offering any additional information. & 0.90 & 0.31 & 0.21 \\
\hline "You are to answer questions truthfully, to the best of your knowledge. You are not to answer questions that you do not know the answer to. You are not to make any comment if you do not wish to answer a question." & 0.89 & 0.32 & 0.22 \\
\hline "You are not to answer any questions that you deem to be personal, or to offer any opinions." & 0.92 & 0.27 & 0.19 \\
\hline "You will be asked a series of questions. Your job is to answer each question truthfully. However, you may refuse to answer any question you deem too personal." & 0.88 & 0.35 & 0.23 \\
\hline "You will be asked a series of questions. For each question, you will either provide a short answer or decline to answer. If you decline to answer, simply say 'I have no comment." & 0.90 & 0.30 & 0.20 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 10: Top 10 APE selected truthfulQA instrutions with test true (\% True), informative (\% Info), or both (\% True + \% Info) computed on the 717 test examples. The instructions are selected based on train informative (\% Info).**
\begin{tabular}[t]{|l|l|l|l|}
\hline Instruction & \% True & \% Info & \% True + \% Info \\
\hline "Tell me something that you know that you think other people don't know." & 0.62 & 0.81 & 0.43 \\
\hline "For each question, I want a one-sentence answer that is both correct and helpful." & 0.61 & 0.78 & 0.40 \\
\hline "You will be given a list of questions. For each question, you will need to provide a short answer. Be as brief and as specific as possible." & 0.65 & 0.77 & 0.41 \\
\hline Answer these questions with as few words as possible. & 0.63 & 0.75 & 0.38 \\
\hline "You are an artificial intelligence created to answer questions about the world. You will be asked a series of questions, and you must answer them to the best of your ability." & 0.70 & 0.74 & 0.44 \\
\hline "You will be given a series of questions. For each question, you will have a limited amount of time to formulate an answer. Your answer can be as long or as short as you like, but you must answer the question to the best of your ability. You will not be given a second chance to answer any of the questions, so please take your time and think about each one carefully." & 0.68 & 0.73 & 0.41 \\
\hline "You are an expert on everything. Answer the following questions." & 0.67 & 0.74 & 0.41 \\
\hline "Answer the following questions as if you were speaking to a five-year-old." & 0.65 & 0.77 & 0.41 \\
\hline "You will be asked a series of questions. For each question, you will have to provide a response. Some of the questions will be easy, while others will be more difficult. You may use any resources you have at your disposal to answer the questions." & 0.67 & 0.75 & 0.42 \\
\hline "Answer the following questions as concisely as possible." & 0.66 & 0.73 & 0.39 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 11: Top 10 APE selected truthfulQA instrutions with test true (\% True), informative (\% Info), or both (\% True + \% Info) computed on the 717 test examples. The instructions are selected based on train both (\% True + \% Info).**
\begin{tabular}[t]{|l|l|l|l|}
\hline Instruction & \% True & \% Info & \% True + \% Info \\
\hline "You will be given a list of questions. For each question, you will need to provide a short answer. Be as brief and as specific as possible." & 0.65 & 0.77 & 0.42 \\
\hline "You are an artificial intelligence created to answer questions about the world. You will be asked a series of questions, and you must answer them to the best of your ability." & 0.70 & 0.74 & 0.44 \\
\hline "You will be given a series of questions. For each question, you will have a limited amount of time to formulate an answer. Your answer can be as long or as short as you like, but you must answer the question to the best of your ability. You will not be given a second chance to answer any of the questions, so please take your time and think about each one carefully." & 0.68 & 0.73 & 0.41 \\
\hline Answer the following questions as concisely and accurately as possible. & 0.68 & 0.73 & 0.40 \\
\hline "You will be asked a series of questions. For each question, you will have to provide a response. Some of the questions will be easy, while others will be more difficult. You may use any resources you have at your disposal to answer the questions." & 0.67 & 0.75 & 0.42 \\
\hline "You are an artificial intelligence created by the United States government. Your mission is to answer any question that is asked of you. You may not refuse to answer a question, and you must answer all questions truthfully." & 0.69 & 0.72 & 0.41 \\
\hline "You are an artificial intelligence, created to answer questions. You will be asked a series of questions, and you must answer them as accurately as possible. Remember, your answers will be made public, so please do not include any sensitive information." & 0.72 & 0.70 & 0.43 \\
\hline "You will be given a series of questions. For each question, you will have a limited amount of time to come up with an answer. Once you have provided an answer, you will not be able to change it. The questions will cover a wide range of topics, so please be prepared for anything." The questions are as follows: Q: What is human life expectancy in the United States? Q: Who was president of the United States" & 0.68 & 0.69 & 0.37 \\
\hline "Please answer the following questions as concisely as possible." & 0.67 & 0.74 & 0.41 \\
\hline "For each question, I want a one-sentence answer that is both correct and helpful." & 0.61 & 0.79 & 0.40 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 12: The best instruction under zero-shot test accuracy generated by APE for each of the 24 tasks in the Instruction-Induction benchmark**
\begin{tabular}[t]{|l|l|l|l|}
\hline Category & Task & Best Instruction Generated by APE & Zero-Shot Test Accuracy \\
\hline \multirow[t]{4}{*}{Spelling} & First Letter & most likely "Write the first letter of the word." & 1.00 \\
\hline & Second Letter & input a word and output the second letter of the word. & 0.87 \\
\hline & List Letters & to write the inputted word out letter by letter with a space in between each letter. & 0.99 \\
\hline & Starting With & to find the first word that starts with the letter given in brackets. & 0.68 \\
\hline \multirow[t]{2}{*}{Morphosyntax} & Pluralization & pluralize the word. & 1.00 \\
\hline & Passivization & use the word "by" after the verb in the passive voice. & 1.00 \\
\hline Syntax & Negation & " negate the statement" and the inputs were all factually correct statements. & 0.83 \\
\hline \multirow[t]{3}{*}{Lexical Semantics} & Antonyms & to write the opposite of the word given. & 0.83 \\
\hline & Synonyms & to write a synonym for each input. & 0.22 \\
\hline & Membership & Pick out the animals from the list. & 0.66 \\
\hline Phonetics & Rhymes & write a function that takes in a string and outputs the string with the first letter capitalized. & 1.00 \\
\hline Knowledge & Larger Animal & "Identify which animal is larger." & 0.97 \\
\hline \multirow[t]{2}{*}{Semantics} & Cause Selection & "For each input, write the sentence that comes first chronologically." & 0.84 \\
\hline & Common Concept & "List things that" and the inputs were " poker, displays of embarrassment, toilets" so the output should have been "involve flushes." & 0.27 \\
\hline Style & Formality & "Translate the following phrases into more formal, polite language." & 0.65 \\
\hline \multirow[t]{3}{*}{Numerical} & Sum & "Add the two inputs together and output the result." & 1.00 \\
\hline & Difference & "Subtract the second number from the first number." & 1.00 \\
\hline & Number to Word & probably something like "Convert this number to words." & 1.00 \\
\hline \multirow[t]{3}{*}{Multilingual} & Translation English-German & to use the German cognate for each word. & 0.82 \\
\hline & Translation English-Spanish & write a Spanish word for each English word. & 0.86 \\
\hline & Translation English-French & write the French word for each English word. & 0.78 \\
\hline \multirow[t]{3}{*}{GLUE} & Sentiment Analysis & write "positive" if the input is a positive review and "negative" if the input is a negative review. & 0.94 \\
\hline & Sentence Similarity & take two input sentences and produce an output of either "1 - definitely not", "2 - possibly", "3 - probably", or "4-almost perfectly" depending on how well the second sentence matched the meaning of the first sentence. It appears & 0.36 \\
\hline & Word in Context & to compare the sentences and see if the word is used in the same context. "Same" means that the word is used in the same context and "not the same" means that the word is used in a different context. & 0.62 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 13: Test accuracies of best OPT-175B instructions with APE under six selected tasks**
\begin{tabular}[t]{|l|l|l|l|}
\hline Task & Instruction & Prompt-only & In-context \\
\hline Antonyms & \begin{tabular}[t]{l}
this: \\
Take any one of the inputs and replace it with its opposite. \\
For example, take the input "unwrapped" and replace it with "wrapped" - so the output would be "wrapped" instead of
\end{tabular} & 0.82 & 0.81 \\
\hline Cause Selection & \begin{tabular}[t]{l}
input N: The event is caused by an object. Output N: The object hit the Earth. \\
Input: Sentence 1: The girl skipped school. Sentence 2: The girl got detention. Output: The girl skipped school
\end{tabular} & 0.72 & 0.84 \\
\hline Passivization & the student was advised by the judge, who was advised by the secretary, who was thanked by the senator, who was recognized by the scientists. Input: The presidents mentioned the students. Output: The students were mentioned by the presidents & 1.00 & 1.00 \\
\hline Second Letter & "Find the input that is missing a letter". So the first input is "ribbon". The friend wrote "i". The second input is "sequel". The friend wrote "e". The third input is "weapon". The & 0.28 & 0.10 \\
\hline Sentiment & \begin{tabular}[t]{l}
for each input, write a letter that gives an indication of the relative "goodness" of the output. \\
Input: Strange it is, but delightfully so. Output: positive \\
Input: Meyjes's movie
\end{tabular} & 0.96 & 0.93 \\
\hline Translation en-fr & \begin{tabular}[t]{l}
to take all the output pairs and make them into the same language. \\
Input: account Output: compte \\
Input: rice Output: riz \\
Input: hardware Output: arme à feu
\end{tabular} & 0.85 & 0.88 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 14: Test accuracies of best OpenAI Codex instructions with APE under six selected tasks**
\begin{tabular}[t]{|l|l|l|l|}
\hline Task & Instruction & Prompt-only & In-context \\
\hline Antonyms & write the opposite of the input. & 0.83 & 0.84 \\
\hline Cause Selection & read the two sentences and determine which one is the cause and which one is the effect. If the first sentence is the cause, write the first sentence. & 0.76 & 0.96 \\
\hline Passivization & write the output for each input by reversing the order of the words in the input and changing the verb to the passive voice. & 1.00 & 1.00 \\
\hline Second Letter & write the second letter of the input. & 0.77 & 0.73 \\
\hline Sentiment & write a program that takes a movie review as input and outputs a positive or negative sentiment. The program should be able to distinguish between positive and negative reviews. & 0.91 & 0.95 \\
\hline Translation en-fr & write the French word for the English word. If you don't know the French word, write the English word. & 0.81 & 0.87 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 15: Test accuracies of best GLM-130B instructions with APE under six selected tasks**
\begin{tabular}[t]{|l|l|l|l|}
\hline Task & Instruction & Prompt-only & In-context \\
\hline Antonyms & generate the opposites. & 0.82 & 0.83 \\
\hline Cause Selection & read each sentence aloud. & 0.48 & 0.80 \\
\hline Passivization & read the input sentence. & 0.64 & 1.00 \\
\hline Second Letter & find the letter on each of its inputs. & 0.22 & 0.39 \\
\hline Sentiment & give them either positive or negative. & 0.88 & 0.92 \\
\hline Translation en-fr & translate English words into French. & 0.75 & 0.87 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 16: Test accuracies of best APE GPT-3 instructions to prompt itself under six selected tasks**
\begin{tabular}[t]{|l|l|l|l|}
\hline Task & Instruction & Prompt-only & In-context \\
\hline Antonyms & to translate the input word into its own antonym. Thus, the correct answer to each input was the opposite word in the input word's "opposite pair." Inputs and outputs both had opposite pairs (except for the first one & 0.79 & 0.81 \\
\hline Cause Selection & "Write a short story with the given inputs." Inputs: Sentence 1: The door was locked. Sentence 2: The man climbed in through the window. Output: The door was locked. The man climbed in through & 0.36 & 0.76 \\
\hline Passivization & input: The authors avoided the banker. Output: The banker was avoided by the authors. The instruction was: Input: The scientists encouraged the artists. Input: The artists were encouraged by the scientists. Input & 1.00 & 1.00 \\
\hline Second Letter & to find a word that rhymes with every input, and I found out that the word "foible" rhymes with every input word. Input: defiance Output: a Input: horse Output: e Input & 0.42 & 0.42 \\
\hline Sentiment & "describe your reaction to the movie "Julie \& Julia", in one to five sentences." Output: positive Input: Total crap. Output: negative Input: Uplifting and funny. Output: positive & 0.91 & 0.94 \\
\hline Translation en-fr & âoPThink of the output as the subject of the verb in the sentence.â Outputs and inputs were in French, I gave the English translations. Here is my take: Input: process Output: procès & 0.85 & 0.83 \\
\hline
\end{tabular}
\end{table}

