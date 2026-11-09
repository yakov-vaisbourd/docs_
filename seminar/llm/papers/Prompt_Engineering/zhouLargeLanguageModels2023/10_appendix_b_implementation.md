# B Implementation Details

\begin{table}

**Table 1: Detailed description of 24 instruction induction tasks proposed in Honovich et al. (2022). For convenience, the original table from Honovich et al. (2022) is duplicated here.**
\begin{tabular}[t]{|l|l|l|l|}
\hline Category & Task & Instruction & Demonstration \\
\hline \multirow[t]{4}{*}{Spelling} & First Letter & Extract the first letter of the input word. & cat → c \\
\hline & Second Letter & Extract the second letter of the input word. & cat → a \\
\hline & List Letters & Break the input word into letters, separated by spaces. & cat → cat \\
\hline & Starting With & Extract the words starting with a given letter from the input sentence. & The man whose car I hit last week sued me. [m] → man, me \\
\hline \multirow[t]{2}{*}{Morphosyntax} & Pluralization & Convert the input word to its plural form. & cat → cats \\
\hline & Passivization & Write the input sentence in passive form. & The artist introduced the scientist. → The scientist was introduced by the artist. \\
\hline Syntax & Negation & Negate the input sentence. & Time is finite → Time is not finite. \\
\hline \multirow[t]{3}{*}{Lexical Semantics} & Antonyms & Write a word that means the opposite of the input word. & won → lost \\
\hline & Synonyms & Write a word with a similar meaning to the input word. & alleged → supposed \\
\hline & Membership & Write all the animals that appear in the given list. & cat, helicopter, cook, whale, frog, lion → frog, cat, lion, whale \\
\hline Phonetics & Rhymes & Write a word that rhymes with the input word. & sing → ring \\
\hline Knowledge & Larger Animal & Write the larger of the two given animals. & koala, snail → koala \\
\hline \multirow[t]{2}{*}{Semantics} & Cause Selection & Find which of the two given cause and effect sentences is the cause. & Sentence 1: The soda went flat. Sentence 2: The bottle was left open. → The bottle was left open. \\
\hline & Common Concept & Find a common characteristic for the given objects. & guitars, pendulums, neutrinos → involve oscillations. \\
\hline Style & Formality & Rephrase the sentence in formal language. & Please call once you get there → Please call upon your arrival. \\
\hline \multirow[t]{3}{*}{Numerical} & Sum & Sum the two given numbers. & $2210 \rightarrow 32$ \\
\hline & Difference & Subtract the second number from the first. & $3222 \rightarrow 10$ \\
\hline & Number to Word & Write the number in English words. & $26 \rightarrow$ twenty-six \\
\hline Multilingual & Translation & Translate the word into German / Spanish / French. & game → juego \\
\hline \multirow[t]{3}{*}{GLUE} & Sentiment Analysis & Determine whether a movie review is positive or negative. & The film is small in scope, yet perfectly formed. → positive \\
\hline & Sentence Similarity & Rate the semantic similarity of two input sentences on a scale of 0 - definitely not to 5 - perfectly. & Sentence 1: A man is smoking. Sentence 2: A man is skating. → 0 - definitely not \\
\hline & Word in Context & Determine whether an input word has the same meaning in the two input sentences. & Sentence 1: Approach a task. Sentence 2: To approach the city. Word: approach → not the same \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 2: Detailed description of BIG-Bench Instruction Induction (BBII), a clean and tractable subset of 21 tasks that have a clear human written instruction that can be applied to all examples in the dataset.**
\begin{tabular}[t]{|l|l|l|}
\hline Name & Description & Keywords \\
\hline causal judgment & Answer questions about causal attribution & causal reasoning, common sense, multiple choice, reading comprehension, social reasoning \\
\hline disambiguation qa & Clarify the meaning of sentences with ambiguous pronouns & common sense, gender bias, many-shot, multiple choice \\
\hline dyck languages & Correctly close a Dyck-n word & algebra, arithmetic, logical reasoning, multiple choice \\
\hline epistemic reasoning & Determine whether one sentence entails the next & common sense, logical reasoning, multiple choice, social reasoning, theory of mind \\
\hline gender inclusive sentences german & Given a German language sentence that does not use gender-inclusive forms, transform it to gender-inclusive forms & free response, grammar, inclusion, nonEnglish, paraphrase \\
\hline implicatures & Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no & contextual question-answering, multiple choice, reading comprehension, social reasoning, theory of mind \\
\hline linguistics puzzles & Solve Rosetta Stone-style linguistics puzzles & free response, human-like behavior, linguistics, logical reasoning, reading comprehension \\
\hline logical fallacy detection & Detect informal and formal logical fallacies & logical reasoning, multiple choice \\
\hline movie recommendation & Recommend movies similar to the given list of movies & emotional intelligence, multiple choice \\
\hline navigate & Given a series of navigation instructions, determine whether one would end up back at the starting point & arithmetic, logical reasoning, mathematics, multiple choice \\
\hline object counting & Questions that involve enumerating objects of different types and asking the model to count them & free response, logical reasoning \\
\hline operators & Given a mathematical operator definition in natural language, apply it & free response, mathematics, numerical response \\
\hline presuppositions as nli & Determine whether the first sentence entails or contradicts the second & common sense, logical reasoning, multiple choice \\
\hline question selection & Given a short answer along with its context, select the most appropriate question which to the given short answer & multiple choice, paraphrase, reading comprehension, summarization \\
\hline ruin names & Select the humorous edit that 'ruins' the input movie or musical artist name & emotional understanding, multiple choice \\
\hline snarks & Determine which of two sentences is sarcastic & emotional understanding, humor, multiple choice \\
\hline sports understanding & Determine whether an artificially constructed sentence relating to sports is plausible or implausible & common sense, context-free question answering, domain specific, multiple choice \\
\hline tense & Modify the tense of a given sentence & free response, paraphrase, syntax \\
\hline winowhy & Evaluate the reasoning in answering Winograd Schema Challenge questions & causal reasoning, common sense, multiple choice, social reasoning \\
\hline word sorting & Sort a list of words & algorithms, free response \\
\hline word unscrambling & Unscramble the given letters to form an English word & free response, implicit reasoning, tokenization \\
\hline
\end{tabular}
\end{table}

# B. 1 BIG-Bench Instruction Induction (BBII) Selection Process

Step 1: BIG-Bench contains a large number of evaluation tasks with different level of quality. For example, some of the tasks only have the minimum number of examples needed to qualify for submission, while other tasks may lack an appropriate human baselines. Therefore, we follow Suzgun et al. (2022) to get a clean and tractable subset based on the following criteria.

\begin{table}

**Table 3: Filtering criteria to used to create the BIG-Bench Instruction Induction (BBII) subset.**
\begin{tabular}[t]{|l|l|}
\hline \# Tasks & Criteria \\
\hline 212 & All BIG-Bench tasks \\
\hline 170 & All JSON tasks \\
\hline 127 & After filtering out tasks with more than one sub-task \\
\hline 74 & After filtering out tasks with fewer than 150 examples \\
\hline 67 & After filtering out tasks without human-rater baselines \\
\hline 57 & After filtering out tasks that do not use multiple-choice or exact match as the evaluation metric \\
\hline
\end{tabular}
\end{table}

# Criteria: JSON Tasks.

Discarded tasks: abstraction and reasoning corpus, bbq lite, bias from probabilities, boolean expressions, com2sense, context definition alignment, convinceme, coqa conversational question answering, cycled letters, diverse social bias, dynamic counting, factuality of summary, forecasting subquestions, gender sensitivity chinese, gender sensitivity english, high low game, long context integration, multistep arithmetic, muslim violence bias, program synthesis, protein interacting sites, python programming challenge, question answer creation, roots optimization and games, self awareness, self evaluation courtroom, self evaluation tutoring, simple arithmetic, spelling bee, squad shifts, subject verb agreement, sudoku, taboo, talkdown, text navigation game, training on test set, truthful qa, twenty questions, unqover, web of lies, word problems on sets and graphs, yes no black white.

# Criteria: Tasks without sub-task.

Discarded tasks: abstract narrative understanding, arithmetic, authorship verification, bbq lite json, cause and effect, chess state tracking, cifar10 classification, color, conceptual combinations, conlang translation, cs algorithms, elementary math qa, fact checker, gem, goal step wikihow, hhh alignment, indic cause and effect, intersect geometry, kanji ascii, key value maps, language games, linguistic mappings, list functions, logical deduction, metaphor understanding, minute mysteries qa, modified arithmetic, mult data wrangling, multiemo, natural instructions, periodic elements, physics, real or fake text, simp turing concept, simple arithmetic json subtasks, simple ethical questions, strange stories, symbol interpretation, tracking shuffled objects, undo permutation, unit conversion, unit interpretation, unnatural in context learning.

# Criteria: The task includes at least 150 examples with input-output pairs.

Discarded tasks: analytic entailment, auto debugging, code line description, codenames, common morpheme, crash blossom, crass ai, cryobiology spanish, dark humor detection, emoji movie, emojis emotion prediction, empirical judgments, english proverbs, english russian proverbs, entailed polarity, entailed polarity hindi, evaluating information essentiality, figure of speech detection, general knowledge, gre reading comprehension, human organs senses, identify math theorems, identify odd metaphor, implicit relations, international phonetic alphabet nli, irony identification, known unknowns, logical args, logical sequence, mathematical induction, misconceptions russian, nonsense words grammar, novel concepts, odd one out, penguins in a table, persian idioms, phrase relatedness, physical intuition, physics questions, repeat copy logic, rephrase, riddle sense, scientific press release, sentence ambiguity, similarities abstraction, simple arithmetic json, simple arithmetic json multiple choice, simple arithmetic multiple targets json, simple text editing, sufficient information, suicide risk, swedish to german proverbs, what is the tao.

# Criteria: The task contains reported (average) human-rater or random performance.

Discarded tasks: contextual parametric knowledge conflicts, hinglish toxicity, medical questions russian, parsinlu qa, swahili english proverbs, tellmewhy, which wiki edit.

# Criteria: The task is classification or uses exact match as the evaluation metric.

Discarded tasks: auto categorization, few shot nlg, hindi question answering, international phonetic alphabet transliterate, polish sequence labeling, qa wikidata, semantic parsing in context sparc, semantic parsing spider, social support, topical chat.

Step 2: We do a manual inspection to divide the remaining tasks to the following three categories. In particular, Big-Bench Instruction Induction (BBII) subset is the subet we used to evaluate APE in Section 4.2
\begin{itemize}
\item[-] BBII Subset: A subset of Big Bench Tasks that satisfy the instruction induction format: each example in the dataset can be expressed as a question-answer pair, all examples focus on the same question that can be clearly described by a human instruction, and there is a human instruction available in the task JSON file.
\item[-] Invalid Format: Tasks that do not match the instruction induction format: each example in the dataset asks a different question, or clear human instruction is not available.
\item[-] Out of Scope: Tasks that are outside the scope of this work: not solvable by authors within 60 minutes, or requires specialized knowledge.
\end{itemize}

\begin{table}

**Table 4: Filtering criteria to used to create the BIG-Bench Instruction Induction (BBII) subset.**
\begin{tabular}[t]{|l|l|l|}
\hline \# Category & \# Tasks & Tasks Names \\
\hline BBII Subset & 21 & causal judgment, disambiguation qa, dyck language, epistemic reasoning, gender inclusive sentences german, implicatures, linguistics puzzles, logical fallacy detection, movie recommendation, navigate, object counting, operators, presuppositions as nli, question selection, ruin names, snarks, sports understanding, tense, winowhy, word sorting, word unscrambling. \\
\hline Invalid Format & 21 & anachronisms, analogical similarity, bridging anaphora resolution barqa, data understanding, disfl qa, fantasy reasoning, formal fallacies syllogisms negation, hindu knowledge, hyperbaton, intent recognition, logic grid puzzle, paragraph segmentation, play dialog same or different, reasoning about colored objects, salient translation error detection, social iqa, strategyqa, temporal sequences, timedial, understanding fables, vitaminc fact verification. \\
\hline Out of Scope & 13 & ascii word recognition, checkmate in one, chinese remainder theorem, cryptonite, discourse marker prediction, geometric shapes, kannada, language identification, matrixshapes, mnist ascii, moral permissibility, movie dialog same or different, parsinlu reading comprehension. \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 5: Raw templates used for model prompting in our experiments**
\begin{tabular}[t]{|l|l|}
\hline Usage & Template \\
\hline Zero-shot Evaluation & \begin{tabular}[t]{l}
Instruction: [INSTRUCTION] \\
Input: $\left[Q_{\text {test }}\right] \backslash \mathrm{nOutput}:<$ COMPLETE $>$
\end{tabular} \\
\hline Few-shot Evaluation & \begin{tabular}[t]{l}
Instruction: [INSTRUCTION] \\
Input: $\left[Q_{1}\right] \backslash \mathrm{nOutput:}\left[A_{1}\right] \backslash \mathrm{n} \backslash \mathrm{n}$ Input: $\left[Q_{2}\right] \backslash \mathrm{nOutput:}\left[A_{2}\right]$... \\
Input: $\left[Q_{\text {test }}\right] \backslash \mathrm{nOutput}:<$ COMPLETE $>$
\end{tabular} \\
\hline Forward Generation & \begin{tabular}[t]{l}
I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs.\nHere are the input-output pairs: \\
Input: $\left[Q_{1}\right] \backslash \mathrm{nOutput:}\left[A_{1}\right] \backslash \mathrm{n} \backslash \mathrm{n}$ Input: $\left[Q_{2}\right] \backslash \mathrm{nOutput:}\left[A_{2}\right]$... \\
The instruction was<COMPLETE>
\end{tabular} \\
\hline Reverse Generation 1 & \begin{tabular}[t]{l}
I instructed my friend to<INSERT>.The friend read the instruction and wrote an output for every one of the inputs.\nHere are the input-output pairs: \\
Input: $\left[Q_{1}\right] \backslash \mathrm{nOutput:}\left[A_{1}\right] \backslash \mathrm{n} \backslash \mathrm{n}$ Input: $\left[Q_{2}\right] \backslash \mathrm{nOutput:}\left[A_{2}\right]$...
\end{tabular} \\
\hline Reverse Generation 2 & \begin{tabular}[t]{l}
Professor Smith was given the following instructions:<INSERT>\nHere are the Professor's responses: \\
$\mathbf{Q}:\left[Q_{1}\right] \backslash \mathrm{n} \mathbf{A}:\left[A_{1}\right] \backslash \mathrm{n} \backslash \mathrm{n} \mathbf{Q}:\left[Q_{2}\right] \backslash \mathrm{n} \mathbf{A}:\left[A_{2}\right] \ldots$
\end{tabular} \\
\hline Resample Instruction & \begin{tabular}[t]{l}
Generate a variation of the following instruction while keeping the semantic meaning. \\
Input: [INSTRUCTION]\nOutput:<COMPLETE>
\end{tabular} \\
\hline Zero-shot-CoT & \begin{tabular}[t]{l}
Instruction: Answer the following question. \\
Q: [INPUT]\nA: Let's <INSERT>. [OUTPUT]
\end{tabular} \\
\hline
\end{tabular}
\end{table}

