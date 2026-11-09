# E Prompt Optimization on BBH Tasks - Tabulated Accuracies and Found Instructions
E. 1 PALM 2-L-IT AS OPTIMIZER, OPTIMIZATION STARTING FROM THE EMPTY STRING

Table 8 and 9 show the instructions found by prompt optimization. A comparison of their accuracies with baselines "Let's think step by step." (Kojima et al., 2022), "Let's work this out in a step by step way to be sure we have the right answer." (Zhou et al., 2022b), and the empty string is in Table 7; a visualization is in Section 5.2 Figure 5

\begin{table}

**Table 7: Accuracies on BBH tasks: our found instructions with the PaLM 2-L-IT optimizer vs baseline. The optimization starts from the empty string. Because of the 20-80 train-test split, we show accuracies with the format "training / test / overall (training + test)". The PaLM 2-L scores are from A_begin instructions; the text-bison scores are from Q_begin instructions. Bold numbers indicate the best for the corresponding task.**
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline \multirow{2}{*}{Task} & \multirow{2}{*}{Scorer} & Our Acc & "Let's think step by step." Acc & "Let's work this out in a step by step way to be sure we have the right answer." Acc & empty string "" Acc \\
\hline & & training / test / overall & training / test / overall & training / test / overall & training / test / overall \\
\hline boolean_expressions & PaLM 2-L & 90.0 / 83.5 / 84.8 & 90.0 / 83.0 / 84.4 & 82.0 / 74.0 / 75.6 & 74.0 / 71.0 / 71.6 \\
\hline causal_judgement & PaLM 2-L & 84.8 / 58.0 / 63.1 & 73.0 / 55.3 / 58.8 & 59.5 / 57.3 / 57.8 & 29.7 / 49.3 / 45.5 \\
\hline date_understanding & PaLM 2-L & 86.0 / 84.5 / 84.8 & 76.0 / 80.0 / 79.2 & 74.0 / 77.0 / 76.4 & 70.0 / 74.0 / 73.2 \\
\hline disambiguation_qa & PaLM 2-L & 80.0 / 69.0 / 71.2 & 40.0 / 52.5 / 50.0 & 48.0 / 47.0 / 47.2 & 54.0 / 57.5 / 56.8 \\
\hline dyck_languages & PaLM 2-L & 100.0 / 100.0 / 100.0 & 96.0 / 94.5 / 94.8 & 100.0 / 93.5 / 94.8 & 94.0 / 95.0 / 94.8 \\
\hline formal_fallacies & PaLM 2-L & 84.0 / 64.0 / 68.4 & 78.0 / 59.5 / 63.2 & 68.0 / 63.0 / 64.0 & 66.0 / 59.0 / 60.4 \\
\hline geometric_shapes & PaLM 2-L & 76.0 / 57.0 / 60.8 & 42.0 / 33.0 / 34.8 & 42.0 / 32.0 / 34.0 & 34.0 / 33.0 / 33.2 \\
\hline hyperbaton & PaLM 2-L & 100.0 / 96.0 / 96.8 & 78.0 / 75.0 / 75.6 & 74.0 / 72.5 / 72.8 & 88.0 / 89.0 / 88.8 \\
\hline logical_deduction_seven_objects & PaLM 2-L & 74.0 / 57.0 / 60.4 & 46.0 / 37.0 / 38.8 & 34.0 / 30.5 / 31.2 & 46.0 / 45.5 / 45.6 \\
\hline movie_recommendation & PaLM 2-L & 92.0 / 90.5 / 90.8 & 62.0 / 52.5 / 54.4 & 52.0 / 48.0 / 48.8 & 80.0 / 83.0 / 82.4 \\
\hline multistep_arithmetic_two & PaLM 2-L & 72.0 / 55.5 / 58.8 & 42.0 / 46.0 / 45.2 & 60.0 / 50.5 / 52.4 & 4.0 / 3.5 / 3.6 \\
\hline navigate & PaLM 2-L & 92.0 / 75.0 / 78.4 & 68.0 / 62.0 / 63.2 & 70.0 / 64.0 / 65.2 & 38.0 / 37.5 / 37.6 \\
\hline object_counting & PaLM 2-L & 84.0 / 86.5 / 86.0 & 36.0 / 46.5 / 44.4 & 60.0 / 62.0 / 61.6 & 28.0 / 27.0 / 27.2 \\
\hline penguins_in_a_table & PaLM 2-L & 86.2 / 71.8 / 74.7 & 79.3 / 64.1 / 67.1 & 62.1 / 58.1 / 58.9 & 72.4 / 69.2 / 69.9 \\
\hline reasoning_about_colored_objects & PaLM 2-L & 98.0 / 85.5 / 88.0 & 82.0 / 79.5 / 80.0 & 82.0 / 75.0 / 76.4 & 42.0 / 35.0 / 36.4 \\
\hline ruin_names & PaLM 2-L & 88.0 / 88.0 / 88.0 & 70.0 / 55.0 / 58.0 & 80.0 / 75.5 / 76.4 & 88.0 / 76.5 / 78.8 \\
\hline salient_translation_error_detection & PaLM 2-L & 62.0 / 67.0 / 66.0 & 42.0 / 50.0 / 48.4 & 58.0 / 46.0 / 48.4 & 56.0 / 56.5 / 56.4 \\
\hline snarks & PaLM 2-L & 85.7 / 83.2 / 83.7 & 60.0 / 62.2 / 61.8 & 54.3 / 53.1 / 53.4 & 51.4 / 60.1 / 58.4 \\
\hline sports_understanding & PaLM 2-L & 98.0 / 88.0 / 90.0 & 50.0 / 46.5 / 47.2 & 60.0 / 52.5 / 54.0 & 52.0 / 41.5 / 43.6 \\
\hline temporal_sequences & PaLM 2-L & 100.0 / 100.0 / 100.0 & 100.0 / 96.0 / 96.8 & 90.0 / 87.0 / 87.6 & 100.0 / 99.5 / 99.6 \\
\hline tracking_shuffled_objects_seven_objects & PaLM 2-L & 32.0 / 16.5 / 19.6 & 58.0 / 61.5 / 60.8 & 54.0 / 55.5 / 55.2 & 14.0 / 23.5 / 21.6 \\
\hline web_of_lies & PaLM 2-L & 62.0 / 52.0 / 54.0 & 46.0 / 41.5 / 42.4 & 24.0 / 31.0 / 29.6 & 54.0 / 54.0 / 54.0 \\
\hline word_sorting & PaLM 2-L & 54.0 / 54.5 / 54.4 & 2.0 / 4.5 / 4.0 & 12.0 / 9.5 / 10.0 & 20.0 / 22.5 / 22.0 \\
\hline boolean_expressions & text-bison & 98.0 / 87.0 / 89.2 & 72.0 / 61.5 / 63.6 & 88.0 / 78.0 / 80.0 & 80.0 / 68.5 / 70.8 \\
\hline causal_judgement & text-bison & 78.4 / 58.0 / 62.0 & 70.3 / 50.7 / 54.5 & 73.0 / 55.3 / 58.8 & 78.4 / 58.0 / 62.0 \\
\hline date_understanding & text-bison & 60.0 / 50.0 / 52.0 & 44.0 / 45.5 / 45.2 & 48.0 / 45.0 / 45.6 & 44.0 / 45.0 / 44.8 \\
\hline disambiguation_qa & text-bison & 68.0 / 73.0 / 72.0 & 4.0 / 6.0 / 5.6 & 4.0 / 15.5 / 13.2 & 52.0 / 68.5 / 65.2 \\
\hline dyck_languages & text-bison & 100.0 / 100.0 / 100.0 & 100.0 / 95.5 / 96.4 & 100.0 / 94.5 / 95.6 & 100.0 / 98.5 / 98.8 \\
\hline formal_fallacies & text-bison & 70.0 / 53.0 / 56.4 & 64.0 / 54.5 / 56.4 & 84.0 / 82.5 / 82.8 & 70.0 / 54.5 / 57.6 \\
\hline geometric_shapes & text-bison & 40.0 / 19.5 / 23.6 & 22.0 / 13.0 / 14.8 & 18.0 / 12.0 / 13.2 & 20.0 / 14.5 / 15.6 \\
\hline hyperbaton & text-bison & 80.0 / 79.5 / 79.6 & 64.0 / 67.5 / 66.8 & 64.0 / 69.0 / 68.0 & 64.0 / 64.0 / 64.0 \\
\hline logical_deduction_seven_objects & text-bison & 66.0 / 53.5 / 56.0 & 56.0 / 58.0 / 57.6 & 56.0 / 56.0 / 56.0 & 58.0 / 56.5 / 56.8 \\
\hline movie_recommendation & text-bison & 98.0 / 90.0 / 91.6 & 68.0 / 63.0 / 64.0 & 66.0 / 62.0 / 62.8 & 68.0 / 64.0 / 64.8 \\
\hline multistep_arithmetic_two & text-bison & 32.0 / 16.5 / 19.6 & 12.0 / 18.0 / 16.8 & 18.0 / 17.5 / 17.6 & 16.0 / 18.5 / 18.0 \\
\hline navigate & text-bison & 72.0 / 61.0 / 63.2 & 56.0 / 55.0 / 55.2 & 60.0 / 56.5 / 57.2 & 56.0 / 57.0 / 56.8 \\
\hline object_counting & text-bison & 72.0 / 62.0 / 64.0 & 58.0 / 57.0 / 57.2 & 62.0 / 55.5 / 56.8 & 50.0 / 57.0 / 55.6 \\
\hline penguins_in_a_table & text-bison & 72.4 / 56.4 / 59.6 & 58.6 / 53.0 / 54.1 & 55.2 / 55.6 / 55.5 & 58.6 / 53.0 / 54.1 \\
\hline reasoning_about_colored_objects & text-bison & 82.0 / 77.0 / 78.0 & 76.0 / 72.5 / 73.2 & 78.0 / 73.0 / 74.0 & 74.0 / 69.5 / 70.4 \\
\hline ruin_names & text-bison & 88.0 / 82.5 / 83.6 & 66.0 / 65.5 / 65.6 & 66.0 / 62.5 / 63.2 & 64.0 / 66.0 / 65.6 \\
\hline salient_translation _error_detection & text-bison & 46.0 / 50.5 / 49.6 & 42.0 / 47.5 / 46.4 & 42.0 / 49.5 / 48.0 & 44.0 / 50.0 / 48.8 \\
\hline snarks & text-bison & 80.0 / 81.8 / 81.5 & 68.6 / 77.6 / 75.8 & 71.4 / 76.2 / 75.3 & 77.1 / 84.6 / 73.1 \\
\hline sports_understanding & text-bison & 94.0 / 82.5 / 84.8 & 86.0 / 79.0 / 80.4 & 90.0 / 81.0 / 82.8 & 38.0 / 44.5 / 43.2 \\
\hline temporal_sequences & text-bison & 78.0 / 81.0 / 80.4 & 36.0 / 43.5 / 42.0 & 32.0 / 45.0 / 42.4 & 36.0 / 43.0 / 41.6 \\
\hline tracking_shuffled_objects_seven_objects & text-bison & 32.0 / 15.5 / 18.8 & 10.0 / 17.0 / 15.6 & 10.0 / 18.0 / 16.4 & 12.0 / 15.5 / 14.8 \\
\hline web_of_lies & text-bison & 62.0 / 50.0 / 52.4 & 48.0 / 45.5 / 46.0 & 48.0 / 44.0 / 44.8 & 52.0 / 51.5 / 51.2 \\
\hline word_sorting & text-bison & 24.0 / 17.5 / 18.8 & 10.0 / 12.0 / 11.6 & 4.0 / 8.0 / 7.2 & 4.0 / 7.5 / 6.8 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 8: BBH task-wise instructions found by prompt optimization with the PaLM 2-L scorer and the PaLM 2-L-IT optimizer. The optimization starts from the empty string.**
\begin{tabular}[t]{|l|l|}
\hline Task & Our Instruction \\
\hline boolean_expressions & A Boolean expression is a well-formed expression consisting of variables, values, and logical operators. The expression must evaluate to a single True or False value. The order of precedence of the logical operators is as follows: NOT, AND, OR, XOR, IMP. Parentheses can be used to group subexpressions and to control the order of evaluation. \\
\hline causal_judgement & When considering questions about causation, a typical person would consider the following factors: whether the action or event was a necessary condition for the outcome to occur, a sufficient condition, a proximate cause, or a foreseeable cause. \\
\hline date_understanding & To find the date X time ago from today, first find today's date. Then subtract X time from today's date. If the current date is the last day of a month, then the date a month ago is the last day of the previous month. If the current date is not the last day of a month, then the date a month ago is the same day of the previous month. For example, if today is March 31, 2023, then the date a month ago is February 28, 2023. If today is April 1, 2023, then the date a month ago is March 1, 2023. \\
\hline disambiguation_qa & Identifying Antecedents of Pronouns: A Comprehensive Guide \\
\hline dyck_languages & First, look for the opening parentheses. Then, count the number of opening parentheses. Finally, close the parentheses in the reverse order that they were opened. \\
\hline formal_fallacies & A deductive argument is one where the conclusion follows necessarily from the premises. If the premises are true, then the conclusion must also be true. An invalid argument is one where it is possible for the premises to be true and the conclusion to be false. \\
\hline geometric_shapes & A closed polygonal chain is a series of connected line segments. The line segments can be straight or curved. The first and last line segments are connected. The line segments do not intersect each other except at their endpoints. A closed polygon can be described by an SVG path element, which starts at a given point, goes to one or more additional points, and then ends at the starting point. The path element can consist of straight line segments, curved segments, or a mixture of both. \\
\hline hyperbaton & The correct adjective order in English is opinion, size, shape, age, color, origin, material, and purpose. If you have more than one adjective of the same type, they are usually placed in order of importance. For example, you would say "a large, old, Pakistani ship" rather than "an old, large, Pakistani ship." There are a few exceptions to these rules, but they are generally followed in most cases. \\
\hline logical_deduction _seven_objects & The following questions will test your ability to use deductive reasoning. You will be given a set of statements about a group of objects. You will then be asked to answer questions about the objects based on the statements. The statements in the questions are logically consistent, so you can use them to deduce the order of the objects. For each question, you must choose the option that is logically consistent with the information in the questions. \\
\hline movie_recommendation & Based on your input, I have analyzed the given movies in terms of genre, plot, tone, audience rating, year of release, director, cast, and reviews. I have also taken into account the given options. The movie that is most similar to the given movies in terms of all these factors is: \\
\hline multistep_arithmetic _two & The order of operations in mathematics is PEMDAS, which stands for Parentheses, Exponents, Multiplication, Division, Addition, and Subtraction. When there are multiple operations of the same precedence, they must be performed from left to right. Note that multiplication and division have the same precedence, as do addition and subtraction. \\
\hline navigate & You will return to the starting point if and only if (1) the total number of steps you take forward is equal to the total number of steps you take back, and (2) the total number of turns you make is a multiple of 180 degrees. \\
\hline object_counting & Here is a list of the objects you mentioned and their corresponding counts: \\
\hline penguins_in_a_table & Here is my new text: \\
\hline reasoning_about _colored_objects & Starting from the leftmost object in the row, I observe the following objects arranged in this order: \\
\hline ruin_names & Which is the funniest pun on the artist or movie name? \\
\hline salient_translation _error_detection & Instructions: Read the German sentence and its English translation carefully, then identify the type of error in the translation and select the correct option. There are six possible types of errors: Named Entities, Numerical Values, Modifiers or Adjectives, Negation or Antonyms, Facts, and Dropped Content. \\
\hline snarks & Identify the sarcastic statement by considering the following factors: incongruity, exaggeration, understatement, context, speaker's intent, and audience's reaction. I will also consider the speaker's tone of voice, facial expressions, and body language. \\
\hline sports_understanding & I will determine if a sentence about an athlete is plausible by first checking if it is grammatically correct. If it is, I will then check if it is consistent with the athlete's sport, position, and real-world statistics. I will also check if it is consistent with the rules of the athlete's sport. If the sentence is consistent with all of these things, I will answer "yes", otherwise I will answer "no". \\
\hline temporal_sequences & The answer is the time that is not mentioned in the given statements. \\
\hline tracking_shuffled_objects _seven_objects & Claire has the blue ball, Gertrude has the black ball, and Dave has the green ball. They are all happy with their new \\
\hline web_of_lies & The answer to a question is yes if there are an odd number of liars before the current speaker, and no if there are an even number of liars before the current speaker. If the current speaker is a truth-teller, they will say the opposite of what the previous person said, while a liar will say the same thing as the previous person said. \\
\hline word_sorting & Alphabetical order of given words: \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 9: BBH task-wise instructions found by prompt optimization with the text-bison scorer and the PaLM 2-L-IT optimizer. The optimization starts from the empty string.**
\begin{tabular}[t]{|l|l|}
\hline Task & Our Instruction \\
\hline boolean_expressions & Not (not False) and not not False is False \\
\hline causal_judgement & A typical person would likely answer the questions about causation as follows: \\
\hline date_understanding & Today is February 28, 2023. It is a Tuesday. Yesterday was Monday, February 27, 2023. Tomorrow will be Wednesday, March 1, 2023. A week ago, it was February 21, 2023, and a month ago, it was January 28, 2023. A year from now, it will be February 28, 2024. The day of the week is important to note because it will help us to correctly answer the questions below. Not all years are leap years that contain February 29. \\
\hline disambiguation_qa & A pronoun is a word that stands in for a noun. The noun that a pronoun refers to is called its antecedent. To identify the antecedent of a pronoun, look for the noun that the pronoun could be referring to. If there is only one possible noun, then that is the antecedent. If there are two or more possible nouns, then the antecedent is ambiguous. Use the context of the sentence to help you determine the correct antecedent. \\
\hline dyck_languages & \{ \} \\
\hline formal_fallacies & How to Evaluate Deductive Validity of an Argument \\
\hline geometric_shapes & What shape is this SVG code drawing, and how many sides does it have? \\
\hline hyperbaton & In English, adjectives are typically placed before nouns in a specific order. The order is: opinion, size, shape, age, color, origin, material, purpose, noun. For example, the sentence "the big, old, red barn" would be considered grammatically correct, while the sentence "the old, big, red barn" would not. Adjectives that come before nouns are called attributive adjectives, while adjectives that come after nouns are called predicative adjectives. \\
\hline logical_deduction _seven_objects & In this logical reasoning task, you will be given a series of paragraphs, each of which describes a set of objects arranged in a fixed order. The statements in each paragraph are logically consistent. You must read each paragraph carefully and use the information given to determine the logical relationships between the objects. You will then be asked a question about the order of the objects. Read each question carefully and choose the option that answers the question correctly. \\
\hline movie_recommendation & What is the highest-rated movie similar to the given movies, with a similar IMDb rating and released in the same year? \\
\hline multistep_arithmetic_two & Let's solve these equations using PEMDAS order of operations. Remember that PEMDAS stands for parentheses, exponents, multiplication and division, and addition and subtraction. \\
\hline navigate & Starting at the origin, facing north, follow the instructions. If your displacement from the origin is zero and your direction is unchanged, then your answer is Yes. Otherwise, your answer is No. \\
\hline object_counting & Let me help you count the items you have. Just list them one by one, separated by commas. I will then count each item and tell you how many items there are in total. \\
\hline penguins_in_a_table & This table shows information about penguins. The columns show the penguin's name, age, height (in cm), and weight (in kg). The penguins are listed in order of their age, from youngest to oldest. \\
\hline reasoning_about _colored_objects & First, read the input carefully. Then, identify all the objects mentioned, their colors, and their positions. Next, visualize the objects and their positions in your mind. Finally, answer the questions accurately based on the information given. Make sure to pay attention to the order of the objects. \\
\hline ruin_names & A humorous edit of an artist or movie name can be created by replacing one or more letters to form a new word or phrase that sounds similar but has a different meaning. The new word or phrase should be relevant to the original word, but it should also be a surprise, which makes the edit funny. For example, the artist or movie name "Rocky" can be changed to "Ricky," and "Schindler's List" can be changed to "Schindler's Lift." Be creative and have fun! \\
\hline salient_translation _error_detection & The following translations from German to English contain a particular error. The error may be one of the following types: Named Entities, Numerical Values, Modifiers or Adjectives, Negation or Antonyms, Facts, or Dropped Content. Please identify the error. \\
\hline snarks & The statement \\
\hline sports_understanding & To determine the plausibility of a sports sentence, I will first identify the sport, athletes, teams, and events mentioned in the sentence. Then, I will use my knowledge of the rules of the sport, the context of the sentence, common sense, and my knowledge of the world to determine whether the sentence is plausible. I will also consider the time period and location, as well as any other relevant information. Finally, I will return a score of 1 for plausible sentences and 0 for implausible ones. \\
\hline temporal_sequences & To determine the time period when a person went to a place, first identify all the time periods when the person's whereabouts are unknown. Then, rule out any time periods during which the person was seen doing something else or the place was closed. The remaining time periods are the possible times when the person could have gone to the place. \\
\hline tracking_shuffled_objects _seven_objects & At the start of the game, Claire has a blue ball. Throughout the game, pairs of people swap balls. Claire ends up with the yellow ball. \\
\hline web_of_lies & People in a group either tell the truth or lie. The truthfulness of a person's statement is determined by the statement of the previous person. If the previous person told the truth, then the current person who says the opposite is lying. If the previous person lied, then the current person who says the opposite is telling the truth. This rule applies to all subsequent statements. \\
\hline word_sorting & Sort the following words alphabetically, ignoring case and punctuation. Print the sorted list. \\
\hline
\end{tabular}
\end{table}

# E. 2 GPT-3.5-turbo AS OPTIMIZER, OPTIMIZATION STARTING FROM THE EMPTY STRING

Table 11,12 and 13 show the instructions found by prompt optimization. Their accuracies are listed in Table 10. Figure 25 visualizes the difference between their accuracies and those of the baselines "Let's think step by step." and the empty string. The optimizations find instructions better than the empty starting point, and most of the found instructions are better than "Let's think step by step".

One caveat in the A_begin instructions (Table 11) is that a lot of the found instructions are imperative or interrogative sentences that are more suitable to be put into "Q:" rather than "A:", like "Solve the sequence by properly closing the parentheses." for dyck_languages and "Which movie option from the given choices ...?" for movie_recommendation. Such styles appear more often here than the PaLM 2-L-IT optimizer results (Table 8), showing PaLM 2-L-IT understands the needed style better. In Section E.3, we show the A_begin optimization results with the non-empty starting point "Let's solve the problem.". Most results there are declarative sentences - more suitable for A_begin.

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-35_1529_1238_855_441.jpg)

**Figure 25: On 23 BBH tasks, the accuracy differences among instructions found by prompt optimization (with the gpt-3.5-turbo optimizer), "Let's think step by step.", and the empty string (optimization starting point).**

\begin{table}

**Table 10: Accuracies on BBH tasks with the gpt-3.5-turbo optimizer that starts from the empty string. The PaLM 2-L scores are from A_begin (left) instructions; the text-bison scores include Q_begin (left) and Q_end (right) instructions.**
\begin{tabular}[t]{|l|l|l|l|}
\hline \multirow[t]{2}{*}{Task} & \multirow[t]{2}{*}{Scorer} & Our Acc (begin) & Our Acc (end) \\
\hline & & training / test / overall & training / test / overall \\
\hline boolean_expressions & PaLM 2-L & 92.0 / 86.5 / 87.6 & N/A \\
\hline causal_judgement & PaLM 2-L & 81.1 / 58.7 / 63.1 & N/A \\
\hline date_understanding & PaLM 2-L & 86.0 / 82.0 / 82.8 & N/A \\
\hline disambiguation_qa & PaLM 2-L & 80.0 / 74.0 / 75.2 & N/A \\
\hline dyck_languages & PaLM 2-L & 100.0 / 100.0 / 100.0 & N/A \\
\hline formal_fallacies & PaLM 2-L & 88.0 / 63.5 / 68.4 & N/A \\
\hline geometric_shapes & PaLM 2-L & 60.0 / 41.0 / 44.8 & N/A \\
\hline hyperbaton & PaLM 2-L & 88.0 / 93.0 / 92.0 & N/A \\
\hline logical_deduction_seven_objects & PaLM 2-L & 76.0 / 56.5 / 60.4 & N/A \\
\hline movie_recommendation & PaLM 2-L & 84.0 / 86.0 / 85.6 & N/A \\
\hline multistep_arithmetic_two & PaLM 2-L & 52.0 / 49.0 / 49.6 & N/A \\
\hline navigate & PaLM 2-L & 76.0 / 67.0 / 68.8 & N/A \\
\hline object_counting & PaLM 2-L & 78.0 / 79.0 / 78.8 & N/A \\
\hline penguins_in_a_table & PaLM 2-L & 82.8 / 72.6 / 74.7 & N/A \\
\hline reasoning_about _colored_objects & PaLM 2-L & 86.0 / 67.5 / 71.2 & N/A \\
\hline ruin_names & PaLM 2-L & 90.0 / 83.0 / 84.4 & N/A \\
\hline salient_translation_error_detection & PaLM 2-L & 62.0 / 65.0 / 64.4 & N/A \\
\hline snarks & PaLM 2-L & 85.7 / 70.6 / 73.6 & N/A \\
\hline sports_understanding & PaLM 2-L & 68.0 / 57.5 / 59.6 & N/A \\
\hline temporal_sequences & PaLM 2-L & 100.0 / 99.5 / 99.6 & N/A \\
\hline tracking_shuffled_objects_seven_objects & PaLM 2-L & 44.0 / 34.5 / 36.4 & N/A \\
\hline web_of_lies & PaLM 2-L & 92.0 / 91.0 / 91.2 & N/A \\
\hline word_sorting & PaLM 2-L & 62.0 / 52.0 / 54.0 & N/A \\
\hline boolean_expressions & text-bison & 84.0 / 78.5 / 79.6 & 80.0 / 78.0 / 78.4 \\
\hline causal_judgement & text-bison & 78.4 / 57.3 / 61.5 & 83.8 / 53.3 / 59.4 \\
\hline date_understanding & text-bison & 52.0 / 45.0 / 46.4 & 64.0 / 52.4 / 54.8 \\
\hline disambiguation_qa & text-bison & 68.0 / 75.5 / 74.0 & 64.0 / 71.5 / 70.0 \\
\hline dyck_languages & text-bison & 100.0 / 99.5 / 99.6 & 100.0 / 100.0 / 100.0 \\
\hline formal_fallacies & text-bison & 70.0 / 54.5 / 57.6 & 74.0 / 53.5 / 57.6 \\
\hline geometric_shapes & text-bison & 28.0 / 15.0 / 17.6 & 48.0 / 28.0 / 32.0 \\
\hline hyperbaton & text-bison & 86.0 / 85.0 / 85.2 & 80.0 / 76.5 / 77.2 \\
\hline logical_deduction_seven_objects & text-bison & 66.0 / 57.5 / 59.2 & 62.0 / 55.0 / 56.4 \\
\hline movie_recommendation & text-bison & 76.0 / 69.5 / 70.8 & 82.0 / 70.5 / 72.8 \\
\hline multistep_arithmetic_two & text-bison & 28.0 / 20.5 / 22.0 & 28.0 / 22.5 / 23.6 \\
\hline navigate & text-bison & 72.0 / 61.0 / 63.2 & 68.0 / 59.5 / 61.2 \\
\hline object_counting & text-bison & 68.0 / 71.0 / 70.4 & 72.0 / 69.0 / 69.6 \\
\hline penguins_in_a_table & text-bison & 65.5 / 59.8 / 61.0 & 79.3 / 53.0 / 58.2 \\
\hline reasoning_about_colored_objects & text-bison & 84.0 / 76.5 / 78.0 & 86.0 / 74.0 / 76.4 \\
\hline ruin_names & text-bison & 80.0 / 74.0 / 75.2 & 74.0 / 75.0 / 74.8 \\
\hline salient_translation_error_detection & text-bison & 44.0 / 50.5 / 49.2 & 48.0 / 51.0 / 50.4 \\
\hline snarks & text-bison & 82.9 / 79.7 / 80.3 & 88.6 / 84.6 / 85.4 \\
\hline sports_understanding & text-bison & 84.0 / 76.5 / 78.0 & 90.0 / 80.0 / 82.0 \\
\hline temporal_sequences & text-bison & 50.0 / 54.5 / 53.6 & 64.0 / 61.5 / 62.0 \\
\hline tracking_shuffled_objects_seven_objects & text-bison & 22.0 / 18.5 / 19.2 & 30.0 / 21.5 / 23.2 \\
\hline web_of_lies & text-bison & 64.0 / 57.5 / 58.8 & 68.0 / 55.0 / 57.6 \\
\hline word_sorting & text-bison & 26.0 / 19.0 / 20.4 & 32.0 / 25.5 / 26.8 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 11: BBH task-wise instructions found by prompt optimization with the PaLM 2-L scorer and the gpt-3.5-turbo optimizer. The optimizations start from the empty string.**
\begin{tabular}[t]{|l|l|}
\hline Task & Our Instruction \\
\hline boolean_expressions & An accurate evaluation of logical expressions involves correctly applying Boolean operators, considering the order of operations, and analyzing the truth values of the operands in accordance with Boolean logic principles. \\
\hline causal_judgement & Understanding causality is critical for accurately assessing cause and effect relationships in various scenarios, leading to well-informed judgments, precise conclusions, and definitive answers to questions about the outcomes involved. \\
\hline date_understanding & What is the specific date mentioned or required in each given problem or question, taking into account all relevant information, available options, and the provided context? Please provide the accurate answer in the format MM/DD/YYYY. \\
\hline disambiguation_qa & Accurately analyze and clarify the pronoun-antecedent relationship in the given sentences, identifying the appropriate referent to eliminate any potential confusion or ambiguity and ensure a precise understanding of the intended meaning. \\
\hline dyck_languages & Solve the sequence by properly closing the parentheses. \\
\hline formal_fallacies & In determining the deductive validity of arguments based on explicit premises, a meticulous analysis of the logical relationships and implications is essential for definitively establishing their soundness, confirming their validity or invalidity, and ensuring a reliable and robust assessment of the arguments at hand. \\
\hline geometric_shapes & The SVG path element with the "d" attribute plays a crucial role in web development, allowing for the precise definition and rendering of various shapes on a webpage. \\
\hline hyperbaton & Understanding the correct order of adjectives is crucial for constructing grammatically accurate and coherent sentences that effectively convey the intended meaning in diverse contexts while ensuring clarity, cohesion, and consistency throughout consistently and effortlessly. \\
\hline logical_deduction _seven_objects & By conducting a meticulous analysis of the given information and ensuring logical consistency within each paragraph, we can accurately determine the precise order or ranking of the mentioned objects, allowing us to confidently and consistently identify the correct answer in every presented scenario with utmost precision and confidence. \\
\hline movie_recommendation & Which movie option from the given choices closely matches the mentioned films in terms of themes, storylines, and characteristics, guaranteeing the highest possible similarity score among them all? \\
\hline multistep_arithmetic_two & Evaluate the given mathematical expressions step by step to determine the correct solutions accurately. \\
\hline navigate & Is it possible to determine, with absolute certainty, whether strictly adhering to the given instructions will unfailingly bring you back to the original starting point without any exceptions, errors, or deviations? \\
\hline object_counting & Determine the total number of objects or entities mentioned in the given list, covering various categories and types, to accurately calculate the overall count. \\
\hline penguins_in_a_table & From the given table, what information can we gather about the mentioned animals and their respective attributes, including names, ages, heights, and weights? \\
\hline reasoning_about _colored_objects & By thoroughly examining the given information, accurately determine the answers for each question by considering the specific characteristics, colors, and positions of the mentioned objects. \\
\hline ruin_names & Select the most amusing and clever alteration from the options provided for the given artist, movie, or title name, and accurately choose the correct answer to test your wit and creativity. \\
\hline salient_translation _error_detection & Thoroughly examine the given translations from German to English and accurately identify any errors by carefully analyzing the text and selecting the appropriate option with meticulous attention to detail, precision, utmost accuracy, and comprehensive understanding of the language for precise evaluation and categorization. \\
\hline snarks & Which option delivers the most devastatingly sarcastic response, brilliantly exposing the sheer absurdity and leaving absolutely no doubt whatsoever in all the given situations? \\
\hline sports_understanding & Maintaining the accuracy, reliability, and integrity of sports event representation is essential for upholding the highest standards of credibility, trustworthiness, and overall quality in conveying information, without any compromise, misrepresentation, or distortion, thereby ensuring the factual accuracy of sports journalism. \\
\hline temporal_sequences & Based on the provided timeline and observed activities, we can accurately determine the possible time range when each individual could have visited their intended destinations and answer questions about their visitation time. \\
\hline tracking_shuffled_objects _seven_objects & An important point to note is that each person in the group starts with one specific book at the beginning of the semester. \\
\hline web_of_lies & Analyzing the consistency and accuracy of statements provided by each person is crucial for determining the truthfulness of individuals in every scenario. \\
\hline word_sorting & Please sort the given words in alphabetical order: The list of words to be sorted contains - \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 12: BBH task-wise Q_begin instructions found by prompt optimization with the text-bison scorer and the gpt-3.5-turbo optimizer. The optimizations start from the empty string.**
\begin{tabular}[t]{|l|l|}
\hline Task & Our Instruction \\
\hline boolean_expressions & Group sub-expressions with parentheses to accurately evaluate logical operations: not, and, and finally or. Determine the resulting value as either True or False. \\
\hline causal_judgement & Consider the intentions and actions of the individuals involved. \\
\hline date_understanding & Determine the one-day difference in the given date and express it in the format MM/DD/YYYY. \\
\hline disambiguation_qa & Determine the precise antecedent of the pronoun in the given sentence and select the correct option or state if it is ambiguous. \\
\hline dyck_languages & Ensure that all opening brackets have a corresponding closing bracket, and that the closing brackets are in the correct order. \\
\hline formal_fallacies & Thoroughly analyze the explicitly provided premises and determine the deductive validity of the argument based on all necessary conditions, implications, exclusions, and dependencies given. \\
\hline geometric_shapes & Analyze the given SVG path element carefully and confidently select the correct option from the provided choices to accurately determine the corresponding shape. Pay close attention to the specific path details and confidently make the most suitable choice. \\
\hline hyperbaton & Select the sentence that strictly adheres to the standard order of adjectives: opinion, size, age, shape, color, origin, material, and purpose. Ensure there are no deviations or alterations in the adjective order. Choose the option without any changes. \\
\hline logical_deduction _seven_objects & Analyze the given information to accurately determine the precise order and ranking of the mentioned objects/people, considering their relationships, positions, and any provided comparisons, for a definitive and logical progression with maximum accuracy and efficiency. \\
\hline movie_recommendation & Based on the movie list provided, carefully consider your preferences and make a well-informed decision. \\
\hline multistep_arithmetic_two & First, simplify any expressions within parentheses following the correct order of operations to accurately evaluate the final answer with efficiency and precision. \\
\hline navigate & Always face forward. Take 10 steps forward. Turn left. Take 5 steps forward. Take 3 steps backward. Finally, take 7 steps forward. Turn around and take 1 step forward. Repeat the previous sequence three times. Follow the given path precisely without any deviations. At the end, turn right and take 11 steps forward. If you follow these instructions, will you return to the starting point? Options: - Yes - No \\
\hline object_counting & Determine the total count of mentioned vegetables accurately and state the final count as the answer. \\
\hline penguins_in_a_table & Analyze the given table to accurately determine the required information based on the provided criteria and attributes of the penguins and giraffes. Utilize efficient problem-solving strategies to arrive at the correct answer. \\
\hline reasoning_about _colored_objects ruin_names & \begin{tabular}[t]{l}
State the color of the object mentioned in the given arrangement with utmost accuracy. \\
Choose the option that offers the most clever and humorous alteration of the given artist or movie name. Let your creativity shine and select the answer that will undoubtedly bring a smile to your face! Make sure to think outside the box!
\end{tabular} \\
\hline salient_translation _error_detection & Analyze the translation and accurately identify the specific error type based on the source text, providing the most appropriate corresponding option. \\
\hline snarks & Choose the option that wickedly embodies sarcasm. \\
\hline sports_understanding & Determine the plausibility of the given statement by evaluating factual accuracy, logical consistency, and contextual relevance, then provide a succinct and well-justified response. \\
\hline temporal_sequences & Identify the optimal time slot for the individual to engage in the mentioned location/activity considering the given sightings and waking up time, taking into account the opening and closing times of the location and the duration of each event. \\
\hline tracking_shuffled_objects _seven_objects & Pay attention to the given information and track the swaps/exchanges carefully to accurately determine the final possession/position/outcome for the specified individual. \\
\hline web_of_lies & To determine the truthfulness of the last person mentioned, analyze the consistency of each statement and count the number of individuals accusing the previous person of lying. If the count of accusers is even, that person tells the truth; if it is odd, that person lies. \\
\hline word_sorting & Alphabetically sort the given list of words, ensuring all words are included and in ascending order. \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 13: BBH task-wise Q_end instructions found by prompt optimization with the text-bison scorer and the gpt-3.5-turbo optimizer. The optimizations start from the empty string.**
\begin{tabular}[t]{|l|l|}
\hline Task & Our Instruction \\
\hline boolean_expressions & Accurately use order of operations and parentheses to evaluate logical expressions and determine truth values efficiently. \\
\hline causal_judgement & Consider all relevant factors, prioritize overall well-being and ethical considerations, make well-informed decisions while foreseeing potential consequences efficiently, and consistently strive for optimal outcomes with empathy and adaptability in a thoughtful and comprehensive manner. \\
\hline date_understanding & Subtract the specified number of days from the given date and format the outcome as MM/DD/YYYY to accurately determine the desired result in an efficient manner. \\
\hline disambiguation_qa & Clearly identify and select the unambiguous antecedent for the pronoun or designate it as "Ambiguous" if it is unclear. \\
\hline dyck_languages & Add the missing closing parentheses. \\
\hline formal_fallacies & Determine the deductive validity of the argument presented based on the explicitly stated premises and reach a definitive conclusion. \\
\hline geometric_shapes & Analyzing the given SVG path element, accurately determine its shape by closely examining its curves and coordinates, then select the correct option. \\
\hline hyperbaton & Choose the option with the correct adjective order in each sentence, prioritizing specific attributes like size, color, and origin. Place the most specific adjective before the more general ones for precise and standardized ordering across all examples. Ensure accurate alignment of the adjectives based on their respective attributes for consistent and standardized ordering. \\
\hline logical_deduction _seven_objects & Determine the precise order of the given objects/participants based on the provided information and establish the final ranking accurately, considering all relevant factors, while maintaining logical consistency with maximum efficiency. \\
\hline movie_recommendation & Choose the most similar option from the choices provided that closely aligns with the given movies' themes, genres, and impact for the most accurate recommendation possible. Make your selection wisely. \\
\hline multistep_arithmetic_two & Carefully follow the order of operations to precisely simplify the expressions within parentheses and efficiently find the accurate final answer. \\
\hline navigate & Always face forward. Take 10 steps forward. Turn right and walk for 5 steps. Then, make a left turn and continue for 9 steps. Proceed by walking 6 steps backward. Finally, turn around and take 200 steps. Accurately track your movements, diligently adhere to the given path, and ensure to return to the starting point without any deviations or obstacles. \\
\hline object_counting & Determine the total count of items mentioned, including all listed items, using an efficient and concise method. State the final count as your answer. \\
\hline penguins_in_a_table & Identify the animal with the maximum measurement (weight, age, or height) in the table and state its name and species. \\
\hline reasoning_about _colored_objects & Determine the color of each item in the given scenario and select the correct color option from the provided choices for accurate responses, ensuring utmost precision and completeness. \\
\hline ruin_names & Choose the option that creatively and hilariously transforms the given artist or movie name. \\
\hline salient_translation _error_detection & Carefully analyze the translations and select the most suitable option from the given choices to rectify the specific error category, ensuring complete precision, accuracy, and faithful representation of the intended meaning, while considering all relevant information in the source text. \\
\hline snarks & Choose the option that cleverly employs sarcasm to defy all expectations and leave everyone utterly dumbfounded, questioning the very essence of their own perception. \\
\hline sports_understanding & Evaluate the plausibility of each given statement and provide a well-supported justification based on logical reasoning, contextual understanding, and relevant evidence to arrive at a definitive and conclusive answer. \\
\hline temporal_sequences & Identify the possible time slot for the desired activity based on the given information and sightings, then select the correct option. \\
\hline tracking_shuffled_objects _seven_objects & Thoroughly analyze the given scenarios, systematically consider all available information, and confidently determine the final outcome with exceptional precision and optimal efficiency, while maintaining a strategic and logical approach throughout the process. \\
\hline web_of_lies & Examine each person's statements meticulously to accurately determine the truth and confidently identify who is telling the truth, enabling you to effectively solve the given problem. \\
\hline word_sorting & Sort the given words alphabetically using spaces as separators while maintaining their original order and including all words. \\
\hline
\end{tabular}
\end{table}

# E. 3 PALM 2-L AS SCORER, GPT-3.5-turbo AS OPTIMIZER, OPTIMIZATION STARTING from "Let's solve the problem."

Figure 26 and Table 14 compare the accuracies of found instructions vs "Let's solve the problem.", "Let's think step by step.", and the instructions in Table 11. Table 15 details the found instructions.

The "Let's" pattern appears more often in the found instructions because of the starting points, and the instructions are more often declarative that are more suitable for A_begin, even if some are semantically far from "Let's solve the problem". In fact, "Let's" was adopted by Zhou et al. (2022b) as a fixed pattern in generated prompts, possibly because of the same reason.

![Figure](./yangLargeLanguageModels2024_images/b795b684-5f37-4eea-bfba-a5cecf51b52e-40_1593_1249_715_441.jpg)

**Figure 26: On 23 BBH tasks, the accuracy differences among instructions found by prompt optimization (with the text-bison scorer and the gpt-3.5-turbo optimizer), "Let's think step by step.", and "Let's solve the problem." (optimization starting point). The found instructions mostly outperform the "Let's think step by step." baseline, the "Let's solve the problem." starting point, and the instructions in Table 11 found by prompt optimization from the empty string.**

\begin{table}

**Table 14: Accuracies on BBH tasks with the PaLM 2-L scorer and the gpt-3.5-turbo optimizer that starts from "Let's solve the problem". The scores are from A_begin instructions.**
\begin{tabular}[t]{|l|l|l|l|}
\hline \multirow[t]{2}{*}{Task} & \multirow[t]{2}{*}{Scorer} & Our Acc & "Let's solve the problem." Acc \\
\hline & & training / test / overall & training / test / overall \\
\hline boolean_expressions & PaLM 2-L & 98.0 / 89.5 / 91.2 & 78.0 / 69.0 / 70.8 \\
\hline causal_judgement & PaLM 2-L & 83.8 / 58.7 / 63.6 & 62.0 / 61.3 / 61.5 \\
\hline date_understanding & PaLM 2-L & 90.0 / 82.0 / 83.6 & 74.0 / 71.0 / 71.6 \\
\hline disambiguation_qa & PaLM 2-L & 78.0 / 68.0 / 70.0 & 52.0 / 54.5 / 54.0 \\
\hline dyck_languages & PaLM 2-L & 100.0 / 100.0 / 100.0 & 94.0 / 97.0 / 96.4 \\
\hline formal_fallacies & PaLM 2-L & 84.0 / 62.0 / 66.4 & 68.0 / 54.0 / 56.8 \\
\hline geometric_shapes & PaLM 2-L & 62.0 / 42.5 / 46.4 & 30.0 / 22.0 / 23.6 \\
\hline hyperbaton & PaLM 2-L & 94.0 / 91.5 / 92.0 & 72.0 / 77.0 / 76.0 \\
\hline logical_deduction_seven_objects & PaLM 2-L & 66.0 / 53.0 / 55.6 & 38.0 / 36.5 / 36.8 \\
\hline movie_recommendation & PaLM 2-L & 88.0 / 88.0 / 88.0 & 66.0 / 76.0 / 74.0 \\
\hline multistep_arithmetic_two & PaLM 2-L & 66.0 / 55.0 / 57.2 & 30.0 / 22.0 / 23.6 \\
\hline navigate & PaLM 2-L & 76.0 / 67.0 / 68.8 & 54.0 / 63.5 / 61.6 \\
\hline object_counting & PaLM 2-L & 96.0 / 92.5 / 93.2 & 58.0 / 58.0 / 58.0 \\
\hline penguins_in_a_table & PaLM 2-L & 86.2 / 70.9 / 74.0 & 69.0 / 72.6 / 71.9 \\
\hline reasoning_about _colored_objects & PaLM 2-L & 88.0 / 69.0 / 72.8 & 78.0 / 69.5 / 71.2 \\
\hline ruin_names & PaLM 2-L & 92.0 / 85.5 / 86.8 & 76.0 / 79.5 / 80.8 \\
\hline salient_translation_error_detection & PaLM 2-L & 66.0 / 67.5 / 67.2 & 30.0 / 35.5 / 34.4 \\
\hline snarks & PaLM 2-L & 88.6 / 76.9 / 79.2 & 80.0 / 70.6 / 72.5 \\
\hline sports_understanding & PaLM 2-L & 72.0 / 63.5 / 65.2 & 60.0 / 50.5 / 52.4 \\
\hline temporal_sequences & PaLM 2-L & 100.0 / 99.5 / 99.6 & 96.0 / 92.5 / 93.2 \\
\hline tracking_shuffled_objects_seven_objects & PaLM 2-L & 56.0 / 63.5 / 62.0 & 42.0 / 51.5 / 49.6 \\
\hline web_of_lies & PaLM 2-L & 56.0 / 58.5 / 58.0 & 0.0 / 4.0 / 3.2 \\
\hline word_sorting & PaLM 2-L & 52.0 / 44.5 / 46.0 & 18.0 / 20.5 / 20.0 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 15: BBH task-wise Q_begin instructions found by prompt optimization with the PaLM 2-L scorer and the gpt-3.5-turbo optimizer. The optimizations start from "Let's solve the problem".**
\begin{tabular}[t]{|l|l|}
\hline Task & Our Instruction \\
\hline boolean_expressions & Let's accurately assess the given conditions and determine their corresponding Boolean values. \\
\hline causal_judgement & Let's conduct a meticulous evaluation of the given scenarios, accurately determine the causal relationships, and provide definitive answers through comprehensive analysis, ensuring a precise understanding of causation and a thorough determination of events in each situation. \\
\hline date_understanding & Let's accurately determine the correct date based on the given information and select the corresponding option in the standard MM/DD/YYYY format with utmost precision and reliability, ensuring the most definitive and reliable solution possible for accurate representation in all scenarios without any room for ambiguity, error, or confusion, and providing the highest level of accuracy and reliability. \\
\hline disambiguation_qa & Let's thoroughly analyze the given sentences to accurately determine the unambiguous antecedents of the pronouns used, ensuring clear understanding, effective communication, and leaving no room for any confusion or ambiguity. \\
\hline dyck_languages & Let's find the correct closing parentheses and brackets for the given sequences. \\
\hline formal_fallacies & Let's thoroughly analyze the explicitly stated premises and draw definitive conclusions to accurately determine the deductive validity of the arguments provided in each question, employing precise and logical reasoning in our assessments for unwavering confidence in our determinations. \\
\hline geometric_shapes & Let's accurately determine the shape represented by the given SVG path element by carefully analyzing its path data and considering all available options for a precise identification. \\
\hline hyperbaton & Let's quickly identify the correct adjective order. \\
\hline logical_deduction _seven_objects & Let's methodically analyze the given information, employ logical reasoning, thoroughly evaluate all relevant details, and accurately determine the solutions for each problem by considering all provided options comprehensively and strategically, ensuring an efficient and effective approach towards arriving at the correct answers. \\
\hline movie_recommendation & Let's uncover the perfect movie recommendation from the options provided, ensuring an exceptional cinematic experience together as we select the most captivating and satisfying choice that will keep us thoroughly engaged and immersed until the very end. \\
\hline multistep_arithmetic_two & Let's tackle the following calculations. \\
\hline navigate & Let's accurately and efficiently determine the correct solution for each given scenario, ensuring the highest level of precision, reliability, and consistency throughout. \\
\hline object_counting & Let's determine the total count of various items/objects/ingredients/animals mentioned in order to accurately and \\
\hline penguins_in_a_table & Let's analyze the given information and determine the correct answer. \\
\hline reasoning_about _colored_objects & Let's systematically analyze the given information and carefully evaluate each answer choice to confidently determine the accurate and optimal solutions, considering all available options and specific details provided in each question for precise and concise responses, ensuring complete accuracy and clarity in our answers. \\
\hline ruin_names & Prepare to have a side-splittingly funny time as we uncover the most clever and hilarious alternatives for these artist or movie names, challenging your wit to guess the correct one with a burst of creativity, humor, and imaginative twists! \\
\hline salient_translation _error_detection & Let's meticulously analyze the provided translations, accurately identifying any errors or discrepancies, and conduct a comprehensive evaluation to ensure the highest level of translation quality and fidelity. By considering contextual nuances, cultural references, linguistic conventions, potential factual errors, and any dropped content, our ultimate aim is to achieve precise and thorough assessments for optimal translation accuracy and adherence to the source text. \\
\hline snarks & Let's expertly determine the sarcastic statement among the given options and confidently provide the definitive answer without any room for doubt or confusion, ensuring absolute precision, clarity, and unwavering expertise in our response, while carefully analyzing the context, tone, and intention behind each statement to achieve unrivaled accuracy and unwavering confidence. \\
\hline sports_understanding temporal_sequences & \begin{tabular}[t]{l}
Let's find the accurate information. \\
The flawless approach
\end{tabular} \\
\hline tracking_shuffled_objects _seven_objects & By meticulously analyzing the given scenarios and accurately determining the final outcomes through a series of trades, swaps, and exchanges among the individuals involved, let's ascertain the conclusive results. \\
\hline web_of_lies & Let's scrutinize each statement provided to accurately determine the truth-teller and uncover the veracity behind their words with unwavering analysis. \\
\hline word_sorting & Employing efficient and precise measures, sort the given list of words in alphabetical order to provide an optimal solution for any sorting problem, ensuring maximum performance and effectiveness. \\
\hline
\end{tabular}
\end{table}
