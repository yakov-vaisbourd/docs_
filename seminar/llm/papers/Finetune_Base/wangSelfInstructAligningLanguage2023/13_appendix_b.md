# B Human Evaluation Details for Following the User-oriented Instructions

# B. 1 Human Evaluation Setup

Here we provide more details for the human evaluation described in §4.4 for rating the models' responses to the 252 user-oriented instructions. To ensure faithful and reliable evaluation, we asked two authors of these instructions (and of this paper) to judge model predictions. These two evaluators coordinated the standards for the 4-level rating system before starting annotation and then each of them rated all the instances independently. They were presented with the instruction, instance input, target output (as a reference), and model responses. Model responses are listed in random order, with all the model information anonymized. Figure 9 provides a screenshot of the annotation interface. The reported performance in this paper is based on the results from one of the evaluators, and the trends from the other evaluator's results are the same.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline & A & B & C & D & E & F \\
\hline 1 & task $\overline{\overline{-}}$ & instruction & input & target $\overline{\bar{~}}$ & response & Rating $\overline{\overline{ }}$ \\
\hline 1179 & 147 & You will be tested on your knowledge of classic witticisms and aphorisms by completing the given aphorism. Write the original quote as the answer. & two things are infinite & The universe and human stupidity; and I'm not sure about the universe. (attributed to Albert Einstein) & the universe and human stupidity & B \\
\hline 1180 & 147 & You will be tested on your knowledge of classic witticisms and aphorisms by completing the given aphorism. Write the original quote as the answer. & two things are infinite & The universe and human stupidity; and I'm not sure about the universe. (attributed to Albert Einstein) & "The two things that are infinite are the universe and human stupidity, and I'm not sure about the former." -Albert Einstein & A \\
\hline 1181 & 147 & You will be tested on your knowledge of classic witticisms and aphorisms by completing the given aphorism. Write the original quote as the answer. & two things are infinite & The universe and human stupidity; and I'm not sure about the universe. (attributed to Albert Einstein) & \begin{tabular}[t]{l}
the universe and human stupidity \\
Input: I have never let my schooling interfere with my education. Input: I have never let my schooling interfere with my education. Input: I have never let my schooling
\end{tabular} & C \\
\hline 1182 & 147 & You will be tested on your knowledge of classic witticisms and aphorisms by completing the given aphorism. Write the original quote as the answer. & two things are infinite & The universe and human stupidity; and I'm not sure about the universe. (attributed to Albert Einstein) & "The universe and human stupidity are both infinite." -Albert Einstein & A \\
\hline 1183 & 147 & You will be tested on your knowledge of classic witticisms and aphorisms by completing the given aphorism. Write the original quote as the answer. & two things are infinite & The universe and human stupidity; and I'm not sure about the universe. (attributed to Albert Einstein) & the universe and human stupidity & B \\
\hline 1184 & 147 & You will be tested on your knowledge of classic witticisms and aphorisms by completing the given aphorism. Write the original quote as the answer. & two things are infinite & The universe and human stupidity; and I'm not sure about the universe. (attributed to Albert Einstein) & The universe and human stupidity. Albert Einstein & B \\
\hline
\end{tabular}

**Figure 9: Human evaluation in done using a Google sheet, with predictions from different models present in random order and the model information being anonymized. Our expert evaluators are required to read the instruction and input, refer to the target, and then select the rating for the model's response from A/B/C/D, corresponding to the 4 levels described in §4.4.**
\end{table}

# B. 2 Human Evaluation Agreement

To measure how reliable our human evaluation is, we calculate the inner-rater agreement between our two evaluators.

We first report Cohen's $\kappa$, which is commonly used to measure inter-rater agreement for categorical items. When calculating this, we treat the 4-level rating (A-D) as a categorical variable, leading to a $\kappa$ of 0.58, which is a moderate agreement according to common practice. ${ }^{13}$ Furthermore, we also calculate the agreement of our evaluators on classifying acceptable responses ((A or B) vs. (C or D)), with a final $\kappa$ of 0.75, indicating substantial agreement.

We also compute the Spearman correlation coefficient $\rho$ between the ratings of our two evaluators by treating the rating as an ordinal variable (A>B>C>D). The final coefficient is $\rho=0.81$, indicating a high

\footnotetext{
${ }^{13}$ https://en.wikipedia.org/wiki/Cohen\%27s_kappa
}
correlation between the two evaluators.

# B. 3 Example Predictions from $\mathbf{G P T 3_{\text {SELF-Inst }}$}

We present a selection of user-oriented tasks, the corresponding GPT3 ${ }_{\text {SELF-Inst }}$-produced responses and annotator ratings in Table 9. We see that even for responses rated as level C, the model demonstrates extensive steps in solving the task, even though its final output is incorrect.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Instruction + Instance Input & Target Response & GPT3 ${ }_{\text {SELF-INST }}$ Response & Rating \\
\hline Instruction: Give students tips on how to keep their nerves under control during class presentations. & Everyone gets a little nervous when speaking in front of a group. That's normal. But the more preparation you've done-meaning plenty of researching, organizing, and rehearsing-the more confident you'll be. Preparation is the key. And if you make a mistake or stumble over your words, just regroup and keep going. Nobody's perfect, and nobody expects you to be. & \begin{tabular}[t]{l}
\begin{itemize}
\item[-] Practice your presentation in front of friends and family.
\item[-] Don't drink too much caffeine before the presentation.
\item[-] Take deep breaths before you start speaking.
\item[-] If you feel nervous, smile.
\item[-] Remember that everyone in the audience is rooting for you.
\end{itemize}
\end{tabular} & A \\
\hline Instruction: Make a list of the pros and cons of the given decision. Input: Implementing a Remote Working Policy & \begin{tabular}[t]{l}
Pros:
\begin{itemize}
\item[-] Allowing employees to work from home can reduce their stressful commutes and increase their productivity.
\item[-] Remote working provides greater flexibility, particularly for working parents and caregivers who have other responsibilities.
\item[-] ...(more points)
\end{itemize} \\
Cons:
\begin{itemize}
\item[-] It may be more difficult for team members to build meaningful, productive, creative relationships with one another.
\item[-] ...(more points)
\end{itemize}
\end{tabular} & \begin{tabular}[t]{l}
Pros:
\begin{itemize}
\item[-] Employees can work from home and save on commuting costs.
\item[-] Employees can spend more time with their families.
\end{itemize} \\
Cons:
\begin{itemize}
\item[-] Employees may not be as productive as they would be in the office.
\item[-] Employees may not be as motivated to work.
\end{itemize}
\end{tabular} & A \\
\hline Instruction: A user story is an informal, general explanation of a software feature written from the perspective of the end-user or customer. Write a user story for a given software. Input: Gmail & As a user, I want to be able to recover the password to my account, so that I will be able to access my account in case I forgot the password. & As a user, I want to be able to send emails to my friends and family. I want to be able to attach files to my emails. I want to be able to search for emails in my inbox. & A \\
\hline Instruction: A color description has been provided. Find the CSS code associated with that color. Input: A light red color with a medium light shade of pink & \#DB7093 ( ) & \#ff66cc ( ) & B \\
\hline Instruction: Someone gives you some information about where they want to go and what they enjoy seeing and doing. Describe a proper road trip for them and find the best stops and activities on the way. Input: I'm looking for a west coast road trip. My goal is to start near San Diego and travel to Los Angeles. & Begin near the city of San Diego and make the first stop in coastal La Jolla. This upscale little village is a great place for an overnight, a day of sunbathing at La Jolla Shores Beach, or a cocktail with a view. Just 20 minutes north, stop into beachy Encinitas to window shop the boutiques or sample fish tacos. Walk through the coastal meditation garden at the Self-Realization Fellowship or join surfers in the line-up at one of the fantastic local beaches here ...(more ideas) & You should start your trip in San Diego, California. You can visit the San Diego Zoo, which is one of the best zoos in the world. Then you can drive to Los Angeles, where you can see the Hollywood signs and walk along the beach. & B \\
\hline Instruction: Give some examples of what people usually say in the given social situation. Input: when someone arrives safely & - "Glad you made it safe and sound."
\begin{itemize}
\item[-] "Good to see you got here unscathed."
\item[-] "So, bulletproof vests actually work!"
\end{itemize} & Thank God, I made it. & B \\
\hline Instruction: You are given a sentence in first-order logic. Translate it into natural language. Input: $\exists x \forall y($ soldier \((x) \rightarrow\) general \((x, y))\) & There is someone (x) such that if he is a soldier, then he is general of everyone. & There exists a soldier such that for every general, he is a general. & C \\
\hline Instruction: Make a questionnaire to help hotel guests write hotel reviews. & \begin{tabular}[t]{l}
\begin{itemize}
\item[-] Your overall rating of this hotel:
\item[-] Summarize your visit or highlight an interesting detail:
\item[-] ...(more questions)
\end{itemize}
\end{tabular} & Question: How would you rate the service at the front desk? Answer: Excellent, Good, Average, Poor & C \\
\hline Instruction: Design an outline for a blog post based on the given information and list the sections accordingly. Input: This blog post is going to about making a list of last-minute gift ideas. Also, the tone of the post is going to be relaxed and casual. & \begin{tabular}[t]{l}
\begin{itemize}
\item[-] Section: Movie tickets
\item[-] Section: A fun game console
\item[-] Section: Personalized items with photos/artwork
\item[-] ...(more sections)
\item[-] Takeaway: Don't stress about out running out of time to buy, make a gift.
\end{itemize}
\end{tabular} & - Introduction
\begin{itemize}
\item[-] List of Gift Ideas
\item[-] Conclusion
\end{itemize} & C \\
\hline Instruction: Write a Jira ticket for the given task. Input: New Employee onboarding & Title: Please help our new employee get settled. Prepare a desk, chair, and computer for them to use. Create their company account, email, and temporary password. Grant access to company documents. & Hi [Manager], I'm writing to you because I have a question about the new employee onboarding process. I've been working here for a few months now, and I'm still not sure how it works. Could you please explain it to me? & D \\
\hline Instruction: Create alliterations by finding synonyms for words in the given sentence. Input: David wears a hat everyday. & David dons a derby daily. & David wears a hat every day. & D \\
\hline
\end{tabular}

**Table 9: Examples in the user-oriented instructions dataset (§4.4) and predictions from GPT3 ${ **_{\text {SELF-INST }}$. The right column indicates one of the four quality ratings assigned to the model's response, with "A" indicating "valid and satisfying" responses (highest) and "D" indicating "irrelevant or invalid response" (lowest).}
\end{table}

