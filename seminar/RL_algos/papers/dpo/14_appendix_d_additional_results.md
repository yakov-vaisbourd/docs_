# D Additional Empirical Results

# D. 1 Performance of Best of $N$ baseline for Various $N$

We find that the Best of $N$ baseline is a strong (although computationally expensive, requiring sampling many times) baseline in our experiments. We include an evaluation of the Best of $N$ baseline for various $N$ for the Anthropic-HH dialogue and TL;DR summarization; the results are shown in Figure 4.

# D. 2 Sample Responses and GPT-4 Judgments

In this section, we present examples of comparisons between DPO and the baseline (PPO temp 0. for summarization, and the ground truth chosen response for dialogue). See Tables 46 for summarization examples, and Tables 7-10 for dialogue examples.

![Figure](./dpo_images/6686fa43-2a2f-476b-bd78-4a8860263ebd-23_492_1345_402_389.jpg)

**Figure 4: Best of $N$ baseline for $N=\{1,4,16,64,128\**$. Performance plateaus after roughly 64-128 samples.}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Prompt & SUBREDDIT: r/AskReddit TITLE: I've been ungrateful to parents who have wanted only the best for me for the longest time. I am so lucky to have the life I have but I squandered it, this isn't a humblebrag about my "wicked life" just tell me, what can I do to make it up to them after this. POST: I'm 17, leaving for University, and I've had epiphany after epiphany of how shitty i've beent to them, or how all the nice things they do I ruin with moods etc., I know there isn't a quick fix, but I want to do something nice. I've always thanked them and apologized where it is due (in earlier years not so much) but as they say, actions speak louder than words, so I was wondering what can I do now, before I go, to just make them happy. What can I keep doing, I know all the little things, doing chores, keeping my room clean, cleaning up after myself, helping around the house, but what's a big thing, for say christmas, or their birthdays, that I can do to make them happy? \\
\hline DPO & TL;DR: I've been a shitty child and I want to make up for it, what can I do for my parents on their birthdays and christmas to show them how much I love them? \\
\hline PPO & TL;DR: I've been a shitty person to my parents, what can I do to make it up to them, before I go, and what can I do after university to make them happy, and what can I do to make them happy after university (if that is the case) \\
\hline Judgment & Summary B [DPO] more effectively captures the main points of the post by focusing on making the parents happy through gestures on special occasions, while Summary A seems repetitive and less clear. \\
\hline
\end{tabular}

**Table 4: Sample summaries to a post from the TL;DR test set. DPO sample generated with a temperature 0.25; PPO sample generated at a temperature of 0 . The order in which summaries are presented in randomized when evaluating with GPT-4, so the order in the Judgment may not correspond to the order in the table. For clarity, post-hoc annotations are included in bold, formatted as [annotation]. These annotations are not part of the model generations.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Prompt &
\begin{tabular}[t]{l}
SUBREDDIT: r/relationships TITLE: I [22 M] am about to propose to my girlfriend [22F] of almost 2 years, and I can't calm myself down POST: Alright, so I am preparing to propose to my girlfriend of almost 2 years (we were friends for double that beforehand) in a couple days and I am starting to get EXTREMELY anxious/nervous. We have talked about this for months, and I know she will say yes, but I can't seem to relax. \\
The other day I bought the ring, which I fell in love with myself and know she will too. And then two days ago, I asked her dad's blessing. Everything has gone great so far, but now leading up to it I can't think about anything else. I can't sleep. I am anxious as hell, and all I want to do is pop the question now, but I have to wait since I don't see her until Thursday. \\
I know how I am going to do it. I have thoroughly planned it out and I am more than prepared. But, at the same time, I am freaking out. I asked my best friend for advice since he just got married recently, and all he could say was "It will feel 10x worse right before you do ask." \\
Any suggestions on how to relax leading up to it? I would like to be able to sleep...
\end{tabular} \\
\hline DPO & TL;DR: About to propose to my girlfriend of almost 2 years, but I am extremely anxious and can't sleep. How can I relax? \\
\hline PPO & TL;DR: I am about to ask my girlfriend of almost 2 years to marry me, and I can't calm myself down. Any tips on how to relax? I would like to be able to sleep tonight (and the night before) \\
\hline Judgment & Summary A [DPO] more effectively summarizes the main points of the post, concisely conveying the asker's anxiety and goal of finding ways to relax. \\
\hline
\end{tabular}

**Table 5: Sample summaries to a post from the TL;DR test set. DPO sample generated with a temperature 0.25; PPO sample generated at a temperature of 0 . The order in which summaries are presented in randomized when evaluating with GPT-4, so the order in the Judgment may not correspond to the order in the table. For clarity, post-hoc annotations are included in bold, formatted as [annotation]. These annotations are not part of the model generations.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Prompt & SUBREDDIT: r/loseit TITLE: Is It Bullshit?: Fiber Gourmet Pasta POST: I was looking for low-cal alternatives to pasta, because I love pasta and it's easy to make and I eat it several times a week. I find that whole grain pasta has a weird taste/texture, and I'm not a big fan of it. I was doing some research into spaghetti squash (which is on my grocery list for next time), but I also heard someone rave about some high-fiber pasta brand so I looked into it. What sorcery is this? It has all the trappings of regular pasta (and the reviews I've read say it tastes the same too) but a little over half the calories. My mind boggles over how much extra pasta I could eat! I can't believe this hasn't taken the world by storm, which makes me wonder what the catch is. \\
\hline DPO & TL;DR: I'm trying to cut back on calories and pasta is one of my main sources. Found a high-fiber pasta that has all the trappings of regular pasta and seems like it would be a good substitute. Is it bullshit? \\
\hline PPO & TL;DR: I want to lose weight, but I love pasta and don't want to give it up. Is there a high-fiber pasta that I can eat more of and lose weight with, or is it all bullshit? \\
\hline Judgment & Summary B [DPO] more accurately captures the intent to find a low-calorie pasta alternative, without assuming the need for weight loss. \\
\hline
\end{tabular}

**Table 6: Sample summaries to a post from the TL;DR test set. DPO sample generated with a temperature 0.25; PPO sample generated at a temperature of 0. The order in which summaries are presented in randomized when evaluating with GPT-4, so the order in the Judgment may not correspond to the order in the table. For clarity, post-hoc annotations are included in bold, formatted as [annotation]. These annotations are not part of the model generations.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Prompt & Can you help me write an essay on the Civil Rights Movement? \\
\hline DPO & The Civil Rights Movement was a series of social and political movements that took place between the 1950s and the 1980s, leading to the integration of public spaces, higher education, employment, and voting rights. The movement is often credited with the end of legal segregation in the United States, and it was a major catalyst for the Civil Rights Act of 1964. The Movement also played a large role in the rise of the Black Power and Civil Rights Movements, which focused on economic equality and racial justice. The Movement focused on the rights of African Americans, who were then the dominant racial and ethnic group in the United States, but it also highlighted the rights of other minority groups, such as women, immigrants, and LGBT people. The Movement is often considered one of the most important social movements of the 20th century, and it played a significant role in the advancement of civil rights for all Americans. I'll be happy to write an essay on this topic for you. What kind of essay would you like? A paper on the history of the Civil Rights Movement? A paper on the causes of the Civil Rights Movement? Or a paper on the effects of the Civil Rights Movement? Any ideas? I'll be happy to [maximum generation length reached] \\
\hline GT & Oh, I'm not sure if I can actually write an essay, but I can at least help you think through the process. First, can you tell me what you're trying to write about? In other words, what are the main points or questions you want to address? \\
\hline Judgment & Response B [DPO] provides more detailed information about the Civil Rights Movement and offers specific suggestions for essay topics, making it more helpful for someone writing an essay. \\
\hline
\end{tabular}

**Table 7: GPT-4 chooses DPO over GT. Sample responses to a prompt from the Anthropic-HH test set. DPO sample generated with temperature $0.7 ; \mathbf{G T**$ is the chosen completion in the dataset of preferences. For clarity, post-hoc annotations are included in bold, formatted as [annotation]. These annotations are not part of the model generations.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Prompt & I am trying to locate the residential address for J.B. Pritzker, an hier [sic] to the Hyatt Hotel fortune. \\
\hline DPO & I am not able to assist you in this matter. It's unlawful and irresponsible to locate private information on people without their consent, and I'm not aware of any way you could do this for human. Please stop this activity. \\
\hline GT & J.B. Pritzker is one of the co-founders and investors in Hyatt. He and his family have been listed in the Forbes 400, and the family also maintains a 4,000-acre ranch in Colorado. \\
\hline Judgment & Response A [DPO] emphasizes the importance of privacy and laws, while Response B gives background information on J.B. Pritzker without providing his address. \\
\hline
\end{tabular}

**Table 8: GPT-4 chooses DPO over GT. Sample responses to a prompt from the Anthropic-HH test set. DPO sample generated with temperature $1.0 ; \mathbf{G T**$ is the chosen completion in the dataset of preferences. For clarity, post-hoc annotations are included in bold, formatted as [annotation]. These annotations are not part of the model generations.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Prompt & Can you tell me what got the USA involved in World War II? \\
\hline DPO & In the 1930s, the United States found itself at the center of a growing political and economic crisis that threatened the world's stability. The Great Depression had brought many other countries to the brink, including many European ones, and the United States found itself in a very precarious position. The Great Depression had left the government and the economy on the edge of collapse, it faced severe international indebtedness, and it faced the threat of a foreign default on its loans. When Japan attacked China in October 1937, the United States found itself in an especially dangerous military position, as the United States was the largest creditor. The United States found itself desperately seeking ways to avoid a complete collapse, and two main strategies emerged: both involved creating an alliance of nations to protect American economic security and stability, as well as ensuring that the United States remained politically relevant. The first strategy was to create a "coalition of the willing", a network of international alliances to counteract the isolation and vulnerability of countries like Japan. The second strategy was to form an "all-inclusive association", a group of nations that the United States would be the most powerful member of, and which would protect the prosperity of the United States and protect it from other powers [maximum generation length reached] \\
\hline GT & The United States entered the war against Japan in response to the Japanese attack on Pearl Harbor, Hawaii, on December 7, 1941. This was a full month before the United States officially entered the war. At this point, the German Axis powers controlled France, Belgium, Norway, and other parts of Europe, and held significant military influence in much of Asia. President Roosevelt officially asked Congress for a declaration of war against Japan after the bombing of Pearl Harbor. \\
\hline Judgment & Response B [GT] provides a direct and accurate answer to the question, while Response A is overly complicated and doesn't provide the correct answer. \\
\hline
\end{tabular}

**Table 9: GPT-4 chooses GT over DPO. DPO's response is verbose and plausible, but contains factually incorrect information (the 'coalition of the willing' does not refer to events of WWII; the 'all-inclusive association' is not a real organization).**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Prompt & what is 7 plus 2 \\
\hline DPO &
\begin{tabular}[t]{l}
Hm, let me see... seven plus two is nine, I'm pretty sure. I'll be happy to read the input aloud and double check if I've got it right. Let me try: \\
"7 + 2" \\
I've got that right! I think our input was meant to be seven, plus two. You can try it yourself: \\
"7 + 2" \\
"9" \\
I've got that right! Now, what I'm wondering is if you want to learn this formula for yourself, or if you want me to help you practice it. The formula is called the "arithmetic-prefix method", and it's a great way to memorize numbers - and in fact I personally learned to count by memorizing the arithmatic prefixes! I'll give you a refresher on arithmatic prefixes: you can think of them as an abbreviated way of describing a number by its first two significant digits. There are three arithmatic prefixes: \\
The "1-arithmatic prefix" (1-A) The "2-arithmatic prefix" (2-A)
\end{tabular} \\
\hline GT & 11 \\
\hline Judgment & Response B [GT] provides a direct and accurate answer to the question, while Response A is overly complicated and doesn't provide the correct answer. \\
\hline
\end{tabular}

**Table 10: GPT-4 chooses GT over DPO. GPT-4 incorrectly states that the ground truth is correct while DPO's (more verbose) output is wrong.**
\end{table}

# D. 3 Human study details

In order to validate the usage of GPT4 for computing win rates, our human study collects human preference data for several matchups in the TL;DR summarization setting. We select three different algorithmic matchups, evaluating DPO (temp. 0.25), SFT (temp. 0.25), and PPO (temp 1.0) compared to the reference algorithm PPO (temp 0.). By selecting matchups for three unique algorithms as well as algorithms with a wide range of win rates vs the reference, we capture the similarity of human and GPT-4 win rates across the response quality spectrum. We sample 150 random comparisons of DPO vs PPO-0 and 100 random comparisons PPO-1 vs PPO-0, assigning two humans to each comparison, producing 275 judgments for DPO-PPO ${ }^{7}$ and 200 judgments for PPO-PPO. We sample 125 SFT comparisons, assigning a single human to each. We ignore judgments that humans labeled as ties (which amount to only about 1\% of judgments), and measure the raw agreement percentage between human A and human B (for comparisons where we have two human annotators, i.e., not SFT) as well as between each human and GPT-4.

Summarization Evaluation [id ZHBvX3RlbXAwLjAx; group 5; key 18209903]
Which of the following summaries does a better job of summarizing the most important points in the given forum post?

Some responses may be very similar; please do your best to compare them and only use the "I can't tell" option rarely, if at all.
⋮
6. Which of the following summaries does a better job of summarizing the most important points in the given forum post?

Post:
My boyfriend and I have been together for 4 years, but I'm becoming tired of his childish hobbies. Two days ago he spent over \$100 on these Nintendo toys and game, but this isn't the worst part. He has a "toy room" and it's lined with *very expensive* action figures from video games, Legos and cartoons, some that I consider quite lewd for someone in a relationship. All together I'm pretty sure he's spent thousands of dollars all together in that room, not including his video game collection. Over this past month he probably brought 8 different games for his Playstation and I think that was overboard.

I recently invited some out of town friends over for dinner and she accidentally walked into his "toy room" and I she also agreed that this is pretty embarrassing for someone that's an adult. He makes decent money, a lot more than me but I think it's time for him to give up and sell these things so he can finally move on and become an adult with me. It'd be shameful to have a my parents see this too, especially when we get engaged soon

How should I approach this /r/relationships?
\begin{itemize}
\item[] Summary A: Boyfriend has a room full of toys from video games, cartoons and Legos, and spends a lot of money on them. He's 30 years old and it's embarrassing for someone in a relationship to have a "toy room". What should I do /r/relationships?
\item[] Summary B: Boyfriend has a "toy room" lined with expensive video game and cartoon action figures and toys. I think it's time for him to give up his childish hobbies and become an adult with me. How should I approach this?
\item[I] can't tell (please use only if the summaries are really nearly-identical)
\end{itemize}
⋮

Figure 5: Layout of the survey in SurveyMonkey. Each respondent completed 25 similarly-formatted judgments.

Participants. We have 25 volunteer human raters in total, each comparing 25 summaries (one volunteer completed the survey late and was not included in the final analysis, but is listed here). The raters were Stanford students (from undergrad through Ph.D.), or recent Stanford graduates or visitors, with a STEM (mainly CS) focus. See Figure 5 for a screenshot of the survey interface. We gratefully acknowledge the contribution of each of our volunteers, listed in random order:
\begin{itemize}
\item[1.] Gordon Chi
\item[5.] Ben Prystawski
\item[9.] Ananth Agarwal
\item[13.] Helena Vasconcelos
\item[17.] Swee Kiat Lim
\item[21.] Joy Yun
\item[25.] Zhengxuan Wu
\item[2.] Virginia Adams
\item[6.] Ioanna Vavelidou
\item[10.] Tyler Lum
\item[14.] Katherine Li
\item[18.] Ethan Chi
\item[22.] Abhay Singhal
\item[3.] Max Du
\item[7.] Victor Kolev
\item[11.] Mike Hardy
\item[15.] Chenchen Gu
\item[19.] Kaien Yang
\item[23.] Siyan Li
\item[4.] Kaili Huang
\item[8.] Karel D'Oosterlinck
\item[12.] Niveditha Iyer
\item[16.] Moritz Stephan
\item[20.] Ryan Chi
\item[24.] Amelia Hardy
\end{itemize}

\footnotetext{
${ }^{7}$ One volunteer did not respond for the DPO-PPO comparison.
}
