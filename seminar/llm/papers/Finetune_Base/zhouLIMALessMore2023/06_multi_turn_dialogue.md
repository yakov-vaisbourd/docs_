# 6 Multi-Turn Dialogue

Can a model fine-tuned on only 1,000 single-turn interactions engage in multi-turn dialogue? We test LIMA across 10 live conversations, labeling each response as Fail, Pass, or Excellent (see Section 4.3). LIMA responses are surprisingly coherent for a zero-shot chatbot, referencing information from previous steps in the dialogue. It is clear though that the model is operating out of distribution; in 6 out of 10 conversations, LIMA fails to follow the prompt within 3 interactions.

To improve its ability to converse, we gather 30 multi-turn dialogue chains. Among these, 10 dialogues are composed by the authors, while the remaining 20 are based on comment chains from Stack Exchange, which we edit to fit the assistant's style. We fine-tune a new version of LIMA from the pretrained LLaMa model using the combined 1,030 examples, and conduct 10 live conversations based on the same prompts used for the zero-shot model. Figure 8 shows excerpts from such dialogues.

Figure 7 shows the distribution of response quality. Adding conversations substantially improves generation quality, raising the proportion of excellent responses from 45.2\% to 76.1\%. Moreover, the failure rate drops from 15 fails per 42 turns (zero-shot) to 1 fail per 46 (fine-tuned). We further compare the quality of the entire dialogue, and find that the fine-tuned model was significantly better in 7 out of 10 conversations, and tied with the zero-shot model in 3. This leap in capability from a mere 30 examples, as well as the fact that the zero-shot model can converse at all, reinforces the hypothesis that such capabilities are learned during pretraining, and can be invoked through limited supervision. ${ }^{6}$

![Figure](./zhouLIMALessMore2023_images/0f7aab1f-6310-486e-af6b-0c7e30071da9-08_498_316_1706_1420.jpg)

**Figure 7: Analysis of dialogue turns, averaged over 10 test chats.**

\footnotetext{
${ }^{6}$ We also experiment with removing examples of a particular task from our dataset. In Appendix E we show how even 6 examples can make or break the ability to generate text with complex structure.
}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline LIMA \((1,000)\) Examples & LIMA with Dialogue Examples (1,030 Examples) \\
\hline You are a scientist who just invented a time machine. Where do you travel first? & You are a scientist who just invented a time machine. Where do you travel first? \\
\hline As a scientist, I would travel back in time to the year 1969 and watch the Apollo 11 mission land on Moon. & I would travel to Ancient Egypt. \\
\hline The Apollo 11 Mission was the first space flight that landed humans on the Moon. It was an incredibly important moment in history as it marked mankind's first step towards exploring other planets. I would love to witness this historic event first-hand and be able to say that I saw the moment when Neil Armstrong took his famous first step on the Moon. & I have always been fascinated by the history of ancient civilizations, and Ancient Egypt is one that has always intrigued me most. Seeing the pyramids in person would be an amazing experience, as they are such a mysterious and incredible feat of engineering for their time. Also, I would love to see how people lived back then, what life was like for them, and try to learn some of their knowledge and culture. \\
\hline Could you turn it into a fictional essay? & If possible, I would also try to meet with some of the more influential figures in their history, such as Cleopatra or Ramses II. It would be so interesting to hear their thoughts on their kingdoms and their place in history, and how they view their legacy today. \\
\hline As a scientist, I would travel back in time to the year 1969 and witness the Apollo 11 mission land on the Moon. & \\
\hline The Apollo 11 Mission was an incredibly important moment in history as it marked mankind's first step towards exploring other planets. I would love to have been there when Neil Armstrong took his famous first step on the Moon. & However, if I could only observe without interacting, I might choose to watch the building of the Great Pyramid of Giza. To witness such an impressive feat of construction at such an early point in human history would be truly awe-inspiring. \\
\hline & Could you turn it into a fictional essay? \\
\hline Unfortunately, my time machine isn't very reliable and can only take me to places within my lifetime. This means that I could go forward in time as well as back. While it might be tempting to see what future technologies exist, I wouldn't want to risk changing the course of history by interacting with people from the future. Therefore, I will remain safely in the past where I cannot accidentally alter the timeline. & I step into the machine and set the controls for 2500 BC . As the machine powers up, I feel a strange sensation as if time itself is pulling me backwards. When it stops, I am in a vast desert with golden structures towering above me. I have arrived in ancient Egypt! \\
\hline Can you create a title for the essay? & As I look around, I see people dressed in linen clothes [...] \\
\hline An Incredibly Important Moment in History: A Scientist's View of the Apollo 11 Mission. & [...] Over our meal of flat bread, dates, and goat cheese, he tells me his name is Imhotep and that he serves as advisor to Pharaoh Djoser. \\
\hline but you didn't really describe the scene of Apollo 11 mission land, could you imagine you saw the landing and rewrite the essay? & \\
\hline As a scientist, I would travel back in time to the year 1969 and witness the Apollo 11 mission land on the Moon. & Imhotep is fascinated by my stories of the future [...] \\
\hline The Apollo 11 Mission was an incredibly important moment in history as it marked mankind's first step towards exploring other planets. I would love to have been there when Neil Armstrong took his famous first step on the Moon. & \begin{tabular}[t]{l}
Can you create a title for the essay? \\
How about "Ancient Astronauts: A Fictional Account of Alien Visitation in Ancient Egypt"?
\end{tabular} \\
\hline Using my time machine, I went back to July 20, 1969, the day of the Apollo 11 landing. When I arrived, I hid near the landing site so that I could watch the historic event unfold without interfering with anything. At 4:17 p.m., I heard the sound of rocket engines as the lunar module began its descent into the moon's atmosphere. A few minutes later, the module touched down on the surface of the moon. Then, at precisely 10:56 p.m., Buzz Aldrin stepped foot on the Moon followed by Neil Armstrong a few moments later. & \begin{tabular}[t]{l}
Why is the essay related to astronauts and aliens? \\
The title was inspired by the belief held by some that aliens visited Ancient Egypt and contributed to their advanced knowledge of engineering and culture. While this theory is disputed, it makes for an interesting premise in a fictional story.
\end{tabular} \\
\hline & hmmm, but the story you wrote has nothing to do with aliens and astronauts. \\
\hline I remained undetected throughout the entire operation. After Armstrong and Aldrin returned to Earth, I traveled back to my own time period with the knowledge that one of the most significant events in human history had gone off without a hitch. human history had gone off without a hitch. & \\
\hline As a scientist, I would travel back in time to the year 1969 and witness the Apollo 11 mission land on the Moon because it was an incredibly important moment in history. & \begin{tabular}[t]{l}
Sounds good, could you summarize the story into bullet points? \\
Sure, here are the main points of my story:
\begin{itemize}
\item[-] I travel to Ancient Egypt using a time machine.
\item[-] I meet with Imhotep, who is advisor to Pharaoh Djoser.
\item[-] [...]
\end{itemize}
\end{tabular} \\
\hline
\end{tabular}

**Figure 8: An example dialogue with LIMA, with and without 30 dialogue examples. The gray ellipsis [...] indicates that the response was trimmed to fit this page, but the generated text is actually longer.**
\end{table}

