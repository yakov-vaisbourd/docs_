# B Qualitative examples

See Figure 10 and Figure 11 for cherry-picked examples of responses to challenging instructions.

![Figure](./chungScalingInstructionFinetunedLanguage2022_images/5b970e92-029a-4684-b206-dafd7ac86e22-25_1498_1604_457_260.jpg)

**Figure 10: Qualitative examples of responses to challenging open-ended questions.**

Zero-shot chain-of-thought commonsense reasoning

Input Text

I see a person walking towards the back of the bus. The person looks as if it is walking backward. Explain why with step-by-step reasoning.

Flan-PaLM output
When a bus drives forward, the person walking towards the back of the bus will appear to be walking backward. Thus, the answer is the bus drives forward.

Zero-shot chain-of-thought commonsense reasoning

Input Text
Answer the following question by reasoning step-by-step.
Am I blood related to my step-aunt's husband's nephew? Definitely yes, definitely no, or maybe?

Flan-PaLM output
My step-aunt is married to my uncle. My uncle's nephew is my cousin. Therefore, the final answer is definitely yes.

Zero-shot speech writing
Input Text
<article about US open Nadal vs Medvedev>
Write me a speech for Rafael Nadal to give for his US Open victory:

Flan-PaLM output
Thank you everyone. It is a great honor to be here today. I would like to congratulate Daniil Medvedev on an incredible match. He is a great champion and he will win many tournaments. I would also like to congratulate the tournament organizers for the great event they put on every year.

Few-shot arithmetic reasoning
Input Text
How many sides does a rectangle and heptagon have, when combined? Answer this question with some math.

A rectangle has 4 sides. A heptagon has 7 sides. In combination they have $4+7=11$ sides.

How many sides does a square and a triangle have combined? Answer this question with some math.

Flan-PaLM output
A square has 4 sides and a triangle has 3 sides. $4+3=7$ sides. A square and a triangle have 7 sides combined.

Zero-shot chain-of-thought arithmetic reasoning
Input Text
Answer the following question by reasoning step-by-step.
An earthquake caused four buildings to collapse. Experts predicted that each following earthquake would have double the number of collapsing buildings as the previous one, since each one would make the foundations less stable. After three more earthquakes, how many buildings had collapsed including those from the first earthquake?

Flan-PaLM output
Step-by-step reasoning process: The second earthquake caused 2 * $4=8$ buildings to collapse.
The third earthquake caused $2 * 8=16$ buildings to collapse.
The fourth earthquake caused $2 * 16=32$ buildings to collapse.
In total, the earthquakes caused $4+8+16+32=60$
buildings to collapse.
The answer is 60.

Few-shot explanation
Input Text
I will explain these jokes:
(1) The problem with kleptomaniacs is that they always take things literally.
Explanation: This joke is wordplay. Someone who "takes things literally" is someone who doesn't fully understand social cues and context, which is a negative trait. But the definition of kleptomania is someone who literally takes things.
(2) Always borrow money from a pessimist. They'll never expect it back.
Explanation: Most people expect you to pay them back when you borrow money, however a pessimist is someone who always assumes the worst, so if you borrow money from them, they will expect that you won't pay them back anyways.
(3) If a parsley farmer gets sued, can they garnish his wages?

Flan-PaLM output
Explanation: Garnishing wages is a legal action where a portion of a person's income is taken to pay off a debt. Garnishing can also refer to the process of adding parsley to a dish.

Figure 11: More qualitative examples of responses to challenging open-ended questions.

