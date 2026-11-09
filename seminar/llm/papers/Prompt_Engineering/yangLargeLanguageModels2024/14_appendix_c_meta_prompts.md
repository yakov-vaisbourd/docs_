# C Meta-Prompts

# C. 1 Meta-Prompt for Math Optimization

Now you will help me minimize a function with two input variables w, b. I have some (w, b) pairs and the function values at those points. The pairs are arranged in descending order based on their function values, where lower values are better.
```
input:
w=18, b=15
value:
10386334
input:
w=17, b=18
value:
9204724
```

Give me a new (w, b) pair that is different from all pairs above, and has a function value lower than any of the above. Do not write code. The output must end with a pair [w, b], where w and b are numerical values.

Figure 19: An example of the meta-prompt for linear regression. The blue text contains solution-score pairs; the orange text are meta-instructions.
```
You are given a list of points with coordinates below: (0): (-4, 5), (1): (17, 76), (2): (-9, 0), (3): (-31,
-86), (4): (53, -35), (5): (26, 91), (6): (65, -33), (7): (26, 86), (8): (-13, -70), (9): (13, 79), (10): (-73,
-86), (11): (-45, 93), (12): (74, 24), (13): (67, -42), (14): (87, 51), (15): (83, 94), (16): (-7, 52), (17):
(-89, 47), (18): (0, -38), (19): (61, 58).
Below are some previous traces and their lengths. The traces are arranged in descending order based
on their lengths, where lower values are better.
<trace> 0,13,3,16,19,2,17,5,4,7,18,8,1,9,6,14,11,15,10,12 </trace>
length:
2254
<trace> 0,18,4,11,9,7,14,17,12,15,10,5,19,3,13,16,1,6,8,2 </trace>
length:
2017
<trace> 0,11,4,13,6,10,8,17,12,15,3,5,19,2,1,18,14,7,16,9 </trace>
length:
1953
<trace> 0,10,4,18,6,8,7,16,14,11,2,15,9,1,5,19,13,12,17,3</trace>
length:
1840
```

Give me a new trace that is different from all traces above, and has a length lower than any of the above. The trace should traverse all points exactly once. The trace should start with <trace> and end with </trace>.

Figure 20: An example of the meta-prompt for Traveling Salesman Problems with problem size $n=20$. The blue text contains solution-score pairs; the orange text are meta-instructions.

# C. 2 Meta-Prompt for Prompt Optimization

Different optimizer models work the best on different styles of meta-prompts. Figure 3 in the main paper shows the meta-prompt for PaLM 2-L-IT; Figure 21 shows that for pre-trained PaLM 2-L; Figure 22 shows that for GPT models.
```
Create a piece of text at the beginning of the answer to enhance the precision in solving diverse grade
school math problems.
Precision: 4 <TEXT>A dime</TEXT>
Precision: 17 <TEXT>The answer is a function. It is</TEXT>
Precision: 19 <TEXT>So how can we find out what this equation means?</TEXT>
Precision: 20 <TEXT>Solutions:</TEXT>
```

Figure 21: An example of the meta-prompt for prompt optimization with pre-trained PaLM 2-L on GSM8K, where the generated instruction will be prepended to the beginning of the scorer LLM output (A_begin in Section 4.1).
```
Your task is to generate the instruction <INS>. Below are some previous instructions with their scores.
The score ranges from 0 to 100.
text:
Let’s figure it out!
score:
61
text:
Let’s solve the problem.
score:
63
(... more instructions and scores ...)
Below are some problems.
Problem:
Q: Alannah, Beatrix, and Queen are preparing for the new school year and have been given books
by their parents. Alannah has 20 more books than Beatrix. Queen has 1/5 times more books than
Alannah. If Beatrix has 30 books, how many books do the three have together?
A: <INS>
Ground truth answer:
140
(... more exemplars ...)
```

Generate an instruction that is different from all the instructions <INS> above, and has a higher score than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>. The instruction should be concise, effective, and generally applicable to all problems above.

Figure 22: An example of the meta-prompt for prompt optimization with GPT models (gpt-3.5-turbo or gpt-4) on GSM8K, where the generated instruction will be prepended to the beginning of the scorer LLM output (A_begin in Section 4.1). The blue text contains solutionscore pairs; the purple text describes the optimization task and output format; the orange text are meta-instructions.

