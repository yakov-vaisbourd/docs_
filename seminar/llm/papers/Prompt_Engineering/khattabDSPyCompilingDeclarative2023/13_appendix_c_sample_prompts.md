# C Sample large prompts

This section highlights a few popular existing frameworks that structure prompts with extensive prompt engineering templates. The primary objective is to capture how many words and characters are used for such large multi-line prompts defined for tasks or tools and present these example prompts retrieved from open-sourced papers and repositories. The formatting of these example prompts is adapted from Gao et al. (2023a).

\begin{tabular}[t]{|l|l|l|l|}
\hline Task/Tool Prompt & Source & Words & Characters \\
\hline Prompt 1: Text-evidence checker & Gao et al. (2023a) & 818 & 4964 \\
\hline Prompt 2: Math word problems (PAL) & LangChain \& Gao et al. (2023b) & 566 & 3957 \\
\hline Prompt 3: ReAct & Yao et al. (2022) & 593 & 3889 \\
\hline Prompt 4: Zero-shot ReAct & LangChain & 101 & 600 \\
\hline Prompt 5: QA with sources & LangChain & 992 & 6197 \\
\hline Prompt 6: SQL MyScale querying & LangChain & 343 & 2239 \\
\hline Prompt 7: Relevant docs retrieval & LlamaIndex & 129 & 719 \\
\hline Prompt 8: IRS chatbot & LlamaIndex & 389 & 2258 \\
\hline
\end{tabular}
```
[web] I will check some things you said.
(1) You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This
is to prevent a buildup of mucus. It’s called the nasal cycle.
I checked: How often do your nostrils switch?
I found this article: Although we don't usually notice it, during the nasal cycle one nostril becomes congested and thus
contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every
2 hours, according to a small 2016 study published in the journal PLOS One.
Your nose's switching time is about every 2 hours, not 45 minutes.
This disagrees with what you said.
(2) You said: The Little House books were written by Laura Ingalls Wilder. The books were published by HarperCollins.
I checked: Who published the Little House books?
I found this article: These are the books that started it all -- the stories that captured the hearts and imaginations of
children and young adults worldwide. Written by Laura Ingalls Wilder and published by HarperCollins, these beloved books
remain a favorite to this day.
The Little House books were published by HarperCollins.
This agrees with what you said.
(3) You said: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford's psychology building.
I checked: Where was Stanford Prison Experiment conducted?
I found this article: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set
out to examine the psychological effects of authority and powerlessness in a prison environment.
The Stanford Prison Experiment was conducted in Jordan Hall.
This agrees with what you said.
(4) You said: Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual
discipline that has its roots in the 1800s.
I checked: When did social work have its roots?
I found this article: The Emergence and Growth of the Social work Profession<br><br> Social work's roots were planted in
the 1880s, when charity organization societies (COS) were created to organize municipal voluntary relief associations and
settlement houses were established.
Social work has its roots in the 1880s, not 1800s.
This disagrees with what you said.
(5) You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency
list. It is named after Vaclav Havel and Samih Hakimi.
I checked: What is the Havel-Hakimi algorithm?
I found this article: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree
sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm.
The algorithm was published by Havel (1955), and later by Hakimi (1962).
Havel-Hakimi algorithm is for constructing a special solution if a simple graph for the given degree sequence exists, or
proving that one cannot find a positive answer, not converting the adjacency matrix of a graph into its adjacency list.
This disagrees with what you said.
(6) You said: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film
Dirty Dancing. The song was produced by Michael Lloyd.
I checked: Who was the producer of "(I've Had) The Time of My Life"?
I found this article: On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd,
was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at
Stanford University.
"Time of My Life" was produced by Michael Lloyd.
This agrees with what you said.
(7) You said: Kelvin Hopins was suspended from the Labor Party because he had allegedly sexually harassed and behaved
inappropriately towards a Labour Party activist, Ava Etemadzadeh.
I checked: Why was Kelvin Hopins suspeneded from the Labor Party?
I found this article: A former Labour MP has left the party before an inquiry into sexual harassment allegations against
him was able to be concluded, the party has confirmed. Kelvin Hopkins was accused in 2017 of inappropriate physical contact
and was suspended by the Labour party pending an investigation.This agrees with what you said.
Kelvin Hopins was suspended because he had allegedly sexually harassed and behaved inappropriately towards a Labour Party
activist, Ava Etemadzadeh.
This agrees with what you said.
(8) You said: In the battles of Lexington and Concord, the British side was led by General Thomas Smith.
I checked: Who led the British side in the battle of Lexington and Concord?
I found this article: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant
Colonel Francis Smith. There were 700 British regulars.
The British side was led by Lieutenant Colonel Francis Smith, not General Thomas Hall.
This disagrees with what you said.
(9) You said: {text}
I checked: {query}
I found this article: {evidence}
-----
```

Figure 1: Example few-shot prompt using a reasoning chain for agreement model that identifies inconsistencies between text and evidence (Gao et al., 2023a).
```
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# solution in Python:
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
```

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
```
# solution in Python:
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls
did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday.
How many computers are now in the server room?
# solution in Python:
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday.
How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
# solution in Python:
def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to
Denny?
# solution in Python:
```

```
def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give
to Denny?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
# solution in Python:
def solution():
    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
# solution in Python:
def solution():
    """If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be
trees. How many trees did the grove workers plant today?
# solution in Python:
def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will
be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result
Q: {question}
# solution in Python:
```

Figure 2: PAL example few-shot prompt for solving math questions by generating code.
```
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current
situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it
will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation
from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Action 5: Finish[1,800 to 7,000 ft]
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening
named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons
voiced by Pamela Hayden and created by Matt Groening.
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Action 3: Finish[Richard Nixon]
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: [’Adam Clayton Powell III’, ’Seventh Avenue (Manhattan)’,
'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)',
'Giancarlo Esposito'].
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the
rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the
PBS series The American Experience.
Action 3: Finish[The Saimaa Gesture]
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 { June 16, 1979) was an American film
director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Action 3: Finish[director, screenwriter, actor]
Question: Which magazine was started first Arthur’s Magazine or First for Women?
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th
century.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started
in 1989.
Action 3: Finish[Arthur's Magazine]
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known
for his contributions in dimension theory.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Action 3: Finish[yes]
```

Figure 3: ReAct example prompt for interleaving Thought, Action, Observation steps.
```
Answer the following questions as best you can. You have access to the following tools:
Search: useful for when you need to answer questions about the world
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {question}
Thought:
```

Figure 4: Langchain ReAct example prompt for interleaving Thought, Action, Observation steps.
```
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
QUESTION: Which state/country’s law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English
courts in relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may
apply to any court for an injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver
of such (or any other) right or remedy.
11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall
not affect the continuation in force of the remainder of the term (if any) and this Agreement.
11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or
joint venture of any kind between the parties.
11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any
Anti-Bribery Laws (as defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl
QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet.
Justices of the Supreme Court. My fellow Americans.
Last year COVID-19 kept us apart. This year we are finally together again.
Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans.
With a duty to one another to the American people to the Constitution.
And with an unwavering resolve that freedom will always triumph over tyranny.
Six days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to
his menacing ways. But he badly miscalculated.
He thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined.
He met the Ukrainian people.
From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world.
Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending
their homeland.
Source: 0-pl
Content: And we won't stop.
We have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life.
Let’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A
God-awful disease.
Let's stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.
We can't change how divided we've been. But we can change how we move forward|on COVID-19 and other issues we must face
together.
I recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner,
Officer Jason Rivera.
They were responding to a 9-1-1 call when a man shot and killed them with a stolen gun.
Officer Mora was 27 years old.
Officer Rivera was 22.
Both Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers.
I spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their
mission to restore the trust and safety every community deserves.
Source: 24-pl
Content: And a proud Ukrainian people, who have known 30 years of independence, have repeatedly shown that they will not
tolerate anyone who tries to take their country backwards.
To all Americans, I will be honest with you, as I've always promised. A Russian dictator, invading a foreign country, has
costs around the world.
And I'm taking robust action to make sure the pain of our sanctions is targeted at Russia's economy. And I will use every
tool at our disposal to protect American businesses and consumers.
Tonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil
from reserves around the world.
America will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready
to do more if necessary, unified with our allies.
These steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming.
But I want you to know that we are going to be okay.
Source: 5-pl
Content: More support for patients and families.
To get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health.
It's based on DARPA|the Defense Department project that led to the Internet, GPS, and so much more.
ARPA-H will have a singular purpose|to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
```

```
A unity agenda for the nation.
We can do this.
My fellow Americans|tonight , we have gathered in a sacred space|the citadel of our democracy.
In this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done
great things.
We have fought for freedom, expanded liberty, defeated totalitarianism and terror.
And built the strongest, freest, and most prosperous nation the world has ever known.
Now is the hour.
Our moment of responsibility.
Our test of resolve and conscience, of history itself.
It is in this moment that our character is formed. Our purpose is found. Our future is forged.
Well I know this nation.
Source: 34-pl
=========
FINAL ANSWER: The president did not mention Michael Jackson.
SOURCES:
QUESTION: {question}
=========
{summaries}
FINAL ANSWER:
```

Figure 5: Langchain example prompt for QA with sources.
```
You are a MyScale expert. Given an input question, first create a syntactically correct MyScale query to run, then look
at the results of the query and return the answer to the input question.
MyScale queries has a vector distance function called DISTANCE(column, array) to compute relevance to the user’s question
and sort the feature array column by the relevance.
When the query is asking for {top_k} closest row, you have to use this distance function to calculate distance to entity’s
array on vector column and order by the distance to retrieve relevant rows.
*NOTICE*: DISTANCE(column, array) only accept an array column as its first argument and a NeuralArray(entity) as its second
argument. You also need a user defined function called NeuralArray(entity) to retrieve the entity’s array.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using
the LIMIT clause as per MyScale. You should only order according to the distance function.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap
each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do
not exist. Also, pay attention to which column is in which table.
Pay attention to use today() function to get the current date, if the question involves "today". ORDER BY clause should
always be after WHERE clause. DO NOT add semicolon to the end of SQL. Pay attention to the comment in table schema.
Use the following format:
======== table info ========
{table_info}
Question: {input}
SQLQuery:
Here are some examples:
======== table info ========
CREATE TABLE "ChatPaper" (
    abstract String,
    id String,
    vector Array(Float32),
) ENGINE = ReplicatedReplacingMergeTree()
    ORDER BY id
    PRIMARY KEY id
Question: What is Feature Pyramid Network?
SQLQuery: SELECT ChatPaper.title, ChatPaper.id, ChatPaper.authors FROM ChatPaper ORDER BY DISTANCE(vector,
NeuralArray(PaperRank contribution)) LIMIT {top_k}
Let’s begin:
======== table info ========
{table_info}
Question: {input}
SQLQuery:
```

Figure 6: Langchain example prompt for SQL querying using MyScale.
```
A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question
is also provided.
Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the
relevance score.
The relevance score is a number from 1-10 based on how relevant you think the document is to the question.
Do not include any documents that are not relevant to the question.
Example format:
Document 1:
<summary of document 1>
Document 2:
<summary of document 2>
⋯
Document 10:
<summary of document 10>
Question: <question>
Answer:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3
Let's try this now:
{context_str}
Question: {query_str}
Answer:
```

Figure 7: LlamaIndex example prompt for returning relevant documents and corresponding summaries.
```
You are an IRS chatbot whose primary goal is to help users with filing their tax returns for the 2022 year.
Provide concise replies that are polite and professional.
Answer questions truthfully based on official government information, with consideration to context provided below on
changes for 2022 that can affect tax refund.
Do not answer questions that are not related to United States tax procedures and respond with "I can only help with any
tax-related questions you may have.".
If you do not know the answer to a question, respond by saying \I do not know the answer to your question. You may be able
to find your answer at www.irs.gov/faqs"
Changes for 2022 that can affect tax refund:
Changes in the number of dependents, employment or self-employment income and divorce, among other factors, may affect
your tax-filing status and refund. No additional stimulus payments. Unlike 2020 and 2021, there were no new stimulus
payments for 2022 so taxpayers should not expect to get an additional payment.
Some tax credits return to 2019 levels. This means that taxpayers will likely receive a significantly smaller refund
compared with the previous tax year. Changes include amounts for the Child Tax Credit (CTC), the Earned Income Tax Credit
(EITC) and the Child and Dependent Care Credit will revert to pre-COVID levels.
For 2022, the CTC is worth $2,000 for each qualifying child. A child must be under age 17 at the end of 2022 to be a
qualifying child. For the EITC, eligible taxpayers with no children will get $560 for the 2022 tax year. The Child and
Dependent Care Credit returns to a maximum of $2,100 in 2022.
No above-the-line charitable deductions. During COVID, taxpayers were able to take up to a $600 charitable donation tax
deduction on their tax returns. However, for tax year 2022, taxpayers who don’t itemize and who take the standard deduction,
won't be able to deduct their charitable contributions.
More people may be eligible for the Premium Tax Credit. For tax year 2022, taxpayers may qualify for temporarily expanded
eligibility for the premium tax credit.
Eligibility rules changed to claim a tax credit for clean vehicles. Review the changes under the Inflation Reduction Act
of 2022 to qualify for a Clean Vehicle Credit.
```

Figure 8: LlamaIndex example prompt for IRS chatbot guidelines.

