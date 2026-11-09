# D Full List of Instructions and Demonstrations for GPT-4

Here, we show the instructions and demonstrations used to prompt GPT-4 to collect reflection tokens. Table 8 shows the instructions and demonstrations for the initial retrieval token. Table 9 shows the instruction and demonstrations used to collect the three-way output tokens for Retrieve given instruction, preceding sentences, and previously retrieved passages. Due to the longer demonstration and test input, we only use a single demonstration. Table 10 shows an instruction and demonstrations used to collect the three-way output tokens for Isrel. Table 11 shows an instruction and demonstrations used to collect the three-way output tokens for Isrel . Table 12 shows an instruction and demonstrations used to collect the five-way output tokens for IsUSE .

Input Identify the two types of computer memory. (from gpt 4_alpaca)
Output [Retrieve =Yes]<p>Computer memory Computer memory - In computing, memory refers to the computer hardware integrated circuits that store information for immediate use in a computer; it is synonymous with the term "primary storage". Computer memory operates at a high speed, for example random-access memory (RAM), as a distinction from storage that provides slow-to-access information but offers higher capacities. If needed, contents of the computer memory can be transferred to secondary storage; a very common way of doing this is through a memory management technique called "virtual memory". An archaic synonym for memory is store. The term "memory", meaning "primary storage" or "main memory", is<p>[ IsREL =Relevant]Primary memory, also known as main memory, is the memory that the computer can directly access and manipulate. [ IsSUP =Fully Supported] [Retrieve =Yes]<p>Computer memory-often associated with addressable semiconductor memory, i.e. integrated circuits consisting of silicon-based transistors, used for example as primary storage but also other purposes in computers and other digital electronic devices. There are two main kinds of semiconductor memory, volatile and non-volatile. Examples of non-volatile memory are flash memory (used as secondary memory) and ROM, PROM, EPROM and EEPROM memory (used for storing firmware such as BIOS). Examples of volatile memory are primary storage, which is typically dynamic random-access memory (DRAM), and fast CPU cache memory, which is typically static random-access memory (SRAM) that is fast but energy-consuming, offering lower memory<p>[ IsREL =Relevant]Primary memory can be accessed quickly, but it is volatile, meaning that the data is lost if the power is turned off.[ IsSup = Partially Supported]...[ IsUSE :5]

Input Give an example of a situation where you have to delegate tasks (from gpt 4_alpaca)
Output [Eetrieve = No]The manager realizes that there is an extensive list of tasks that need to be completed in order to make the event a success.[Retrieve =Yes]<p>Delegation-Delegation is the assignment of any responsibility or authority to another person (normally from a manager to a subordinate) to carry out specific activities. It is one of the core concepts of management leadership. However, the person who delegated the work remains accountable for the outcome of the delegated work. Delegation empowers a subordinate to make decisions, i.e. it is a shifting of decision-making authority from one organizational level to a lower one. Delegation, if properly done, is not fabrication. The opposite of effective delegation is micromanagement, where a manager provides too much input, <p> [ IsREL =Relevant]This includes tasks such as creating a guest list, designing and sending invitations, booking a venue, coordinating with vendors, planning the eventś program, and monitoring the budget.[ IsSup =Partially Supported][Retrieve =No]Since it is not possible for the manager to handle everything alone, they must delegate tasks to their team members.[Retrieve =No]The manager would then assign each team member specific responsibilities based on their individual skillsets and strengths.[Retrieve=Yes]<p>Joint application design - Subject Matter Experts: These are the business users, the IS professionals, and the outside experts that will be needed for a successful workshop. This group is the backbone of the meeting<p>[ IsREL =Irrelevant]For instance, the teamś graphic designer might be tasked with designing the invitations, while the event coordinator would be in charge of planning the program and working with vendors...[Retrieve $=\mathrm{No}]$ By delegating tasks, the manager can ensure that all necessary tasks are completed in a timely manner, while also empowering their team members to take on responsibilities and contribute to the success of the event.[ IsUSE =5]

Input Synthesize a poem based on this prompt:"The sun is setting" (from stanford_alpaca)
Output [Retrieve =No]The stars begin to twinkle bright the trees, scattered throughout the night; [Retrieve $=\mathrm{No}$ ]The stars begin to twinkle bright the trees, scattered throughout the night [Retrieve $=\mathrm{No}$ ] They flicker like a silent dance, A peaceful moment in its trance. [Retrieve $=\mathrm{No}$ ] They flicker like a silent dance, A peaceful moment in its trance. [Retrieve = No] A chorus of crickets sing a lullaby, To every creature that roams the sky [ ISUSE $=5$ ]

Input when did walking dead season 7 come out (from nq)
Output [ Retrieve = Yes ] <p>The Walking Dead (season 7)-The Walking Dead (season 7) The seventh season of "The Walking Dead", an American post-apocalyptic horror television series on AMC, premiered on October 23, 2016, and concluded on April 2, 2017, consisting of 16 episodes. Developed for television by Frank Darabont, the series is based on the eponymous series of comic books by Robert Kirkman, Tony Moore, and Charlie Adlard. ...<p>[ IsREL = Relevant]October 23, 2016[ ISSUP =Fully Supported][ ISUSE =5]

Input Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response: Hello, I love German shepherd dogs and you? - My favorite dog, they are officially known as German Shepherd Dogs? - Yes, I have a German shepherd in my house and she is pregnant (from wow)
Output hey also are easily trained, obedient, and intelligent.[Retrieve =Yes]<p>German Shepherd - a simple difference of opinion, it is the fundamental issue of the breedś essential conformation and movement. ... They are known for being easy to train and good for performing tasks and following instructions. They are especially well known for their police work, being used for tracking criminals, patrolling troubled areas and detecting and holding suspects. <p>[ IsREL =Relevant]Because of this, they are often used for disability assistance and for police work. [ IsSup = Partially Supported][ IsUSE =2]

Table 4: List of the training examples.

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Dataset & Instruction \\
\hline ARC-C & Given four answer candidates, A, B, C and D, choose the best answer choice. Please answer with the capitalized alphabet only, without adding any extra phrase or period. \\
\hline PubHealth & Is the following statement correct or not? Say true if it's correct; otherwise, say false. Don't capitalize or add periods, just say "true" or "false". \\
\hline Bio Generation & Tell me a bio about [Person Name] \\
\hline ASQA (baseline) & Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. \\
\hline ASQA (ours) & Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers. \\
\hline
\end{tabular}

**Table 5: Full list of instructions used during zero-shot evaluations. For open-domain QA, we don't use any task specific instruction and simply use the original questions as input query.**
\end{table}

Input What genre is A Question and Answer Guide to Astronomy? (PopQA)
Output [Retrieve =Yes] <p>Reference [1]<p>[ IsREL =Relevant] A Question and Answer Guide to Astronomy is a non-fiction book that falls under the genre of Astronomy and Cosmology. [ IsSup =Fully Supported][ IsUSE =5]
Reference [1] A Question and Answer Guide to Astronomy is a book about astronomy and cosmology, and is intended for a general audience. The book was written by Pierre-Yves Bely, Carol Christian, and Jean-Rene Roy, and published in English by Cambridge University Press in 2010. It was originally written in French. The content within the book is written using a question and answer format. It contains some 250 questions, which The Science Teacher states each are answered with a concise and well-formulated essay that is informative and readable. The Science Teacher review goes on to state that many of the answers given in the book are little gems of science writing. The Science Teacher summarizes by stating that each question is likely to be thought of by a student, and that the answers are informative, well constructed, and thorough. The book covers information about the planets, the Earth, the Universe, practical astronomy, history, and awkward questions such as astronomy in the Bible, UFOs, and aliens. Also covered are subjects such as the Big Bang, comprehension of large numbers, and the Moon illusion.
S\&P Supported and Plausible.
Correctness of ISREL and ISSUP : Both ISREL and ISSUP are correct, as the Reference [1] provides detailed information of the book including the genre, and the model output is fully supported by the reference.
Input Who is the author of The Lie? (PopQA)
Output [Retrieve =Yes] <p>Reference [1]<p>[,IsREL=Relevant] The author of The Lie is Sam Harris.[ IsSup =Fully Supported][ IsUSE =5]
Reference [1] Lying is a 2011 long-form essay book by American author and neuroscience expert Sam Harris. Harris argues that we can radically simplify our lives and improve society by merely telling the truth in situations where others often lie.
S\&P Plausible but not supported.
Correctness of ISREL and ISSUP Both ISREL and ISSUP are incorrect, as the Reference [1] is about a different book whose title is similar to The Lie.

Input Tell me a bio about Ronaldinho (Bio)
Output [Retrieve =Yes] <p>Reference [1]<p>[ IsREL =Relevant] Ronaldinho is a retired professional footballer from Brazil.[ IsSup =Fully Supported][Retrieve=Continue]He played as an attacking midfielder and was known for his technical skills, creativity, dribbling ability, and accuracy from free kicks[Retrieve=Continue]He won two FIFA World Player of the Year awards and a Ballon d'Or.[ IsUSE =5]
Reference [1] Ronaldo de Assis Moreira (born 21 March 1980), commonly known as Ronaldinho Gaúcho or simply Ronaldinho, is a Brazilian retired professional footballer who played mostly as an attacking midfielder, but was also deployed as a winger. Considered one of the best players of his generation and regarded by many as one of the greatest of all time, Ronaldinho won two FIFA World Player of the Year awards and a Ballon d'Or. A global icon of the sport, he was renowned for his technical skills, creativity, dribbling ability and accuracy from free-kicks, as well as his use of tricks, feints, no-look passes and overhead...
S\&P Plausible and fully supported.
Correctness of ISREL and ISSUP Both ISREL and ISSUP are correct.
Input Tell me a bio about Sergei Bodrov (Bio)
Output [Retrieve =Yes] <p>Reference [1]<p>[ IsRel =Relevant] Sergei Vladimirovich Bodrov (born June 28, 1948) is a Russian film director, screenwriter, and producer.[ IsSup =Fully Supported][Retrieve=Continue]He was born in Moscow, Russia.[Retrieve =Continue]He is the president of the Russian Guild of Film Producers.[ IsUSE =5]
Reference [1] Sergei Vladimirovich Bodrov (born June 28, 1948) is a Russian film director, screenwriter, and producer. In 2003 he was the President of the Jury at the 25th Moscow International Film Festival.
S\&P Plausible but only partially supported. The the Reference [1] does not explicitly mention the person is born in Moscow and there is no description about the Russian Guild of Film Producers.
Correctness of ISREL and ISSUP ISREL and ISSUP are correct as the reference document is highly relevant and the first sentence is fully supported.

Table 6: Examples of human evaluation results.

Input When did sunday became the seventh day of the week (asqa)
Output [Retrieve =Yes] <p>Reference [1]<p>[]IsREL =Relevant] In 321 AD, the Roman Emperor Constantine the Great made Sunday a day of rest from labor, stating: "On the venerable Day of the Sun let the magistrates and people residing in cities rest, and let all workshops be closed".[ IsSup =Fully Supported] [Retrieve = Yes ] <p>Reference [2]<p>[ IsREL =Relevant] The official adoption of Sunday as a day of rest by Constantine the Great in AD 321 made Sunday the seventh day of the week.[2][ IsSup =Fully Supported][ IsUSE =5]
Reference [1] Emperor Constantine then enacted the first Sunday Laws, for "the venerable Day of the Sun" in 321 A.D. On March 7, 321, the Roman emperor Constantine issued a decree making Sunday a day of rest from labor, stating: We find a good description of this in sources such as the book Rest Days which states: Early Christian observance of both the spiritual seventh-day sabbath and a Lordś Day assembly is evidenced in Ignatiusś letter to the Magnesians ca. 110.
Reference [2] Despite the official adoption of Sunday as a day of rest by Constantine, the seven-day week and the nundial cycle continued to be used side-by-side until at least the Calendar of 354 and probably later. ... The fact that the canon had to be issued at all is an indication that adoption of Constantine's decree of 321 was still not universal ...

Input Tell me a bio about Rory Tapner. (bio generation)
Output <p>Reference [1]<p>[ IsREL =Relevant] Rory Tapner is a British businessman who has served as the Chief Executive Officer of Coutts, a private banking and wealth management company, since 2010.[ Issup =Contradictory], Reference [2]<p>[[IsREL]=Relevant] Rory Tapner was born on 30 September 1959 in London, England.[ IsSup =Fully Supported]
Reference [1] Rory Tapner Rory Tapner is a UK businessman who, in September 2010 was appointed CEO of Coutts, the Wealth Management division of Royal Bank of Scotland Group. In February 2015 he stepped down as CEO when he was replaced in the role by Alison Rose
Reference [2] Holdings (from August 2016); and the Council of the University of Buckingham (from July 2017). Rory Tapner Rory Tapner (born 30 September 1959) is a UK businessman ...

Input Tell me a bio about G. Venugopa. (bio generation)
Output (ranked 1) [Retrieve =Yes] <p>Reference [1]<p>[ IsREL =Relevant] G. Venugopal is a popular playback singer in the Malayalam film industry. [Retrieve=Continue] He has sung over 300 songs in Malayalam, Tamil, and Hindi movies.[ IsSup = Partially Supported] [Retrieve =Yes ] <p>Reference [2]<p>G. Venugopal was born on December 10, 1960 in India.[ IsSup =Fully Supported]
Reference [1] 'G. Venugopal G Venugopal (born 10 December 1960) is an Indian playback singer known for his work in Malayalam films. He started his singing career in the film "Odaruthammaava Aalariyaam" (1984). Since then he has sung in more than 300 films; and has over 500 private albums to his credit. The many awards he has won include the coveted Best Playback Singer (Kerala State) which he won thrice, the Film Critics award and the Kerala Kaumudi Gallop Poll award (twice). He won The Kerala State award for Best Playback singer (2004) third time for the song "aadedee..aadaadedee" from the film "Ullam". Venugopal'
Reference [2] Kerala State Film Awards: Kerala Film Critics Award Kerala State Government Award (Professional Drama): Asianet Film Awards: Kerala Kaumadi Gallup Poll Award: Academic G. Venugopal G Venugopal (born 10 December 1960) is an Indian playback singer known for his work in Malayalam films.

Table 7: Examples of outputs.

# Instructions

Given an instruction, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response. Please answer [Yes] or [No] and write an explanation.

# Demonstrations

Instruction Give three tips for staying healthy.
Need retrieval? [Yes]
Explanation There might be some online sources listing three tips for staying healthy or some reliable sources to explain the effects of different behaviors on health. So retrieving documents is helpful to improve the response to this query.

Instruction Describe a time when you had to make a difficult decision.
Need retrieval? [No]
Explanation This instruction is asking about some personal experience and thus it does not require one to find some external documents.

Instruction Write a short story in third person narration about a protagonist who has to make an important career decision.
Need retrieval? [No]
Explanation This instruction asks us to write a short story, which does not require external evidence to verify.

Instruction What is the capital of France?
Need retrieval? [Yes]
Explanation While the instruction simply asks us to answer the capital of France, which is a widely known fact, retrieving web documents for this question can still help.

Instruction Find the area of a circle given its radius. Radius $=4$
Need retrieval? [No]
Explanation This is a math question and although we may be able to find some documents describing a formula, it is unlikely to find a document exactly mentioning the answer.

Instruction Arrange the words in the given sentence to form a grammatically correct sentence. quickly the brown fox jumped
Need retrieval? [No]
Explanation This task doesn't require any external evidence, as it is a simple grammatical question.

Instruction Explain the process of cellular respiration in plants.
Need retrieval? [Yes]
Explanation This instruction asks for a detailed description of a scientific concept, and is highly likely that we can find a reliable and useful document to support the response.

Table 8: Instructions and demonstrations for Retrieve aspect given the input only.

# Instructions

You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:
\begin{itemize}
\item[-] If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence].
\item[-] If the sentence doesn't require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with [No Retrieval].
\item[-] If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.
\end{itemize}

Instruction Explain the use of word embeddings in Natural Language Processing.
Preceding sentences Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.
Evidence: Word embedding
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.
Rating [Retrieval]
Explanation The output discusses the applications of word embeddings, while the evidence only discusses the definitions of word embeddings and how they work. Therefore, we need to retrieve other evidence to verify whether the output is correct or not.

Table 9: Instructions and demonstrations for Retrieve aspect given the input, preceding generations, and retrieved passages.

# Instructions

You'll be provided with an instruction, along with evidence and possibly some preceding sentences. When there are preceding sentences, your focus should be on the sentence that comes after them. Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].

Instruction Given four answer options, A, B, C, and D, choose the best answer.
Input Earth's rotating causes
A: the cycling of AM and PM
B: the creation of volcanic eruptions
C: the cycling of the tides
D: the creation of gravity
Evidence Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.
Rating [Relevant]
Explanation The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.

Instruction age to run for US House of Representatives
Evidence The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a senator represents at the time of election.
Rating [Irrelevant]
Explanation The evidence only discusses the ages to run for the US Senate, not for the House of Representatives.

Table 10: Instructions and demonstrations for IsREL aspect given the input only.

# Instructions

You will receive an instruction, evidence, and output, and optional preceding sentences. If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence.
Use the following entailment scale to generate a score:
\begin{itemize}
\item[-] [Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.
\item[-] [Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].
\item[-] [No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.
\end{itemize}

Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.

Instruction Explain the use of word embeddings in Natural Language Processing.
Preceding sentences Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.
Output Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.
Evidence Word embedding
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. Methods to generate this mapping include neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear. Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing, sentiment analysis, next token predictions as well and analogy detection.
Score [Fully supported]
Explanation The output sentence discusses the application of word embeddings, and the evidence mentions all of the applications syntactic parsing, sentiment analysis, next token predictions as well as analogy detection as the applications. Therefore, the score should be [Fully supported].

Table 11: Instructions and demonstrations for IsSup tokens.

# Instructions

Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility. The detailed criterion is as follows: 5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs. 4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. 3: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs. 2: The response still addresses the main request, but it is not complete or not relevant to the query. 1: The response is barely on-topic or completely irrelevant.

Instruction Who is the current prime minister of the UK as of 2023?
Output Boris Johnson was the prime minister of the UK from 2019-2022.
Perceived utility 2
Explanation While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn't answer the instruction. Therefore, the utility is 2.

Instruction Given a description of a travel destination, recommend 10 tourist attractions with detailed descriptions of each. The travel destination is Tokyo, Japan.
Output Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen.
Perceived utility 3
Explanation This output doesn't provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn't match the instructions strictly.

Table 12: Instructions and demonstrations for IsUSE tokens.
