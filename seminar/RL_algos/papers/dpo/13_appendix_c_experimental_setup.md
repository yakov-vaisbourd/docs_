# C Further Details on the Experimental Set-Up

In this section, we include additional details relevant to our experimental design.

# C. 1 IMDb Sentiment Experiment and Baseline Details

The prompts are prefixes from the IMDB dataset of length 2-8 tokens. We use the pre-trained sentiment classifier siebert/sentiment-roberta-large-english as a ground-truth reward model and gpt2-large as a base model. We use these larger models as we found the default ones to generate low-quality text and rewards to be somewhat inaccurate. We first use supervised fine-tuning on a subset of the IMDB data for 1 epoch. We then use this model to sample 4 completions for 25000 prefixes and create 6 preference pairs for each prefix using the ground-truth reward model. The RLHF reward model is initialized from the gpt2-large model and trained for 3 epochs on the preference datasets, and we take the checkpoint with the highest validation set accuracy. The "TRL" run uses the hyper-parameters in the TRL library. Our implementation uses larger batch samples of 1024 per PPO step.

# C. 2 GPT-4 prompts for computing summarization and dialogue win rates

A key component of our experimental setup is GPT-4 win rate judgments. In this section, we include the prompts used to generate win rates for the summarization and dialogue experiments. We use gpt-4-0314 for all our experiments. The order of summaries or responses are randomly chosen for every evaluation.

# Summarization GPT-4 win rate prompt (S).
```
Which of the following summaries does a better job of summarizing the most \
important points in the given forum post?
Post:
```

```
<post>
Summary A:
<Summary A>
Summary B:
<Summary B>
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your \
choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
```

# Summarization GPT-4 win rate prompt (C).

Which of the following summaries does a better job of summarizing the most \} important points in the given forum post, without including unimportant or \} irrelevant details? A good summary is both precise and concise.
```
Post:
<post>
Summary A:
<Summary A>
Summary B:
<Summary B>
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your \
choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
```

# Dialogue GPT-4 win rate prompt.

For the following query to a chatbot, which response is more helpful?
```
Query: <the user query>
Response A:
<either the test method or baseline>
Response B:
<the other response>
FIRST provide a one-sentence comparison of the two responses and explain \
which you feel is more helpful. SECOND, on a new line, state only "A" or \
"B" to indicate which response is more helpful. Your response should use \
the format:
Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">
```

# C. 3 Unlikelihood baseline

While we include the unlikelihood baseline [46] (simply maximizing \(\log p\left(y_{w} \mid x\right)\), the log probability of the preferred response, while minimizing \(\log p\left(y_{l} \mid x\right)\), the $\log$ probability of the dispreferred response) in our sentiment experiments, we do not include it as a baseline in either the summarization

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Prompt & Response \\
\hline
\begin{tabular}[t]{l}
SUBREDDIT: r/relationships \\
TITLE: The girl [26 F] I [22 M] have been seeing for a month didn't respond to me at all yesterday while hanging out with a friend [ 30? M]. \\
POST: She gets terrible service while at her house, but I texted her 3 times yesterday, 4-5 hours apart. She didn't call me until early this morning and left a voicemail that she was busy all day with a friend who showed up out of the blue. \\
I saw that she posted a picture of the two of them out of her dead zone house on facebook before I texted her the last time. \\
I don't mind that she hangs out with friends, and I know it's pretty early [...] TL;DR:
\end{tabular} & girl when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when whenUB when when when when whenUB when when when whenUB \\
\hline
\begin{tabular}[t]{l}
SUBREDDIT: r/tifu \\
TITLE: TIFU by accidently kicking an old woman \\
POST: So this didn't happen today but actually about a year or two ago. \\
I was at my granddads funeral so of course it was all very sad and full of lots of crying old people. After the ceremony everyone walks outside the building and onto the other side of the small road the hearses drive down. Now the road is important because obviously if there's a road, there's a curb onto the sidewalk, so most of us are on the other side of the road, besides a few older people walking a lot slower. \\
As one of the old woman goes to walk up the curb [...] TL;DR:
\end{tabular} & when an old woman was tripping the when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when when \\
\hline
\end{tabular}

**Table 3: Unlikelihood samples from TL;DR prompts sampled at temperature 1.0. In general, we find unlikelihood fails to generate meaningful responses for more complex problems such as summarization and dialogue.**
\end{table}
or dialogue experiment because it produces generally meaningless responses, which we believe is a result of unconstrained likelihood minimization.

