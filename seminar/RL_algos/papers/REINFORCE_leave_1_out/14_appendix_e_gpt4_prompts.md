# E GPT-4 Evaluation Prompts

TL;DR Summarize: Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise.

Post: \{instruction\}
Summary (A): \{output_1\}
Summary (B): \{output_2\}
FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "Summary (A)" or " Summary (B)" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation> Preferred: <"Summary (A)" or "Summary (B)">
Anthropic-HH: For the following query to a chatbot assistant, which response is more helpful?
Query: instruction

Response (A): \{output_1\}
Response (B): \{output_2\}
FIRST provide a one-sentence comparison of the two responses and explain which you feel is more helpful. SECOND, on a new line, state only "Response (A)" or "Response (B)" to indicate which response is more helpful. If they are equally good or bad, state "Neither". Your response should use the format:
Comparison: <one-sentence comparison and explanation> Preferred: <"Response (A)" or "Response 1(B)" or "Neither">

