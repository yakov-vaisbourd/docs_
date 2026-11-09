# B Prompting Formats for Scorer LLM

Figure 14, 15, and 16 show examples of the Q_begin, Q_end, and A_begin prompting formats when the "QA" pattern is present. The "QA" pattern is eliminated when prompting instruction-tuned scorer models like text-bison with the Q_begin and Q_end formats (Figure 17 and 18).

Q: \{instruction\}
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

A:

Figure 14: The Q_begin prompting format on a GSM8K test exemplar with the "QA" pattern.

Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? \{instruction\}

A:

Figure 15: The Q_end prompting format on a GSM8K test exemplar with the "QA" pattern.

Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

A: \{instruction\}

Figure 16: The A_begin prompting format on a GSM8K test exemplar.
\{instruction\}
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Figure 17: The Q_begin prompting format on a GSM8K test exemplar without the "QA" pattern.

Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? \{instruction\}

Figure 18: The Q_end prompting format on a GSM8K test exemplar without the "QA" pattern.

