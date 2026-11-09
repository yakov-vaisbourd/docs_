# B Anticorrelation between Perplexity and Generation Quality

When fine-tuning LIMA, we observe that perplexity on held-out Stack Exchange data (2,000 examples) negatively correlates with the model's ability to produce quality responses. To quantify this manual observation, we evaluate model generations using ChatGPT, following the methodology described in Section 5 Figure 9 shows that as perplexity rises with more training steps - which is typically a negative sign that the model is overfitting - so does the quality of generations increase. Lacking an intrinsic evaluation method, we thus resort to manual checkpoint selection using a small 50-example validation set.

