# 5 Discussion

We present TailSFT as a drop-in replacement for standard SFT that filters already-fit sequences to improve coverage after the SFT stage, which produces improvements in post-RL performance. TailSFT is derived using a combination of theoretical analysis and controlled empirics, and significantly outperforms standard SFT in our language modeling experiments. More broadly, TailSFT uses the coverage principle to target the interaction between the SFT and RL stages of language model training (Chen et al., 2026a). We believe our results motivate a shift in how language models should be optimized, toward targeting the full training trajectory rather than fitting each stage independently.

