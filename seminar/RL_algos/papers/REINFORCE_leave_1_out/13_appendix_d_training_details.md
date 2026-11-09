# D Training Details

Below are additional details on training and data preprocessing.
Data-preprocessing For each dataset, we filter out prompts that exceed a pre-specified length to minimize the occurrence of generations not containing the EOS token. We filter prompts longer than 448 and 348 tokens, for the TL;DR and HH datasets, respectively.
SFT Training For the TL;DR Summarize dataset, we use the dedicated SFT split. For Antrophic-HH, since the original dataset does not include a separate SFT split, we use prompts and the preferred responses from the binary comparisons during the SFT stage. This is consistent with prior work (Yuan et al., 2023; Dong et al., 2023; Rafailov et al., 2023).

In terms of training hyperparameters, for the Pythia models, similar to previous work (Touvron et al., 2023b; Bai et al., 2022a), we train for 2 epochs with an initial learning rate of 2e-5 in both summarization and dialogue tasks. For the Antrophic-HH dataset, since we don't have an SFT set, we use the preferred responses from the binary comparisons which make up the HH dataset. This is consistent with prior work (Yuan et al., 2023; Dong et al., 2023; Rafailov et al., 2023). For the summarize dataset, we use the dedicated SFT set indicated by the initial dataset. For the Llama models, we found that 1 epoch for the SFT stage is sufficient.
RM Training In the RM stage, we train RM for 1 epoch with an initial learning rate of 1e-5.
For both RM and SFT training, we use a cosine decay learning schedule (Loshchilov and Hutter, 2016) and a 0.03 warm-up ratio.
Preference Training For TL;DR Summarize dataset where we only experiment with Pythia models, we train each variant for 600 steps with a rollout batch-size of 512. We use a $\beta$ value of 0.03.

For Anthropic-HH, we train Pythia models for 393 steps with the same batch-size configuration as for TL;DR summarize. As for Llama models, we follow the setup in (Dong et al., 2023) and use 2048 rollout and step batch size over 2 epochs. We use $\beta=0.10$ for all Anthropic-HH experiments unless otherwise noted. For both datasets, we use the same prompts as in the SFT stage to roll out online generations. Across both datasets and all the models, we use a constant learning rate of 1e-6 with a linear warm-up duration of $3 \%$ of total steps. Learning-rates were chosen after a sweep of $\left\{1 \times 10^{-6}, 1 \times 10^{-5}, 2 \times 10^{-5}\right\}$ for RAFT and RLOO, and $\left\{1 \times 10^{-6}, 1 \times 10^{-5}\right\}$ for PPO and Vanilla PG. For all algorithms, we take 2 gradient steps for each batch.

