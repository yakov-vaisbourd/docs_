# A Additional Results

Below we present additional experimental results on the RLOO estimator.

# A. 1 RLOO vs RAFT Sample Efficiency

Figure 5 plots the test reward for RAFT and RLOO regardless of the value of $k$ vs online samples generated, normalized by batch size which shows RLOO's efficiency.

![Figure](./REINFORCE_leave_1_out_images/7b6ccb24-aeaf-4405-a575-200815561112-15_528_1559_595_250.jpg)

**Figure 5: RL00 and RAFT Rewards are plotted at various points through-out training based on the total \# of samples seen (regardless of whether $k=2$ or $k=4$ ). RL00 is more efficient than RAFT in-terms of online sample use, regardless of $k$ and dataset.**

