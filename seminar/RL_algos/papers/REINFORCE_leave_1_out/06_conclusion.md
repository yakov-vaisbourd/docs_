# 6 Conclusion

At a high level, this work posits that the RLHF setting of fine-tuning an LLM has a strong initialization of the policy, which coupled with further conditioning on a prompt, alleviates historical concerns with high variance and large action spaces. We support this position with empirical results, showing that techniques such as REINFORCE \& RLOO rarely used in traditional RL settings due to high variance outperform PPO. We also show the improved robustness of RLOO relative to iterative fine-tuning methods like RAFT - achieving a best of both worlds sweet spot.

