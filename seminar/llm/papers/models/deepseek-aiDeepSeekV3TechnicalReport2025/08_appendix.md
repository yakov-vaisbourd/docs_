# Appendix

# A. Contributions and Acknowledgments

\begin{tabular}[t]{|l|l|}
\hline Research \& Engineering & Lecong Zhang \\
\hline Aixin Liu & Liang Zhao \\
\hline Bing Xue & Litong Wang \\
\hline Bingxuan Wang & Liyue Zhang \\
\hline Bochao Wu & Mingchuan Zhang \\
\hline Chengda Lu & Minghua Zhang \\
\hline Chenggang Zhao & Minghui Tang \\
\hline Chengqi Deng & Panpan Huang \\
\hline Chenyu Zhang* & Peiyi Wang \\
\hline Chong Ruan & Qiancheng Wang \\
\hline Damai Dai & Qihao Zhu \\
\hline Daya Guo & Qinyu Chen \\
\hline Dejian Yang & Qiushi Du \\
\hline Deli Chen & Ruiqi Ge \\
\hline Erhang Li & Ruisong Zhang \\
\hline Fangyun Lin & Ruizhe Pan \\
\hline Fucong Dai & Runji Wang \\
\hline Fuli Luo* & Runxin Xu \\
\hline Guangbo Hao & Ruoyu Zhang \\
\hline Guanting Chen & Shanghao Lu \\
\hline Guowei Li & Shangyan Zhou \\
\hline H. Zhang & Shanhuang Chen \\
\hline Han Bao* & Shengfeng Ye \\
\hline Hanwei Xu & Shirong Ma \\
\hline Haocheng Wang* & Shiyu Wang \\
\hline Haowei Zhang & Shuiping Yu \\
\hline Honghui Ding & Shunfeng Zhou \\
\hline Huajian Xin* & Shuting Pan \\
\hline Huazuo Gao & Tao Yun \\
\hline Hui Qu & Tian Pei \\
\hline Jianzhong Guo & Wangding Zeng \\
\hline Jiashi Li & Wanjia Zhao* \\
\hline Jiawei Wang* & Wen Liu \\
\hline Jingchang Chen & Wenfeng Liang \\
\hline Jingyang Yuan & Wenjun Gao \\
\hline Junjie Qiu & Wenqin Yu \\
\hline Junlong Li & Wentao Zhang \\
\hline Junxiao Song & Xiao Bi \\
\hline Kai Dong & Xiaodong Liu \\
\hline Kai Hu* & Xiaohan Wang \\
\hline Kaige Gao & Xiaokang Chen \\
\hline Kang Guan & Xiaokang Zhang \\
\hline Kexin Huang & Xiaotao Nie \\
\hline Kuai Yu & Xin Cheng \\
\hline Lean Wang & Xin Liu \\
\hline
\end{tabular}

\begin{tabular}[t]{|l|l|}
\hline Xin Xie & Zhigang Yan \\
\hline Xingchao Liu & Zhihong Shao \\
\hline Xingkai Yu & Zhiyu Wu \\
\hline Xinyu Yang & Zhuoshu Li \\
\hline Xinyuan Li & Zihui Gu \\
\hline Xuecheng Su & Zijia Zhu \\
\hline Xuheng Lin & Zijun Liu* \\
\hline Y.K. Li & Zilin Li \\
\hline Y.Q. Wang & Ziwei Xie \\
\hline Y.X. Wei & Ziyang Song \\
\hline Yang Zhang & Ziyi Gao \\
\hline Yanhong Xu & Zizheng Pan \\
\hline Yao Li & \\
\hline Yao Zhao & \\
\hline Yaofeng Sun & Data Annotation \\
\hline Yaohui Wang & Bei Feng \\
\hline Yi Yu & \\
\hline Yichao Zhang & J.L. Cai \\
\hline Yifan Shi & \\
\hline Yiliang Xiong & Lei Xu \\
\hline Ying He & Meng Li \\
\hline Yishi Piao & Ning Tian \\
\hline Yisong Wang & R.J. Chen \\
\hline Yixuan Tan & R.L. Jin \\
\hline Yiyang Ma* & Ruyi Chen \\
\hline Yiyuan Liu & \\
\hline Yongqiang Guo & Shuang Zhou \\
\hline Yu Wu & Tianyu Sun \\
\hline Yuan Ou & X.Q. Li \\
\hline Yuduan Wang & Xiangyue Jin \\
\hline Yue Gong & Xiaojin Shen \\
\hline Yuheng Zou & Xiaosha Chen \\
\hline Yujia He & Xiaowen Sun \\
\hline Yunfan Xiong & Xiaoxiang Wang \\
\hline Yuxiang Luo & Xinnan Song \\
\hline Yuxiang You & Xinyi Zhou \\
\hline Yuxuan Liu & Y.X. Zhu \\
\hline Yuyang Zhou & Yanhong Xu \\
\hline Z.F. Wu & Yanping Huang \\
\hline Z.Z. Ren & Yaohui Li \\
\hline Zehui Ren & Yi Zheng \\
\hline Zhangli Sha & Yuchen Zhu \\
\hline Zhe Fu & Yunxian Ma \\
\hline Zhean Xu & Zhen Huang \\
\hline Zhenda Xie & Zhipeng Xu \\
\hline Zhengyan Zhang & Zhongyu Zhang \\
\hline Zhewen Hao & \\
\hline Zhibin Gou & Business \& Compliance \\
\hline Zhicheng Ma & Dongjie Ji \\
\hline
\end{tabular}

Jian Liang
Jin Chen
Leyi Xia
Miaojun Wang
Mingming Li
Peng Zhang
Shaoqing Wu
Shengfeng Ye
T. Wang
W.L. Xiao
Wei An
Xianzu Wang
Xinxia Shan
Ying Tang
Yukun Zha
Yuting Yan
Zhen Zhang

Within each role, authors are listed alphabetically by the first name. Names marked with * denote individuals who have departed from our team.

# B. Ablation Studies for Low-Precision Training

![Figure](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-47_419_1494_1068_284.jpg)

**Figure 10 | Loss curves comparison between BF16 and FP8 training. Results are smoothed by Exponential Moving Average (EMA) with a coefficient of 0.9.**

# B.1. FP8 v.s. BF16 Training

We validate our FP8 mixed precision framework with a comparison to BF16 training on top of two baseline models across different scales. At the small scale, we train a baseline MoE model comprising approximately 16B total parameters on 1.33T tokens. At the large scale, we train a baseline MoE model comprising approximately 230B total parameters on around 0.9T tokens. We show the training curves in Figure 10 and demonstrate that the relative error remains below $0.25 \%$ with our high-precision accumulation and fine-grained quantization strategies.

# B.2. Discussion About Block-Wise Quantization

Although our tile-wise fine-grained quantization effectively mitigates the error introduced by feature outliers, it requires different groupings for activation quantization, i.e., 1×128 in forward pass and 128x1 for backward pass. A similar process is also required for the activation gradient. A straightforward strategy is to apply block-wise quantization per 128x128 elements like the way we quantize the model weights. In this way, only transposition is required for backward. Therefore, we conduct an experiment where all tensors associated with Dgrad are quantized on a block-wise basis. The results reveal that the Dgrad operation which computes the activation gradients and back-propagates to shallow layers in a chain-like manner, is highly sensitive to precision. Specifically, block-wise quantization of activation gradients leads to
model divergence on an MoE model comprising approximately 16B total parameters, trained for around 300B tokens. We hypothesize that this sensitivity arises because activation gradients are highly imbalanced among tokens, resulting in token-correlated outliers (Xi et al., 2023). These outliers cannot be effectively managed by a block-wise quantization approach.

# C. Expert Specialization Patterns of the 16B Aux-Loss-Based and Aux-Loss-Free Models

We record the expert load of the 16B auxiliary-loss-based baseline and the auxiliary-loss-free model on the Pile test set. The auxiliary-loss-free model tends to have greater expert specialization across all layers, as demonstrated in Figure 10.
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-49_336_1487_297_258.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-49_331_1485_646_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-49_331_1485_996_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-49_329_1485_1343_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-49_334_1485_1693_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-49_164_1485_2042_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-49_155_1480_2218_263.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-49_69_923_2399_553.jpg)
(a) Layers 1-7
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-51_299_1485_331_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-51_336_1485_646_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-51_334_1485_996_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-51_334_1485_1345_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-51_331_1485_1697_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-51_329_1483_2045_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-51_69_923_2399_553.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-52_299_1485_331_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-52_334_1485_644_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-52_329_1485_996_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-52_334_1485_1343_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-52_329_1485_1695_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-52_329_1485_2042_260.jpg)
![](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-52_69_923_2399_553.jpg)

![Figure](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-53_803_1494_943_255.jpg)

**Figure 10 | Expert load of auxiliary-loss-free and auxiliary-loss-based models on three domains in the Pile test set. The auxiliary-loss-free model shows greater expert specialization patterns than the auxiliary-loss-based one. The relative expert load denotes the ratio between the actual expert load and the theoretically balanced expert load.**

