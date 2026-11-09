# B Inference Latency Introduced by Adapter Layers

Adapter layers are external modules added to a pre-trained model in a sequential manner, whereas our proposal, LoRA, can be seen as external modules added in a parallel manner. Consequently, adapter layers must be computed in addition to the base model, inevitably introducing additional latency. While as pointed out in Rücklé et al. (2020), the latency introduced by adapter layers can be mitigated when the model batch size and/or sequence length is large enough to full utilize the hardware parallelism. We confirm their observation with a similar latency study on GPT-2 medium and point out that there are scenarios, notably online inference where the batch size is small, where the added latency can be significant.

We measure the latency of a single forward pass on an NVIDIA Quadro RTX8000 by averaging over 100 trials. We vary the input batch size, sequence length, and the adapter bottleneck dimension $r$. We test two adapter designs: the original one by Houlsby et al. (2019), which we call Adapter ${ }^{\mathrm{H}}$, and a recent, more efficient variant by Lin et al. (2020), which we call Adapter ${ }^{\text {L }}$. See Section 5.1 for more details on the designs. We plot the slow-down in percentage compared to the no-adapter baseline in Figure 5.

![Figure](./huLoRALowRankAdaptation2021_images/2f8c4be1-43fa-4600-99ef-0945e0605540-17_692_1250_1268_436.jpg)

**Figure 5: Percentage slow-down of inference latency compared to the no-adapter \((r=0)\) baseline. The top row shows the result for Adapter ${ **^{\mathrm{H}}$ and the bottom row Adapter ${ }^{\mathrm{L}}$. Larger batch size and sequence length help to mitigate the latency, but the slow-down can be as high as over $30 \%$ in an online, short-sequence-length scenario. We tweak the colormap for better visibility.}

