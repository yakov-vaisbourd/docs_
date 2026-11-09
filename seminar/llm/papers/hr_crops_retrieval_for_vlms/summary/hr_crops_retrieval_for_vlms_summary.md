# Summary

Project page: <https://nimrodshabtay.github.io/AwaRes/>

The paper introduces AwaRes, a framework designed to resolve the trade-off between accuracy and computational efficiency in Vision-Language Models (VLMs).

# Summary

**Problem.** VLMs face a trade-off between accuracy and efficiency when processing high-resolution images:
- High resolution captures fine-grained details (e.g.\ small text, chart values) but incurs large visual token counts.
- Low resolution is cheap but loses critical information.
- The demand for high resolution is spatially sparse - most queries require fine detail only in a small region of the image.

**Prior approaches.** Two families of methods address this trade-off:
- **Fixed-budget token pruning** - reduces the visual token count globally (e.g.\ by dropping low-attention tokens), but the budget is query-agnostic and may discard spatially critical regions.
- **Adaptive resolution escalation** - iteratively increases resolution or requests additional crops based on model confidence, but incurs multiple sequential inference passes and high latency (4.3s for the closest baseline vs.\ sub-second for AwaRes).

**Method.** The authors propose **AwaRes**, a spatial-on-demand framework:
- The model first processes a low-resolution global view, then uses tool-calling to selectively retrieve only the high-resolution crops relevant to the query. Crops are drawn from a fixed candidate set of 9 regions (4 quadrants, center, 4 half-image slices, full image).
- **Coupled decision policy:** In a single first-turn generation, the model jointly decides (a) whether cropping is needed and (b) which specific crops to request - both decisions are made together rather than in separate steps. This is implemented by having the model emit structured tool-call tokens (e.g.\ `<crop>["q1","q3"]</crop>`) as part of its first-turn output, specifying the desired crop IDs.
- Once the tool returns the requested high-resolution crop images, they are appended to the context as additional visual tokens and the model continues its answer in a second turn - reusing the first-turn KV cache so the extra inference cost is minimal.

**Training.** Supervision is constructed automatically (no manual spatial annotations) in two stages. First, a judge LLM compares low- vs.\ high-resolution answers to label whether cropping is needed. Second, an oracle grounding model localizes the evidence and maps it to the crop candidate set. This yields tool-use trajectories used for:
- **Cold-start SFT** - trains the coupled decision policy on the curated trajectories, with upweighted loss on tool-call tokens.
- **Multi-turn GRPO** - refines the policy using a composite reward combining semantic answer correctness with an asymmetric crop-cost penalty that discourages unnecessary crop calls and penalizes missed crops more heavily than over-use.

**Results.** AwaRes matches near-full-resolution accuracy (80.3\% vs.\ 80.5\% baseline average across 6 benchmarks: ChartQA, DocVQA, OCRBench, POPE, RealWorldQA, V*-Bench) while using only 36\% of visual tokens. It outperforms both fixed-budget token pruning methods and adaptive resolution escalation baselines in accuracy and efficiency, with sub-second latency vs.\ 4.3s for the closest adaptive baseline.
