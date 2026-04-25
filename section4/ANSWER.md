Question B — Evaluating LLM Output Quality

If my manager asked whether our summarisation model is performing well, I would avoid answering based on intuition and instead set up a structured evaluation process.

First, I would create a reliable benchmark dataset. I’d collect around 300–500 internal reports from different teams such as finance, HR, operations, and legal, so the dataset reflects real usage. For each report, I’d ask human reviewers to write high-quality reference summaries. They would follow a clear rubric covering factual accuracy, completeness, clarity, and brevity. For important samples, I’d use two reviewers and resolve disagreements to improve consistency.

Next, I’d use automated metrics to measure quality at scale. I would track ROUGE scores to compare overlap with reference summaries and BERTScore to measure semantic similarity. ROUGE is useful for monitoring trends, but it can unfairly penalise summaries that use different wording. BERTScore handles paraphrasing better, but neither metric guarantees factual correctness. Because of that, I’d also run factual checks for critical details like names, dates, numbers, and action items.

Human evaluation is still essential. Every week, I’d randomly sample outputs and ask reviewers to score them on usefulness, readability, missing information, and hallucinations. Hallucination rate would be one of the most important metrics, since a fluent but incorrect summary can damage trust.

To detect regressions when prompts or models change, I’d maintain a frozen benchmark set and compare every new version against the current production model. If factual accuracy drops or hallucinations rise beyond a threshold, I would block deployment. I’d also run A/B tests on live traffic before full rollout.

For non-technical stakeholders, I’d simplify everything into a dashboard showing summary quality, factual error rate, average processing time, and user satisfaction. For example: “The new version is 10% more accurate, 25% faster, and reduced factual mistakes from 5% to 2%.”

Question C — On-Premise LLM Deployment

For a defence or government client requiring fully offline AI inference, my first priority would be security, reliability, and predictable performance. Since the hardware is a single server with 2× NVIDIA A100 80GB GPUs, there is enough capacity to run strong open-source models locally.

I would start by benchmarking a few practical candidates such as Meta Llama 3 8B Instruct, Meta Llama 3 70B Instruct, and Mistral AI Mistral 7B Instruct. My default starting point would be an 8B model, because it usually gives the best balance between quality and latency. If the task required deeper reasoning, I would then test a larger 70B model.

For memory sizing, an 8B model in FP16 uses roughly 16 GB just for weights, while a 70B model needs around 140 GB, so it would need both GPUs with tensor parallelism. With INT4 quantisation, memory usage drops significantly, making deployment easier and faster. I would test AWQ or GPTQ quantisation because they usually preserve quality well.

For serving, I would prefer vLLM because it is efficient, supports batching, and performs well with modern GPUs. If maximum optimisation was required, I would also benchmark TensorRT-LLM.

For the required latency of under 3 seconds with a 500-token input, an 8B quantised model should comfortably meet the target, often responding in around 1–2 seconds depending on output length. A 70B model may still meet the target, but with less headroom.

My deployment setup would include containerised local inference, internal-only network access, local logging, health monitoring, and failover restart scripts. If concurrency increased later, I would likely choose a smaller model with retrieval augmentation rather than a slower large model.