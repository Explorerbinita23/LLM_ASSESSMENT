# Section 1: Diagnose a Failing LLM Pipeline

## Scenario

A customer support chatbot powered by GPT-4o performed well during testing, but after launch users reported:

1. Confidently incorrect answers about product pricing  
2. Occasional responses in English when users wrote in Hindi or Arabic  
3. Latency increasing from 1.2s to 8–12s over two weeks  

No prompt or model changes were made after launch.

---

# Problem 1: Hallucinated Product Pricing

## Diagnosis Log

## Step 1: Reproduce the Failure

My first step would be to reproduce the issue using representative pricing queries such as:

- What is Product A price?
- Is Product B discounted?
- What is the annual subscription fee?
- What is the latest premium plan cost?

I would test multiple times and record:

- Whether answers are consistent
- Whether answers are confident despite uncertainty
- Whether answers cite any retrieved source
- Whether different phrasings produce different prices

This helps determine whether the issue is deterministic, retrieval-related, or randomness-related.

---

## Step 2: Determine Whether Pricing Comes From Retrieval or Model Memory

Pricing is dynamic business data and should usually come from a trusted source such as:

- Pricing database
- Product API
- Updated pricing documents
- Internal knowledge base

I would inspect the pipeline for:

- Retrieved documents attached to the prompt
- Timestamp of indexed pricing documents
- Whether pricing tables are actually present in retrieved context
- Cases where no relevant documents were retrieved but the model still answered

If the model answers without verified pricing context, hallucination risk is high.

---

## Step 3: Rule Out Other Causes

### Prompt Issue

I would inspect whether the prompt contains language such as:

- “Always answer helpfully”
- “Do your best even if information is missing”

This can unintentionally encourage guessing.

### Temperature Issue

If temperature is high, responses may vary.  
I would compare repeated outputs at:

- Temperature = 0.0
- Temperature = 0.2
- Temperature = 0.7

If wrong answers persist even at low temperature, temperature is not the root cause.

### Knowledge Cutoff Issue

If the model gives historically old prices, knowledge cutoff may contribute.  
However, pricing should not rely on pretrained memory in production.

---

## Root Cause Assessment

Most likely primary cause: **Retrieval issue**

Likely contributing cause: **Knowledge cutoff**

Reason:

- Pricing changes frequently
- Model should not infer commercial values from memory
- Missing or stale retrieval commonly leads to confident wrong pricing

---

## Concrete Fix

## Immediate Fix

Route pricing questions to a real-time source of truth:

```text
User asks pricing
→ Detect pricing intent
→ Query pricing API / database
→ LLM formats answer only

### Prompt Guardrail

If verified pricing data is not provided in context,
state that current pricing is unavailable rather than guessing.
Monitoring

Track:
Pricing hallucination rate
% pricing answers backed by source data
Retrieval hit rate for pricing intents

# Problem 2: Language Switching (Hindi / Arabic → English)

## Diagnosis Log
### Step 1: Reproduce

I would test multilingual inputs such as:

Hindi question
Arabic question
Mixed Hindi-English message
Multi-turn conversation with language switch

Goal:

Measure how often the assistant defaults to English
Identify whether failure happens on first turn or later turns

# Problem 2: Language Switching (Hindi / Arabic → English)

## Diagnosis Log

### Step 1: Reproduce the Failure

My first step would be to reproduce the issue using multilingual user inputs such as:

- Hindi query asking product information  
- Arabic query asking account support  
- Mixed Hindi-English message  
- Multi-turn conversation where user switches language

I would observe:

- Whether the response matches the user’s language  
- Whether the first reply is correct but later replies switch to English  
- Whether certain languages fail more often than others  
- Whether responses are partially translated

This helps determine whether the issue is prompt-related, memory-related, or language detection-related.

---

### Step 2: Inspect Prompt Hierarchy

Most LLM chat systems follow instruction priority:

```text
System Prompt > Developer Prompt > User Message
A common failure mode is a system prompt like:

You are a professional support assistant. Respond clearly in English.

Even when users write in Hindi or Arabic, the higher-priority system instruction may override user language preference.

Root Cause Mechanism

The model follows the strongest instruction available.
If English is explicitly or implicitly prioritized in the system prompt, multilingual behavior becomes unreliable.

Concrete Fix (Specific + Testable + Language-Agnostic)

Replace the system prompt with:

You are a multilingual customer support assistant.

Always reply in the same language as the user's latest message.

If the user switches language, switch accordingly.

Do not default to English unless the user writes in English.

Preserve the user's script, tone, and formality.
Additional Reliability Layer

Use automatic language detection before inference:

Detected language: Arabic
Respond only in Arabic.
Monitoring

Track:

Language mismatch rate
Response language vs user language
Failure rate by language

# Problem 3: Latency Increased from 1.2s to 8–12s
 Diagnosis Log

Since no prompt or model changes were made, I would investigate scaling and infrastructure causes rather than model regression.

Step 1: Check Prompt Size Growth

As usage grows, conversation history often becomes longer. Larger prompts increase inference time.

I would inspect:

Average prompt tokens by week
Average completion tokens by week
Average conversation turns per session

If token counts steadily increased, latency growth is expected.

Step 2: Check Retrieval Layer Performance

If the chatbot uses RAG, a growing document index may slow retrieval over time.

I would inspect:

Retrieval latency percentiles
Number of indexed documents
Query throughput
Cache hit rate

Slow retrieval often adds several seconds before model inference begins.

Step 3: Check API Queueing / Rate Limits

As user traffic grows, external LLM providers may throttle requests or queue them.

I would inspect:

Requests per minute trend
Provider response times
Retry counts
HTTP 429 / timeout errors
Distinct Causes That Could Produce This Pattern
Cause 1: Prompt Token Growth

Longer chat history increases processing time.

Cause 2: Retrieval Slowdown

Larger vector databases or poor indexing increase search latency.

Cause 3: Increased Traffic / Queueing

Higher usage causes API delays or throttling.

Additional Possible Causes
Cold starts
Autoscaling lag
Database contention
Network retries
Logging overhead
What I Would Investigate First
First Priority: Prompt Token Growth

Reason:

Very common in chatbots
Happens with no code changes
Easy to verify using logs
Directly affects model latency
Concrete Fixes
If Prompt Growth
Keep only recent turns
Summarize older conversation history
Use memory compression
If Retrieval Slowdown
Use ANN indexing (FAISS / HNSW)
Add metadata filtering
Cache common results
If API Queueing
Use async requests
Increase provider tier
Add fallback provider
Monitoring

Track:

P50 / P95 / P99 latency
Prompt tokens per request
Retrieval latency
API wait time
Error / retry rate

###Post-Mortem Summary (Non-Technical Stakeholders)

Three separate issues caused the chatbot decline after launch. First, incorrect pricing responses were likely caused by the assistant answering from incomplete or outdated information instead of a live pricing source. We will connect all pricing questions directly to verified pricing systems so answers remain accurate. Second, occasional English replies to Hindi or Arabic users were caused by internal instructions that unintentionally prioritized English. We have updated the assistant so it always responds in the customer’s language. Third, slower response times were caused by growth-related scaling issues such as longer conversations, larger search indexes, and increased traffic. We are reducing prompt size, optimizing search speed, and improving capacity management. No evidence suggests the core model itself degraded. These changes will improve accuracy, language consistency, and response speed.