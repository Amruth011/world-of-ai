<div align="center">

# ✨ PART 2 — The Generative AI Revolution

### How ChatGPT, Claude, and modern LLMs actually work. No magic — just math and data.

[← Part 1](./PART1-Foundations.md) | [Back to Home](./README.md) | [Part 3 →](./PART3-AgenticAI.md)

</div>

---

> 💡 **Building on Part 1:** You already understand neural networks and Transformers. Now we'll see what happens when you scale them up to billions of parameters and train on the entire internet.

---

## 📋 What You'll Learn in Part 2

- [1. Generative AI — The Big Picture](#1-generative-ai--the-big-picture)
- [2. Large Language Models — How They Work](#2-large-language-models--how-they-work)
- [3. Reasoning Models — The 2025-2026 Breakthrough](#3-reasoning-models--the-2025-2026-breakthrough)
- [4. Multimodal Models — Beyond Text](#4-multimodal-models--beyond-text)
- [5. Fine-Tuning and Alignment](#5-fine-tuning-and-alignment)
- [6. Model Compression and Optimization](#6-model-compression-and-optimization)

---

## 1. Generative AI — The Big Picture

### 🤔 What is Generative AI?

> **Generative AI** creates new content — text, images, audio, video, code — that didn't exist before. It doesn't retrieve or copy; it *generates*.

**Discriminative vs Generative — the key difference:**

```
Discriminative AI:   "Is this photo a cat or a dog?" → classify
Generative AI:       "Draw me a photo of a cat on the moon" → create
```

**Why 2022-2026 is the Generative AI era:**

```
2017 → Transformer architecture invented
2018 → BERT shows self-supervised learning scales
2020 → GPT-3: first model to show true emergent abilities
2022 → ChatGPT: 100M users in 60 days — fastest product in history
2023 → GPT-4, Claude 2, Gemini — race to the frontier
2024 → Multimodal becomes default, RAG and agents go mainstream
2025 → Reasoning models (o3, DeepSeek R1), agentic AI deployed in production
2026 → Agents ARE the product. Reasoning is standard. 1M context windows.
```

---

### 🗂️ Types of Generative Models

| Type | What it generates | Key models |
|---|---|---|
| **LLMs** | Text, code | GPT-4o, Claude 3.7, Gemini 2.5, LLaMA 3 |
| **Diffusion Models** | Images, video, audio | Stable Diffusion, DALL-E 3, Sora, Veo 2 |
| **GANs** | Images, video (older) | StyleGAN, BigGAN |
| **VAEs** | Images, latent representations | DALL-E 1 (used VQ-VAE) |
| **Multimodal Models** | Text + images + audio + video | GPT-4o, Gemini 2.0, Claude 3.5 Sonnet |

---

## 2. Large Language Models — How They Work

### 🤔 What is an LLM in plain English?

> An LLM is a very large Transformer model trained to predict the next token in a sequence. That's it. Everything else — conversation, coding, reasoning — is an *emergent property* of doing this at massive scale.

**The emergent abilities mystery:**
- At 1 billion parameters: can complete sentences
- At 10 billion parameters: can answer questions, summarize
- At 100 billion parameters: can reason, code, translate, explain
- Nobody designed these abilities — they emerged from scale

---

### ⚙️ How LLMs Are Trained — The 3-Stage Pipeline

```
Stage 1: PRE-TRAINING
  Input:  The entire internet + books + code + papers
  Task:   Predict the next token
  Result: A model that understands language deeply
  Cost:   $1M - $100M+ in compute
  Time:   Weeks to months on thousands of GPUs

         ↓

Stage 2: SUPERVISED FINE-TUNING (SFT)
  Input:  High-quality human-written conversations
  Task:   Learn to follow instructions helpfully
  Result: Model that's useful for chat/tasks
  Cost:   Much cheaper than pre-training

         ↓

Stage 3: RLHF / DPO — Alignment
  Input:  Human preferences (which response is better?)
  Task:   Learn to generate responses humans prefer
  Result: Helpful, harmless, honest assistant
  Cost:   Moderate — but data is expensive to collect
```

---

### 🔤 Tokenization — How LLMs See Text

> LLMs don't see words or characters — they see **tokens**, which are chunks of text.

**Why this matters:**

```python
"Hello, world!"  →  ["Hello", ",", " world", "!"]  →  [9906, 11, 1917, 0]
"ChatGPT"       →  ["Chat", "G", "PT"]              →  [14690, 38, 2849]
"AI"            →  ["AI"]                            →  [15836]
```

**Key points:**
- English averages ~1.3 tokens per word
- Numbers are split into single digits (expensive!)
- Rare words get split into subword pieces
- **Context window** = max tokens the model can "see" at once
- In 2026: GPT-4o = 128K tokens, Gemini 2.5 Pro = 1M tokens

**Tokenization methods:**
- **BPE (Byte-Pair Encoding)** — GPT models
- **WordPiece** — BERT
- **SentencePiece** — T5, LLaMA
- **tiktoken** — OpenAI's fast BPE tokenizer

---

### 🎛️ Key Generation Parameters — Controlling the Output

| Parameter | What it does | Values | Effect |
|---|---|---|---|
| **Temperature** | Controls randomness | 0 = deterministic, 2 = very random | High T: creative; Low T: focused |
| **Top-p (nucleus)** | Keep only top P% probability tokens | 0.1 - 1.0 | Lower = more conservative |
| **Top-k** | Keep only top K tokens | 1 - 100 | Lower = more focused |
| **Max tokens** | Max length of response | 1 - 128K | Hard stop on output |
| **Stop sequences** | Strings that stop generation | `["\n\n", "###"]` | Control output format |

**When to use what:**
```
Coding tasks     → Temperature 0, Top-p 0.95 (deterministic, accurate)
Creative writing → Temperature 1.0-1.3, Top-p 0.9 (varied, creative)
Factual Q&A      → Temperature 0.1-0.3 (accurate, consistent)
Brainstorming    → Temperature 1.2-1.5 (diverse outputs)
```

---

### 🧠 LLM Architecture Types

```
ENCODER-ONLY (BERT, RoBERTa)
  Input text → bidirectional attention → rich contextual representation
  Best for:   Classification, NER, semantic search, embeddings
  Training:   Masked language modeling (predict masked tokens)

DECODER-ONLY (GPT, LLaMA, Claude, Gemini)
  Input text → unidirectional attention → next token prediction
  Best for:   Text generation, conversation, code generation
  Training:   Causal language modeling (predict next token)
  Note:       All frontier LLMs in 2026 are decoder-only

ENCODER-DECODER (T5, BART)
  Input text → encoder → decoder → output text
  Best for:   Translation, summarization, question answering
  Training:   Sequence-to-sequence tasks
```

---

### 📊 The LLM Landscape in 2026

#### Frontier Closed Models (API access only)

| Model | Company | Strengths | Context |
|---|---|---|---|
| **GPT-4o** | OpenAI | Multimodal, fast, reliable | 128K |
| **o3 / o4-mini** | OpenAI | Best at math and coding (reasoning) | 200K |
| **Claude 3.7 Sonnet** | Anthropic | Coding, reasoning, long documents | 200K |
| **Gemini 2.5 Pro** | Google | Thinking, 1M context, multimodal | 1M |
| **Grok 3** | xAI | Real-time web access | 128K |
| **Command R+** | Cohere | Enterprise RAG | 128K |

#### Open Source Models (weights available)

| Model | Company | Parameters | Why it's notable |
|---|---|---|---|
| **LLaMA 3.3** | Meta | 70B | Best open source general model |
| **DeepSeek V3** | DeepSeek | 671B (MoE) | Matches GPT-4o at fraction of cost |
| **DeepSeek R1** | DeepSeek | 671B (MoE) | Open source reasoning model matching o3 |
| **Mistral Large 2** | Mistral | 123B | Strong European open weights model |
| **Qwen 2.5** | Alibaba | 7B-72B | Best multilingual open model |
| **Phi-4** | Microsoft | 14B | Best small model for its size |
| **Gemma 3** | Google | 1B-27B | Efficient, deployable on-device |

> 🔥 **The big story of 2025:** DeepSeek R1 matched OpenAI o1 performance at ~10x lower training cost, sent shockwaves through the AI industry, and proved the open-source frontier is closing rapidly.

---

### 💬 Prompt Engineering — The Art of Talking to LLMs

> **Prompt Engineering** is the skill of designing inputs to LLMs to get the best possible outputs.

#### Zero-Shot Prompting

No examples — just ask directly.

```
Prompt: "Classify this review as positive, negative, or neutral:
'The product was okay but delivery was slow.'"

Output: "Neutral"
```

#### Few-Shot Prompting

Provide examples to show the pattern.

```
Prompt: "Classify these reviews:
'Great product, fast delivery!' → Positive
'Terrible quality, broke after 1 day' → Negative
'The product was okay but delivery was slow.' → ?"

Output: "Neutral"
```

#### Chain-of-Thought (CoT)

Ask the model to reason step by step. Dramatically improves complex reasoning.

```
Without CoT:
Q: "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have?"
A: "11"  (sometimes wrong)

With CoT:
Q: "...Think step by step."
A: "Roger starts with 5 balls. He buys 2 × 3 = 6 more balls. Total: 5 + 6 = 11 balls."  ✓
```

#### System Prompts — Defining the AI's Persona

```python
system_prompt = """You are a senior Python engineer at a FAANG company.
You write clean, efficient, well-documented code.
You always explain your reasoning.
You flag potential bugs and suggest improvements."""

user_message = "Write a function to find duplicates in a list"
```

#### Structured Output — Force JSON

```python
prompt = """Extract the following information from this text and return as JSON:
- name
- age
- occupation

Text: "John Smith is a 35-year-old software engineer from London."

Return only valid JSON, no other text."""

# Output: {"name": "John Smith", "age": 35, "occupation": "software engineer"}
```

#### Tool Use / Function Calling

```python
# Define a tool
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    }
}]

# LLM decides to call the tool
response = llm.invoke("What's the weather in Bangalore?", tools=tools)
# LLM returns: {"tool": "get_weather", "args": {"location": "Bangalore", "unit": "celsius"}}
```

---

### 🧠 In-Context Learning — A Remarkable Emergent Property

> Without any weight updates, LLMs can learn new tasks just from examples in the prompt.

```
Prompt: "Translate English to French:
  sea otter → loutre de mer
  peppermint → menthe poivrée
  plush giraffe → girafe en peluche
  cheese → ?"

Output: "fromage"
```

The model never updated its weights — it learned from context. This is in-context learning.

**Why it matters:** You can adapt LLMs to new tasks instantly, without expensive fine-tuning.

---

## 3. Reasoning Models — The 2025-2026 Breakthrough

### 🤔 What are Reasoning Models?

> Reasoning models **think before they answer** — spending more compute at inference time to work through problems step by step before producing a final response.

**The key insight:** Instead of generating one answer immediately, spend tokens on *thinking* — exploring approaches, catching mistakes, trying alternatives — then produce the final answer.

```
Standard LLM:
User: "What is 17 × 23?"
Model: "391" (immediate, sometimes wrong)

Reasoning Model:
User: "What is 17 × 23?"
Model: [thinking: Let me work this out...
        17 × 20 = 340
        17 × 3 = 51
        340 + 51 = 391]
Output: "391" (verified correct)
```

---

### 📊 Reasoning Models in 2026

| Model | Company | Thinking | Best at |
|---|---|---|---|
| **o3** | OpenAI | Internal (hidden) | Math olympiad, competitive coding |
| **o4-mini** | OpenAI | Internal (hidden) | Fast + cheap reasoning |
| **Claude 3.7 Sonnet** | Anthropic | Extended thinking (visible!) | Complex coding, research |
| **DeepSeek R1** | DeepSeek | Chain-of-thought (visible) | Open source reasoning, multilingual |
| **Gemini 2.5 Pro** | Google DeepMind | Internal + visible | Very long reasoning chains |

> 🌟 **DeepSeek R1 is historic:** Open source, matches o1 on benchmarks, training cost ~$5M vs OpenAI's estimated $100M+. Shows reasoning capability can be achieved efficiently.

---

### ⚖️ When to Use Reasoning Models vs Standard Models

| Task | Use Reasoning | Use Standard |
|---|---|---|
| Complex math / olympiad problems | ✅ | |
| Multi-step code debugging | ✅ | |
| Research synthesis | ✅ | |
| Simple Q&A | | ✅ (faster, cheaper) |
| Text summarization | | ✅ |
| Simple code generation | | ✅ |
| Real-time chat | | ✅ (latency matters) |

---

## 4. Multimodal Models — Beyond Text

### 🤔 What are Multimodal Models?

> **Multimodal models** natively handle multiple types of data — text, images, audio, and video — in a single model.

In 2024-2026, **multimodal became the default**. Every frontier model is now multimodal.

---

### 🎨 Image Generation — Diffusion Models in Depth

**The process:**

```
Text prompt: "A photorealistic photo of an astronaut riding a horse on Mars"
       ↓
Text encoder (CLIP) → converts text to embedding
       ↓
Diffusion model → starts with pure noise, denoises 20-50 steps
       ↓
Image decoder → converts latent representation to pixels
       ↓
High quality image generated
```

**State of the art (2026):**
- **Stable Diffusion 3** — open source, run locally
- **DALL-E 3** — best prompt following
- **Midjourney v7** — most aesthetically pleasing
- **Ideogram 2.0** — best text rendering in images
- **Flux** — fast, high quality open source

---

### 🎬 Video Generation — 2025-2026 Explosion

| Model | Company | Max length | Quality |
|---|---|---|---|
| **Sora** | OpenAI | ~1 min | Photorealistic |
| **Veo 2** | Google | ~2 min | Photorealistic |
| **Runway Gen-3** | Runway | ~10 sec | Artistic |
| **Kling** | Kuaishou | ~2 min | High quality |
| **Pika** | Pika Labs | ~3 sec | Fast generation |

---

### 🔊 Audio and Voice Generation

| Use case | Tools | Quality |
|---|---|---|
| **Text-to-speech** | ElevenLabs, OpenAI TTS | Near-indistinguishable from human |
| **Voice cloning** | ElevenLabs, Resemble AI | Clone any voice from 30 seconds |
| **Music generation** | Suno, Udio, MusicGen | Full songs with vocals |
| **Real-time voice AI** | GPT-4o voice, Gemini Live | Sub-100ms latency |

---

## 5. Fine-Tuning and Alignment

### 🤔 Why Fine-Tune?

> Pre-trained LLMs are general-purpose. Fine-tuning specializes them for specific tasks, styles, or domains.

```
Base LLM (knows everything broadly)
       ↓ fine-tuning
Medical LLM (expert at clinical notes)
Legal LLM (expert at contract analysis)
Code LLM (expert at specific codebase)
Customer service bot (knows your product)
```

---

### 🔧 Fine-Tuning Methods

#### Full Fine-Tuning

Update ALL model weights on your dataset.

- **Pros:** Best performance, complete adaptation
- **Cons:** Extremely expensive (7B model needs 8x A100 GPUs), risk of catastrophic forgetting
- **When:** You have huge custom dataset and budget, need maximum performance

#### PEFT — Parameter-Efficient Fine-Tuning

Update only a **small fraction** of weights. Much cheaper, almost as good.

##### LoRA (Low-Rank Adaptation) — The Most Popular Method

**The idea:**

```
Original weight matrix W (huge, frozen)
       ↓
Add two small matrices A and B where: ΔW = A × B
       
Instead of updating W (7B parameters), only update A and B (~millions of parameters)

Total trainable params: <1% of full model
Performance: 95-99% of full fine-tuning
```

```python
# Using LoRA with HuggingFace
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,           # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1
)
model = get_peft_model(base_model, config)
# Now only ~16M params are trainable instead of 7B
```

##### QLoRA — Fine-Tune on Consumer Hardware

**The idea:** Quantize the base model to 4-bit (saves 75% memory), then train LoRA adapters on top.

**Practical impact:** Fine-tune a 7B model on a single RTX 3090 (24GB VRAM). Previously required 8x A100s.

##### Other PEFT Methods

| Method | How it works | When to use |
|---|---|---|
| **DoRA** | LoRA with separate magnitude and direction | Better than LoRA on some tasks |
| **Prefix Tuning** | Learn soft prompts prepended to input | NLP classification tasks |
| **Adapter Layers** | Small trainable layers inserted between frozen layers | Modular, swappable adapters |

---

### 🎯 Alignment — Making LLMs Helpful and Safe

> **Alignment** = training AI systems to behave the way humans intend — helpful, honest, and harmless.

#### RLHF — Reinforcement Learning from Human Feedback

The method that turned GPT-3 into ChatGPT.

```
Step 1: Collect demonstrations
  Human writes ideal responses to 1000s of prompts

Step 2: Train reward model
  Humans rank multiple responses (A > B > C)
  Train a model to predict human preferences

Step 3: RL fine-tuning (PPO)
  Use the reward model as feedback signal
  Fine-tune the LLM to maximize human preference scores

Result: Model that gives responses humans actually want
```

#### DPO — Direct Preference Optimization

> **Simpler than RLHF, widely adopted in 2025.**

Instead of training a separate reward model, directly optimize on preference pairs:

```python
# For each prompt, you have:
preferred_response = "Here's a helpful explanation..."
rejected_response  = "I cannot help with that."

# DPO directly trains the model to prefer the better response
# No separate reward model needed
# More stable training, simpler pipeline
```

**Used by:** Mistral, Meta LLaMA 3, many open source models

#### GRPO — Group Relative Policy Optimization

> DeepSeek R1's secret weapon for reasoning.

Instead of comparing to a reward model, compare multiple responses generated by the model itself:

```
Generate N responses to a math problem
Grade each response (correct/incorrect)
Train the model to increase probability of correct responses
      relative to the group average

No reward model, no reference model, simpler and cheaper
```

#### Constitutional AI (Anthropic)

> Train the model using AI feedback against a set of principles.

```
Principle: "Be honest. Don't claim to know things you don't know."

Step 1: Model generates response
Step 2: Critique model evaluates: "Does this violate the principle?"
Step 3: Model revises based on critique
Step 4: Use revised responses as training data

Result: Model internalizes the principles
```

---

## 6. Model Compression and Optimization

### 🤔 Why Compress Models?

> A 70B parameter model needs 140GB of RAM to run at full precision. That's expensive. Compression makes models smaller and faster with minimal quality loss.

---

### 📦 Quantization — Smaller Numbers, Smaller Model

> Replace 32-bit (FP32) or 16-bit (FP16) numbers with smaller ones.

```
FP32:  32 bits per number  → 7B model = 28GB
FP16:  16 bits per number  → 7B model = 14GB
INT8:   8 bits per number  → 7B model =  7GB
INT4:   4 bits per number  → 7B model = 3.5GB
```

**The tradeoff:** Lower precision = smaller model + faster inference + slight quality loss

**Popular formats:**

| Format | Bits | Who uses it | Notes |
|---|---|---|---|
| **GGUF** | 2-8 bit | llama.cpp, Ollama | Run LLMs on CPU/Mac |
| **AWQ** | 4 bit | Fast GPU inference | Better quality than naive INT4 |
| **GPTQ** | 4 bit | GPU serving | Post-training quantization |
| **BF16** | 16 bit | Training standard | Best quality/speed tradeoff |

---

### ✂️ Pruning — Remove the Unnecessary

> Find and remove weights that don't contribute much to the output.

```
Unstructured pruning: Set individual weights to 0
Structured pruning:   Remove entire neurons, heads, or layers

Result: Sparser model that runs faster
Challenge: Can hurt quality significantly if done aggressively
```

---

### 👨‍🏫 Knowledge Distillation — Small Student, Large Teacher

> Train a small model (student) to mimic a large model (teacher).

```
Teacher: GPT-4 (1T+ parameters, expensive)
       ↓ generates outputs on many inputs
Student: Phi-4 14B (trained to match teacher outputs)

Result: Phi-4 gets GPT-4-level behavior on many tasks at 1% of the size

Real example: DeepSeek R1-Distill series — small models with R1's reasoning ability
```

---

### ⚡ Speculative Decoding — 2-3x Faster Generation

> Use a small fast model to draft tokens, then verify with the large model.

```
Draft model (small, fast) generates 5 tokens quickly
Large model (slow, accurate) verifies all 5 in one pass

If all correct: save 5 model calls, gained speed
If wrong: fall back to large model for correction

Net effect: 2-3x faster generation with identical output quality
```

---

### 🧩 MoE — Mixture of Experts

> Instead of using the entire model for every token, only activate the relevant "expert" sub-networks.

```
Traditional dense model:
  Input token → ALL 70B parameters activated → output

MoE model (e.g., Mixtral 8×7B):
  Input token → routing network decides which 2 of 8 experts to use
              → only 14B parameters activated → output

Total parameters: 56B (8 × 7B)
Active parameters: 14B per token
Speed: Same as 14B model
Quality: Close to 56B model
```

**Models using MoE:** Mixtral, GPT-4 (rumored), Gemini (multiple versions), DeepSeek V3

---

## ✅ Part 2 Summary

You now understand:

- ✅ What Generative AI is and why 2022-2026 is its era
- ✅ How LLMs are trained — pre-training, SFT, RLHF
- ✅ Tokenization, context windows, and generation parameters
- ✅ Prompt engineering — zero-shot, few-shot, CoT, structured output, function calling
- ✅ Reasoning models and why DeepSeek R1 shocked the AI industry
- ✅ Multimodal models — images, video, audio generation
- ✅ Fine-tuning — full, LoRA, QLoRA, adapters
- ✅ Alignment — RLHF, DPO, GRPO, Constitutional AI
- ✅ Model compression — quantization, pruning, distillation, MoE, speculative decoding

---

<div align="center">

**This is where most people stop. But the real value is in what comes next.**

## [→ Continue to Part 3: Agentic AI — The Frontier](./PART3-AgenticAI.md)

*"LLMs that just answer questions are 2023. In 2026, AI agents plan, reason, and act. Let's build them."*

---

[← Part 1](./PART1-Foundations.md) | [Back to Home](./README.md) | [Part 3 →](./PART3-AgenticAI.md)

</div>
