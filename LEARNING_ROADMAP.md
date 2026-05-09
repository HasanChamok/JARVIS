# JARVIS Learning Roadmap
## Everything you need — Beginner to Pro

---

## PHASE 0 — Python Foundations (Week 1–2)
*If you're new to Python, start here. Skip if comfortable.*

### Core concepts to learn
1. Variables, data types (int, float, str, list, dict)
2. Functions — def, return, arguments
3. Classes — __init__, self, methods, inheritance
4. File I/O — open(), read(), write(), JSON
5. Error handling — try/except
6. Modules and imports

### Best resources (free)
- **Python.org tutorial**: https://docs.python.org/3/tutorial/
- **Automate the Boring Stuff** (free online): https://automatetheboringstuff.com
- **freeCodeCamp Python course** (YouTube): search "freeCodeCamp Python full course"

### Book
- *"Python Crash Course"* — Eric Matthes (best beginner book, very practical)

---

## PHASE 1 — Audio & numpy (Week 2–3)
*Understanding how sound works as numbers.*

### Key concepts
- Sample rate — what 16000 Hz means
- numpy arrays — shape, dtype, operations
- Audio as float32 arrays — amplitude, volume
- VAD — Voice Activity Detection
- Zero-crossing rate for noise filtering

### Resources
- numpy documentation: https://numpy.org/doc/stable/user/quickstart.html
- **Book**: *"Python for Audio Signal Processing"* — search on GitHub for free notebooks
- **Research Paper**: "A comparative study of Voice Activity Detection algorithms"
  (Search Google Scholar — understanding VAD is essential)

### Practice project
Visualise your own voice: record 5 seconds, plot the waveform with matplotlib.
```python
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
audio = sd.rec(5*16000, samplerate=16000, channels=1, dtype='float32')
sd.wait()
plt.plot(audio)
plt.show()
```

---

## PHASE 2 — Threading & Concurrency (Week 3–4)
*Making things happen simultaneously.*

### Key concepts
- threading.Thread — creating and starting threads
- threading.Lock — preventing collisions
- threading.Event — signalling between threads
- queue.Queue — safe data passing between threads
- daemon threads — what they are and why they matter
- The GIL — Python's Global Interpreter Lock (and why it doesn't matter for JARVIS)

### Resources
- Python threading docs: https://docs.python.org/3/library/threading.html
- **Book**: *"Python Concurrency with asyncio"* — Matthew Fowler
  (covers threading + async, very practical)
- **Article**: "Understanding Python's GIL" — Real Python (realpython.com/python-gil)

### Practice project
Build a stopwatch that displays time in the terminal while listening for "stop":
Two threads — one counting, one reading keyboard input.

---

## PHASE 3 — GPU & PyTorch (Week 4–6)
*Understanding your RTX 4050 and how AI uses it.*

### Key concepts
- What a tensor is (like numpy array but GPU-aware)
- CUDA — NVIDIA's GPU programming platform
- float16 vs float32 vs int8 — precision vs VRAM tradeoffs
- Loading models to GPU
- Memory management — torch.cuda.empty_cache()
- Batch processing — why GPUs love batches

### Resources
- PyTorch official tutorial: https://pytorch.org/tutorials/beginner/basics/intro.html
- **Book**: *"Deep Learning with PyTorch"* — Eli Stevens (free PDF on Manning)
- **Fast.ai course** (free): https://course.fast.ai — hands-on, GPU-first approach

### Research Papers (start here)
1. *"Attention Is All You Need"* — Vaswani et al. 2017
   The paper that invented transformers. Read abstract + architecture section.
   Link: https://arxiv.org/abs/1706.03762

2. *"BERT: Pre-training of Deep Bidirectional Transformers"* — Devlin et al. 2018
   Link: https://arxiv.org/abs/1810.04805

### Practice project
Move a tensor to GPU, do matrix multiplication, measure time vs CPU:
```python
import torch, time
a = torch.randn(1000, 1000)
b = torch.randn(1000, 1000)
# CPU
t0 = time.time(); c = a @ b; print(f"CPU: {time.time()-t0:.4f}s")
# GPU
a, b = a.cuda(), b.cuda()
t0 = time.time(); c = a @ b; torch.cuda.synchronize()
print(f"GPU: {time.time()-t0:.4f}s")
```

---

## PHASE 4 — Speech & NLP (Week 6–8)
*How machines understand language.*

### Key concepts
- Tokenisation — how text becomes numbers
- Embeddings — words as vectors in space
- Attention mechanism — the core of transformers
- Encoder vs decoder models
- Whisper architecture — CTC + attention for ASR
- Temperature, top-p, top-k — LLM generation parameters

### Resources
- **"The Illustrated Transformer"**: https://jalammar.github.io/illustrated-transformer/
  (Best visual explanation of attention — read this before the Vaswani paper)
- **"The Illustrated BERT"**: https://jalammar.github.io/illustrated-bert/
- Andrej Karpathy "Let's build GPT": https://www.youtube.com/watch?v=kCc8FmEb1nY
  (Builds a transformer from scratch — 2 hours, worth every minute)

### Research Papers
3. *"Robust Speech Recognition via Large-Scale Weak Supervision"* — Whisper paper
   Link: https://arxiv.org/abs/2212.04356

4. *"Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"*
   Link: https://arxiv.org/abs/1712.05884

---

## PHASE 5 — RAG & Vector Databases (Week 8–10)
*Giving JARVIS long-term memory about your life.*

### Key concepts
- Embeddings for semantic search
- Cosine similarity — measuring meaning distance
- Chunking strategies — how to split documents
- ChromaDB — local vector database
- sentence-transformers — efficient local embedding models
- Hybrid search — combining vector + keyword

### Resources
- ChromaDB docs: https://docs.trychroma.com
- sentence-transformers: https://www.sbert.net
- **Article**: "Building RAG systems" — LangChain blog

### Research Papers
5. *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* — Lewis et al.
   The original RAG paper.
   Link: https://arxiv.org/abs/2005.11401

6. *"Dense Passage Retrieval for Open-Domain Question Answering"* — Karpukhin et al.
   Link: https://arxiv.org/abs/2004.04906

### Practice project
Index your Documents folder, then ask JARVIS questions about your own files.

---

## PHASE 6 — Fine-tuning (Month 3–4)
*Making JARVIS sound and think like you.*

### Key concepts
- Transfer learning — why fine-tuning works
- LoRA — Low-Rank Adaptation (trains 0.1% of parameters)
- QLoRA — quantised LoRA (fits on your RTX 4050)
- GGUF format — quantised model files for Ollama
- Instruction format — how to format your training data
- Overfitting — how to avoid training too much

### Tools
- **Unsloth**: https://github.com/unslothai/unsloth
  Fastest LoRA fine-tuning library, optimised for consumer GPUs
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl
  More features, more config options
- **Ollama Modelfile**: for loading your fine-tuned model

### Research Papers
7. *"LoRA: Low-Rank Adaptation of Large Language Models"* — Hu et al. 2021
   Link: https://arxiv.org/abs/2106.09685

8. *"QLoRA: Efficient Finetuning of Quantized LLMs"* — Dettmers et al. 2023
   Link: https://arxiv.org/abs/2305.14314

### Data collection
JARVIS already saves every conversation to data/memory.json.
After 2–3 months of daily use, export it and format for fine-tuning:

```python
import json

memory = json.load(open("data/memory.json"))
turns  = memory["turns"]
training_data = []

for i in range(0, len(turns) - 1, 2):
    if turns[i]["role"] == "user" and turns[i+1]["role"] == "assistant":
        training_data.append({
            "instruction": "You are JARVIS, a personal AI assistant.",
            "input":  turns[i]["content"],
            "output": turns[i+1]["content"],
        })

json.dump(training_data, open("training_data.json","w"), indent=2)
print(f"Generated {len(training_data)} training pairs")
```

---

## PHASE 7 — Future GPU Upgrade (When ready)
*When you add more VRAM — what changes.*

### What a GPU upgrade unlocks
- Bigger Whisper model (medium/large) → better accuracy
- Larger LLM (7B, 13B, 70B parameters) → smarter responses
- Faster inference → lower latency
- Vision models (LLaVA) → JARVIS can see your screen
- Multiple simultaneous models

### Config changes needed (just edit config.py)
```python
# After upgrading:
WHISPER_MODEL = "large-v3"     # from "base.en"
OLLAMA_MODEL  = "llama3.1:70b" # from "llama3.2"
COMPUTE_TYPE  = "float16"      # stays the same
```

JARVIS is already designed for this — the config.py approach means
you change 2 lines and get a more powerful system instantly.

---

## Book Reading Order
1. *Python Crash Course* — Eric Matthes (foundations)
2. *Automate the Boring Stuff* — Al Sweigart (free, practical Python)
3. *Deep Learning with PyTorch* — Eli Stevens (GPU + tensors)
4. *Python Concurrency with asyncio* — Matthew Fowler (threading)
5. *Designing Machine Learning Systems* — Chip Huyen (big picture)
6. *The Hundred-Page Machine Learning Book* — Andriy Burkov (theory, compact)

---

## YouTube Channels (watch in this order)
1. **Andrej Karpathy** — builds AI from scratch, no fluff
2. **Yannic Kilcher** — explains research papers clearly
3. **Sentdex** — practical Python + ML projects
4. **Two Minute Papers** — latest research, accessible
5. **Machine Learning Street Talk** — deep technical interviews

---

## How to read a research paper (as a beginner)
1. Read the **abstract** — what's the problem, what's the solution?
2. Look at all the **figures and tables** — visuals tell you 80% of the story
3. Read the **introduction** — context and motivation
4. Skip the **math** on first pass — come back later
5. Read the **results section** — what did they actually achieve?
6. Read the **conclusion**
7. Then tackle the **method section** once you understand the goal

You don't need to understand every equation to learn from papers.
The intuition comes first. The math comes later.

---

*Updated as JARVIS grows. Add your own notes here.*
