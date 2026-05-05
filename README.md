# Domain-Adaptive RAG: Does Continued Pre-Training of the Retriever Help?

**CS505: Natural Language Processing - Final Project**
**Boston University, Spring 2026**
**Akhil Elamanchili (aelamanc@bu.edu)**

## Overview

This project investigates whether continued pre-training of a dense passage retriever on domain-specific text improves end-to-end retrieval-augmented generation (RAG) performance in the biomedical domain. Using PubMedQA with an expanded 11,000-passage corpus, I compare four retrieval strategies feeding a Llama 3.2 3B generator:

- **BM25** — non-neural keyword baseline
- **Contriever** — general-purpose dense retriever (frozen)
- **Contriever + DAPT** — continued contrastive pre-training on PubMed abstracts
- **Contriever + LoRA** — parameter-efficient adaptation via low-rank adapters

### Key Findings

| Retriever | Recall@5 | MRR | Accuracy |
|---|---|---|---|
| BM25 | 0.924 | 0.876 | 0.585 |
| Contriever (general) | 0.793 | 0.697 | 0.575 |
| Contriever + DAPT | 0.949 | 0.893 | 0.595 |
| Contriever + LoRA | **0.947** | **0.880** | **0.645** |

- General Contriever collapses on the harder 11K corpus vs. BM25 (−13.1 Recall@5)
- Both domain-adapted variants recover and surpass BM25
- LoRA achieves near-identical retrieval to full DAPT while training only ~1% of parameters and in ~25 min vs. ~45 min
- LoRA produces stronger end-to-end accuracy (+6% over BM25)

---

## Repository Structure

```
├── notebooks/
│   ├── pubmedqa_baselines.ipynb         # BM25 + general Contriever baselines (11K corpus)
│   ├── contriever_dapt.ipynb            # Continued pre-training on PubMed abstracts
│   ├── contriever_lora.ipynb            # LoRA adaptation of Contriever
│   └── rag_evaluation.ipynb             # End-to-end RAG with Llama 3.2 3B
├── results/
│   ├── baseline_results_expanded.json   # BM25 + Contriever retrieval metrics
│   ├── dapt_results.json                # DAPT retrieval metrics
│   ├── lora_results.json                # LoRA retrieval metrics
│   ├── all_results.json                 # Full results (retrieval + accuracy)
│   └── error_analysis.json             # Failure type counts by retriever
└── README.md
```

---

## Setup

All notebooks are designed to run on **Google Colab with a T4 GPU**.

### Dependencies

Each notebook installs its own dependencies in the first cell. The core packages are:

```
datasets
transformers
torch==2.1.0
faiss-cpu
rank_bm25
peft
numpy
tqdm
```

### HuggingFace Access

Notebook 4 (RAG evaluation) requires access to **Llama 3.2 3B-Instruct**, which is a gated model. Request access at:
[https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

Then add your HuggingFace token in the login cell.

---

## Running the Notebooks

Run the notebooks in the order listed below. Each saves outputs to Google Drive (`/content/drive/MyDrive/cs505_project/`) for persistence across sessions.

### Notebook 1 — Baselines (`pubmedqa_baselines.ipynb`)
- Loads PubMedQA labeled split (1,000 queries)
- Builds expanded corpus with 10,000 distractor abstracts from PubMedQA artificial split
- Evaluates BM25 and general Contriever
- Saves: `baseline_results_expanded.json`, `corpus_embeddings_expanded.npy`, `query_embeddings.npy`

### Notebook 2 — DAPT (`contriever_dapt.ipynb`)
- Continues Contriever's contrastive pre-training on ~211K PubMed abstracts
- Random cropping objective: 64-token spans, temperature τ=0.05
- Hyperparameters: batch_size=64, lr=1e-5, 3 epochs, 500 steps/epoch
- Saves: `contriever_dapt_best/` checkpoint, `dapt_results.json`, `corpus_emb_dapt.npy`

### Notebook 3 — LoRA (`contriever_lora.ipynb`)
- Applies LoRA to Contriever attention layers (r=16, α=32, target: query+value)
- Same contrastive objective as DAPT, lr=1e-4, ~1.2M trainable parameters
- Saves: `contriever_lora_best/` checkpoint, `lora_results.json`, `corpus_emb_lora.npy`

### Notebook 4 — RAG Evaluation (`rag_evaluation.ipynb`)
- Loads all retrieval results and pre-computed embeddings from Drive
- Runs RAG pipeline: retrieve top-5 passages → Llama 3.2 3B-Instruct → yes/no/maybe
- Evaluates on 200-query sample (~4.5s per query on T4)
- Computes NLI-based faithfulness and error analysis by failure type
- Saves: `all_results.json`, `error_analysis.json`, per-retriever prediction files

---

## Methods

### Corpus Construction

The original PubMedQA corpus (1,000 passages) is trivially easy for any retriever. I augment it with 10,000 distractor abstracts from the PubMedQA artificial split, creating an 11,000-passage corpus that forces the retriever to discriminate among topically similar biomedical documents.

### Contrastive Pre-Training Objective

Following Contriever's original training procedure, positive pairs are formed by randomly cropping two 64-token spans from the same document. The in-batch NT-Xent loss with temperature τ=0.05 is:

$$\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{s(e_i^1, e_i^2)/\tau}}{\sum_{j=1}^{N}e^{s(e_i^1, e_j^2)/\tau}} + \log\frac{e^{s(e_i^2, e_i^1)/\tau}}{\sum_{j=1}^{N}e^{s(e_i^2, e_j^1)/\tau}}\right]$$

where $s(\cdot, \cdot)$ is cosine similarity and $N$ is the batch size.

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Target modules | query, value |
| Dropout | 0.1 |
| Trainable params | ~1.2M / 110M (1.1%) |
| Learning rate | 1e-4 |

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{elamanchili2026domainadaptiverag,
  title  = {Domain-Adaptive {RAG}: Does Continued Pre-Training of the Retriever Help?},
  author = {Elamanchili, Akhil},
  year   = {2026},
  note   = {CS505 Final Project, Boston University}
}
```

---

## References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Izacard et al. (2022). Unsupervised Dense Information Retrieval with Contrastive Learning. *TMLR*.
- Gururangan et al. (2020). Don't Stop Pretraining. *ACL*.
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
- Wang et al. (2022). GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval. *NAACL*.
- Jin et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. *EMNLP*.