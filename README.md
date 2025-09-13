````markdown
# LLMNER

An experimental repository for **Named Entity Recognition (NER)** based on **Large Language Models (LLMs)**.

---

## Introduction

LLMNER is an experimental repository designed to explore and evaluate the performance of large language models on NER tasks.  
It includes prompt-based methods (zero-shot/few-shot inference), LLM-based classifier wrappers, as well as auxiliary training and evaluation scripts.

---

## Repository Structure (Example)

```text
LLMNER/
├── data/                # Example datasets, dataset conversion scripts
├── llm-cls/             # LLM-based classification/inference implementations
├── train/               # Training or fine-tuning related scripts (if any)
├── chat/                # Prompt templates, interactive scripts
├── result/              # Prediction outputs and evaluation results
├── requirements.txt     # Python dependencies
├── README.md            # This file
````

---

## Installation

It is recommended to use a virtual environment to isolate dependencies:

```bash
git clone https://github.com/dreamer-tx/LLMNER.git
cd LLMNER
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Evaluation Metrics

It is recommended to compute and save the following metrics:

* Global: Precision, Recall, F1 (micro / macro)
* Per entity type: Precision / Recall / F1 for each class

---

## Possible Extensions

* Add more prompt templates and perform automated comparisons.
* Support multilingual or domain-specific (medical, legal) entity sets.

---

