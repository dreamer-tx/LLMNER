# LLMNER A Named Entity Recognition (NER) tool and experimental repository based on Large Language Models (LLM).

---

## Project Introduction
LLMNER is an experimental repository for exploring and evaluating the performance of Large Language Models in Named Entity Recognition tasks. It includes prompt-based methods , implementations of LLM-based classifiers, and auxiliary training/evaluation scripts.

---

## Repository Structure (Example)
text
LLMNER/
├── data/                # Example data, dataset conversion scripts
├── llm-cls/             # LLM-based classification/inference implementation
├── train/               # Scripts related to training or fine-tuning (if any)
├── chat/                # Prompt templates, interaction scripts
├── result/              # Output predictions and evaluation results
├── requirements.txt     # Python dependencies
├── README.md            # This file
---

## Installation
It is recommended to use a virtual environment to isolate dependencies:
bash
git clone https://github.com/dreamer-tx/LLMNER.git 
cd LLMNER
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
---

## Evaluation Metrics
It is recommended to calculate and save the following metrics:
* Global: Precision, Recall, F1 (micro / macro)
* By entity type: Precision / Recall / F1 for each type

---

## Scalable Directions
* Add more prompt templates and conduct automated comparisons.
* Support multi-language or domain-specific (medical, legal) entity sets.
