# Abductive Event Reasoning: Towards Real-World Event Causal Inference for Large Language Models

This repository contains the codebase and research materials for our project on **Abductive Event Reasoning (AER)**, focused on SemEval-2026 Task-12.

The goal of this project is to improve the capability of Large Language Models (LLMs) to identify the direct, root cause of an event using real-world background documents. Since real-world events rarely happen in isolation, the challenge lies in distinguishing actual structural causes from mere contextual distractors.

## Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Authors](#authors)

## Overview
LLMs typically excel at pattern recognition but falter at causal inference in noisy environments. Our approach mitigates this by applying a dual-stage Retrieval-Augmented Generation (RAG) pipeline to filter out irrelevant information, coupled with efficient parameter-efficient fine-tuning (PEFT) on Llama 3.2 models (3B and 8B). 

## Methodology

### 1. Dual-Stage Document Filtering
The context windows for background documents were extensively large and noisy. We processed this by:
- **Lexical Search**: Splitting raw text into ~300-character chunks and using **BM25s** to find the top 10 relevant blocks.
- **Semantic Reranking**: Utilizing Faiss (`sentence-transformers/all-MiniLM-L6-v2`) to rank these blocks, eventually combining the top 4 most relevant chunks with 2 distractor chunks to enhance the model's resilience to noise.

### 2. Few-Shot Inference
We evaluated Llama 3.2 8B using zero, two, five, and eight-shot prompting. Results indicated that models tend to over-predict and get distracted by background contexts, with performance plateauing around the 5-shot mark.

### 3. Fine-tuning
We fine-tuned the Llama 3.2 3B and 8B models using [Unsloth](https://github.com/unslothai/unsloth) with 4-bit quantization (LoRA/QLoRA) on Google Colab. Fine-tuning allowed the model to internalize the task's causal logic instead of relying strictly on contextual prompt patterns, successfully mitigating output format and hallucination problems.

## Repository Structure

- `dataset/`: Directory containing the target events, documents, and QA JSON formats (`train_data`, `dev_data`, `sample_data`).
- `zero_shot.ipynb`: Notebook exploring zero-shot and few-shot inference pipelines using FAISS and BM25 document filtering.
- `My_Code_for_Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning (1).ipynb`: Training script to fine-tune Llama 3.2 models on the AER dataset using Unsloth. Includes code to serve a Flask-based inference API via ngrok.
- `BPC_NLP.pdf`: The full technical research report detailing experimental setups, discussions, and ablation studies.
- `zero_shot.py` / `My_Code_for_....py`: Extracted python script equivalents for rapid code review.

## Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install torch transformers sentence-transformers faiss-cpu bm25s PyStemmer flask pyngrok
```
To run the fine-tuning code, install the Unsloth library:
```bash
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git@nightly git+https://github.com/unslothai/unsloth-zoo.git
```

## Usage

### Evaluation (Few-Shot)
Run `zero_shot.ipynb` to evaluate the base Llama models. The notebook processes the datasets, builds the BM25 and FAISS embeddings, and calculates metrics (accuracy, precision, recall) across different N-shot regimes.

### Fine-Tuning
Execute `My_Code_for_Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning (1).ipynb` to initialize the Unsloth trainer. The script:
1. Loads the preprocessed AER dataset.
2. Augments prompts with FAISS-retrieved documents.
3. Fine-tunes the model adapters.
4. Serves the fine-tuned model as a local API endpoint on port 5000.

## Results

- **Few-Shot**: Showed limited capabilities; models struggled to distinguish direct causes from contextual factors, resulting in low accuracy and over-prediction.
- **Fine-Tuning**: 
  - **Llama 3.2 8B**: Reached **0.80 accuracy** and an F1 score of 0.88.
  - **Llama 3.2 3B**: Reached **0.67 accuracy** and an F1 score of 0.82.
  
*Fine-tuning on task-specific examples successfully eliminated formatting inconsistencies and significantly enhanced the models' logical causal deduction skills.*

## Authors
- **Hwanseo Lee**
- **Bharat Agrawal**

**Advisor:** Tsedeniya Kinfe Temesgen  
*Technical University of Munich*
