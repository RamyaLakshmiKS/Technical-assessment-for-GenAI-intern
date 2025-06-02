# Technical Assessment for Gen-AI Intern

## Overview
This repository contains my submission for the Gen-AI Intern technical assessment. The project demonstrates a modular Retrieval-Augmented Generation (RAG) pipeline for SEC filings, robust evaluation, and cost tracking. The work is organized by task, with each folder containing code, requirements, and a reflection.

## Folder Structure
- `Task 1/` — 10-K/10-Q Retrieval QA pipeline (RAG, chunking, embedding, vector store, Gemini LLM, demo Q&A, reflection)
- `Task 3/` — Automatic Chain Evaluator & Cost Ledger (unit tests, F1/cost tracking, summary table, reflection)
- `config/` — API keys and configuration (not included in version control)
- `data/` — Downloaded SEC filings and FAISS vector index

## Task Summaries

### Task 1: 10-K/10-Q Retrieval QA
- Built a modular RAG pipeline to answer questions about the latest SEC filings for ten companies.
- Used chunking, embedding (HuggingFace), FAISS vector store, and Gemini LLM for RetrievalQA.
- Answers are citation-aware and concise.
- See `Task 1/README.md` and `Task 1/reflection.md` for details.

### Task 2: (Not Completed)
- **Note:** I was unable to complete Task 2 (presumably a different or intermediate task) due to time constraints.

### Task 3: Automatic Chain Evaluator & Cost Ledger
- Developed an evaluation framework to unit-test the RetrievalQA chain against golden Q-A pairs.
- Tracks F1 score, token/cost usage, and enforces quality/cost constraints.
- See `Task 3/README.md` and `Task 3/reflection.md` for details.

## Reflections
Each task folder contains a `reflection.md` file addressing:
- Why I chose the method or approach
- What surprised me during development
- What my next steps would be given additional time

## How to Run
See the `README.md` in each task folder for setup and usage instructions.

---
**Note:** All code is reproducible, modular, and uses only public packages. Please see individual task folders for more details and reflections.
