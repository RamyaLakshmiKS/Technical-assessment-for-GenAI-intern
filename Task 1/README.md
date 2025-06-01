# Task 1: 10-K Retrieval QA

## Overview
This folder contains all necessary scripts and configuration for Task 1 of the technical assessment: building a Retrieval-Augmented Generation (RAG) pipeline to answer questions about the latest SEC filings for ten companies.

## Rubric Checklist & Implementation Mapping

| Rubric Requirement | Implementation |
|--------------------|----------------|
| Download latest 10-K/10-Q filings for 10 companies | `ingest_10k.py` downloads and saves the latest 10-Q filings for 10 companies using `sec_downloader` |
| Chunk filings | `RecursiveCharacterTextSplitter` in `ingest_10k.py` |
| Embed and store in FAISS | `HuggingFaceEmbeddings` + `FAISS` in `ingest_10k.py` |
| RetrievalQA chain using LLM | `qa_10k_gemini.py` (swap in Gemini LLM as needed) |
| Answer two demo questions per company | `qa_10k_gemini.py` loops over companies and asks two questions each |
| Cite-aware, concise, well-chunked responses | QA script prints answer and source document for each response |
| Reproducibility & organization | All scripts, requirements, and instructions included |

## Usage Instructions

1. **Install requirements**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Ingest and index filings**
   ```powershell
   python ingest_10k.py
   ```
   This will download filings, chunk, embed, and create a FAISS index.
3. **Run RetrievalQA demo**
   ```powershell
   python qa_10k_gemini.py
   ```
   This will answer two demo questions per company and print answers with citations.

## Notes
- If 10-Ks are unavailable, the pipeline uses 10-Qs for demonstration.
- To use Gemini LLM, replace the OpenAI LLM in `qa_10k_gemini.py` with Gemini integration as per your API access.
- All data and secrets are ignored from version control for security and reproducibility.

---
**Contact:** ra.kuppasundarar@ufl.edu
