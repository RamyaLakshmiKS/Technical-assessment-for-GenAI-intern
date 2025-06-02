# Task 3: Automatic Chain Evaluator & Cost Ledger

This folder contains the code and resources for Task 3 of the technical assessment: evaluating a LangChain RetrievalQA chain using golden Q-A pairs, tracking token/cost usage, and reporting results with robust error handling.

## Overview
- Loads at least five golden Q-A test cases for Apple (AAPL) 10-K.
- Runs each test through the RetrievalQA chain.
- Computes exact-match F1 scores (case/punctuation-insensitive).
- Tracks prompt/completion tokens and USD cost.
- Presents a summary table and enforces minimum F1/maximum cost constraints.

See `evaluator.py` for implementation details.
