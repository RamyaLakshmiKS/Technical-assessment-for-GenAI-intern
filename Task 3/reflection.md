## Reflection: Task 2 - Automatic Chain Evaluator & Cost Ledger

**Why I chose this method or approach:**
I chose a RetrievalQA-based evaluation framework with exact-match F1 scoring because it provides a robust, interpretable, and reproducible way to measure generative QA performance. The use of F1 (case/punctuation-insensitive) balances recall and precision, which is important for real-world LLM outputs that may not match gold answers word-for-word. Token and cost tracking leverages LLM response metadata, with fallback to pricing estimates for Gemini, ensuring cost-awareness and reproducibility. The modular design (separate ingestion, retrieval, evaluation) makes the pipeline easy to debug and extend.

**What surprised me during development:**
I was surprised by how much LLM outputs can vary in phrasing, even with strong extraction prompts. Even when the answer is present in the retrieved context, the LLM sometimes prefers to synthesize or summarize rather than copy verbatim. Retrieval quality is highly sensitive to chunk size, overlap, and the number of retrieved chunks (k). I also found that Gemini API sometimes omits token/cost metadata, requiring fallback estimation. Finally, some gold answers are not present verbatim in the filings, which limits achievable F1.

**What my next steps would be given additional time:**
- Preprocess SEC filings to remove HTML tags and noise before chunking, for cleaner retrieval and extraction.
- Try even larger chunk overlaps (e.g., chunk_size=1000, chunk_overlap=900) to maximize answer coverage.
- Explore relaxing the F1 metric or using semantic similarity if gold answers are not present verbatim in the filings.
- Investigate and tune the retriever filter and metadata to ensure all relevant chunks are included.
- Experiment with chunking strategies, prompt engineering, and alternative LLMs for better extraction.
- Expand the evaluation to cover multiple companies and more diverse question types.
- Integrate the evaluation into a CI pipeline for automated regression and cost monitoring.
- Build a UI to inspect retrieval and answer quality interactively.