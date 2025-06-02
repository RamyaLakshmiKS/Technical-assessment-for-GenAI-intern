## Reflection: Task 1 - 10-K Retrieval QA

**Why I chose this method or approach:**
I chose a modular Retrieval-Augmented Generation (RAG) pipeline using chunking, embedding, and vector search because it enables scalable, efficient retrieval and synthesis from large, unstructured SEC filings. Recursive chunking with overlap balances recall and precision, while FAISS provides fast, memory-efficient similarity search. Using a modern LLM (Gemini) for answer synthesis allows for flexible, citation-aware responses. The pipeline is designed for reproducibility and extensibility.

**Why I used 10-Q instead of 10-K filings:**
I used 10-Q filings instead of 10-K because, at the time of development, the latest 10-K filings were not yet available for all selected companies. 10-Qs are the most recent comprehensive filings and provide up-to-date financial and risk information, ensuring the pipeline demonstrates retrieval and synthesis on the freshest available data for each company.

**What surprised me during development:**
I was surprised by how sensitive retrieval quality is to chunk size, overlap, and the number of retrieved chunks. Even with strong prompts, LLMs sometimes synthesize or summarize rather than extract verbatim, especially when context is noisy or fragmented. Cleaning and preprocessing filings (e.g., removing HTML) is critical for high-quality retrieval. API quota and cost management also became important practical concerns.

**What my next steps would be given additional time:**
- Preprocess SEC filings to remove HTML tags and noise before chunking, for cleaner retrieval and extraction.
- Experiment with chunking strategies, overlap, and retriever parameters for optimal recall/precision.
- Add more granular and semantic evaluation metrics (e.g., BLEU, ROUGE, embedding similarity).
- Expand the pipeline to support more companies, more question types, and more robust error handling.
- Integrate the workflow into a CI pipeline for automated regression and cost monitoring.
- Build a UI to inspect retrieval and answer quality interactively.
