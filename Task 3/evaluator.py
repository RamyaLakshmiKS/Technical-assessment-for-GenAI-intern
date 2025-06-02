import os
import re
import csv
import string
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import toml

def normalize_text(text):
    # Lowercase, remove punctuation, and extra whitespace
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def f1_score(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"}
    )
    faiss_path = "../data/faiss_index_10k/final_index.faiss"
    vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    secrets = toml.load("../config/secrets.toml")
    gemini_api_key = secrets.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    retriever = vector_store.as_retriever(search_kwargs={
        "filter": lambda m: "/AAPL/" in m.get("source", "").replace("\\", "/"),
        "k": 20}
    )
    # Stronger prompt for verbatim extraction
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a financial analyst assistant. "
            "Answer the question using only the provided context. "
            "Copy the answer verbatim from the context. Do not summarize, rephrase, or infer. "
            "If the answer is a list, copy the list exactly as it appears. "
            "If the answer is not found, say 'Not found in context.'\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer:"
        )
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa_chain

def main():
    # Load test cases
    df = pd.read_csv("golden_qa_apple.csv")
    qa_chain = load_chain()
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    for idx, row in df.iterrows():
        question, gold = row["question"], row["answer"]
        try:
            result = qa_chain.invoke({"query": question})
            pred = result.get("result", "")
            # Print retrieved context for debugging
            sources = result.get("source_documents", [])
            print(f"\n--- Retrieved context for Q{idx+1}: {question} ---")
            for i, doc in enumerate(sources):
                print(f"[Chunk {i+1}]: {doc.page_content[:300]}...\n")  # Print first 300 chars of each chunk
            

            usage = getattr(result.get("llm_output", {}), "usage", None) or result.get("llm_output", {}).get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
            completion_tokens = usage.get("completion_tokens", 0) if usage else 0
            cost = usage.get("total_cost", 0.0) if usage else 0.0
            # If cost is not provided, estimate (Gemini pricing: $0.00025/1K input, $0.0005/1K output tokens)
            if not cost:
                cost = (prompt_tokens / 1000 * 0.00025) + (completion_tokens / 1000 * 0.0005)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_cost_usd += cost
            f1 = f1_score(pred, gold)
            results.append({
                "Question": question,
                "Gold": gold,
                "Prediction": pred,
                "F1 Score": round(f1, 3),
                "Cost (cents)": round(cost * 100, 3)
            })
        except Exception as e:
            results.append({
                "Question": question,
                "Gold": gold,
                "Prediction": "[ERROR]",
                "F1 Score": 0.0,
                "Cost (cents)": 0.0,
                "Error": str(e)
            })
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    mean_f1 = np.mean([r["F1 Score"] for r in results])
    print(f"\nMean F1 Score: {mean_f1:.3f}")
    print(f"Total Cost (USD): ${total_cost_usd:.4f}")
    if mean_f1 < 0.6:
        raise AssertionError(f"Mean F1 score {mean_f1:.3f} is below threshold 0.6")
    if total_cost_usd > 0.10:
        raise AssertionError(f"Total cost ${total_cost_usd:.4f} exceeds $0.10 budget")

if __name__ == "__main__":
    main()
