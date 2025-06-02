# QA script for Task 1: RetrievalQA using Gemini LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import toml
import os

COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JNJ", "V"]
TICKER_TO_NAME = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta",
    "NVDA": "Nvidia",
    "BRK-B": "Berkshire Hathaway",
    "JNJ": "Johnson & Johnson",
    "V": "Visa"
}
DEMO_QUESTIONS = [
    "What does [Company] list as its three primary sources of revenue?",
    "Summarize the biggest risk [Company] cites about supply chain concentration."
]
FAISS_INDEX_PATH = "./data/faiss_index_10k/final_index.faiss"
SECRETS_PATH = "config/secrets.toml"

def load_embeddings_and_vector_store():
    # Use smaller chunk size and more overlap for better context
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return embeddings, vector_store

def load_gemini_llm():
    secrets = toml.load(SECRETS_PATH)
    gemini_api_key = secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in config/secrets.toml")
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    return llm

def get_company_retriever(vector_store, company):
    def company_filter(metadata):
        src = metadata.get("source", "")
        # Normalize path separators for cross-platform compatibility
        src = src.replace("\\", "/")
        return f"/{company}/" in src
    return vector_store.as_retriever(search_kwargs={"filter": company_filter})

def run_qa(qa_chain, question):
    try:
        # Custom prompt: encourage synthesis and table/list extraction
        custom_prompt = (
            "You are a financial analyst assistant. "
            "If the answer is not explicit, synthesize from the context. "
            "If the answer is a list or table, extract it. "
            "If you cannot answer, say so concisely. "
            "\nContext follows:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        # Use RetrievalQA with custom prompt if possible
        # (LangChain's RetrievalQA supports custom prompt via chain_type_kwargs)
        if hasattr(qa_chain, 'chain_type_kwargs'):
            qa_chain.chain_type_kwargs = {"prompt": custom_prompt}
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "[No answer returned]")
        sources = result.get("source_documents", [])
        # Fallback: if no answer, show top 2 relevant chunks
        if (not answer or answer.strip().lower().startswith("i am sorry") or answer.strip().lower().startswith("[no answer")) and sources:
            fallback = "\n\n[No direct answer found. Top relevant context:]\n"
            for i, doc in enumerate(sources[:2]):
                snippet = doc.page_content[:300].replace('\n', ' ')
                fallback += f"  [Source {i+1}]: {doc.metadata.get('source','N/A')}\n    Context: {snippet}\n"
            answer += fallback
        return answer, sources
    except Exception as e:
        print(f"[ERROR] QA chain failed: {e}")
        return f"[Error: {e}]", []

def format_answer(answer, sources):
    output = f"A: {answer}\n"
    if sources:
        output += "Citations:\n"
        for i, doc in enumerate(sources):
            output += f"  [Source {i+1}]: {doc.metadata.get('source','N/A')}\n"
    else:
        output += "[No source documents retrieved.]\n"
    return output

def main():
    embeddings, vector_store = load_embeddings_and_vector_store()
    llm = load_gemini_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    for company in COMPANIES:
        display_name = TICKER_TO_NAME[company]
        print(f"\n=== {company} ===")
        for question in DEMO_QUESTIONS:
            q = question.replace("[Company]", display_name)
            retriever = get_company_retriever(vector_store, company)
            qa_chain.retriever = retriever
            print(f"Q: {q}")
            answer, sources = run_qa(qa_chain, q)
            print(format_answer(answer, sources))

if __name__ == "__main__":
    main()
