# Import libraries
import os
from sec_downloader import Downloader
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import logging
from tqdm import tqdm
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_10k_filings(companies, output_dir="./data/10k_filings"):
    os.makedirs(output_dir, exist_ok=True)
    downloader = Downloader(output_dir, email_address="ra.kuppasundarar@ufl.edu")
    print("Downloader methods:", dir(downloader))
    for ticker in companies:
        logging.info(f"Downloading 10-Q for {ticker}")
        try:
            metadatas = downloader.get_filing_metadatas(ticker)
            print(f"Metadata for {ticker}: {metadatas}")  # DEBUG: print the metadata structure
            if not metadatas:
                logging.error(f"No 10-Q filings found for {ticker}")
                continue
            # Find the most recent 10-Q filing (use attribute access)
            ten_qs = [m for m in metadatas if getattr(m, "form_type", "").upper() == "10-Q"]
            if not ten_qs:
                logging.error(f"No 10-Q filings found for {ticker} in metadata")
                continue
            # Sort by filing_date descending if available
            ten_qs.sort(key=lambda m: getattr(m, "filing_date", ""), reverse=True)
            accession_number = ten_qs[0].accession_number
            filing_url = ten_qs[0].primary_doc_url
            # Download the filing manually using requests
            import requests
            headers = {
                "User-Agent": "ra.kuppasundarar@ufl.edu", 
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov"
            }
            filing_response = requests.get(filing_url, headers=headers)
            if filing_response.status_code == 200:
                # Save as txt file in output_dir, named by ticker and accession number
                ticker_dir = os.path.join(output_dir, ticker)
                os.makedirs(ticker_dir, exist_ok=True)
                file_path = os.path.join(ticker_dir, f"{accession_number}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(filing_response.text)
                logging.info(f"Downloaded and saved filing for {ticker} to {file_path}")
            else:
                logging.error(f"Failed to download filing for {ticker} from {filing_url} (status {filing_response.status_code})")
        except Exception as e:
            logging.error(f"Failed to download 10-Q for {ticker}: {e}")

def parse_and_chunk_filings(filings_dir):
    documents = []
    for file_path in glob.glob(os.path.join(filings_dir, "**/*.txt"), recursive=True):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        doc = Document(page_content=text, metadata={"source": file_path})
        documents.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_and_save_vector_store(texts, output_path="./data/faiss_index_10k", batch_size=32):
    logging.info("Loading SentenceTransformer model 'all-mpnet-base-v2'")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    os.makedirs(output_path, exist_ok=True)
    logging.info("Creating FAISS vector store")
    vector_store = FAISS.from_documents(texts, embeddings)
    final_path = os.path.join(output_path, "final_index.faiss")
    vector_store.save_local(final_path)
    logging.info(f"Final FAISS vector store saved to {final_path}")
    return vector_store

def main():
    companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JNJ", "V"]
    # Clean the filings directory before downloading
    filings_dir = "./data/10k_filings"
    if os.path.exists(filings_dir):
        for root, dirs, files in os.walk(filings_dir):
            for file in files:
                os.remove(os.path.join(root, file))

    download_10k_filings(companies)
    texts = parse_and_chunk_filings(filings_dir)
    logging.info(f"Number of document chunks: {len(texts)}")
    if not texts:
        logging.error("No document chunks found. FAISS index will not be created.")
        return
    create_and_save_vector_store(texts, output_path="./data/faiss_index_10k")
    logging.info("10-Q ingestion and vector store creation complete.")

if __name__ == "__main__":
    main()
