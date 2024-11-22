import os
import shutil
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader

#CHROMA_PATH = give the path for your database for storing the pdf
#DATA_PATH = give the path where u have your pdf
#OPENAI_API_KEY = enter your API key here



def main():
    try:
        print("Generating data store...")
        generate_data_store()
        print("Data store generation completed successfully.")
    except Exception as e:
        print(f"Error in main: {str(e)}")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Example: Printing content of the 10th document chunk
    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

    return chunks


def save_to_chroma(chunks):
    """Save chunks of documents to Chroma database."""
    # Clear out the database first if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Create a new database from the documents
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()


if __name__ == "__main__":
    main()
