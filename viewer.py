import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd 
import streamlit as st
import argparse

# parser = argparse.ArgumentParser(description='Set the Chroma DB path to view collections')
# parser.add_argument('db')

pd.set_option('display.max_columns', 4)



def view_collections(dir):
    st.markdown("### DB Path: %s" % dir)
    client = chromadb.PersistentClient(path="./TSLA_vector")
    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # collection = client.get_collection(name="", embedding_function=embeddings)
    # client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
    #                                     persist_directory=dir
    #                                 ))

    # This might take a while in the first execution if Chroma wants to download
    # the embedding transformer
    print(client.list_collections())

    st.header("Collections")

    for collection in client.list_collections():
        data = collection.get()

        ids = data['ids']
        embeddings = data["embeddings"]
        metadata = data["metadatas"]
        documents = data["documents"]

        df = pd.DataFrame.from_dict(data)
        st.markdown("### Collection: **%s**" % collection.name)
        st.dataframe(df)

if __name__ == "__main__":
    try:
        # args = parser.parse_args()
        # print("Opening database: %s" % args.db)
        view_collections('./TSLA_vector')
    except:
        pass

