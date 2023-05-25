from langchain.vectorstores import Chroma

def create_db(split_docs,embedding_function,collection_name,persist_directory):
    import os
    if not os.path.exists(persist_directory):
       os.makedirs(persist_directory)
    vectorstore = Chroma.from_documents(split_docs,embedding_function,collection_name=collection_name,persist_directory=persist_directory)
    vectorstore.persist()

def get_similar_docs(query, top_k, embedding_function,collection_name,persist_directory):
    vectordb = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embedding_function)
    return vectordb.similarity_search(query,top_k,include_metadata=True)