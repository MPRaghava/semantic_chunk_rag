from sentence_transformers import SentenceTransformer
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader 
import streamlit as st
import pickle
import os
# from torch import cosine_similarity
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ##  About
    This app is an LLM powered app built using:
    - [Streamlit] (https://streamlit.io/)
    - [langchain](https://python.langchain.com/)
    - [Groq-Gemma - 7B](gemma2-9b-it) LLM model
                
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer]')

def main():
    st.write("Chat with PDF  🗨️")
    pdf = st.file_uploader("Upload your PDF file.",type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
       
        text =""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        sentences = sent_tokenize(text)
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        embeddings = model.encode(sentences)

        threshold = 0.65  # similarity threshold
        chunks =[]
        current_chunk =[sentences[0]]

        for i in range(1,len(sentences)):
            sim = cosine_similarity(
                [embeddings[i-1]],[embeddings[i]]
            ) [0][0]

        if sim > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append("".join(current_chunk))
            current_chunk = [sentences[i]]

        
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        st.write(chunks)
             
        llm = ChatGroq(model_name ="gemma2-9b-it",api_key ="api_key")
       
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vector_store = pickle.load(f)
            st.write("Embeddings are loaded from Disk")    
        else:
             embeddings = HuggingFaceEmbeddings(
             model_name ="sentence-transformers/all-MiniLM-L6-v2"
             )
             vector_store = FAISS.from_texts(chunks, embeddings)
             with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vector_store,f)

             st.write("Embeddings computation completed.")

        #Accept user question or Query
        query = st.text_input("Ask questions about your PDF document:")
        st.write(query)
        

        if query:
            candidates = vector_store.similarity_search(query=query,k=3)
            st.write(candidates)

           
            chain = load_qa_chain(llm =llm, chain_type ="stuff")
            response = chain.run(input_documents = candidates,question = query)
            st.write(response)
            # st.write(candidates)

        # st.write(chunks)
        # st.write(text)

if __name__ == '__main__':
    main()

