from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Gita")
    st.header("Ask Gita 💬")

    st.subheader("Choose a Question:")
    st.write("1. What is the significance of Lord Krishna's teachings in the Bhagavad Gita?")
    st.write("2. Explain the concept of 'Yoga' as described in the Bhagavad Gita.")
    st.write("3. Discuss the importance of selfless action and its role in spiritual growth according to the Bhagavad Gita.")
    st.write("4. Explore the concept of 'Dharma' and its relevance in the Bhagavad Gita.")

    # upload file (default PDF)
    default_pdf_path = "PDF/Bhagavad_Gita.pdf"  # Replace with the path to your default PDF
    default_pdf_reader = PdfReader(default_pdf_path)
    default_text = ""
    for page in default_pdf_reader.pages:
        default_text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    default_chunks = text_splitter.split_text(default_text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    default_knowledge_base = FAISS.from_texts(default_chunks, embeddings)

    # show user input
    user_question = st.text_input("Ask a question about Gita:")

    if user_question:
        if st.checkbox("Use Gita PDF", value=True):
            knowledge_base = default_knowledge_base
        else:
            pdf = st.file_uploader("Upload Gita PDF", type="pdf")
            if pdf is not None:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # split into chunks
                chunks = text_splitter.split_text(text)
                knowledge_base = FAISS.from_texts(chunks, embeddings)

        # retrieve answers
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)

        st.write(response)

if __name__ == '__main__':
    main()
