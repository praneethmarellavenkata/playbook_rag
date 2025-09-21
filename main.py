import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

class DensoRAGSystem:
    
    def __init__(self):
        load_dotenv()
        self.qa_chain = None
        self.initialized = False
        
    def initialize(self):
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found! Please set OPENAI_API_KEY in your .env file")
            
            loader = PyPDFLoader("Denso - Workflow.pdf")
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            
            embeddings = OpenAIEmbeddings()
            
            vector_store = FAISS.from_documents(docs, embeddings)

            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            
            self.initialized = True
            return True
            
        except Exception as e:
            raise Exception(f"Error initializing RAG system: {str(e)}")
    
    def ask_question(self, question):

        if not self.initialized or not self.qa_chain:
            raise Exception("RAG system not initialized. Call initialize() first.")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                'answer': result['result'],
                'source_documents': result['source_documents']
            }
        except Exception as e:
            raise Exception(f"Error processing question: {str(e)}")
    


rag_system = DensoRAGSystem()


def initialize_system():
    return rag_system.initialize()

def ask_question(question):
    return rag_system.ask_question(question)


