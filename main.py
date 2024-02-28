# all in one
import torch
from IPython.display import Markdown, display
from langchain import PromptTemplate
from langchain import HuggingFacePipeline

from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import json

class ChromaShip:
    """
    Class representing a document in the Chroma vector store.
    """
    def __init__(self, content, metadata=None):
        """
        Constructor for ChromaShip class.

        Parameters:
        - content: str, the content of the document.
        - metadata: any, additional metadata associated with the document.
        """
        self.page_content: str = content
        self.metadata = metadata

class KFCRAG():
    """
    Class for KFCRAG functionality.
    """
    def __init__(self, json_file='menu.json'):
        """
        Constructor for KFCRAG class.

        Parameters:
        - json_file: str, the path to the JSON file containing data.
        """
        self.json_file = json_file
        self.load_json()
        self.embed()
        self.create_rag()
        self.create_pipeline()
        self.create_llm()
        self.create_qa_chain()

    def load_json(self):
        """
        Load data from a JSON file and prepare documents for Chroma vector store.
        """
        with open(self.json_file) as f:
            data = json.load(f)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)

        self.documents = []
        for key, value in data.items():
            if key == 'Location':
                docs = [f'Location: {value["name"]}']
                self.documents.extend([ChromaShip(content=doc, metadata=None) for doc in docs])
            elif key == 'Menus':
                for menu_code, detail_menu in value.items():
                    text = f'Menu code : {menu_code} , '
                    text+=f'Menu Name : {detail_menu["name"]} , '
                    text+=f'Menu Price : {detail_menu["price"]} , '
                    text+=f'Menu contents : {detail_menu["contents"]} , '
                    docs= [text]
                    self.documents.extend([ChromaShip(content=doc, metadata=None) for doc in docs])
            else:

                for item, item_value in value.items():

                    
                    text = f'Item code : {item} , Item Name : {item_value[0]} , Item Price : {item_value[1]} , '
                    detail_data = item_value[2]

                    if 'nutritionalInfo' in detail_data:
                        for detail_key, detail_value in detail_data['nutritionalInfo'].items():
                            text+= 'nutritional info : '
                            if detail_key == 'kcal':
                                text+= f'calories : {detail_value} , '
                                
                            else:
                                text+= f'{detail_key} : {detail_value} , '
                                
                    if 'available' in detail_data:
                        text+= f'available : {["No", "Yes"][int(detail_data["available"])]} , '
                        
                    else:
                        text+= f'available : Yes , '
                        
                    docs = [text]    
                    self.documents.extend([ChromaShip(content=doc, metadata=None) for doc in docs])
            

    def embed(self):
        """
        Embed documents into the Chroma vector store.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True})

        self.db = Chroma.from_documents(self.documents, self.embeddings, persist_directory="db_njscuba")

    def create_rag(self):
        """
        Create a RAG (Retrieval-Augmented Generation) model.
        """
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config
        )

    def create_pipeline(self):
        """
        Create a text generation pipeline using the RAG model.
        """
        self.text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=400,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,)

    def create_llm(self):
        """
        Create a language model pipeline using the text generation pipeline.
        """
        self.llm = HuggingFacePipeline(pipeline=self.text_generation_pipeline)

    def create_qa_chain(self):
        """
        Create a QA (Question-Answering) chain using the language model and Chroma retriever.
        """
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.db.as_retriever())

    def query(self, q):
        """
        Run a query on the QA chain.

        Parameters:
        - q: str, the query string.

        Returns:
        - result: any, the result of the query.
        """
        return self.qa_chain.run(q)

if __name__ == '__main__':
    kfc_rag = KFCRAG()
    print(kfc_rag.query('Give me an orange chocolate milkshake, medium'))
