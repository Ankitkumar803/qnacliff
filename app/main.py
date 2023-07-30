from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma

app = FastAPI()

class Question(BaseModel):
    question: str
class Company(BaseModel):
    company: str
class MAX_NEW_TOKEN(BaseModel):
    max_new_token: int
    
instructor_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# chroma db client
client = chromadb.HttpClient(host="13.232.139.161", port=8000)
# list all collections
print(client.list_collections())



@app.post("/qna")
def assess_diversification(question:Question, company:Company, max_new_token: MAX_NEW_TOKEN ):
    try:
        print(question.question, type(question.question) , company.company, max_new_token.max_new_token , type(max_new_token.max_new_token))
        print(question , company , max_new_token)
        print(type(question) , type(company) , type(max_new_token))
        try:
            print(type(str(question)), str(question))
        except:
            pass
        # query = question
        query = "In what all sectors does this company works?"
        company1 = 'Tata_motors-mini'
        db = Chroma(client=client, collection_name= company1 , embedding_function=instructor_embeddings)
        docs = db.similarity_search(query, k=3 ) # k = 3
        print(docs[0].page_content)
        
        context = ""
        for doc in docs:
            context += "\n" + doc.page_content
        question = query 
        template = """<|prompt|>Use the following pieces of context to answer the question at the end, don't use information outside the context.

        If you don't know the answer, just say that you don't know, don't try to make up an answer. 


        {context}
        Question: {question}
        Helpful Answer:<|endoftext|><|answer|>"""
        prompt = template.format(context=context, question=question)


        # hyperparameters for llm
        payload = {
          "inputs": prompt,
          "parameters": {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.5,
            "max_new_tokens": 200,  #max_new_token ,   # defalut = 20 , max = 512 ,      input + output = 1512 limit
            "repetition_penalty": 1.03 ,
            "stop": ["\nUser:","<|endoftext|>","</s>"],
            "return_full_text":False
          }
        }

        
        # in case if end-point starting with "qna-falcon-7b-22112000-" is not there 
        ENDPOINT = "End-point starting with qna-falcon-7b-22112000- is not found"
        import boto3
        # Get the SageMaker client
        sagemaker_client = boto3.client('sagemaker')
        # List all of the endpoints
        endpoints = sagemaker_client.list_endpoints()
        # Filter the list to only include endpoints that are in the "InService" status
        in_service_endpoints = sagemaker_client.list_endpoints(StatusEquals='InService')
        
        for endpoint in in_service_endpoints['Endpoints']:
            if "qna-falcon-7b-22112000-" in endpoint['EndpointName']:
                ENDPOINT = endpoint['EndpointName']
        print(ENDPOINT)

        
        # send request to endpoint
        # response = llm.predict(payload)
        # import boto3
        runtime = boto3.client('runtime.sagemaker')
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                            ContentType= 'application/json',
                                            Body=json.dumps(payload))
        result = json.loads(response['Body'].read().decode())

        assistant = result[0]["generated_text"]
        print(assistant)      
        return {
            assistant
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/correlation")
# def calculate_correlation(ticker_symbols: TickerSymbols):
#     try:
        
#         return {}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
