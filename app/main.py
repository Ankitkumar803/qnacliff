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
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
import json
from typing import Dict

app = FastAPI()

class Question_user(BaseModel):
    question_user: str
class Company(BaseModel):
    company: str
class MAX_NEW_TOKEN(BaseModel):
    max_new_token: int
    
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



@app.post("/qna")
def assess_diversification(question_user:Question_user, company:Company, max_new_token: MAX_NEW_TOKEN ):
    try:

        
        # in case if end-point starting with "qna-falcon-7b-22112000-" is not there 
        ENDPOINT = "End-point starting with qna-llama-7b-22112000- is not found"
        import boto3
        # Get the SageMaker client
        sagemaker_client = boto3.client('sagemaker')
        # List all of the endpoints
        endpoints = sagemaker_client.list_endpoints()
        # Filter the list to only include endpoints that are in the "InService" status
        in_service_endpoints = sagemaker_client.list_endpoints(StatusEquals='InService')

        for endpoint in in_service_endpoints['Endpoints']:
            if "qna-llama-7b-22112000-" in endpoint['EndpointName']:
                ENDPOINT = endpoint['EndpointName']
        print(ENDPOINT)


        MAX_NEW_TOKENS = max_new_token.max_new_token
        if MAX_NEW_TOKENS == 0:
            MAX_NEW_TOKENS = 512
        print(MAX_NEW_TOKENS)


                
        #hosted on sagemaker
        endpoint = ENDPOINT


        class ContentHandler(LLMContentHandler):
                content_type = "application/json"
                accepts = "application/json"

                # def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                #     input_str = json.dumps({"inputs": prompt, **model_kwargs})
                #     return input_str.encode('utf-8')

                def transform_input(self, prompt: str , model_kwargs: Dict) -> bytes:
                    request = {'inputs': prompt,
                                "parameters": { "do_sample": True,
                                            "top_p": 0.9,
                                            "temperature": 0.85,
                                            "max_new_tokens": MAX_NEW_TOKENS,
                                            "stop": ["<|endoftext|>", "</s>"],
                                            "return_full_text":False,
                                            "repetition_penalty": 1.03  #,
                                            # "early_stopping": True
                                                }, **model_kwargs}

                    input_str = json.dumps(request)
                    print(input_str)
                    return input_str.encode('utf-8')

                def transform_output(self, output: bytes) -> str:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]['generated_text']

        content_handler = ContentHandler()

        sm_llm=SagemakerEndpoint(
                endpoint_name=endpoint,
            credentials_profile_name="default",
                region_name="ap-south-1",
                # model_kwargs= parameters,
                content_handler=content_handler,
            )

        print("sm_llm: ", sm_llm)

        import chromadb
        from chromadb.config import Settings
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA


        client = chromadb.HttpClient(host="13.232.139.161", port=8000)
        vectordb = Chroma(client=client, collection_name= company.company , embedding_function=embedding_function)


        # vectordb = Chroma(persist_directory="db", embedding_function=embedding, collection_name="docs")
        retriever = vectordb.as_retriever(search_kwargs={'k':5})
        print("retriever: ", retriever)

        qa_chain = RetrievalQA.from_chain_type(llm=sm_llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        return_source_documents=True)

        ans = qa_chain(question_user.question_user)
        print("************************")
        print(ans["result"])

        return {
            ans["result"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/correlation")
# def calculate_correlation(ticker_symbols: TickerSymbols):
#     try:
        
#         return {}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
