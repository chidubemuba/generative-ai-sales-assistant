from pinecone import Pinecone
from openai import OpenAI
from typing import List
from transformers import pipeline
import sys
sys.path.append('../')
sys.path.extend('../')
from app.settings import inject_settings
from datetime import datetime
import uuid
from vector_db_uploader import embed_text

settings = inject_settings()
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("salesloft-vista")
client = OpenAI(api_key=settings.OPENAI_API_KEY)

query_string = "What is the best discount you can give me? Are you flexible?"
query_vector = embed_text(client=client, text=query_string)


document_payload_response = index.query(
                                        vector = query_vector,
                                        top_k=3
                                    )

# for document in document_payload_response:
#     print("#"*40)
#     print(document)
#     print("\n")