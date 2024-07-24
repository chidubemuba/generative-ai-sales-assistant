
from pinecone import Pinecone
from openai import OpenAI
from typing import List
from transformers import pipeline
from app.settings import inject_settings
from datetime import datetime
import uuid

settings = inject_settings()
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("salesloft-vista")
classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k = 1)

def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding='utf-8') as file:
        return file.read()

def chunk_text(text: str, chunk_size: int) -> list:
    chunks = []
    curr_chunk = ""

    for word in text.split():
        if len(curr_chunk) + len(word) < chunk_size:
            curr_chunk += word + " "
        else:
            chunks.append(curr_chunk)
            curr_chunk = word + " "

    chunks.append(curr_chunk)
    return chunks

def assign_genre(chunk: str) -> str:
    prediction = classifier(chunk)
    # print(prediction)
    genre = prediction[0][0]['label']
    # genre = max(prediction, key = lambda x: x['score'])['label']
    return genre

def upload_to_pinecone(text: str, genre: str):
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.embeddings.create(
        input= text,
        model="text-embedding-3-small"
    )
    timestamp = datetime.now().isoformat()
    embeddings = response.data[0].embedding

    index.upsert(
        vectors=[
            {
            "id": uuid.uuid4().hex, 
            "values": embeddings, 
            "metadata": {"genre": genre, "year": timestamp, "textchunk": text}
            }
        ]
    )


if __name__ == '__main__':
    file_path = "C:/Users/ubach/Downloads/Salesloft Vista Hackerathon/sl-vista-backend/devtools/Pricing best practices.txt"
    transcript = read_file(file_path)
    chunks = chunk_text(transcript, 1000)
    for chunk in chunks:
        genre = assign_genre(chunk)
        upload_to_pinecone(chunk, genre)
