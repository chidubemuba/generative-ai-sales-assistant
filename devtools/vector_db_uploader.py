
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
import os

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

def flatten_list_comprehension(nested_list):
    """
    Flatten a list of lists into a single list.

    Args:
        nested_list (list of list): A list where each element is itself a list.

    Returns:
        list: A single list containing all the elements of the sublists.
    """
    return [item for sublist in nested_list for item in sublist]

def process_text_files(directory, read_file, chunk_text, flatten_list_comprehension, chunk_size=200):
    """
    Processes text files in a given directory by reading their contents, chunking the text, 
    and flattening the list of chunks.

    Args:
        directory (str): The directory containing the text files.
        read_file (function): A function that takes a filename as input and returns the file's contents as a string.
        chunk_text (function): A function that takes a text string and a chunk size, and returns a list of text chunks.
        flatten_list_comprehension (function): A function that takes a list of lists and flattens it into a single list.
        chunk_size (int, optional): The size of each text chunk. Defaults to 200.

    Returns:
        list: A flattened list of text chunks.
    """
    # List all files in the specified directory
    files = os.listdir(directory)
    
    # Filter out only text files
    text_files = [file for file in files if file.endswith('.txt')]
    
    # Read the contents of each text file
    list_of_transcripts = [read_file(os.path.join(directory, file)) for file in text_files]
    
    # Chunk each transcript into smaller pieces
    list_of_chunks = [chunk_text(transcript, chunk_size) for transcript in list_of_transcripts]
    
    # Flatten the list of chunks into a single list
    flat_chunk_list = flatten_list_comprehension(list_of_chunks)
    
    return flat_chunk_list

def assign_genre(chunk: str) -> str:
    prediction = classifier(chunk)
    # print(prediction)
    genre = prediction[0][0]['label']
    # genre = max(prediction, key = lambda x: x['score'])['label']
    return genre

def embed_text(client, text):
    response = client.embeddings.create(
        input= text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def upload_to_pinecone(text: str, genre: str):
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    embeddings = embed_text(client, text)
    timestamp = datetime.now().isoformat()

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
    file_path = "/Users/carlos.salas/Documents/sl-vista-backend/devtools/Pricing best practices.txt"
    directory = os.getcwd()
    chunks = process_text_files(directory, read_file, chunk_text, flatten_list_comprehension, chunk_size=200)
    for idx,chunk in enumerate(chunks, start = 1):
        print(f"classifying chunk {idx}")
        genre = assign_genre(chunk)
        print(f"Uploading chunk {idx} to Pinecone\n")
        upload_to_pinecone(chunk, genre)
