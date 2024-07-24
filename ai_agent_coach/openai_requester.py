from pinecone import Pinecone
from openai import OpenAI 
import json
import sys
sys.path.append('../')
from devtools.vector_db_uploader import embed_text


def search_documents(client, query_string, index, top_k=10):
    """
    Perform a search in the document index using an embedded query vector and retrieve the top matching documents.

    Args:
        client: The client used to embed the query text.
        query_string (str): The text string to be embedded and used as a query.
        index: The index object used for querying the embedded vectors.
        top_k (int, optional): The number of top matches to retrieve. Defaults to 10.

    Returns:
        list: A list of document texts corresponding to the top matches.
    """
    # Embed the query string into a vector representation
    query_vector = embed_text(client=client, text=query_string)
    
    # Query the index with the embedded vector
    document_payload_response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract the text chunks from the query results
    text_chunks = [match.metadata for match in document_payload_response['matches']]
    
    # Retrieve the document texts from the metadata
    documents = [doc['textchunk'] for doc in text_chunks]
    
    return documents


def concatenate_documents(documents):
    """
    Concatenate a list of documents into a single string with each document
    prefixed by its index.

    Args:
        documents (list of str): A list of document strings to be concatenated.

    Returns:
        str: A single string where each document is prefixed by its index and separated by newlines.
    """
    concatenated_string_list = []

    # Iterate over the documents with an index starting from 1
    for idx, doc in enumerate(documents, start=1):
        # Append the formatted document string to the list
        concatenated_string_list.append(f"Context document #{idx}: {doc}\n")
    
    # Join all document strings into a single string with newlines
    concatenated_document_string = "\n".join(concatenated_string_list)
    
    return concatenated_document_string

def build_prompt(client, index, transcript, top_k = 5):

    system = f"""You are a sales assistant. You role is to support and guide your seller user as they are closing a SAAS deal. You will be provided with a transcription of the latest spoken info from the call, along with some document context containing best practices. Your job is to adapt the best practices context to the live transcript to generate a personalized recommendation for your seller. Generate your recommendation/coaching response as bullet points. Limit them to 2 bullet points and keep them short enough to read in < 5 seconds and display in a small modal. \nYour response MUST be a valid JSON object with your predictions. If it is not a JSON object you will FAIL THIS TASK. 
    """

   # Get the documents
    documents = search_documents(client, transcript, index, top_k=top_k)
    concatenated_doc_context = concatenate_documents(documents)

    user = f"""# Live Transcript String:{transcript}\n\n# Best Practice Context:\n{concatenated_doc_context}\n\n# Your coaching response. Use the following format:{{"recommendations": ["<recommendation 1>", "<recommendation 2>"]}}
    """

    return system,user

def run_llm_call(client, index, parameters, transcription):
    # build prompts
    system, user = build_prompt(client= client, 
                            index =index, 
                            transcript=transcription, 
                            top_k = parameters['top_k'])

    # Create a chat completion
    response = client.chat.completions.create(
        model=parameters['model'],
        temperature = parameters['temperature'],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format = parameters['response_format'],
        max_tokens = parameters['max_tokens'],
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        
    )
    
    # Extract usage stats
    usage_stats = dict(dict(response)["usage"])

    # Convert the response to a dictionary that can be serialized
    response_dict = {
        "id": response.id,
        "created": response.created,
        "model": response.model,
        "object": response.object,
        "system_fingerprint": response.system_fingerprint,
        "completion_tokens": usage_stats['completion_tokens'],
        "prompt_tokens": usage_stats['prompt_tokens'],
        "total_tokens": usage_stats['total_tokens'],
        "choices": [{"content": choice.message.content} for choice in response.choices]
    }

    return response_dict

def get_recommendation(client, index, parameters, transcription):
    """
    Call an LLM API to get recommendations based on a transcription and return a list of recommendations.

    Args:
        client: The client used to call the LLM API.
        index: The index object used in the LLM API call (if applicable).
        parameters (dict): Parameters for the LLM API call.
        transcription (str): The transcription text to be used in the LLM API call.

    Returns:
        list: A list of recommendations extracted from the LLM API response.
    """
    # Call the LLM API with the provided parameters and transcription
    response_dict = run_llm_call(client, index, parameters, transcription)
    
    # Extract the recommendation string from the API response
    recommendation_str = response_dict['choices'][0]['content']
    
    # Parse the recommendation string into a dictionary
    recommendation_dict = json.loads(recommendation_str)
    
    # Extract the list of recommendations from the dictionary
    recommendations_list = recommendation_dict.get('recommendations', [])
    
    return recommendations_list






if __name__ == "__main__":
    from pinecone import Pinecone
    from openai import OpenAI
    import sys
    sys.path.append('../')
    sys.path.extend('../')
    from app.settings import inject_settings
    settings = inject_settings()

    # Set pinecone  client, index, and openai client
    pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pinecone_client.Index("salesloft-vista")
    client = OpenAI(api_key= settings.OPENAI_API_KEY)


