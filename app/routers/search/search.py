from fastapi import APIRouter
import pinecone
from app.settings import inject_settings
from pinecone import Pinecone

router = APIRouter()
settings = inject_settings()

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("salesloft-vista")

async def text_embedder(text: str):
    """
    Call to Openai embeddings here
    """
    pass


@router.post("/search")
async def search(input_text):
    text_embedding = await text_embedder(text=input_text)
    pinecone_result = index.query(
        vector=text_embedding,
        top_k=10,
        include_metadata=True,
    )
    return {"result": pinecone}
