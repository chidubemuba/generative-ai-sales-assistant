
from pinecone import Pinecone
from app.settings import inject_settings
settings = inject_settings()

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("salesloft-vista")


def upload_to_pinecone():
    pass

if __name__ == '__main__':
    upload_to_pinecone