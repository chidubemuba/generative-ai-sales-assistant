from fastapi import APIRouter


router = APIRouter()

def classify(input_chunk):
    prediction = 'GENERAL'
    if input_chunk == 'pricing':
        prediction = "PRICE"
    return prediction

@router.post('/chunk-intent-detection')
async def chunk_detection(input_chunk):
    classification = classify(input_chunk=input_chunk)
    if classification == 'PRICE':
        return {"response": {"Recommendation": "The price of the product is $20."}}
