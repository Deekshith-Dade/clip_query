from typing import Union

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from torchvision import transforms
import chromadb
import pdb
from embedding_functions import ClipImageEmbeddingFunction, ClipTextEmbeddingFunction


CHROMA_PATH = "chroma"
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None, r: Union[str, None] = None):
    return {"item_id": item_id, "q": q, "r" : r}

@app.get("/query_with_image")
async def text_query_with_image(file: UploadFile = File(...), n: Union[int, None] = 1):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
                image = image.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to CLIP input size
            transforms.ToTensor()  # Convert to torch tensor
        ])
        image = transform(image)
        embedding_function = ClipImageEmbeddingFunction()
        embedding = embedding_function(image)[0]
        
        
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        text_collection = client.get_collection("chroma_text")
        
        results = text_collection.query(embedding, n_results=n)
        source_urls = [metadata["source"] for metadata in results["metadatas"][0]]
        
        
        return JSONResponse(content={"image_size": image.shape, "filename": file.filename, "result_ids": source_urls}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e), "check": "bad"}, status_code=400)
    

@app.get("/query_with_text")
async def image_query_with_text(q: Union[str, None] = "mars", n: Union[int, None] = 1):
    text = q
    embedding_function = ClipTextEmbeddingFunction()
    embedding = embedding_function(text)
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    image_collection = client.get_collection("chroma_images")
    
    results = image_collection.query(embedding, n_results=n)
    print(results)
    
    return JSONResponse(content={"query": q, "n":n, "result": results["metadatas"]}, status_code=200)
