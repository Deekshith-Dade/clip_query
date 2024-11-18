from langchain_community.document_loaders import WebBaseLoader
import pdb
import json
import requests
import io
from torchvision import transforms

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
import chromadb
from embedding_functions import ClipTextEmbeddingFunction, ClipImageEmbeddingFunction
import bs4
from PIL import Image

CHROMA_PATH = "chroma"

def main():
    urls = [
        "https://science.nasa.gov/mission/tgo/",
        "https://science.nasa.gov/mission/mars-sample-return/",
        "https://science.nasa.gov/mission/mars-2020-perseverance/",
        "https://science.nasa.gov/mission/maven/",
        "https://science.nasa.gov/resource/nasas-perseverance-observes-observation-rock/",
        "https://www.nhm.ac.uk/discover/planet-mars.html"
    ]
    docs = load_documents(urls)
    chunks = split_documents(docs)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    text_collection = client.get_or_create_collection("chroma_text", metadata={"hnsw:space": "cosine"})
    
    add_text_to_chroma(chunks, text_collection)
    
    with open("images.json", "r") as file:
        images = json.load(file)
    
    image_collection = client.get_or_create_collection("chroma_images", metadata={"hnsw:space": "cosine"})
    add_images_to_chroma(images, image_collection)
    



def add_images_to_chroma(images, collection):
    ided_images = calculate_image_ids_and_embeddings(images, embedding_fun=ClipImageEmbeddingFunction())
    
    existing_items = collection.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing images in DB: {len(existing_ids)}")
    
    new_images = []
    for image in ided_images:
        if image["id"] not in existing_ids:
            new_images.append(image)
    
    if len(new_images):
        print(f"Adding {len(new_images)} new images to DB")
        collection.add(
            embeddings=[image["embedding"] for image in new_images],
            ids=[image["id"] for image in new_images],
            metadatas = [{"source": image["source"], "url": image["url"]} for image in new_images]
        )
        print("Added new images to DB")
    else:
        print("No new images to add to DB")

def calculate_image_ids_and_embeddings(images, embedding_fun=None):
    new_images = []
    for i in range(len(images)):
        image = images[i]
        image_id = f"{image['url']}"
        if embedding_fun:
            
            img = Image.open(io.BytesIO(requests.get(image["url"]).content))
            # convert to RGB if image is not in RGB format
            if img.mode != "RGB":
                img = img.convert("RGB")
            transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to CLIP input size
            transforms.ToTensor()  # Convert to torch tensor
            ])
            img = transform(img)
            image_embedding = embedding_fun(img)[0].tolist()
        new_images.append({"id": image_id, "embedding": image_embedding, "source": image["source"], "url": image["url"]})
    return new_images

def load_documents(urls):
    loader = WebBaseLoader(
        urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                ["article", "main", "div", "section", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
            )
        )
        )
    docs = loader.load()
    
    return docs

def split_documents(documents: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def add_text_to_chroma(chunks: list, collection):
    chunk_with_ids = calculate_chunk_ids_and_embeddings(chunks, embedding_fun=ClipTextEmbeddingFunction())
    
    existing_items = collection.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    new_chunks = []
    for chunk in chunk_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new documents to DB")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        collection.add(
            embeddings=[chunk.metadata["embedding"] for chunk in new_chunks],
            ids=new_chunk_ids,
            # remove embeddings from metadata
            metadatas=[{k: v for k, v in chunk.metadata.items() if k != "embedding"} for chunk in new_chunks]
            # metadatas=[chunk.metadata for chunk in new_chunks],
        )
        print("Added new documents to DB")
    else:
        print("No new documents to add to DB")
    

def calculate_chunk_ids_and_embeddings(chunks, embedding_fun=None):
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        current_page_id = f"{source}:{"web"}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        
        if embedding_fun:
            chunk.metadata["embedding"] = embedding_fun(chunk.page_content)[0].tolist()
        chunk.metadata["id"] = chunk_id
    return chunks


if __name__ == "__main__":
    main()
