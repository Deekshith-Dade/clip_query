import torch
from transformers import CLIPProcessor, CLIPModel
from chromadb import Documents, EmbeddingFunction, Embeddings

loader = torch.load('clip_finetune_36.pt', map_location=torch.device('cpu'), weights_only=True)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model_state_dict = loader['model_state_dict']
model.load_state_dict(model_state_dict)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


class ClipTextEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        text = input
        inputs = processor(text, return_tensors='pt', padding=True, truncation=True)
        text_features = model.get_text_features(**inputs)
        text_embeddings = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_embeddings.tolist()
    
    def embed_documents(self, documents: Documents) -> Embeddings:
        return self(documents)
    

class ClipImageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        images = input
        inputs = processor(images=images, return_tensors='pt', do_rescale=False)
        image_features = model.get_image_features(**inputs)
        return image_features.tolist()
    
    def embed_documents(self, documents: Documents) -> Embeddings:
        return self(documents)
 