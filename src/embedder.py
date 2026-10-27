import torch
from PIL import Image
import open_clip
# The embedder uses the BiomedCLIP model from Microsoft, which is designed for biomedical image and text embeddings. 
# It provides methods to convert images and text into 512-dimensional embeddings suitable for various downstream tasks
# such as similarity search or clustering.


# Model id, Hugging Face Hub
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

class BiomedClipEmbedder:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load BiomedCLIP, its tokenizer, and preprocessing transforms
        self.model, _ , self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def embed_batch(self, image_paths: list[str]) -> list[list[float]]:
        """Batch image embedding."""
        tensors = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(self.preprocess(img))

        batch_tensor = torch.stack(tensors).to(self.device) 

        with torch.no_grad():
            features = self.model.encode_image(batch_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().tolist()


    def embed_image(self, image_path: str) -> list[float]:
        """Image to 512-dimensional embedding."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            # L2 normalization for cosine similarity
            features /= features.norm(dim=-1, keepdim=True)
            
        return features.squeeze(0).cpu().numpy().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Text to 512-dimensional embedding."""
        text_tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)
            
        return features.squeeze(0).cpu().numpy().tolist()