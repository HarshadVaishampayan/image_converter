import os
import platform
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)
from sentence_transformers import SentenceTransformer

def detect_environment() -> str:
    """Detect if running on desktop or mobile (very basic heuristic)."""
    import platform
    system = platform.system().lower()
    if system in ["linux", "windows", "darwin"]:
        if "ANDROID_ROOT" in os.environ:  # Android on Linux
            return "mobile"
        return "desktop"
    elif system in ["ios"]:
        return "mobile"
    else:
        return "desktop"

def get_device() -> torch.device:
    """Detects and returns the appropriate device for PyTorch."""
    if torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")
    print("CUDA and MPS not available, using CPU")
    return torch.device("cpu")

def load_models(env: str, device: torch.device):
    """Load configuration for captioning and multiple embedding models."""
    if env == "desktop":
        caption_model_name = "florence-community/Florence-2-large"  # native transformers port
        clip_name = "sentence-transformers/clip-ViT-L-14"
        siglip_name = "google/siglip2-so400m-patch14-384"  # SigLIP 2 (shape-optimized 400M)
        batch_size = 2 
        beam_size = 1 
    else:
        caption_model_name = "florence-community/Florence-2-base"
        clip_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        siglip_name = "google/siglip2-base-patch16-224"
        batch_size = 1
        beam_size = 1

    return caption_model_name, clip_name, siglip_name, batch_size, beam_size

class SigLIPWrapper:
    """Wrapper for SigLIP / SigLIP 2 models to provide a SentenceTransformer-compatible interface."""
    def __init__(self, model_name, device):
        from transformers import AutoProcessor, AutoModel
        import torch
        print(f"Loading SigLIP ({model_name}) via Transformers...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
        ).to(device).eval()
        self.device = device

    def encode(self, input_data, convert_to_tensor=True, device=None):
        import torch
        from PIL import Image
        
        # Determine if input is a single item or a list
        is_single_input = False
        if isinstance(input_data, (str, Image.Image)):
            is_single_input = True
            input_data = [input_data]
        
        is_text = isinstance(input_data[0], str)

        with torch.no_grad():
            if is_text:
                inputs = self.processor(
                    text=input_data, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=64, 
                    return_tensors="pt"
                ).to(self.device)
                features = self.model.get_text_features(**inputs)
            else:
                inputs = self.processor(images=input_data, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)

            # transformers 5 may return a ModelOutput instead of a bare tensor
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            
            # Normalize for cosine similarity
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        
        if is_single_input:
            features = features[0]

        if convert_to_tensor:
            return features
        return features.cpu().numpy()

class ModelManager:
    """Singleton class to manage model instances."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            print("Initializing ModelManager...")
            self.device = get_device()
            self.env = detect_environment()
            # Load model configuration
            (self.caption_model_name, 
             self.clip_name, 
             self.siglip_name,
             self.batch_size, 
             self.beam_size) = load_models(self.env, self.device)
            
            # Initialize model placeholders
            self.caption_processor = None
            self.caption_model = None
            self.clip_model = None
            self.siglip_model = None
            
            self.initialized = True
            print(f"ModelManager initialized on device {self.device}")
    
    def get_clip_model(self):
        if self.clip_model is None:
            print("Loading CLIP model...")
            self.clip_model = SentenceTransformer(
                self.clip_name, 
                model_kwargs={"dtype": torch.float16}
            ).to(self.device)
        return self.clip_model

    def get_siglip_model(self):
        if self.siglip_model is None:
            self.siglip_model = SigLIPWrapper(self.siglip_name, self.device)
        return self.siglip_model

    def get_embedding_model(self, model_type="clip"):
        if model_type == "clip":
            return self.get_clip_model()
        elif model_type == "siglip":
            return self.get_siglip_model()
        else:
            raise ValueError(f"Unknown embedding model type: {model_type}")
    
    def get_caption_model(self):
        """Get Captioning models (Florence-2)."""
        if self.caption_model is None:
            print(f"Loading Caption model ({self.caption_model_name})...")
            token = os.getenv("HF_TOKEN")
            self.caption_processor = AutoProcessor.from_pretrained(
                self.caption_model_name,
                token=token,
            )
            # Use float32 for Florence-2 on MPS for stability
            self.caption_model = AutoModelForImageTextToText.from_pretrained(
                self.caption_model_name,
                token=token,
                dtype=torch.float32,
            ).to(self.device).eval()
                
        return self.caption_processor, self.caption_model, self.beam_size, self.batch_size