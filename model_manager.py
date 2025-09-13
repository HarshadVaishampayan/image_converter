import os
import platform
import torch
from transformers import (
    BlipForConditionalGeneration, 
    BlipProcessor, 
    AutoProcessor, 
    Blip2ForConditionalGeneration
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

def load_models(env: str):
    """Load BLIP + CLIP models and config depending on environment type."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if env == "desktop":
        blip_name = "Salesforce/blip-image-captioning-large"
        clip_name = "clip-ViT-B-32"
        batch_size = 2
        beam_size = 2
    else:
        blip_name = "Salesforce/blip-image-captioning-small"
        clip_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        batch_size = 2
        beam_size = 2

    return blip_name, clip_name, batch_size, beam_size, device

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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.env = detect_environment()
            # Load model configuration
            self.blip_name, self.clip_name, self.batch_size, self.beam_size, _ = load_models(self.env)
            # Initialize model placeholders
            self.blip_processor = None
            self.blip_model = None
            self.clip_model = None
            self.blip2_processor = None
            self.blip2_model = None
            self.initialized = True
            print(f"ModelManager initialized with beam_size={self.beam_size}, batch_size={self.batch_size}")
    
    def get_clip_model(self):
        if self.clip_model is None:
            print("Loading CLIP model...")
            self.clip_model = SentenceTransformer(self.clip_name).to(self.device)
        return self.clip_model
    
    def get_blip_models(self):
        """Get BLIP models and their configuration parameters."""
        if self.blip_model is None:
            print("Loading BLIP models...")
            self.blip_processor = BlipProcessor.from_pretrained(self.blip_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_name).to(self.device)
        return self.blip_processor, self.blip_model, self.beam_size, self.batch_size

    def get_blip2_models(self):
        if self.blip2_model is None:
            print("Loading BLIP-2 models...")
            self.blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-6.7b-coco",
                device_map="auto"
            ).to(self.device)
        return self.blip2_processor, self.blip2_model, self.beam_size