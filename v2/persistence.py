import numpy as np
import uuid
import os
import json
from typing import List, Dict, Optional, Any
import re
import torch

import chromadb
from chromadb.utils import embedding_functions

from model_manager import ModelManager 
from metrics import Metrics # Import Metrics

# Default ChromaDB location: v2/chroma_db, independent of the working directory
DEFAULT_DB_PATH = os.environ.get("IMAGE_CONVERTER_DB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Helper function for embedding normalization
def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector to unit length."""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

class PersistenceAPI:
    """
    Persistence API using ChromaDB for metadata and vector indexing.
    """

    def __init__(self, collection_name: str = None, path: str = None, embedding_model_type: str = "clip"):
        path = path or DEFAULT_DB_PATH
        self.path = path
        self.embedding_model_type = embedding_model_type
        self.model_manager = ModelManager() 
        
        # Determine collection name based on model if not provided
        if collection_name is None:
            collection_name = f"images_{embedding_model_type}"
            
        print(f"Initializing PersistenceAPI with {embedding_model_type.upper()} | collection='{collection_name}' at path='{path}'")
        
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        print(f"PersistenceAPI initialized successfully with ChromaDB (cosine distance)")

    def _get_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using selected encoder and normalize."""
        try:
            model = self.model_manager.get_embedding_model(self.embedding_model_type)
            with torch.no_grad():
                text_emb = model.encode([text], convert_to_tensor=True)
            emb_np = text_emb[0].cpu().numpy()
            return normalize_embedding(emb_np).tolist()
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to generate text embedding for '{text[:50]}...': {e}")
            return None

    def _get_batch_text_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for a batch of text strings efficiently."""
        if not texts:
            return []
        try:
            model = self.model_manager.get_embedding_model(self.embedding_model_type)
            with torch.no_grad():
                batch_embs = model.encode(texts, convert_to_tensor=True)
            
            # Convert to normalized numpy arrays
            embs_np = batch_embs.cpu().numpy()
            return [normalize_embedding(e) for e in embs_np]
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to generate batch text embeddings: {e}")
            return [None] * len(texts)

    def _rebuild_index(self):
        """Clears and potentially re-populates the ChromaDB collection."""
        print(f"\n=== Rebuilding ChromaDB collection: {self.collection_name} ===")
        try:
            # Delete and re-create the collection to clear all data
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("ChromaDB collection rebuilt (cleared).")
        except Exception as e:
            print(f"Error rebuilding ChromaDB collection: {e}")
            raise
        print(f"=== Rebuild Complete ===")
    
    def insert_or_update(self, file_path: str, embedding: List[float],
                         captions: Optional[List[str]] = None,
                         description: Optional[str] = None,
                         metadata: Optional[Dict] = None,
                         width: Optional[int] = None,
                         height: Optional[int] = None) -> str:
        """Insert new image + embedding + captions + detailed description + metadata or update if exists."""
        metrics = Metrics()
        metrics.start("DB Upsert (Inner)")
        
        # Check if an entry for this file_path already exists
        existing_entries = self.collection.get(
            where={"file_path": file_path},
            include=['metadatas']
        )
        
        image_id = None
        if existing_entries and existing_entries['ids']:
            for meta in existing_entries['metadatas']:
                if 'image_id' in meta:
                    image_id = meta['image_id']
                    break
            if image_id:
                self.collection.delete(where={"image_id": image_id})
            else:
                image_id = str(uuid.uuid4())
        else:
            image_id = str(uuid.uuid4())

        ids = []
        embeddings_to_add = []
        metadatas_to_add = []
        documents_to_add = []

        # Normalize the image embedding passed in
        normalized_img_emb = normalize_embedding(np.array(embedding, dtype=np.float32)).tolist()

        # 1. Add/Update Image Embedding
        image_entry_id = f"img-{image_id}"
        base_metadata = {
            "image_id": image_id,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "width": width,
            "height": height,
            "type": "image",
            "captions": json.dumps(captions) if captions else "[]",
            "description": description if description else "",
        }
        if metadata:
            base_metadata.update(metadata)

        ids.append(image_entry_id)
        embeddings_to_add.append(normalized_img_emb)
        metadatas_to_add.append(base_metadata)
        doc_content = description if description else (captions[0] if captions else os.path.basename(file_path))
        documents_to_add.append(doc_content)

        # 2. Add/Update Caption Embeddings (Short Captions)
        if captions:
            metrics.start("Text Embedding")
            caption_embeddings = self._get_batch_text_embeddings(captions)
            metrics.stop("Text Embedding")
            
            for i, (caption_text, caption_embedding) in enumerate(zip(captions, caption_embeddings)):
                if caption_embedding is not None:
                    ids.append(f"cap-{image_id}-{i}")
                    embeddings_to_add.append(caption_embedding.tolist())
                    metadatas_to_add.append({
                        "image_id": image_id,
                        "file_path": file_path,
                        "type": "caption",
                        "caption_text": caption_text,
                    })
                    documents_to_add.append(caption_text)

        # 3. Add/Update Description Embedding (Detailed Description)
        if description:
            sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', description) if s.strip()]
            sentences = [s for s in sentences if len(s) >= 10]
            
            if sentences:
                metrics.start("Text Embedding")
                sentence_embeddings = self._get_batch_text_embeddings(sentences)
                metrics.stop("Text Embedding")
                
                for i, (sentence, sentence_embedding) in enumerate(zip(sentences, sentence_embeddings)):
                    if sentence_embedding is not None:
                        ids.append(f"desc-{image_id}-{i}")
                        embeddings_to_add.append(sentence_embedding.tolist())
                        metadatas_to_add.append({
                            "image_id": image_id,
                            "file_path": file_path,
                            "type": "description_chunk",
                            "full_description_snippet": description[:100] + "...",
                        })
                        documents_to_add.append(sentence)

        if ids:
            metrics.start("ChromaDB Write")
            self.collection.upsert(
                embeddings=embeddings_to_add,
                metadatas=metadatas_to_add,
                documents=documents_to_add,
                ids=ids
            )
            metrics.stop("ChromaDB Write")

        metrics.stop("DB Upsert (Inner)")
        return image_id

    def find_similar_images(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Return top-k most similar images to a query embedding."""
        print(f"\nSearching for {top_k} similar images in collection {self.collection_name}...")
        
        if not self.collection.count():
            print("ChromaDB collection is empty. No search possible.")
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 5, 
            include=['distances', 'metadatas']
        )
        
        found_results = []
        seen_image_ids = set()

        if results and results['metadatas'] and results['distances']:
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            sorted_results = sorted(zip(distances, metadatas), key=lambda x: x[0])

            for distance, metadata in sorted_results:
                image_id = metadata.get('image_id')
                if image_id and image_id not in seen_image_ids:
                    # Get full image data, not just the minimal image record
                    # This ensures captions and description are up-to-date and complete
                    image_data = self.get_full_image_data(metadata['file_path'])
                    if image_data:
                        # Append the full image_data, but add the score from this specific match
                        image_data['score'] = distance
                        image_data['matched_by'] = metadata.get('type', 'unknown')
                        found_results.append(image_data)
                        seen_image_ids.add(image_id)
                        if len(found_results) >= top_k:
                            break
        
        print(f"Returning {len(found_results)} unique image results.")
        return found_results

    def cleanup_empty_text_embeddings(self) -> int:
        """
        Finds and removes ChromaDB entries (captions and description chunks)
        that have empty text content.
        Returns the number of entries removed.
        """
        print(f"\nCleaning up empty text embeddings in collection: {self.collection_name}...")
        
        ids_to_delete = []

        # Find empty captions
        empty_caption_results = self.collection.get(
            where={"$and": [{"type": "caption"}, {"caption_text": ""}]},
            include=[]
        )
        if empty_caption_results['ids']:
            ids_to_delete.extend(empty_caption_results['ids'])
            print(f"Found {len(empty_caption_results['ids'])} empty caption entries.")

        # Find empty description chunks (documents that are empty strings)
        # Note: ChromaDB's document filter might be tricky for truly empty strings.
        # We might need to iterate or query for very short ones.
        # For now, let's rely on the filtering in caption_image.py to prevent new ones.
        # For existing ones, `document: ""` would be needed, but ChromaDB might not index empty documents.
        # A safer bet is to query `metadatas` for `type: "description_chunk"` and then check their `documents`
        # if the query by document itself doesn't work.
        
        # Let's try querying for all description chunks and filtering in Python
        all_desc_chunks = self.collection.get(
            where={"type": "description_chunk"},
            include=['documents']
        )
        for i, doc_id in enumerate(all_desc_chunks['ids']):
            if not all_desc_chunks['documents'][i] or len(all_desc_chunks['documents'][i].strip()) < 5: # Small threshold
                ids_to_delete.append(doc_id)
        print(f"Found {len(ids_to_delete) - len(empty_caption_results['ids'])} (possibly empty or very short) description chunk entries.")


        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} entries with empty text content...")
            self.collection.delete(ids=ids_to_delete)
            print("Cleanup complete.")
            return len(ids_to_delete)
        else:
            print("No empty text embeddings found for cleanup.")
            return 0


    def get_image(self, image_id: str) -> Optional[Dict]:
        """Retrieve image record by image_id."""
        if not image_id:
            return None

        # Directly fetch the main image entry using its known ID format
        results = self.collection.get(
            ids=[f"img-{image_id}"],
            include=['metadatas']
        )

        if results and results['metadatas']:
            image_metadata = results['metadatas'][0]
            return {
                "image_id": image_id,
                "file_name": image_metadata.get('file_name'),
                "file_path": image_metadata.get('file_path'),
                "width": image_metadata.get('width'),
                "height": image_metadata.get('height'),
                "captions": json.loads(image_metadata.get('captions', '[]')),
                "description": image_metadata.get('description', ''),
                "metadata": { 
                    k:v for k,v in image_metadata.items() if k not in ["image_id", "file_path", "file_name", "width", "height", "type", "captions", "description"]
                }
            }
        return None

    def list_images(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """List images with pagination."""
        # Use ChromaDB's built-in pagination
        image_entries = self.collection.get(
            where={"type": "image"},
            include=['metadatas'],
            limit=limit,
            offset=offset
        )

        results = []
        if image_entries and image_entries['metadatas']:
            for metadata in image_entries['metadatas']:
                image_id = metadata.get('image_id')
                if image_id:
                    results.append({
                        "image_id": image_id,
                        "file_name": metadata.get('file_name'),
                        "file_path": metadata.get('file_path'),
                        "width": metadata.get('width'),
                        "height": metadata.get('height'),
                        "captions": json.loads(metadata.get('captions', '[]')),
                        "description": metadata.get('description', ''),
                        "metadata": { 
                            k:v for k,v in metadata.items() if k not in ["image_id", "file_path", "file_name", "width", "height", "type", "captions", "description"]
                        }
                    })
        
        return results

    def save_index(self):
        print("ChromaDB PersistentClient handles saving automatically.")
        pass

    def load_index(self) -> bool:
        print("ChromaDB PersistentClient handles loading automatically.")
        return True

    def get_full_image_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves all stored information for a given image file_path from ChromaDB.
        Aggregates image, caption, and description_chunk entries.
        """
        results = self.collection.get(
            where={"file_path": file_path},
            include=['metadatas', 'documents', 'embeddings']
        )

        if not results['ids']:
            return None

        image_data = {
            "image_id": None,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "width": None,
            "height": None,
            "metadata": {},
            "description": "",
            "captions": [],
            "embedding": None,
            "description_chunks": [],
        }

        # First pass: find the main image entry and its ID
        for i, metadata in enumerate(results['metadatas']):
            if metadata.get("type") == "image":
                image_data["image_id"] = metadata.get("image_id")
                image_data["width"] = metadata.get("width")
                image_data["height"] = metadata.get("height")
                image_data["description"] = metadata.get("description", "")
                
                # Copy other metadata, excluding our internal keys
                for k, v in metadata.items():
                    if k not in ["image_id", "file_path", "file_name", "width", "height", "type", "description", "captions"]:
                        image_data["metadata"][k] = v
                
                # Attempt to parse captions if stored as JSON string
                stored_captions_str = metadata.get("captions")
                if stored_captions_str:
                    try:
                        image_data["captions"] = json.loads(stored_captions_str)
                    except json.JSONDecodeError:
                        image_data["captions"] = [stored_captions_str] # Fallback
                
                image_data["embedding"] = results['embeddings'][i]
                break
        
        if not image_data["image_id"]:
            return None

        # Second pass: gather all associated captions and description chunks
        for i, metadata in enumerate(results['metadatas']):
            if metadata.get("image_id") == image_data["image_id"]:
                if metadata.get("type") == "caption":
                    if metadata.get("caption_text") not in image_data["captions"]:
                        image_data["captions"].append(metadata.get("caption_text"))
                elif metadata.get("type") == "description_chunk":
                    image_data["description_chunks"].append({
                        "document": results['documents'][i],
                        "metadata": metadata,
                        "embedding_snippet": results['embeddings'][i][:5] # Store a snippet
                    })

        return image_data
