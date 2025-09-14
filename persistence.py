import sqlite3
import faiss
import numpy as np
import uuid
import os
import pickle
from typing import List, Dict, Optional
from model_manager import ModelManager 

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector to unit length."""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

class PersistenceAPI:
    """
    Persistence API using SQLite (metadata store) + FAISS (ephemeral vector index).
    Embeddings are persisted in SQLite. FAISS index is rebuilt at startup.
    """

    def __init__(self, db_path: str = "images.db", dim: int = 512, index_path: str = "faiss.index"):
        print(f"Initializing PersistenceAPI with db_path={db_path}, dim={dim}")
        self.db_path = db_path
        self.dim = dim  # embedding dimensionality
        self.index_path = index_path
        self._init_db()
        self.index = None  # Initialize as None, will be built on-demand
        self.id_map = {}
        self.model_manager = ModelManager()  # Use ModelManager instead of direct CLIP model
        print(f"PersistenceAPI initialized successfully")

    # ------------------------------
    # Initialization
    # ------------------------------
    def _init_db(self):
        """Initialize SQLite tables if not exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_path TEXT UNIQUE,
                    width INTEGER,
                    height INTEGER,
                    embedding BLOB,
                    captions TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization failed: {e}")
            raise
        finally:
            conn.close()

    def _init_faiss(self):
        """Initialize FAISS index and populate from DB."""
        self.index = faiss.IndexFlatL2(self.dim)  # L2 distance
        self.id_map = {}  # mapping from FAISS idx → image_id
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild FAISS index from SQLite database including both image and caption embeddings."""
        print("\n=== Starting FAISS Index Rebuild ===")
        
        # Initialize new FAISS index
        print(f"Initializing new FAISS index with dimension {self.dim}")
        self.index = faiss.IndexFlatL2(self.dim)
        self.id_map = {}
        
        print("Fetching data from SQLite database...")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT image_id, embedding, captions FROM images")
        rows = cur.fetchall()
        conn.close()
        print(f"Found {len(rows)} images in database")

        vectors = []
        image_count = 0
        caption_count = 0
        skipped_count = 0

        for idx, (image_id, emb_blob, captions) in enumerate(rows):
            # Add image embedding
            print(f"\nProcessing image {idx + 1}/{len(rows)}: {image_id}")
            img_emb = np.frombuffer(emb_blob, dtype=np.float32)
            if img_emb.shape[0] != self.dim:
                print(f"  ⚠️ Warning: Skipping malformed image embedding (dim={img_emb.shape[0]})")
                skipped_count += 1
                continue
            vectors.append(img_emb)
            self.id_map[len(vectors)-1] = (image_id, 'image')
            image_count += 1
            print(f"  ✓ Added image embedding")
            
            # Add caption embeddings if available
            if captions:
                caption_list = captions.split(',')
                print(f"  Processing {len(caption_list)} captions")
                for caption in caption_list:
                    if not caption.strip():
                        continue
                    print(f"    Processing caption: {caption[:50]}...")
                    caption_emb = self._get_text_embedding(caption)
                    if caption_emb is not None:
                        vectors.append(caption_emb)
                        self.id_map[len(vectors)-1] = (image_id, 'caption')
                        caption_count += 1
                        print(f"    ✓ Added caption embedding")
                    else:
                        print(f"    ⚠️ Failed to generate caption embedding")

        if vectors:
            print("\nAdding vectors to FAISS index...")
            mat = np.vstack(vectors).astype("float32")
            self.index.add(mat)
            print(f"\n=== Index Rebuild Complete ===")
            print(f"Total vectors added: {len(vectors)}")
            print(f"  - Image embeddings: {image_count}")
            print(f"  - Caption embeddings: {caption_count}")
            print(f"  - Skipped embeddings: {skipped_count}")
        else:
            print("\n⚠️ No vectors available to rebuild index")

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using CLIP text encoder."""
        try:
            clip_model = self.model_manager.get_clip_model()  # Get model from ModelManager
            text_emb = clip_model.encode([text], convert_to_tensor=True)
            return text_emb[0].cpu().numpy()
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to generate text embedding: {e}")
            return None

    # ------------------------------
    # Core CRUD
    # ------------------------------
    def insert_or_update(self, file_path: str, embedding: List[float],
                    captions: Optional[List[str]] = None,
                    metadata: Optional[Dict] = None,
                    width: Optional[int] = None,
                    height: Optional[int] = None) -> str:
        """Insert new image + embedding + metadata or update if exists."""
        print(f"\nProcessing image: {file_path}")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Check if image exists by file_path
        cur.execute("SELECT image_id FROM images WHERE file_path = ?", (file_path,))
        existing_image = cur.fetchone()

        # Ensure embedding is normalized the same way as queries
        embedding_arr = np.array(embedding, dtype=np.float32)
        normalized_embedding = normalize_embedding(embedding_arr)  # Use local function
        
        # Convert to bytes for storage
        embedding_bytes = normalized_embedding.tobytes()

        if existing_image:
            image_id = existing_image[0]
            print(f"Updating existing image with id={image_id}")
            # Update existing image
            updates = []
            values = []

            if embedding is not None:
                updates.append("embedding = ?")
                values.append(embedding_bytes)
            if captions is not None:
                updates.append("captions = ?")
                values.append(",".join(captions))
            if metadata is not None:
                updates.append("metadata = ?")
                values.append(str(metadata))
            if width is not None:
                updates.append("width = ?")
                values.append(width)
            if height is not None:
                updates.append("height = ?")
                values.append(height)

            if updates:
                query = f"UPDATE images SET {', '.join(updates)} WHERE image_id = ?"
                values.append(image_id)
                cur.execute(query, tuple(values))
                conn.commit()
                # Remove rebuild call here
        else:
            # Insert new image
            image_id = str(uuid.uuid4())
            print(f"Inserting new image with id={image_id}")
            file_name = file_path.split("/")[-1]
            captions_str = ",".join(captions) if captions else None
            metadata_str = str(metadata) if metadata else None
            
            cur.execute("""
                INSERT INTO images (image_id, file_name, file_path, width, height, embedding, captions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, file_name, file_path, width, height, 
                  embedding_bytes,
                  captions_str, metadata_str))
            conn.commit()
            # Remove rebuild call here

        conn.close()
        print(f"Operation completed for image: {file_path}")
        return image_id

    def save_index(self):
        """Save FAISS index to disk."""
        if self.index is not None:
            print(f"Saving FAISS index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)
            # Save id_map alongside the index
            with open(f"{self.index_path}.map", 'wb') as f:
                pickle.dump(self.id_map, f)
            print("Index saved successfully")

    def load_index(self) -> bool:
        """Load FAISS index from disk if it exists."""
        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_path}")
            try:
                self.index = faiss.read_index(self.index_path)
                # Load id_map
                with open(f"{self.index_path}.map", 'rb') as f:
                    self.id_map = pickle.load(f)
                print(f"Loaded index with {self.index.ntotal} vectors")
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
                return False
        return False

    # ------------------------------
    # Querying
    # ------------------------------
    def find_similar_images(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Return top-k most similar images to a query embedding."""
        print(f"\nSearching for {top_k} similar images...")
        
        # Try to load existing index first
        if self.index is None:
            if not self.load_index():
                print("No existing index found, building new index...")
                self._rebuild_index()
                self.save_index()  # Save the newly built index
            
        if self.index.ntotal == 0:
            print("FAISS index is empty, rebuilding...")
            self._rebuild_index()
            
            if self.index.ntotal == 0:
                print("No vectors available in database to rebuild index")
                return []

        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        print(f"Query vector shape: {query_vec.shape}")
        
        dists, idxs = self.index.search(query_vec, top_k * 2)  # Get more results as some might be captions
        
        results = []
        seen_images = set()  # Track unique images
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for rank, idx in enumerate(idxs[0]):
            if idx == -1 or idx not in self.id_map:
                continue
                
            image_id, entry_type = self.id_map[idx]
            
            # Skip if we've already included this image
            if image_id in seen_images:
                continue
                
            cursor.execute("SELECT file_path, captions, metadata FROM images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            if row:
                results.append({
                    "image_id": image_id,
                    "file_path": row[0],
                    "captions": row[1].split(",") if row[1] else [],
                    "metadata": row[2],
                    "score": float(dists[0][rank]),
                    "matched_by": entry_type
                })
                seen_images.add(image_id)
                print(f"Match {len(results)}: {row[0]} (score: {float(dists[0][rank])}, type: {entry_type})")
                
                if len(results) >= top_k:
                    break
        
        conn.close()
        print(f"Returning {len(results)} unique image results")
        return results

    def get_image(self, image_id: str) -> Optional[Dict]:
        """Retrieve image record by id."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "image_id": row[0],
            "file_name": row[1],
            "file_path": row[2],
            "width": row[3],
            "height": row[4],
            "captions": row[6].split(",") if row[6] else [],
            "metadata": row[7]
        }

    def list_images(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """List images with pagination."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT image_id, file_path, captions, metadata FROM images LIMIT ? OFFSET ?", (limit, offset))
        rows = cur.fetchall()
        conn.close()

        return [{
            "image_id": r[0],
            "file_path": r[1],
            "captions": r[2].split(",") if r[2] else [],
            "metadata": r[3]
        } for r in rows]

