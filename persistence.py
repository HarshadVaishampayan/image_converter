import sqlite3
import faiss
import numpy as np
import uuid
from typing import List, Dict, Optional

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

    def __init__(self, db_path: str = "images.db", dim: int = 512):
        print(f"Initializing PersistenceAPI with db_path={db_path}, dim={dim}")
        self.db_path = db_path
        self.dim = dim  # embedding dimensionality
        self._init_db()
        self.index = faiss.IndexFlatL2(self.dim)
        self.id_map = {}
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
        """Load embeddings from SQLite into FAISS index."""
        print(f"Rebuilding FAISS index...")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT image_id, embedding FROM images")
        rows = cur.fetchall()
        conn.close()

        self.index.reset()
        self.id_map = {}

        vectors = []
        for idx, (image_id, emb_blob) in enumerate(rows):
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if emb.shape[0] != self.dim:
                print(f"Warning: Skipping malformed embedding for image_id={image_id}")
                continue
            vectors.append(emb)
            self.id_map[idx] = image_id

        if vectors:
            mat = np.vstack(vectors).astype("float32")
            self.index.add(mat)
            print(f"FAISS index rebuilt with {len(vectors)} vectors")
        else:
            print("No vectors available to rebuild index")

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

                # Update FAISS index
                if embedding is not None:
                    print("Updating embedding in FAISS index")
                    self._rebuild_index()
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

            # Update FAISS index
            print("Adding embedding to FAISS index")
            self._rebuild_index()

        conn.close()
        print(f"Operation completed for image: {file_path}")
        return image_id

    # ------------------------------
    # Querying
    # ------------------------------
    def find_similar_images(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Return top-k most similar images to a query embedding."""
        print(f"\nSearching for {top_k} similar images...")
        
        if self.index.ntotal == 0:
            print("FAISS index is empty, attempting to rebuild...")
            self._rebuild_index()
            
            # Check again after rebuild
            if self.index.ntotal == 0:
                print("No vectors available in database to rebuild index")
                return []

        # Validate query embedding dimension
        query_vec = np.array(query_embedding, dtype=np.float32)
        print(f"Input query vector shape: {query_vec.shape}")
        
        if len(query_vec) != self.dim:
            raise ValueError(f"Query embedding dimension ({len(query_vec)}) does not match index dimension ({self.dim})")
        
        # Reshape to 2D array for FAISS
        query_vec = query_vec.reshape(1, -1)
        print(f"Reshaped query vector: {query_vec.shape}")
        
        try:
            dists, idxs = self.index.search(query_vec, top_k)
            print(f"Found {len(idxs[0])} matches")
            print(f"Distances: {dists[0]}")
        except AssertionError as e:
            print(f"FAISS dimension mismatch: Expected {self.dim}, got {len(query_embedding)}")
            raise ValueError(f"Embedding dimension mismatch: Expected {self.dim}, got {len(query_embedding)}")
        
        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for rank, idx in enumerate(idxs[0]):
            if idx == -1 or idx not in self.id_map:
                print(f"Skipping invalid index: {idx}")
                continue
            image_id = self.id_map[idx]
            cursor.execute("SELECT file_path, captions, metadata FROM images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            if row:
                results.append({
                    "image_id": image_id,
                    "file_path": row[0],
                    "captions": row[1].split(",") if row[1] else [],
                    "metadata": row[2],
                    "score": float(dists[0][rank])
                })
                print(f"Match {rank+1}: {row[0]} (score: {float(dists[0][rank])})")
        
        conn.close()
        print(f"Returning {len(results)} results")
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

