import sqlite3
import faiss
import numpy as np
import uuid
from typing import List, Dict, Optional

class PersistenceAPI:
    """
    Persistence API using SQLite (metadata store) + FAISS (ephemeral vector index).
    Embeddings are persisted in SQLite. FAISS index is rebuilt at startup.
    """

    def __init__(self, db_path: str = "images.db", dim: int = 512):
        self.db_path = db_path
        self.dim = dim  # embedding dimensionality
        self._init_db()
        self._init_faiss()

    # ------------------------------
    # Initialization
    # ------------------------------
    def _init_db(self):
        """Initialize SQLite tables if not exist."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                file_name TEXT,
                file_path TEXT,
                width INTEGER,
                height INTEGER,
                embedding BLOB,
                captions TEXT,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _init_faiss(self):
        """Initialize FAISS index and populate from DB."""
        self.index = faiss.IndexFlatL2(self.dim)  # L2 distance
        self.id_map = {}  # mapping from FAISS idx → image_id
        self._rebuild_index()

    def _rebuild_index(self):
        """Load embeddings from SQLite into FAISS index."""
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
                continue  # skip malformed
            vectors.append(emb)
            self.id_map[idx] = image_id

        if vectors:
            mat = np.vstack(vectors).astype("float32")
            self.index.add(mat)

    # ------------------------------
    # Core CRUD
    # ------------------------------
    def insert_or_update(self, file_path: str, embedding: List[float],
                     captions: Optional[List[str]] = None,
                     metadata: Optional[Dict] = None,
                     width: Optional[int] = None,
                     height: Optional[int] = None) -> str:
        """Insert new image + embedding + metadata or update if exists."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Check if image exists by file_path
        cur.execute("SELECT image_id FROM images WHERE file_path = ?", (file_path,))
        existing_image = cur.fetchone()

        if existing_image:
            image_id = existing_image[0]
            # Update existing image
            updates = []
            values = []

            if embedding is not None:
                updates.append("embedding = ?")
                values.append(np.array(embedding, dtype=np.float32).tobytes())
            if captions is not None:
                updates.append("captions = ?")
                values.append(",".join(captions))
            if metadata is not None:
                updates.append("metadata = ?")
                values.append(str(metadata))

            if updates:
                query = f"UPDATE images SET {', '.join(updates)} WHERE image_id = ?"
                cur.execute(query, tuple(values) + (image_id,))
                conn.commit()

                # Update FAISS index
                self._rebuild_index()
                return image_id
            else:
                return image_id  # No updates needed
        else:
            # Insert new image
            image_id = str(uuid.uuid4())
            file_name = file_path.split("/")[-1]
            captions_str = ",".join(captions) if captions else ""
            metadata_str = str(metadata) if metadata else ""

            emb_arr = np.array(embedding, dtype=np.float32)
            emb_blob = emb_arr.tobytes()

            cur.execute("""
                INSERT INTO images (image_id, file_name, file_path, width, height, embedding, captions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, file_name, file_path, width, height, emb_blob, captions_str, metadata_str))
            conn.commit()

            # Update FAISS index in-memory
            self.index.add(np.expand_dims(emb_arr, axis=0))
            self.id_map[len(self.id_map)] = image_id

            return image_id

    def delete_image(self, image_id: str) -> bool:
        """Delete image (requires FAISS rebuild)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
        deleted = cur.rowcount > 0
        conn.commit()
        conn.close()

        if deleted:
            self._rebuild_index()
        return deleted

    def update_image(self, image_id: str, embedding: Optional[List[float]] = None,
                     captions: Optional[List[str]] = None,
                     metadata: Optional[Dict] = None) -> bool:
        """Update embedding, captions, or metadata."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        updates = []
        values = []

        if embedding is not None:
            updates.append("embedding = ?")
            values.append(np.array(embedding, dtype=np.float32).tobytes())
        if captions is not None:
            updates.append("captions = ?")
            values.append(",".join(captions))
        if metadata is not None:
            updates.append("metadata = ?")
            values.append(str(metadata))

        if not updates:
            return False

        values.append(image_id)
        query = f"UPDATE images SET {', '.join(updates)} WHERE image_id = ?"
        cur.execute(query, tuple(values))
        conn.commit()
        updated = cur.rowcount > 0
        conn.close()

        if updated and embedding is not None:
            self._rebuild_index()
        return updated

    # ------------------------------
    # Querying
    # ------------------------------
    def find_similar_images(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Return top-k most similar images to a query embedding."""
        if self.index.ntotal == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        dists, idxs = self.index.search(query_vec, top_k)

        results = []
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        for rank, idx in enumerate(idxs[0]):
            if idx == -1 or idx not in self.id_map:
                continue
            image_id = self.id_map[idx]
            cur.execute("SELECT file_path, captions, metadata FROM images WHERE image_id = ?", (image_id,))
            row = cur.fetchone()
            if row:
                results.append({
                    "image_id": image_id,
                    "file_path": row[0],
                    "captions": row[1].split(",") if row[1] else [],
                    "metadata": row[2],
                    "score": float(dists[0][rank])
                })
        conn.close()
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

