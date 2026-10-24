import os
import sys

from flask import Flask, abort, jsonify, render_template, send_file, request

# Make sibling modules importable regardless of the working directory
V2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V2_DIR)

import subprocess

import requests

from find_similar import search_similar
from persistence import PersistenceAPI # Import PersistenceAPI
from caption_image import _get_image_paths, process_folder_parallel_caption
from jobs import JobRunner
from duplicates import bp as duplicates_bp

DEFAULT_PORT = 47821  # deliberately not 5000 (macOS AirPlay) / 8080

app = Flask(__name__)
app.register_blueprint(duplicates_bp)


@app.before_request
def _only_local_hosts():
    """Reject DNS-rebinding requests: the app is served to this machine only."""
    if (request.host or "").split(":")[0] not in ("127.0.0.1", "localhost"):
        abort(403)
    # JSON API POSTs must carry a custom header (forces a CORS preflight that
    # this server never approves, so other websites can't trigger them).
    if request.method == "POST" and request.path.startswith("/api/") \
            and request.headers.get("X-Requested-By") != "image-converter":
        abort(403)

# Initialize PersistenceAPI globally (once)
# This prevents "Too many open files" errors by reusing the ChromaDB client
api_siglip = PersistenceAPI(embedding_model_type='siglip')
# api_clip = PersistenceAPI(embedding_model_type='clip') # Only if we need to support clip in GUI

def get_images_from_db(query: str = None, image_id: str = None, top_k: int = 6, sensitivity: float = 0.5):
    """
    Fetch similar images based on query text or by finding similar images to a given image_id.
    Returns list of image data with URLs for display.
    """
    try:
        # Use the globally initialized API instance
        api = api_siglip # Or api_clip if requested by embedding_model_type, but for simplicity, we assume siglip for now
        if image_id:
            print(f"Searching for images similar to image ID: {image_id}")
            # First get the file path for the reference image from ChromaDB
            ref_image_record = api.get_image(image_id)
            
            if not ref_image_record: # Check if the record itself is None
                print(f"Reference image not found for ID: {image_id} (or missing type 'image' entry).")
                return []
            
            if not ref_image_record.get('file_path'): # Then check for file_path
                print(f"Reference image record found for ID: {image_id}, but missing 'file_path'.")
                return []
                
            # Use the file path to search for similar images
            results = search_similar(image_path=ref_image_record['file_path'], top_k=top_k, sensitivity=sensitivity, embedding_model_type='siglip')
        else:
            # Use text query to search
            results = search_similar(query_text=query, top_k=top_k, sensitivity=sensitivity, embedding_model_type='siglip')

        # Format results for display
        images = []
        for result in results:
            images.append({
                'id': result.get('image_id', ''),
                'url': f"/images/{result.get('image_id', '')}",
                'file_path': result.get('file_path', ''),
                'score': f"{result.get('hybrid_score', 0):.4f}",
                'captions': result.get('captions', []),
                'description': result.get('description', '')
            })
        return images
    except Exception as e:
        print(f"Error fetching images: {e}")
        return []

import random

@app.route('/', methods=['GET'])
def search_form():
    """Render the search form."""
    # Use the globally initialized API instance
    api = api_siglip 
    # List a few images by default
    initial_images = []
    try:
        # Fetch a larger pool to randomize from
        pool = api.list_images(limit=50)
        if pool:
            # Pick 10 random images (or fewer if pool is small)
            selected = random.sample(pool, min(len(pool), 10))
            for img in selected:
                initial_images.append({
                    'id': img['image_id'],
                    'url': f"/images/{img['image_id']}",
                    'file_path': img.get('file_path', ''),
                    'score': '',
                    'captions': img['captions'],
                    'description': img.get('description', '')
                })
    except Exception as e:
        print(f"Error loading initial images: {e}")

    return render_template('search.html',
                           active='search',
                           images=initial_images,
                           query=None,
                           matches=10,
                           sensitivity=0.5,
                           reference_id=None)

@app.route('/search', methods = ['POST'])
def image_grid():
    """
    Display image grid based on search criteria.
    Query params:
        q: text query for semantic search
        id: image_id to find similar images
    """
    query = request.form.get('query')
    image_id = request.form.get('image_id')
    matches = request.form.get('matches')
    sensitivity = request.form.get('sensitivity')

    print(f"Search parameters - query: {query}, image_id: {image_id}, matches: {matches}, sensitivity: {sensitivity}")

    # Validate matches parameter
    try:
        matches = int(matches) if matches else 10
        if matches <= 0 or matches > 50:
            matches = 10
    except Exception:
        matches = 10

    # Validate sensitivity parameter
    try:
        sensitivity = float(sensitivity) if sensitivity else 0.5
        if sensitivity <= 0 or sensitivity > 2:
            sensitivity = 0.5
    except Exception:
        sensitivity = 0.5

    images = []
    if query:
        images = get_images_from_db(
            query=query, 
            top_k=matches, 
            sensitivity=sensitivity
        )
    elif image_id:
        images = get_images_from_db(
            image_id=image_id,
            top_k=matches, 
            sensitivity=sensitivity
        )
        
    return render_template('search.html',
                         active='search',
                         images=images,
                         query=query,
                         matches=matches,
                         sensitivity=sensitivity,
                         reference_id=image_id)

@app.route('/images/<image_id>')
def serve_image(image_id):
    """Serve an image from the filesystem based on its ID in the database."""
    try:
        # Use the globally initialized API instance
        api = api_siglip
        # Get image record from ChromaDB
        image_record = api.get_image(image_id)

        if not image_record or not image_record.get('file_path'):
            print(f"Image not found for ID: {image_id}")
            return "Image not found", 404

        file_path = image_record['file_path']
        if not os.path.exists(file_path):
            print(f"File not found at path: {file_path}")
            return "File not found", 404

        print(f"Serving image {image_id} from path: {file_path}")
        return send_file(file_path)

    except Exception as e: # Catch all exceptions, not just sqlite3.Error
        print(f"Error serving image: {e}")
        return "Server error", 500

@app.route('/rebuildindex', methods=['POST'])
def rebuild_index():
    """Clear the SigLIP collection (destructive, so POST + custom header only)."""
    if request.headers.get("X-Requested-By") != "image-converter":
        abort(403)
    api_siglip._rebuild_index()
    return jsonify({"ok": True})


# ------------------------------------------------------- shared JSON API ----
@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(409)
def _api_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": e.description}), e.code
    return e


@app.post('/api/pick-folder')
def pick_folder():
    """Open the native macOS folder chooser (used by Search and Duplicates)."""
    body = request.get_json(silent=True) or {}
    start = body.get("start")
    prompt = (body.get("prompt") or "Choose a folder").replace('"', "")
    script = f'POSIX path of (choose folder with prompt "{prompt}"'
    if start and os.path.isdir(start):
        script += ' default location (POSIX file "%s")' % start.replace('"', '')
    script += ")"
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True,
                             text=True, timeout=600)
    except Exception as e:
        return jsonify({"path": None, "error": str(e)})
    if out.returncode != 0:
        return jsonify({"path": None})  # user cancelled
    path = out.stdout.strip()
    return jsonify({"path": path.rstrip("/") if len(path) > 1 else path})


# ------------------------------------------------------------- indexing ----
OLLAMA_MODEL = "gemma3:12b-it-qat"  # the model caption_image.py asks Ollama for
index_runner = JobRunner()


def _ollama_status():
    """(ok, message) for the Ollama captioner."""
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=1).json()
    except Exception:
        return False, "Ollama isn't running on this machine."
    names = {m.get("name") for m in tags.get("models", [])}
    if OLLAMA_MODEL not in names:
        return False, f"Ollama is running but the {OLLAMA_MODEL} model isn't pulled."
    return True, "ok"


def _indexed_paths():
    entries = api_siglip.collection.get(where={"type": "image"}, include=["metadatas"])
    return {m.get("file_path") for m in entries.get("metadatas") or []}


def _plan_index(body):
    """Resolve the folder and list the images an index run would process."""
    folder = os.path.realpath(os.path.expanduser((body.get("folder") or "").strip()))
    if not os.path.isdir(folder):
        abort(400, "Folder does not exist.")
    recursive = bool(body.get("recursive", True))
    found = _get_image_paths(folder, recursive=recursive)
    todo = found
    if body.get("skip_existing", True):
        done = _indexed_paths()
        todo = [p for p in found if p not in done]
    return folder, recursive, found, todo


@app.get('/api/index/options')
def index_options():
    ok, msg = _ollama_status()
    return jsonify({"ollama_ok": ok, "ollama_msg": msg,
                    "busy": index_runner.current is not None and index_runner.current.status == "running",
                    "current": index_runner.current.to_json() if index_runner.current else None})


@app.post('/api/index/preview')
def index_preview():
    folder, recursive, found, todo = _plan_index(request.get_json(force=True))
    return jsonify({"folder": folder, "found": len(found), "new": len(todo)})


@app.post('/api/index/start')
def index_start():
    body = request.get_json(force=True)
    caption_model = body.get("caption_model", "florence2")
    if caption_model not in ("florence2", "ollama"):
        abort(400, "caption_model must be florence2 or ollama")
    if caption_model == "ollama":
        ok, msg = _ollama_status()
        if not ok:
            abort(400, msg)
    folder, recursive, found, todo = _plan_index(body)
    if not todo:
        abort(400, "No new images to index in that folder.")

    def consumer(result):
        emb = result["embedding"]
        api_siglip.insert_or_update(
            file_path=result["image_path"],
            embedding=emb.tolist() if hasattr(emb, "tolist") else emb,
            captions=[c["text"] for c in result.get("captions", [])],
            description=result.get("description"),
            metadata=result.get("metadata", {}),
            width=result.get("width"),
            height=result.get("height"),
        )

    def work(job):
        total = len(todo)
        job.progress(0, total, f"Loading models… ({total:,} images to index)")
        indexed = 0
        for _ in process_folder_parallel_caption(
                path=folder, consumers=[consumer], embedding_model_type="siglip",
                caption_model_type=caption_model, image_paths=todo):
            indexed += 1
            job.progress(indexed, total, f"Indexed {indexed:,} of {total:,} images…")
            job.check_cancelled()
        if indexed < total:
            job.error = (f"{total - indexed:,} image(s) were skipped: they couldn't be read "
                         "or the captioner returned no description.")
        return indexed

    job = index_runner.start("index", work, done_message="Indexed {n:,} image(s).")
    if job is None:
        abort(409, "An indexing run is already in progress.")
    return jsonify(job.to_json())


@app.get('/api/index/job/<job_id>')
def index_job(job_id):
    job = index_runner.get(job_id) or abort(404)
    return jsonify(job.to_json())


@app.post('/api/index/job/<job_id>/cancel')
def index_cancel(job_id):
    job = index_runner.get(job_id) or abort(404)
    job.cancelled = True
    return jsonify(job.to_json())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the Image Converter web app.")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.environ.get('IMAGE_CONVERTER_PORT', DEFAULT_PORT)))
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(f"Image Converter running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
