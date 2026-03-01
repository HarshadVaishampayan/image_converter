from flask import Flask, render_template, send_file, request
from find_similar import search_similar
import os
from persistence import PersistenceAPI # Import PersistenceAPI

app = Flask(__name__)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

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
                    'score': '',
                    'captions': img['captions'],
                    'description': img.get('description', '')
                })
    except Exception as e:
        print(f"Error loading initial images: {e}")

    return render_template('grid.html', 
                           images=initial_images, 
                           columns=3, 
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
        
    return render_template('grid.html', 
                         images=images, 
                         columns=3, 
                         query=query,
                         matches=matches,
                         sensitivity=sensitivity)

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

@app.route('/rebuildindex', methods=['GET'])
def rebuild_index():
    # Code to rebuild the search index goes here
    search_similar(force_rebuild=True, embedding_model_type='siglip')
    return 'Done'
