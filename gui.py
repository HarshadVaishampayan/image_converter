from flask import Flask, render_template, send_file, request
from find_similar import search_similar
import sqlite3
import os

app = Flask(__name__)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

def get_images_from_db(query: str = None, image_id: str = None, top_k: int = 6, sensitivity: float = 1.5):
    """
    Fetch similar images based on query text or by finding similar images to a given image_id.
    Returns list of image data with URLs for display.
    """
    try:
        if image_id:
            # First get the file path for the reference image
            conn = sqlite3.connect('images.db')
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM images WHERE image_id = ?", (image_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                print(f"Reference image not found for ID: {image_id}")
                return []
                
            # Use the file path to search for similar images
            results = search_similar(image_path=result[0], top_k=top_k, sensitivity=sensitivity)
        else:
            # Use text query to search
            results = search_similar(query_text=query, top_k=top_k, sensitivity=sensitivity)

        # Format results for display
        images = []
        for result in results:
            images.append({
                'id': result.get('image_id', ''),
                'url': f"/images/{result.get('image_id', '')}",
                'score': f"{result.get('score', 0):.4f}",
                'captions': result.get('captions', [])
            })
        return images
    except Exception as e:
        print(f"Error fetching images: {e}")
        return []

@app.route('/', methods=['GET'])
def search_form():
    """Render the search form."""
    return render_template('grid.html', 
                           images=[], 
                           columns=3, 
                           query=None, 
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
    matches = request.form.get('matches')
    sensitivity = request.form.get('sensitivity')

    print(f"Search parameters - query: {query}, id: {request.args.get('id')}, matches: {matches}, sensitivity: {sensitivity}")

    # Validate matches parameter
    try:
        matches = int(matches) if matches else 10
        if matches <= 0 or matches > 20:
            matches = 10
    except Exception:
        matches = 10

    # Validate sensitivity parameter
    try:
        sensitivity = float(sensitivity) if sensitivity else 1.5
        if sensitivity <= 0 or sensitivity > 2:
            sensitivity = 1.5
    except Exception:
        sensitivity = 1.5

    images = []
    if query:
        images = get_images_from_db(
            query=query, 
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
        # Connect to database and fetch file path
        conn = sqlite3.connect('images.db')
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM images WHERE image_id = ?", (image_id,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            print(f"Image not found for ID: {image_id}")
            return "Image not found", 404

        file_path = result[0]
        if not os.path.exists(file_path):
            print(f"File not found at path: {file_path}")
            return "File not found", 404

        print(f"Serving image {image_id} from path: {file_path}")
        return send_file(file_path)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "Database error", 500
    except Exception as e:
        print(f"Error serving image: {e}")
        return "Server error", 500

@app.route('/rebuildindex', methods=['GET'])
def rebuild_index():
    # Code to rebuild the search index goes here
    search_similar(force_rebuild=True)
    return 'Done'

if __name__ == '__main__':
    app.run(debug=True, port=5000)