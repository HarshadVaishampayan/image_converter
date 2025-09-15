from flask import Flask, render_template, send_file, request
from find_similar import search_similar
import sqlite3
import os

app = Flask(__name__)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

def get_images_from_db(query: str = None, image_id: str = None):
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
            results = search_similar(image_path=result[0])
        else:
            # Use text query to search
            results = search_similar(query_text=query)

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

@app.route('/search')
def image_grid():
    """
    Display image grid based on search criteria.
    Query params:
        q: text query for semantic search
        id: image_id to find similar images
    """
    query = request.args.get('query')
    image_id = request.args.get('id')
    
    if not query and not image_id:
        # Return empty grid instead of error
        return render_template('grid.html', 
                            images=[], 
                            columns=3, 
                            query=None, 
                            reference_id=None)
        
    images = get_images_from_db(query=query, image_id=image_id)
    return render_template('grid.html', 
                         images=images, 
                         columns=3, 
                         query=query, 
                         reference_id=image_id)

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)