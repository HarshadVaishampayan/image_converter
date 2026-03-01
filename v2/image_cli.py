import click
import os
import sys
import subprocess
import time
import json
import traceback

# Add the v1 directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from convert_image import convert_image
from describe_image import describe_image
from persistence import PersistenceAPI
from caption_image import process_folder_parallel_caption, make_persistence_consumer, print_consumer
from find_similar import search_similar # Import search_similar
from tabulate import tabulate

@click.group()
def cli():
    """A command-line tool for image processing and search."""
    pass

@cli.command()
@click.option('--model', default='siglip', type=click.Choice(['clip', 'siglip']), help='Embedding model type to purge.')
def purge(model):
    """Delete all data from the database and reset the index."""
    if click.confirm(f'This will delete all indexed images and embeddings for {model.upper()}. Are you sure?'):
        api = PersistenceAPI(embedding_model_type=model)
        api._rebuild_index()
        click.echo("Database purged successfully.")

@cli.command()
@click.option('--limit', default=20, help='Maximum number of images to list.')
@click.option('--offset', default=0, help='Number of images to skip.')
@click.option('--model', default='siglip', type=click.Choice(['clip', 'siglip']), help='Embedding model type to list.')
def list_images(limit, offset, model):
    """List indexed images in the database."""
    api = PersistenceAPI(embedding_model_type=model)
    images = api.list_images(limit=limit, offset=offset)
    
    if not images:
        click.echo(f"No images found in the {model.upper()} collection.")
        return

    table_data = []
    for img in images:
        caps = ", ".join(img['captions'])
        if not caps:
            caps = "[MISSING]"
        elif len(caps) > 50: 
            caps = caps[:47] + "..."
        
        desc = img.get('description', '')
        if not desc:
            desc = "[MISSING]"
        elif len(desc) > 50: 
            desc = desc[:47] + "..."
        
        table_data.append([
            img['image_id'][:8], 
            os.path.basename(img['file_path']),
            caps,
            desc
        ])
    
    headers = ["ID (Short)", "Filename", "Captions", "Description"]
    click.echo(f"\n--- {model.upper()} Collection ---")
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

@cli.command()
@click.option('--model', default='siglip', type=click.Choice(['clip', 'siglip']), help='Embedding model type.')
def stats(model):
    """Show statistics about the indexed collection."""
    api = PersistenceAPI(embedding_model_type=model)
    count = api.collection.count()
    all_entries = api.collection.get(where={"type": "image"}, include=[])
    image_count = len(all_entries['ids'])
    
    click.echo(f"Model: {model.upper()}")
    click.echo(f"Total Unique Images: {image_count}")
    click.echo(f"Total Vector Entries: {count}")

@cli.command()
@click.option('--ip', required=True, help='Path to the image file or directory.')
@click.option('--embed-model', default='siglip', type=click.Choice(['clip', 'siglip']), help='Embedding model to use.')
@click.option('--cap-model', default='ollama', type=click.Choice(['florence2', 'ollama']), help='Captioning/Description model to use.')
@click.option('--chroma-path', default='./chroma_db', help='Path to the ChromaDB data directory.')
def describe(ip, embed_model, cap_model, chroma_path):
    """Describe images and index them using the selected models."""
    click.echo(f"-START INDEXING (Embed: {embed_model.upper()} | Cap: {cap_model.upper()})-")
    
    persist_consumer = make_persistence_consumer(path=chroma_path, embedding_model_type=embed_model)
    
    for _ in process_folder_parallel_caption(
        path=ip,
        consumers=[print_consumer, persist_consumer],
        embedding_model_type=embed_model,
        caption_model_type=cap_model
    ):
        pass
    click.echo("=END=")

@cli.command()
@click.option('--image', type=str, help='Path to the image to search with.')
@click.option('--query', type=str, help='Query string to search with.')
@click.option('--embed-model', default='siglip', type=click.Choice(['clip', 'siglip']), help='Embedding model to use.')
@click.option('--top-k', type=int, default=6, help='The number of top matches to return.')
@click.option('--sensitivity', type=float, default=0.5, help='The score threshold to filter results.')
def search(image, query, embed_model, top_k, sensitivity):
    """Find similar images using Hybrid Search (Embedding + Keywords)."""
    try:
        results = search_similar(
            query_text=query,
            image_path=image,
            top_k=top_k,
            sensitivity=sensitivity,
            embedding_model_type=embed_model
        )
            
        if not results:
            click.echo("No similar images found.")
            return

        for result in results:
            click.echo(f"Path: {result['file_path']}")
            click.echo(f"Score: {result['hybrid_score']:.4f} (Visual Dist: {result['score']:.4f})")
            click.echo(f"Captions: {result['captions']}")
            click.echo(f"Description: {result.get('description', '[MISSING]')}")
            click.echo("-" * 20)
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--embed-model', default='siglip', type=click.Choice(['clip', 'siglip']), help='Embedding model type of the collection to clean.')
def cleanup_embeddings(embed_model):
    """Finds and removes ChromaDB entries with empty text content (captions/descriptions)."""
    api = PersistenceAPI(embedding_model_type=embed_model)
    removed_count = api.cleanup_empty_text_embeddings()
    if removed_count > 0:
        click.echo(f"Successfully removed {removed_count} empty text embedding entries from the {embed_model.upper()} collection.")
    else:
        click.echo(f"No empty text embedding entries found in the {embed_model.upper()} collection.")

@cli.command()
@click.option('--image-path', required=True, help='Path to the image file to inspect.')
@click.option('--embed-model', default='siglip', type=click.Choice(['clip', 'siglip']), help='Embedding model type used for the image.')
def inspect(image_path, embed_model):
    """Inspect all stored data for a given image in the database."""
    api = PersistenceAPI(embedding_model_type=embed_model)
    image_data = api.get_full_image_data(image_path)

    if not image_data:
        click.echo(f"No data found for image: {image_path} in collection for {embed_model.upper()}")
        return

    click.echo(f"\n--- Inspection Results for {image_path} ({embed_model.upper()} Collection) ---")
    click.echo(f"Image ID: {image_data.get('image_id')}")
    click.echo(f"File Path: {image_data.get('file_path')}")
    click.echo(f"File Name: {image_data.get('file_name')}")
    click.echo(f"Dimensions: {image_data.get('width')}x{image_data.get('height')}")
    click.echo(f"Primary Embedding (first 5 vals): {image_data.get('embedding', [])[:5]}...")
    click.echo("\nMetadata:")
    for k, v in image_data.get('metadata', {}).items():
        click.echo(f"  {k}: {v}")
    
    click.echo("\nDescription (Unified):")
    click.echo(f"  {image_data.get('description')}")

    captions = image_data.get('captions')
    if captions:
        click.echo("\nCaptions:")
        for i, cap in enumerate(captions):
            click.echo(f"  - {cap}")

    desc_chunks = image_data.get('description_chunks')
    if desc_chunks:
        click.echo(f"\nDescription Chunks ({len(desc_chunks)} total):")
        for i, chunk in enumerate(desc_chunks):
            click.echo(f"  Chunk {i+1}:")
            click.echo(f"    Document: {chunk.get('document')}")
            click.echo(f"    Embedding Snippet (first 5 vals): {chunk.get('embedding_snippet', [])[:5]}...")
            click.echo(f"    Type: {chunk.get('metadata', {}).get('type')}")
    
    click.echo("-" * (len(image_path) + 40))

@cli.command()
@click.option('--host', default='127.0.0.1', help='The interface to bind to.')
@click.option('--port', default=5000, type=int, help='The port to listen on.')
@click.option('--debug', is_flag=True, help='Enable debug mode.')
def web(host, port, debug):
    """Run the Flask web application."""
    os.environ['FLASK_APP'] = 'gui.py'
    if debug: os.environ['FLASK_DEBUG'] = '1'
    flask_cmd = [sys.executable, '-m', 'flask', 'run', '--host', host, '--port', str(port)]
    subprocess.run(flask_cmd)

if __name__ == '__main__':
    cli()
