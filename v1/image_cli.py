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
from caption_image import process_folder_parallel_blip, make_persistence_consumer, print_consumer
from find_similar import search_similar # Import search_similar

@click.group()
def cli():
    """A command-line tool for image processing and search."""
    pass

@cli.command()
@click.option('--ip', required=True, help='Path to the input image.')
@click.option('--op', help='Path to save the converted image (optional; default: n_<input_image> in same folder).')
@click.option('--w', required=True, type=int, help='Width of the output image.')
@click.option('--h', type=int, help='Height of the output image (optional; if not set, height is scaled proportionally).')
@click.option('--f', required=True, type=click.Choice(['jpeg', 'png']), help='Output image format (jpeg or png).')
@click.option('--c', type=click.Choice(['greyscale', 'rgb']), help='Color mode: greyscale or rgb (optional).')
def convert(ip, op, w, h, f, c):
    """Convert and resize an image."""
    params = {
        'input_path': ip,
        'output_path': op,
        'width': w,
        'format': f
    }
    if h is not None:
        params['height'] = h
    if c is not None:
        params['color'] = c

    try:
        convert_image(params)
        click.echo('Image conversion complete.')
    except ValueError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)


@cli.command()
@click.option('--ip', required=True, help='Path to the image file or directory.')
@click.option('--m', default='llava:7b-v1.5-q4_1', help='Model to use (default: llava:7b-v1.5-q4_1).')
@click.option('--provider', default='ollama', type=click.Choice(['ollama', 'huggingface']), help='Provider to use: ollama (default) or huggingface.')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda']), help='Device to use for Hugging Face: auto (default), cpu, or cuda.')
@click.option('--db-path', default='images.db', help='Path to the SQLite database for persistence.')
@click.option('--dim', default=512, type=int, help='Embedding dimension for persistence.')
def describe(ip, m, provider, device, db_path, dim):
    """Describe an image or images in a directory using Ollama or Hugging Face models, and persist results."""
    click.echo("-START-")
    start_str = f"START: {ip}"
    click.echo(start_str)
    param_str = f"Invoking describe with parameters: image_path={ip}, model={m}, provider={provider}, device={device}"
    click.echo(param_str)
    
    output_lines = []
    output_lines.append(start_str)
    output_lines.append(param_str)

    start_time = time.time()

    is_dir = os.path.isdir(ip)

    if is_dir:
        click.echo(f"Processing directory: {ip}")
        cache_file = "processed_cache.jsonl" # This is from caption_image.py original logic
        persist_consumer = make_persistence_consumer(db_path=db_path, dim=dim)
        
        for _ in process_folder_parallel_blip(
            path=ip,
            consumers=[print_consumer, persist_consumer],
            cache_file=cache_file,
        ):
            pass
        description = "Directory processing complete."
    else:
        try:
            description = describe_image(ip, model=m, provider=provider, device=device)
        except Exception as e:
            description = f"Error describing image: {e}\n{traceback.format_exc()}"
            click.echo(description, err=True)
            sys.exit(1)
    
    time_taken = time.time() - start_time
    click.echo(description)
    output_lines.append(description)
    time_str = f"Time Taken: {time_taken:.2f} seconds"
    click.echo(time_str)
    output_lines.append(time_str)
    
    end_str = f"END: {ip}"
    click.echo(end_str)
    output_lines.append(end_str)

    image_dir = os.path.dirname(ip) if not is_dir else ip
    metadata_path = os.path.join(image_dir, 'metadata.txt')
    
    # In run_describe_image.py, metadata.txt was deleted and appended to. Replicating behavior.
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    with open(metadata_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        f.write('\n' + ('-'*40) + '\n')
    click.echo("=END=")


@cli.command()
@click.option('--image', type=str, help='Path to the image to search with.')
@click.option('--query', type=str, help='Query string to search with.')
@click.option('--rebuild-index', is_flag=True, help='Force rebuild of FAISS index.')
@click.option('--top-k', type=int, default=6, help='The number of top matches to return.')
@click.option('--sensitivity', type=float, default=1.5, help='The score threshold to filter results.')
def search(image, query, rebuild_index, top_k, sensitivity):
    """Find similar images based on image or query string."""
    try:
        results = search_similar(
            query_text=query,
            image_path=image,
            force_rebuild=rebuild_index,
            top_k=top_k,
            sensitivity=sensitivity
        )
            
        if not results:
            click.echo("No similar images found.")
            return

        for result in results:
            click.echo(f"Path: {result['file_path']}")
            click.echo(f"Score: {result['score']:.4f}")
            click.echo(f"Captions: {result['captions']}")
            click.echo("-" * 20)
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--host', default='127.0.0.1', help='The interface to bind to.')
@click.option('--port', default=5000, type=int, help='The port to listen on.')
@click.option('--debug', is_flag=True, help='Enable debug mode.')
def web(host, port, debug):
    """Run the Flask web application."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gui_path = os.path.join(current_dir, 'gui.py')
    
    cmd = [sys.executable, gui_path]
    if host != '127.0.0.1':
        cmd.extend(['--host', host]) # gui.py doesn't currently support --host, but Flask's app.run does.
                                   # We'll need to modify gui.py to accept this if we want to use it.
    if port != 5000:
        cmd.extend(['--port', str(port)]) # Same as above for --port
    if debug:
        cmd.append('--debug') # Same as above for --debug

    click.echo(f"Starting web server on http://{host}:{port} with debug={debug}...")
    click.echo("Press Ctrl+C to stop.")
    
    try:
        # We need to run gui.py directly using python -m flask run or update gui.py to handle args
        # For now, let's assume gui.py runs with default host/port and debug is controlled within it.
        # To make it truly configurable from cli, gui.py needs modification.
        # For simplicity, let's just run gui.py, and it will use its own app.run parameters.
        # Or, we can use flask run command.
        
        # A more robust way to run a Flask app from a CLI entry point:
        # Set environment variables for Flask
        os.environ['FLASK_APP'] = 'gui.py'
        if debug:
            os.environ['FLASK_DEBUG'] = '1'
        
        # Call flask run directly
        # The command should be 'flask run --host <host> --port <port>'
        flask_cmd = [sys.executable, '-m', 'flask', 'run', '--host', host, '--port', str(port)]
        
        # If gui.py has a __name__ == '__main__': block with app.run(),
        # we can just execute gui.py and it will respect its own args.
        # But if we want to pass args from 'image_cli web', we need to adapt gui.py
        # or use `flask run`. Given the current gui.py structure, it's better to use `flask run`.

        subprocess.run(flask_cmd)

    except FileNotFoundError:
        click.echo("Error: Flask is not found. Please ensure it's installed.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error starting web server: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()
