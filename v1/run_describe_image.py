import argparse
from describe_image import describe_image

def main():
    parser = argparse.ArgumentParser(description='Describe an image using Ollama or Hugging Face models.')
    parser.add_argument('--ip', required=True, dest='image_path', help='Path to the image file')
    parser.add_argument('--m', dest='model', default='llava:7b-v1.5-q4_1', help='Model to use (default: llava:7b-v1.5-q4_1)')
    parser.add_argument('--provider', dest='provider', default='ollama', choices=['ollama', 'huggingface'], help='Provider to use: ollama (default) or huggingface')
    parser.add_argument('--device', dest='device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use for Hugging Face: auto (default), cpu, or cuda')
    args = parser.parse_args()

    print("-START-")
    output_lines = []
    start_str = f"START: {args.image_path}"
    end_str = f"END: {args.image_path}"
    print(start_str)
    output_lines.append(start_str)
    param_str = f"Invoking describe_image with parameters: image_path={args.image_path}, model={args.model}, provider={args.provider}, device={args.device}"
    print(param_str)
    output_lines.append(param_str)
    import time
    start_time = time.time()
    description = describe_image(args.image_path, model=args.model, provider=args.provider, device=args.device)
    time_taken = time.time() - start_time
    print(description)
    output_lines.append(description)
    time_str = f"Time Taken: {time_taken:.2f} seconds"
    print(time_str)
    output_lines.append(time_str)
    print(end_str)
    output_lines.append(end_str)

    import os
    image_dir = os.path.dirname(args.image_path)
    metadata_path = os.path.join(image_dir, 'metadata.txt')
    # Delete metadata.txt if it exists
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    with open(metadata_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        f.write('\n' + ('-'*40) + '\n')
    print("=END=")

if __name__ == '__main__':
    main()
