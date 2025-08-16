import argparse
from describe_image import describe_image

def main():
    parser = argparse.ArgumentParser(description='Describe an image using Ollama LLaVA model.')
    parser.add_argument('--ip', required=True, dest='image_path', help='Path to the image file')
    parser.add_argument('--m', dest='model', default='llava:7b-v1.5-q4_1', help='Ollama model to use (default: llava:7b-v1.5-q4_1)')
    args = parser.parse_args()

    print(f"Invoking describe_image with parameters: image_path={args.image_path}, model={args.model}")
    description = describe_image(args.image_path, model=args.model)
    print(description)

if __name__ == '__main__':
    main()
