import argparse
from convert_image import convert_image


def main():
    parser = argparse.ArgumentParser(description='Convert and resize an image.')
    parser.add_argument('--ip', required=True, dest='input_path', help='Path to the input image')
    parser.add_argument('--op', dest='output_path', help='Path to save the converted image')
    parser.add_argument('--w', required=True, dest='width', help='Width of the output image')
    parser.add_argument('--h', dest='height', help='Height of the output image (optional)')
    parser.add_argument('--f', required=True, dest='format', choices=['jpeg', 'png'], help='Output image format (jpeg or png)')
    parser.add_argument('--c', dest='color', choices=['greyscale', 'rgb'], help='Color mode: greyscale or rgb (optional)')
    args = parser.parse_args()

    params = {
        'input_path': args.input_path,
        'output_path': args.output_path,
        'width': args.width,
        'format': args.format
    }
    if args.height:
        params['height'] = args.height
    if args.color:
        params['color'] = args.color

    convert_image(params)
    print('Image conversion complete.')

if __name__ == '__main__':
    main()
