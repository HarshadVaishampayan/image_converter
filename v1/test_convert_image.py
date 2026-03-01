import unittest
import os
import tempfile
from PIL import Image
import pytest
from convert_image import convert_image

def create_test_image(path, size=(100, 100), color=(255, 0, 0)):
    img = Image.new('RGB', size, color)
    img.save(path, format='JPEG')

def test_convert_image_resize_and_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'input.jpg')
        output_path = os.path.join(tmpdir, 'output.png')
        create_test_image(input_path, size=(200, 100))
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'width': '50',
            'format': 'png'
        }
        convert_image(params)
        assert os.path.exists(output_path)
        with Image.open(output_path) as out_img:
            assert out_img.size[0] == 50
            # Height should be scaled proportionally
            assert out_img.size[1] == 25
            assert out_img.format == 'PNG'

def test_convert_image_with_height():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'input.jpg')
        output_path = os.path.join(tmpdir, 'output.jpeg')
        create_test_image(input_path, size=(100, 100))
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'width': '80',
            'height': '60',
            'format': 'jpeg'
        }
        convert_image(params)
        assert os.path.exists(output_path)
        with Image.open(output_path) as out_img:
            assert out_img.size == (80, 60)
            assert out_img.format == 'JPEG'

def test_convert_image_greyscale():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'input.jpg')
        output_path = os.path.join(tmpdir, 'output_grey.jpeg')
        create_test_image(input_path, size=(100, 100))
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'width': '50',
            'format': 'jpeg',
            'color': 'greyscale'
        }
        convert_image(params)
        assert os.path.exists(output_path)
        with Image.open(output_path) as out_img:
            assert out_img.mode == 'L'
            assert out_img.size[0] == 50
            assert out_img.format == 'JPEG'

def test_convert_image_rgb():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'input.jpg')
        output_path = os.path.join(tmpdir, 'output_rgb.jpeg')
        create_test_image(input_path, size=(100, 100))
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'width': '50',
            'format': 'jpeg',
            'color': 'rgb'
        }
        convert_image(params)
        assert os.path.exists(output_path)
        with Image.open(output_path) as out_img:
            assert out_img.mode == 'RGB'
            assert out_img.size[0] == 50
            assert out_img.format == 'JPEG'

def test_invalid_format():
    with pytest.raises(ValueError):
        convert_image({'input_path': 'a', 'output_path': 'b', 'width': '10', 'format': 'bmp'})

def test_missing_required_params():
    with pytest.raises(ValueError):
        convert_image({'output_path': 'b', 'width': '10', 'format': 'png'})
    with pytest.raises(ValueError):
        convert_image({'input_path': 'a', 'width': '10', 'format': 'png'})
    with pytest.raises(ValueError):
        convert_image({'input_path': 'a', 'output_path': 'b', 'format': 'png'})