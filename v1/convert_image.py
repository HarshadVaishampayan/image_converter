from PIL import Image
import piexif

def convert_image(inputimageparams):
	"""
	Takes a dictionary (map) of input image parameters (strings to strings)
	and performs image conversion logic.
	Optional parameter:
		color: 'greyscale' or 'rgb' (case-insensitive)
	"""
	import os
	input_path = inputimageparams.get('input_path')
	output_path = inputimageparams.get('output_path')
	width = inputimageparams.get('width')
	height = inputimageparams.get('height')
	fmt = inputimageparams.get('format', '').lower()
	color = inputimageparams.get('color', '').lower() if inputimageparams.get('color') else None
	# Accept both 'jpeg' and 'jpg' as JPEG
	if fmt == 'jpg':
		fmt = 'jpeg'

	if fmt not in ('jpeg', 'png'):
		raise ValueError('Invalid format. Only jpeg and png are supported.')
	if not input_path or not width:
		raise ValueError('input_path and width are required.')

	if not os.path.exists(input_path):
		raise ValueError(f"Input file does not exist: {input_path}")

	if not output_path or output_path.strip() == '':
		dir_name = os.path.dirname(input_path)
		base_name = os.path.basename(input_path)
		output_file = 'n_' + base_name
		output_path = os.path.join(dir_name, output_file)

	width = int(width)
	# Overwrite output file if it exists
	if os.path.exists(output_path):
		os.remove(output_path)
	with Image.open(input_path) as img:
		orig_width, orig_height = img.size
		exif_bytes = None
		if fmt == 'jpeg':
			try:
				exif_dict = piexif.load(img.info.get('exif', b''))
				exif_bytes = piexif.dump(exif_dict)
			except Exception:
				exif_bytes = None
		if height:
			height = int(height)
		else:
			# Maintain aspect ratio
			height = int((width / orig_width) * orig_height)
		img = img.resize((width, height))
		# Convert color if requested
		if color == 'greyscale':
			img = img.convert('L')
		elif color == 'rgb':
			img = img.convert('RGB')
		save_kwargs = {'format': fmt.upper()}
		if exif_bytes and fmt == 'jpeg':
			save_kwargs['exif'] = exif_bytes
		img.save(output_path, **save_kwargs)
