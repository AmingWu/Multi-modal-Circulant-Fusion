import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
VCR_IMAGES_DIR = os.path.join('/dev/shm/', 'data', 'vcr1images')
VCR_ANNOTS_DIR = os.path.join('/dev/shm/', 'data')

if not os.path.exists(VCR_IMAGES_DIR):
    raise ValueError("Update config.py with where you saved VCR images to.")