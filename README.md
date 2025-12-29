## Load annotations
<pre>
import json

# Path to the annotations file (relative to CWD)
json_path = 'data/phase_1/annotations/train.json'

# Load the JSON data
with open(json_path, 'r') as f:
    annotations_data = json.load(f)

# Check the structure of the JSON file
print(annotations_data.keys())  # This will show the top-level keys of the JSON

</pre>

## Load image
<pre>
import os

image_folder = 'data/phase_1/train'
# Get the image info for image_id 1 (example)
image_id = 1
image_info = next(image for image in annotations_data['images'] if image['id'] == image_id)
image_file = image_info['file_name']  # This gives us the image filename

# Full path to the image
image_path = f'{image_folder}/{image_file}'
print(f"Image path: {image_path}")

# Get the annotations for this image
image_annotations = [ann for ann in annotations_data['annotations'] if ann['image_id'] == image_id]

print(image_path, image_annotations)

</pre>

## Visualize image with annotations
<pre>
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Open the image
image = Image.open(image_path)

# Display the image
fig, ax = plt.subplots(1)
ax.imshow(image)

# Draw bounding boxes for each annotation
for ann in image_annotations:
    bbox = ann['bbox']  # [x, y, width, height]
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# Show the image with bounding boxes
plt.show()

</pre>
