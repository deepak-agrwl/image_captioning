import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Paths
image_path = '../../resources/input/flickr30k/flickr30k_images/flickr30k_images/1000366164.jpg'
csv_path = '../../resources/input/flickr30k/flickr30k_images/results.csv'

# Load captions
with open(csv_path, 'r') as f:
    lines = [line.rstrip(',\n') for line in f]
    rows = [line.split('|') for line in lines]
df = pd.DataFrame(rows[1:], columns=[col.strip() for col in rows[0]])
captions = df[df['image_name'].str.strip() == '1000366164.jpg']['comment'].str.strip().tolist()

# Load image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image and captions
plt.figure(figsize=(6, 7))
plt.imshow(image)
plt.axis('off')
plt.title('1000092795.jpg')

# Display captions below the image
caption_text = '\n'.join([f'{i+1}. {cap}' for i, cap in enumerate(captions)])
plt.figtext(0.5, 0.01, caption_text, wrap=True, ha='center', fontsize=12)
plt.show()