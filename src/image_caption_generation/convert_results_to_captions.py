import csv
import os

# Paths (edit if needed)
RESULTS_CSV = '/Users/kapil.singhal/CascadeProjects/image_captioning/resources/input/flickr30k/flickr30k_images/results.csv'
CAPTIONS_CSV = '/Users/kapil.singhal/CascadeProjects/image_captioning/resources/input/flickr30k/flickr30k_images/captions.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(CAPTIONS_CSV), exist_ok=True)

with open(RESULTS_CSV, 'r', encoding='utf-8') as infile, open(CAPTIONS_CSV, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile, delimiter='|', skipinitialspace=True)
    print(f"[DEBUG] Parsed fieldnames: {reader.fieldnames}")
    writer = csv.writer(outfile)
    writer.writerow(['image', 'caption'])
    row_count = 0
    total_rows = 0
    for row in reader:
        total_rows += 1
        if row['image_name'] is None or row['comment'] is None:
            print(f"[DEBUG] Skipping row due to missing data: {row}")
            continue
        image = row['image_name'].strip()
        caption = row['comment'].strip()
        writer.writerow([image, caption])
        row_count += 1

print(f"Converted {RESULTS_CSV} to {CAPTIONS_CSV} (columns: image, caption)")
print(f"Total rows from results.csv: {total_rows}")
print(f"Total rows from captions.csv: {row_count}")
