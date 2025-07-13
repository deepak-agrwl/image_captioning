import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert results.csv to captions.csv with specified columns.")
    parser.add_argument('--results_csv', type=str, required=True, help='Path to input results.csv file')
    parser.add_argument('--captions_csv', type=str, required=True, help='Path to output captions.csv file')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.captions_csv), exist_ok=True)

    with open(args.results_csv, 'r', encoding='utf-8') as infile, open(args.captions_csv, 'w', newline='', encoding='utf-8') as outfile:
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

    print(f"Converted {args.results_csv} to {args.captions_csv} (columns: image, caption)")
    print(f"Total rows from results.csv: {total_rows}")
    print(f"Total rows from captions.csv: {row_count}")

if __name__ == "__main__":
    main()