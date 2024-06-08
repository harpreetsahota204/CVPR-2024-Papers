#!/usr/bin/env python3

import argparse
from pathlib import Path
import time
from tqdm import tqdm
import pandas as pd
import arxiv

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Function to extract arXiv ID from URL
    def extract_arxiv_id(url):
        if isinstance(url, str):
            return url.split('/')[-1]
        return ""

    # Create the pdfs directory
    pdf_dir = Path('arxiv_pdfs')
    pdf_dir.mkdir(exist_ok=True)

    # Ensure 'pdf_path' column is treated as string from the beginning
    if 'pdf_path' not in df.columns:
        df['pdf_path'] = ""
    else:
        df['pdf_path'] = df['pdf_path'].astype(str)

    # Iterate over the DataFrame with tqdm for progress monitoring
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Initialize the arXiv client
        client = arxiv.Client()
        arxiv_id = extract_arxiv_id(row['arXiv_link'])

        # Skip rows with no valid arXiv ID
        if not arxiv_id:
            continue

        try:
            # Construct the search query
            search = arxiv.Search(id_list=[arxiv_id])

            # Fetch the first result
            first_result = next(client.results(search))

            # Define the PDF path
            pdf_path = pdf_dir / f"{arxiv_id}.pdf"

            # Update DataFrame with the title and summary
            df.at[index, 'arXiv_title'] = first_result.title
            df.at[index, 'summary'] = first_result.summary
            df.at[index, 'primary_category'] = first_result.primary_category
            df.at[index, 'categories'] = first_result.categories
            
            # Download the PDF if it doesn't already exist
            if not pdf_path.exists():
                first_result.download_pdf(dirpath=pdf_dir, filename=f"{arxiv_id}.pdf")
                df.at[index, 'pdf_path'] = f"{arxiv_id}.pdf"

        except Exception as e:
            print(f"Error fetching data for arXiv ID {arxiv_id}: {e}")

        # Save progress every 10 iterations
        if index % 10 == 0:
            df.to_csv(output_csv, index=False)
            time.sleep(5)

    # Final save
    df.to_csv(output_csv, index=False)
    print("DataFrame update complete and saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download PDFs from arXiv and update CSV.")
    parser.add_argument('input_csv', help="Path to the input CSV file")
    parser.add_argument('output_csv', help="Path to the output CSV file")

    args = parser.parse_args()
    main(args.input_csv, args.output_csv)