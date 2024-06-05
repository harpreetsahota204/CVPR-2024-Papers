import csv
import os
import pandas as pd
import re
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Constants
URL = 'https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers'
CSV_FILE = 'cvpr2024_accepted_papers.csv'
PROGRESS_FILE = 'cvpr2024_accepted_papers_with_arxiv_links.csv'
MAX_WORKERS = 1
REQUEST_DELAY = 1

def fetch_page(url):
    """
    Fetch the HTML content of the given URL.

    Args:
        url (str): The URL to fetch.

    Returns:
        str: The HTML content of the page.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_papers(html_content):
    """
    Parse the HTML content to extract paper titles, authors, and links.

    Args:
        html_content (str): The HTML content of the page.

    Returns:
        list: A list of lists containing paper title, authors, and link.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    papers_section = soup.find('h1', string='CVPR 2024 Accepted Papers').find_next('table')
    data = []

    for row in papers_section.find_all('tr'):
        td = row.find('td')
        if td:
            title_tag = td.find('a')
            link = title_tag['href'] if title_tag else None
            title = title_tag.get_text(strip=True) if title_tag else (td.find('strong').get_text(strip=True) if td.find('strong') else "No title found")
            authors_tag = td.find('i')
            authors = authors_tag.get_text(strip=True).replace(u'\xa0', u' ') if authors_tag else 'No authors listed'
            if title:
                data.append([title, authors, link if link else " "])
    return data

def write_to_csv(data, filename):
    """
    Write the extracted data to a CSV file.

    Args:
        data (list): The data to write to the CSV file.
        filename (str): The name of the CSV file.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Authors', 'Link'])
        writer.writerows(data)

def search_arxiv_link(title):
    """
    Search for the arXiv link of a paper using DuckDuckGo search.

    Args:
        title (str): The title of the paper.

    Returns:
        tuple: A tuple containing the arXiv link and another link if found.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(title, timelimit='y', max_results=2)
        if results:
            for result in results:
                href = result.get('href', '')
                if 'arxiv.org/abs' in href or 'arxiv.org/html' in href or 'cvpr.thecvf.com' in href:
                    return href, ''
            return '', results[0].get('href', '')
    except Exception as e:
        print(f"Error searching for title '{title}': {e}")
    return '', ''

def process_title(row):
    """
    Process a row to find the arXiv link and another link for the paper.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        tuple: A tuple containing the index, arXiv link, and another link.
    """
    index = row.name
    title = row['Title'] + " " + " ".join(row['Authors'].split('·')[:2])
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title).strip()
    arxiv_link, other_link = search_arxiv_link(title)
    return index, arxiv_link, other_link

def load_progress(filename, df):
    """
    Load the progress from a CSV file if it exists.

    Args:
        filename (str): The name of the progress CSV file.
        df (pd.DataFrame): The DataFrame containing the paper data.

    Returns:
        tuple: A tuple containing the progress DataFrame and the last processed index.
    """
    try:
        progress_df = pd.read_csv(filename)
        last_index = progress_df[['arXiv_link', 'other_link']].last_valid_index() + 1 if not progress_df[['arXiv_link', 'other_link']].isna().all().all() else 0
    except (FileNotFoundError, pd.errors.EmptyDataError):
        progress_df = df.copy()
        progress_df['arXiv_link'] = ''
        progress_df['other_link'] = ''
        last_index = 0
    return progress_df, last_index

def update_links(df, progress_df, last_index):
    """
    Update the DataFrame with arXiv and other links for each paper.

    Args:
        df (pd.DataFrame): The DataFrame containing the paper data.
        progress_df (pd.DataFrame): The DataFrame to store progress.
        last_index (int): The index to start processing from.
    """
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_title, row) for index, row in df.iloc[last_index:].iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Titles"):
            index, arxiv_link, other_link = future.result()
            progress_df.at[index, 'arXiv_link'] = arxiv_link
            progress_df.at[index, 'other_link'] = other_link
            time.sleep(REQUEST_DELAY)
            progress_df.to_csv(PROGRESS_FILE, index=False)

def main():
    """
    Main function to orchestrate the scraping and processing of papers.
    """
    html_content = fetch_page(URL)
    data = parse_papers(html_content)
    write_to_csv(data, CSV_FILE)
    print("Data has been written to cvpr2024_accepted_papers.csv")

    df = pd.read_csv(CSV_FILE)
    progress_df, last_index = load_progress(PROGRESS_FILE, df)
    update_links(df, progress_df, last_index)
    print("Processing complete.")

if __name__ == "__main__":
    main()