import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json # NEW: Import json for saving metadata

# NEW: Import Selenium components
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# --- Configuration ---
# Base directory where extracted content will be saved (now specifically for news)
BASE_OUTPUT_DIR = Path("./data/news")
# Maximum number of articles to scrape per site to prevent excessive downloads
MAX_ARTICLES_TO_SCRAPE_PER_SITE =  50
# Delay between requests to avoid overwhelming the server (in seconds)
REQUEST_DELAY = 1 
# Path to the file containing a list of URLs to scrape
URL_LIST_FILE = "urls_to_scrape.txt"

# Selenium specific configuration
CHROMEDRIVER_PATH = "./chromedriver" # Path to ChromeDriver in the current directory
SELENIUM_PAGE_LOAD_TIMEOUT = 30 # Max time to wait for a page to load in Selenium (seconds)
SCROLL_PAUSE_TIME = 2 # Time to pause after each scroll (seconds)
SCROLL_COUNT = 3 # Number of times to scroll down to load more content (adjust as needed)

# Number of parallel threads/browsers to run
MAX_WORKERS = 3 # Adjust based on your system's CPU/RAM/network capabilities


# --- Ensure Base Output Directory Exists ---
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Ensuring base output directory exists: {BASE_OUTPUT_DIR}")

def is_valid_url(url, base_url_parsed):
    """
    Checks if a URL is valid and is from the same domain as BASE_URL.
    """
    parsed_url = urlparse(url)
    return parsed_url.scheme in ['http', 'https'] and base_url_parsed.netloc == parsed_url.netloc

def initialize_driver():
    """Initializes and returns a headless Chrome WebDriver."""
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode (no UI)
    options.add_argument("--no-sandbox") 
    options.add_argument("--disable-dev-shm-usage") 
    options.add_argument("--disable-gpu") 
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36") 
    
    service = Service(CHROMEDRIVER_PATH)
    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(SELENIUM_PAGE_LOAD_TIMEOUT)
        return driver
    except WebDriverException as e:
        print(f"Error initializing WebDriver: {e}")
        print(f"Please ensure ChromeDriver is installed at {CHROMEDRIVER_PATH} and matches your Chrome browser version.")
        print("You can download it from: https://chromedriver.chromium.org/downloads")
        return None

def get_page_content_selenium(driver, url):
    """
    Fetches the content of a given URL using Selenium to handle dynamic content.
    """
    try:
        driver.get(url)
        WebDriverWait(driver, SELENIUM_PAGE_LOAD_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        for _ in range(SCROLL_COUNT):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
        
        return driver.page_source
    except TimeoutException:
        print(f"  Timeout loading page: {url}. Content might be incomplete.")
        return driver.page_source 
    except WebDriverException as e:
        print(f"  Error fetching {url} with Selenium: {e}")
        return None

def extract_text_from_html(html_content):
    """
    Extracts main readable text from HTML content using BeautifulSoup.
    More aggressive removal of irrelevant/UI elements for cleaner article text.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    for element in soup([
        "script", "style", "nav", "footer", "header", "aside", "form",
        "iframe", "img", "button", "input", "select", "textarea", 
        "noscript", "meta", "link", 
        "svg", "canvas", "audio", "video", "source", "track", 
        ".sidebar", ".ad", ".advertisement", ".promo", ".widget", 
        "[id*='ad']", "[class*='ad']", "[id*='promo']", "[class*='promo']", 
        "[role='banner']", "[role='contentinfo']", "[role='navigation']", 
        ".comments", "[id*='comment']", "[class*='comment']", 
        ".feed-item-vote", ".vote-buttons", 
        ".user-info", ".user-flair", 
        ".icon", ".thumb", ".spoiler", 
        ".bottom-bar", ".top-bar" 
    ]):
        element.extract()

    text = soup.get_text()
    
    lines = (line.strip() for line in text.splitlines())
    cleaned_text = '\n'.join(line for line in lines if line)
    return cleaned_text

def scrape_single_site(base_url_input):
    """
    Scrapes a single site: initializes its own driver, scrapes links, and saves content.
    Designed to be run in a separate thread.
    """
    driver = None
    try:
        driver = initialize_driver()
        if not driver:
            print(f"  Skipping {base_url_input} due to WebDriver initialization failure.")
            return

        BASE_URL = base_url_input
        parsed_base_url = urlparse(BASE_URL)

        # Get current timestamp for the folder name
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Use the actual domain as site_origin for metadata, clean for folder name
        site_origin_for_metadata = parsed_base_url.netloc
        site_name_for_folder = site_origin_for_metadata.replace('.', '_').replace('www_','')

        output_dir_with_timestamp = BASE_OUTPUT_DIR / f"{site_name_for_folder}_{timestamp_str}"
        output_dir_with_timestamp.mkdir(parents=True, exist_ok=True)
        print(f"Articles from {BASE_URL} will be saved to: {output_dir_with_timestamp}")

        visited_urls = set()
        urls_to_visit_level_1 = []

        print(f"Starting scrape from main page: {BASE_URL}")
        main_page_html = get_page_content_selenium(driver, BASE_URL)
        
        if main_page_html:
            visited_urls.add(BASE_URL)
            soup = BeautifulSoup(main_page_html, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(BASE_URL, href)

                if is_valid_url(full_url, parsed_base_url) and full_url not in visited_urls:
                    if any(ext in full_url for ext in ['.html', '.htm', '.php', '.asp', '.aspx', '/post/', '/article/', '/news/']):
                        urls_to_visit_level_1.append(full_url)
                        visited_urls.add(full_url)
                    elif parsed_base_url.netloc == "patriots.win" and "/p/" in full_url:
                        urls_to_visit_level_1.append(full_url)
                        visited_urls.add(full_url)
                    elif not urlparse(full_url).path.strip('/') or not urlparse(full_url).fragment:
                        urls_to_visit_level_1.append(full_url)

        print(f"Found {len(urls_to_visit_level_1)} unique links on the main page for {BASE_URL}.")

        articles_scraped_count = 0
        for i, article_url in enumerate(urls_to_visit_level_1):
            if articles_scraped_count >= MAX_ARTICLES_TO_SCRAPE_PER_SITE:
                print(f"Reached maximum articles ({MAX_ARTICLES_TO_SCRAPE_PER_SITE}) for {BASE_URL}. Stopping.")
                break

            print(f"Processing article {i+1}/{len(urls_to_visit_level_1)} from {BASE_URL}: {article_url}")
            article_html = get_page_content_selenium(driver, article_url)
            
            if article_html:
                article_text = extract_text_from_html(html_content=article_html)
                if article_text and len(article_text) > 200:
                    # Create a simple filename from the URL path, ensuring it's valid for FS
                    filename_base = urlparse(article_url).path.strip('/').replace('/', '_').replace('.', '_')
                    if not filename_base:
                        filename_base = "index_article_" + str(i)
                    
                    # Ensure filename is not too long and is unique
                    safe_filename_base = filename_base[:100].strip('_') # Limit length
                    url_hash = str(hash(article_url))[:8] 
                    final_filename = f"{safe_filename_base}_{url_hash}"

                    # Define paths for text and metadata files
                    text_output_path = output_dir_with_timestamp / f"{final_filename}.txt"
                    metadata_output_path = output_dir_with_timestamp / f"{final_filename}.json"

                    # Prepare metadata dictionary
                    article_metadata = {
                        "publication_date": timestamp_str, # Use the folder's timestamp
                        "site_origin": site_origin_for_metadata, # Use the cleaned domain
                        "url": article_url, # Original URL of the article
                        "category": "news" # Explicitly tag the category
                    }

                    try:
                        # Save the article text
                        with open(text_output_path, "w", encoding="utf-8") as f:
                            f.write(article_text)
                        
                        # Save the metadata
                        with open(metadata_output_path, "w", encoding="utf-8") as f:
                            json.dump(article_metadata, f, indent=4)

                        print(f"  Saved: {text_output_path} and {metadata_output_path}")
                        articles_scraped_count += 1
                    except IOError as e:
                        print(f"  Error saving files for {article_url}: {e}")
                else:
                    print(f"  Skipping {article_url}: No substantial text extracted or content too short.")
            
            time.sleep(REQUEST_DELAY)

        print(f"\nScraping for {BASE_URL} complete. Total articles saved: {articles_scraped_count}")
    finally:
        if driver:
            driver.quit() # Ensure the browser is closed after scraping this site

if __name__ == "__main__":
    print(f"Attempting to read URLs from {URL_LIST_FILE}")
    if not Path(URL_LIST_FILE).exists():
        print(f"Error: URL list file '{URL_LIST_FILE}' not found.")
        print("Please create this file in the same directory as scrape_news.py and add one URL per line.")
    else:
        with open(URL_LIST_FILE, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]

        if not urls:
            print(f"No URLs found in '{URL_LIST_FILE}'. Exiting.")
        else:
            print(f"Found {len(urls)} URLs to scrape. Starting parallel scraping with {MAX_WORKERS} workers...")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                executor.map(scrape_single_site, urls)
            print("\nAll scraping tasks submitted and completed.")

