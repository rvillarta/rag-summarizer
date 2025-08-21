import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json 
import re 
import shutil # Added for shutil.rmtree for clean restart

# NEW: Import for LLM and subprocess for starting llamafile
from langchain_openai import ChatOpenAI
import subprocess

# Selenium components
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options # FIX: Added missing import for Options

# --- Configuration ---
# Base directory where extracted content will be saved (now specifically for news)
BASE_OUTPUT_DIR = Path("./data/news")
# Maximum number of articles to scrape per site to prevent excessive downloads
MAX_ARTICLES_TO_SCRAPE_PER_SITE = 3
# Delay between requests to avoid overwhelming the server (in seconds)
REQUEST_DELAY = 1 
# Path to the file containing a list of URLs to scrape
URL_LIST_FILE = "urls_to_scrape.txt"

# Selenium specific configuration
CHROMEDRIVER_PATH = "./chromedriver" # Path to ChromeDriver in the current directory
SELENIUM_PAGE_LOAD_TIMEOUT = 30 # Max time to wait for a page to load in Selenium (seconds)
SCROLL_PAUSE_TIME = 2 # Time to pause after each scroll (seconds)
SCROLL_COUNT = 3 # Number of times to scroll down to load more content (adjust as needed)

# LLM Configuration for Scraper's Summarization
LLAMAFILE_NAME = "Mistral-7B-Instruct-v0.3.Q5_K_M.llamafile" 
LLAMAFILE_API_BASE = "http://localhost:8080/v1" # Ensure this matches your llamafile server
GPU_OFFLOAD_LAYERS = 999 

LLAMAFILE_SUMMARIZER_PROMPT = """You are a highly concise summarizer.
Given the following article text, provide a summary that is 1 to 3 sentences long.
Focus on the main topic and key takeaway. This summary will be used as a short headline description.

Article:
{article_text}

Summary:"""

# Logging for Llamafile server startup
LLAMAFILE_LOG_FILE = "llamafile_server.log"

# Number of parallel threads/browsers to run
MAX_WORKERS = 3 # Adjust based on your system's CPU/RAM/network capabilities


# --- Ensure Base Output Directory Exists ---
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Ensuring base output directory exists: {BASE_OUTPUT_DIR}")

# NEW: Moved from app.py
def start_llamafile_server():
    """
    Starts the llamafile server as a subprocess and waits for it to become ready.
    Implements a robust retry mechanism, logging llamafile's output to a file.
    """
    llamafile_path = Path(LLAMAFILE_NAME)
    if not llamafile_path.exists():
        print(f"Error: Llamafile '{LLAMAFILE_NAME}' not found in the current directory.")
        print("Please download it from Hugging Face and place it here, then run 'chmod +x {LLAMAFILE_NAME}'.")
        return None, "Llamafile executable not found."

    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLAMAFILE_API_BASE, api_key="sk-no-key-required", timeout=5.0) 
        client.models.list() 
        print("Llamafile server appears to be already running and responsive.")
        return None, None # Return None for process and error if already running
    except Exception:
        pass # Llamafile not running, proceed to start it

    print(f"Attempting to start llamafile server with GPU offload: {LLAMAFILE_NAME} -ngl {GPU_OFFLOAD_LAYERS}...")
    print(f"Llamafile server output will be logged to: {LLAMAFILE_LOG_FILE}")

    log_file_handle = open(LLAMAFILE_LOG_FILE, 'w') # Open in write mode to clear previous logs
    try:
        command_str = ( # Moved command_str definition to ensure it's always defined before use in subprocess.Popen
            f"./{llamafile_path.name} --server --host 0.0.0.0 --port 8080 "
            f"-ngl {GPU_OFFLOAD_LAYERS} > {LLAMAFILE_LOG_FILE} 2>&1"
        )
        if GPU_OFFLOAD_LAYERS == 0:
            command_str = f"./{llamafile_path.name} --server --host 0.0.0.0 --port 8080 > {LLAMAFILE_LOG_FILE} 2>&1"

        process = subprocess.Popen(
            command_str, 
            cwd=os.getcwd(),
            shell=True # Use shell=True to run the command string as is
        )
        print(f"Llamafile server (with GPU offload) started with PID: {process.pid}.")
        print("Waiting for llamafile server to become ready (up to 240 seconds)...")

        start_time = time.time()
        timeout = 240
        check_interval = 5

        from openai import OpenAI
        while time.time() - start_time < timeout:
            try:
                client = OpenAI(base_url=LLAMAFILE_API_BASE, api_key="sk-no-key-required", timeout=10.0)
                client.models.list() # A small API call to check if the server is responsive
                print("\nLlamafile server successfully connected and is ready!")
                return process, None # Return the process object and no error
            except Exception:
                if process.poll() is not None: # Check if process terminated unexpectedly
                    print("\nLlamafile server process terminated unexpectedly during startup.")
                    print(f"Please check '{LLAMAFILE_LOG_FILE}' for error details.")
                    return None, f"Llamafile process terminated unexpectedly. Check {LLAMAFILE_LOG_FILE}"
                time.sleep(check_interval)
        
        # If loop finishes, it timed out
        if process.poll() is None:
            print("\nLlamafile server did not become ready within the timeout period.")
            print(f"Please check '{LLAMAFILE_LOG_FILE}' for potential startup logs.")
            return None, f"Llamafile server timed out during startup. Check {LLAMAFILE_LOG_FILE}"
        else: # Process terminated, but not caught by process.poll() inside loop
            print("\nLlamafile server process terminated unexpectedly during startup check (after loop).")
            print(f"Please check '{LLAMAFILE_LOG_FILE}' for error details.")
            return None, f"Llamafile process terminated unexpectedly. Check {LLAMAFILE_LOG_FILE}"

    except Exception as e:
        print(f"Failed to launch llamafile server. Error details: {e}")
        print(f"Please ensure '{LLAMAFILE_NAME}' is executable (chmod +x) and compatible with your system.")
        return None, f"Failed to launch llamafile server: {e}"
    finally:
        log_file_handle.close() # Ensure the log file handle is closed

def initialize_llm_for_summarization():
    """Initializes and returns an LLM instance for summarization within the scraper."""
    try:
        llm = ChatOpenAI(
            model_name="llamafile",
            openai_api_key="sk-no-key-required",
            openai_api_base=LLAMAFILE_API_BASE,
            temperature=0.0, # Keep low for concise summaries
            request_timeout=60.0 # Timeout for LLM calls during scraping
        )
        # Test connection again to be sure after server start
        llm.invoke("Hello", max_tokens=10) # Small test call
        print("LLM for scraper summarization initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM for scraper summarization: {e}")
        return None

def is_valid_url(url, base_url_parsed):
    """
    Checks if a URL is valid and is from the same domain as BASE_URL.
    """
    parsed_url = urlparse(url)
    return parsed_url.scheme in ['http', 'https'] and base_url_parsed.netloc == parsed_url.netloc

def initialize_driver():
    """Initializes and returns a headless Chrome WebDriver."""
    options = Options() # Options class is now imported
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
    Prioritizes common article/main content elements for cleaner text.
    If specific containers are not found, it falls back to a broader cleanup.
    Also extracts the <title> tag text.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract title first
    page_title = soup.find('title')
    title_text = page_title.get_text(strip=True) if page_title else ""

    # Aggressively remove common irrelevant UI elements
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
        ".bottom-bar", ".top-bar",
        ".header-container", ".footer-container", ".navbar", ".masthead",
        ".social-share", ".read-more", ".related-articles", ".pagination",
        ".search-form", ".breadcrumb", ".sub-menu", ".menu-item", ".menu",
        ".byline", 
        ".comment-form", ".tags", ".categories", ".article-meta",
        "figcaption", 
        "figure" 
    ]):
        element.extract()

    # Try to find common article/main content containers
    content_elements = soup.find_all([
        'article', 'main', 
        lambda tag: tag.name == 'div' and (
            'article' in tag.get('class', []) or 'post' in tag.get('class', []) or 
            'story' in tag.get('class', []) or 'main-content' in tag.get('id', '') or 
            'content' in tag.get('id', '') or 'entry-content' in tag.get('class', []) or
            'article-body' in tag.get('class', []) or 'post-body' in tag.get('class', []) or
            'news-article' in tag.get('class', []) or 'news-content' in tag.get('class', []) or
            'entry-content' in tag.get('id', '') or 'article__content' in tag.get('class', []) or
            'article-text' in tag.get('class', []) or 'itemBody' in tag.get('class', []) or
            'post-content' in tag.get('class', [])
        ),
        lambda tag: tag.name == 'section' and (
            'article' in tag.get('class', []) or 'main-content' in tag.get('id', '')
        ),
    ])

    text_parts = []
    if content_elements:
        for tag in content_elements:
            text_parts.append(tag.get_text(separator=' ', strip=True)) 
    else:
        text_parts.append(soup.get_text(separator=' ', strip=True)) 
    
    raw_text = " ".join(text_parts)
    cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()

    return cleaned_text, title_text # Return both cleaned text and title

def scrape_single_site(base_url_input, llm_summarizer):
    """
    Scrapes a single site: initializes its own driver, scrapes links, and saves content.
    Generates a summary for each article using the provided LLM.
    Designed to be run in a separate thread.
    """
    BASE_URL = base_url_input # Moved BASE_URL assignment here to ensure it's always defined
    driver = None
    articles_saved_for_site = 0 # Counter for this site
    try:
        driver = initialize_driver()
        if not driver:
            print(f"  Skipping {BASE_URL} due to WebDriver initialization failure.")
            return

        parsed_base_url = urlparse(BASE_URL)

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        site_origin_for_metadata = parsed_base_url.netloc
        site_name_for_folder = site_origin_for_metadata.replace('.', '_').replace('www_','')

        output_dir_with_timestamp = BASE_OUTPUT_DIR / f"{site_name_for_folder}_{timestamp_str}"
        output_dir_with_timestamp.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Starting scrape for {BASE_URL} ---")
        print(f"  Articles will be saved to: {output_dir_with_timestamp}")

        visited_urls = set()
        urls_to_visit_level_1 = []

        print(f"  Scanning main page: {BASE_URL}")
        main_page_html = get_page_content_selenium(driver, BASE_URL)
        
        if main_page_html:
            visited_urls.add(BASE_URL)
            soup = BeautifulSoup(main_page_html, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(BASE_URL, href)

                if is_valid_url(full_url, parsed_base_url) and full_url not in visited_urls:
                    path = urlparse(full_url).path.lower()
                    
                    is_potential_article = False
                    if re.search(r'/\d{4}/\d{2}/\d{2}/', path) or re.search(r'/\d{2}-\d{2}-\d{4}/', path):
                        is_potential_article = True
                    elif any(keyword in path for keyword in ['/article/', '/story/', '/news/', '/post/', '/politics/', '/us/', '/world/', '/economy/', '/tech/', '/health/']): 
                        is_potential_article = True
                    elif (path.endswith(('.html', '.htm')) and 
                          any(keyword in path for keyword in ['news', 'politics', 'world', 'us', 'business', 'tech', 'health', 'economy'])): 
                        is_potential_article = True
                    elif parsed_base_url.netloc == "patriots.win" and "/p/" in path:
                        is_potential_article = True

                    excluded_patterns = [
                        '/category/', '/tag/', '/author/', '/topic/', '/about/', '/contact/', '/privacy/',
                        '/terms/', '/archive/', '/feed/', '/rss/', '/sitemap.xml', '/ads.txt',
                        '/community/', '/communities', 
                        '/login', '/register', '/signup', '/profile/', '/user/',
                        '/legal/', '/jobs/', '/careers/', '/shop/', '/store/', '/donate/', 
                        '/faq/', '/help/', '/support/', '/apps/',
                        '/video/', '/videos/', '/watch', 'youtube.com', 'youtu.be', 'vimeo.com', 
                        '/live/', '/livestream/', 
                        '.mp4', '.mov', '.avi', '.wmv', '.flv' 
                    ]
                    is_explicitly_excluded = any(ep in full_url for ep in excluded_patterns) or (path == '/' or path == '/index.html') 

                    if is_potential_article and not is_explicitly_excluded:
                        urls_to_visit_level_1.append(full_url)
                        visited_urls.add(full_url)
        else:
            print(f"  Warning: Could not fetch main page content for {BASE_URL}. Skipping link discovery.")

        print(f"  Found {len(urls_to_visit_level_1)} unique potential article links on the main page for {BASE_URL}.")

        for i, article_url in enumerate(urls_to_visit_level_1):
            if articles_saved_for_site >= MAX_ARTICLES_TO_SCRAPE_PER_SITE:
                print(f"  Reached maximum articles ({MAX_ARTICLES_TO_SCRAPE_PER_SITE}) for {BASE_URL}. Stopping.")
                break

            print(f"  Processing article {i+1}/{len(urls_to_visit_level_1)} from {BASE_URL}: {article_url}")
            article_html = get_page_content_selenium(driver, article_url)
            
            if article_html:
                article_text, page_title = extract_text_from_html(html_content=article_html)
                
                if article_text and len(article_text) > 200: # Ensure substantial text is extracted
                    # NEW: Generate article summary
                    article_summary = ""
                    if llm_summarizer:
                        try:
                            summary_messages = [
                                {"role": "user", "content": LLAMAFILE_SUMMARIZER_PROMPT.format(article_text=article_text[:4000])} # Limit text to first 4000 chars for summary
                            ]
                            summary_response = llm_summarizer.invoke(summary_messages)
                            article_summary = summary_response.content.strip()
                            # print(f"    Generated summary for {page_title or article_url[:50]}: {article_summary[:100]}...") # Too verbose for many articles
                        except Exception as e:
                            print(f"    Error generating summary for {article_url}: {e}. Skipping summary generation.")
                    else:
                        print("    LLM summarizer not available, skipping summary generation.")


                    filename_base = urlparse(article_url).path.strip('/').replace('/', '_').replace('.', '_')
                    if not filename_base:
                        filename_base = "index_article_" + str(i)
                    
                    safe_filename_base = filename_base[:100].strip('_') 
                    url_hash = str(hash(article_url))[:8] 
                    final_filename = f"{safe_filename_base}_{url_hash}"

                    text_output_path = output_dir_with_timestamp / f"{final_filename}.txt"
                    metadata_output_path = output_dir_with_timestamp / f"{final_filename}.json"

                    article_metadata = {
                        "publication_date": timestamp_str, 
                        "site_origin": site_origin_for_metadata, 
                        "url": article_url, 
                        "category": "news", 
                        "original_title": page_title,
                        "article_summary": article_summary # NEW: Add the generated summary
                    }

                    try:
                        # Save the article text
                        with open(text_output_path, "w", encoding="utf-8") as f:
                            f.write(article_text)
                        
                        # Save the metadata (including the summary)
                        with open(metadata_output_path, "w", encoding="utf-8") as f:
                            json.dump(article_metadata, f, indent=4)

                        print(f"  Saved article: {text_output_path.name}")
                        articles_saved_for_site += 1
                    except IOError as e:
                        print(f"  Error saving files for {article_url}: {e}")
                else:
                    print(f"  Skipping {article_url}: No substantial text extracted or content too short (< 200 chars).")
            else:
                print(f"  Skipping {article_url}: Failed to fetch HTML content.")
            
            time.sleep(REQUEST_DELAY)

        print(f"\n--- Scraping for {BASE_URL} complete. Total articles saved: {articles_saved_for_site} ---")
    except Exception as e:
        print(f"  An error occurred during scraping {BASE_URL}: {e}")
    finally:
        if driver:
            driver.quit() 

if __name__ == "__main__":
    print(f"Attempting to read URLs from {URL_LIST_FILE}")
    if not Path(URL_LIST_FILE).exists():
        print(f"Error: URL list file '{URL_LIST_FILE}' not found.")
        print("Please create this file in the same directory as scrape_news.py and add one URL per line.")
    else:
        urls = []
        with open(URL_LIST_FILE, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]

        if not urls:
            print(f"No URLs found in '{URL_LIST_FILE}'. Exiting.")
        else:
            llamafile_process = None
            llm_for_scraper = None
            total_urls_processed = len(urls)
            total_articles_saved = 0
            
            try:
                # Attempt to start llamafile server (or confirm it's running)
                llamafile_process, error_message = start_llamafile_server()
                if error_message:
                    print(f"Failed to start/connect to Llamafile server: {error_message}")
                    print("Cannot proceed with scraping as LLM for summarization is required.")
                else:
                    llm_for_scraper = initialize_llm_for_summarization()
                    if not llm_for_scraper:
                        print("LLM for summarization could not be initialized after server start. Skipping scraping.")
                    else:
                        print(f"Found {total_urls_processed} URLs to scrape. Starting parallel scraping with {MAX_WORKERS} workers...")
                        # Use a partial function to pass the llm_summarizer to each thread
                        from functools import partial
                        scrape_func = partial(scrape_single_site, llm_summarizer=llm_for_scraper)
                        
                        # Using ThreadPoolExecutor.map to get results from each task
                        # No need to process results directly here for total_articles_saved, as scrape_single_site prints per-site totals
                        list(ThreadPoolExecutor(max_workers=MAX_WORKERS).map(scrape_func, urls))
                        
                        print("\nAll scraping tasks submitted and completed.")
                        print(f"\nOverall scraping session finished.")

            except Exception as e:
                print(f"An unexpected error occurred during scraper execution: {e}")
            finally:
                if llamafile_process:
                    print("Terminating llamafile server from scraper...")
                    llamafile_process.terminate()
                    llamafile_process.wait(timeout=5)
                    if llamafile_process.poll() is None:
                        print("Llamafile process did not terminate gracefully, forcing kill.")
                        llamafile_process.kill()

