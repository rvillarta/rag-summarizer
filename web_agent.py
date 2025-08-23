import os
import shutil
import time
from pathlib import Path
import json
from datetime import datetime
import re
import queue
import requests 

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate 
from langchain.docstore.document import Document as LcDocument

from bs4 import BeautifulSoup 

# Tool imports for web search
from googlesearch import search as google_search_tool # For Google search
from duckduckgo_search import DDGS # NEW: For DuckDuckGo search

# --- Configuration (from app.py for consistency) ---
BASE_DATA_DIR = Path("data")
WEB_DIR = BASE_DATA_DIR / "web"
BASE_VECTOR_DB_DIR = Path("vectorstores")
WEB_KB_NAME = "web_content"
WEB_VECTOR_DB_PATH = BASE_VECTOR_DB_DIR / WEB_KB_NAME

# --- Prompts for Web Agent's LLM (kept for potential future use) ---
WEB_SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are an expert summarizer. Your task is to condense the following web page content into a concise, informative summary.
    Focus on the main topic, key facts, and any significant claims or arguments.
    Maintain neutrality and avoid adding external information.
    
    Web Page Content (Source: {source_url}):
    {context}

    Concise Summary:"""
)

class WebAgent:
    def __init__(self, llm: ChatOpenAI, embeddings_model: SentenceTransformerEmbeddings, 
                 retriever_settings: dict, web_kb_path: Path, response_queue: queue.Queue,
                 current_step: list, total_steps: int, debug_logging_enabled: bool,
                 web_agent_settings: dict):
        
        self.llm = llm 
        self.embeddings_model = embeddings_model
        self.retriever_settings = retriever_settings
        self.web_kb_path = web_kb_path
        self.response_queue = response_queue
        self.current_step = current_step
        self.total_steps = total_steps
        self.debug_logging_enabled = debug_logging_enabled
        self.web_agent_settings = web_agent_settings 

        self.web_vectorstore = self._initialize_web_vectorstore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.retriever_settings.get("text_splitter", {}).get("chunk_size", 1000),
            chunk_overlap=self.retriever_settings.get("text_splitter", {}).get("chunk_overlap", 200)
        )
        self.last_scrape_time = {} 
        
        self._log_debug(f"WebAgent initialized. Web KB Path: {self.web_kb_path}")

    def _log_debug(self, message):
        """Helper for conditional debug logging within the agent."""
        if self.debug_logging_enabled:
            print(f"DEBUG [WebAgent]: {message}")

    def _update_progress(self, stage_name: str):
        """Helper to update initialization progress."""
        self.current_step[0] += 1
        percentage = int((self.current_step[0] / self.total_steps) * 100)
        self.response_queue.put({"type": "init_progress", "stage": stage_name, "percentage": percentage})
        print(f"[app.py] Progress: {stage_name} ({percentage}%)") 

    def _initialize_web_vectorstore(self):
        """Initializes or loads the persistent web vector store."""
        self._log_debug(f"Initializing web vector store at {self.web_kb_path}")
        if self.web_kb_path.exists() and any(self.web_kb_path.iterdir()):
            web_vectorstore = Chroma(
                persist_directory=str(self.web_kb_path),
                embedding_function=self.embeddings_model
            )
            self._log_debug("Existing web vector store loaded successfully.")
        else:
            web_vectorstore = Chroma.from_documents(
                documents=[], 
                embedding=self.embeddings_model,
                persist_directory=str(self.web_kb_path)
            )
            self._log_debug("New web vector store created.")
        return web_vectorstore

    def set_llm(self, llm: ChatOpenAI):
        """Sets the LLM instance for the WebAgent after it's fully initialized."""
        self.llm = llm
        self._log_debug("WebAgent's LLM instance set.")

    def perform_web_search(self, query: str) -> list: # FIX: Removed num_results parameter
        """
        Performs a web search across multiple providers and returns a combined list of top URLs.
        """
        self._log_debug(f"Performing multi-provider web search for: '{query}'")
        all_urls = set() # Use a set to automatically handle deduplication
        
        # Get search settings from config
        max_search_results = self.web_agent_settings.get("max_search_results", 3)
        search_providers = self.web_agent_settings.get("search_providers", ["google"]) # Default to Google

        if "google" in search_providers:
            try:
                self._log_debug(f"  Searching Google for '{query}' (top {max_search_results} results)")
                urls_generator = google_search_tool(query, num_results=max_search_results, lang='en')
                for url in urls_generator:
                    if url and (url.startswith('http://') or url.startswith('https://')):
                        all_urls.add(url)
                        self._log_debug(f"    Google URL: {url}")
            except Exception as e:
                print(f"Error during Google search for '{query}': {e}")
        
        if "duckduckgo" in search_providers:
            try:
                self._log_debug(f"  Searching DuckDuckGo for '{query}' (top {max_search_results} results)")
                ddg_results = DDGS().text(keywords=query, max_results=max_search_results)
                for result in ddg_results:
                    url = result.get('href')
                    if url and (url.startswith('http://') or url.startswith('https://')):
                        all_urls.add(url)
                        self._log_debug(f"    DuckDuckGo URL: {url}")
            except Exception as e:
                print(f"Error during DuckDuckGo search for '{query}': {e}")

        if "bing" in search_providers:
            print(f"  Bing search requested. Note: Direct free scraping of Bing is unreliable.")
            print(f"  Consider using an API like Azure Cognitive Search (Bing Web Search API) for reliable Bing results.")
            # Placeholder for Bing API integration
            # Example with requests to Bing API (requires API key)
            # bing_api_key = os.getenv("BING_SEARCH_V7_SUBSCRIPTION_KEY")
            # if bing_api_key:
            #     headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
            #     params = {"q": query, "count": max_search_results}
            #     response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
            #     response.raise_for_status()
            #     search_results = response.json()
            #     for result in search_results.get("webPages", {}).get("value", []):
            #         url = result.get("url")
            #         if url:
            #             all_urls.add(url)
            #             self._log_debug(f"    Bing API URL: {url}")
            # else:
            #     print("  BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable not set. Skipping Bing search.")
        
        final_urls = list(all_urls)
        self._log_debug(f"  Total unique URLs found across providers: {len(final_urls)}")
        return final_urls

    def _scrape_url_content(self, url: str) -> str:
        """
        Scrapes content from a single URL using requests and BeautifulSoup.
        Returns cleaned text content.
        """
        self._log_debug(f"Scraping content from URL: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            scrape_timeout = self.web_agent_settings.get("scrape_timeout", 10)
            response = requests.get(url, headers=headers, timeout=scrape_timeout)
            response.raise_for_status() 
            raw_content = response.text
            
            soup = BeautifulSoup(raw_content, 'html.parser')
            
            for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
                script_or_style.extract()
            
            text = soup.get_text()
            text = os.linesep.join([s for s in text.splitlines() if s.strip()])
            text = text.strip()
            
            self._log_debug(f"  Scraped {len(text)} characters from {url}")
            return text
        except requests.exceptions.RequestException as e:
            print(f"Network/HTTP error scraping URL '{url}': {e}")
            return ""
        except Exception as e:
            print(f"Error parsing/cleaning content from URL '{url}': {e}")
            return ""

    def scrape_and_ingest_urls(self, urls: list, original_query: str) -> list[LcDocument]:
        """
        Scrapes raw content from a list of URLs, processes it, and ingests into the web vector store.
        Returns a list of Langchain Documents representing the scraped content.
        """
        newly_ingested_docs = []

        for i, url in enumerate(urls):
            self.response_queue.put({"type": "web_progress", "stage": f"Scraping & ingesting web page {i+1}/{len(urls)}: {url}"})
            
            if url in self.last_scrape_time and (datetime.now() - self.last_scrape_time[url]).days < 1:
                self._log_debug(f"  Skipping recently scraped URL: {url}")
                continue

            scraped_text = self._scrape_url_content(url)
            if not scraped_text:
                self._log_debug(f"  No content scraped from {url}. Skipping ingestion.")
                continue

            sanitized_filename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', url)
            if len(sanitized_filename) > 100: 
                sanitized_filename = sanitized_filename[:90] + "_" + str(hash(url))[-8:]
            
            file_path_base = WEB_DIR / f"{sanitized_filename}.txt"
            metadata_path = WEB_DIR / f"{sanitized_filename}.json"

            try:
                with open(file_path_base, 'w', encoding='utf-8') as f:
                    f.write(scraped_text)
                
                metadata = {
                    "source": f"web_scrape_{sanitized_filename}.txt",
                    "file_path": str(file_path_base),
                    "category": WEB_KB_NAME,
                    "original_url": url,
                    "scraped_at": datetime.now().isoformat(),
                    "title": self._extract_title_from_url(url), 
                    "raw_content_length": len(scraped_text)
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                self.last_scrape_time[url] = datetime.now()
                self._log_debug(f"  Stored raw web content for {url} to {file_path_base}")

                lc_doc = LcDocument(page_content=scraped_text, metadata=metadata)
                
                splits = self.text_splitter.split_documents([lc_doc])
                
                if splits:
                    self.web_vectorstore.add_documents(splits)
                    self._log_debug(f"  Ingested {len(splits)} chunks from {url} into web_vectorstore.")
                    newly_ingested_docs.extend(splits)
                else:
                    self._log_debug(f"  No valid splits generated for {url}. Skipping add to vectorstore.")

            except Exception as e:
                print(f"Error storing/ingesting scraped content for {url}: {e}")
        
        self.web_vectorstore.persist() 
        return newly_ingested_docs

    def get_web_retriever(self, k: int = 4, score_threshold: float = None):
        """
        Returns a retriever for the web vector store, applying configured settings.
        """
        search_kwargs = {"k": k}
        search_type = self.retriever_settings.get("search_type", "similarity")

        if search_type == "mmr":
            search_kwargs["fetch_k"] = self.retriever_settings.get("fetch_k", 20)
            search_kwargs["lambda_mult"] = self.retriever_settings.get("lambda_mult", 0.5)
        
        return self.web_vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def _extract_title_from_url(self, url: str) -> str:
        """Attempts to extract a readable title from a URL."""
        path_parts = Path(url).parts
        if path_parts:
            last_part = path_parts[-1]
            title = re.sub(r'\.(html|htm|php|asp|aspx)$', '', last_part, flags=re.IGNORECASE)
            title = title.replace('-', ' ').replace('_', ' ').strip()
            title = ' '.join(word.capitalize() for word in title.split())
            if not title:
                return f"Web Page from {url.split('//')[-1].split('/')[0]}"
            return title
        return f"Web Page from {url.split('//')[-1].split('/')[0]}" 

