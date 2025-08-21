#!/bin/bash

cd /home/rvillarta/rag-summarizer &&
source venv/bin/activate &&
python3 scrape.py 2>&1;
deactivate
