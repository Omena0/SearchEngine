from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.robotparser import RobotFileParser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect
from threading import Thread
import urllib.parse
import numpy as np
import validators
import time as t
import requests
import warnings
import cursor
import urllib
import json
import gzip
import nltk
import sys
import os

headers = {
    'Accept-Language': 'en-US',
    'User-Agent': 'FrCrawler',
    'From': 'omena0mc@gmail.com'
}

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

os.system('cls')
cursor.hide()

doCrawl = True

def compress_url(url):
    # Remove query and fragment
    url = normalize_url(url)

    # Remove scheme and useless www.
    url = url.removeprefix('https://').removeprefix('http://').removeprefix('www.')

    return url.strip()

robotsCache = {}
robotsFileCache = {}
def is_allowed_by_robots(url):
    if url in robotsCache:
        return robotsCache[url]

    parsed_url = urllib.parse.urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    robots_url = urllib.parse.urljoin(base_url, "/robots.txt")

    if robots_url in robotsFileCache:
        return robotsFileCache[robots_url].can_fetch(headers['User-Agent'], url)

    try:
        rp = _get_robots_file(robots_url, url)
    except:
        return True  # Default True

    result = rp.can_fetch(headers['User-Agent'], url)
    robotsCache[url] = result

    return result

def _get_robots_file(robots_url, url):
    rp = RobotFileParser(robots_url)
    rp.read()

    robotsFileCache[robots_url] = rp

    return rp

def normalize_url(url):
    if not url: return None
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'

    # Removes query and fragment parts
    try:
        parsed = urllib.parse.urlparse(url)
        return urllib.parse.urlunparse(
            (
                parsed.scheme or 'https',
                parsed.netloc,
                parsed.path,
                parsed.params,
                '',
                ''
            )
        ).removesuffix('/')
    except Exception:
        return None

def extract_keywords(text):
    # Tokenize and clean
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    keywords = cleanup_keywords([
        token.strip()
        for token in tokens
        if token.isalnum()
        and token not in stop_words
        and any(c.isalpha() for c in token)
    ])

    # TF-IDF for keyword weighting
    vectorizer = TfidfVectorizer(max_features=150)
    tfidf_matrix = vectorizer.fit_transform([' '.join(keywords)])
    feature_names = vectorizer.get_feature_names_out()

    return [url for url,score in sorted(
        zip(feature_names, tfidf_matrix.toarray()[0]),
        key=lambda x: x[1],
        reverse=True,
    )[:30]]

def extract_meaningful_text(soup):
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # Drop blank lines
    text = ' '.join(chunk for chunk in chunks if chunk)

    return text

def cleanup_keywords(keywords:list):
    return [
        keyword.strip()
        for keyword in keywords
        if (keyword.isalnum() and any(c.isalpha() for c in keyword)) and keyword.strip != ''
    ]


### Actual crawler
def crawl(url):
    try:
        url = normalize_url(url)

        # Can we crawl this?
        if not url or not validators.url(url):
            return None

        if url in crawled or url in sites:
            return None

        if not is_allowed_by_robots(url):
            return None

        if not doCrawl:
            return None

        # Parse html
        response = requests.get(url, timeout=0.5, headers=headers)
        site = response.text
        soup = BeautifulSoup(site, 'html.parser')
        text = extract_meaningful_text(soup)

        # Make sure we have a real website
        if response.status_code != 200:
            return None

        outgoing_links = []

        # Get links
        for link in soup.find_all('a'):
            href = link.get('href')
            href = normalize_url(href)

            if not doCrawl:
                return None

            # Can we crawl this?
            if not href or not validators.url(href):
                continue

            if not is_allowed_by_robots(href):
                continue

            href = compress_url(href)

            if href in to_crawl or href in crawled or href in sites:
                continue

            # Add to outgoing
            outgoing_links.append(href)

            # Add to jobs
            to_crawl.add(href)

            # Conserve some memory
            while len(to_crawl) > 50_000:
                to_crawl.pop()

        # Extract page metadata
        return { # page_data
            'url': compress_url(url),
            'title': soup.title.string if soup.title else '',
            'lang': detect(soup.title.string),
            'keywords': extract_keywords(text),
            'links': outgoing_links
        }

    except Exception:
        return None


# Load sites
if os.path.exists('sites.json.gz'):
    with gzip.open('sites.json.gz') as f:
        sites = json.load(f)

    print(f'Loaded {len(sites)} entries from sites.json.gz')

else:
    sites = {}
    print('Creating sites.')

# Load to_crawl.json.gz
if os.path.exists('to_crawl.json.gz'):
    with gzip.open('to_crawl.json.gz') as f:
        to_crawl = set(json.load(f))

    print(f'Loaded {len(to_crawl)} entries from to_crawl.json.gz')

else:
    to_crawl = {'https://en.wikipedia.org/wiki/Main_Page', 'https://minecraft.wiki/', 'https://google.com/', 'https://github.com/'}
    print('Initialized to_crawl')

# Load crawled.json.gz
if os.path.exists('crawled.json.gz'):
    with gzip.open('crawled.json.gz') as f:
        crawled = set(json.load(f))

    print(f'Loaded {len(crawled)} entries from crawled.json.gz')

else:
    crawled = set()
    print('Initialized crawled')


def worker():
    while doCrawl:
        while not to_crawl and doCrawl:
            t.sleep(0.3)

        url = to_crawl.pop()
        crawled.add(compress_url(url))
        page_data = crawl(url)

        if not doCrawl:
            return

        if page_data:
            sites[url] = page_data


workers = []

print('Starting crawl...')


for _ in range(150):
    w = Thread(target=worker,daemon=True)
    workers.append(w)
    w.start()

print(f'{len(workers)} Threads started. (ctrl-c to stop crawling)')

try:
    while True:
        print(f'Crawled: {len(sites):6<} To crawl: {len(to_crawl):6<} \r',end='')

except KeyboardInterrupt:
    doCrawl = False


print(f'Saving {len(sites)} entries to sites.json.gz')
with gzip.open('sites.json.gz', 'wt') as f:
    json.dump(sites, f, separators=(',', ':'))

all_keywords = set()


# Cleanup keywords
for url, page in sites.items():
    all_keywords.update(page['keywords'])

all_keywords = sorted(cleanup_keywords(all_keywords))

# Save everything
with open('keywords.txt', 'w', errors='ignore') as f:
    f.writelines([f'{i}\n' for i in all_keywords])

print(f'Saving {len(to_crawl)} entries to to_crawl.json.gz')
with gzip.open('to_crawl.json.gz', 'wt') as f:
    json.dump(list(to_crawl), f, separators=(',', ':'))

print(f'Saving {len(crawled)} entries to crawled.json.gz')
with gzip.open('crawled.json.gz', 'wt') as f:
    json.dump(list(crawled), f, separators=(',', ':'))

print('Done!')

sys.exit(0)
