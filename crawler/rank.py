import urllib.parse
import numpy as np
import gzip
import json
import os

# Load search index
if os.path.exists('sites.json.gz'):
    with gzip.open('sites.json.gz') as f:
        sites = json.load(f)

    print(f'Loaded {len(sites)} entries from sites.json.gz')

else:
    sites = {}
    print('Creating sites.')

# Load crawled.json.gz
if os.path.exists('crawled.json.gz'):
    with gzip.open('crawled.json.gz') as f:
        crawled = set(json.load(f))

    print(f'Loaded {len(crawled)} entries from crawled.json.gz')

else:
    crawled = set()
    print('Initialized crawled')


class PageRankCalculator:
    def __init__(self, urls, damping_factor=0.85, epsilon=1e-8, max_iterations=100):
        self.urls = list(urls)
        self.n = len(urls)
        self.damping_factor = damping_factor
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.adjacency_matrix = np.zeros((self.n, self.n))

    def build_adjacency_matrix(self, link_graph):
        url_to_index = {url: idx for idx, url in enumerate(self.urls)}

        for source, destinations in link_graph.items():
            source_idx = url_to_index.get(source)
            if source_idx is None:
                continue

            for dest in destinations:
                dest_idx = url_to_index.get(dest)
                if dest_idx is not None:
                    self.adjacency_matrix[source_idx, dest_idx] = 1

        # Normalize columns
        column_sums = self.adjacency_matrix.sum(axis=0)
        self.adjacency_matrix /= column_sums + (column_sums == 0)

    def calculate(self, link_graph):
        self.build_adjacency_matrix(link_graph)

        pagerank = np.ones(self.n) / self.n

        for _ in range(self.max_iterations):
            prev_pagerank = pagerank.copy()

            pagerank = (1 - self.damping_factor) / self.n + \
                       self.damping_factor * self.adjacency_matrix.T.dot(prev_pagerank)

            if np.sum(np.abs(pagerank - prev_pagerank)) < self.epsilon:
                break

        return dict(zip(self.urls, pagerank))

# After crawling, calculate PageRank
def calculate_pagerank():
    pagerank_calculator = PageRankCalculator(crawled)
    return pagerank_calculator.calculate(sites)

def cleanup_keywords(keywords:list):
    return [
        keyword.strip()
        for keyword in keywords
        if (keyword.isalnum() and any(c.isalpha() for c in keyword)) and keyword.strip != ''
    ]

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
    except:
        return None


print('\nRanking pages...')

pagerank = calculate_pagerank()

search_index = []
search_index_urls = set()

# Update pagerank in search_index
for page in sites.values():
    page['url'] = normalize_url(page['url'])

    if page['url'] in search_index_urls:
        continue

    page['keywords'] = cleanup_keywords(page['keywords'])

    # Add pagerank
    page['pagerank'] = round(pagerank.get(page['url'], 0),6)

    # Remove keys other than url, title, keywords and pagerank
    page = {k: page[k] for k in ('url', 'title', 'lang', 'keywords', 'pagerank')}

    # Add to search index
    search_index.append(page)
    search_index_urls.add(page['url'])



print(f'Ranked {len(search_index)} pages')

print(f'Saving {len(search_index)} entries to search_index.json.gz')
with gzip.open('search_index.json.gz', 'wt') as f:
    json.dump(search_index, f, separators=(',', ':'))

print('Done!')
