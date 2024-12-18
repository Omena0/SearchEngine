from flask import Flask, request, render_template, jsonify
from collections import defaultdict
from functools import lru_cache
from fuzzywuzzy import fuzz
import urllib.parse
import math
import json
import gzip
import re

app = Flask(__name__)

# Load the search index
try:
    with gzip.open('../crawler/search_index.json.gz') as f:
        search_index = json.load(f)
except FileNotFoundError:
    print("Error: search_index.json file not found.")
    search_index = {}

# Create an inverted index for faster searching
inverted_index = defaultdict(list)
for page in search_index:
    for keyword in page['keywords']:
        inverted_index[keyword].append(page)

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
        )
    except Exception as e:
        print(e)
        return None

title_keywords = 1.5
fuzz_bonus   = 3_0
exact_bonus  = 3.0
path_bonus   = 2_0
home_bonus   = 5.0

popular_pages = {'wikipedia.org','github.com','youtube.com','google.com'}

@lru_cache(maxsize=250)
def search(query, lang=None, start=0, num=None,):  # sourcery skip: low-code-quality
    print(f"Searching for: {query} lang:{lang} [{start}:{num}]")
    query_terms = query.lower().split()
    matching_pages = defaultdict(float)

    # Phrase matching
    query_phrase = re.escape(query.lower())

    for term in query_terms:
        if term in inverted_index:
            for page in inverted_index[term]:
                page['url'] = normalize_url(page['url'])

                # Lang parameter
                if lang and page['lang'] != lang:
                    continue

                # Use TF-IDF and PageRank for scoring
                tf = page['keywords'].count(term) / len(page['keywords'])
                idf = math.log(len(search_index) / len(inverted_index[term]))
                tfidf = tf * idf
                score = tfidf * page['pagerank'] + 1

                # Fuzzy matching
                if page['title']:
                    score *= fuzz.partial_ratio(term, page['title'].lower())/(100-fuzz_bonus)

                    # Phrase matching
                    if re.search(query_phrase, page['title'].lower()):
                        score *= title_keywords

                    # Boost exact matches
                    if page['title'].lower().strip() == query.lower().strip():
                        score *= exact_bonus

                # Prioritize short urls
                parsed_url = urllib.parse.urlparse(page['url'])
                score *= path_bonus/(len(parsed_url.path+parsed_url.netloc)+10)

                # Prioritize home pages
                if parsed_url.path in {'','/'}:
                    score *= home_bonus

                # Boost popular pages
                if any({i in page['url'] for i in popular_pages}) and score > 4:
                    score *= 1.5

                matching_pages[page['url']] += score

    # Sort results by score
    sorted_results = sorted(matching_pages.items(), key=lambda x: x[1], reverse=True)

    # Return top results
    return len(sorted_results), [
        {
            'url': url,
            'title': next((page['title'] for page in inverted_index[query_terms[0]] if page['url'] == url), ''),
            'score': round(score,4)
        }
        for url, score in sorted_results[start:start+num if num else None]
        if score >= 0.1
    ]

def getParam(text: str, param: str) -> tuple:
    param_start = text.find(f"{param}:")
    if param_start != -1:
        param_end = text.find(" ", param_start)
        if param_end == -1:
            param_end = len(text)
        param_value = text[param_start + len(param) + 1:param_end]
        modified_query = text[:param_start] + text[param_end:]
        return modified_query.strip(), param_value
    return text.strip(), None

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search')
def search_api():
    query = request.args.get('q', None)
    start = request.args.get('start', 0, type=int)
    num = request.args.get('num', 50, type=int)
    if not query:
        return jsonify({'results': [], 'total_results': 0})  # Return an empty list and total_results as 0 if no query is provided

    # Parse advanced search params
    # Example query: 'wikipedia lang:fi'
    query, lang = getParam(query, 'lang')

    num, results = search(query, lang=lang, start=start, num=num)

    return jsonify({'results': results, 'total_results': num})

if __name__ == '__main__':
    app.run(debug=True)

