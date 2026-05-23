# @openai_report_only.py

import os
import re
import json
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def extract_query_from_html(soup):
    """
    Extract the user query from the HTML.
    The query is typically in the first user message.
    """
    # Find user messages
    user_messages = soup.find_all('div', {'data-message-author-role': 'user'})
    if user_messages:
        # Get the first user message
        first_user_msg = user_messages[0]
        # Find the text content
        text_div = first_user_msg.find('div', class_='whitespace-pre-wrap')
        if text_div:
            return text_div.get_text(strip=True)
    
    # Fallback: try to find text in the main content area
    return ""


def extract_report_from_html(html_content):
    """
    Parses the provided HTML content from a ChatGPT page to extract
    the main report content into markdown format.
    """
    soup = BeautifulSoup(html_content, 'lxml')

    # Strategy 1: Find the Report Container with deep-research-result class
    report_container = soup.find('div', class_='deep-research-result')

    if not report_container:
        # Strategy 2: Try to find markdown prose div (exact class match)
        report_container = soup.find('div', class_='markdown prose')
        
        if not report_container:
            # Strategy 3 & 4: Find all divs with markdown and prose in class, 
            # including those in assistant messages, and select the longest one
            def has_markdown_prose(x):
                if not x:
                    return False
                class_str = ' '.join(x) if isinstance(x, list) else str(x)
                return 'markdown' in class_str and 'prose' in class_str
            
            # Find all markdown prose divs (both standalone and in assistant messages)
            markdown_containers = []
            
            # First, try to find standalone markdown prose divs
            standalone_divs = soup.find_all('div', class_=has_markdown_prose)
            markdown_containers.extend(standalone_divs)
            
            # Then, find markdown prose divs in assistant messages
            assistant_msgs = soup.find_all('div', {'data-message-author-role': 'assistant'})
            for msg in assistant_msgs:
                markdown_div = msg.find('div', class_=has_markdown_prose)
                if markdown_div and markdown_div not in markdown_containers:
                    markdown_containers.append(markdown_div)
            
            if markdown_containers:
                # Select the longest one (most likely the main report)
                report_container = max(markdown_containers, key=lambda x: len(x.get_text()))
            else:
                print("Error: Could not find any suitable report container.")
                return None

    # Convert HTML to Markdown
    report_html = str(report_container)
    markdown_report = md(report_html, heading_style="ATX").strip()

    return markdown_report


def extract_url_mapping_from_report(markdown_report):
    """
    Extract URL mappings from the markdown report.
    Returns a dict mapping abbreviated domain names to lists of full URLs.
    If a domain name appears multiple times with different URLs, all URLs are preserved.
    Also returns a list of all unique URLs for reference.
    e.g., {'en.wikipedia.org': ['https://en.wikipedia.org/wiki/Page1', 'https://en.wikipedia.org/wiki/Page2']}
    """
    url_mapping = {}  # domain -> list of URLs
    all_urls = []
    
    # Pattern to find markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(link_pattern, markdown_report)
    
    for text, url in matches:
        # Extract domain from text (e.g., "ndtv.com" or "en.wikipedia.org")
        domain = text.strip()
        if url.startswith('http'):
            # Remove trailing slash from URL
            url = remove_trailing_slash(url)
            
            # Store the mapping - allow multiple URLs for same domain
            if domain not in url_mapping:
                url_mapping[domain] = []
            if url not in url_mapping[domain]:
                url_mapping[domain].append(url)
            
            # Also store by domain extracted from URL
            url_domain = extract_domain_from_url(url)
            if url_domain:
                if url_domain not in url_mapping:
                    url_mapping[url_domain] = []
                if url not in url_mapping[url_domain]:
                    url_mapping[url_domain].append(url)
            
            # Collect all URLs
            if url not in all_urls:
                all_urls.append(url)
    
    return url_mapping, all_urls


def remove_trailing_slash(url):
    """
    Remove trailing slash from URL.
    e.g., 'https://www.example.com/' -> 'https://www.example.com'
    """
    if url and url.endswith('/'):
        return url[:-1]
    return url


def extract_domain_from_url(url):
    """
    Extract the domain name from a URL.
    e.g., 'https://www.ndtv.com/path' -> 'ndtv.com'
    """
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        return match.group(1)
    return None


def convert_markdown_links_to_at_format(markdown_text):
    """
    Convert markdown hyperlinks [text](url) to @url format.
    e.g., [ndtv.com](https://www.ndtv.com/...) -> @https://www.ndtv.com/...
    Also adds commas to separate multiple consecutive @url patterns.
    e.g., @url1@url2@url3 -> @url1, @url2, @url3
    """
    # Pattern to match markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    def replace_link(match):
        url = match.group(2)
        # Remove trailing slash before converting
        url = remove_trailing_slash(url)
        return f' @{url} '
    
    # Replace all markdown links with @url format
    converted_text = re.sub(link_pattern, replace_link, markdown_text)
    
    # Add commas to separate consecutive @url patterns
    # Pattern: @url followed immediately by @url (no space or comma between)
    consecutive_url_pattern = r'(@https?://[^\s\)]+)(@https?://[^\s\)]+)'
    
    def add_comma(match):
        url1 = match.group(1)
        url2 = match.group(2)
        return f' {url1} , {url2} '
    
    # Keep replacing until no more consecutive @url patterns found
    prev_text = ''
    while prev_text != converted_text:
        prev_text = converted_text
        converted_text = re.sub(consecutive_url_pattern, add_comma, converted_text)
    
    # Clean up multiple consecutive spaces (but preserve single spaces)
    converted_text = re.sub(r' {2,}', ' ', converted_text)
    
    return converted_text


def parse_openai_html(html_content):
    """Parse OpenAI/ChatGPT deep research HTML trace into structured dict.

    Args:
        html_content: Raw HTML string of the ChatGPT deep research page.

    Returns:
        dict with keys: query, final_report, all_source_links, summary_citations, chain_of_research
    """
    soup = BeautifulSoup(html_content, 'lxml')

    result = {
        'query': '',
        'chain_of_research': {},
        'final_report': ''
    }

    # 1. Extract query
    result['query'] = extract_query_from_html(soup)

    # 2. Extract final report
    result['final_report'] = extract_report_from_html(html_content) or ''

    # 3. Extract URL mapping from report for later replacement (before converting links)
    url_mapping, all_urls = extract_url_mapping_from_report(result['final_report'])

    # 4. Extract research trajectory
    trajectory = extract_detailed_research_trajectory(html_content, url_mapping)
    result['chain_of_research'] = trajectory
    
    # 5. Convert markdown hyperlinks in final_report to @url format
    result['final_report'] = convert_markdown_links_to_at_format(result['final_report'])
    
    # 6. Extract all_source_links from all search entries
    all_source_links = []
    for key in sorted(trajectory.keys()):
        if key.startswith('search_'):
            urls = trajectory[key]
            if isinstance(urls, list):
                for url in urls:
                    if url:
                        url = remove_trailing_slash(url)
                        if url not in all_source_links:
                            all_source_links.append(url)
    result['all_source_links'] = all_source_links
    
    # 7. Extract summary_citations from final_report (@url format)
    summary_citations = []
    # Pattern to match @url format
    at_url_pattern = r'@(https?://[^\s\)]+)'
    matches = re.findall(at_url_pattern, result['final_report'])
    for url in matches:
        if url:
            url = remove_trailing_slash(url)
            if url not in summary_citations:
                summary_citations.append(url)
    result['summary_citations'] = summary_citations
    
    return result


def extract_detailed_research_trajectory(html_content, url_mapping):
    """
    Extract detailed research trajectory with reasoning and search pairs.
    Format: reasoning_1 -> search_1 -> reasoning_2 -> search_2 -> ...
    
    Strategy:
    1. Find all '<div class="flex items-start justify-start gap-2">' markers - each defines a search
    2. Extract URL (href) after each marker
    3. Extract reasoning text (from <p data-start="0">) before each marker
    """
    trajectory = {}
    
    # Step 1: Find all search markers
    search_marker = '<div class="flex items-start justify-start gap-2">'
    marker_positions = [m.start() for m in re.finditer(re.escape(search_marker), html_content)]
    
    if not marker_positions:
        return trajectory
    
    # Step 2: For each marker, extract URL and reasoning
    reasoning_idx = 1
    search_idx = 1
    
    # Find the trajectory start (look for "Searched for" to determine where trajectory begins)
    first_searched_for = re.search(r'Searched for', html_content)
    trajectory_start = first_searched_for.start() - 5000 if first_searched_for else 0
    trajectory_start = max(0, trajectory_start)
    
    # Filter markers to only those in the trajectory region
    marker_positions = [pos for pos in marker_positions if pos > trajectory_start]
    
    for i, pos in enumerate(marker_positions):
        # Extract ALL links after this marker, grouped by link text (domain name)
        after_content = html_content[pos:pos+3000]
        soup = BeautifulSoup(after_content, 'lxml')
        
        # Find all links with href
        links = soup.find_all('a', href=True)
        
        if links:
            # Group links by their text (domain name)
            # If multiple links have same text but different URLs, keep all URLs
            domain_to_urls = {}  # domain_name -> list of full URLs
            
            for link in links:
                href = link.get('href', '').strip()
                link_text = link.get_text(strip=True)
                
                if not href.startswith('http'):
                    continue
                
                # Extract domain from link text or from URL
                if link_text:
                    # Use link text as the key (e.g., "en.wikipedia.org")
                    domain_key = link_text
                else:
                    # If no text, extract domain from URL
                    domain_key = extract_domain_from_url(href) or href
                
                # Add this URL to the domain's list
                if domain_key not in domain_to_urls:
                    domain_to_urls[domain_key] = []
                
                # Use full URL from mapping if available and URL is incomplete
                domain = extract_domain_from_url(href)
                url_path = href.split(domain, 1)[1] if domain else ''
                
                if url_path and url_path != '/' and len(url_path) > 1:
                    # Full URL with path - use it directly (remove trailing slash)
                    href = remove_trailing_slash(href)
                    if href not in domain_to_urls[domain_key]:
                        domain_to_urls[domain_key].append(href)
                else:
                    # Short URL - try mapping
                    if domain and domain in url_mapping:
                        # url_mapping[domain] is now a list of URLs
                        mapped_urls = url_mapping[domain]
                        for mapped_url in mapped_urls:
                            mapped_url = remove_trailing_slash(mapped_url)
                            if mapped_url not in domain_to_urls[domain_key]:
                                domain_to_urls[domain_key].append(mapped_url)
                    else:
                        href = remove_trailing_slash(href)
                        if href not in domain_to_urls[domain_key]:
                            domain_to_urls[domain_key].append(href)
            
            # Flatten all URLs (preserving all URLs for same domain)
            urls = []
            for domain_key, url_list in domain_to_urls.items():
                for url in url_list:
                    if url not in urls:
                        urls.append(url)
            
            # Extract reasoning BEFORE this marker
            # For first marker: from trajectory_start to this marker
            # For subsequent markers: from END of previous marker's URL section to this marker
            if i == 0:
                before_start = trajectory_start
            else:
                # Skip past the previous marker's URL section (about 800 chars)
                before_start = marker_positions[i-1] + 800
            
            before_content = html_content[before_start:pos]
            reasoning = extract_reasoning_from_marker_region(before_content)
            
            # Add reasoning if exists
            if reasoning.strip():
                trajectory[f'reasoning_{reasoning_idx}'] = reasoning.strip()
                reasoning_idx += 1
            
            # Add search URLs (can be multiple for same domain name)
            trajectory[f'search_{search_idx}'] = urls
            search_idx += 1
    
    # Extract final reasoning after the last search
    if marker_positions:
        last_marker = marker_positions[-1]
        # Skip past the URL section
        final_start = last_marker + 800
        final_region = html_content[final_start:final_start+15000]
        final_reasoning = extract_reasoning_from_marker_region(final_region)
        if final_reasoning.strip():
            trajectory[f'reasoning_{reasoning_idx}'] = final_reasoning.strip()
    
    return trajectory


def extract_reasoning_from_marker_region(html_region):
    """
    Extract reasoning text from a region, looking for:
    1. class="mb-0! text-base" - contains main reasoning text  
    2. class="text-token-text-primary flex text-base" - contains "Searched for X" text
    3. <p data-start="0"> - fallback pattern
    """
    import re
    from bs4 import BeautifulSoup
    
    # Use a list to maintain order and track positions
    text_items = []  # List of (position, text)
    
    # Pattern 1: class="mb-0! text-base" followed by text in <p> tags
    pattern1 = r'class="mb-0! text-base"[^>]*>([^<]+)</p>'
    for match in re.finditer(pattern1, html_region):
        text = match.group(1).strip()
        if text and len(text) > 10:
            # Only skip if text is very short UI elements (not content with these words)
            if len(text) < 50 and any(skip in text.lower() for skip in ['chatgpt', 'copy code', 'share']):
                continue
            text_items.append((match.start(), text))
    
    # Pattern 2: class="text-token-text-primary flex text-base" with "Searched for X"
    pattern2 = r'class="text-token-text-primary flex text-base"[^>]*>(?:<div[^>]*>)*(Searched for [^<]+)'
    for match in re.finditer(pattern2, html_region):
        text = match.group(1).strip()
        if text:
            text_items.append((match.start(), text))
    
    # Pattern 3: Use BeautifulSoup to find <p data-start="0"> tags
    soup = BeautifulSoup(html_region, 'lxml')
    
    # Get all text from <p> tags with data-start="0"
    for p in soup.find_all('p', {'data-start': '0'}):
        text = p.get_text(strip=True)
        if text and len(text) > 20:
            # Skip "Read more" links and UI elements
            if text.startswith('Read ') and len(text) < 50:
                continue
            # Only skip if text is very short UI elements
            if len(text) < 50 and any(skip in text.lower() for skip in ['chatgpt', 'copy code', 'share']):
                continue
            # Find approximate position in html_region for ordering
            pos = html_region.find(text[:30]) if len(text) > 30 else html_region.find(text)
            if pos == -1:
                pos = len(html_region)  # Put at end if not found
            # Check for duplicates
            existing_texts = [t for _, t in text_items]
            if text not in existing_texts and not any(text in t or t in text for t in existing_texts):
                text_items.append((pos, text))
    
    # Sort by position to maintain order
    text_items.sort(key=lambda x: x[0])
    
    # Extract just the texts
    texts = [t for _, t in text_items]
    
    return '\n\n'.join(texts)


def extract_reasoning_from_region(html_region):
    """
    Extract reasoning text content from an HTML region.
    Looks for text that appears to be reasoning/thinking content.
    """
    soup = BeautifulSoup(html_region, 'lxml')
    
    # Get all text content
    text = soup.get_text(separator='\n')
    
    # Clean and filter lines
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Filter out very short lines and UI elements
        if line and len(line) > 30:
            # Skip obvious UI/SVG/code elements
            if not any(skip in line.lower() for skip in ['copy', 'share', 'button', 'path d=', 'viewbox', 'xmlns', 'svg']):
                # Check if it looks like reasoning content
                if any(keyword in line.lower() for keyword in [
                    "i'm", "i'll", "searching", "found", "looking", "checking", 
                    "gathering", "assessing", "exploring", "reviewing", "credibility",
                    "navigated", "collapsed", "detected", "alerted", "incident",
                    "watch", "apple", "news", "article", "source", "report"
                ]):
                    lines.append(line)
    
    result = '\n\n'.join(lines)
    return result


def extract_urls_from_region(html_region, url_mapping):
    """
    Extract URLs from an HTML region.
    First tries to find href links with full URLs, then looks for abbreviated domains and maps them.
    Prioritizes full URLs from the report over partial URLs found in the HTML.
    """
    urls = []
    soup = BeautifulSoup(html_region, 'lxml')
    
    # First, look for abbreviated domains in text and map them to full URLs
    text = soup.get_text()
    # Pattern to match domains like ndtv.com, fox7austin.com
    domain_pattern = r'\b([a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|org|net|edu|gov))\b'
    domains = re.findall(domain_pattern, text)
    
    for domain in set(domains):
        # Check if we have a mapping for this domain (full URLs from report)
        if domain in url_mapping:
            # url_mapping[domain] is now a list of URLs
            full_urls = url_mapping[domain]
            for full_url in full_urls:
                full_url = remove_trailing_slash(full_url)
                if full_url not in urls:
                    urls.append(full_url)
    
    # If no URLs found from mapping, try to find href links
    if not urls:
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '')
            if href.startswith('http') and not any(skip in href for skip in ['cdn.', 'oaistatic', '.js', '.css']):
                # Check if we have a better (full) URL in the mapping for this domain
                href_domain = extract_domain_from_url(href)
                if href_domain and href_domain in url_mapping:
                    # url_mapping[href_domain] is now a list of URLs
                    full_urls = url_mapping[href_domain]
                    for full_url in full_urls:
                        full_url = remove_trailing_slash(full_url)
                        if full_url not in urls:
                            urls.append(full_url)
                else:
                    href = remove_trailing_slash(href)
                    if href not in urls:
                        urls.append(href)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls




if __name__ == '__main__':
    # Define the input HTML directory and the desired output JSON directory
    input_html_dir = '/Users/zhanyuxiao/Desktop/Agent/HalluBench/data/benchmark/close-source/OpenAI/raw_html'
    output_json_dir = '/Users/zhanyuxiao/Desktop/Agent/HalluBench/data/benchmark/close-source/OpenAI/json'
    
    # Ensure output directory exists
    os.makedirs(output_json_dir, exist_ok=True)
    
    # Process all HTML files in the input directory
    html_files = [f for f in os.listdir(input_html_dir) if f.endswith('.html')]
    
    if not html_files:
        print(f"No HTML files found in '{input_html_dir}'")
    else:
        print(f"Found {len(html_files)} HTML file(s) to process")
        
        for html_file in html_files:
            input_html_file = os.path.join(input_html_dir, html_file)
            base_name = os.path.splitext(html_file)[0]
            print(f"Processing: {html_file}")
            output_json_file = os.path.join(output_json_dir, f'{base_name}.json')
            
            try:
                # Read the source HTML file
                with open(input_html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Parse the HTML and extract all components
                result = parse_openai_html(html_content)

                if result:
                    # Save to JSON file
                    with open(output_json_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"✓ Successfully processed '{html_file}' -> '{base_name}.json'")
                else:
                    print(f"✗ Failed to extract data from '{html_file}'.")

            except FileNotFoundError:
                print(f"✗ Error: The input file '{input_html_file}' was not found.")
            except Exception as e:
                print(f"✗ Error processing '{html_file}': {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nProcessing complete. Processed {len(html_files)} file(s).")
