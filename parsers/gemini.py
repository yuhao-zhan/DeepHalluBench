import json
import re
import sys
import os

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from markdownify import markdownify as md

def _normalize_url(url):
    """Normalize URL by removing strange postfixes and finding the cleanest version."""
    if not url:
        return None
    
    # Check if URL contains Chinese characters - if so, return None to filter it out
    import re
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')  # Unicode range for Chinese characters
    if chinese_pattern.search(url):
        return None
    
    # Remove trailing slashes
    url = url.rstrip('/')
    
    # Remove common strange postfixes that appear in the data
    strange_postfixes = [
        ')',  # Remove trailing parentheses
        '/Research',  # Remove Research postfix
        '/AI',  # Remove AI postfix
        'Research',  # Remove Research without slash
        'Machine',  # Remove Machine postfix
        'Distributed',  # Remove Distributed postfix
    ]
    
    for postfix in strange_postfixes:
        if url.endswith(postfix):
            url = url[:-len(postfix)]
    
    # Remove trailing slashes again after postfix removal
    url = url.rstrip('/')
    
    return url

def _deduplicate_and_normalize_urls(urls):
    """Deduplicate URLs by normalizing them and keeping unique URLs including those with different query parameters."""
    # Group URLs by their normalized base, but preserve URLs with different query parameters
    url_groups = {}
    
    for url in urls:
        if not url:
            continue
            
        normalized = _normalize_url(url)
        if normalized is None:  # Skip URLs with Chinese characters
            continue
            
        # Create a base key by removing query parameters for grouping
        if '?' in normalized:
            base_url = normalized.split('?')[0]
        else:
            base_url = normalized
            
        if base_url not in url_groups:
            url_groups[base_url] = []
        url_groups[base_url].append(normalized)
    
    # For each group, keep unique URLs (including those with different query parameters)
    clean_urls = []
    for base_url, variations in url_groups.items():
        # Remove exact duplicates first
        unique_variations = list(set(variations))
        
        # If we have multiple variations with different query parameters, keep them all
        if len(unique_variations) > 1:
            # Check if they have different query parameters
            query_params = set()
            for variation in unique_variations:
                if '?' in variation:
                    query_params.add(variation)
                else:
                    query_params.add(variation)
            
            # If we have URLs with different query parameters, keep them all
            if len(query_params) > 1:
                clean_urls.extend(sorted(query_params))
            else:
                # Just one unique variation, keep it
                clean_urls.append(unique_variations[0])
        else:
            # Only one variation, keep it
            clean_urls.append(unique_variations[0])
    
    return sorted(clean_urls)

def _extract_all_urls_from_text(text_content):
    """Extract all URLs from text content using regex pattern matching."""
    import re
    
    # Multiple regex patterns to catch different URL formats in the report
    url_patterns = [
        r'@(https?://[^\s<>"{}|\\^`\[\]]+)',  # URLs with @ prefix (like @https://...)
        r'\[([^\]]+)\]\((https?://[^)]+)\)',  # Markdown links [text](url) - extract URL part
        r'\[(https?://[^\]]+)\]',              # URLs in square brackets [url]
        r'https?://[^\s<>"{}|\\^`\[\]]+',     # Standard URLs
    ]
    
    urls = []
    for pattern in url_patterns:
        found_urls = re.findall(pattern, text_content)
        if pattern.startswith(r'@'):
            # For @ prefixed URLs, remove the @ symbol for normalization
            urls.extend([url for url in found_urls])
        elif pattern.startswith(r'\['):
            # For markdown link URLs, extract the URL part (second group)
            if pattern == r'\[([^\]]+)\]\((https?://[^)]+)\)':
                # Pattern: [text](url) - extract URL part
                urls.extend([url[1] for url in found_urls if len(url) > 1])
            else:
                # Pattern: [url] - extract URL part directly
                urls.extend([url for url in found_urls])
        else:
            urls.extend(found_urls)
    
    return [_normalize_url(url) for url in urls]

def _extract_all_urls_from_html(soup):
    """Extract all URLs from HTML content including href attributes and text content."""
    urls = set()
    
    # Extract URLs from href attributes
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http'):
            urls.add(_normalize_url(href))
    
    # Extract URLs from text content
    text_content = soup.get_text()
    text_urls = _extract_all_urls_from_text(text_content)
    urls.update(text_urls)
    
    return list(urls)

def parse_gemini_html(html_content, markdown_file_path=None, base_url="https://gemini.google.com/"):
    """Parse Gemini deep research HTML trace into structured dict.

    Args:
        html_content: Raw HTML string of the Gemini deep research page.
        markdown_file_path: Optional path to a markdown report file. If provided,
            its content is used as final_report and for citation replacement.
        base_url: Base URL for resolving relative links.

    Returns:
        dict with keys: query, final_report, all_source_links, summary_citations, chain_of_research
    """
    soup = BeautifulSoup(html_content, 'lxml')
    output = {
        "query": "",
        "chain_of_research": {},
        "final_report": "",
        "all_source_links": [],
        "summary_citations": []
    }

    # --- 1. Extract the Query ---
    first_user_query = soup.find('user-query')
    if first_user_query:
        query_parts = [p.get_text(strip=True) for p in first_user_query.select('.query-text-line')]
        output['query'] = "\n\n".join(query_parts)

    # --- 2. Extract Chain of Research (with Plan Merging) ---
    chain_of_research = {}
    all_links = set()
    immersive_panel = soup.find('deep-research-immersive-panel')

    if immersive_panel:
        steps = immersive_panel.select('deep-research-confirmation-widget, browse-chip-list, thought-item')
        search_count, plan_count, observation_count = 0, 0, 0
        is_after_search = False
        last_key = None

        for step in steps:
            if step.name == 'deep-research-confirmation-widget':
                plan_count += 1
                plan_key = f"plan_{plan_count}"
                title_el = step.find(class_='research-step-title')
                desc_el = step.find(class_='research-step-description')
                chain_of_research[plan_key] = {
                    "title": title_el.get_text(strip=True) if title_el else "Initial Plan",
                    "description": desc_el.get_text(strip=True, separator='\n') if desc_el else ""
                }
                last_key = plan_key
            elif step.name == 'browse-chip-list':
                search_count += 1
                search_key = f"search_{search_count}"
                links = [_normalize_url(urljoin(base_url, a['href'])) for a in step.find_all('a', class_='browse-chip')]
                chain_of_research[search_key] = links
                all_links.update(links)
                is_after_search = True
                last_key = search_key
            elif step.name == 'thought-item':
                # In Mind2Web-2 HTML, thought-item structure is:
                # <div class="thought-header">header text</div>
                # <div class="gds-body-m gds-italic">body text</div>
                # NOT <div class="thought-body">
                header_el = step.find(class_='thought-header')
                # Find the body by looking for the next div after thought-header
                body_el = None
                if header_el:
                    body_el = header_el.find_next_sibling('div')
                else:
                    # If no thought-header, find all divs with gds-body-m and use the first one
                    divs = step.find_all('div', class_=lambda c: c and 'gds-body-m' in str(c))
                    if divs:
                        header_el = None
                        body_el = divs[0]
                
                header = header_el.get_text(strip=True) if header_el else ""
                body = body_el.get_text(strip=True) if body_el else ""
                
                if not header and not body:
                    continue  # Skip empty thought-items

                if is_after_search:
                    observation_count += 1
                    obs_key = f"observation_{observation_count}"
                    chain_of_research[obs_key] = body
                    is_after_search = False
                    last_key = obs_key
                else:
                    if last_key and last_key.startswith('plan_'):
                        chain_of_research[last_key]['title'] += f"\n{header}"
                        chain_of_research[last_key]['description'] += f"\n{body}"
                    else:
                        plan_count += 1
                        plan_key = f"plan_{plan_count}"
                        chain_of_research[plan_key] = {
                            "title": header,
                            "description": body
                        }
                        last_key = plan_key

    output['chain_of_research'] = chain_of_research
    
    # --- 3. Create a Master Source Map from the "Used Sources" list ---
    source_map = []
    source_lists = soup.find('deep-research-source-lists')
    if source_lists:
        # Try to find "报告中使用的来源" first, then fall back to English "Sources used in the report"
        used_sources_header = source_lists.find(
            lambda tag: tag.name == 'span' and ('报告中使用的来源' in tag.get_text() or 'Sources used in the report' in tag.get_text())
        )
        if used_sources_header:
            button_container = used_sources_header.find_parent('button').find_parent()
            used_sources_list = button_container.find_next_sibling('div', class_='source-list')
            if used_sources_list:
                for link_tag in used_sources_list.select('browse-web-item a'):
                    if link_tag.has_attr('href'):
                        source_map.append(_normalize_url(urljoin(base_url, link_tag['href'])))
        output['summary_citations'] = source_map

    # --- 4. Handle Final Report ---
    html_report_urls = []
    if markdown_file_path:
        # Use markdown file content as final_report
        try:
            with open(markdown_file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Extract URLs from the bottom of the markdown file for citation replacement
            markdown_source_map = _extract_urls_from_markdown(markdown_content)
            
            # Remove the "引用的著作" or "Works cited" section and everything after it
            markdown_content = _remove_references_section(markdown_content)
            
            # Replace newlines with \n and handle other special characters
            # markdown_content = markdown_content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            
            # Replace citation patterns like ".1", ".2", ".15", etc. with URLs from markdown
            # Process from highest to lowest to avoid conflicts (e.g., ".15" before ".1")
            # Must handle both regular dots ".1" and escaped dots "\.1"
            # Use multiline flag to match across line boundaries
            for i in range(len(markdown_source_map), 0, -1):  # Process from highest to lowest
                citation_num = str(i)
                # Pattern matches both ".1" and "\.1" (escaped dot) formats
                # - (?<![0-9]) - not preceded by digit (avoids matching "3.1" as ".1")
                # - (?:\\\\.|\\.) - either escaped dot "\\.1" or regular dot ".1"
                # - exact citation number
                # - (?![0-9]) - not followed by digit (avoids matching ".10" when looking for ".1")
                # Note: In regex string, "\\\\\\.\"" becomes "\\.\"" in pattern (backslash+dot)
                #       and "\\.\" becomes "\.\" in pattern (just dot)
                pattern = f'(?<![0-9])(?:\\\\\\.|\\.){citation_num}(?![0-9])'
                replacement = f' @{markdown_source_map[i-1]} '
                markdown_content = re.sub(pattern, replacement, markdown_content, flags=re.MULTILINE)
            
            # Ensure all URLs in the content have @ prefix for consistency
            # Convert markdown links [text](url) to @url format
            markdown_content = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'@\2', markdown_content)
            
            # Convert direct URLs to @url format (but avoid URLs that already have @)
            markdown_content = re.sub(r'(?<!@)(https?://[^\s<>"{}|\\^`\[\]]+)', r'@\1', markdown_content)
            
            output['final_report'] = markdown_content
            
        except FileNotFoundError:
            print(f"Warning: Markdown file '{markdown_file_path}' not found. Using HTML parsing instead.")
            # Fallback to original HTML parsing method
            output['final_report'], html_report_urls = _parse_final_report_from_html(soup, source_map)
        except Exception as e:
            print(f"Warning: Error reading markdown file: {e}. Using HTML parsing instead.")
            # Fallback to original HTML parsing method
            output['final_report'], html_report_urls = _parse_final_report_from_html(soup, source_map)
    else:
        # Use original HTML parsing method
        output['final_report'], html_report_urls = _parse_final_report_from_html(soup, source_map)

    # --- 5. Compile All Source Links ---
    # all_source_links should only contain URLs from search/browse steps (all_links)
    # This is the collection of all URLs found during the research process
    all_source_links = sorted(list(all_links))
    
    # Normalize and deduplicate all source links
    all_unique_urls = list(set(all_source_links))
    clean_urls = _deduplicate_and_normalize_urls(all_unique_urls)
    output['all_source_links'] = clean_urls
    
    # summary_citations is already populated from the '报告中使用的来源' section (line 252)
    # Normalize and deduplicate it
    normalized_citations = []
    for url in output['summary_citations']:
        normalized = _normalize_url(url)
        if normalized and normalized not in normalized_citations:
            normalized_citations.append(normalized)
    
    output['summary_citations'] = sorted(normalized_citations)

    return output

def _parse_final_report_from_html(soup, source_map):
    """Helper function to parse final report from HTML when markdown file is not available."""
    report_container = soup.find('div', id='extended-response-markdown-content')
    if report_container:
        # Remove all implicit source carousels as they are redundant
        for carousel in report_container.select('sources-carousel-inline'):
            carousel.decompose()

        # Replace all source-footnote superscripts with direct markdown links
        for footnote in report_container.select('source-footnote'):
            try:
                index = int(footnote.sup['data-turn-source-index'])
                if 1 <= index <= len(source_map):
                    url = source_map[index - 1]
                    # Create a new <a> tag
                    new_link = soup.new_tag('a', href=url)
                    new_link.string = f"[{index}]" 
                    footnote.replace_with(new_link)
                else:
                    footnote.decompose() # Remove if index is out of bounds
            except (KeyError, ValueError, TypeError):
                footnote.decompose() # Remove malformed footnotes

        report_html = ''.join(str(c) for c in report_container.contents)
        # Convert the modified HTML to Markdown. markdownify will handle the <a> tags.
        markdown_content = md(report_html, heading_style="ATX").strip()
        
        # Extract URLs from the processed HTML content
        processed_urls = _extract_all_urls_from_html(report_container)
        
        return markdown_content, processed_urls
    
    return "", []

def _extract_urls_from_markdown(markdown_content):
    """Extracts URLs from the bottom numbered list section of a markdown file for citation replacement."""
    urls = []
    lines = markdown_content.split('\n')
    
    # Find "引用的著作" or "Works cited" header
    start_index = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if '引用的著作' in stripped or 'Works cited' in stripped:
            start_index = i + 1  # Start from the line after the header
            break
    
    if start_index == -1:
        return urls  # No reference section found
    
    # Extract URLs from the numbered list
    for i in range(start_index, len(lines)):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check if this is a numbered line
        expected_num = len(urls) + 1
        if not line.startswith(f'{expected_num}.'):
            # If not the expected number, the list has ended
            if urls:  # If we've already extracted some URLs, stop
                break
            continue
        
        # Extract URL from the line
        if '[' in line and '](' in line and ')' in line:
            # Markdown link format [text](url)
            url_match = re.search(r'\]\((https?://[^)]+)\)', line)
            if url_match:
                urls.append(url_match.group(1))
        elif re.search(r'https?://[^\s]+', line):
            # Direct URL in the line
            url_match = re.search(r'(https?://[^\s]+)', line)
            if url_match:
                urls.append(url_match.group(1))
    
    return urls

def _remove_references_section(markdown_content):
    """Removes the "引用的著作" or "Works cited" section and everything after it from the markdown content."""
    lines = markdown_content.split('\n')
    start_index = -1
    for i, line in enumerate(lines):
        # Check for Chinese "引用的著作" or English "Works cited"
        if line.strip().startswith('#### **引用的著作**') or line.strip().startswith('#### **Works cited**'):
            start_index = i
            break
    if start_index != -1:
        return '\n'.join(lines[:start_index])
    return markdown_content


def process_file_pairs(report_md_dir, cor_dir, json_output_dir):
    """
    Process pairs of files from markdown/ and raw_html/ directories.
    Only process files when both .md and .html exist.
    
    Args:
        report_md_dir (str): Path to markdown directory
        cor_dir (str): Path to raw_html directory  
        json_output_dir (str): Path to output json directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Get all .md files from markdown directory
    md_files = [f for f in os.listdir(report_md_dir) if f.endswith('.md')]
    md_basenames = {os.path.splitext(f)[0] for f in md_files}
    
    # Get all .html files from raw_html directory
    html_files = [f for f in os.listdir(cor_dir) if f.endswith('.html')]
    html_basenames = {os.path.splitext(f)[0] for f in html_files}
    
    # Find pairs (files with same basename)
    pairs = md_basenames.intersection(html_basenames)
    
    # Find files without pairs
    md_only = md_basenames - html_basenames
    html_only = html_basenames - md_basenames
    
    print(f"Found {len(pairs)} file pairs to process")
    print(f"Files in markdown/ without HTML counterpart: {len(md_only)}")
    print(f"Files in raw_html/ without MD counterpart: {len(html_only)}")
    
    # Process each pair
    processed_count = 0
    for basename in sorted(pairs):
        md_file_path = os.path.join(report_md_dir, f"{basename}.md")
        html_file_path = os.path.join(cor_dir, f"{basename}.html")
        json_file_path = os.path.join(json_output_dir, f"{basename}.json")
        
        try:
            print(f"Processing pair: {basename}")
            
            # Read HTML file
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML with corresponding markdown file
            result = parse_gemini_html(html_content, md_file_path)

            # Save JSON result
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            processed_count += 1
            print(f"  ✓ Saved {json_file_path}")
            
        except FileNotFoundError as e:
            print(f"  ✗ File not found: {e}")
        except Exception as e:
            print(f"  ✗ Error processing {basename}: {e}")
    
    print(f"\nProcessing complete: {processed_count}/{len(pairs)} pairs processed successfully")
    
    # Report files without pairs
    if md_only:
        print(f"\nFiles in markdown/ without HTML counterpart:")
        for filename in sorted(md_only):
            print(f"  - {filename}.md")
    
    if html_only:
        print(f"\nFiles in raw_html/ without MD counterpart:")
        for filename in sorted(html_only):
            print(f"  - {filename}.html")

if __name__ == '__main__':
    # Define directories
    # report_md_dir = "/Users/zhanyuxiao/Desktop/Agent/First_week_0714/Data/Mind2Web-2/markdown"
    # cor_dir = "/Users/zhanyuxiao/Desktop/Agent/First_week_0714/Data/Mind2Web-2/raw_html"
    # json_output_dir = "/Users/zhanyuxiao/Desktop/Agent/First_week_0714/Data/Mind2Web-2/json"
    

    report_md_dir = "/Users/zhanyuxiao/Desktop/Agent/First_week_0714/Data/BrowseComp/No_answer_30/markdown"
    cor_dir = "/Users/zhanyuxiao/Desktop/Agent/First_week_0714/Data/BrowseComp/No_answer_30/raw_html"
    json_output_dir = "/Users/zhanyuxiao/Desktop/Agent/First_week_0714/Data/BrowseComp/No_answer_30/json"

    # Process all file pairs
    process_file_pairs(report_md_dir, cor_dir, json_output_dir)