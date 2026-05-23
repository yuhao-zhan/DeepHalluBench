import json
import os
import re
import copy
from bs4 import BeautifulSoup, Tag, NavigableString

def normalize_url(url: str) -> str:
    """
    Remove trailing slash from URL.
    
    Args:
        url: URL string
        
    Returns:
        URL with trailing slash removed (if it had one)
    """
    if not url or not isinstance(url, str):
        return url
    # Remove trailing slash, but keep it if URL is just "http://" or "https://"
    if url.endswith('/') and url not in ('http://', 'https://'):
        return url[:-1]
    return url

def convert_to_markdown(tag: Tag) -> str:
    """
    Converts HTML tag content to markdown format, preserving structure
    and converting links to markdown hyperlink format [text](url).
    """
    if tag is None:
        return ""
    
    tag_copy = copy.deepcopy(tag)
    result = []
    
    def process_element(elem):
        """Recursively process elements to convert to markdown"""
        if isinstance(elem, NavigableString):
            text = str(elem).strip()
            return text if text else ""
        
        if not isinstance(elem, Tag):
            return ""
        
        tag_name = elem.name
        
        if tag_name == 'h3':
            text = ''.join([process_element(child) for child in elem.children])
            return f"\n### {text}\n" if text.strip() else ""
        elif tag_name == 'h4':
            text = ''.join([process_element(child) for child in elem.children])
            return f"\n#### {text}\n" if text.strip() else ""
        elif tag_name == 'p':
            text = ''.join([process_element(child) for child in elem.children])
            return f"{text}\n" if text.strip() else ""
        elif tag_name == 'ul':
            items = []
            for li in elem.find_all('li', recursive=False):
                item_text = ''.join([process_element(child) for child in li.children])
                if item_text.strip():
                    items.append(f"- {item_text.strip()}")
            return "\n".join(items) + "\n" if items else ""
        elif tag_name == 'ol':
            items = []
            for idx, li in enumerate(elem.find_all('li', recursive=False), 1):
                item_text = ''.join([process_element(child) for child in li.children])
                if item_text.strip():
                    items.append(f"{idx}. {item_text.strip()}")
            return "\n".join(items) + "\n" if items else ""
        elif tag_name == 'li':
            text = ''.join([process_element(child) for child in elem.children])
            return text
        elif tag_name == 'strong':
            text = ''.join([process_element(child) for child in elem.children])
            return f"**{text}**" if text.strip() else ""
        elif tag_name == 'span':
            # Check if this span contains citation links
            classes = elem.get('class', [])
            if isinstance(classes, list):
                classes_str = ' '.join(classes)
            else:
                classes_str = str(classes)
            
            # Check if this is a citation span (contains inline-flex and has citation links)
            if 'inline-flex' in classes_str or 'flex-row' in classes_str:
                citation_links = elem.find_all('a', class_=lambda x: x and 'citation' in (' '.join(x) if isinstance(x, list) else str(x)))
                if citation_links:
                    # Extract all citation URLs and format them
                    urls = []
                    for link in citation_links:
                        href = link.get('href', '')
                        if href:
                            urls.append(f"@{normalize_url(href)}")
                    if urls:
                        # Return formatted URLs with comma separator
                        return " " + " , ".join(urls) + " "
            
            # Regular span - process children normally
            text = ''.join([process_element(child) for child in elem.children])
            return text
        elif tag_name == 'a':
            # Check if this is a citation link
            classes = elem.get('class', [])
            if isinstance(classes, list):
                classes_str = ' '.join(classes)
            else:
                classes_str = str(classes)
            
            href = elem.get('href', '')
            if href:
                # Check if parent is a citation span - if so, skip processing here (will be handled by span)
                parent = elem.parent
                if parent and isinstance(parent, Tag):
                    parent_classes = parent.get('class', [])
                    if isinstance(parent_classes, list):
                        parent_classes_str = ' '.join(parent_classes)
                    else:
                        parent_classes_str = str(parent_classes)
                    if 'inline-flex' in parent_classes_str or 'flex-row' in parent_classes_str:
                        # This is inside a citation span, skip processing here
                        return ""
                
                # Regular link - return normalized URL with spaces around it
                return f" @{normalize_url(href)} "
            link_text = ''.join([process_element(child) for child in elem.children])
            return link_text
        elif tag_name == 'table':
            # Convert table to markdown format
            rows = []
            headers = []
            thead = elem.find('thead')
            if thead:
                for th in thead.find_all('th'):
                    header_text = ''.join([process_element(child) for child in th.children])
                    headers.append(header_text.strip())
                if headers:
                    rows.append("| " + " | ".join(headers) + " |")
                    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            tbody = elem.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    cells = []
                    for td in tr.find_all('td'):
                        cell_text = ''.join([process_element(child) for child in td.children])
                        cells.append(cell_text.strip().replace('\n', ' '))
                    if cells:
                        rows.append("| " + " | ".join(cells) + " |")
            
            return "\n".join(rows) + "\n" if rows else ""
        elif tag_name == 'hr':
            return "\n---\n"
        else:
            # For other tags, just process children
            return ''.join([process_element(child) for child in elem.children])
    
    result = process_element(tag_copy)
    # Clean up extra newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    # Clean up multiple consecutive spaces (but preserve single spaces)
    result = re.sub(r' {2,}', ' ', result)
    return result.strip()

def extract_links_from_text(text: str) -> list:
    """Extract all URLs from text using regex"""
    url_pattern = r'https?://[^\s\)]+'
    return re.findall(url_pattern, text)

def parse_grok_html(html_content: str) -> dict:
    """Parse Grok deep research HTML trace into structured dict.

    Args:
        html_content: Raw HTML string of the Grok deep research page.

    Returns:
        dict with keys: query, final_report, all_source_links, summary_citations, chain_of_research
    """
    result_data = {
        "query": "", 
        "chain_of_research": {}, 
        "final_report": "",
        "all_source_links": [], 
        "summary_citations": []
    }
    
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract query from the response-content-markdown div
    # The query is in a <p> tag with "whitespace-pre-wrap" style inside response-content-markdown
    query_container = soup.find('div', class_='response-content-markdown')
    if query_container:
        # Find the first <p> tag that contains the query text
        query_para = query_container.find('p', class_='break-words')
        if query_para:
            # Get all text content including from nested lists
            query_parts = []
            current = query_para.next_sibling
            # Include the paragraph and its following siblings until we hit the h3
            query_parts.append(query_para.get_text(separator='\n', strip=True))
            while current:
                if isinstance(current, Tag):
                    if current.name == 'h3':
                        break
                    if current.name in ['ol', 'ul']:
                        # Extract list items
                        for li in current.find_all('li', recursive=False):
                            text = li.get_text(separator=' ', strip=True)
                            if text:
                                query_parts.append(f"{text}")
                    elif current.name == 'p':
                        text = current.get_text(separator=' ', strip=True)
                        if text:
                            query_parts.append(text)
                current = current.next_sibling
            if query_parts:
                # Clean up query: remove backslash escapes
                query_text = '\n\n'.join(query_parts).strip()
                query_text = query_text.replace('\\', '')
                result_data["query"] = query_text

    # Extract final report content
    # Find the div with message-bubble class that contains w-full max-w-none
    # This is the response container (not the query)
    report_container = None
    
    # Find all divs with message-bubble class
    for div in soup.find_all('div', class_=lambda x: x and 'message-bubble' in str(x)):
        classes = div.get('class', [])
        if isinstance(classes, list):
            classes_str = ' '.join(classes)
        else:
            classes_str = str(classes)
        
        # Check if it has w-full max-w-none (this indicates it's the response container, not the query)
        if 'w-full' in classes_str and 'max-w-none' in classes_str:
            report_container = div
            break
    
    if report_container:
        # Find the response-content-markdown div inside this container
        markdown_div = report_container.find('div', class_='response-content-markdown')
        if not markdown_div:
            # Fallback: use the report_container itself
            markdown_div = report_container
        
        # Find the start marker: h3 with "Key Incidents" or similar
        report_start_h3 = None
        for h3 in markdown_div.find_all('h3'):
            text = h3.get_text(strip=True)
            if 'Key Incidents' in text or 'Incidents' in text:
                report_start_h3 = h3
                break
        
        # Find the end marker: paragraph with "Key Citations"
        key_citations_para = None
        for p in markdown_div.find_all('p'):
            strong = p.find('strong')
            if strong and 'Key Citations' in strong.get_text():
                key_citations_para = p
                break
        
        if report_start_h3:
            # Collect all elements from report_start_h3 to before key_citations_para
            report_elements = soup.new_tag('div')
            report_elements.append(copy.deepcopy(report_start_h3))
            
            # Get all siblings after the h3 until we hit key_citations_para
            current = report_start_h3.next_sibling
            while current:
                if current == key_citations_para:
                    break
                
                # Add current element to report
                if isinstance(current, Tag):
                    report_elements.append(copy.deepcopy(current))
                
                # Move to next sibling
                current = current.next_sibling
        else:
            # If no h3 found, extract all content from markdown_div (excluding Key Citations)
            report_elements = soup.new_tag('div')
            for child in markdown_div.children:
                if isinstance(child, Tag):
                    # Skip if it's the Key Citations paragraph
                    if child == key_citations_para:
                        break
                    report_elements.append(copy.deepcopy(child))
                elif child == key_citations_para:
                    break
        
        # Convert to markdown
        result_data["final_report"] = convert_to_markdown(report_elements)
        
        # Clean up final_report: remove duplicate @url@url patterns
        report_text = result_data["final_report"]
        # Remove patterns like @url@url (same URL repeated consecutively)
        report_text = re.sub(r'@(https?://[^\s\)]+)@\1', r'@\1', report_text)
        result_data["final_report"] = report_text

    # Parse chain_of_research: extract URLs from elements with class 'flex flex-col gap-1 px-6 py-3'
    # Each element with this class corresponds to one search
    # Find the report container to exclude URLs that are in the report
    report_container = soup.find('div', class_='response-content-markdown')
    report_urls = set()
    if report_container:
        for link in report_container.find_all('a', href=True):
            report_urls.add(normalize_url(link['href']))
    
    # Find all elements with the specific class
    target_class = 'flex flex-col gap-1 px-6 py-3'
    search_elements = soup.find_all(attrs={'class': lambda x: x and target_class in (' '.join(x) if isinstance(x, list) else str(x))})
    
    # Track all URLs globally to avoid duplicates across searches
    all_seen_urls_global = set()
    
    # Process each search element
    for idx, search_elem in enumerate(search_elements, 1):
        search_urls = []
        search_seen = set()  # Track URLs within this search to avoid duplicates
        
        # Find all links in this search element
        for link in search_elem.find_all('a', href=True):
            url = link['href']
            if url.startswith(('http://', 'https://')):
                url = normalize_url(url)
                
                # Skip if it's in the report
                if url in report_urls:
                    continue
                
                # Only add if not already added globally (avoid duplicates across searches)
                if url not in all_seen_urls_global and url not in search_seen:
                    search_urls.append(url)
                    search_seen.add(url)
                    all_seen_urls_global.add(url)
        
        # Add this search to chain_of_research if it has URLs
        if search_urls:
            result_data["chain_of_research"][f"search_{idx}"] = search_urls
            result_data["all_source_links"].extend(search_urls)
    
    # Remove duplicates and normalize URLs in all_source_links
    normalized_all_source_links = []
    for url in result_data["all_source_links"]:
        normalized_url = normalize_url(url)
        if normalized_url not in normalized_all_source_links:
            normalized_all_source_links.append(normalized_url)
    result_data["all_source_links"] = normalized_all_source_links

    # summary_citations is already normalized and deduplicated when extracted from final_report
    
    # Normalize URLs in chain_of_research
    for key in list(result_data["chain_of_research"].keys()):
        if isinstance(result_data["chain_of_research"][key], list):
            normalized_urls = []
            for url in result_data["chain_of_research"][key]:
                normalized_url = normalize_url(url)
                if normalized_url not in normalized_urls:
                    normalized_urls.append(normalized_url)
            result_data["chain_of_research"][key] = normalized_urls
    
    # Normalize URLs in final_report (@url format)
    if result_data["final_report"]:
        # Find all @url patterns and normalize them
        report_text = result_data["final_report"]
        # Pattern to match @url
        url_pattern = r'@(https?://[^\s\)]+)'
        def replace_url(match):
            url = match.group(1)
            normalized = normalize_url(url)
            return f"@{normalized}"
        result_data["final_report"] = re.sub(url_pattern, replace_url, report_text)
        
        # Extract all URLs from final_report text (in @url format) for summary_citations
        # This ensures we capture all URLs that actually appear in the final report
        # Do this AFTER normalization so we get normalized URLs
        normalized_report_text = result_data["final_report"]
        summary_urls = re.findall(url_pattern, normalized_report_text)
        # Deduplicate (URLs are already normalized)
        normalized_summary_urls = []
        for url in summary_urls:
            if url not in normalized_summary_urls:
                normalized_summary_urls.append(url)
        result_data["summary_citations"] = normalized_summary_urls

    return result_data

def main():
    # Process all HTML files in the directory
    html_dir = "/Users/zhanyuxiao/Desktop/Agent/HalluBench/data/benchmark/close-source/Grok/raw_html"
    json_dir = "/Users/zhanyuxiao/Desktop/Agent/HalluBench/data/benchmark/close-source/Grok/json"
    
    if not os.path.exists(html_dir):
        print(f"Error: HTML directory not found at {html_dir}")
        return
    
    # Ensure JSON directory exists
    os.makedirs(json_dir, exist_ok=True)
    
    # Find all HTML files in the directory (excluding subdirectories)
    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html') and os.path.isfile(os.path.join(html_dir, f))]
    
    if not html_files:
        print(f"No HTML files found in {html_dir}")
        return
    
    print(f"Found {len(html_files)} HTML file(s) to process\n")
    
    # Process each HTML file
    for html_filename in sorted(html_files):
        html_file_path = os.path.join(html_dir, html_filename)
        base_name = os.path.splitext(html_filename)[0]
        json_filename = f"{base_name}.json"
        json_file_path = os.path.join(json_dir, json_filename)
        
        print(f"Processing: {html_filename}")
        
        try:
            # Read the HTML file
            # Read the HTML file
            with open(html_file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            # Parse the HTML content
            result = parse_grok_html(html_content)
            
            # Save the result to a JSON file
            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump(result, file, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved to {json_filename}")
            
            # Print summary for this file
            reasoning_sections = [k for k in result["chain_of_research"].keys() if k.startswith("reasoning")]
            search_sections = [k for k in result["chain_of_research"].keys() if k.startswith("search")]
            
            print(f"  - Query: {result['query'][:80]}..." if result['query'] else "  - Query: (empty)")
            print(f"  - Reasoning sections: {len(reasoning_sections)}")
            print(f"  - Search sections: {len(search_sections)}")
            print(f"  - Total source links: {len(result['all_source_links'])}")
            print(f"  - Summary citations: {len(result['summary_citations'])}")
            print(f"  - Final report length: {len(result['final_report'])} characters")
            
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {html_filename}: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Processing completed! Processed {len(html_files)} file(s).")

if __name__ == "__main__":
    main()