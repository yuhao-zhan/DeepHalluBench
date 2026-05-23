import json
import re
import os
from pathlib import Path
from bs4 import BeautifulSoup

def remove_trailing_slash(url):
    """
    Remove trailing slash from URL if present.
    """
    if url and isinstance(url, str):
        return url.rstrip('/')
    return url

def extract_from_html(html_content):
    """
    Parses HTML to extract the query, a structured chain of research, and all source links.
    Structure: reasoning_1 -> search_1 -> reasoning_2 -> search_2 -> ...
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # 1. Extract the user query
    query_span = soup.select_one('span[data-lexical-text="true"]')
    query = query_span.get_text(strip=True) if query_span else ""

    # 2. Extract and structure the chain of research
    # First, collect all reasoning and search pairs
    steps_data = []
    all_source_links = set()

    research_steps = soup.select('div[role="listitem"].group\\/goal')

    for step_idx, step in enumerate(research_steps):
        # Extract reasoning text from the current step
        reasoning_div = step.select_one('span.pr-sm.block > div.relative')
        reasoning_text = ""
        if reasoning_div:
            reasoning_text = reasoning_div.get_text(strip=True, separator='\n')
        
        # Extract search queries from <p class="px-two"> elements in this step
        # These appear after the reasoning text
        search_queries = []
        for p_tag in step.select('p.px-two'):
            query_text = p_tag.get_text(strip=True)
            if query_text:
                search_queries.append(query_text)
        
        # Extract URLs from "Reviewing sources" section
        # Find all text nodes containing "Reviewing sources"
        reviewing_sources_texts = step.find_all(string=lambda text: text and 'Reviewing sources' in text)
        
        urls = []
        if reviewing_sources_texts:
            # Get the parent div of the first "Reviewing sources" text
            reviewing_sources_div = reviewing_sources_texts[0].find_parent('div')
            
            if reviewing_sources_div:
                # Find the next reasoning step to know where to stop
                next_step = research_steps[step_idx + 1] if step_idx + 1 < len(research_steps) else None
                
                # Find the container div that holds the links (usually has class containing 'container' or 'step-card')
                # The links are typically in a div that comes after "Reviewing sources"
                links_container = None
                for div in reviewing_sources_div.find_all_next('div', limit=30):
                    classes = div.get('class', [])
                    class_str = ' '.join(classes) if classes else ''
                    # Look for the div that contains multiple links and has container/step-card class
                    if ('container' in class_str or 'step-card' in class_str):
                        links_in_div = div.find_all('a', href=True)
                        if len(links_in_div) >= 3:  # Usually there are multiple links
                            # Make sure this container is still within the current step
                            if next_step is None or div not in next_step.find_all('div'):
                                links_container = div
                                break
                
                # If we found a container, extract links from it
                if links_container:
                    for link in links_container.find_all('a', href=True):
                        url = link.get('href')
                        if url and url.startswith('http'):
                            url = remove_trailing_slash(url)
                            if url not in urls:
                                urls.append(url)
                                all_source_links.add(url)
                else:
                    # Fallback: extract all links after "Reviewing sources" within current step
                    # Stop when we reach the next step
                    for link in reviewing_sources_div.find_all_next('a', href=True):
                        # Stop if we've reached the next step
                        if next_step:
                            # Check if this link belongs to the next step
                            if link in next_step.find_all('a'):
                                # Check if it's part of next step's reasoning (definitely stop)
                                next_reasoning = next_step.select_one('span.pr-sm.block > div.relative')
                                if next_reasoning and link in next_reasoning.find_all('a'):
                                    break
                                # If it's in next step but not reasoning, it's part of next step's search
                                # So we should stop here too
                                break
                        
                        url = link.get('href')
                        if url and url.startswith('http'):
                            url = remove_trailing_slash(url)
                            # Avoid duplicates
                            if url not in urls:
                                urls.append(url)
                                all_source_links.add(url)
        
        # Store step data
        if reasoning_text and reasoning_text != "完成":
            # Add search queries to the reasoning if they exist
            reasoning_with_searches = reasoning_text
            if search_queries:
                search_lines = [f'\n\nsearch for "{q}"' for q in search_queries]
                reasoning_with_searches += "".join(search_lines)
            
            steps_data.append({
                'reasoning': reasoning_with_searches,
                'urls': urls
            })
    
    # Now build the chain, merging consecutive reasonings without search
    chain_of_research = {}
    reasoning_count = 1
    search_count = 1
    reasoning_buffer = []
    
    for step_data in steps_data:
        reasoning = step_data['reasoning']
        urls = step_data['urls']
        
        # Add reasoning to buffer
        reasoning_buffer.append(reasoning)
        
        # If we have URLs, flush the buffer and add search
        if urls:
            # Merge all buffered reasonings
            merged_reasoning = "\n\n".join(reasoning_buffer)
            chain_of_research[f"reasoning_{reasoning_count}"] = merged_reasoning
            reasoning_count += 1
            reasoning_buffer = []
            
            # Add search
            chain_of_research[f"search_{search_count}"] = urls
            search_count += 1

    # If there are remaining reasonings in buffer (no final search), add them as the last reasoning
    if reasoning_buffer:
        merged_reasoning = "\n\n".join(reasoning_buffer)
        chain_of_research[f"reasoning_{reasoning_count}"] = merged_reasoning

    # Remove trailing slashes from all URLs before returning
    all_source_links_cleaned = [remove_trailing_slash(url) for url in all_source_links]
    return query, chain_of_research, sorted(list(set(all_source_links_cleaned)))

def extract_report_from_html(html_content):
    """
    Extracts the report section from HTML and converts it to Markdown format.
    The report starts after the 'Finished' text and includes all content after it.
    Links are appended to text in the format: text@url
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Find the 'Finished' text node
    finished_element = None
    for element in soup.find_all(string=lambda text: text and 'Finished' in text.strip()):
        # Check if the text is exactly 'Finished' or contains it
        if 'Finished' in element.strip():
            finished_element = element.find_parent()
            break
    
    # If not found by string search, try finding by text content
    if not finished_element:
        for div in soup.find_all('div'):
            text = div.get_text(strip=True)
            if text == 'Finished':
                finished_element = div
                break
    
    if not finished_element:
        return "", []
    
    # Find the markdown-content div that comes after 'Finished'
    # This is where the actual report content is stored
    report_container = None
    for element in finished_element.find_all_next('div'):
        element_id = element.get('id', '')
        if 'markdown-content' in element_id:
            report_container = element
            break
    
    # If markdown-content not found, try to find the prose container after Finished
    if not report_container:
        for element in finished_element.find_all_next('div'):
            classes = element.get('class', [])
            class_str = ' '.join(classes) if classes else ''
            # Look for prose container which typically contains the report
            if 'prose' in class_str and 'dark:prose-invert' in class_str:
                report_container = element
                break
    
    # If still not found, extract all content after Finished, excluding research steps
    if not report_container:
        # Get all elements after Finished, but skip research steps
        all_elements = []
        for element in finished_element.find_all_next(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'hr', 'blockquote']):
            # Skip elements that are part of the research chain
            if element.find_parent('div', role='listitem'):
                continue
            all_elements.append(element)
        
        # Convert to Markdown
        md_lines = []
        summary_citations = set()
        
        for element in all_elements:
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                text = element.get_text(strip=True)
                if text:
                    md_lines.append(f"{'#' * level} {text}\n")
            elif element.name == 'hr':
                md_lines.append("---\n")
            elif element.name == 'p':
                paragraph_text = process_paragraph_with_links(element, summary_citations)
                if paragraph_text:
                    md_lines.append(f"{paragraph_text}\n")
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li', recursive=False):
                    li_text = process_paragraph_with_links(li, summary_citations)
                    if li_text:
                        prefix = "- " if element.name == 'ul' else "1. "
                        md_lines.append(f"{prefix}{li_text}\n")
            elif element.name == 'li':
                li_text = process_paragraph_with_links(element, summary_citations)
                if li_text:
                    md_lines.append(f"{li_text}\n")
            elif element.name == 'blockquote':
                text = element.get_text(strip=True)
                if text:
                    md_lines.append(f"> {text}\n")
        
        summary_citations_cleaned = [remove_trailing_slash(url) for url in summary_citations]
        summary_citations = sorted(list(set(summary_citations_cleaned)))
        return "\n".join(md_lines).strip(), summary_citations
    
    # Extract content from markdown-content or prose container
    all_elements = []
    for element in report_container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'hr', 'blockquote']):
        # Skip if element is before or contains Finished
        if finished_element in element.find_all() or element == finished_element:
            continue
        all_elements.append(element)
    
    # Convert to Markdown
    md_lines = []
    summary_citations = set()
    
    for element in all_elements:
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(element.name[1])
            text = element.get_text(strip=True)
            if text:
                md_lines.append(f"{'#' * level} {text}\n")
        elif element.name == 'hr':
            md_lines.append("---\n")
        elif element.name == 'p':
            paragraph_text = process_paragraph_with_links(element, summary_citations)
            if paragraph_text:
                md_lines.append(f"{paragraph_text}\n")
        elif element.name in ['ul', 'ol']:
            for li in element.find_all('li', recursive=False):
                li_text = process_paragraph_with_links(li, summary_citations)
                if li_text:
                    prefix = "- " if element.name == 'ul' else "1. "
                    md_lines.append(f"{prefix}{li_text}\n")
        elif element.name == 'li':
            li_text = process_paragraph_with_links(element, summary_citations)
            if li_text:
                md_lines.append(f"{li_text}\n")
        elif element.name == 'blockquote':
            text = element.get_text(strip=True)
            if text:
                md_lines.append(f"> {text}\n")
    
    # Extract all unique URLs for summary_citations and remove trailing slashes
    summary_citations_cleaned = [remove_trailing_slash(url) for url in summary_citations]
    summary_citations = sorted(list(set(summary_citations_cleaned)))
    
    return "\n".join(md_lines).strip(), summary_citations

def process_paragraph_with_links(p_tag, citations_set):
    """
    Processes a paragraph tag, extracting text and appending links in @url format.
    For citation spans, the URL is appended to the preceding text.
    All http URLs must have @ prefix, and consecutive URLs are separated by comma.
    """
    # Get all text nodes and elements in order
    parts = []
    
    # Process the paragraph recursively, maintaining order
    def process_node(node):
        if isinstance(node, str):
            # Text node
            text = node.strip()
            if text:
                parts.append(('text', text))
        elif hasattr(node, 'name'):
            if node.name == 'strong':
                text = node.get_text(strip=True)
                if text:
                    parts.append(('strong', text))
            elif node.name == 'br':
                parts.append(('br', '\n'))
            elif node.name == 'a' and node.get('href'):
                url = remove_trailing_slash(node['href'])
                link_text = node.get_text(strip=True)
                parts.append(('link', (link_text, url)))
                citations_set.add(url)
            elif node.name == 'span' and 'citation' in node.get('class', []):
                # Citation span - find the link inside
                citation_link = node.find('a', href=True)
                if citation_link:
                    url = remove_trailing_slash(citation_link['href'])
                    parts.append(('citation', url))
                    citations_set.add(url)
            else:
                # Recursively process children
                for child in node.children:
                    process_node(child)
    
    # Process all children of the paragraph in order
    for child in p_tag.children:
        process_node(child)
    
    # Build the result string
    result_parts = []
    i = 0
    
    while i < len(parts):
        part_type, part_content = parts[i]
        
        if part_type == 'text':
            # Check if next part is a citation
            if i + 1 < len(parts) and parts[i + 1][0] == 'citation':
                # Append citation URL to text
                url = parts[i + 1][1]
                result_parts.append(f"{part_content} @{url} ")
                i += 2
            else:
                result_parts.append(part_content)
                i += 1
        elif part_type == 'strong':
            # Check if next part is a citation
            if i + 1 < len(parts) and parts[i + 1][0] == 'citation':
                url = parts[i + 1][1]
                result_parts.append(f"**{part_content}** @{url} ")
                i += 2
            else:
                result_parts.append(f"**{part_content}**")
                i += 1
        elif part_type == 'br':
            result_parts.append(part_content)
            i += 1
        elif part_type == 'link':
            link_text, url = part_content
            # Check for consecutive URLs
            consecutive_urls = [url]
            j = i + 1
            while j < len(parts) and parts[j][0] in ['link', 'citation']:
                if parts[j][0] == 'link':
                    _, next_url = parts[j][1]
                    consecutive_urls.append(next_url)
                else:
                    consecutive_urls.append(parts[j][1])
                j += 1
            
            # Format URLs: @url1, @url2, @url3
            if len(consecutive_urls) > 1:
                url_str = " , ".join([f"@{u}" for u in consecutive_urls])
                if result_parts:
                    result_parts[-1] += f" {url_str} "
                else:
                    result_parts.append(f"{link_text} {url_str} ")
                i = j
            else:
                # Single URL
                if result_parts:
                    result_parts[-1] += f" @{url} "
                else:
                    result_parts.append(f"{link_text} @{url} ")
                i += 1
        elif part_type == 'citation':
            url = part_content
            # Check for consecutive URLs
            consecutive_urls = [url]
            j = i + 1
            while j < len(parts) and parts[j][0] in ['link', 'citation']:
                if parts[j][0] == 'link':
                    _, next_url = parts[j][1]
                    consecutive_urls.append(next_url)
                else:
                    consecutive_urls.append(parts[j][1])
                j += 1
            
            # Format URLs: @url1, @url2, @url3
            if len(consecutive_urls) > 1:
                url_str = " , ".join([f"@{u}" for u in consecutive_urls])
                if result_parts:
                    result_parts[-1] += f" {url_str} "
                else:
                    result_parts.append(f" {url_str} ")
                i = j
            else:
                # Single URL
                if result_parts:
                    result_parts[-1] += f" @{url} "
                else:
                    result_parts.append(f" @{url} ")
                i += 1
        else:
            i += 1
    
    result = " ".join(result_parts)
    # Clean up multiple spaces but preserve line breaks
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r' \n', '\n', result)
    result = re.sub(r'\n ', '\n', result)
    
    # Ensure all http URLs have @ prefix (for any URLs that might have been missed)
    result = re.sub(r'(?<!@)(https?://[^\s\)\]>]+)', r'@\1', result)
    
    return result
    
def parse_perplexity_html(html_content):
    """Parse Perplexity deep research HTML trace into structured dict.

    Args:
        html_content: Raw HTML string of the Perplexity deep research page.

    Returns:
        dict with keys: query, final_report, all_source_links, summary_citations, chain_of_research
    """
    query, chain, all_links = extract_from_html(html_content)
    report_content, summary_citations = extract_report_from_html(html_content)

    return {
        "query": query,
        "chain_of_research": chain,
        "all_source_links": all_links,
        "final_report": report_content,
        "summary_citations": summary_citations,
    }


def process_single_file(html_filepath, output_filepath):
    """Process a single HTML file and generate JSON output (file-to-file convenience)."""
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        final_data = parse_perplexity_html(html_content)
    except FileNotFoundError:
        print(f"Error: HTML file not found at '{html_filepath}'")
        return False
    except Exception as e:
        print(f"An error occurred while processing the HTML file '{html_filepath}': {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully created '{output_filepath}'")
        return True
    except Exception as e:
        print(f"An error occurred while writing the JSON file '{output_filepath}': {e}")
        return False

def main():
    """
    Main function to process all HTML files in a directory.
    """
    # --- Configuration ---
    HTML_DIR = '/Users/zhanyuxiao/Desktop/Agent/HalluBench/data/benchmark/close-source/Perplexity/raw_html'
    OUTPUT_DIR = '/Users/zhanyuxiao/Desktop/Agent/HalluBench/data/benchmark/close-source/Perplexity/json'
    # -------------------

    # Get all HTML files in the directory
    html_dir_path = Path(HTML_DIR)
    if not html_dir_path.exists():
        print(f"Error: HTML directory not found at '{HTML_DIR}'")
        return
    
    html_files = list(html_dir_path.glob('*.html'))
    
    if not html_files:
        print(f"No HTML files found in '{HTML_DIR}'")
        return
    
    print(f"Found {len(html_files)} HTML file(s) to process\n")
    
    # Process each HTML file
    success_count = 0
    for html_file in html_files:
        print(f"Processing: {html_file.name}")
        
        # Generate output filename (same name but .json extension)
        output_filename = html_file.stem + '.json'
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        
        if process_single_file(str(html_file), output_filepath):
            success_count += 1
        print()  # Empty line for readability
    
    print(f"Processing complete: {success_count}/{len(html_files)} files processed successfully")

if __name__ == "__main__":
    main()