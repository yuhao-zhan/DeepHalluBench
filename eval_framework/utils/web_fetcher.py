"""
Web Content Fetcher for DeepHalluBench

Fetches and caches web content from URLs for chunk-level scoring.
Uses the same approach as the original HalluBench pipeline:
1. Primary: Jina AI via aiohttp (https://r.jina.ai/{url})
2. Fallback: Playwright + BeautifulSoup

Usage:
    fetcher = WebContentFetcher(cache_dir="./cache")
    web_content = fetcher.fetch(urls, trajectory_name)
    # Returns {url: cleaned_text_content} dict
    # Cached at cache_dir/web_content_cache/cache_{trajectory_name}.json
"""

import json
import os
import re
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

DESKTOP_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)


class WebContentFetcher:
    """Fetches web content from URLs and caches it for reuse."""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.web_cache_dir = os.path.join(cache_dir, "web_content_cache")
        os.makedirs(self.web_cache_dir, exist_ok=True)

    def load_cache(self, trajectory_name: str) -> Dict[str, str]:
        """Load cached web content for a trajectory."""
        cache_file = os.path.join(
            self.web_cache_dir, f"cache_{trajectory_name}.json"
        )
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                print(f"  📖 Loaded {len(content)} URLs from web cache: {cache_file}")
                return content
            except Exception as e:
                print(f"  ⚠️ Failed to load web cache: {e}")
        return {}

    def save_cache(self, web_content: Dict[str, str], trajectory_name: str) -> None:
        """Save web content to cache."""
        cache_file = os.path.join(
            self.web_cache_dir, f"cache_{trajectory_name}.json"
        )
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(web_content, f, ensure_ascii=False, indent=2)
            print(f"  💾 Saved {len(web_content)} URLs to web cache: {cache_file}")
        except Exception as e:
            print(f"  ⚠️ Failed to save web cache: {e}")

    def fetch(
        self,
        urls: List[str],
        trajectory_name: str,
        force_refetch: bool = False,
    ) -> Dict[str, str]:
        """
        Fetch web content for a list of URLs, using cache when available.

        Args:
            urls: List of URLs to fetch
            trajectory_name: Name of the trajectory for caching
            force_refetch: If True, refetch even if cached

        Returns:
            Dictionary mapping URLs to their content
        """
        if not urls:
            return {}

        # Try loading from cache first
        if not force_refetch:
            cached = self.load_cache(trajectory_name)
            if cached:
                # Return cached content, but try to fetch any new URLs
                missing = [u for u in urls if u not in cached]
                if missing:
                    print(f"  🌐 {len(missing)} URLs not in cache, attempting fetch...")
                    try:
                        fetched = asyncio.run(self._fetch_urls_async(missing))
                        cached.update(fetched)
                        self.save_cache(cached, trajectory_name)
                    except Exception as e:
                        print(f"  ⚠️ Async fetch failed: {e}")
                return {u: cached.get(u, "") for u in urls if u in cached}

        # Fetch all URLs
        print(f"  🌐 Fetching content for {len(urls)} URLs...")
        try:
            fetched = asyncio.run(self._fetch_urls_async(urls))
        except Exception as e:
            print(f"  ⚠️ Async fetch failed: {e}, trying offline fallback")
            fetched = {}
            for url in urls:
                fetched[url] = ""

        # Save to cache
        self.save_cache(fetched, trajectory_name)
        return fetched

    async def _fetch_urls_async(self, urls: List[str]) -> Dict[str, str]:
        """Fetch URLs using aiohttp + Jina AI, with parallel batches."""
        import multiprocessing

        all_results = {}

        # Process in batches for better memory management
        cpu_cores = multiprocessing.cpu_count()
        batch_size = max(1, min(10, len(urls) // max(cpu_cores, 1)))
        max_concurrent = min(cpu_cores * 2, max(1, len(urls) // max(batch_size, 1)))

        url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch: List[str]) -> Dict[str, str]:
            async with semaphore:
                return await self._fetch_batch_async(batch)

        tasks = [process_batch(batch) for batch in url_batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(batch_results):
            if isinstance(result, dict):
                all_results.update(result)
            else:
                logger.warning(f"Batch {i+1} failed: {result}")

        return all_results

    async def _fetch_batch_async(self, urls: List[str]) -> Dict[str, str]:
        """Fetch a batch of URLs in parallel."""
        import aiohttp

        results = {}
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_single_with_retry(session, url) for url in urls]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed:
                if isinstance(result, tuple) and len(result) == 2:
                    url, content = result
                    results[url] = content
        return results

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Extract clean text from HTML using BeautifulSoup if available."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            # Remove script/style elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Collapse multiple newlines
            return re.sub(r'\n{3,}', '\n\n', text)
        except Exception:
            # Fallback: rough HTML tag stripping
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    async def _fetch_single_with_retry(
        self, session, url: str, max_retries: int = 3
    ) -> Tuple[str, str]:
        """
        Fetch a single URL with retry logic.
        Tries: 1) Direct HTTP GET, 2) Jina AI reader, 3) Playwright+BeautifulSoup.
        """
        import aiohttp
        from aiohttp import ClientTimeout

        # Attempt 1: Direct HTTP GET with desktop UA
        for attempt in range(max_retries):
            try:
                async with session.get(
                    url,
                    headers={"User-Agent": DESKTOP_UA, "Accept": "text/html,application/xhtml+xml"},
                    timeout=ClientTimeout(total=15),
                    allow_redirects=True,
                ) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "")
                        if "text/html" in content_type or "text/plain" in content_type:
                            content = await response.text()
                            if len(content) > 100:
                                return url, self._html_to_text(content)
                    # Non-200 or too short -> break to next method
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                break

        # Attempt 2: Jina AI reader
        for attempt in range(max_retries):
            try:
                jina_url = f"https://r.jina.ai/{url}"
                async with session.get(
                    jina_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        if "ERROR" not in content.upper() and len(content) > 100:
                            return url, content
                    # Non-200 or error content -> try Playwright fallback
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                break

        # Attempt 3: Playwright fallback
        try:
            playwright_content = await self._fetch_with_playwright(url)
            if playwright_content and not playwright_content.startswith("[Error]"):
                return url, playwright_content
        except Exception as e:
            return url, f"[Error] Playwright: {e}"

        return url, f"[Error] Failed to fetch: all 3 methods returned no content"

    async def _fetch_with_playwright(self, url: str) -> str:
        """Fetch URL using Playwright + BeautifulSoup (fallback method)."""
        try:
            from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
            from bs4 import BeautifulSoup
        except ImportError:
            return "[Error] Playwright or BeautifulSoup not installed"

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=DESKTOP_UA,
                    locale="en-US",
                    viewport={"width": 1280, "height": 800}
                )
                page = await context.new_page()
                try:
                    await page.goto(url, timeout=60000)
                    html = await page.content()
                    soup = BeautifulSoup(html, "lxml")

                    # Make links absolute
                    for a in soup.find_all("a", href=True):
                        text = a.get_text(strip=True) or "[No Text]"
                        raw_href = a["href"]
                        full_href = urljoin(url, raw_href)
                        a.replace_with(f"{text} ({full_href})")

                    if soup.body:
                        content = soup.body.get_text(separator="\n", strip=True)
                    else:
                        content = soup.get_text(separator="\n", strip=True)

                    await context.close()
                    await browser.close()
                    return content
                except PlaywrightTimeoutError:
                    await context.close()
                    await browser.close()
                    return "[Error] Timeout"
                except Exception as e:
                    await context.close()
                    await browser.close()
                    return f"[Error] {e!r}"
        except Exception as e:
            return f"[Error] Playwright: {e}"
