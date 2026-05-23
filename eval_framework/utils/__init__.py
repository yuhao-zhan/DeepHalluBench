"""
LLM Client Utility for DeepHalluBench

Provides a unified interface for calling various LLM APIs:
- OpenAI
- Anthropic
- Aliyun (DeepSeek)
- Other OpenAI-compatible APIs
"""

import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Container for LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Providers:
    - openai: OpenAI API (https://api.openai.com/v1)
    - anthropic: Anthropic API (https://api.anthropic.com)
    - aliyun: Aliyun/DashScope API (https://dashscope.aliyuncs.com)
    - deepseek: DeepSeek API (https://api.deepseek.com)
    """

    PROVIDER_URLS = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deepseek": "https://api.deepseek.com",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: int = 120,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: API key for the provider
            model: Model name to use
            provider: Provider name (openai, anthropic, aliyun, deepseek)
            base_url: Custom base URL (overrides provider default)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.base_url = base_url or self.PROVIDER_URLS.get(self.provider, self.PROVIDER_URLS["openai"])
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Initialize the appropriate client
        self._client = None
        # Only initialize if API key is valid
        if api_key and api_key not in ("YOUR_API_KEY_HERE", "", None):
            self._init_client()

    def _init_client(self):
        """Initialize the underlying HTTP client."""
        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("⚠️ Anthropic package not installed. Install with: pip install anthropic")
                self._client = None
        else:
            # Use OpenAI client for most providers (OpenAI-compatible)
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                print("⚠️ OpenAI package not installed. Install with: pip install openai")
                self._client = None

    @classmethod
    def from_config(cls, llm_config) -> "LLMClient":
        """Create client from LLMConfig object."""
        return cls(
            api_key=llm_config.api_key,
            model=llm_config.model,
            provider=llm_config.provider,
            base_url=llm_config.base_url,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
        )

    @classmethod
    def from_env(cls) -> Optional["LLMClient"]:
        """Create client from environment variables."""
        api_key = os.getenv("LLM_API_KEY", "")
        if not api_key:
            print("⚠️ LLM_API_KEY not set in environment")
            return None

        return cls(
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            provider=os.getenv("LLM_PROVIDER", "openai").lower(),
            base_url=os.getenv("LLM_BASE_URL"),
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        retry: int = 3,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            retry: Number of retries on failure

        Returns:
            Generated text content
        """
        if not self._client:
            raise RuntimeError("LLM client not initialized. Check API key and package installation.")

        # Fail fast for placeholder API keys
        if self.api_key in ("YOUR_API_KEY_HERE", "", None) or not self.api_key:
            raise RuntimeError("API key not configured. Set LLM_API_KEY environment variable or provide api_key in config.")

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        for attempt in range(retry):
            try:
                if self.provider == "anthropic":
                    response = self._client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    return response.content[0].text
                else:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": user_prompt})
                    response = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                    )
                    return response.choices[0].message.content

            except Exception as e:
                print(f"⚠️ LLM generation attempt {attempt + 1} failed: {e}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        return ""

    def generate_batch(
        self,
        prompts: list,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_workers: int = 4,
    ) -> list:
        """
        Generate responses for multiple prompts in parallel.

        Args:
            prompts: List of user prompts
            system_prompt: System prompt (shared)
            max_tokens: Override max tokens
            temperature: Override temperature
            max_workers: Max parallel workers

        Returns:
            List of generated responses
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(prompts)

        def generate_single(args):
            i, prompt = args
            try:
                return i, self.generate(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"⚠️ Batch generation failed for prompt {i}: {e}")
                return i, ""

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_single, (i, p)): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result

        return results
