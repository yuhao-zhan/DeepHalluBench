"""
Configuration module for DeepHalluBench evaluation.

Supports loading configuration from environment variables or JSON files.

Example:
    from eval_framework.config import EvaluationConfig, ConfigManager

    # From environment
    config = ConfigManager.from_env()

    # From file
    config = ConfigManager.from_file("config.json")
"""

import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 4096
    temperature: float = 0.1


@dataclass
class NLIConfig:
    """Configuration for NLI model."""
    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    model_path: Optional[str] = None
    device: str = "cuda"
    num_labels: int = 3
    threshold: float = 0.99


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "BAAI/bge-m3"
    model_path: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512


@dataclass
class RerankerConfig:
    """Configuration for reranking model."""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_path: Optional[str] = None
    device: str = "cuda:0"
    batch_size: int = 32
    use_amp: bool = True


@dataclass
class EvaluationConfig:
    """Main configuration for DeepHalluBench evaluation."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    nli_threshold: float = 0.95
    similarity_threshold: float = 0.4
    top_k_chunks: int = 5
    chunk_overlap: int = 3
    chunk_size: int = 15

    num_gpus: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    cache_dir: str = "./cache"
    output_dir: str = "./results"
    max_workers: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "max_tokens": self.llm.max_tokens,
            },
            "nli": {
                "model_name": self.nli.model_name,
                "threshold": self.nli.threshold,
            },
            "embedding": {"model_name": self.embedding.model_name},
            "reranker": {"model_name": self.reranker.model_name},
            "nli_threshold": self.nli_threshold,
            "similarity_threshold": self.similarity_threshold,
            "top_k_chunks": self.top_k_chunks,
            "num_gpus": self.num_gpus,
            "gpu_ids": self.gpu_ids,
        }


class ConfigManager:
    """Manages configuration loading and saving."""

    @staticmethod
    def from_env() -> EvaluationConfig:
        """Load configuration from environment variables."""
        config = EvaluationConfig()

        # LLM configuration
        config.llm.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        config.llm.api_key = os.getenv("LLM_API_KEY", "")
        config.llm.model = os.getenv("LLM_MODEL", "gpt-4o")
        config.llm.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

        if config.llm.provider == "anthropic":
            config.llm.base_url = "https://api.anthropic.com"
        elif config.llm.provider in ["aliyun", "deepseek"]:
            config.llm.base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            config.llm.model = config.llm.model or "deepseek-v3"

        # Model configuration
        config.nli.model_name = os.getenv("NLI_MODEL", config.nli.model_name)
        config.nli.threshold = float(os.getenv("NLI_THRESHOLD", str(config.nli.threshold)))

        config.embedding.model_name = os.getenv("EMBEDDING_MODEL", config.embedding.model_name)
        config.reranker.model_name = os.getenv("RERANKER_MODEL", config.reranker.model_name)

        # GPU configuration
        gpu_ids_str = os.getenv("GPU_IDS", "0")
        config.gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",")]
        config.num_gpus = int(os.getenv("NUM_GPUS", str(config.num_gpus)))

        # Evaluation parameters
        config.nli_threshold = float(os.getenv("NLI_THRESHOLD", str(config.nli_threshold)))

        # Directories
        config.cache_dir = os.getenv("CACHE_DIR", config.cache_dir)
        config.output_dir = os.getenv("OUTPUT_DIR", config.output_dir)

        return config

    @staticmethod
    def from_file(config_path: str) -> EvaluationConfig:
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)

        config = EvaluationConfig()

        if "llm" in data:
            config.llm.provider = data["llm"].get("provider", config.llm.provider)
            config.llm.model = data["llm"].get("model", config.llm.model)
            config.llm.api_key = data["llm"].get("api_key", config.llm.api_key)
            config.llm.base_url = data["llm"].get("base_url", config.llm.base_url)

        if "nli" in data:
            config.nli.model_name = data["nli"].get("model_name", config.nli.model_name)
            config.nli.threshold = data["nli"].get("threshold", config.nli.threshold)

        config.nli_threshold = data.get("nli_threshold", config.nli_threshold)
        config.gpu_ids = data.get("gpu_ids", config.gpu_ids)
        config.num_gpus = data.get("num_gpus", config.num_gpus)
        config.cache_dir = data.get("cache_dir", config.cache_dir)
        config.output_dir = data.get("output_dir", config.output_dir)

        return config
