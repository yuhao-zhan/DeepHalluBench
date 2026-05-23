"""
DeepHalluBench Configuration

Unified configuration system for all models and API settings.
Supports multiple LLM providers: OpenAI, Anthropic, Aliyun (DeepSeek), and more.
"""

import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    provider: str = "openai"  # openai, anthropic, aliyun, deepseek, etc.
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 4096
    temperature: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key[:10] + "..." if len(self.api_key) > 10 else self.api_key,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@dataclass
class NLIConfig:
    """Configuration for NLI model."""
    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    model_path: Optional[str] = None  # Local model path if using local model
    device: str = "cuda"
    num_labels: int = 3  # entailment, contradiction, neutral
    threshold: float = 0.99  # High confidence threshold for NLI

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "num_labels": self.num_labels,
            "threshold": self.threshold,
        }


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "BAAI/bge-large-zh-v1.5"  # Default model for multilingual support
    model_path: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }


@dataclass
class RerankerConfig:
    """Configuration for reranking model."""
    model_name: str = "BAAI/bge-reranker-large"
    model_path: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 32
    use_amp: bool = True  # Automatic mixed precision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "use_amp": self.use_amp,
        }


@dataclass
class EvaluationConfig:
    """Main configuration for DeepHalluBench evaluation."""
    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Model settings
    nli: NLIConfig = field(default_factory=NLIConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    # Evaluation parameters
    nli_threshold: float = 0.95
    similarity_threshold: float = 0.4
    top_k_chunks: int = 5
    chunk_overlap: int = 3
    chunk_size: int = 15  # sentences per chunk

    # GPU settings
    num_gpus: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    # Cache settings
    cache_dir: str = "./cache"
    output_dir: str = "./results"

    # Parallel processing
    max_workers: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.to_dict(),
            "nli": self.nli.to_dict(),
            "embedding": self.embedding.to_dict(),
            "reranker": self.reranker.to_dict(),
            "nli_threshold": self.nli_threshold,
            "similarity_threshold": self.similarity_threshold,
            "top_k_chunks": self.top_k_chunks,
            "chunk_overlap": self.chunk_overlap,
            "chunk_size": self.chunk_size,
            "num_gpus": self.num_gpus,
            "gpu_ids": self.gpu_ids,
            "cache_dir": self.cache_dir,
            "output_dir": self.output_dir,
            "max_workers": self.max_workers,
        }


class ConfigManager:
    """Manages configuration loading and saving."""

    DEFAULT_CONFIG = EvaluationConfig()

    @staticmethod
    def from_env() -> EvaluationConfig:
        """Load configuration from environment variables."""
        config = EvaluationConfig()

        # LLM configuration
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        config.llm.provider = llm_provider
        config.llm.api_key = os.getenv("LLM_API_KEY", "")
        config.llm.model = os.getenv("LLM_MODEL", "gpt-4o")
        config.llm.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

        # Set provider-specific defaults
        if llm_provider == "anthropic":
            if not config.llm.base_url or config.llm.base_url == "https://api.openai.com/v1":
                config.llm.base_url = "https://api.anthropic.com"
            config.llm.model = config.llm.model or "claude-3-5-sonnet-20241022"
        elif llm_provider in ["aliyun", "deepseek"]:
            config.llm.base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            config.llm.model = config.llm.model or "deepseek-v3"

        # Model configuration
        config.nli.model_name = os.getenv("NLI_MODEL", config.nli.model_name)
        config.nli.model_path = os.getenv("NLI_MODEL_PATH")
        config.nli.threshold = float(os.getenv("NLI_THRESHOLD", str(config.nli.threshold)))

        config.embedding.model_name = os.getenv("EMBEDDING_MODEL", config.embedding.model_name)
        config.embedding.model_path = os.getenv("EMBEDDING_MODEL_PATH")

        config.reranker.model_name = os.getenv("RERANKER_MODEL", config.reranker.model_name)
        config.reranker.model_path = os.getenv("RERANKER_MODEL_PATH")

        # GPU configuration
        gpu_ids_str = os.getenv("GPU_IDS", "0")
        config.gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",")]
        config.num_gpus = int(os.getenv("NUM_GPUS", str(config.num_gpus)))

        # Evaluation parameters
        config.nli_threshold = float(os.getenv("NLI_THRESHOLD", str(config.nli_threshold)))
        config.similarity_threshold = float(os.getenv("SIM_THRESHOLD", str(config.similarity_threshold)))
        config.top_k_chunks = int(os.getenv("TOP_K_CHUNKS", str(config.top_k_chunks)))

        # Cache settings
        config.cache_dir = os.getenv("CACHE_DIR", config.cache_dir)
        config.output_dir = os.getenv("OUTPUT_DIR", config.output_dir)

        return config

    @staticmethod
    def from_file(config_path: str) -> EvaluationConfig:
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)

        config = EvaluationConfig()

        # Load LLM config
        if "llm" in data:
            llm_data = data["llm"]
            config.llm.provider = llm_data.get("provider", config.llm.provider)
            config.llm.model = llm_data.get("model", config.llm.model)
            config.llm.api_key = llm_data.get("api_key", config.llm.api_key)
            config.llm.base_url = llm_data.get("base_url", config.llm.base_url)
            config.llm.max_tokens = llm_data.get("max_tokens", config.llm.max_tokens)
            config.llm.temperature = llm_data.get("temperature", config.llm.temperature)

        # Load NLI config
        if "nli" in data:
            nli_data = data["nli"]
            config.nli.model_name = nli_data.get("model_name", config.nli.model_name)
            config.nli.model_path = nli_data.get("model_path")
            config.nli.threshold = nli_data.get("threshold", config.nli.threshold)

        # Load embedding config
        if "embedding" in data:
            emb_data = data["embedding"]
            config.embedding.model_name = emb_data.get("model_name", config.embedding.model_name)
            config.embedding.model_path = emb_data.get("model_path")

        # Load reranker config
        if "reranker" in data:
            rer_data = data["reranker"]
            config.reranker.model_name = rer_data.get("model_name", config.reranker.model_name)
            config.reranker.model_path = rer_data.get("model_path")

        # Load other settings
        config.nli_threshold = data.get("nli_threshold", config.nli_threshold)
        config.similarity_threshold = data.get("similarity_threshold", config.similarity_threshold)
        config.top_k_chunks = data.get("top_k_chunks", config.top_k_chunks)
        config.gpu_ids = data.get("gpu_ids", config.gpu_ids)
        config.num_gpus = data.get("num_gpus", config.num_gpus)
        config.cache_dir = data.get("cache_dir", config.cache_dir)
        config.output_dir = data.get("output_dir", config.output_dir)

        return config

    @staticmethod
    def save_config(config: EvaluationConfig, config_path: str):
        """Save configuration to a JSON file."""
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    @staticmethod
    def create_example_config(example_path: str = "./config_example.json"):
        """Create an example configuration file."""
        example_config = EvaluationConfig()
        # Mask the API key in example
        example_config.llm.api_key = "YOUR_API_KEY_HERE"
        ConfigManager.save_config(example_config, example_path)
        print(f"Example configuration saved to {example_path}")
