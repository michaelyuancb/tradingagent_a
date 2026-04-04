import importlib.util
import os
from pathlib import Path
from pprint import pformat


LAST_CONFIG_PATH = Path(__file__).with_name("last_config.py")
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
LOCAL_DATA_DIR = os.getenv(
    "TRADINGAGENTS_LOCAL_DATA_DIR",
    os.path.join(PROJECT_DIR, "local_data"),
)

DEFAULT_CONFIG = {
    "project_dir": PROJECT_DIR,
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        PROJECT_DIR,
        "dataflows/data_cache",
    ),
    "local_data_dir": LOCAL_DATA_DIR,
    "data_tools_cache_dir": os.path.join(LOCAL_DATA_DIR, "data_tools", "cache"),
    "data_tools_snapshot_dir": os.path.join(LOCAL_DATA_DIR, "data_tools", "snapshots"),
    "market_data_dir": os.path.join(LOCAL_DATA_DIR, "market_tools"),
    "agent_output_dir": os.path.join(LOCAL_DATA_DIR, "agents"),
    "backtest_output_dir": os.path.join(LOCAL_DATA_DIR, "backtests"),
    "market_region": "cn_a",
    # LLM 配置
    "llm_provider": "azure",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": "YOUR_AZURE_API_URL",
    "azure_api_version": "2024-12-01-preview",
    "content_filter_max_retries": 2,
    "content_filter_skip_message": "Skipped due to Azure content policy filter.",
    # 不同提供方的思考参数配置
    "google_thinking_level": "high",      # 例如 "high"、"minimal"
    "openai_reasoning_effort": "high",    # 可选 "medium"、"high"、"low"
    "anthropic_effort": "high",           # 可选 "high"、"medium"、"low"
    # 分析师报告、辩论内容与最终决策的输出语言。
    # 如存在必须保留的机器可读标记，应仅保留该标记本身的英文形式。
    "internal_language": "English",
    "final_output_language": "Chinese",
    "output_language": "Chinese",
    "selected_analysts": ["market", "social", "news", "fundamentals"],
    "research_depth": 2,
    # 辩论与讨论轮次配置
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # 超时配置（秒）
    "timeout": 180,
    # 数据供应商配置
    # 类别级配置（该类别下工具默认沿用）
    "data_vendors": {
        "core_stock_apis": "akshare",
        "technical_indicators": "akshare",
        "fundamental_data": "akshare",
        "news_data": "akshare",
    },
    # 工具级配置（优先级高于类别级）
    "tool_vendors": {
        # 示例："get_market_news": "akshare",
    },
}


def load_last_config() -> dict:
    """
    加载上一次 CLI 保存的配置。

    返回：
        dict: 上一次保存的配置；若文件不存在或内容无效，则返回空字典。
    """
    if not LAST_CONFIG_PATH.exists():
        return {}

    try:
        spec = importlib.util.spec_from_file_location("tradingagents.last_config", LAST_CONFIG_PATH)
        if spec is None or spec.loader is None:
            return {}

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = getattr(module, "LAST_CONFIG", {})
        if isinstance(config, dict):
            return config.copy()
    except Exception:
        return {}

    return {}


def build_runtime_config() -> dict:
    """
    构建运行时配置。

    返回：
        dict: 以默认配置为基底，并叠加上一次 CLI 保存配置后的结果。
    """
    runtime_config = DEFAULT_CONFIG.copy()
    runtime_config.update(load_last_config())
    return runtime_config


def save_last_config(config: dict) -> None:
    """
    将当前配置写入 `last_config.py`。

    参数：
        config: 需要持久化保存的配置映射。

    返回：
        None: 无返回值。
    """
    serializable_config = config.copy()
    LAST_CONFIG_PATH.write_text(
        "# 该文件由 CLI 自动生成，用于保存上一次确认后的配置。\n"
        f"LAST_CONFIG = {pformat(serializable_config, width=100, sort_dicts=False)}\n",
        encoding="utf-8",
    )
