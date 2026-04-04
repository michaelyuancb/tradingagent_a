import logging
import os
import time
from contextlib import contextmanager
from http.client import RemoteDisconnected
import sys
from typing import Annotated

import akshare as ak
import pandas as pd
import requests
from akshare.utils import tqdm as ak_tqdm
from stockstats import wrap

from .a_share_common import (
    format_date_for_api,
    get_previous_trade_date,
    normalize_ashare_symbol,
    to_exchange_prefixed_symbol,
    to_plain_symbol,
)
from .config import get_config

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_akshare_progress():
    """
    临时关闭 AkShare 内部 tqdm 进度条输出。

    返回：
        None: 无返回值。
    """
    replacements = {}
    silent_get_tqdm = lambda enable=True: (lambda iterable, *args, **kwargs: iterable)

    replacements[(ak_tqdm, "get_tqdm")] = ak_tqdm.get_tqdm
    ak_tqdm.get_tqdm = silent_get_tqdm

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("akshare."):
            continue
        if hasattr(module, "get_tqdm"):
            replacements[(module, "get_tqdm")] = getattr(module, "get_tqdm")
            setattr(module, "get_tqdm", silent_get_tqdm)
    try:
        yield
    finally:
        for (module, attr), original in replacements.items():
            setattr(module, attr, original)


def _is_retryable_akshare_error(exc: Exception) -> bool:
    """
    判断异常是否属于可重试的 AkShare 网络错误。

    参数：
        exc: 待判断的异常对象。

    返回：
        bool: 条件满足时返回 True，否则返回 False。
    """
    if isinstance(exc, (requests.exceptions.RequestException, RemoteDisconnected, TimeoutError)):
        return True

    message = str(exc)
    retryable_markers = (
        "Remote end closed connection without response",
        "Connection aborted",
        "Read timed out",
        "ConnectTimeout",
        "Max retries exceeded",
    )
    return any(marker in message for marker in retryable_markers)


def fetch_with_cache(
    func,
    retries: int = 3,
    retry_delay: float = 1.0,
    log_failure_as_exception: bool = True,
):
    """
    执行 AkShare 获取函数，并在异常时补充上下文日志后重新抛出。
    
    参数：
        func: 用于抓取外部数据的可调用对象。
        retries: 最大重试次数。
        retry_delay: 首次重试前的等待秒数，后续按次数递增。
        log_failure_as_exception: 最终失败时是否打印异常堆栈。
    
    返回：
        Any: 外部查询返回的数据。
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries or not _is_retryable_akshare_error(exc):
                if log_failure_as_exception:
                    logger.exception("A-share data fetch failed after %s attempt(s): %s", attempt, exc)
                else:
                    logger.warning("A-share data fetch failed after %s attempt(s): %s", attempt, exc)
                raise
            logger.warning("A-share data fetch attempt %s/%s failed: %s", attempt, retries, exc)
            time.sleep(retry_delay * attempt)
    raise last_exc


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    将 A 股 OHLCV 数据规范化为 stockstats 所需格式。
    
    参数：
        data: 输入数据。
    
    返回：
        pd.DataFrame: 处理后的数据表。
    """
    renamed = data.rename(
        columns={
            "日期": "Date",
            "date": "Date",
            "开盘": "Open",
            "open": "Open",
            "最高": "High",
            "high": "High",
            "最低": "Low",
            "low": "Low",
            "收盘": "Close",
            "close": "Close",
            "成交量": "Volume",
            "amount": "Volume",
        }
    ).copy()

    renamed["Date"] = pd.to_datetime(renamed["Date"], errors="coerce")
    renamed = renamed.dropna(subset=["Date"])

    price_cols = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in renamed.columns]
    renamed[price_cols] = renamed[price_cols].apply(pd.to_numeric, errors="coerce")
    renamed = renamed.dropna(subset=["Close"])
    renamed[price_cols] = renamed[price_cols].ffill().bfill()

    return renamed.loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]]


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """
    在无未来函数前提下获取带缓存的 A 股 OHLCV 数据。
    
    参数：
        symbol: 待分析标的的 A 股股票代码。
        curr_date: 当前分析或交易日期，格式为 YYYY-MM-DD。
    
    返回：
        pd.DataFrame: 处理后的数据表。
    """
    config = get_config()
    normalized_symbol = normalize_ashare_symbol(symbol)
    aligned_trade_date = get_previous_trade_date(curr_date)
    curr_date_dt = pd.to_datetime(aligned_trade_date)

    today_date = pd.Timestamp.today().normalize()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{normalized_symbol.replace('.', '_')}-akshare-qfq-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        try:
            data = fetch_with_cache(
                lambda: ak.stock_zh_a_hist(
                    symbol=to_plain_symbol(symbol),
                    period="daily",
                    start_date=format_date_for_api(start_str),
                    end_date=format_date_for_api(end_str),
                    adjust="qfq",
                ),
                log_failure_as_exception=False,
            )
        except Exception:  # noqa: BLE001
            with _suppress_akshare_progress():
                data = fetch_with_cache(
                    lambda: ak.stock_zh_a_hist_tx(
                        symbol=to_exchange_prefixed_symbol(symbol).lower(),
                        start_date=format_date_for_api(start_str),
                        end_date=format_date_for_api(end_str),
                        adjust="qfq",
                    )
                )
        data.to_csv(data_file, index=False, encoding="utf-8-sig")

    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_date_dt]

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """
    仅保留报告日期不晚于 curr_date 的报表行或列。
    
    参数：
        data: 输入数据。
        curr_date: 当前分析或交易日期，格式为 YYYY-MM-DD。
    
    返回：
        pd.DataFrame: 处理后的数据表。
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)

    if "REPORT_DATE" in data.columns:
        filtered = data.copy()
        filtered["REPORT_DATE"] = pd.to_datetime(filtered["REPORT_DATE"], errors="coerce")
        return filtered[filtered["REPORT_DATE"] <= cutoff]

    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    if mask.any():
        return data.loc[:, mask]
    return data


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        """
        返回股票技术指标对象。
        
        参数：
            symbol: 待分析标的的 A 股股票代码。
            indicator: 需要计算或查询的技术指标名称。
            curr_date: 当前分析或交易日期，格式为 YYYY-MM-DD。
        
        返回：
            Any: 当前查询结果。
        """
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
