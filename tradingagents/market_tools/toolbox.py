from pathlib import Path

import pandas as pd


class LocalMarketDataToolbox:
    """基于本地 tick 数据提供市场工具能力。"""

    def __init__(self, root_dir: str):
        """
        初始化市场数据工具箱。

        参数：
            root_dir: 本地 tick 数据根目录。

        返回：
            None: 无返回值。
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_ticks(self, symbol: str, trade_date: str, ticks: pd.DataFrame) -> Path:
        """
        保存某个标的某日的 tick 数据。

        参数：
            symbol: 标的代码。
            trade_date: 交易日期，格式为 YYYY-MM-DD。
            ticks: tick 数据表，至少包含 timestamp、price、volume 列。

        返回：
            Path: 保存后的文件路径。
        """
        file_path = self._tick_file_path(symbol, trade_date)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ticks.to_csv(file_path, index=False, encoding="utf-8-sig")
        return file_path

    def load_ticks(self, symbol: str, trade_date: str) -> pd.DataFrame:
        """
        读取某个标的某日的 tick 数据。

        参数：
            symbol: 标的代码。
            trade_date: 交易日期，格式为 YYYY-MM-DD。

        返回：
            pd.DataFrame: 读取后的 tick 数据表。
        """
        file_path = self._tick_file_path(symbol, trade_date)
        if not file_path.exists():
            raise FileNotFoundError(f"未找到 tick 数据文件：{file_path}")
        ticks = pd.read_csv(file_path)
        ticks["timestamp"] = pd.to_datetime(ticks["timestamp"])
        return ticks.sort_values("timestamp").reset_index(drop=True)

    def resample_ticks(self, ticks: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
        """
        将 tick 数据重采样为 K 线序列。

        参数：
            ticks: 输入 tick 数据表。
            rule: Pandas 重采样规则，例如 1min、5min。

        返回：
            pd.DataFrame: 包含 open、high、low、close、volume 的 K 线数据。
        """
        normalized = ticks.copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
        normalized = normalized.sort_values("timestamp").set_index("timestamp")

        price_ohlc = normalized["price"].resample(rule).ohlc()
        volume = normalized["volume"].resample(rule).sum().rename("volume")
        bars = price_ohlc.join(volume).dropna().reset_index()
        return bars

    def build_bars(self, symbol: str, trade_date: str, rule: str = "1min") -> pd.DataFrame:
        """
        读取本地 tick 数据并直接生成 K 线序列。

        参数：
            symbol: 标的代码。
            trade_date: 交易日期，格式为 YYYY-MM-DD。
            rule: Pandas 重采样规则，例如 1min、5min。

        返回：
            pd.DataFrame: 生成后的 K 线数据。
        """
        ticks = self.load_ticks(symbol, trade_date)
        return self.resample_ticks(ticks, rule=rule)

    def get_execution_price(
        self,
        symbol: str,
        trade_date: str,
        side: str,
        decision_time: str | None = None,
    ) -> float:
        """
        根据本地 tick 数据估算交易执行价格。

        参数：
            symbol: 标的代码。
            trade_date: 交易日期，格式为 YYYY-MM-DD。
            side: 交易方向，仅用于未来扩展。
            decision_time: 决策时间，格式兼容 pandas 时间解析。

        返回：
            float: 估算出的执行价格。
        """
        ticks = self.load_ticks(symbol, trade_date)
        if decision_time is None:
            return float(ticks.iloc[0]["price"])

        timestamp = pd.Timestamp(decision_time)
        matched = ticks[ticks["timestamp"] >= timestamp]
        if matched.empty:
            return float(ticks.iloc[-1]["price"])
        return float(matched.iloc[0]["price"])

    def _tick_file_path(self, symbol: str, trade_date: str) -> Path:
        """
        生成某个标的某日 tick 文件路径。

        参数：
            symbol: 标的代码。
            trade_date: 交易日期，格式为 YYYY-MM-DD。

        返回：
            Path: 约定的 tick 文件路径。
        """
        safe_symbol = symbol.replace(".", "_")
        return self.root_dir / safe_symbol / f"{trade_date}.csv"
