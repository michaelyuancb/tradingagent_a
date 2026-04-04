import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class LocalArtifactStore:
    """负责缓存与快照文件的本地存储。"""

    def __init__(self, cache_dir: str, snapshot_dir: str):
        """
        初始化本地工件存储。

        参数：
            cache_dir: 工具调用缓存目录。
            snapshot_dir: 周期性采集快照目录。

        返回：
            None: 无返回值。
        """
        self.cache_dir = Path(cache_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def build_cache_key(self, tool_name: str, params: dict[str, Any]) -> str:
        """
        为工具调用构建稳定缓存键。

        参数：
            tool_name: 工具名称。
            params: 本次调用的参数字典。

        返回：
            str: 基于工具名与参数生成的哈希键。
        """
        normalized = {
            "tool_name": tool_name,
            "params": self._normalize_value(params),
        }
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def load_cache(self, tool_name: str, params: dict[str, Any]) -> tuple[Any, Path] | None:
        """
        尝试读取工具调用缓存。

        参数：
            tool_name: 工具名称。
            params: 本次调用的参数字典。

        返回：
            tuple[Any, Path] | None: 命中时返回结果值与载荷路径，否则返回 None。
        """
        manifest_path = self._cache_manifest_path(tool_name, params)
        if not manifest_path.exists():
            return None

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_path = manifest_path.parent / manifest["payload_file"]
        if not payload_path.exists():
            return None
        return self._read_payload(payload_path, manifest["payload_format"]), payload_path

    def save_cache(
        self,
        tool_name: str,
        params: dict[str, Any],
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        将工具结果写入缓存目录。

        参数：
            tool_name: 工具名称。
            params: 本次调用的参数字典。
            value: 需要保存的结果值。
            metadata: 附加元数据。

        返回：
            Path: 保存后的载荷文件路径。
        """
        cache_key = self.build_cache_key(tool_name, params)
        directory = self.cache_dir / tool_name / cache_key
        directory.mkdir(parents=True, exist_ok=True)
        return self._write_artifact(directory, tool_name, params, value, metadata or {})

    def save_snapshot(
        self,
        tool_name: str,
        params: dict[str, Any],
        value: Any,
        snapshot_group: str = "daily",
        snapshot_date: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        将工具结果写入可审计快照目录。

        参数：
            tool_name: 工具名称。
            params: 本次调用的参数字典。
            value: 需要保存的结果值。
            snapshot_group: 快照分组名。
            snapshot_date: 快照日期，格式为 YYYY-MM-DD。
            metadata: 附加元数据。

        返回：
            Path: 保存后的载荷文件路径。
        """
        snapshot_day = snapshot_date or date.today().isoformat()
        cache_key = self.build_cache_key(tool_name, params)
        directory = self.snapshot_dir / snapshot_group / snapshot_day / tool_name / cache_key
        directory.mkdir(parents=True, exist_ok=True)
        return self._write_artifact(directory, tool_name, params, value, metadata or {})

    def _cache_manifest_path(self, tool_name: str, params: dict[str, Any]) -> Path:
        """
        返回缓存 manifest 文件路径。

        参数：
            tool_name: 工具名称。
            params: 本次调用的参数字典。

        返回：
            Path: manifest 文件路径。
        """
        cache_key = self.build_cache_key(tool_name, params)
        return self.cache_dir / tool_name / cache_key / "manifest.json"

    def _write_artifact(
        self,
        directory: Path,
        tool_name: str,
        params: dict[str, Any],
        value: Any,
        metadata: dict[str, Any],
    ) -> Path:
        """
        将工具结果及 manifest 写入指定目录。

        参数：
            directory: 保存目录。
            tool_name: 工具名称。
            params: 本次调用的参数字典。
            value: 需要保存的结果值。
            metadata: 附加元数据。

        返回：
            Path: 保存后的载荷文件路径。
        """
        payload_format = self._infer_payload_format(value)
        payload_path = directory / f"payload.{self._payload_suffix(payload_format)}"
        self._write_payload(payload_path, value, payload_format)

        manifest = {
            "tool_name": tool_name,
            "params": self._normalize_value(params),
            "payload_file": payload_path.name,
            "payload_format": payload_format,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": self._normalize_value(metadata),
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload_path

    def _infer_payload_format(self, value: Any) -> str:
        """
        推断结果值应采用的保存格式。

        参数：
            value: 工具返回值。

        返回：
            str: 存储格式标识。
        """
        if isinstance(value, pd.DataFrame):
            return "csv"
        if isinstance(value, (dict, list, tuple)):
            return "json"
        return "text"

    def _payload_suffix(self, payload_format: str) -> str:
        """
        返回载荷格式对应的文件后缀。

        参数：
            payload_format: 存储格式标识。

        返回：
            str: 文件后缀。
        """
        return {
            "csv": "csv",
            "json": "json",
            "text": "txt",
        }[payload_format]

    def _write_payload(self, payload_path: Path, value: Any, payload_format: str) -> None:
        """
        将结果值写入文件。

        参数：
            payload_path: 目标文件路径。
            value: 需要保存的结果值。
            payload_format: 存储格式标识。

        返回：
            None: 无返回值。
        """
        if payload_format == "csv":
            value.to_csv(payload_path, index=False, encoding="utf-8-sig")
            return
        if payload_format == "json":
            payload_path.write_text(
                json.dumps(self._normalize_value(value), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return
        payload_path.write_text(str(value), encoding="utf-8")

    def _read_payload(self, payload_path: Path, payload_format: str) -> Any:
        """
        按格式读取已保存结果。

        参数：
            payload_path: 载荷文件路径。
            payload_format: 存储格式标识。

        返回：
            Any: 读取后的结果对象。
        """
        if payload_format == "csv":
            return pd.read_csv(payload_path)
        if payload_format == "json":
            return json.loads(payload_path.read_text(encoding="utf-8"))
        return payload_path.read_text(encoding="utf-8")

    def _normalize_value(self, value: Any) -> Any:
        """
        将对象规范化为可 JSON 序列化结构。

        参数：
            value: 待规范化对象。

        返回：
            Any: 规范化后的对象。
        """
        if isinstance(value, dict):
            return {
                str(key): self._normalize_value(val)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        return value
