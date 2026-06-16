"""统一的 YAML/JSON 文件读写管理器。

负责所有明文配置文件的原子读写，确保队列操作实时落盘。
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class FileStorageManager:
    """沙盒数据的统一文件存储管理器。

    所有 Agent 配置、案卷记录、文书终稿的读写都通过此类完成，
    杜绝直接文件操作导致的冲突和数据不一致。
    """

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created sandbox data directory: {self.base_dir}")

    # ── 案件路径管理 ──

    def get_case_agent_path(self, case_id: str, party_role: str) -> Path:
        """获取基于案件的 Agent 配置路径。

        Args:
            case_id: 案件 ID (如 '9')
            party_role: 当事人角色 ('plaintiff' 或 'defendant')

        Returns:
            Path: cases/case_{case_id}/{party_role}/
        """
        case_key = str(case_id)
        if case_key.startswith("case_"):
            case_key = case_key[5:]
        return self.base_dir / "cases" / f"case_{case_key}" / party_role

    def get_case_runtime_path(self, case_id: str) -> Path:
        """获取案件级运行态文件路径。"""
        case_key = str(case_id)
        if case_key.startswith("case_"):
            case_key = case_key[5:]
        return self.base_dir / "cases" / f"case_{case_key}" / "runtime.yaml"

    def migrate_client_to_case_structure(
        self, old_client_path: Path, case_id: str, party_role: str
    ) -> Path:
        """将旧的客户端数据迁移到新的案件结构。

        Args:
            old_client_path: 旧路径 (如 agents/clients/client_001)
            case_id: 案件 ID
            party_role: 当事人角色

        Returns:
            Path: 新路径
        """
        import shutil

        new_path = self.get_case_agent_path(case_id, party_role)

        if new_path.exists():
            logger.warning(f"Target path already exists: {new_path}")
            return new_path

        new_path.mkdir(parents=True, exist_ok=True)

        # 复制配置文件
        old_config = old_client_path / "config.yaml"
        if old_config.exists():
            shutil.copy2(old_config, new_path / "config.yaml")
            logger.info(f"Migrated {old_client_path} -> {new_path}")

        return new_path

    # ── Agent 配置读写 ──

    def load_agent_config(self, agent_path: Union[str, Path]) -> dict:
        """加载 Agent 的 config.yaml。"""
        config_file = Path(agent_path) / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"Config root is not a mapping, using empty config instead: {config_file}")
            data = {}
        logger.debug(f"Loaded config from {config_file}")
        return data

    def load_case_runtime(self, case_id: str) -> dict:
        """加载案件级 runtime.yaml。"""
        runtime_file = self.get_case_runtime_path(case_id)
        if not runtime_file.exists():
            raise FileNotFoundError(f"Case runtime not found: {runtime_file}")
        with open(runtime_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"Case runtime root is not a mapping, using empty runtime instead: {runtime_file}")
            data = {}
        return data

    def save_case_runtime(self, case_id: str, data: dict) -> None:
        """原子写入案件级 runtime.yaml。"""
        runtime_file = self.get_case_runtime_path(case_id)
        runtime_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = runtime_file.with_suffix(".yaml.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        tmp_file.replace(runtime_file)
        logger.debug(f"Saved case runtime to {runtime_file}")

    def save_agent_config(self, agent_path: Union[str, Path], data: dict) -> None:
        """原子写入 Agent 的 config.yaml。"""
        config_file = Path(agent_path) / "config.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = config_file.with_suffix(".yaml.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        tmp_file.replace(config_file)
        logger.debug(f"Saved config to {config_file}")

    def update_agent_field(
        self, agent_path: Union[str, Path], field_key: str, new_data: Any
    ) -> None:
        """精准更新 config.yaml 中的单个字段。"""
        config = self.load_agent_config(agent_path)
        config[field_key] = new_data
        self.save_agent_config(agent_path, config)
        logger.debug(f"Updated field '{field_key}' in {agent_path}")

    # ── 队列操作（立即落盘） ──

    def append_to_queue(
        self, agent_path: Union[str, Path], queue_field: str, item: str
    ) -> None:
        """向队列字段追加元素并立即落盘。队列中不重复保存同一案件。"""
        config = self.load_agent_config(agent_path)
        queue = config.get(queue_field, [])
        if not isinstance(queue, list):
            queue = []
        normalized_queue = []
        for existing in queue:
            if existing not in normalized_queue:
                normalized_queue.append(existing)
        if item and item not in normalized_queue:
            normalized_queue.append(item)
        config[queue_field] = normalized_queue
        self.save_agent_config(agent_path, config)

    def pop_from_queue(
        self, agent_path: Union[str, Path], queue_field: str
    ) -> Optional[str]:
        """从队列字段弹出首个元素并立即落盘。"""
        config = self.load_agent_config(agent_path)
        queue = config.get(queue_field, [])
        if not queue:
            return None
        item = queue.pop(0)
        config[queue_field] = queue
        self.save_agent_config(agent_path, config)
        return item

    # ── 案卷记录读写 ──

    def save_case_record(
        self, case_dir: Union[str, Path], stage_name: str, data: dict
    ) -> None:
        """保存案卷阶段记录到律所 /cases/ 目录。"""
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        record_file = case_dir / f"{stage_name}_record.json"
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved case record: {record_file}")

    def load_case_record(
        self, case_dir: Union[str, Path], stage_name: str
    ) -> Optional[dict]:
        """读取案卷阶段记录。"""
        record_file = Path(case_dir) / f"{stage_name}_record.json"
        if not record_file.exists():
            return None
        with open(record_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_case_document(
        self, case_dir: Union[str, Path], doc_name: str, content: str
    ) -> None:
        """保存文书终稿（起诉状、答辩状等）。"""
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        doc_file = case_dir / f"{doc_name}.txt"
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved document: {doc_file}")

    def load_case_document(
        self, case_dir: Union[str, Path], doc_name: str
    ) -> Optional[str]:
        """读取文书终稿。"""
        doc_file = Path(case_dir) / f"{doc_name}.txt"
        if not doc_file.exists():
            return None
        with open(doc_file, "r", encoding="utf-8") as f:
            return f.read()

    # ── 通用 YAML/JSON 读写 ──

    def load_yaml(self, filepath: Union[str, Path]) -> dict:
        """加载任意 YAML 文件。"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"YAML root is not a mapping, using empty data instead: {filepath}")
            return {}
        return data

    def save_yaml(self, filepath: Union[str, Path], data: dict) -> None:
        """保存任意 YAML 文件。"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
