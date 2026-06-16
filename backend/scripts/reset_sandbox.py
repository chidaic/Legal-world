"""重置沙盒存档脚本 - 清空所有 Agent 状态和检查点。"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add backend to path
_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))

from src.core.file_storage_manager import FileStorageManager
import shutil

SANDBOX_DATA_DIR = _backend_dir / "sandbox_data"


def reset_client_config(storage: FileStorageManager, agent_dir: Path) -> None:
    """重置当事人配置到初始状态。"""
    config = storage.load_agent_config(agent_dir)
    config.pop("chat_history_summary", None)
    long_term_memory = config.get("long_term_memory", {})
    if not isinstance(long_term_memory, dict):
        long_term_memory = {}
    long_term_memory.pop("stage_summaries", None)

    # 重置状态字段
    config["case_state"] = "空闲"
    config["long_term_memory"] = {
        "case_background": "",
        "core_demands": "",
    }

    # 清空地图状态
    if "map_state" in config:
        config["map_state"]["sitting"] = None

    storage.save_agent_config(agent_dir, config)
    print(f"[OK] 重置当事人: {config.get('profile', {}).get('name', 'Unknown')}")


def reset_lawyer_config(storage: FileStorageManager, agent_dir: Path) -> None:
    """重置律师配置到初始状态。"""
    config = storage.load_agent_config(agent_dir)
    config.pop("chat_history_summary", None)
    long_term_memory = config.get("long_term_memory", {})
    if not isinstance(long_term_memory, dict):
        long_term_memory = {}
    long_term_memory.pop("stage_summaries", None)

    # 重置状态字段
    config["current_handling_case"] = None
    config["case_queue"] = []
    config["long_term_memory"] = {
        "case_summary": "",
        "legal_relationship": "",
        "dispute_focus": "",
    }

    # 清空地图状态
    if "map_state" in config:
        config["map_state"]["sitting"] = None

    storage.save_agent_config(agent_dir, config)
    print(f"[OK] 重置律师: {config.get('profile', {}).get('name', 'Unknown')}")


def reset_judge_config(storage: FileStorageManager, agent_dir: Path) -> None:
    """重置法官配置到初始状态。"""
    config = storage.load_agent_config(agent_dir)
    config.pop("chat_history_summary", None)

    # 重置状态字段（法官通常没有持久状态，但保险起见）

    storage.save_agent_config(agent_dir, config)
    print(f"[OK] 重置法官: {config.get('profile', {}).get('name', 'Unknown')}")


def reset_checkpoints(checkpoint_dir: Path) -> None:
    """清空所有检查点文件。"""
    if not checkpoint_dir.exists():
        print("[OK] 检查点目录不存在，跳过")
        return

    for file in checkpoint_dir.glob("*.yaml"):
        file.unlink()
        print(f"[OK] 删除检查点: {file.name}")


def reset_output(output_dir: Path) -> None:
    """清空所有输出文件。"""
    if not output_dir.exists():
        print("[OK] 输出目录不存在，跳过")
        return

    for case_dir in output_dir.iterdir():
        if case_dir.is_dir():
            shutil.rmtree(case_dir)
            print(f"[OK] 删除输出: {case_dir.name}")


def main():
    """主函数：重置所有沙盒数据。"""
    print("=" * 60)
    print("开始重置沙盒存档...")
    print("=" * 60)

    storage = FileStorageManager(base_dir=SANDBOX_DATA_DIR)

    # 1. 重置所有当事人
    print("\n[1/5] 重置当事人配置...")
    clients_dir = SANDBOX_DATA_DIR / "agents" / "clients"
    if clients_dir.exists():
        for client_dir in clients_dir.iterdir():
            if client_dir.is_dir() and (client_dir / "config.yaml").exists():
                reset_client_config(storage, client_dir)

    # 2. 重置所有律师
    print("\n[2/5] 重置律师配置...")
    law_firms_dir = SANDBOX_DATA_DIR / "law_firms"
    if law_firms_dir.exists():
        for firm_dir in law_firms_dir.iterdir():
            if firm_dir.is_dir():
                lawyers_dir = firm_dir / "lawyers"
                if lawyers_dir.exists():
                    for lawyer_dir in lawyers_dir.iterdir():
                        if lawyer_dir.is_dir() and (lawyer_dir / "config.yaml").exists():
                            reset_lawyer_config(storage, lawyer_dir)

    # 3. 重置所有法官
    print("\n[3/5] 重置法官配置...")
    court_system_dir = SANDBOX_DATA_DIR / "court_system"
    if court_system_dir.exists():
        for court_dir in court_system_dir.iterdir():
            if court_dir.is_dir():
                judges_dir = court_dir / "judges"
                if judges_dir.exists():
                    for judge_dir in judges_dir.iterdir():
                        if judge_dir.is_dir() and (judge_dir / "config.yaml").exists():
                            reset_judge_config(storage, judge_dir)

    # 4. 清空检查点
    print("\n[4/5] 清空检查点...")
    checkpoint_dir = SANDBOX_DATA_DIR / "checkpoints"
    reset_checkpoints(checkpoint_dir)

    # 5. 清空输出
    print("\n[5/5] 清空输出文件...")
    output_dir = SANDBOX_DATA_DIR / "output"
    reset_output(output_dir)

    print("\n" + "=" * 60)
    print("沙盒存档重置完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
