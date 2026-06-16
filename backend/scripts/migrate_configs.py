"""配置迁移脚本：将旧版 config.yaml 转换为沙盒新格式。

读取 backend/memory/players/default_player/agents/ 下的旧配置，
转换并写入 backend/sandbox_data/ 对应目录。
旧目录保留不删除。
"""

import sys
from pathlib import Path

import yaml

# 确保项目路径可用
_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))

OLD_BASE = _backend_dir / "memory" / "players" / "default_player" / "agents"
NEW_BASE = _backend_dir / "sandbox_data"


def migrate_client(old_dir: Path, new_id: str, party_role: str = "plaintiff"):
    """迁移当事人配置。"""
    old_config_path = old_dir / "config.yaml"
    if not old_config_path.exists():
        print(f"  跳过（无 config.yaml）: {old_dir}")
        return

    with open(old_config_path, "r", encoding="utf-8") as f:
        old = yaml.safe_load(f) or {}

    char = old.get("character", {})
    new_config = {
        "profile": {
            "name": char.get("name", ""),
            "age": char.get("age"),
            "gender": char.get("gender", ""),
            "occupation": char.get("occupation", ""),
            "personality": char.get("personality", ""),
            "speaking_style": char.get("speaking_style", ""),
        },
        "long_term_memory": [],
        "dataset_path": "",
        "case_id": "",
        "party_role": party_role,
        "case_state": "空闲",
    }

    # 清理 None 值
    new_config["profile"] = {k: v for k, v in new_config["profile"].items() if v is not None}

    new_dir = NEW_BASE / "agents" / "clients" / new_id
    new_dir.mkdir(parents=True, exist_ok=True)
    out_path = new_dir / "config.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] 当事人 {char.get('name', '')} -> {out_path}")


def migrate_lawyer(old_dir: Path, firm_id: str, new_id: str):
    """迁移律师配置。"""
    old_config_path = old_dir / "config.yaml"
    if not old_config_path.exists():
        print(f"  跳过（无 config.yaml）: {old_dir}")
        return

    with open(old_config_path, "r", encoding="utf-8") as f:
        old = yaml.safe_load(f) or {}

    char = old.get("character", {})
    new_config = {
        "profile": {
            "name": char.get("name", ""),
            "lawyer_id": new_id,
            "seniority": "Partner",
        },
        "long_term_memory": [],
        "current_handling_case": None,
        "case_queue": [],
    }

    new_dir = NEW_BASE / "law_firms" / firm_id / "lawyers" / new_id
    new_dir.mkdir(parents=True, exist_ok=True)
    out_path = new_dir / "config.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] 律师 {char.get('name', '')} -> {out_path}")


def migrate_judge(old_dir: Path, court_id: str, new_id: str):
    """迁移法官配置。"""
    old_config_path = old_dir / "config.yaml"
    if not old_config_path.exists():
        print(f"  跳过（无 config.yaml）: {old_dir}")
        return

    with open(old_config_path, "r", encoding="utf-8") as f:
        old = yaml.safe_load(f) or {}

    char = old.get("character", {})
    new_config = {
        "profile": {
            "name": char.get("name", ""),
        },
        "current_handling_case": None,
        "case_queue": [],
    }

    new_dir = NEW_BASE / "court_system" / court_id / "judges" / new_id
    new_dir.mkdir(parents=True, exist_ok=True)
    out_path = new_dir / "config.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] 法官 {char.get('name', '')} -> {out_path}")


def create_lawyer_roster(firm_id: str, firm_name: str, lawyers: list):
    """创建律所名册。"""
    roster = {
        "firm_name": firm_name,
        "firm_id": firm_id,
        "lawyers": lawyers,
    }
    roster_path = NEW_BASE / "law_firms" / firm_id / "lawyer_roster.yaml"
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    with open(roster_path, "w", encoding="utf-8") as f:
        yaml.dump(roster, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] 律所名册 -> {roster_path}")


def main():
    print("=" * 50)
    print("配置迁移：旧格式 → 沙盒新格式")
    print("=" * 50)

    # 迁移当事人
    print("\n[当事人]")
    client_dir = OLD_BASE / "client"
    if client_dir.exists():
        for i, sub in enumerate(sorted(client_dir.iterdir())):
            if sub.is_dir():
                migrate_client(sub, f"client_{i+1:03d}", "plaintiff" if i == 0 else "defendant")
    else:
        print("  旧当事人目录不存在，跳过")

    # 迁移律师
    print("\n[律师]")
    lawyer_dir = OLD_BASE / "lawyer"
    if lawyer_dir.exists():
        lawyers_for_roster = []
        for i, sub in enumerate(sorted(lawyer_dir.iterdir())):
            if sub.is_dir():
                firm = "law_firm_A" if i % 2 == 0 else "law_firm_B"
                lid = f"lawyer_{chr(65 + i % 2)}{i // 2 + 1:02d}"
                migrate_lawyer(sub, firm, lid)
                with open(sub / "config.yaml", "r", encoding="utf-8") as f:
                    old = yaml.safe_load(f) or {}
                char = old.get("character", {})
                lawyers_for_roster.append({
                    "id": lid,
                    "name": char.get("name", ""),
                    "specialty": char.get("specialty_areas", ["综合"]),
                    "seniority": "Partner",
                    "status": "available",
                })
        # 创建名册
        create_lawyer_roster("law_firm_A", "金杜律师事务所",
                             [l for i, l in enumerate(lawyers_for_roster) if i % 2 == 0])
        create_lawyer_roster("law_firm_B", "君合律师事务所",
                             [l for i, l in enumerate(lawyers_for_roster) if i % 2 == 1])
    else:
        print("  旧律师目录不存在，跳过")

    # 迁移法官
    print("\n[法官]")
    judge_dir = OLD_BASE / "judge"
    if judge_dir.exists():
        for i, sub in enumerate(sorted(judge_dir.iterdir())):
            if sub.is_dir():
                court = "basic_court" if i == 0 else "intermediate_court"
                jid = f"judge_{'basic' if i == 0 else 'inter'}_{i+1:02d}"
                migrate_judge(sub, court, jid)
    else:
        print("  旧法官目录不存在，跳过")

    print("\n迁移完成！旧目录保留不删除。")


if __name__ == "__main__":
    main()
