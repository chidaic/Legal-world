"""数据迁移脚本: 将旧的客户端数据结构迁移到新的案件结构。

旧结构:
    sandbox_data/agents/clients/client_001/config.yaml (party_role: plaintiff, case_id: '9')
    sandbox_data/agents/clients/client_002/config.yaml (party_role: defendant, case_id: '9')

新结构:
    sandbox_data/cases/case_9/plaintiff/config.yaml
    sandbox_data/cases/case_9/defendant/config.yaml
"""

import logging
import shutil
import sys
from pathlib import Path

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.file_storage_manager import FileStorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def migrate_clients(sandbox_data_dir: Path, dry_run: bool = True) -> dict:
    """迁移所有客户端数据到新的案件结构。

    Args:
        sandbox_data_dir: sandbox_data 目录路径
        dry_run: 如果为 True，只打印操作不实际执行

    Returns:
        迁移报告字典
    """
    storage = FileStorageManager(sandbox_data_dir)
    clients_dir = sandbox_data_dir / "agents" / "clients"

    if not clients_dir.exists():
        logger.warning(f"客户端目录不存在: {clients_dir}")
        return {"status": "skipped", "reason": "no_clients_dir"}

    report = {
        "total": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": 0,
        "details": [],
    }

    for client_dir in sorted(clients_dir.iterdir()):
        if not client_dir.is_dir():
            continue

        config_file = client_dir / "config.yaml"
        if not config_file.exists():
            logger.warning(f"跳过 (无配置文件): {client_dir.name}")
            report["skipped"] += 1
            continue

        report["total"] += 1

        try:
            # 读取配置
            config = storage.load_agent_config(str(client_dir))
            case_id = config.get("case_id", "")
            party_role = config.get("party_role", "plaintiff")

            if not case_id:
                logger.warning(f"跳过 {client_dir.name}: 缺少 case_id")
                report["skipped"] += 1
                report["details"].append({
                    "client": client_dir.name,
                    "status": "skipped",
                    "reason": "no_case_id",
                })
                continue

            # 计算新路径
            new_path = storage.get_case_agent_path(case_id, party_role)

            if new_path.exists():
                logger.info(f"跳过 {client_dir.name}: 目标路径已存在 {new_path}")
                report["skipped"] += 1
                report["details"].append({
                    "client": client_dir.name,
                    "case_id": case_id,
                    "party_role": party_role,
                    "status": "skipped",
                    "reason": "target_exists",
                    "target": str(new_path),
                })
                continue

            # 执行迁移
            if dry_run:
                logger.info(f"[DRY RUN] 将迁移: {client_dir} -> {new_path}")
            else:
                new_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_file, new_path / "config.yaml")
                logger.info(f"✓ 迁移成功: {client_dir.name} -> {new_path}")

            report["migrated"] += 1
            report["details"].append({
                "client": client_dir.name,
                "case_id": case_id,
                "party_role": party_role,
                "status": "migrated",
                "source": str(client_dir),
                "target": str(new_path),
            })

        except Exception as e:
            logger.error(f"迁移失败 {client_dir.name}: {e}")
            report["errors"] += 1
            report["details"].append({
                "client": client_dir.name,
                "status": "error",
                "error": str(e),
            })

    return report


def print_report(report: dict) -> None:
    """打印迁移报告。"""
    print("\n" + "=" * 60)
    print("迁移报告")
    print("=" * 60)
    print(f"总计: {report['total']}")
    print(f"已迁移: {report['migrated']}")
    print(f"跳过: {report['skipped']}")
    print(f"错误: {report['errors']}")
    print("=" * 60)

    if report["details"]:
        print("\n详细信息:")
        for detail in report["details"]:
            status = detail["status"]
            client = detail.get("client", "N/A")
            if status == "migrated":
                print(f"  [OK] {client}: {detail.get('source')} -> {detail.get('target')}")
            elif status == "skipped":
                reason = detail.get("reason", "unknown")
                print(f"  [SKIP] {client}: 跳过 ({reason})")
            elif status == "error":
                error = detail.get("error", "unknown")
                print(f"  [ERROR] {client}: 错误 - {error}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="迁移客户端数据到新的案件结构")
    parser.add_argument(
        "--sandbox-dir",
        type=Path,
        default=Path(__file__).parent.parent / "sandbox_data",
        help="sandbox_data 目录路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印操作不实际执行",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际执行迁移 (默认为 dry-run)",
    )

    args = parser.parse_args()

    sandbox_dir = args.sandbox_dir.resolve()
    dry_run = not args.execute

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN 模式 - 不会实际修改文件")
        logger.info("使用 --execute 参数执行实际迁移")
        logger.info("=" * 60)

    logger.info(f"Sandbox 目录: {sandbox_dir}")

    if not sandbox_dir.exists():
        logger.error(f"目录不存在: {sandbox_dir}")
        sys.exit(1)

    # 执行迁移
    report = migrate_clients(sandbox_dir, dry_run=dry_run)

    # 打印报告
    print_report(report)

    # 保存报告
    if not dry_run:
        report_file = sandbox_dir / "migration_report.yaml"
        with open(report_file, "w", encoding="utf-8") as f:
            yaml.dump(report, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
