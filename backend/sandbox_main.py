"""法律AI小镇沙盒主入口 (sandbox_main.py)。

配置驱动：从 sandbox_data/ 自动发现所有 Agent，
由 FSM + ScenarioOrchestrator 驱动全流程。
"""

import asyncio
import logging
import sys
from pathlib import Path

_backend_dir = Path(__file__).parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from src.core.event_bus import EventBus, EventType
from src.core.file_storage_manager import FileStorageManager
from src.orchestration.case_fsm import CaseStateMachine
from src.orchestration.agent_registry import AgentRegistry
from src.orchestration.scenario_orchestrator import ScenarioOrchestrator
from src.simulation.map_engine import MockFrontendEngine
from src.utils.memory_initializer import initialize_client_memory, initialize_lawyer_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("sandbox")

SANDBOX_DATA_DIR = _backend_dir / "sandbox_data"


async def main():
    """沙盒主循环 — 配置驱动，零硬编码。"""

    # 1. 基础设施
    storage = FileStorageManager(base_dir=SANDBOX_DATA_DIR)
    event_bus = EventBus()
    map_engine = MockFrontendEngine(speed_factor=0.5)
    fsm = CaseStateMachine(
        event_bus,
        storage,
        state_change_notifier=getattr(map_engine, "broadcast_state_change", None),
    )

    # 2. 从 sandbox_data/ 自动发现所有 Agent（传入 map_engine）
    registry = AgentRegistry(SANDBOX_DATA_DIR, event_bus, storage, map_engine)
    registry.discover_all()

    # 3. 场景编排器
    orchestrator = ScenarioOrchestrator(registry, event_bus, fsm, storage, SANDBOX_DATA_DIR)

    # 4. 初始化记忆 & 统计活跃案件（支持断点续传）
    active_cases = []
    resume_cases = []  # 需要恢复的案件（状态不是空闲或等待前台）

    for client in registry.get_agents_by_type("client"):
        if not client.config_path:
            continue
        config = storage.load_agent_config(client.config_path)

        # 只在首次运行时初始化状态（如果没有 case_state 字段）
        if "case_state" not in config:
            storage.update_agent_field(client.config_path, "case_state", "空闲")

        # 初始化当事人结构化长期记忆（从数据集预填充 case_background）
        initialize_client_memory(storage, client.config_path)

        current_state = config.get("case_state", "空闲")
        if config.get("party_role") == "plaintiff":
            if current_state in ["空闲", "等待前台接待"]:
                # 新案件：从头开始
                active_cases.append((client, config))
            elif current_state not in ["已结案"]:
                # 进行中的案件：需要恢复
                resume_cases.append((client, config, current_state))
                logger.info(f"检测到未完成案件: {client.name} (状态: {current_state})")

    # Reset lawyer queues & 初始化律师结构化长期记忆
    for lawyer in registry.get_agents_by_type("lawyer"):
        if not lawyer.config_path:
            continue
        try:
            # 不重置 current_handling_case，保留断点续传信息
            # storage.update_agent_field(lawyer.config_path, "current_handling_case", None)
            storage.update_agent_field(lawyer.config_path, "case_queue", [])
            initialize_lawyer_memory(storage, lawyer.config_path)
        except FileNotFoundError:
            pass

    event_bus.set_expected_cases(len(active_cases) + len(resume_cases))

    # 5. 启动
    logger.info("=" * 60)
    logger.info("法律AI小镇沙盒启动")
    logger.info(f"发现 {len(active_cases)} 个新案件")
    logger.info(f"发现 {len(resume_cases)} 个待恢复案件")
    logger.info("=" * 60)

    # 启动新案件
    for client, config in active_cases:
        case_id = f"case_{config.get('case_id', '1')}"
        # Determine target firm (round-robin or first available)
        firms = list(registry._firms.keys())
        target_firm = firms[0] if firms else "law_firm_A"

        await map_engine.move_to_location(client.agent_id, f"{target_firm}_lobby")
        await event_bus.publish(EventType.PLAINTIFF_ARRIVED, {
            "client_id": client.agent_id,
            "case_id": case_id,
            "target_firm": target_firm,
            "party_role": "plaintiff",
            "client_path": client.config_path,
        })

    # 恢复进行中的案件
    for client, config, current_state in resume_cases:
        case_id = f"case_{config.get('case_id', '1')}"
        firms = list(registry._firms.keys())
        target_firm = firms[0] if firms else "law_firm_A"

        logger.info(f"恢复案件 {case_id}: {client.name} (状态: {current_state})")

        # 根据状态恢复 Agent 位置
        if current_state == "原告咨询中":
            # 恢复咨询场景：当事人和律师都在椅子上
            # 找到对应的律师
            lawyer_agent = None
            for lawyer in registry.get_agents_by_type("lawyer"):
                if lawyer.config_path:
                    lawyer_cfg = storage.load_agent_config(lawyer.config_path)
                    if lawyer_cfg.get("current_handling_case") == case_id:
                        lawyer_agent = lawyer
                        break

            if lawyer_agent:
                logger.info(f"找到律师 {lawyer_agent.name}，恢复咨询场景")

                # 定义椅子位置
                client_chair = f"{target_firm}_front_desk_chair_left"
                lawyer_chair = f"{target_firm}_front_desk_chair_right"

                # Step 1: 当事人出生并移动到椅子
                client_loc = map_engine.registry.get(client_chair)
                if client_loc:
                    await map_engine.spawn_agent(
                        agent_id=client.agent_id,
                        name=client.name,
                        character_name=getattr(client, "character_name", "Adam"),
                        birth_loc_id=client_chair,  # 直接在椅子位置出生
                        role="client",
                    )
                    await map_engine.sit_agent(client.agent_id, client_chair)
                    logger.info(f"当事人 {client.name} 已恢复到椅子位置")

                # Step 2: 律师出生并移动到对面椅子
                lawyer_loc = map_engine.registry.get(lawyer_chair)
                if lawyer_loc:
                    await map_engine.spawn_agent(
                        agent_id=lawyer_agent.agent_id,
                        name=lawyer_agent.name,
                        character_name=getattr(lawyer_agent, "character_name", "Adam"),
                        birth_loc_id=lawyer_chair,  # 直接在椅子位置出生
                        role="lawyer",
                    )
                    await map_engine.sit_agent(lawyer_agent.agent_id, lawyer_chair)
                    logger.info(f"律师 {lawyer_agent.name} 已恢复到椅子位置")

                # Step 3: 继续咨询场景
                await event_bus.publish(EventType.ENTER_PLAINTIFF_CONSULTATION, {
                    "case_id": case_id,
                    "lawyer_id": lawyer_agent.agent_id,
                    "client_id": client.agent_id,
                    "map_prefix": target_firm,
                    "client_path": client.config_path,
                })
            else:
                logger.warning(f"案件 {case_id} 找不到对应律师，无法恢复")
                # 自动结案
                await event_bus.publish(EventType.CASE_CLOSED, {
                    "case_id": case_id,
                    "client_path": client.config_path,
                    "participant_ids": [],
                })
        else:
            logger.warning(f"案件 {case_id} 状态 {current_state} 的断点续传尚未实现")
            # 自动结案
            await event_bus.publish(EventType.CASE_CLOSED, {
                "case_id": case_id,
                "client_path": client.config_path,
                "participant_ids": [],
            })

    # 6. 等待所有案件结案（带防卡死检查）
    await event_bus.spin_until_all_closed(
        timeout=3600,
        storage_manager=storage,
        agent_registry=registry,
        check_interval=30.0,  # 每 30 秒检查一次
    )

    logger.info("=" * 60)
    logger.info("沙盒模拟结束")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
