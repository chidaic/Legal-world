"""虚拟地图引擎 (MapEngine) — 后端占位的前端行为阻流层。

在真实前端上线前，用 asyncio.sleep 模拟人物寻路、动画播放等延时，
防止后端秒级吞吐导致沙盒"闪现"。
"""

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TownAvatarInterface(ABC):
    """前端行动画与物理阻滞约定。"""

    @abstractmethod
    async def move_to_location(self, agent_id: str, dest_loc_id: str) -> bool:
        """让指定人物模型寻路移动到目标点。"""
        ...

    @abstractmethod
    async def play_animation(self, agent_id: str, animation_name: str, duration: float) -> None:
        """播放规定时长的动作（如 talking, typing, reading）。"""
        ...

    @abstractmethod
    async def show_bubble(self, agent_id: str, text: str, duration: float) -> None:
        """头顶弹出气泡文本并在倒计时后消散。"""
        ...


class MockFrontendEngine(TownAvatarInterface):
    """纯后端的占位器 Mock 实现。

    用 asyncio.sleep 模拟寻路和动画延时。
    可通过 speed_factor 调节模拟速度（0 = 无延时，1 = 正常）。
    """

    def __init__(self, speed_factor: float = 1.0):
        self.speed_factor = speed_factor

    async def move_to_location(self, agent_id: str, dest_loc_id: str) -> bool:
        delay = 2.0 * self.speed_factor
        if delay > 0:
            logger.info(f"[Map] {agent_id} → {dest_loc_id} (寻路 {delay:.1f}s)")
            await asyncio.sleep(delay)
        logger.info(f"[Map] {agent_id} 抵达 {dest_loc_id}")
        return True

    async def play_animation(self, agent_id: str, animation_name: str, duration: float) -> None:
        delay = duration * self.speed_factor
        if delay > 0:
            logger.info(f"[Map] {agent_id} 播放动画 '{animation_name}' ({delay:.1f}s)")
            await asyncio.sleep(delay)

    async def show_bubble(self, agent_id: str, text: str, duration: float) -> None:
        delay = duration * self.speed_factor
        logger.info(f"[Map] {agent_id} 气泡: {text[:30]}...")
        if delay > 0:
            await asyncio.sleep(delay)
