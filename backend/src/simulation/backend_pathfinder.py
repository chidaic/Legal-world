from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


MAP_BOUNDS = {
    "x": -768,
    "y": -1024,
    "width": 2048,
    "height": 1280,
}

COLLISION_LAYER_NAMES = {
    "wall",
    "desk",
    "tree",
    "flower",
    "decoration",
    "chair",
}


class BackendPathfinder:
    """Mirror of the frontend grid pathfinder for travel-time estimation."""

    def __init__(self, map_json_path: str | Path):
        self.map_json_path = Path(map_json_path)
        with open(self.map_json_path, "r", encoding="utf-8") as f:
            map_data = json.load(f)

        self.tile_width = int(map_data.get("tilewidth", 16) or 16)
        self.tile_height = int(map_data.get("tileheight", 16) or 16)
        world_bounds = self._compute_world_bounds(map_data)
        self.grid_offset_x = world_bounds["x"]
        self.grid_offset_y = world_bounds["y"]
        self.grid_width = int(math.ceil(world_bounds["width"] / self.tile_width))
        self.grid_height = int(math.ceil(world_bounds["height"] / self.tile_height))
        self.inflation_radius = 0
        self.cost_radius = 3

        self.collision_grid: list[list[bool]] = [
            [False for _ in range(self.grid_width)]
            for _ in range(self.grid_height)
        ]
        self.inflated_grid: list[list[bool]] = []
        self.cost_grid: list[list[int]] = []

        self._populate_collision_grid(map_data.get("layers", []))
        self._create_inflated_grid()
        logger.info(
            "[BackendPathfinder] Loaded %s, grid=%sx%s, bounds=(%.2f, %.2f, %.2f, %.2f)",
            self.map_json_path,
            self.grid_width,
            self.grid_height,
            self.grid_offset_x,
            self.grid_offset_y,
            self.grid_offset_x + self.grid_width * self.tile_width,
            self.grid_offset_y + self.grid_height * self.tile_height,
        )

    def estimate_travel_duration(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        move_speed_px_per_second: float,
    ) -> float | None:
        path = self.find_path(start_x, start_y, end_x, end_y)
        if path is None:
            return None

        speed = max(float(move_speed_px_per_second or 0.0), 1.0)
        total_distance = 0.0
        prev_x = float(start_x)
        prev_y = float(start_y)
        for point in path:
            next_x = float(point["x"])
            next_y = float(point["y"])
            total_distance += math.hypot(next_x - prev_x, next_y - prev_y)
            prev_x = next_x
            prev_y = next_y
        return total_distance / speed

    def find_path(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> list[dict[str, float]] | None:
        path = self._find_path_internal(start_x, start_y, end_x, end_y, use_inflated=True)
        if path is None:
            path = self._find_path_internal(start_x, start_y, end_x, end_y, use_inflated=False)
        return path

    def _populate_collision_grid(self, layers: list[dict[str, Any]]) -> None:
        for layer, offset_x, offset_y in self._iter_tile_layers(layers, 0.0, 0.0):
            if str(layer.get("name", "")).lower() not in COLLISION_LAYER_NAMES:
                continue
            if layer.get("visible", True) is False:
                continue

            for tile_x, tile_y in self._iter_filled_tiles(layer):
                world_x = offset_x + tile_x * self.tile_width
                world_y = offset_y + tile_y * self.tile_height
                self._mark_collision_tile(world_x, world_y)

    def _compute_world_bounds(self, map_data: dict[str, Any]) -> dict[str, float]:
        min_x = float(MAP_BOUNDS["x"])
        min_y = float(MAP_BOUNDS["y"])
        max_x = min_x + float(MAP_BOUNDS["width"])
        max_y = min_y + float(MAP_BOUNDS["height"])

        def update_bounds(x1: float, y1: float, x2: float, y2: float) -> None:
            nonlocal min_x, min_y, max_x, max_y
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

        def walk_layers(
            layers: list[dict[str, Any]],
            parent_offset_x: float,
            parent_offset_y: float,
        ) -> None:
            for layer in layers:
                layer_type = layer.get("type", "")
                offset_x = (
                    parent_offset_x
                    + float(layer.get("offsetx", 0.0) or 0.0)
                    + float(layer.get("x", 0.0) or 0.0) * self.tile_width
                )
                offset_y = (
                    parent_offset_y
                    + float(layer.get("offsety", 0.0) or 0.0)
                    + float(layer.get("y", 0.0) or 0.0) * self.tile_height
                )

                if layer_type == "group":
                    walk_layers(layer.get("layers", []), offset_x, offset_y)
                    continue

                if layer_type == "tilelayer":
                    if "chunks" in layer:
                        for chunk in layer.get("chunks", []):
                            chunk_x = offset_x + float(chunk.get("x", 0) or 0) * self.tile_width
                            chunk_y = offset_y + float(chunk.get("y", 0) or 0) * self.tile_height
                            chunk_w = float(chunk.get("width", 0) or 0) * self.tile_width
                            chunk_h = float(chunk.get("height", 0) or 0) * self.tile_height
                            update_bounds(chunk_x, chunk_y, chunk_x + chunk_w, chunk_y + chunk_h)
                    else:
                        layer_w = float(layer.get("width", 0) or 0) * self.tile_width
                        layer_h = float(layer.get("height", 0) or 0) * self.tile_height
                        update_bounds(offset_x, offset_y, offset_x + layer_w, offset_y + layer_h)
                    continue

                if layer_type == "objectgroup":
                    for obj in layer.get("objects", []):
                        obj_x = offset_x + float(obj.get("x", 0.0) or 0.0)
                        obj_y = offset_y + float(obj.get("y", 0.0) or 0.0)
                        obj_w = float(obj.get("width", 0.0) or 0.0)
                        obj_h = float(obj.get("height", 0.0) or 0.0)
                        update_bounds(obj_x, obj_y, obj_x + obj_w, obj_y + obj_h)

        walk_layers(map_data.get("layers", []), 0.0, 0.0)

        padding_tiles = 2
        min_x = math.floor(min_x / self.tile_width) * self.tile_width - padding_tiles * self.tile_width
        min_y = math.floor(min_y / self.tile_height) * self.tile_height - padding_tiles * self.tile_height
        max_x = math.ceil(max_x / self.tile_width) * self.tile_width + padding_tiles * self.tile_width
        max_y = math.ceil(max_y / self.tile_height) * self.tile_height + padding_tiles * self.tile_height

        return {
            "x": float(min_x),
            "y": float(min_y),
            "width": float(max_x - min_x),
            "height": float(max_y - min_y),
        }

    def _iter_tile_layers(
        self,
        layers: list[dict[str, Any]],
        parent_offset_x: float,
        parent_offset_y: float,
    ):
        for layer in layers:
            layer_type = layer.get("type", "")
            offset_x = (
                parent_offset_x
                + float(layer.get("offsetx", 0.0) or 0.0)
                + float(layer.get("x", 0.0) or 0.0) * self.tile_width
            )
            offset_y = (
                parent_offset_y
                + float(layer.get("offsety", 0.0) or 0.0)
                + float(layer.get("y", 0.0) or 0.0) * self.tile_height
            )

            if layer_type == "group":
                yield from self._iter_tile_layers(
                    layer.get("layers", []),
                    offset_x,
                    offset_y,
                )
            elif layer_type == "tilelayer":
                yield layer, offset_x, offset_y

    @staticmethod
    def _iter_filled_tiles(layer: dict[str, Any]):
        if "chunks" in layer:
            for chunk in layer.get("chunks", []):
                width = int(chunk.get("width", 0) or 0)
                chunk_x = int(chunk.get("x", 0) or 0)
                chunk_y = int(chunk.get("y", 0) or 0)
                for index, gid in enumerate(chunk.get("data", [])):
                    if not gid:
                        continue
                    yield chunk_x + (index % width), chunk_y + (index // width)
            return

        width = int(layer.get("width", 0) or 0)
        start_x = int(layer.get("startx", layer.get("x", 0)) or 0)
        start_y = int(layer.get("starty", layer.get("y", 0)) or 0)
        for index, gid in enumerate(layer.get("data", [])):
            if not gid:
                continue
            yield start_x + (index % width), start_y + (index // width)

    def _mark_collision_tile(self, world_x: float, world_y: float) -> None:
        gx_start = max(
            0,
            math.floor((world_x - self.grid_offset_x) / self.tile_width) - 1,
        )
        gx_end = min(
            self.grid_width - 1,
            math.floor((world_x + self.tile_width - self.grid_offset_x) / self.tile_width) + 1,
        )
        gy_start = max(
            0,
            math.floor((world_y - self.grid_offset_y) / self.tile_height) - 1,
        )
        gy_end = min(
            self.grid_height - 1,
            math.floor((world_y + self.tile_height - self.grid_offset_y) / self.tile_height) + 1,
        )

        for gy in range(gy_start, gy_end + 1):
            center_y = self.grid_offset_y + gy * self.tile_height + self.tile_height / 2
            if not (world_y <= center_y < world_y + self.tile_height):
                continue
            for gx in range(gx_start, gx_end + 1):
                center_x = self.grid_offset_x + gx * self.tile_width + self.tile_width / 2
                if world_x <= center_x < world_x + self.tile_width:
                    self.collision_grid[gy][gx] = True

    def _create_inflated_grid(self) -> None:
        self.inflated_grid = [row[:] for row in self.collision_grid]
        self.cost_grid = [
            [1 for _ in range(self.grid_width)]
            for _ in range(self.grid_height)
        ]

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not self.collision_grid[y][x]:
                    continue

                for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                    for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            self.inflated_grid[ny][nx] = True

                for dy in range(-self.cost_radius, self.cost_radius + 1):
                    for dx in range(-self.cost_radius, self.cost_radius + 1):
                        nx = x + dx
                        ny = y + dy
                        if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                            continue
                        distance = abs(dx) + abs(dy)
                        additional_cost = max(0, (self.cost_radius - distance + 1) * 2)
                        self.cost_grid[ny][nx] = max(
                            self.cost_grid[ny][nx],
                            1 + additional_cost,
                        )

    def world_to_grid(self, world_x: float, world_y: float) -> dict[str, int]:
        return {
            "x": math.floor((world_x - self.grid_offset_x) / self.tile_width),
            "y": math.floor((world_y - self.grid_offset_y) / self.tile_height),
        }

    def grid_to_world(self, grid_x: int, grid_y: int) -> dict[str, float]:
        return {
            "x": grid_x * self.tile_width + self.grid_offset_x + self.tile_width / 2,
            "y": grid_y * self.tile_height + self.grid_offset_y + self.tile_height / 2,
        }

    def _is_walkable(self, grid_x: int, grid_y: int, use_inflated: bool) -> bool:
        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
            return False
        grid = self.inflated_grid if use_inflated else self.collision_grid
        return not grid[grid_y][grid_x]

    def _find_nearest_walkable(self, grid_x: int, grid_y: int, use_inflated: bool):
        for radius in range(1, 21):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx = grid_x + dx
                    ny = grid_y + dy
                    if self._is_walkable(nx, ny, use_inflated):
                        return {"x": nx, "y": ny}
        return None

    def _get_move_cost(self, grid_x: int, grid_y: int) -> int:
        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
            return 999
        return self.cost_grid[grid_y][grid_x]

    def _find_path_internal(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        use_inflated: bool,
    ) -> list[dict[str, float]] | None:
        start_grid = self.world_to_grid(start_x, start_y)
        end_grid = self.world_to_grid(end_x, end_y)

        if not self._is_walkable(start_grid["x"], start_grid["y"], use_inflated):
            nearest = self._find_nearest_walkable(start_grid["x"], start_grid["y"], use_inflated)
            if nearest is None:
                return None
            start_grid = nearest

        if not self._is_walkable(end_grid["x"], end_grid["y"], use_inflated):
            nearest = self._find_nearest_walkable(end_grid["x"], end_grid["y"], use_inflated)
            if nearest is None:
                return None
            end_grid = nearest

        if start_grid["x"] == end_grid["x"] and start_grid["y"] == end_grid["y"]:
            return [{"x": end_x, "y": end_y}]

        open_list: list[dict[str, Any]] = []
        closed_set: set[tuple[int, int]] = set()

        start_node = {
            "x": start_grid["x"],
            "y": start_grid["y"],
            "g": 0,
            "h": self._heuristic(start_grid["x"], start_grid["y"], end_grid["x"], end_grid["y"]),
            "f": 0,
            "parent": None,
        }
        start_node["f"] = start_node["g"] + start_node["h"]
        open_list.append(start_node)

        directions = (
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
        )
        max_iterations = self.grid_width * self.grid_height
        iterations = 0

        while open_list and iterations < max_iterations:
            iterations += 1
            open_list.sort(key=lambda node: node["f"])
            current = open_list.pop(0)

            if current["x"] == end_grid["x"] and current["y"] == end_grid["y"]:
                return self._reconstruct_path(current, end_x, end_y)

            current_key = (current["x"], current["y"])
            closed_set.add(current_key)

            for dx, dy in directions:
                neighbor_x = current["x"] + dx
                neighbor_y = current["y"] + dy
                neighbor_key = (neighbor_x, neighbor_y)
                if neighbor_key in closed_set:
                    continue
                if not self._is_walkable(neighbor_x, neighbor_y, use_inflated):
                    continue

                move_cost = self._get_move_cost(neighbor_x, neighbor_y) if use_inflated else 1
                g_cost = current["g"] + move_cost
                h_cost = self._heuristic(neighbor_x, neighbor_y, end_grid["x"], end_grid["y"])
                f_cost = g_cost + h_cost

                existing = next(
                    (node for node in open_list if node["x"] == neighbor_x and node["y"] == neighbor_y),
                    None,
                )
                if existing is not None:
                    if g_cost < existing["g"]:
                        existing["g"] = g_cost
                        existing["f"] = f_cost
                        existing["parent"] = current
                    continue

                open_list.append(
                    {
                        "x": neighbor_x,
                        "y": neighbor_y,
                        "g": g_cost,
                        "h": h_cost,
                        "f": f_cost,
                        "parent": current,
                    }
                )

        return None

    @staticmethod
    def _heuristic(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def _reconstruct_path(
        self,
        end_node: dict[str, Any],
        final_x: float,
        final_y: float,
    ) -> list[dict[str, float]]:
        path: list[dict[str, float]] = []
        current = end_node
        while current is not None:
            path.insert(0, self.grid_to_world(current["x"], current["y"]))
            current = current["parent"]

        if path:
            path.pop(0)

        if path:
            path[-1] = {"x": final_x, "y": final_y}
        else:
            path.append({"x": final_x, "y": final_y})
        return path
