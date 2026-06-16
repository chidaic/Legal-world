"""Location registry loaded from the town map JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_building_prefix(building: str) -> str:
    """Normalize caller-provided building ids to registry prefixes."""
    compact = "".join(ch for ch in str(building or "").strip().lower() if ch.isalnum())
    if compact == "lawfirma":
        return "lawfirmA"
    if compact == "lawfirmb":
        return "lawfirmB"
    if compact == "courta":
        return "courtA"
    if compact == "courtb":
        return "courtB"
    return str(building or "").strip()


def _make_unique_location_id(store: dict[str, "Location"], loc_id: str) -> str:
    """Rename duplicate location ids while keeping numeric suffixes readable."""
    if loc_id not in store:
        return loc_id

    stem = loc_id.rstrip("0123456789")
    suffix = loc_id[len(stem):]
    if suffix.isdigit():
        next_index = int(suffix) + 1
        candidate = f"{stem}{next_index}"
        while candidate in store:
            next_index += 1
            candidate = f"{stem}{next_index}"
    else:
        next_index = 2
        candidate = f"{loc_id}_{next_index}"
        while candidate in store:
            next_index += 1
            candidate = f"{loc_id}_{next_index}"

    logger.warning("Duplicate location id %s detected, renamed to %s", loc_id, candidate)
    return candidate


def _build_waiting_spot_locations(
    building_prefix: str,
    sofas: list[tuple[str, "Location"]],
    front_desk: "Location | None",
    zone: "BuildingZone | None",
) -> list[tuple[str, "Location"]]:
    """Generate indoor standing queue spots for a law firm."""
    if sofas:
        sorted_sofas = sorted(sofas, key=lambda item: (item[1].y, item[1].x, item[0]))
        spots: list[tuple[str, Location]] = []
        for index, (_, sofa_loc) in enumerate(sorted_sofas, start=1):
            if building_prefix == "lawfirmB":
                stand_x = sofa_loc.x - 28.0
                stand_direction = "left"
            else:
                if sofa_loc.direction == "left":
                    stand_x = sofa_loc.x + 24.0
                    stand_direction = "left"
                elif sofa_loc.direction == "right":
                    stand_x = sofa_loc.x - 24.0
                    stand_direction = "right"
                else:
                    stand_x = sofa_loc.x
                    stand_direction = "down"
            stand_y = sofa_loc.y
            spots.append((
                f"{building_prefix}_wait_{index}",
                Location(x=stand_x, y=stand_y, direction=stand_direction),
            ))

        y_steps = [
            second[1].y - first[1].y
            for first, second in zip(sorted_sofas, sorted_sofas[1:])
            if abs(second[1].y - first[1].y) > 1.0
        ]
        step = min(y_steps, key=lambda value: abs(abs(value) - 24.0)) if y_steps else 24.0
        anchor = spots[-1][1]
        for extra_index in range(1, 5):
            candidate_y = anchor.y + step * extra_index
            if zone and not (zone.y <= candidate_y <= zone.y + zone.height):
                candidate_y = anchor.y - step * extra_index
            if zone and not (zone.y <= candidate_y <= zone.y + zone.height):
                continue
            spots.append((
                f"{building_prefix}_wait_{len(spots) + 1}",
                Location(x=anchor.x, y=candidate_y, direction=anchor.direction),
            ))
        return spots

    if not front_desk:
        return []

    base_direction = "left" if building_prefix == "lawfirmB" else "right"
    base_x = front_desk.x + (74.0 if building_prefix == "lawfirmB" else -74.0)
    spots = []
    for index in range(6):
        candidate_y = front_desk.y + 24.0 * index
        if zone and not (zone.y <= candidate_y <= zone.y + zone.height):
            continue
        spots.append((
            f"{building_prefix}_wait_{index + 1}",
            Location(x=base_x, y=candidate_y, direction=base_direction),
        ))
    return spots


@dataclass
class Location:
    x: float
    y: float
    direction: str = ""
    role: str = ""


@dataclass
class BuildingZone:
    x: float
    y: float
    width: float
    height: float


@dataclass
class LocationRegistry:
    birth_locations: dict[str, Location] = field(default_factory=dict)
    building_zones: dict[str, BuildingZone] = field(default_factory=dict)
    lawfirm_chairs: dict[str, Location] = field(default_factory=dict)
    lawfirm_sofas: dict[str, Location] = field(default_factory=dict)
    lawfirm_waiting_spots: dict[str, Location] = field(default_factory=dict)
    lawfirm_interactions: dict[str, Location] = field(default_factory=dict)
    court_chairs: dict[str, Location] = field(default_factory=dict)

    def get(self, loc_id: str) -> Location | None:
        for store in (
            self.birth_locations,
            self.lawfirm_chairs,
            self.lawfirm_sofas,
            self.lawfirm_waiting_spots,
            self.lawfirm_interactions,
            self.court_chairs,
        ):
            if loc_id in store:
                return store[loc_id]
        return None

    def get_available_sofa(self, building: str, occupied: set[str] | None = None) -> str | None:
        occupied = occupied or set()
        prefix = f"{_normalize_building_prefix(building)}_sofa"
        for loc_id in sorted(self.lawfirm_sofas):
            if loc_id.startswith(prefix) and loc_id not in occupied:
                return loc_id
        return None

    def get_available_waiting_spot(
        self,
        building: str,
        occupied: set[str] | None = None,
    ) -> tuple[str | None, Location | None]:
        occupied = occupied or set()
        prefix = f"{_normalize_building_prefix(building)}_wait_"
        for loc_id in sorted(self.lawfirm_waiting_spots):
            if loc_id.startswith(prefix) and loc_id not in occupied:
                return loc_id, self.lawfirm_waiting_spots[loc_id]
        return None, None

    def get_meeting_chair_pair(
        self,
        building: str,
        occupied: set[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Return a free client/lawyer chair pair.

        The map already marks lawyer-side chairs as `direction=right` and client-side
        chairs as `direction=left`. Pairing them by nearest y-coordinate is much more
        robust than the previous hard-coded chair-name mapping and also tolerates
        duplicate chair names in the map source.
        """
        occupied = occupied or set()
        prefix = f"{_normalize_building_prefix(building)}_chair"

        right_chairs: list[tuple[str, Location]] = []
        left_chairs: list[tuple[str, Location]] = []
        for loc_id, loc in sorted(self.lawfirm_chairs.items()):
            if not loc_id.startswith(prefix) or loc_id in occupied:
                continue
            if loc.direction == "right":
                right_chairs.append((loc_id, loc))
            elif loc.direction == "left":
                left_chairs.append((loc_id, loc))

        if not right_chairs or not left_chairs:
            return None, None

        right_chairs.sort(key=lambda item: (item[1].y, item[1].x, item[0]))
        remaining_left = sorted(left_chairs, key=lambda item: (item[1].y, item[1].x, item[0]))
        for lawyer_chair_id, lawyer_loc in right_chairs:
            best_index = min(
                range(len(remaining_left)),
                key=lambda idx: (
                    abs(remaining_left[idx][1].y - lawyer_loc.y),
                    abs(remaining_left[idx][1].x - lawyer_loc.x),
                    remaining_left[idx][0],
                ),
            )
            client_chair_id, _ = remaining_left.pop(best_index)
            return client_chair_id, lawyer_chair_id

        return None, None


def _parse_object_properties(obj: dict[str, Any]) -> dict[str, str]:
    props = {}
    for prop in obj.get("properties", []):
        props[prop["name"]] = prop["value"]
    return props


def _parse_group_layers(
    group: dict[str, Any],
    building_prefix: str,
    registry: LocationRegistry,
) -> None:
    offset_x = group.get("offsetx", 0)
    offset_y = group.get("offsety", 0)

    for layer in group.get("layers", []):
        if layer.get("type", "") != "objectgroup":
            continue

        layer_name = layer.get("name", "")
        total_ox = offset_x + layer.get("offsetx", 0)
        total_oy = offset_y + layer.get("offsety", 0)

        if layer_name in ("chair", "chair_location"):
            for obj in layer.get("objects", []):
                name = obj.get("name", "")
                if not name:
                    continue
                props = _parse_object_properties(obj)
                loc = Location(
                    x=obj["x"] + total_ox,
                    y=obj["y"] + total_oy,
                    direction=props.get("direction", ""),
                    role=name,
                )
                if building_prefix == "lawfirmB" and name.startswith("sofa"):
                    loc.direction = "left"
                loc_id = name if name.startswith(f"{building_prefix}_") else f"{building_prefix}_{name}"

                if building_prefix.startswith("court"):
                    target_store = registry.court_chairs
                elif name.startswith("sofa"):
                    target_store = registry.lawfirm_sofas
                else:
                    target_store = registry.lawfirm_chairs

                target_store[_make_unique_location_id(target_store, loc_id)] = loc

        elif layer_name == "interaction":
            for obj in layer.get("objects", []):
                name = obj.get("name", "").replace(" ", "_")
                if not name:
                    continue
                loc_id = name if name.startswith(f"{building_prefix}_") else f"{building_prefix}_{name}"
                registry.lawfirm_interactions[loc_id] = Location(
                    x=obj["x"] + total_ox,
                    y=obj["y"] + total_oy,
                )


def load_registry_from_map(map_path: str | Path) -> LocationRegistry:
    map_path = Path(map_path)
    with open(map_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    registry = LocationRegistry()

    for layer in data.get("layers", []):
        layer_type = layer.get("type", "")
        layer_name = layer.get("name", "")

        if layer_type == "objectgroup":
            if layer_name == "birth-location":
                for obj in layer.get("objects", []):
                    name = obj.get("name", "")
                    if name:
                        registry.birth_locations[name] = Location(x=obj["x"], y=obj["y"])

            elif layer_name == "location":
                for obj in layer.get("objects", []):
                    name = obj.get("name", "") or obj.get("type", "")
                    if name:
                        registry.building_zones[name] = BuildingZone(
                            x=obj["x"],
                            y=obj["y"],
                            width=obj.get("width", 0),
                            height=obj.get("height", 0),
                        )

            elif layer_name == "interaction":
                for obj in layer.get("objects", []):
                    name = obj.get("name", "").replace(" ", "_")
                    if not name:
                        continue
                    registry.lawfirm_interactions[name] = Location(x=obj["x"], y=obj["y"])

        elif layer_type == "group":
            group_name = layer_name
            prefix = group_name[0].lower() + group_name[1:]
            _parse_group_layers(layer, prefix, registry)

    for building_prefix in sorted(
        key for key in registry.building_zones.keys() if key.startswith("lawfirm")
    ):
        sofas = [
            (loc_id, loc)
            for loc_id, loc in sorted(registry.lawfirm_sofas.items())
            if loc_id.startswith(f"{building_prefix}_")
        ]
        front_desk = registry.lawfirm_interactions.get(f"{building_prefix}_front_desk")
        zone = registry.building_zones.get(building_prefix)
        for loc_id, loc in _build_waiting_spot_locations(building_prefix, sofas, front_desk, zone):
            registry.lawfirm_waiting_spots[loc_id] = loc

    logger.info(
        "LocationRegistry loaded: %d birth, %d zones, %d chairs, %d sofas, %d waits, %d court, %d interactions",
        len(registry.birth_locations),
        len(registry.building_zones),
        len(registry.lawfirm_chairs),
        len(registry.lawfirm_sofas),
        len(registry.lawfirm_waiting_spots),
        len(registry.court_chairs),
        len(registry.lawfirm_interactions),
    )
    return registry
