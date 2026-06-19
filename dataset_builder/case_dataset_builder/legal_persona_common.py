import copy
import os
from typing import Any, Dict, Iterable, List, Optional

from . import utils


PROFILE_FIELD_NAME = "legal_persona_profile"

LEGAL_PERSONA_FIELDS = [
    ("legal_literacy_level", "法律素养水平"),
    ("information_disclosure_willingness", "信息披露意愿"),
    ("emotional_stability", "情绪稳定性"),
    ("narrative_proficiency", "叙事表达能力"),
]

LEGAL_PERSONA_FIELD_KEYS = [field_key for field_key, _ in LEGAL_PERSONA_FIELDS]
LEGAL_PERSONA_FIELD_LABELS_ZH = {
    field_key: field_label_zh for field_key, field_label_zh in LEGAL_PERSONA_FIELDS
}
LEGAL_PERSONA_FIELD_LABELS_EN_TO_ZH = {
    field_key: field_label_zh for field_key, field_label_zh in LEGAL_PERSONA_FIELDS
}
LEGAL_PERSONA_FIELD_LABELS_ZH_TO_EN = {
    field_label_zh: field_key for field_key, field_label_zh in LEGAL_PERSONA_FIELDS
}

LEVEL_HIGH = "high"
LEVEL_MEDIUM = "medium"
LEVEL_LOW = "low"

LEVEL_VALUE_ALIASES = {
    "高": LEVEL_HIGH,
    "中": LEVEL_MEDIUM,
    "低": LEVEL_LOW,
    "high": LEVEL_HIGH,
    "medium": LEVEL_MEDIUM,
    "low": LEVEL_LOW,
    "mid": LEVEL_MEDIUM,
    "middle": LEVEL_MEDIUM,
}

DEFAULT_RANDOM_PROBABILITIES = {
    LEVEL_HIGH: 0.25,
    LEVEL_MEDIUM: 0.5,
    LEVEL_LOW: 0.25,
}

LEGACY_PERSONA_KEYS = {
    PROFILE_FIELD_NAME,
    "big_five_personality",
    "big_5_traits",
    "behavioral_style",
}

PRESET_PERSONAS = {
    "P01": {
        "name": "理想协作型",
        "profile": {
            "legal_literacy_level": LEVEL_HIGH,
            "information_disclosure_willingness": LEVEL_HIGH,
            "emotional_stability": LEVEL_HIGH,
            "narrative_proficiency": LEVEL_HIGH,
        },
        "difficulty": "简单",
        "description": "理解快、配合高、表达清楚，是最理想的当事人。",
        "observation_point": "模型能否高效推进、少走弯路",
    },
    "P02": {
        "name": "理性谨慎型",
        "profile": {
            "legal_literacy_level": LEVEL_HIGH,
            "information_disclosure_willingness": LEVEL_MEDIUM,
            "emotional_stability": LEVEL_HIGH,
            "narrative_proficiency": LEVEL_HIGH,
        },
        "difficulty": "简单-中等",
        "description": "理解能力强，但不会一开始把所有信息都交出来，需要律师逐步问。",
        "observation_point": "模型能否做有层次的追问",
    },
    "P03": {
        "name": "配合但吃力型",
        "profile": {
            "legal_literacy_level": LEVEL_MEDIUM,
            "information_disclosure_willingness": LEVEL_HIGH,
            "emotional_stability": LEVEL_MEDIUM,
            "narrative_proficiency": LEVEL_MEDIUM,
        },
        "difficulty": "中等",
        "description": "愿意配合，但理解和表达都一般，需要律师适度引导。",
        "observation_point": "模型能否稳定做信息整理",
    },
    "P04": {
        "name": "情绪牵引型",
        "profile": {
            "legal_literacy_level": LEVEL_MEDIUM,
            "information_disclosure_willingness": LEVEL_HIGH,
            "emotional_stability": LEVEL_LOW,
            "narrative_proficiency": LEVEL_MEDIUM,
        },
        "difficulty": "中等偏难",
        "description": "愿意讲，也不隐瞒，但情绪容易带偏谈话，需要安抚和拉回主题。",
        "observation_point": "模型能否兼顾安抚与推进",
    },
    "P05": {
        "name": "表达散乱型",
        "profile": {
            "legal_literacy_level": LEVEL_MEDIUM,
            "information_disclosure_willingness": LEVEL_HIGH,
            "emotional_stability": LEVEL_HIGH,
            "narrative_proficiency": LEVEL_LOW,
        },
        "difficulty": "中等",
        "description": "态度好，也愿意说，但叙述不成结构，关键事实混在细节里。",
        "observation_point": "模型能否做结构化提取",
    },
    "P06": {
        "name": "保守试探型",
        "profile": {
            "legal_literacy_level": LEVEL_MEDIUM,
            "information_disclosure_willingness": LEVEL_LOW,
            "emotional_stability": LEVEL_MEDIUM,
            "narrative_proficiency": LEVEL_MEDIUM,
        },
        "difficulty": "中等偏难",
        "description": "不会明显对抗，但会保留信息，先观察律师是否值得信任。",
        "observation_point": "模型能否建立信任、逐步打开信息",
    },
    "P07": {
        "name": "外强内乱型",
        "profile": {
            "legal_literacy_level": LEVEL_MEDIUM,
            "information_disclosure_willingness": LEVEL_MEDIUM,
            "emotional_stability": LEVEL_LOW,
            "narrative_proficiency": LEVEL_HIGH,
        },
        "difficulty": "中等偏难",
        "description": "表达能力不错，但情绪波动大，容易反复确认、质疑判断。",
        "observation_point": "模型能否处理不稳定高表达用户",
    },
    "P08": {
        "name": "朴素直白型",
        "profile": {
            "legal_literacy_level": LEVEL_LOW,
            "information_disclosure_willingness": LEVEL_HIGH,
            "emotional_stability": LEVEL_HIGH,
            "narrative_proficiency": LEVEL_MEDIUM,
        },
        "difficulty": "中等",
        "description": "法律理解弱，但态度坦诚、沟通稳定，属于“好带但基础差”。",
        "observation_point": "模型能否把法律话术翻译成人话",
    },
    "P09": {
        "name": "低素养防御型",
        "profile": {
            "legal_literacy_level": LEVEL_LOW,
            "information_disclosure_willingness": LEVEL_LOW,
            "emotional_stability": LEVEL_MEDIUM,
            "narrative_proficiency": LEVEL_LOW,
        },
        "difficulty": "困难",
        "description": "不懂法、表达乱、还不愿多说，信息缺口较大。",
        "observation_point": "模型能否持续挖掘关键事实",
    },
    "P10": {
        "name": "沉默抗拒型",
        "profile": {
            "legal_literacy_level": LEVEL_MEDIUM,
            "information_disclosure_willingness": LEVEL_LOW,
            "emotional_stability": LEVEL_LOW,
            "narrative_proficiency": LEVEL_MEDIUM,
        },
        "difficulty": "困难",
        "description": "听得懂一些，但不想说太多，对追问容易烦躁或回避。",
        "observation_point": "模型能否在不激化冲突下推进",
    },
    "P11": {
        "name": "对抗失序型",
        "profile": {
            "legal_literacy_level": LEVEL_LOW,
            "information_disclosure_willingness": LEVEL_LOW,
            "emotional_stability": LEVEL_LOW,
            "narrative_proficiency": LEVEL_LOW,
        },
        "difficulty": "困难",
        "description": "最难的一类：不懂法、不愿说、情绪差、表达也乱。",
        "observation_point": "模型极限处理能力",
    },
    "P12": {
        "name": "隐蔽高能型",
        "profile": {
            "legal_literacy_level": LEVEL_HIGH,
            "information_disclosure_willingness": LEVEL_LOW,
            "emotional_stability": LEVEL_HIGH,
            "narrative_proficiency": LEVEL_HIGH,
        },
        "difficulty": "困难",
        "description": "看起来理性、冷静、表达清楚，但会有选择地隐藏不利信息。",
        "observation_point": "模型能否识别“表面配合、实际保留”",
    },
}

PRESET_PERSONA_IDS = list(PRESET_PERSONAS.keys())
PRESET_NAME_TO_ID = {
    preset["name"]: preset_id for preset_id, preset in PRESET_PERSONAS.items()
}


def load_cases(input_path: str) -> List[Dict[str, Any]]:
    data = utils.load_json(input_path)
    if not isinstance(data, list):
        raise ValueError("输入文件必须是案件列表 JSON。")
    if not data:
        raise ValueError("输入文件为空，没有可处理的案件。")
    return data


def save_cases(data: List[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    utils.save_json(data, output_path)


def get_extracted_info(case_data: Dict[str, Any]) -> Dict[str, Any]:
    extracted_info = case_data.get("extracted_info")
    return extracted_info if isinstance(extracted_info, dict) else case_data


def get_party_container(extracted_info: Dict[str, Any], side: str) -> Any:
    party_info = extracted_info.get("party_info", {})
    if isinstance(party_info, dict) and side in party_info:
        return party_info.get(side)
    return extracted_info.get(side)


def get_party_records(party_container: Any) -> List[Dict[str, Any]]:
    if isinstance(party_container, list):
        return [item for item in party_container if isinstance(item, dict)]
    if isinstance(party_container, dict):
        if "name" in party_container:
            return [party_container]
        ordered_keys = sorted(
            key for key, value in party_container.items() if isinstance(value, dict)
        )
        return [party_container[key] for key in ordered_keys]
    return []


def iter_parties(case_data: Dict[str, Any]) -> Iterable[tuple[str, Dict[str, Any]]]:
    extracted_info = get_extracted_info(case_data)
    for side in ("plaintiff", "defendant"):
        party_container = get_party_container(extracted_info, side)
        for party in get_party_records(party_container):
            yield side, party


def strip_legacy_persona_fields(data: Any) -> Any:
    if isinstance(data, dict):
        for key in list(data.keys()):
            if key in LEGACY_PERSONA_KEYS:
                del data[key]
            else:
                strip_legacy_persona_fields(data[key])
    elif isinstance(data, list):
        for item in data:
            strip_legacy_persona_fields(item)
    return data


def prepare_case(case_data: Dict[str, Any]) -> Dict[str, Any]:
    updated_case = copy.deepcopy(case_data)
    strip_legacy_persona_fields(updated_case)
    return updated_case


def normalize_process_range(
    total_count: int,
    start_idx: int,
    process_count: Optional[int],
) -> tuple[int, int]:
    if start_idx < 0 or start_idx >= total_count:
        raise ValueError(f"start_idx 超出范围，应在 0 到 {total_count - 1} 之间。")
    if process_count is None:
        end_idx = total_count
    else:
        end_idx = min(start_idx + process_count, total_count)
    return start_idx, end_idx


def case_identifier(case_data: Dict[str, Any], fallback_idx: int) -> Any:
    return case_data.get("original_id", case_data.get("id", fallback_idx))


def copy_profile(profile: Dict[str, str]) -> Dict[str, str]:
    return {key: value for key, value in profile.items()}


def normalize_level(value: Any) -> str:
    if value is None:
        return LEVEL_MEDIUM

    raw_text = str(value).strip()
    lower_text = raw_text.lower()

    if lower_text in LEVEL_VALUE_ALIASES:
        return LEVEL_VALUE_ALIASES[lower_text]
    if raw_text in LEVEL_VALUE_ALIASES:
        return LEVEL_VALUE_ALIASES[raw_text]

    if "高" in raw_text and "低" not in raw_text:
        return LEVEL_HIGH
    if "低" in raw_text and "高" not in raw_text:
        return LEVEL_LOW
    if "中" in raw_text:
        return LEVEL_MEDIUM

    if "high" in lower_text:
        return LEVEL_HIGH
    if "low" in lower_text:
        return LEVEL_LOW
    if "medium" in lower_text or "mid" in lower_text:
        return LEVEL_MEDIUM

    return LEVEL_MEDIUM


def normalize_profile(raw_profile: Dict[str, Any]) -> Dict[str, str]:
    normalized = {}
    for field_key, field_label_zh in LEGAL_PERSONA_FIELDS:
        value = None
        if isinstance(raw_profile, dict):
            if field_key in raw_profile:
                value = raw_profile[field_key]
            elif field_label_zh in raw_profile:
                value = raw_profile[field_label_zh]
        normalized[field_key] = normalize_level(value)
    return normalized


def validate_random_probabilities(
    probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    merged = dict(DEFAULT_RANDOM_PROBABILITIES)
    if probabilities:
        for key, value in probabilities.items():
            normalized_key = normalize_level(key)
            merged[normalized_key] = float(value)

    total = sum(merged.values())
    if total <= 0:
        raise ValueError("随机概率之和必须大于 0。")

    return {
        LEVEL_HIGH: merged[LEVEL_HIGH] / total,
        LEVEL_MEDIUM: merged[LEVEL_MEDIUM] / total,
        LEVEL_LOW: merged[LEVEL_LOW] / total,
    }


def resolve_preset(preset_ref: str) -> tuple[str, Dict[str, Any]]:
    ref = preset_ref.strip()
    upper_ref = ref.upper()
    if upper_ref in PRESET_PERSONAS:
        return upper_ref, PRESET_PERSONAS[upper_ref]
    if ref in PRESET_NAME_TO_ID:
        preset_id = PRESET_NAME_TO_ID[ref]
        return preset_id, PRESET_PERSONAS[preset_id]
    raise ValueError(f"未找到预设人格：{preset_ref}")


def format_preset_list() -> str:
    lines = []
    for preset_id, preset in PRESET_PERSONAS.items():
        profile = preset["profile"]
        lines.append(
            f"{preset_id} {preset['name']} | "
            f"legal_literacy_level={profile['legal_literacy_level']} "
            f"information_disclosure_willingness={profile['information_disclosure_willingness']} "
            f"emotional_stability={profile['emotional_stability']} "
            f"narrative_proficiency={profile['narrative_proficiency']} | "
            f"难度={preset['difficulty']} | "
            f"{preset['description']}"
        )
    return "\n".join(lines)
