import json
import random
import re
from typing import Any, Dict, Optional

from . import utils
from .legal_persona_common import (
    DEFAULT_RANDOM_PROBABILITIES,
    LEGAL_PERSONA_FIELDS,
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    PRESET_PERSONAS,
    PROFILE_FIELD_NAME,
    case_identifier,
    copy_profile,
    get_extracted_info,
    iter_parties,
    load_cases,
    normalize_process_range,
    normalize_profile,
    prepare_case,
    resolve_preset,
    save_cases,
    validate_random_probabilities,
)


LEGAL_PERSONA_PROMPT_TEMPLATE = """
你是法律当事人人格建模专家。请基于给定案件材料，判断指定当事人在以下四个维度上的等级。

要求：
1. 每个维度只能输出 high、medium、low 三者之一。
2. 必须严格依据下列等级判断标准，不得改写、混用、扩展或发明新的等级。
3. 如果案件材料信息有限，也要基于现有材料做出最可能的判断。
4. 只返回 JSON 对象，不要输出解释、代码块或其他内容。

四个维度的等级判断标准如下：

一、法律素养水平
高：你具备较高的法律素养。你能够理解基本法律概念、程序步骤和律师的专业分析。你在表达诉求时较有结构，能区分事实、判断和目标。你愿意围绕法律问题进行沟通，并能较快理解律师提出的策略含义与风险。
中：你具备中等法律素养。你对法律程序和基本规则有朴素理解，但理解不系统、不稳定。你能大致听懂律师的分析，但对专业判断仍需要进一步解释。你通常能表达核心诉求，但不总能准确组织为法律问题。
低：你法律素养较低。你主要从个人经历和直观公平感出发理解案件，对程序、概念和法律边界缺乏稳定认识。你更容易从生活经验而不是法律框架表达问题，需要律师持续引导、解释和重述，才能逐步理解自己的处境与选择。

二、信息披露意愿
高：你有较高的信息披露意愿。主动、完整地陈述案件事实，包括对自己不利的事实；在律师追问时不会回避；愿意提供所有相关证据材料
中：你具有中等信息披露意愿。陈述主要事实但可能遗漏某些细节（非故意隐瞒，而是认为不重要）；在律师追问下会补充信息；对敏感问题需要建立信任后才愿意回答
低：你信息披露意愿较低。你对信息暴露保持明显谨慎，倾向于保留、弱化或延后披露可能影响自身利益的内容。面对追问时，你更可能回避、模糊、缩短回答，或只给出最低限度的信息。只有在信任明显提升后，你才可能逐步开放。

三、情绪稳定性
高：你的情绪稳定性较高。能冷静客观地陈述事实；面对不利分析能理性接受；沟通简洁有条理；能配合律师的信息收集节奏
中：你的情绪稳定性处于中等水平。你会受到情绪影响，但通常仍能在引导下回到案件本身。你可能重复确认、表达担忧、短暂偏离主题，或对不确定性表现出敏感。只要律师给予一定解释、安抚或结构化引导，你仍能继续配合沟通。
低：你的情绪稳定性较低。对案件结果有强烈预期，不易接受律师的专业判断；可能质疑律师的专业能力；在庭审中可能出现不配合代理律师指导的行为，面对不利信息时，你更难维持持续讨论，也更难稳定吸收律师建议。

四、叙事表达能力
高：你具有较高的叙事组织能力。能按时间线有条理地叙述事件经过；能区分主要事实与次要细节；能准确回忆关键日期、金额等具体信息；表达清晰、逻辑连贯
中：你的叙事组织能力处于中等水平。你能够说明案件的大致经过，但在顺序、重点和细节准确性上并不总是稳定。你有时会遗漏节点、重复信息或在关键处表达不够清楚，但经过追问后通常可以补足主要事实。
低：你的叙事组织能力较低。你在叙述中较难稳定区分主次、顺序和重点，容易出现跳跃、混杂、重复或结构不清的表达。关键信息常常需要律师通过多轮拆解、追问和重组才能提炼出来。

请严格按照下面格式返回：
{{
  "legal_literacy_level": "high|medium|low",
  "information_disclosure_willingness": "high|medium|low",
  "emotional_stability": "high|medium|low",
  "narrative_proficiency": "high|medium|low"
}}

案件材料：
{case_context}

指定当事人：
{role_label}
"""


def deep_get(data: Dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("plea", "claim", "defense", "reasons", "judgment_result"):
            if key in value:
                return to_text(value.get(key))
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "；".join(to_text(item) for item in value if item not in (None, ""))
    return str(value)


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = to_text(value).strip()
        if text:
            return text
    return ""


def build_case_context(case_data: Dict[str, Any], side: str) -> str:
    extracted_info = get_extracted_info(case_data)
    first_instance = extracted_info.get("first_instance", {})

    sections = []

    case_cause = to_text(extracted_info.get("case_cause")).strip()
    if case_cause:
        sections.append(f"案由：{case_cause}")

    case_background = to_text(extracted_info.get("case_background")).strip()
    if case_background:
        sections.append(f"案件背景：{case_background}")

    facts_and_reasons = first_non_empty(
        deep_get(extracted_info, "facts_and_reasons"),
        deep_get(first_instance, "facts_and_reasons"),
    )
    if facts_and_reasons:
        sections.append(f"案件事实与理由：{facts_and_reasons}")

    if side == "plaintiff":
        plaintiff_claim = deep_get(first_instance, "plaintiff_claim", "claim")
        plaintiff_claim_text = to_text(plaintiff_claim).strip()
        if plaintiff_claim_text:
            sections.append(f"原告诉请：{plaintiff_claim_text}")
    else:
        defendant_plea = first_non_empty(
            deep_get(extracted_info, "defendant_plea", "plea"),
            deep_get(first_instance, "defendant_plea", "plea"),
            deep_get(first_instance, "defendant_plea"),
        )
        if defendant_plea:
            sections.append(f"被告答辩：{defendant_plea}")

    court_finding = to_text(deep_get(first_instance, "court_finding")).strip()
    if court_finding:
        sections.append(f"法院查明：{court_finding}")

    return "\n".join(sections)


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def choose_weighted_level(
    rng: random.Random,
    probabilities: Dict[str, float],
) -> str:
    levels = [LEVEL_HIGH, LEVEL_MEDIUM, LEVEL_LOW]
    weights = [probabilities[LEVEL_HIGH], probabilities[LEVEL_MEDIUM], probabilities[LEVEL_LOW]]
    return rng.choices(levels, weights=weights, k=1)[0]


def generate_random_profile(
    rng: random.Random,
    probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    normalized_probabilities = validate_random_probabilities(probabilities)
    return {
        field_key: choose_weighted_level(rng, normalized_probabilities)
        for field_key, _ in LEGAL_PERSONA_FIELDS
    }


def resolve_preset_config(
    preset_id: Optional[str] = None,
    plaintiff_preset: Optional[str] = None,
    defendant_preset: Optional[str] = None,
) -> tuple[str, str]:
    if plaintiff_preset and defendant_preset:
        return resolve_preset(plaintiff_preset)[0], resolve_preset(defendant_preset)[0]

    if preset_id:
        resolved_id, _ = resolve_preset(preset_id)
        return (
            resolve_preset(plaintiff_preset or resolved_id)[0],
            resolve_preset(defendant_preset or resolved_id)[0],
        )

    if not plaintiff_preset or not defendant_preset:
        raise ValueError(
            "preset 模式下请提供 preset_id，或同时提供 plaintiff_preset 和 defendant_preset。"
        )

    return resolve_preset(plaintiff_preset)[0], resolve_preset(defendant_preset)[0]


def extract_llm_persona_for_party(
    case_data: Dict[str, Any],
    party: Dict[str, Any],
    side: str,
    model_name: str,
) -> Dict[str, str]:
    party_name = party.get("name", "未知当事人")
    role_label = f"{'原告' if side == 'plaintiff' else '被告'}{party_name}"
    case_context = build_case_context(case_data, side)

    prompt = LEGAL_PERSONA_PROMPT_TEMPLATE.format(
        case_context=case_context,
        role_label=role_label,
    )
    response = utils.get_completion(
        prompt,
        [],
        1,
        model_name=model_name,
    )[0]
    parsed = extract_json_object(response)
    return normalize_profile(parsed)


def enrich_case_with_llm_persona(
    case_data: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    updated_case = prepare_case(case_data)
    for side, party in iter_parties(updated_case):
        party[PROFILE_FIELD_NAME] = extract_llm_persona_for_party(
            updated_case,
            party,
            side,
            model_name=model_name,
        )
    return updated_case


def enrich_case_with_random_persona(
    case_data: Dict[str, Any],
    rng: random.Random,
    probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    updated_case = prepare_case(case_data)
    normalized_probabilities = validate_random_probabilities(probabilities)
    for _, party in iter_parties(updated_case):
        party[PROFILE_FIELD_NAME] = generate_random_profile(
            rng,
            normalized_probabilities,
        )
    return updated_case


def enrich_case_with_preset_persona(
    case_data: Dict[str, Any],
    plaintiff_preset_id: str,
    defendant_preset_id: str,
) -> Dict[str, Any]:
    updated_case = prepare_case(case_data)
    _, plaintiff_preset = resolve_preset(plaintiff_preset_id)
    _, defendant_preset = resolve_preset(defendant_preset_id)

    for side, party in iter_parties(updated_case):
        preset = plaintiff_preset if side == "plaintiff" else defendant_preset
        party[PROFILE_FIELD_NAME] = copy_profile(preset["profile"])
    return updated_case


def enrich_case_with_mode(
    case_data: Dict[str, Any],
    mode: str = "random",
    llm_model: str = "gpt-5.2",
    rng: Optional[random.Random] = None,
    random_probabilities: Optional[Dict[str, float]] = None,
    preset_id: Optional[str] = None,
    plaintiff_preset: Optional[str] = None,
    defendant_preset: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_mode = mode.strip().lower()

    if normalized_mode == "llm":
        return enrich_case_with_llm_persona(case_data, model_name=llm_model)

    if normalized_mode == "random":
        effective_rng = rng or random.Random()
        return enrich_case_with_random_persona(
            case_data,
            rng=effective_rng,
            probabilities=random_probabilities,
        )

    if normalized_mode == "preset":
        resolved_plaintiff, resolved_defendant = resolve_preset_config(
            preset_id=preset_id,
            plaintiff_preset=plaintiff_preset,
            defendant_preset=defendant_preset,
        )
        return enrich_case_with_preset_persona(
            case_data,
            plaintiff_preset_id=resolved_plaintiff,
            defendant_preset_id=resolved_defendant,
        )

    raise ValueError(f"不支持的 mode: {mode}")


def process_cases(
    input_path: str,
    output_path: str,
    mode: str = "random",
    start_idx: int = 0,
    process_count: Optional[int] = None,
    llm_model: str = "gpt-5.2",
    random_seed: Optional[int] = None,
    random_probabilities: Optional[Dict[str, float]] = None,
    preset_id: Optional[str] = None,
    plaintiff_preset: Optional[str] = None,
    defendant_preset: Optional[str] = None,
) -> str:
    data = load_cases(input_path)
    total_count = len(data)
    start_idx, end_idx = normalize_process_range(total_count, start_idx, process_count)

    rng = random.Random(random_seed)
    normalized_probabilities = validate_random_probabilities(random_probabilities)
    updated_data = [prepare_case(case) for case in data]

    success_count = 0
    failed_count = 0

    print(f"正在加载数据文件: {input_path}")
    print(f"数据加载完成，共 {total_count} 条记录")
    print(f"本次处理范围: 第 {start_idx} 条 到第 {end_idx - 1} 条")
    print(f"生成模式: {mode}")

    if mode == "random":
        print(f"随机概率: {normalized_probabilities}")
        if random_seed is not None:
            print(f"随机种子: {random_seed}")
    elif mode == "preset":
        resolved_plaintiff, resolved_defendant = resolve_preset_config(
            preset_id=preset_id,
            plaintiff_preset=plaintiff_preset,
            defendant_preset=defendant_preset,
        )
        print(
            f"预设人格: 原告={resolved_plaintiff} {PRESET_PERSONAS[resolved_plaintiff]['name']} | "
            f"被告={resolved_defendant} {PRESET_PERSONAS[resolved_defendant]['name']}"
        )
    elif mode == "llm":
        print(f"LLM 模型: {llm_model}")

    for idx in range(start_idx, end_idx):
        case_id = case_identifier(data[idx], idx)
        print(f"\n{'=' * 60}")
        print(f"正在处理第 {idx} 条案件 (ID: {case_id})")

        try:
            updated_data[idx] = enrich_case_with_mode(
                updated_data[idx],
                mode=mode,
                llm_model=llm_model,
                rng=rng,
                random_probabilities=normalized_probabilities,
                preset_id=preset_id,
                plaintiff_preset=plaintiff_preset,
                defendant_preset=defendant_preset,
            )
            success_count += 1
            print(f"[OK] 第 {idx} 条处理成功")
        except Exception as exc:
            failed_count += 1
            print(f"[FAILED] 第 {idx} 条处理失败: {exc}")

    save_cases(updated_data, output_path)

    print(f"\n{'=' * 60}")
    print("处理完成")
    print(f"成功: {success_count} 条")
    print(f"失败: {failed_count} 条")
    print(f"输出文件: {output_path}")
    return output_path

