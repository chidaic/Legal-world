from __future__ import annotations

import argparse
import asyncio
import copy
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from tqdm import tqdm

from . import utils
from .generate_legal_profile import (
    LEGAL_PERSONA_PROMPT_TEMPLATE,
    build_case_context,
    enrich_case_with_random_persona,
    extract_json_object,
    normalize_profile,
)
from .legal_persona_common import (
    DEFAULT_RANDOM_PROBABILITIES,
    PROFILE_FIELD_NAME,
    get_extracted_info,
    iter_parties,
)
from .matched_cases_extract import (
    build_extraction_prompt,
    extract_clean_json as extract_stage_json,
    merge_with_original,
)
from .matched_generate_consultation_questions import (
    process_single_case_async as generate_questions_for_case,
)


MAX_CONCURRENCY_LIMIT = 60


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    final_output_path: Path
    stage1_output_path: Path
    stage2_output_path: Path
    start_id: Optional[int] = 1
    end_id: Optional[int] = None
    process_count: Optional[int] = None
    resume: bool = True
    force_rerun_stage1: bool = False
    force_rerun_stage2: bool = False
    force_rerun_stage3: bool = False
    max_concurrency_extract: int = 20
    max_concurrency_persona: int = 20
    max_concurrency_questions: int = 20
    extract_timeout_seconds: Optional[float] = 600
    persona_timeout_seconds: Optional[float] = 300
    question_timeout_seconds: Optional[float] = 300
    extract_model: Optional[str] = None
    persona_model: Optional[str] = None
    question_model: Optional[str] = None
    persona_mode: str = "llm"
    persona_random_seed: int = 42
    persona_random_probabilities: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RANDOM_PROBABILITIES)
    )
    persona_rebalance_medium_after_llm: bool = True
    persona_medium_rebalance_weights: dict[str, float] = field(
        default_factory=lambda: {"high": 2.0, "medium": 1.0, "low": 2.0}
    )
    save_every: int = 1
    dry_run: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a matched civil case consultation dataset from a public raw JSON file."
    )
    parser.add_argument("input_path", help="Path to matched_cases_*_raw.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for stage outputs. Default: ./outputs/matched_dataset_run_01",
    )
    parser.add_argument(
        "--final-output",
        default=None,
        help="Final JSON path. Default: <output-dir>/light_case_dataset.json",
    )
    parser.add_argument(
        "--final-name",
        default="light_case_dataset.json",
        help="Final file name when --final-output is not set.",
    )
    parser.add_argument("--model", default=None, help="Default model for all LLM stages.")
    parser.add_argument("--api-base-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--extract-model", default=None, help="Model for stage 1 extraction.")
    parser.add_argument("--persona-model", default=None, help="Model for stage 2 persona generation.")
    parser.add_argument("--question-model", default=None, help="Model for stage 3 Q&A generation.")
    parser.add_argument("--persona-mode", choices=["llm", "random"], default="llm")
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--process-count", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=20)
    parser.add_argument("--extract-concurrency", type=int, default=None)
    parser.add_argument("--persona-concurrency", type=int, default=None)
    parser.add_argument("--question-concurrency", type=int, default=None)
    parser.add_argument("--extract-timeout", type=float, default=600)
    parser.add_argument("--persona-timeout", type=float, default=300)
    parser.add_argument("--question-timeout", type=float, default=300)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing stage outputs.")
    parser.add_argument("--force", action="store_true", help="Force rerun all stages.")
    parser.add_argument("--force-stage1", action="store_true")
    parser.add_argument("--force-stage2", action="store_true")
    parser.add_argument("--force-stage3", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and print planned outputs only.")
    return parser


def resolve_config(args: argparse.Namespace) -> PipelineConfig:
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON does not exist: {input_path}")
    if input_path.is_dir() or input_path.suffix.lower() != ".json":
        raise ValueError(f"Input path must be a JSON file: {input_path}")

    if args.final_output:
        final_output_path = Path(args.final_output).expanduser().resolve()
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else final_output_path.parent
        )
    else:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else (Path.cwd() / "outputs" / "matched_dataset_run_01").resolve()
        )
        final_output_path = output_dir / args.final_name

    output_dir.mkdir(parents=True, exist_ok=True)
    utils.configure(model_name=args.model, api_base_url=args.api_base_url)

    return PipelineConfig(
        input_path=input_path,
        output_dir=output_dir,
        final_output_path=final_output_path,
        stage1_output_path=output_dir / "stage1_extracted.json",
        stage2_output_path=output_dir / "stage2_persona.json",
        start_id=args.start_id,
        end_id=args.end_id,
        process_count=args.process_count,
        resume=not args.no_resume,
        force_rerun_stage1=args.force or args.force_stage1,
        force_rerun_stage2=args.force or args.force_stage2,
        force_rerun_stage3=args.force or args.force_stage3,
        max_concurrency_extract=args.extract_concurrency or args.max_concurrency,
        max_concurrency_persona=args.persona_concurrency or args.max_concurrency,
        max_concurrency_questions=args.question_concurrency or args.max_concurrency,
        extract_timeout_seconds=args.extract_timeout,
        persona_timeout_seconds=args.persona_timeout,
        question_timeout_seconds=args.question_timeout,
        extract_model=args.extract_model or args.model or utils.MODEL_NAME,
        persona_model=args.persona_model or args.model or utils.MODEL_NAME,
        question_model=args.question_model or args.model or utils.MODEL_NAME,
        persona_mode=args.persona_mode,
        save_every=max(1, args.save_every),
        dry_run=args.dry_run,
    )


def load_raw_cases(input_path: Path) -> list[dict[str, Any]]:
    data = utils.load_json(input_path)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of matched cases.")
    if not data:
        raise ValueError("Input JSON is empty.")
    return data


def case_original_id(case_data: dict[str, Any], fallback_idx: int = 0) -> Any:
    if "original_id" in case_data:
        return case_data["original_id"]
    if "id" in case_data:
        return case_data["id"]
    return fallback_idx


def case_numeric_id(case_data: dict[str, Any], fallback_idx: int = 0) -> int:
    value = case_original_id(case_data, fallback_idx)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback_idx


def sort_cases_by_original_id(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(cases, key=lambda item: case_numeric_id(item))


def index_cases_by_original_id(cases: Iterable[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {case_original_id(case_data, idx): case_data for idx, case_data in enumerate(cases)}


def select_raw_cases_by_id_range(
    raw_cases: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    selected = []
    for idx, case_data in enumerate(raw_cases):
        numeric_id = case_numeric_id(case_data, idx + 1)
        if config.start_id is not None and numeric_id < config.start_id:
            continue
        if config.end_id is not None and numeric_id > config.end_id:
            continue
        selected.append(case_data)

    if config.process_count is not None:
        selected = selected[: config.process_count]
    if not selected:
        raise ValueError("No cases selected by start_id/end_id/process_count.")
    return selected


def load_existing_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = utils.load_json(path)
    return data if isinstance(data, list) else []


def save_cases(data: list[dict[str, Any]], path: Path) -> None:
    utils.save_json(sort_cases_by_original_id(data), path)


async def run_with_timeout(awaitable: Any, timeout_seconds: Optional[float]) -> Any:
    if timeout_seconds in (None, 0):
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=timeout_seconds)


def get_concurrency(value: int) -> int:
    return min(MAX_CONCURRENCY_LIMIT, max(1, int(value)))


def stage1_is_complete(case_data: dict[str, Any]) -> bool:
    extracted_info = case_data.get("extracted_info")
    return isinstance(extracted_info, dict) and "error" not in extracted_info


def stage2_is_complete(case_data: dict[str, Any]) -> bool:
    extracted_info = get_extracted_info(case_data)
    if not isinstance(extracted_info, dict) or "error" in extracted_info:
        return False

    has_party = False
    for _, party in iter_parties(case_data):
        has_party = True
        profile = party.get(PROFILE_FIELD_NAME)
        if not isinstance(profile, dict):
            return False
        normalized = normalize_profile(profile)
        if not all(normalized.get(field) for field in normalized):
            return False
    return has_party


def stage3_is_complete(case_data: dict[str, Any]) -> bool:
    if not stage2_is_complete(case_data):
        return False

    has_party = False
    for _, party in iter_parties(case_data):
        has_party = True
        questions = party.get("questions")
        if not isinstance(questions, list) or not questions:
            return False
    return has_party


def parse_year(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None


def estimate_party_age(case_data: dict[str, Any], party: dict[str, Any]) -> Optional[int]:
    birth_year = parse_year(str(party.get("birth_date", "")))
    if birth_year is None:
        return None

    extracted_info = get_extracted_info(case_data)
    first_instance = extracted_info.get("first_instance", {})
    second_instance = extracted_info.get("second_instance", {})
    judgment_year = (
        parse_year(str(first_instance.get("judgment_date", "")))
        or parse_year(str(second_instance.get("judgment_date", "")))
    )
    if judgment_year is None:
        return None

    age = judgment_year - birth_year
    return age if 0 < age < 120 else None


def build_party_specific_case_context(
    case_data: dict[str, Any],
    side: str,
    party: dict[str, Any],
) -> str:
    base_context = build_case_context(case_data, side)
    extracted_info = get_extracted_info(case_data)

    party_lines = [
        f"指定当事人姓名：{party.get('name', '')}",
        f"指定当事人类型：{party.get('type', '')}",
    ]
    if party.get("gender"):
        party_lines.append(f"指定当事人性别：{party.get('gender')}")
    if party.get("ethnicity"):
        party_lines.append(f"指定当事人民族：{party.get('ethnicity')}")
    if party.get("birth_date"):
        party_lines.append(f"指定当事人出生日期：{party.get('birth_date')}")

    estimated_age = estimate_party_age(case_data, party)
    if estimated_age is not None:
        party_lines.append(f"指定当事人在案件裁判年份的大致年龄：{estimated_age}岁")

    if party.get("address"):
        party_lines.append(f"指定当事人住址/所在地：{party.get('address')}")
    if party.get("representative"):
        party_lines.append(f"指定当事人法定代表人/经营者：{party.get('representative')}")

    role_desc = "一审原告" if side == "plaintiff" else "一审被告"
    appellant = extracted_info.get("appellant")
    if appellant == "原告" and side == "plaintiff":
        role_desc += "；二审上诉人"
    elif appellant == "被告" and side == "defendant":
        role_desc += "；二审上诉人"
    elif appellant == "原告" and side == "defendant":
        role_desc += "；二审被上诉人"
    elif appellant == "被告" and side == "plaintiff":
        role_desc += "；二审被上诉人"
    party_lines.append(f"指定当事人在案件中的参与身份：{role_desc}")

    return "\n".join(party_lines + ["", base_context])


def build_persona_random_rng(case_data: dict[str, Any], fallback_idx: int, config: PipelineConfig) -> random.Random:
    return random.Random(f"{config.persona_random_seed}:{case_original_id(case_data, fallback_idx)}")


def choose_rebalanced_level(rng: random.Random, weights: dict[str, float]) -> str:
    levels = ["high", "medium", "low"]
    choice_weights = [float(weights[level]) for level in levels]
    return rng.choices(levels, weights=choice_weights, k=1)[0]


def rebalance_medium_levels(
    profile: dict[str, str],
    rng: random.Random,
    weights: dict[str, float],
) -> dict[str, str]:
    rebalanced_profile = dict(profile)
    for field_key, level in rebalanced_profile.items():
        if level == "medium":
            rebalanced_profile[field_key] = choose_rebalanced_level(rng, weights)
    return rebalanced_profile


async def extract_single_case_async(
    raw_case: dict[str, Any],
    case_index: int,
    config: PipelineConfig,
) -> dict[str, Any]:
    original_id = case_original_id(raw_case, case_index + 1)
    prompt = build_extraction_prompt(raw_case)

    try:
        response, _ = await utils.aget_completion(
            prompt,
            history=[],
            flag=1,
            model_name=config.extract_model,
        )
        extracted_info = extract_stage_json(response)
        merged_info = merge_with_original(extracted_info, raw_case)
        return {"original_id": original_id, "extracted_info": merged_info}
    except Exception as exc:
        return {"original_id": original_id, "extracted_info": {"error": str(exc)}}


async def extract_persona_for_party_async(
    case_data: dict[str, Any],
    party: dict[str, Any],
    side: str,
    config: PipelineConfig,
) -> dict[str, str]:
    party_name = party.get("name", "未知当事人")
    role_label = f"{'原告' if side == 'plaintiff' else '被告'}{party_name}"
    case_context = build_party_specific_case_context(case_data, side, party)
    prompt = LEGAL_PERSONA_PROMPT_TEMPLATE.format(
        case_context=case_context,
        role_label=role_label,
    )
    response, _ = await utils.aget_completion(
        prompt,
        history=[],
        flag=1,
        model_name=config.persona_model,
    )
    parsed = extract_json_object(response)
    return normalize_profile(parsed)


async def enrich_case_with_persona_async(
    case_data: dict[str, Any],
    case_index: int,
    config: PipelineConfig,
) -> dict[str, Any]:
    persona_rng = build_persona_random_rng(case_data, case_index, config)
    persona_mode = config.persona_mode.strip().lower()

    if persona_mode == "random":
        updated_case = enrich_case_with_random_persona(
            case_data,
            rng=persona_rng,
            probabilities=config.persona_random_probabilities,
        )
        updated_case.pop("stage2_error", None)
        return updated_case

    updated_case = copy.deepcopy(case_data)
    updated_case.pop("stage2_error", None)
    extracted_info = get_extracted_info(updated_case)
    if not isinstance(extracted_info, dict) or "error" in extracted_info:
        return updated_case

    for side, party in iter_parties(updated_case):
        profile = await extract_persona_for_party_async(updated_case, party, side, config)
        if config.persona_rebalance_medium_after_llm:
            profile = rebalance_medium_levels(
                profile,
                rng=persona_rng,
                weights=config.persona_medium_rebalance_weights,
            )
        party[PROFILE_FIELD_NAME] = profile

    return updated_case


async def generate_questions_for_case_async(
    case_data: dict[str, Any],
    case_index: int,
    config: PipelineConfig,
) -> dict[str, Any]:
    updated_case = copy.deepcopy(case_data)
    updated_case.pop("stage3_error", None)
    outcome = await generate_questions_for_case(
        updated_case,
        case_index,
        model_name=config.question_model,
    )
    if outcome.get("status") != "success":
        updated_case["stage3_error"] = outcome.get("reason", "question generation failed")
    return updated_case


async def run_stage1_extract_async(
    raw_cases: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    print("\nStage 1: extracting structured case facts")
    existing_map: dict[Any, dict[str, Any]] = {}
    if config.resume and config.stage1_output_path.exists() and not config.force_rerun_stage1:
        existing_map = index_cases_by_original_id(load_existing_cases(config.stage1_output_path))

    result_map: dict[Any, dict[str, Any]] = {}
    pending_cases = []
    for idx, raw_case in enumerate(raw_cases):
        original_id = case_original_id(raw_case, idx + 1)
        existing_case = existing_map.get(original_id)
        if existing_case and stage1_is_complete(existing_case):
            result_map[original_id] = existing_case
            continue
        if existing_case:
            result_map[original_id] = existing_case
        pending_cases.append((idx, raw_case))

    completed = sum(1 for item in result_map.values() if stage1_is_complete(item))
    print(f"Total: {len(raw_cases)} | completed: {completed} | pending: {len(pending_cases)}")
    if not pending_cases:
        return sort_cases_by_original_id(result_map.values())

    semaphore = asyncio.Semaphore(get_concurrency(config.max_concurrency_extract))
    save_every = max(1, config.save_every)
    processed_since_save = 0
    progress_bar = tqdm(total=completed + len(pending_cases), initial=completed, desc="stage1", unit="case")

    async def worker(idx: int, raw_case: dict[str, Any]) -> dict[str, Any]:
        try:
            async with semaphore:
                return await run_with_timeout(
                    extract_single_case_async(raw_case, idx, config),
                    config.extract_timeout_seconds,
                )
        except asyncio.TimeoutError:
            original_id = case_original_id(raw_case, idx + 1)
            return {
                "original_id": original_id,
                "extracted_info": {"error": f"stage1 timed out after {config.extract_timeout_seconds}s"},
            }

    try:
        for task in asyncio.as_completed([asyncio.create_task(worker(idx, case)) for idx, case in pending_cases]):
            result = await task
            result_map[case_original_id(result)] = result
            processed_since_save += 1
            if processed_since_save >= save_every:
                save_cases(list(result_map.values()), config.stage1_output_path)
                processed_since_save = 0
            progress_bar.update(1)
    finally:
        progress_bar.close()

    results = sort_cases_by_original_id(result_map.values())
    save_cases(results, config.stage1_output_path)
    return results


async def run_stage2_persona_async(
    stage1_cases: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    print("\nStage 2: generating legal persona profiles")
    existing_map: dict[Any, dict[str, Any]] = {}
    if config.resume and config.stage2_output_path.exists() and not config.force_rerun_stage2:
        existing_map = index_cases_by_original_id(load_existing_cases(config.stage2_output_path))

    result_map: dict[Any, dict[str, Any]] = {}
    pending_cases = []
    for idx, case_data in enumerate(stage1_cases):
        original_id = case_original_id(case_data, idx)
        existing_case = existing_map.get(original_id)
        if existing_case and stage2_is_complete(existing_case):
            result_map[original_id] = existing_case
            continue
        result_map[original_id] = copy.deepcopy(existing_case or case_data)
        if stage1_is_complete(case_data):
            pending_cases.append((idx, case_data))

    completed = sum(1 for item in result_map.values() if stage2_is_complete(item))
    print(f"Total: {len(stage1_cases)} | completed: {completed} | pending: {len(pending_cases)}")
    if not pending_cases:
        return sort_cases_by_original_id(result_map.values())

    semaphore = asyncio.Semaphore(get_concurrency(config.max_concurrency_persona))
    save_every = max(1, config.save_every)
    processed_since_save = 0
    progress_bar = tqdm(total=completed + len(pending_cases), initial=completed, desc="stage2", unit="case")

    async def worker(idx: int, case_data: dict[str, Any]) -> dict[str, Any]:
        try:
            async with semaphore:
                task = enrich_case_with_persona_async(case_data, idx, config)
                timeout = None if config.persona_mode == "random" else config.persona_timeout_seconds
                return await run_with_timeout(task, timeout)
        except asyncio.TimeoutError:
            timed_out_case = copy.deepcopy(case_data)
            timed_out_case["stage2_error"] = f"stage2 timed out after {config.persona_timeout_seconds}s"
            return timed_out_case
        except Exception as exc:
            failed_case = copy.deepcopy(case_data)
            failed_case["stage2_error"] = str(exc)
            return failed_case

    try:
        for task in asyncio.as_completed([asyncio.create_task(worker(idx, case)) for idx, case in pending_cases]):
            result = await task
            result_map[case_original_id(result)] = result
            processed_since_save += 1
            if processed_since_save >= save_every:
                save_cases(list(result_map.values()), config.stage2_output_path)
                processed_since_save = 0
            progress_bar.update(1)
    finally:
        progress_bar.close()

    results = sort_cases_by_original_id(result_map.values())
    save_cases(results, config.stage2_output_path)
    return results


async def run_stage3_questions_async(
    stage2_cases: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    print("\nStage 3: generating consultation questions")
    existing_map: dict[Any, dict[str, Any]] = {}
    if config.resume and config.final_output_path.exists() and not config.force_rerun_stage3:
        existing_map = index_cases_by_original_id(load_existing_cases(config.final_output_path))

    result_map: dict[Any, dict[str, Any]] = {}
    pending_cases = []
    for idx, case_data in enumerate(stage2_cases):
        original_id = case_original_id(case_data, idx)
        existing_case = existing_map.get(original_id)
        if existing_case and stage3_is_complete(existing_case):
            result_map[original_id] = existing_case
            continue
        result_map[original_id] = copy.deepcopy(existing_case or case_data)
        if stage2_is_complete(case_data):
            pending_cases.append((idx, case_data))

    completed = sum(1 for item in result_map.values() if stage3_is_complete(item))
    print(f"Total: {len(stage2_cases)} | completed: {completed} | pending: {len(pending_cases)}")
    if not pending_cases:
        return sort_cases_by_original_id(result_map.values())

    semaphore = asyncio.Semaphore(get_concurrency(config.max_concurrency_questions))
    save_every = max(1, config.save_every)
    processed_since_save = 0
    progress_bar = tqdm(total=completed + len(pending_cases), initial=completed, desc="stage3", unit="case")

    async def worker(idx: int, case_data: dict[str, Any]) -> dict[str, Any]:
        try:
            async with semaphore:
                return await run_with_timeout(
                    generate_questions_for_case_async(case_data, idx, config),
                    config.question_timeout_seconds,
                )
        except asyncio.TimeoutError:
            timed_out_case = copy.deepcopy(case_data)
            timed_out_case["stage3_error"] = f"stage3 timed out after {config.question_timeout_seconds}s"
            return timed_out_case
        except Exception as exc:
            failed_case = copy.deepcopy(case_data)
            failed_case["stage3_error"] = str(exc)
            return failed_case

    try:
        for task in asyncio.as_completed([asyncio.create_task(worker(idx, case)) for idx, case in pending_cases]):
            result = await task
            result_map[case_original_id(result)] = result
            processed_since_save += 1
            if processed_since_save >= save_every:
                save_cases(list(result_map.values()), config.final_output_path)
                processed_since_save = 0
            progress_bar.update(1)
    finally:
        progress_bar.close()

    results = sort_cases_by_original_id(result_map.values())
    save_cases(results, config.final_output_path)
    return results


def write_run_summary(
    config: PipelineConfig,
    selected_raw_cases: list[dict[str, Any]],
    final_count: Optional[int] = None,
) -> None:
    lines = [
        "Matched case dataset pipeline",
        "",
        f"input_file: {config.input_path}",
        f"output_dir: {config.output_dir}",
        f"selected_cases: {len(selected_raw_cases)}",
        f"start_id: {config.start_id}",
        f"end_id: {config.end_id}",
        f"process_count: {config.process_count}",
        "",
        "outputs:",
        f"- stage1: {config.stage1_output_path}",
        f"- stage2: {config.stage2_output_path}",
        f"- final: {config.final_output_path}",
        "",
        "models:",
        f"- extract_model: {config.extract_model}",
        f"- persona_model: {config.persona_model}",
        f"- question_model: {config.question_model}",
        f"- persona_mode: {config.persona_mode}",
        "",
        "concurrency:",
        f"- extract: {config.max_concurrency_extract}",
        f"- persona: {config.max_concurrency_persona}",
        f"- questions: {config.max_concurrency_questions}",
    ]
    if final_count is not None:
        lines += ["", f"final_cases: {final_count}"]
    (config.output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_pipeline_async(config: PipelineConfig) -> None:
    raw_cases = load_raw_cases(config.input_path)
    selected_raw_cases = select_raw_cases_by_id_range(raw_cases, config)
    write_run_summary(config, selected_raw_cases)

    first_id = case_original_id(selected_raw_cases[0])
    last_id = case_original_id(selected_raw_cases[-1])
    print(f"Input: {config.input_path}")
    print(f"Output dir: {config.output_dir}")
    print(f"Selected cases: {len(selected_raw_cases)} (id {first_id} - {last_id})")

    if config.dry_run:
        print("Dry run complete. No LLM calls were made.")
        return

    start_time = time.time()
    stage1_cases = await run_stage1_extract_async(selected_raw_cases, config)
    stage2_cases = await run_stage2_persona_async(stage1_cases, config)
    final_cases = await run_stage3_questions_async(stage2_cases, config)
    write_run_summary(config, selected_raw_cases, final_count=len(final_cases))

    elapsed = time.time() - start_time
    print("\nPipeline complete")
    print(f"Final output: {config.final_output_path}")
    print(f"Final cases: {len(final_cases)}")
    print(f"Elapsed: {elapsed:.1f}s")


def main() -> None:
    args = build_arg_parser().parse_args()
    config = resolve_config(args)
    asyncio.run(run_pipeline_async(config))


if __name__ == "__main__":
    main()

