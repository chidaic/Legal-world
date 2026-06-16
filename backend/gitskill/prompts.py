from __future__ import annotations

from typing import Any

from src.prompts.prompt_assembler import PromptAssembler
from src.utils.prompt_profile import is_test_prompt_profile


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_structured_plain(payload: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(payload, dict):
        if not payload:
            return f"{prefix}(empty object)"
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_render_structured_plain(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {_stringify_scalar(value)}")
        return "\n".join(lines)
    if isinstance(payload, list):
        if not payload:
            return f"{prefix}(empty list)"
        lines: list[str] = []
        for index, item in enumerate(payload, start=1):
            if isinstance(item, dict):
                stage_name = item.get("stage")
                summary_text = item.get("summary")
                if stage_name or summary_text:
                    label = f"{prefix}- item_{index}"
                    if stage_name:
                        label += f" [{stage_name}]"
                    lines.append(label)
                    if summary_text:
                        lines.append(f"{prefix}  摘要: {_stringify_scalar(summary_text)}")
                    for key, value in item.items():
                        if key in {"stage", "summary"}:
                            continue
                        if isinstance(value, (dict, list)):
                            lines.append(f"{prefix}  {key}:")
                            lines.append(_render_structured_plain(value, indent + 2))
                        else:
                            lines.append(f"{prefix}  {key}: {_stringify_scalar(value)}")
                    continue
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}- item_{index}:")
                lines.append(_render_structured_plain(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_stringify_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{_stringify_scalar(payload)}"


def _render_stage_summaries(stage_summaries: list[Any]) -> str:
    if not stage_summaries:
        return "（无阶段摘要）"
    return _render_structured_plain(stage_summaries)


def _select_prompt_variant(prod_prompt: str, test_prompt: str) -> str:
    return test_prompt if is_test_prompt_profile() else prod_prompt


def build_stage_one_system_prompt(profile: dict[str, Any], long_term_memory: dict[str, Any]) -> str:
    scenario_prompt = _select_prompt_variant(
        """你现在处于“案件结束后的首轮反思阶段”。
你的唯一目标，是基于评测结果和阶段摘要，产出这个案件当前版本的完整 `reflection.md`。

要求：
1. 你不是在继续办案，而是在复盘。
2. 这一阶段不能调用任何工具，也不要假设自己已经读过原始文书或完整对话。
3. 输出必须是完整 Markdown，不要包在代码块里。
4. 输出内容要覆盖整个 `reflection.md`，不要只输出增量片段。
5. 需要明确写出：
   - 这次案件暴露出的有效方法
   - 这次案件暴露出的错误或风险
   - 当前还缺哪些材料，导致暂时不能下最终 Skill 结论
6. 即使当前还不能沉淀 Skill，也要把有价值的反思写清楚，不能留空。

建议使用如下标题组织全文：
# 案件反思
## 案件信息
## 评测关键信号
## 第一轮反思
## 候选方法与警示
## 需要补读的材料
## 当前暂定结论
""",
        """你现在处于“案件结束后的首轮反思阶段”。
目标：基于评测结果和阶段摘要，直接输出完整 `reflection.md`。

要求：
1. 只做复盘，不继续办案。
2. 不调用工具。
3. 输出完整 Markdown，不要代码块。
4. 至少写清：有效方法、错误风险、还缺什么材料、当前暂定结论。
""",
    )
    return PromptAssembler.build(
        profile=profile,
        long_term_memory=long_term_memory,
        scenario_prompt=scenario_prompt,
    )


def build_stage_one_instruction(
    case_metadata: dict[str, Any],
    eval_summary: dict[str, Any],
    eval_full_report: dict[str, Any],
    stage_summaries: list[Any],
) -> str:
    prefix = _select_prompt_variant(
        "请基于以下输入，直接产出这个案件当前版本的完整 reflection.md。\n\n",
        "请直接输出当前完整 reflection.md。\n\n",
    )
    return (
        prefix +
        "【案件元信息】\n"
        f"{_render_structured_plain(case_metadata)}\n\n"
        "【评测汇总结果】\n"
        f"{_render_structured_plain(eval_summary)}\n\n"
        "【评测完整报告】\n"
        "其中已经包含各阶段评分、评分理由、参考答案 GT 以及待评内容。请优先基于这里做反思。\n"
        f"{_render_structured_plain(eval_full_report)}\n\n"
        "【各阶段摘要】\n"
        f"{_render_stage_summaries(stage_summaries)}"
    )


def build_stage_two_system_prompt(profile: dict[str, Any], long_term_memory: dict[str, Any]) -> str:
    scenario_prompt = _select_prompt_variant(
        """你现在处于“补读与补反思阶段”。
你的目标，是在现有 `reflection.md` 基础上决定是否需要补读材料，并在补读后输出更新后的完整 `reflection.md`。

你现在可以使用：
- `read_case_artifact`：按需读取当前案件的阶段结果、文书全文或对话片段

要求：
1. 不要无脑全读，只补读真正影响结论的材料。
2. 如果调用了工具，必须把你读了什么、为什么读、读到了什么，写进“补读记录”。
3. 输出仍然必须是完整 Markdown，不要包在代码块里。
4. 输出内容要覆盖整个 `reflection.md`，不能只输出新增段落。
5. 这一阶段只做案件复盘和补读，不做 Skill 的新增、更新、查重或合并判断。
6. `artifact catalog` 里除了路径，还会告诉你每个文件大致是什么内容，以及 JSON 文件有哪些顶层 key。你应先根据这些说明决定读哪个文件。
7. 对结构化 artifact，优先用 `field` 只读取真正需要的部分，而不是默认读整个文件。

建议补充或更新如下标题：
## 补读记录
## 第二轮反思
## 暂定方法与风险
""",
        """你现在处于“补读与补反思阶段”。
目标：只在必要时补读材料，然后输出更新后的完整 `reflection.md`。

要求：
1. 不要无脑全读。
2. 如果补读，要记录读了什么、为什么读、读到什么。
3. 输出完整 Markdown，不要代码块。
4. 本阶段不做 Skill 写入判断。
""",
    )
    return PromptAssembler.build(
        profile=profile,
        long_term_memory=long_term_memory,
        scenario_prompt=scenario_prompt,
    )


def build_stage_two_instruction(
    reflection_markdown: str,
    artifact_catalog_markdown: str,
) -> str:
    prefix = _select_prompt_variant(
        "请在当前 reflection.md 基础上继续复盘。只有当确实需要时，才使用工具补读。\n\n",
        "请在当前 reflection.md 基础上继续复盘；只有必要时才补读。\n\n",
    )
    return (
        prefix +
        "【当前 reflection.md】\n"
        f"{reflection_markdown}\n\n"
        "【artifact catalog】\n"
        f"{artifact_catalog_markdown}"
    )


def build_stage_three_system_prompt(profile: dict[str, Any], long_term_memory: dict[str, Any]) -> str:
    scenario_prompt = _select_prompt_variant(
        """你现在处于“Skill 判断与写入阶段”。
你的目标，是基于当前 `reflection.md` 做最终判断：这次案件应当沉淀 0 个、1 个，还是多个 Skill。

你现在可以使用：
- `load_skill`：读取当前 main/private 中已有的 Skill
- `upsert_skill_file`：每次调用写入一个私有 Skill 文件夹下的 `SKILL.md`

要求：
1. 先判断这次案件真正沉淀下来的可复用方法，到底是 0 个、1 个，还是多个；不要机械拆碎。
2. 只有当几个方法确实彼此独立、可单独复用时，才拆成多个 Skill。
3. 如果你决定写入一个或多个 Skill，必须先调用 `load_skill` 读取主干里的 `skill-creator`，把它当作 `SKILL.md` 写作参考模板。
4. 这一阶段才允许做 Skill 查重、判断是 `new` 还是 `update`，以及最终写入。
5. 如果最终结论是 `none`，不要调用 `upsert_skill_file`。
6. 输出内容仍然要覆盖整个 `reflection.md`，并在文末加入最终判定，以及你实际写入的每个 Skill 文件夹路径与 action。

写 Skill 时必须使用完整 `SKILL.md`。内容应当可跨案复用，应简洁清晰，不要写本案专属细节。
""",
        """你现在处于“Skill 判断与写入阶段”。
目标：基于当前 `reflection.md` 判断沉淀 0/1/多个 Skill，并在需要时写入。

要求：
1. 先判断是否真的值得沉淀。
2. 写入前先读取 `skill-creator`。
3. 若结论是 none，不要写文件。
4. 输出完整 `reflection.md`，文末写最终判定和写入路径。
""",
    )
    return PromptAssembler.build(
        profile=profile,
        long_term_memory=long_term_memory,
        scenario_prompt=scenario_prompt,
    )


def build_stage_three_instruction(
    reflection_markdown: str,
    case_cause_dir: str,
) -> str:
    prefix = _select_prompt_variant(
        "请先读取必要的已有 Skill，再对当前 reflection.md 做最终判定。\n",
        "请先读取必要 Skill，再做最终判定。\n",
    )
    return (
        prefix +
        "先思考这次案件应当沉淀 0 个、1 个，还是多个 Skill。\n"
        "如果你准备写入或更新 Skill，先加载 `skill-creator` 作为写作模板参考。\n"
        f"如果要写入 Skill，`relative_skill_dir` 必须以 `{case_cause_dir}/` 开头，并且必须指向具体的 Skill 文件夹。\n\n"
        "【当前 reflection.md】\n"
        f"{reflection_markdown}\n\n"
        "【写入约束】\n"
        f"- 当前案由目录：{case_cause_dir}\n"
        "- 你可以选择 none，或者写入 1 个或多个 Skill\n"
        "- 每次 upsert_skill_file 只处理一个 Skill 文件夹下的 SKILL.md\n"
        "- 最终输出仍然要是完整 reflection.md"
    )
