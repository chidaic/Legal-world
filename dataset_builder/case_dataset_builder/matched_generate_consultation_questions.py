"""
生成当事人法律咨询问题及参考答案脚本（人格引导的一步式生成）

策略：
对每个案件只做 1 次 LLM 调用，同时为所有当事人生成成组的
question + reference_answer。

核心原则：
1. 参考答案仍然必须基于案件真实审理信息，保证可评分、可核对。
2. 但“生成哪些问答对”要受当事人的法律人格影响：
   - 人格决定其更可能关注的法律点
   - 人格决定问题的口吻、具体度、情绪感、信息暴露程度、结构感
3. 问题与答案在同一次生成中配套输出，避免“两步法”里答案先固定导致问题风格受限。
"""

import json

from . import utils


PERSONA_FIELD_LABELS = {
    "legal_literacy_level": "法律素养水平",
    "information_disclosure_willingness": "信息披露意愿",
    "emotional_stability": "情绪稳定性",
    "narrative_proficiency": "叙事表达能力",
}

PERSONA_LEVEL_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}


def safe_text(value):
    """将不同结构的数据安全转成文本"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = safe_text(item)
            if text:
                parts.append(text)
        return "；".join(parts)
    if isinstance(value, dict):
        for key in ("plea", "defense", "claim", "reasons", "judgment_result", "evidence"):
            if key in value:
                return safe_text(value.get(key))
        if "name" in value:
            content = []
            for key in ("plea", "defense", "claim", "reasons", "evidence", "dispute"):
                if key in value:
                    field_text = safe_text(value.get(key))
                    if field_text:
                        content.append(field_text)
            if content:
                return f"{value.get('name', '')}：{'；'.join(content)}"
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def first_structured_item(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return None


def build_full_case_context(extracted_info):
    """构建完整的案件上下文信息（包括一审二审）"""
    parts = []
    parts.append(f"案件类别: {extracted_info.get('case_cause', '未知')}")
    parts.append(f"案件背景: {extracted_info.get('case_background', '无')}")
    
    first = extracted_info.get('first_instance', {})
    if first:
        parts.append("\n=== 一审信息 ===")
        if first.get('plaintiff_claim'):
            parts.append(f"原告诉讼请求: {safe_text(first['plaintiff_claim'])}")
        if first.get('facts_and_reasons'):
            parts.append(f"事实与理由: {first['facts_and_reasons']}")
        if first.get('defendant_plea'):
            parts.append(f"被告答辩: {safe_text(first['defendant_plea'])}")
        if first.get('court_finding'):
            parts.append(f"法院查明事实: {first['court_finding']}")
        if first.get('evidence'):
            evidence = first['evidence']
            if evidence.get('plaintiff_evidence'):
                ev_texts = []
                for k, v in evidence['plaintiff_evidence'].items():
                    ev_texts.append(safe_text(v))
                parts.append(f"原告证据: {'; '.join(ev_texts)}")
            if evidence.get('defendant_evidence'):
                ev_texts = []
                for k, v in evidence['defendant_evidence'].items():
                    ev_texts.append(safe_text(v))
                parts.append(f"被告证据: {'; '.join(ev_texts)}")
        if first.get('court_opinion'):
            parts.append(f"一审法院认为: {first['court_opinion']}")
        if first.get('final_judgment'):
            parts.append(f"一审判决结果: {safe_text(first['final_judgment'])}")
        if first.get('legal_basis'):
            parts.append(f"一审法律依据: {first['legal_basis']}")
    
    second = extracted_info.get('second_instance', {})
    if second:
        parts.append("\n=== 二审信息 ===")
        if second.get('appellant_claim'):
            ac = first_structured_item(second['appellant_claim'])
            if ac:
                parts.append(f"上诉人: {ac.get('name', '')}")
                parts.append(f"上诉请求: {safe_text(ac.get('claim', []))}")
                if ac.get('reasons'):
                    parts.append(f"上诉理由: {ac['reasons']}")
            else:
                parts.append(f"上诉请求: {safe_text(second['appellant_claim'])}")
        if second.get('appellee_defense'):
            ad = first_structured_item(second['appellee_defense'])
            if ad and ad.get('name'):
                parts.append(f"被上诉人: {ad.get('name', '')}")
            parts.append(f"被上诉人答辩: {safe_text(second['appellee_defense'])}")
        if second.get('court_finding'):
            parts.append(f"二审查明事实: {second['court_finding']}")
        if second.get('court_opinion'):
            parts.append(f"二审法院认为: {second['court_opinion']}")
        if second.get('final_judgment'):
            parts.append(f"二审判决结果: {safe_text(second['final_judgment'])}")
        if second.get('legal_basis'):
            parts.append(f"二审法律依据: {second['legal_basis']}")
    
    return '\n'.join(parts)


def extract_clean_json(response_text):
    """从LLM响应中提取JSON，处理可能的markdown代码块包裹"""
    response_clean = response_text.strip()
    if response_clean.startswith('```'):
        lines = response_clean.split('\n')
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip().startswith('```') and not in_json:
                in_json = True
                continue
            elif line.strip() == '```':
                break
            elif in_json:
                json_lines.append(line)
        response_clean = '\n'.join(json_lines)
    return json.loads(response_clean)


def build_parties_description(party_info):
    """
    构建所有当事人的描述文本，用于prompt中
    返回 (描述文本, 角色key列表)
    角色key列表如: [("plaintiff", "原告", "张三"), ("defendant", "被告", "李四"), ...]
    """
    parties = []
    desc_parts = []
    
    plaintiff = party_info.get('plaintiff')
    if plaintiff:
        if isinstance(plaintiff, dict):
            parties.append(("plaintiff", "原告", plaintiff.get('name', '未知')))
            desc_parts.append(f"- 原告: {plaintiff.get('name', '未知')}（{plaintiff.get('type', '未知')}）")
        elif isinstance(plaintiff, list):
            for i, p in enumerate(plaintiff):
                if not isinstance(p, dict):
                    continue
                parties.append((f"plaintiff_{i}", f"原告{i+1}", p.get('name', '未知')))
                desc_parts.append(f"- 原告{i+1}: {p.get('name', '未知')}（{p.get('type', '未知')}）")
    
    defendant = party_info.get('defendant')
    if defendant:
        if isinstance(defendant, dict):
            parties.append(("defendant", "被告", defendant.get('name', '未知')))
            desc_parts.append(f"- 被告: {defendant.get('name', '未知')}（{defendant.get('type', '未知')}）")
        elif isinstance(defendant, list):
            for i, d in enumerate(defendant):
                parties.append((f"defendant_{i}", f"被告{i+1}", d.get('name', '未知')))
                desc_parts.append(f"- 被告{i+1}: {d.get('name', '未知')}（{d.get('type', '未知')}）")
    
    third_party = party_info.get('third_party')
    if third_party and isinstance(third_party, dict):
        parties.append(("third_party", "第三人", third_party.get('name', '未知')))
        desc_parts.append(f"- 第三人: {third_party.get('name', '未知')}（{third_party.get('type', '未知')}）")
    
    return '\n'.join(desc_parts), parties


def normalize_persona_level(value):
    """将人格等级统一归一化为 high / medium / low"""
    if value is None:
        return None

    text = str(value).strip().lower()
    mapping = {
        "high": "high",
        "medium": "medium",
        "mid": "medium",
        "middle": "medium",
        "low": "low",
        "高": "high",
        "中": "medium",
        "低": "low",
    }
    if text in mapping:
        return mapping[text]

    raw_text = str(value).strip()
    if "高" in raw_text and "低" not in raw_text:
        return "high"
    if "低" in raw_text and "高" not in raw_text:
        return "low"
    if "中" in raw_text:
        return "medium"
    if "high" in text:
        return "high"
    if "low" in text:
        return "low"
    if "med" in text or "mid" in text:
        return "medium"
    return None


def get_party_data_by_key(party_info, party_key):
    """根据内部角色 key 取回当事人原始数据"""
    if party_key == "plaintiff":
        plaintiff = party_info.get("plaintiff")
        return plaintiff if isinstance(plaintiff, dict) else None

    if party_key.startswith("plaintiff_"):
        plaintiff = party_info.get("plaintiff")
        if isinstance(plaintiff, list):
            idx = int(party_key.split("_")[1])
            if 0 <= idx < len(plaintiff) and isinstance(plaintiff[idx], dict):
                return plaintiff[idx]
        return None

    if party_key == "defendant":
        defendant = party_info.get("defendant")
        return defendant if isinstance(defendant, dict) else None

    if party_key.startswith("defendant_"):
        defendant = party_info.get("defendant")
        if isinstance(defendant, list):
            idx = int(party_key.split("_")[1])
            if 0 <= idx < len(defendant) and isinstance(defendant[idx], dict):
                return defendant[idx]
        return None

    if party_key == "third_party":
        third_party = party_info.get("third_party")
        return third_party if isinstance(third_party, dict) else None

    return None


def extract_persona_profile(party):
    """提取并归一化当事人的法律人格画像"""
    if not isinstance(party, dict):
        return None

    raw_profile = party.get("legal_persona_profile")
    if not isinstance(raw_profile, dict):
        return None

    profile = {}
    for field_key in PERSONA_FIELD_LABELS:
        normalized = normalize_persona_level(
            raw_profile.get(field_key, raw_profile.get(PERSONA_FIELD_LABELS[field_key]))
        )
        if normalized:
            profile[field_key] = normalized

    return profile or None


def build_persona_style_instruction(profile):
    """将人格画像转换成问答生成时的风格指令"""
    if not profile:
        return "未提供人格画像，按普通当事人口吻生成问题即可。"

    instructions = []

    literacy = profile.get("legal_literacy_level")
    if literacy == "high":
        instructions.append("优先关注责任、证据、法律后果等较明确的法律问题，问题中可自然使用较清晰的法律概念")
    elif literacy == "medium":
        instructions.append("优先关注常见责任与维权问题，表达可带少量法律意味，但整体仍以普通人表达为主")
    elif literacy == "low":
        instructions.append("优先关注直观损失、公平感和结果影响，表达应更口语化，少用专业术语")

    disclosure = profile.get("information_disclosure_willingness")
    if disclosure == "high":
        instructions.append("生成问题时可以自然带出较具体的背景事实，不回避关键细节")
    elif disclosure == "medium":
        instructions.append("生成问题时可说出核心事实，但不必把所有细节都放进问题里")
    elif disclosure == "low":
        instructions.append("生成问题时应更谨慎保守，可适度模糊或收着说，不主动暴露过多对自己不利的细节")

    stability = profile.get("emotional_stability")
    if stability == "high":
        instructions.append("语气保持冷静、克制、简洁")
    elif stability == "medium":
        instructions.append("语气可以带一点担心、确认或犹豫，但整体仍可正常沟通")
    elif stability == "low":
        instructions.append("语气可以更焦虑、急迫、敏感或带一点对抗感，但不要失控")

    narrative = profile.get("narrative_proficiency")
    if narrative == "high":
        instructions.append("问题结构可以更完整，重点清楚，条件和诉求较明确")
    elif narrative == "medium":
        instructions.append("问题表达基本清楚，但可以保留一点普通人提问时的不完全结构化")
    elif narrative == "low":
        instructions.append("问题应更短、更碎片化一些，避免过度完整和过分工整")

    return "；".join(instructions) + "。"


def build_parties_persona_description(party_info, parties):
    """构建 prompt 中的人格画像说明"""
    desc_parts = []

    for party_key, role_name, name in parties:
        party = get_party_data_by_key(party_info, party_key)
        profile = extract_persona_profile(party)

        if not profile:
            desc_parts.append(f"【{role_name}（{name}）】未提供法律人格画像。")
            continue

        profile_text = "；".join(
            f"{PERSONA_FIELD_LABELS[field_key]}={PERSONA_LEVEL_LABELS[level]}"
            for field_key, level in profile.items()
        )
        style_text = build_persona_style_instruction(profile)
        desc_parts.append(
            f"【{role_name}（{name}）】法律人格画像：{profile_text}。"
            f"问答生成风格要求：{style_text}"
        )

    return "\n".join(desc_parts)


async def batch_generate_qa_pairs_async(parties, full_case_context, case_info, party_info, model_name=None):
    """
    一次性为所有当事人同时生成 question + reference_answer 对
    """
    roles_text = '\n'.join([f"- {role_name}（{name}）" for _, role_name, name in parties])
    persona_desc = build_parties_persona_description(party_info, parties)

    qa_json = {}
    for _, role_name, name in parties:
        key = f"{role_name}（{name}）"
        qa_json[key] = [
            {
                "question": "问题1",
                "reference_answer": "参考答案1"
            },
            {
                "question": "问题2",
                "reference_answer": "参考答案2"
            },
            {
                "question": "问题3",
                "reference_answer": "参考答案3"
            }
        ]

    prompt = f'''你是一名法律咨询客户模拟器兼法律分析专家。现在给定一个案件的完整审理信息、纠纷背景和各当事人的法律人格画像，
请你同时为每个当事人生成最多3组【咨询问题 + 参考答案】。

任务目标：
1. 先根据该当事人的法律人格，判断他/她在【纠纷发生后、第一次起诉前】最可能会关注哪些法律问题
2. 再围绕这些问题，同时生成对应的标准参考答案
3. 问题与答案必须一一对应、成对输出

重要前提：
- 当事人此时还没有起诉，也不知道法律结果
- 问题必须像真实当事人会问的
- 一审和二审有差异时，以二审为准

问题要求（重要）：
1. 用第一人称表达
2. 问题必须具体、明确，避免过于开放或模糊
3. 每个问题不超过30个字
4. 问题的口吻、详略、专业度、情绪感、信息暴露程度、表达结构，必须符合对应当事人的法律人格画像
5. 不同当事人的问题风格应有区分，不要全部写成同一种平铺直叙的问法
6. 问题应明确、具体，参考类型：
   - 事实认定
   - 责任划分或权利义务
   - 法律依据
   - 具体后果或可行结论
   - 法律流程环节等问题

参考答案要求（重要）：
1. 必须能够直接回答对应问题
2. 以第三人称、客观、简洁的方式表述，每个答案2-4句话
3. 不要提及"法院认定"、"法院判决"、"一审/二审"等审判过程，只陈述事实和法律结论



当事人列表：
{roles_text}

纠纷类别: {case_info.get('case_cause', '未知')}
纠纷背景（仅帮助理解纠纷场景）: {case_info.get('case_background', '无')}

各当事人的法律人格画像与问答生成风格要求：
{persona_desc}

以下是完整的案件审理信息（仅供提取事实和法律依据）：
{full_case_context}

请严格以JSON格式返回，格式如下：
{json.dumps(qa_json, ensure_ascii=False, indent=4)}
    '''

    try:
        response, _ = await utils.aget_completion(prompt, [], 1, model_name=model_name)
        result = extract_clean_json(response)
        return result
    except Exception as e:
        print(f"  ⚠ 批量生成问答失败: {e}")
        return {}


async def process_single_case_async(case_data, case_index=None, model_name=None):
    """
    处理单个案件：1次LLM调用同时为所有当事人生成Q&A
    """
    extracted_info = case_data.get('extracted_info', {})
    party_info = extracted_info.get('party_info', {})
    
    case_info = {
        'case_cause': extracted_info.get('case_cause', ''),
        'case_background': extracted_info.get('case_background', '')
    }
    
    full_case_context = build_full_case_context(extracted_info)
    _, parties = build_parties_description(party_info)
    case_label = f"案例 {case_index + 1}" if case_index is not None else "案例"
    
    if not parties:
        print(f"[{case_label}] ⚠ 未找到当事人信息，跳过")
        return {"status": "failed", "reason": "未找到当事人信息", "qa_count": 0}
    
    print(f"[{case_label}] 正在结合案件材料与法律人格批量生成问答...")
    qa_dict = await batch_generate_qa_pairs_async(
        parties,
        full_case_context,
        case_info,
        party_info,
        model_name=model_name,
    )

    if not qa_dict:
        return {"status": "failed", "reason": "未能生成问答", "qa_count": 0}
    
    # 组装并写回数据
    qa_count = 0
    for party_key, role_name, name in parties:
        dict_key = f"{role_name}（{name}）"
        qa_pairs = qa_dict.get(dict_key, [])
        if not isinstance(qa_pairs, list):
            qa_pairs = []

        normalized_qa_pairs = []
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            normalized_qa_pairs.append(
                {
                    "question": qa.get("question", "咨询问题"),
                    "reference_answer": qa.get("reference_answer", ""),
                }
            )
        qa_count += len(normalized_qa_pairs)
        print(f"[{case_label}] → {dict_key}: {len(normalized_qa_pairs)} 个问答对")
        
        # 写回到对应的当事人数据中
        if party_key == "plaintiff":
            party_info['plaintiff']['questions'] = normalized_qa_pairs
        elif party_key.startswith("plaintiff_"):
            idx = int(party_key.split("_")[1])
            party_info['plaintiff'][idx]['questions'] = normalized_qa_pairs
        elif party_key == "defendant":
            party_info['defendant']['questions'] = normalized_qa_pairs
        elif party_key.startswith("defendant_"):
            idx = int(party_key.split("_")[1])
            party_info['defendant'][idx]['questions'] = normalized_qa_pairs
        elif party_key == "third_party":
            party_info['third_party']['questions'] = normalized_qa_pairs
    
    return {"status": "success", "reason": "", "qa_count": qa_count}
