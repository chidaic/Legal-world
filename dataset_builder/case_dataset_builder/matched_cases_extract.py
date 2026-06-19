"""
一审二审配对案件数据信息抽取脚本
适配 matched_cases_split 目录下的新 JSON 格式数据
将一审二审文书内容合并喂给 LLM 提取结构化信息，
再拼接原始数据中的案由、审理法院、裁判日期、法律依据等字段。
"""

import json
import re


def clean_nbsp(text):
    """
    清理文本中的 &nbsp; 及其变体
    """
    if not text:
        return text
    # 清理各种形式的 &nbsp;
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&nbsp', ' ')
    text = text.replace('\xa0', ' ')
    # 合并连续的多个空格为单个空格
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def build_extraction_prompt(case_data):
    """
    构建信息抽取的 Prompt
    
    将一审和二审的标题、案件类型、审理程序、文书内容、当事人合并后喂给LLM。
    
    Args:
        case_data: 单条配对案件数据，包含 first_instance 和 second_instance
    
    Returns:
        完整的提示词
    """
    first = case_data.get('first_instance', {})
    second = case_data.get('second_instance', {})
    
    # 组装一审文本
    first_text = f"【一审】\n"
    first_text += f"标题：{clean_nbsp(first.get('标题', ''))}\n"
    first_text += f"案件类型：{clean_nbsp(first.get('案件类型', ''))}\n"
    first_text += f"审理程序：{clean_nbsp(first.get('审理程序', ''))}\n"
    first_text += f"当事人：{clean_nbsp(first.get('当事人', ''))}\n"
    first_text += f"文书内容：\n{clean_nbsp(first.get('文书内容', ''))}\n"
    
    # 组装二审文本
    second_text = f"\n【二审】\n"
    second_text += f"标题：{clean_nbsp(second.get('标题', ''))}\n"
    second_text += f"案件类型：{clean_nbsp(second.get('案件类型', ''))}\n"
    second_text += f"审理程序：{clean_nbsp(second.get('审理程序', ''))}\n"
    second_text += f"当事人：{clean_nbsp(second.get('当事人', ''))}\n"
    second_text += f"文书内容：\n{clean_nbsp(second.get('文书内容', ''))}\n"
    
    doc_text = first_text + second_text
    
    prompt = '''
你是法律文书专家，现给定一组配对的民事一审和二审裁判文书，你需要从中抽取完整的结构化信息。

## 抽取要求

请从裁判文书中提取以下信息：

### 1. 基本信息
- case_background: 案件起因与背景（**绝对禁止**包含任何诉讼过程、判决结果、"一审"、"二审"、"起诉"等词汇。必须还原为一段纯粹从未打过官司的纠纷事实描述。）

### 2. 当事人信息
请首先准确判断当事人类型（`type`）：
- **自然人**: 具体的人（如"张伟"）。
- **法人/非法人组织**: 公司、企业、基地、工厂、店铺、个体工商户字号等（如"某某有限公司"、"某某基地"）。

针对不同类型，提取/生成不同的字段：
- **自然人** 需包含: name, type, gender, ethnicity, birth_date, address
- **法人/非法人组织** 需包含: name, type, address, representative(法定代表人/经营者)
  *(注意：法人/组织绝不应有 gender, ethnicity, birth_date 字段)*

- plaintiff: 原告信息（一审原告）
- defendant: 被告信息（一审被告）

### 3. 上诉人信息
- appellant: 原告/被告   只填写原告或者被告

### 4. 一审信息 (first_instance)
- judges: 一审审判员/审判长姓名列表
- plaintiff_claim: 原告诉讼请求
  - name: 原告姓名
  - claim: 诉讼请求列表
- facts_and_reasons: 事实和理由
- defendant_plea: 被告辩称
  - name: 被告姓名
  - plea: 被告辩称内容
- court_finding: 法院认定事实
- evidence: 证据信息
  - plaintiff_evidence: 原告证据（包含evidence和dispute字段）
  - defendant_evidence: 被告证据（包含evidence和dispute字段）
- court_opinion: 法院认为
- final_judgment: 最终判决
  - judgment_result: 判决结果列表

### 5. 二审信息 (second_instance)
- judges: 二审审判员/审判长姓名列表
- appellant_claim: 上诉人上诉请求
  - name: 上诉人姓名
  - claim: 上诉请求列表
  - reasons: 上诉事实与理由
- appellee_defense: 被上诉人答辩
  - name: 被上诉人姓名
  - defense: 答辩意见
- court_finding: 本院查明  法院认定案件的事实内容
- new_evidence: 新证据（如有）
  - appellant_evidence: 上诉人证据
  - appellee_evidence: 被上诉人证据
- court_opinion: 本院认为
- final_judgment: 最终判决
  - judgment_result: 判决结果列表

## 注意事项
1. **姓名替换**：如果文书中出现"张某"、"李某某"等隐去真实姓名的写法，**请务必将其替换为合理的、真实的中文姓名，不要保留"某"这种写法。
2. **缺失信息自动补全**：
   - **对于自然人**：如果文书未提及**性别、民族、出生日期或具体住址**，**请务必结合上下文生成合理、真实的虚拟数据填充**。字段不得为空。
   - **对于法人/组织**：如果文书未提及**具体住所地**或**法定代表人/经营者**，**请务必生成合理的数据填充**。字段不得为空。
3. **严格区分当事人类型**：
   - 看到"基地"、"公司"、"厂"、"店"等名称，必须归类为"法人"（或组织），**绝对禁止**为其生成性别、出生日期、民族等自然人属性。
4. **彻底清洗诉讼痕迹**：在提取 `case_background` 时，请假装你是一个在纠纷刚发生、还未去法院时旁观的记者。只记录发生了什么事，不记录谁告了谁，谁判了谁。
5. 完全忠于原文提取案情、证据、判决等核心事实，不要修改。
6. **审判员姓名**：从文书末尾署名处准确提取审判长、审判员的姓名，一审和二审分别提取。

## 输出格式
请严格按照以下JSON格式输出，注意根据 type 选择不同的当事人字段结构：

{{
    "case_background": "案件背景详述（纯事实，无诉讼痕迹）",
    
    "party_info": {{
        "plaintiff": {{
            "name": "原告姓名",
            "type": "自然人" 或 "法人",
            
            // 仅当 type="自然人" 时输出以下4个字段：
            "gender": "性别(必填)",
            "ethnicity": "民族(必填)",
            "birth_date": "出生日期(必填)",
            "address": "住址(必填)",

            // 仅当 type="法人" 时输出以下2个字段：
            "address": "住所地(必填)",
            "representative": ""
        }},
        "defendant": {{
            "name": "被告姓名",
            "type": "自然人" 或 "法人",
            
            // 仅当 type="自然人" 时输出：
            "gender": "...", 
            "ethnicity": "...",
            "birth_date": "...",
            "address": "...",

             // 仅当 type="法人" 时输出：
            "address": "...",
            "representative": "..."
        }}
    }},
    
    "appellant": "上诉人(原告/被告) 只填写原告或者被告
",
    
    "first_instance": {{
        "judges": ["审判长/审判员姓名"],
        "plaintiff_claim": {{
            "name": "原告姓名",
            "claim": [
                "诉讼请求1", 
                "..."
            ]
        }},
        "facts_and_reasons": "事实和理由",
        "defendant_plea": {{
            "name": "被告姓名",
            "plea": "被告辩称"
        }},
        "court_finding": "法院认定事实",
        "evidence": {{
            "plaintiff_evidence": {{
                "evidence_1": {{
                    "evidence": "证据内容",
                    "dispute": "法院对证据内容的观点认定"
                }}
            }},
            "defendant_evidence": {{
                "evidence_1": {{ "evidence": "证据内容", "dispute": "法院对证据内容的观点认定" }}
            }}
        }},
        "court_opinion": "法院认为",
        "final_judgment": {{
            "judgment_result": ["判决1", "..."]
        }}
    }},
    
    "second_instance": {{
        "judges": ["审判长/审判员姓名"],
        "appellant_claim": {{
            "name": "上诉人姓名",
            "claim": ["..."],
            "reasons": "..."
        }},
        "appellee_defense": {{
            "name": "被上诉人姓名",
            "plea": "..."
        }},
        "court_finding": "...",
        "new_evidence": {{
            "appellant_evidence": {{}},
            "appellee_evidence": {{}}
        }},
        "court_opinion": "...",
        "final_judgment": {{
            "judgment_result": ["..."]
        }}
    }}
}}

## 待抽取的裁判文书内容：

{doc_text}
'''
    return prompt.format(doc_text=doc_text)


def extract_clean_json(response_text):
    """从模型响应中提取干净的 JSON，兼容 markdown 代码块和前后说明文本"""
    response_clean = (response_text or "").strip()
    if not response_clean:
        raise json.JSONDecodeError("空响应", response_clean, 0)

    if response_clean.startswith("```"):
        lines = response_clean.splitlines()
        json_lines = []
        inside_block = False
        for line in lines:
            if line.strip().startswith("```") and not inside_block:
                inside_block = True
                continue
            if line.strip() == "```" and inside_block:
                break
            if inside_block:
                json_lines.append(line)
        response_clean = "\n".join(json_lines).strip()

    if not response_clean.startswith("{"):
        start = response_clean.find("{")
        end = response_clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            response_clean = response_clean[start:end + 1]

    return json.loads(response_clean)


def merge_with_original(extracted, case_data):
    """
    将 LLM 提取结果 与 原始数据中的字段合并，组成完整的 JSON 数据。
    
    拼接规则：
    - 顶层：拼接 案由 (case_cause)
    - first_instance：拼接 案号 (case_number)、审理法院 (court)、裁判日期 (judgment_date)、法律依据 (legal_basis)
    - second_instance：拼接 案号 (case_number)、审理法院 (court)、裁判日期 (judgment_date)、法律依据 (legal_basis)
    
    Args:
        extracted: LLM 提取的结构化数据
        case_data: 原始配对案件数据
    
    Returns:
        合并后的完整数据
    """
    if "error" in extracted:
        return extracted
    
    first_raw = case_data.get('first_instance', {})
    second_raw = case_data.get('second_instance', {})
    
    # 获取案由：优先取非空的
    case_cause = first_raw.get('案由', '') or second_raw.get('案由', '')
    extracted['case_cause'] = case_cause
    
    # 拼接一审子对象的原始数据
    if 'first_instance' in extracted:
        extracted['first_instance']['case_number'] = first_raw.get('案号', '')
        extracted['first_instance']['court'] = first_raw.get('审理法院', '')
        extracted['first_instance']['judgment_date'] = first_raw.get('裁判日期', '')
        extracted['first_instance']['legal_basis'] = first_raw.get('法律依据', '')
        extracted['first_instance']['url'] = first_raw.get('网页链接', '')
    
    # 拼接二审子对象的原始数据
    if 'second_instance' in extracted:
        extracted['second_instance']['case_number'] = second_raw.get('案号', '')
        extracted['second_instance']['court'] = second_raw.get('审理法院', '')
        extracted['second_instance']['judgment_date'] = second_raw.get('裁判日期', '')
        extracted['second_instance']['legal_basis'] = second_raw.get('法律依据', '')
        extracted['second_instance']['url'] = second_raw.get('网页链接', '')
    
    return extracted


