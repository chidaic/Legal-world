# Matched Case Dataset Builder

用于把一审/二审配对后的裁判文书 raw JSON 构造成法律咨询问答数据集。

本目录只包含数据构造代码，不包含爬虫、统计、画图、J-one 评测构造脚本或原始数据。

公开数据集地址：<https://huggingface.co/datasets/Chidaic/legal-world>

## 输入格式

输入文件应为 `matched_cases_*_raw.json`，每条数据大致包含：

```json
{
  "id": 1,
  "first_instance": {
    "标题": "...",
    "文书内容": "...",
    "案由": "..."
  },
  "second_instance": {
    "标题": "...",
    "文书内容": "...",
    "案由": "..."
  }
}
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置 API

使用 OpenAI-compatible 接口：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_api_base_url"
export OPENAI_MODEL="your_model_name"
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="your_api_base_url"
$env:OPENAI_MODEL="your_model_name"
```

## 运行

```bash
python -m case_dataset_builder.pipeline <input_raw_json> --output-dir <output_dir>
```

也可以显式指定模型：

```bash
python -m case_dataset_builder.pipeline <input_raw_json> --output-dir <output_dir> --model <model_name>
```

## 输出文件

运行后会在 `output_dir` 下生成：

- `stage1_extracted.json`：结构化案件信息。
- `stage2_persona.json`：加入法律人格画像后的案件数据。
- `light_case_dataset.json`：最终问答数据集。
- `run_summary.txt`：运行摘要。

## 代码说明

- `case_dataset_builder/pipeline.py`：主入口，负责完整流程。
- `case_dataset_builder/matched_cases_extract.py`：抽取案件结构化信息。
- `case_dataset_builder/generate_legal_profile.py`：生成法律人格画像。
- `case_dataset_builder/matched_generate_consultation_questions.py`：生成咨询问题和参考答案。
- `case_dataset_builder/legal_persona_common.py`：人格字段和当事人处理工具。
- `case_dataset_builder/utils.py`：JSON 读写和 API 调用。
