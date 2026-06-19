# 代码文件介绍

这个文件夹是一个独立的数据构造小包，目标是把已经配对好的一审/二审裁判文书 raw JSON 构造成最终的法律咨询问答数据集。

它不包含爬虫、文书下载、一审二审匹配、统计分析、画图脚本，也不包含 J-one 评测数据构造代码。

## 整体流程

主流程在 `case_dataset_builder/pipeline.py` 中，分为三步：

1. 读取 `matched_cases_*_raw.json`。
2. 从一审/二审裁判文书中抽取结构化案件信息，生成 `stage1_extracted.json`。
3. 为当事人生成法律人格画像，生成 `stage2_persona.json`。
4. 基于案件信息和人格画像生成咨询问题与参考答案，生成 `light_case_dataset.json`。

## 顶层文件

### `README.md`

简单使用说明，包括输入格式、依赖安装、API 配置、运行命令和输出文件。

### `requirements.txt`

运行所需的最小依赖：

- `openai`：调用 OpenAI-compatible 模型接口。
- `tqdm`：显示处理进度条。

### `CODE_OVERVIEW.md`

当前文件，说明代码结构和各文件作用。

## `case_dataset_builder/` 目录

### `__init__.py`

Python 包标记文件，使 `case_dataset_builder` 可以通过 `python -m case_dataset_builder.pipeline` 方式运行。

### `pipeline.py`

主入口文件。

主要职责：

- 解析命令行参数。
- 校验输入 raw JSON。
- 管理输出目录。
- 串联三个阶段。
- 控制并发和超时。
- 支持断点续跑。
- 写出 `run_summary.txt`。

推荐运行入口：

```bash
python -m case_dataset_builder.pipeline <input_raw_json> --output-dir <output_dir>
```

### `matched_cases_extract.py`

Stage 1 的核心逻辑。

主要职责：

- 构造案件信息抽取 prompt。
- 将一审和二审文书内容合并后交给 LLM。
- 从 LLM 返回内容中提取 JSON。
- 把原始数据中的案由、案号、法院、裁判日期、法律依据、网页链接等字段合并回结构化结果。

输出结果会进入 `stage1_extracted.json`。

### `generate_legal_profile.py`

Stage 2 的核心逻辑。

主要职责：

- 定义法律人格画像 prompt。
- 为当事人生成四个维度的人格等级：
  - 法律素养水平
  - 信息披露意愿
  - 情绪稳定性
  - 叙事表达能力
- 支持 LLM 生成人格画像。
- 保留随机人格画像生成函数，便于调试或低成本构造。

输出结果会进入 `stage2_persona.json`。

### `matched_generate_consultation_questions.py`

Stage 3 的核心逻辑。

主要职责：

- 整理完整案件上下文。
- 读取当事人的法律人格画像。
- 为每个当事人生成咨询问题。
- 为每个问题生成对应参考答案。
- 将生成结果写回当事人字段中的 `questions`。

输出结果会进入最终的 `light_case_dataset.json`。

### `legal_persona_common.py`

法律人格相关的公共工具。

主要职责：

- 定义人格字段名和等级。
- 归一化人格等级，例如把中文“高/中/低”统一为 `high/medium/low`。
- 遍历案件中的原告、被告当事人。
- 清理旧的人格字段。
- 提供预设人格配置。

### `utils.py`

通用工具文件。

主要职责：

- 读取 JSON。
- 保存 JSON。
- 调用 OpenAI-compatible Chat Completions API。
- 从环境变量读取 API 配置。

不会硬编码 API Key 或本机路径。

相关环境变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `LEGAL_DATASET_MAX_RETRIES`
- `LEGAL_DATASET_MAX_TOKENS`
- `LEGAL_DATASET_TEMPERATURE`

## 产物说明

运行主流程后，输出目录中通常会有：

- `stage1_extracted.json`：案件结构化抽取结果。
- `stage2_persona.json`：加入法律人格画像后的结果。
- `light_case_dataset.json`：最终数据集。
- `run_summary.txt`：运行摘要。

