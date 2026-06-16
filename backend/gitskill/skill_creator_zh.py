from __future__ import annotations


SKILL_CREATOR_ZH = """---
name: skill-creator
description: 创建高质量 Skill 的指南。当用户想要创建新 Skill，或更新一个已有 Skill 以扩展 Codex 在专门知识、工作流或工具集成方面的能力时，应使用本指南。
metadata:
  short-description: 创建或更新 Skill
---

# Skill Creator

本 Skill 提供创建高质量 Skill 的指导。

## 关于 Skill

Skill 是模块化、自包含的文件夹，用专门知识、工作流和工具来扩展 Codex 的能力。可以把它理解为某个领域或任务的“入职指南”：它把通用 Agent 转化为具备程序性知识的专门 Agent，而这些程序性知识通常不是任何基础模型都能完整掌握的。

### Skill 能提供什么

1. 专门工作流：面向特定领域的多步骤流程。
2. 工具集成：处理特定文件格式或 API 的操作说明。
3. 领域知识：公司内部知识、数据模式、业务逻辑。
4. 打包资源：用于复杂、重复任务的脚本、参考资料和素材。

## 核心原则

### 简洁是关键

上下文窗口是一种公共资源。Skill 会与 Codex 所需的一切共享上下文窗口，包括 system prompt、对话历史、其他 Skill 的元数据，以及用户的真实请求。

默认假设：Codex 已经非常聪明。只添加 Codex 尚不具备的上下文。审视每一条信息：Codex 真的需要这段解释吗？这段内容值得它消耗的 token 吗？

优先使用简洁示例，而不是冗长解释。

### 设置合适的自由度

根据任务的脆弱程度和变化程度，匹配说明的具体程度：

高自由度（文本式说明）：当多种做法都合理、决策依赖上下文、或主要靠启发式判断时使用。

中自由度（伪代码或带参数脚本）：当存在推荐模式、允许一定变化、或配置会影响行为时使用。

低自由度（具体脚本、少量参数）：当操作脆弱且容易出错、一致性非常关键、或必须遵循特定步骤顺序时使用。

可以把 Codex 想成在探索路径：狭窄悬崖桥需要明确护栏（低自由度），开阔场地则允许多条路线（高自由度）。

### 保护验证完整性

迭代 Skill 时，可以使用子 Agent 验证 Skill 是否能在真实任务上工作，或某个怀疑的问题是否真实存在。这种方式适合在修订后独立检查 Skill 的行为、输出或失败模式。只有在能够启动新子 Agent 时才使用。

使用子 Agent 做验证时，要把它视为评估面。目标是判断 Skill 是否能泛化，而不是判断另一个 Agent 能否从泄露的上下文中重构答案。

优先传递原始产物，例如示例 prompt、输出、diff、日志或 trace。只提供完成验证所需的最小任务上下文。除非验证明确需要，否则不要传递预期答案、怀疑的 bug、计划中的修复方案或你自己的先验结论。

### Skill 的结构

每个 Skill 都由必需的 SKILL.md 文件和可选的打包资源组成：

```text
skill-name/
|-- SKILL.md (必需)
|   |-- YAML frontmatter 元数据 (必需)
|   |   |-- name: (必需)
|   |   `-- description: (必需)
|   `-- Markdown 使用说明 (必需)
|-- agents/ (推荐)
|   `-- openai.yaml - Skill 列表和标签使用的 UI 元数据
`-- 打包资源 (可选)
    |-- scripts/          - 可执行代码，例如 Python、Bash 等
    |-- references/       - 按需加载进上下文的文档
    `-- assets/           - 输出中使用的文件，例如模板、图标、字体等
```

#### SKILL.md（必需）

每个 SKILL.md 都包含：

- Frontmatter（YAML）：包含 `name` 和 `description` 字段。Codex 只读取这些字段来判断何时使用该 Skill，因此必须清晰、完整地描述 Skill 是什么，以及什么时候应该使用它。
- Body（Markdown）：使用该 Skill 的说明和指导。只有在 Skill 被触发之后才会加载。

#### Agents 元数据（推荐）

- 面向 UI 的 Skill 列表和标签元数据。
- 生成字段前先阅读 `references/openai_yaml.md`，并遵循其中的描述和约束。
- 通过阅读 Skill 创建面向人的 `display_name`、`short_description` 和 `default_prompt`。
- 用确定性方式生成：把这些值作为 `--interface key=value` 传给 `scripts/generate_openai_yaml.py` 或 `scripts/init_skill.py`。
- 更新时：验证 `agents/openai.yaml` 仍然与 SKILL.md 匹配；如果已经过期则重新生成。
- 只有用户明确提供时，才包含其他可选界面字段，例如图标、品牌色。
- 字段定义和示例见 `references/openai_yaml.md`。

#### 打包资源（可选）

##### 脚本（`scripts/`）

用于需要确定性可靠性，或会被反复重写的任务的可执行代码，例如 Python、Bash 等。

- 何时包含：同样的代码会被反复重写，或任务需要确定性可靠性。
- 示例：PDF 旋转任务中的 `scripts/rotate_pdf.py`。
- 好处：节省 token、结果确定、可在不加载进上下文的情况下执行。
- 注意：Codex 仍可能需要阅读脚本，以便打补丁或做环境相关调整。

##### 参考资料（`references/`）

用于按需加载进上下文、帮助 Codex 制定流程和思考的文档及参考材料。

- 何时包含：Codex 工作时应参考的文档。
- 示例：金融数据模式的 `references/finance.md`、公司 NDA 模板的 `references/mnda.md`、公司政策的 `references/policies.md`、API 规格的 `references/api_docs.md`。
- 使用场景：数据库 schema、API 文档、领域知识、公司政策、详细工作流指南。
- 好处：保持 SKILL.md 精简，只在 Codex 判断需要时加载。
- 最佳实践：如果文件很大（超过 1 万词），在 SKILL.md 中提供 grep 搜索模式。
- 避免重复：信息应该只存在于 SKILL.md 或 reference 文件之一，不要两边都放。除非确实是 Skill 的核心内容，否则详细信息优先放到 references 文件中；这样 SKILL.md 保持精简，同时信息仍可被发现，不会占满上下文窗口。SKILL.md 中只保留必要的程序性说明和工作流指导；把详细参考材料、schema 和示例移到 reference 文件。

##### 素材（`assets/`）

不会被加载进上下文，而是会被 Codex 用于最终输出的文件。

- 何时包含：Skill 需要最终输出会用到的文件。
- 示例：品牌素材 `assets/logo.png`、PowerPoint 模板 `assets/slides.pptx`、HTML/React 样板 `assets/frontend-template/`、字体 `assets/font.ttf`。
- 使用场景：模板、图片、图标、样板代码、字体、会被复制或修改的示例文档。
- 好处：把输出资源与说明文档分离，使 Codex 能使用文件而不必把它们加载进上下文。

#### 不要在 Skill 中包含什么

Skill 应只包含直接支持其功能的必要文件。不要创建多余的文档或辅助文件，包括：

- README.md
- INSTALLATION_GUIDE.md
- QUICK_REFERENCE.md
- CHANGELOG.md
- 等等

Skill 只应包含 AI Agent 完成当前工作所需的信息。它不应包含创建过程的辅助上下文、安装和测试流程、面向用户的说明文档等。额外文档只会增加混乱和噪声。

### 渐进披露设计原则

Skill 使用三级加载系统来高效管理上下文：

1. 元数据（name + description）：始终在上下文中，约 100 词。
2. SKILL.md 正文：Skill 触发时加载，少于 5000 词。
3. 打包资源：Codex 按需加载；脚本可以在不读入上下文窗口的情况下执行，因此容量不受上下文直接限制。

#### 渐进披露模式

为减少上下文膨胀，SKILL.md 正文应只保留必要内容，并控制在 500 行以内。接近该限制时，应拆分到独立文件。拆分内容到其他文件时，必须从 SKILL.md 明确引用，并说明什么时候读取它们，确保 Skill 使用者知道这些文件存在以及何时使用。

关键原则：当一个 Skill 支持多个变体、框架或选项时，SKILL.md 只保留核心工作流和选择指南。把特定变体的细节（模式、示例、配置）移到独立参考文件。

模式 1：高层指南加参考文件

```markdown
# PDF Processing

## Quick start

Extract text with pdfplumber:
[code example]

## Advanced features

- **Form filling**: See [FORMS.md](FORMS.md) for complete guide
- **API reference**: See [REFERENCE.md](REFERENCE.md) for all methods
- **Examples**: See [EXAMPLES.md](EXAMPLES.md) for common patterns
```

Codex 只在需要时加载 FORMS.md、REFERENCE.md 或 EXAMPLES.md。

模式 2：按领域组织

对于包含多个领域的 Skill，按领域组织内容，避免加载无关上下文：

```text
bigquery-skill/
|-- SKILL.md (overview and navigation)
`-- reference/
    |-- finance.md (revenue, billing metrics)
    |-- sales.md (opportunities, pipeline)
    |-- product.md (API usage, features)
    `-- marketing.md (campaigns, attribution)
```

当用户询问销售指标时，Codex 只读取 sales.md。

类似地，对于支持多个框架或变体的 Skill，也按变体组织：

```text
cloud-deploy/
|-- SKILL.md (workflow + provider selection)
`-- references/
    |-- aws.md (AWS deployment patterns)
    |-- gcp.md (GCP deployment patterns)
    `-- azure.md (Azure deployment patterns)
```

当用户选择 AWS 时，Codex 只读取 aws.md。

模式 3：条件细节

展示基础内容，链接高级内容：

```markdown
# DOCX Processing

## Creating documents

Use docx-js for new documents. See [DOCX-JS.md](DOCX-JS.md).

## Editing documents

For simple edits, modify the XML directly.

**For tracked changes**: See [REDLINING.md](REDLINING.md)
**For OOXML details**: See [OOXML.md](OOXML.md)
```

Codex 只在用户需要这些功能时读取 REDLINING.md 或 OOXML.md。

重要指导：

- 避免深层嵌套引用：references 与 SKILL.md 保持一层关系。所有参考文件都应直接从 SKILL.md 链接。
- 组织较长参考文件：超过 100 行的文件，应在顶部包含目录，便于 Codex 预览时理解完整范围。

## Skill 创建流程

Skill 创建包含以下步骤：

1. 通过具体示例理解 Skill。
2. 规划可复用的 Skill 内容，例如脚本、参考资料、素材。
3. 初始化 Skill，运行 `init_skill.py`。
4. 编辑 Skill，实施资源并编写 SKILL.md。
5. 验证 Skill，运行 `quick_validate.py`。
6. 根据真实使用情况迭代，并对复杂 Skill 做前向测试。

除非有明确理由说明某一步不适用，否则应按顺序执行。

### Skill 命名

- 只使用小写字母、数字和连字符；将用户提供的标题规范化为连字符格式，例如 `Plan Mode` 转为 `plan-mode`。
- 生成名称时，控制在 64 个字符以内，字符只包括字母、数字和连字符。
- 优先使用简短的、以动词引导的短语来描述动作。
- 当按工具做命名空间能提高可理解性或触发准确性时，可以这样做，例如 `gh-address-comments`、`linear-address-issue`。
- Skill 文件夹名称必须与 Skill 名称完全一致。

### 第 1 步：通过具体示例理解 Skill

只有在 Skill 的使用模式已经非常清楚时才跳过本步骤。即使是在处理已有 Skill，本步骤仍然有价值。

要创建高质量 Skill，必须清楚理解它会如何被具体使用。这种理解可以来自用户直接给出的示例，也可以来自生成的示例，但生成示例需要通过用户反馈验证。

例如，在创建 image-editor Skill 时，相关问题包括：

- 这个 image-editor Skill 应支持什么功能？编辑、旋转，还是其他功能？
- 能否给一些这个 Skill 会如何使用的例子？
- 我能想到用户可能会说“去掉这张图片里的红眼”或“旋转这张图片”。你还设想过其他使用方式吗？
- 用户说什么内容时应触发这个 Skill？
- 你希望把这个 Skill 创建在哪里？如果没有偏好，我会放到 `$CODEX_HOME/skills`；如果未设置 `CODEX_HOME`，则放到 `~/.codex/skills`，这样 Codex 可以自动发现它。

为了避免让用户负担过重，不要在一条消息里问太多问题。先问最重要的问题，再按需要继续追问，以提高有效性。

当已经清楚 Skill 应支持哪些功能时，结束本步骤。

### 第 2 步：规划可复用的 Skill 内容

要把具体示例转化为有效 Skill，需要对每个示例进行分析：

1. 思考如果从零开始，如何完成这个示例任务。
2. 识别在反复执行这些工作流时，哪些脚本、参考资料和素材会有帮助。

示例：创建 `pdf-editor` Skill 来处理“帮我旋转这个 PDF”这类请求时，分析结果是：

1. 旋转 PDF 每次都要重写同样的代码。
2. 在 Skill 中存放 `scripts/rotate_pdf.py` 会很有帮助。

示例：设计 `frontend-webapp-builder` Skill 来处理“帮我做一个 todo app”或“做一个记录步数的 dashboard”这类请求时，分析结果是：

1. 编写前端 Web 应用每次都需要类似的 HTML/React 样板。
2. 在 Skill 中存放包含 HTML/React 项目样板文件的 `assets/hello-world/` 模板会很有帮助。

示例：创建 `big-query` Skill 来处理“今天有多少用户登录？”这类请求时，分析结果是：

1. 查询 BigQuery 每次都需要重新发现表结构和表关系。
2. 在 Skill 中存放记录表结构的 `references/schema.md` 会很有帮助。

为了确定 Skill 内容，应分析每个具体示例，并形成需要包含的可复用资源清单：脚本、参考资料和素材。

### 第 3 步：初始化 Skill

到这一步，就应真正创建 Skill。

只有在要开发的 Skill 已经存在时才跳过本步骤；这种情况下继续下一步。

运行 `init_skill.py` 之前，先询问用户希望在哪里创建 Skill。如果用户没有指定位置，默认使用 `$CODEX_HOME/skills`；如果未设置 `CODEX_HOME`，则回退到 `~/.codex/skills`，以便 Skill 可被自动发现。

从零创建新 Skill 时，始终运行 `init_skill.py` 脚本。该脚本会方便地生成一个新的 Skill 模板目录，并自动包含 Skill 所需的一切，使创建流程更高效、更可靠。

用法：

```bash
scripts/init_skill.py <skill-name> --path <output-directory> [--resources scripts,references,assets] [--examples]
```

示例：

```bash
scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills"
scripts/init_skill.py my-skill --path "${CODEX_HOME:-$HOME/.codex}/skills" --resources scripts,references
scripts/init_skill.py my-skill --path ~/work/skills --resources scripts --examples
```

该脚本会：

- 在指定路径创建 Skill 目录。
- 生成带有正确 frontmatter 和 TODO 占位符的 SKILL.md 模板。
- 用通过 `--interface key=value` 传入的、由 Agent 生成的 `display_name`、`short_description` 和 `default_prompt` 创建 `agents/openai.yaml`。
- 根据 `--resources` 可选创建资源目录。
- 当设置 `--examples` 时可选添加示例文件。

初始化后，按需要自定义 SKILL.md 并添加资源。如果使用了 `--examples`，请替换或删除占位文件。

通过阅读 Skill 生成 `display_name`、`short_description` 和 `default_prompt`，再作为 `--interface key=value` 传给 `init_skill.py`，或者用下面命令重新生成：

```bash
scripts/generate_openai_yaml.py <path/to/skill-folder> --interface key=value
```

只有在用户明确提供时，才包含其他可选界面字段。完整字段描述和示例见 `references/openai_yaml.md`。

### 第 4 步：编辑 Skill

编辑新生成或已有的 Skill 时，要记住这个 Skill 是给另一个 Codex 实例使用的。只放对 Codex 有帮助、且并非显而易见的信息。思考哪些程序性知识、领域细节或可复用素材能帮助另一个 Codex 实例更有效地完成这些任务。

经过重大修订后，或者 Skill 特别棘手时，应使用子 Agent 在真实任务或产物上做前向测试。这样做时，应传递被验证的产物，而不是你对问题的诊断；prompt 要足够通用，使成功依赖可迁移推理，而不是隐藏答案。

#### 从可复用 Skill 内容开始

开始实施时，先处理上面识别出的可复用资源：`scripts/`、`references/` 和 `assets/` 文件。注意，这一步可能需要用户输入。例如实现 `brand-guidelines` Skill 时，用户可能需要提供要存放到 `assets/` 的品牌素材或模板，或要存放到 `references/` 的文档。

新增脚本必须实际运行测试，确保没有 bug，且输出符合预期。如果有许多相似脚本，只需测试有代表性的样本，在完成时间和可信度之间取得平衡。

如果使用了 `--examples`，删除 Skill 不需要的占位文件。只创建真正需要的资源目录。

#### 更新 SKILL.md

写作指南：始终使用祈使句或不定式表达。

##### Frontmatter

编写包含 `name` 和 `description` 的 YAML frontmatter：

- `name`：Skill 名称。
- `description`：这是 Skill 的主要触发机制，帮助 Codex 理解何时使用该 Skill。
  - 同时包含 Skill 做什么，以及具体的触发条件或上下文。
  - 所有“什么时候使用”的信息都写在这里，而不是正文里。正文只有在触发后才会加载，因此正文中的 “When to Use This Skill” 小节对 Codex 的触发判断没有帮助。
  - `docx` Skill 的 description 示例：“Comprehensive document creation, editing, and analysis with support for tracked changes, comments, formatting preservation, and text extraction. Use when Codex needs to work with professional documents (.docx files) for: (1) Creating new documents, (2) Modifying or editing content, (3) Working with tracked changes, (4) Adding comments, or any other document tasks”

YAML frontmatter 不要包含其他字段。

##### Body

编写如何使用该 Skill 及其打包资源的说明。

### 第 5 步：验证 Skill

Skill 开发完成后，验证 Skill 文件夹，尽早发现基础问题：

```bash
scripts/quick_validate.py <path/to/skill-folder>
```

验证脚本会检查 YAML frontmatter 格式、必需字段和命名规则。如果验证失败，修复报告的问题后再次运行命令。

### 第 6 步：迭代

测试 Skill 后，你可能发现 Skill 足够复杂，需要前向测试；用户也可能要求改进。

用户测试通常发生在刚使用完 Skill 后，此时对 Skill 表现的上下文还很新。

前向测试和迭代工作流：

1. 在真实任务中使用 Skill。
2. 观察困难或低效之处。
3. 判断应如何更新 SKILL.md 或打包资源。
4. 实施修改并再次测试。
5. 如果合理且合适，进行前向测试。

## 前向测试

前向测试是启动子 Agent，用最小上下文对 Skill 做压力测试。

子 Agent 不应知道自己是在测试 Skill。应把它当作一个被用户要求完成任务的 Agent。给子 Agent 的 prompt 应类似：

`Use $skill-x at /path/to/skill-x to solve problem y`

而不是：

`Review the skill at /path/to/skill-x; pretend a user asks you to...`

前向测试决策规则：

- 倾向于进行前向测试。
- 如果你认为前向测试存在以下风险，应先请求批准：
  - 会花很长时间；
  - 需要用户提供额外批准；
  - 会修改线上生产系统。

在这些情况下，向用户展示你拟定的 prompt，并请求两点：是否同意，以及是否有建议修改。

前向测试注意事项：

- 对独立测试使用新的线程。
- 像真实用户一样传递 Skill 和请求。
- 传递原始产物，而不是你的结论。
- 避免展示预期答案或计划中的修复。
- 每轮迭代后从源产物重建上下文。
- 审阅子 Agent 的输出、推理和产物。
- 避免让 Agent 在迭代之间能发现磁盘上的残留产物；清理子 Agent 产物，避免额外污染。

如果前向测试只有在子 Agent 看到泄露上下文时才成功，应先收紧 Skill 或前向测试设置，再信任结果。
"""
