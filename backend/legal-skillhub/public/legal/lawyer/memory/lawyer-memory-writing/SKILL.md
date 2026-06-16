---
name: lawyer-memory-writing
description: 在需要更新律师案件长期记忆 memory.yaml 时使用，指导你按字段级 JSON operations 更新案件摘要、证据、法律分析和客户画像。
---



# 固定字段

只能操作以下末级字段：

```text
case_facts.case_summary
case_facts.evidence_ledger
legal_analysis.legal_frame
legal_analysis.dispute_focus
client_brief.client_profile
client_brief.client_demand_list
```

# 调用格式

调用 `save_lawyer_memory` 时只提交需要修改的字段级 JSON operations

每个 operation 只能包含三个值：

```json
{
  "field": "case_facts.case_summary",
  "operation": "revise",
  "content": "要覆盖或追加的具体文本"
}
```

完整示例：

```json
{
  "operations": [
    {
      "field": "case_facts.case_summary",
      "operation": "revise",
      "content": "案件背景：...\n\n关键事实经过：...\n\n当前程序进展：..."
    },
    {
      "field": "case_facts.evidence_ledger",
      "operation": "expand",
      "content": "新增证据：二审提交沟通记录截图，用于证明对方参与过还款协商。"
    }
  ]
}
```

# 操作类型

- `revise`
  新信息与旧判断存在冲突，或需要把旧信息精化得更具体、更准确、更紧凑。
  工具会用 `content` 覆盖该字段旧值。

- `expand`
  旧判断仍成立，只是在原字段基础上新增条目、新细节、新证据或新进展。
  工具会把 `content` 追加到该字段旧值之后

# 字段写法

- `case_facts.case_summary`
  全案压缩摘要，覆盖案件背景与关键事实经过、当前案件程序进展/最新结论。

- `case_facts.evidence_ledger`
  证据台账。写证据内容、来源、证明对象、证明力、对应争点。

- `legal_analysis.legal_frame`
  法律关系定性、请求权基础、适用法条方向、抗辩框架。

- `legal_analysis.dispute_focus`
  当前核心争议点，以及哪些点已清楚、哪些仍待查明或补证。

- `client_brief.client_profile`
  客户画像：沟通方式、配合度、风险偏好、诉求强度、调解态度等。

- `client_brief.client_demand_list`
  当事人当前明确提出的诉求、上诉请求或答辩意见列表。

# 文书后写入规则

- CD/DD/AD/AR 文书起草结束后，`client_brief.client_demand_list` 和 `case_facts.evidence_ledger` 必须严格依据刚起草完成的文书正文，这时候使用`revise`。
- 如果文书正文与 LC 咨询、旧记忆或客户临场表达冲突，以刚完成的文书正文为准。
- AD/AR 阶段的 `case_facts.evidence_ledger` 只写二审阶段获取到的新证据；没有二审新证据时，对该字段使用 `revise` 写入“无二审新证据”。
- 不得把文书没有列明的诉求、答辩意见或证据补写进记忆。

# 写入原则

- 只提交要改的字段，没变化的字段不要提交。
- 如果当前字段为空则需要进行操作。
- `content` 必须是纯文本字符串，可以包含换行。
- 不得编造事实、证据、日期、金额、程序结论。
- 不确定的内容不要写成确定事实。
