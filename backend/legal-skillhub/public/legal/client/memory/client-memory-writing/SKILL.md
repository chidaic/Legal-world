---
name: client-memory-writing
description: 在需要更新当事人案件长期记忆 memory.yaml 时使用，指导你按字段级 JSON operations 更新案件自述、推进进展和核心诉求。
---



# 固定字段

只能操作以下末级字段：

```text
case_knowledge.self_narrative
case_knowledge.case_stage
demands.core_demands
```

# 调用格式

调用 `save_client_memory` 时只提交需要修改的字段级 JSON operations，不要提交整份 YAML。

每个 operation 只能包含三个值：

```json
{
  "field": "case_knowledge.case_stage",
  "operation": "expand",
  "content": "已完成起诉状起草，准备进入立案。"
}
```

完整示例：

```json
{
  "operations": [
    {
      "field": "case_knowledge.case_stage",
      "operation": "expand",
      "content": "已完成法律咨询，下一步准备整理借条和银行流水。"
    },
    {
      "field": "demands.core_demands",
      "operation": "revise",
      "content": "核心诉求是尽快追回本金；如对方一次性支付主要本金，可以考虑调解。"
    }
  ]
}
```

# 操作类型

- `revise`
  新信息与旧判断存在冲突，或需要把旧信息精化得更具体、更准确、更紧凑。
  工具会用 `content` 覆盖该字段旧值。

- `expand`
  旧判断仍成立，只是在原字段基础上新增条目、新细节或新进展。
  工具会把 `content` 追加到该字段旧值之后

# 字段写法

- `case_knowledge.self_narrative`
  当事人自己眼里事情是怎么发生的。案件背景、完整经过、事实叙事；不写和律师推进到了哪一步。

- `case_knowledge.case_stage`
  当前案件随着和律师推进到了什么状态。偏程序进展、当前卡点、已完成动作、下一步安排。

- `demands.core_demands`
  当事人当前核心诉求、底线、优先级、最想达到什么结果、最担心什么风险。

# 写入原则

- 只提交要改的字段，没变化的字段不要提交。
- 如果当前字段为空则需要进行操作。
- `content` 必须是纯文本字符串，可以包含换行。
- `self_narrative` 写案件经过，`case_stage` 写推进进展，不要混写。
- 不得编造事实、证据、日期、金额、程序结论。
- 不确定的内容不要写成确定事实。
