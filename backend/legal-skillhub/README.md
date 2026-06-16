# LEGALWORLD Public Skill Hub

This directory contains the public Skill instructions required by the LEGALWORLD backend runtime.

The Skill hub is intentionally limited to reusable procedural guidance. It should not contain generated case trajectories, private party data, model outputs, logs, evaluation results, or local deployment configuration.

## Layout

```text
public/
  legal/
    client/
      memory/
        client-memory-writing/
          SKILL.md
    lawyer/
      document-drafting/
        lawyer-complaint-drafting/
          SKILL.md
        lawyer-defense-drafting/
          SKILL.md
        lawyer-appeal-drafting/
          SKILL.md
        lawyer-appeal-response-drafting/
          SKILL.md
      memory/
        lawyer-memory-writing/
          SKILL.md
```

## Included Skills

- `client-memory-writing`: updates client-side long-term memory with field-level JSON operations.
- `lawyer-memory-writing`: updates lawyer-side case memory, evidence ledger, legal analysis, and client brief fields.
- `lawyer-complaint-drafting`: guides complaint drafting in the complaint-drafting stage.
- `lawyer-defense-drafting`: guides defense drafting in the defense-drafting stage.
- `lawyer-appeal-drafting`: guides appeal drafting in the appeal-drafting stage.
- `lawyer-appeal-response-drafting`: guides appeal response drafting in the appeal-response stage.

## Public-Release Rules

- Keep all files UTF-8 encoded.
- Use anonymized examples only.
- Do not add real party names, case numbers, addresses, phone numbers, IDs, model traces, or runtime outputs.
- Do not add `.env`, credentials, private API endpoints, or local absolute paths.
- Keep generated or learned private Skills outside this public directory unless they have been reviewed and anonymized.
