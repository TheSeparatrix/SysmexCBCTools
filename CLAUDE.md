# CLAUDE.md

## API design

Public APIs follow [BloodCounts! ADR 0001](https://github.com/bloodcounts/.github/blob/main/docs/adr/0001-apis-unified-and-shaped-like-familiar-libraries.md): the same names and call shapes as the other BloodCounts! packages, and the conventions of the library users already know for the job (scikit-learn for transformers, pandas for data).

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (TheSeparatrix/SysmexCBCTools) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
