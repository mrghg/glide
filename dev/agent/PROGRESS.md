# Progress

_The durable state shared between agent sessions and the human operator.
Sessions read it first and write to it last. Entries follow the template in
[INSTRUCTIONS.md](INSTRUCTIONS.md) §8. Keep it current; prune MERGED entries
into the "History" section monthly._

## Current state

- Physics tag: none yet (pre-freeze).
- Golden fixtures: none yet (task A0).
- Benchmark baseline: none yet (task A0).
- Active allocations: none.

## Data inventory

_Filled by the human operator. Every cube the tasks may use, with its path.
Agents do not download data._

| Path | Domain | Period | Levels | Cadence | Size | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `<<PLACEHOLDER>>` | | | | | | smoke |
| `<<PLACEHOLDER>>` | | | | | | reference multi-site |

Restricted reference data (never in the repo): FLEXPART outputs
`<<PLACEHOLDER>>`; NAME footprints `<<PLACEHOLDER>>`; EDGAR `<<PLACEHOLDER>>`;
ETEX `<<PLACEHOLDER>>`; FLEXPART v11 source `<<PLACEHOLDER>>`.

## Open BLOCKED items

_None._

## Task log

_One entry per task, newest first. Status: CLAIMED | PR OPEN #n | BLOCKED |
REWORK | MERGED._

## Proposed tasks

_Agents append proposals here (follow-ups discovered during a task). The human
moves accepted ones into QUEUE.md._

## History

_MERGED entries moved here monthly._
