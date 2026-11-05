# Per-task progress records

One file per task, `<ID>.md`, using the template in
[INSTRUCTIONS.md](../INSTRUCTIONS.md) §8.

- An agent creates its task's file as the first commit on the task branch and
  pushes it at once. The pushed branch is the claim that other sessions see.
- The file reaches `main` only when the human merges the PR. A file present on
  `main` therefore means the task is merged; there is no separate MERGED status.
- Files are never deleted. Together they are the history of the work.
- The human harvests `proposed follow-ups` from newly merged files into
  [QUEUE.md](../QUEUE.md) during the weekly review.
