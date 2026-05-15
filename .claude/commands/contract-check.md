---
description: Audit current code against the SSOT §6 interface contracts
---

Audit the repository's current code against the interface contracts in
`docs/SSOT.md` §6 (Contracts A–F).

For each contract, compare the implemented or stubbed public signature in
`src/` against the contract: function/class names, argument names and order,
type hints, and return shape. For Contract A, check the `corpus.parquet`
schema instead.

Report every mismatch as a list with `file:line` references and the exact
discrepancy. This is a read-only review — do not modify any code.
