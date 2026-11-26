# Trained K8V4 QSA: change map and validation

This note covers the default-off trained decode and optional XMX-prefill changes in PR #3. It is a review map, not evidence that all models, cache states, or throughput targets are qualified. The sidecar and launch requirements are in [trained K8V4 QSA](trained-k8v4-qsa.md) and [XMX prefill](trained-qsa-xmx-prefill.md).

## Why the changes belong together

* `trained_qsa.hpp/.cpp` validate and load the trained indexer weights separately from the base checkpoint; `trained_qsa_device.hpp/.cpp` maintain per-layer selector history and turn each row's query into a causal top-512 plan. Neither changes the KV encoding.
* `trained_qsa_pages.hpp` converts logical block IDs into ascending logical pages/masks, adds the current block once, and retains dense fallback for an all-visible row. The actual physical page is resolved by the serving slot's page table, not by the trained selector.
* `model.hpp/.cpp`, `step_model.hpp`, `batch_engine.hpp/.cpp` and `sources.cmake` connect the indexer to model/slot lifecycle. The coverage guard keeps restored or incomplete selector history on dense attention even when target KV is reusable. Only accepted rows publish history; unsupported or mixed layouts retain stock dispatch.
* `qsa.hpp/.cpp` and `gqa.hpp`, `gqa.cpp`, `gqa_seg.hpp/.cpp`, `gqa_fa.cpp` provide the trained selector and masked K8V4 page walk. XMX prefill packs BF16 operands and shares a group page union, but each row retains its own causal mask. It can change FP32 scores and selections; default dense prefill remains unchanged unless explicitly enabled.
* `server-env.md`, `trained-k8v4-qsa.md`, and `trained-qsa-xmx-prefill.md` document defaults, flag interactions, sidecar provenance, dense fallback and precision limits. `test_qsa_host.cpp` checks the host page handoff without claiming device or model parity.

This PR does **not** add native K8V4 tree verification (PR #4), a DFlash drafter loader (PR #1), private benchmark release barriers, cross-slot resident-summary copies, asynchronous profiling, or deployment changes. PR #4 also touches GQA files: independent branch reviews are not a substitute for a combined integration test.

## Regression coverage and boundaries

The `qsa-host` test covers the all-visible dense handoff, mandatory current-block insertion, invalid/future/duplicate selections, nonidentity page tables and the mixed dense/sparse per-row/per-head plan. `server` tests protect adjacent serving behavior. Run the repository's unchanged host suite using `ctest --test-dir build -L host --output-on-failure` after building host targets, with the appropriate oneAPI environment.

Before changing defaults or making a general performance claim, also test on device: exact ordered causal IDs, masks and nonidentity physical pages; mixed eligible/ineligible slots; partial acceptance and rejected branches; cache capture, restoration, trim, eviction and ownership; and dense fallback for missing/incomplete history, trees and unsupported configurations. Qualify XMX/FP16 precision separately from FP32 and compare complete outputs, not only retrieval. The current public XMX path is limited to one slot and requires contiguous selector coverage; target KV hits alone are not proof of that coverage.

## Results provenance

The PR description and its linked comments distinguish the exact public-head BMG build/host tests and 36,864-ID XMX dispatch/retrieval smoke from separate downstream long-context experiments. The public smoke retrieved the expected answer but the full texts differed; it is not a token-parity or general speed result. The versioned sidecar has its own SHA-256 and license information; it does not fix missing GPTQ metadata or supply a BF16 bridge head in an incompatible base checkpoint. Capture overhead and diagnostic timing are not stock critical-path measurements.

Missing sweeps include repeated uninstrumented short/mid/near-limit C1 prefill and decode, genuine simultaneous C2–C4 decode, and the above cache/state/fallback matrix. Keep incremental-prefill and short-to-long scaling goals separate from throughput claims and preserve the user's benchmark methodology.
