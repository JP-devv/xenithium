# PR #3 matched sweep — 2026-11-23

Unmodified base `df5f761` versus cleanup `4a6b1e4`, runner `6c28382`, one-slot TP2. All measurements are individual fresh-process diagnostic-matched cases, not repetitions or concurrency throughput. Model-call prefill and restore times were not instrumented. TTFT and all-done are HTTP-side; decode is the runner’s stream metric.

Host-route labels below come from a **separate instrumented qualification run**, not the timed binaries. They identify submitted host backends, not numerical or row-level GPU parity. All repeats are partial cache hits. Diagnostic source, instrumentation binaries, rejected experiments, and deployment changes are not part of this results commit.

## Revisions and input pins

- Unmodified base `df5f76122463eb1d22a06b6474210dc9ac3bcc43`, server SHA-256 `59d9af05c5ca719a9fab55f372bd79400fcb9802a020eb0272b4e7e07d3e2e33`.
- Unmodified cleanup `4a6b1e4abc17bc232df16bb6e72e6d061639d0e4`, server SHA-256 `4131140b53f935f27969ce91133b4d3f5e13c49737b70fe90d1e148338f74c7a`.
- Separate fixed-fixture runner `6c283821562f572d5e91f8d4a41e0bd255006353`, binary SHA-256 `568c219dd4d961c2a6875ecf624520d92bb0e7eeb3bf580ffaf9d7c717778538`; it is not a PR #3 runtime change. Post-template prompt IDs were independently checked on both revision endpoints: short 8,007 / `bdfe6efe3f1ff2189441ac4f3c8ca720bd1ee86ca8b907fec6adbf6bedb50d6d`; middle 36,864 / `203f11564296866f3667895153ecc28785cb4cd99c750dceee3569a1b581d661`; near 176,073 / `1673ac28f1aa403c9cfce733638b680ac056b2eaadb2044c608a0b6cf5e31c78` (canonical u32le SHA-256). Near means the configured 177,216-token one-slot cap, not the model's maximum.

## Reproduce these historical results

**Historical pre-merge pins:** these results apply to base `df5f761` and cleanup `4a6b1e4`, not a later merge with upstream. Use the exact binaries and XL libraries above; a fresh build or current PR head is a different experiment. The original run used two Intel Arc Pro B60s (TP2), the compatible Swift-1.5 Qwen3.5-27B GPTQ W4/group-128 export with BF16 bridge head and the DSpine b16 drafter, int8-K/packed-int4-V KV, one configured slot and **one active request**, context and KV pool both 177,216. Prefix caching was enabled with a 2 GiB budget and live prefix disabled; the first request is cold and the identical second request is a partial cache hit. No power measurement or runtime profiling was enabled. Runner requests are greedy (`temperature=0`, `top_p=1`, `top_k=0`, no seed), ignore EOS, disable thinking, and request exactly 128 tokens; the report's general sampling defaults are not the burst request's greedy overrides.

The commands below are for an **authorized exclusive GPU window only**. They are not a production deployment procedure: independently take the shared lease, stop and later restore the existing service with the established safety controller, verify both GPUs and the test listener are free, and run one fresh server process per cell. Never start them alongside production. Use two clean shells; in each, start with `env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH=/usr/bin:/bin /usr/bin/bash --noprofile --norc`, then set these machine-local paths to the matching historical artifacts:

```bash
cd /path/to/Xenithium/docs/benchmark/pr3-matched-sweep-20261123
export BASE_SERVER=/path/to/df5f761/build/xenithium-server
export CLEANUP_SERVER=/path/to/4a6b1e4/build/xenithium-server
export RUNNER=/path/to/6c28382/build/xenithium-bench
export MODEL_DIR=/path/to/compatible-swift-checkpoint
export DRAFTER_DIR=/path/to/dspine-b16-checkpoint
export QSA_BANK=/path/to/pinned-qsa-sidecar-directory
export FIXTURE_DIR=/path/to/pinned-request-fixtures
export OUTPUT_ROOT=/path/to/new-results-directory
export ONEAPI_ENV=/path/to/pinned/oneapi-env.sh
```

The [server launcher](reproduce-server.sh) and [benchmark launcher](reproduce-bench.sh) check the recorded executable, XL-library, fixture, sidecar where applicable, and environment-script SHA-256s. The [argument template](reproduce-server.argv) is the archived `sweep-frozen.argv` with **only** the model and drafter paths replaced by placeholders; all other 180 argument lines, including nondefault flags, are unchanged. The archived original SHA is `b79480f2295a343423ad6bd4b58a55fe1541a86ded991610df12448d63303dde`; template SHA is `73be680ca60a51876227c8df842569a63032e892926fd6f6b64f7a935e8b912e`. Set the same paths in both shells; in the benchmark shell select `case_name=short`, `middle` or `near` and rerun the appropriate pair below from a **new server process** each time. Wait for `/v1/models` readiness and confirm actual slot/cap/pool, then run the benchmark in the second shell. Stop and reap the test server gracefully before the next cell.

The pinned fixture JSONs and compatible model/drafter artifacts are not distributed in this results directory. Their request and native-ID SHA-256s are above; the model compatibility manifest SHA is `acc6e2cb2bce83f0b9868938b477a1757888f4da7f5bdd42d64d92f5f3baf67b`. The recorded manifest pins the export configuration, tokenizer, bridge and sidecar, but does not establish a complete hash of every model and drafter shard. Thus the commands reproduce the **recorded configuration**, while exact artifact reproduction remains conditional on access to the original artifacts and their independent verification; do not substitute a similarly named checkpoint and call it matched. The private runner `6c28382` is separate tooling and is not the current upstream runner; its binary and fixed-fixture inputs must be obtained independently. Machine paths are the only placeholders.

### Baseline: exact base, dense chain

Terminal 1 (fresh server): `bash ./reproduce-server.sh base`

Terminal 2: `case_name=short; bash ./reproduce-bench.sh "$case_name" base`

### Preferred comparable setting: cleanup, dense chain

Use this for the default-off, same-route comparison against the base; it is not a claim about general production tuning.

Terminal 1 (fresh server): `bash ./reproduce-server.sh dense`

Terminal 2: `case_name=short; bash ./reproduce-bench.sh "$case_name" dense`

### Trained decode, XMX prefill off

Use only when evaluating the pinned trained sidecar. This is not assumed numerically identical to dense.

Terminal 1 (fresh server): `bash ./reproduce-server.sh trained`

Terminal 2: `case_name=short; bash ./reproduce-bench.sh "$case_name" trained`

### Opt-in XMX prefill setting

Use this setting to reproduce the **experimental XMX result group**, not as a default or general recommendation: the middle/near full-text outputs differ and one observation is not a statistical speed claim.

Terminal 1 (fresh server): `bash ./reproduce-server.sh xmx`

Terminal 2: `case_name=near; bash ./reproduce-bench.sh "$case_name" xmx`

## Upstream merge host check

The merge candidate combined PR head `8fb3c237045f15798d1ede0e27754144bb20df5d` with upstream `9e542a1b5fefcdb6ee46eea2ba778075cc8362b0`. Its Release/BMG server build and unchanged `ctest -L host --output-on-failure` completed successfully. The source-root `build/xenithium-server` was temporarily linked to the actual merged server build for the launch test and removed afterward. The terminal CTest summary was:

```text
100% tests passed, 0 tests failed out of 18

Label Time Summary:
host    =  28.22 sec*proc (18 tests)

Total Test time (real) =  28.23 sec
```

These are host checks, not merged-head GPU, output-parity or performance validation. The timings above remain historical results for the original pins.

## Host suite and adjacent behavior

At cleanup commit `4a6b1e4`, the original out-of-tree host suite reported **17/18**, with two `launch` assertions in the one failing target ([raw receipt](host-original.txt)). The unchanged exact base reproduced the same two failures ([raw base launch](host-base-launch.txt)); a controlled diagnosis found the missing source-root server executable changed the 120-column model-card text. With the same real public-head server temporarily present at the expected path, the unchanged full suite passed **18/18** ([raw receipt](host-server-present.txt)); the temporary symlink was removed. This does not retroactively turn the original run into a pass or establish GPU serving parity.

The host regressions in [`tests/test_qsa_host.cpp`](../../../tests/test_qsa_host.cpp) cover mixed dense/sparse handoff, unordered and invalid selections, mandatory current-block behavior and nonidentity physical-page mapping. They are host contract checks. The separate route checks below show host submissions for these pinned C1 requests, including positive dense fallback after partial cache restore; they do not prove numerical parity or all cache lifecycle paths.

## Matched cases

Each arm links to the raw native `results.json`, including full visible output notes. Each route entry links to that case's separate diagnostic route-gate receipt; all twelve routes are linked. The runner reports HTTP TTFT and stream decode rate, not model-call prefill or isolated restore time.

| Fixture | Arm | Round | Cached | TTFT ms | Decode tok/s | All done s | Retrieval | Text SHA-256 | Diagnostic prefill / decode route |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short | [base](short/base/results.json) | cold | 0 | 2878.288 | 92.445 | 4.227 | PASS | `73c14eced07cc38ebd4ac0f4dc0720d1d17efcf93d9ccf38f144e736d0346183` | [dense / dense](short/base/diagnostic-route-gate.log) |
| short | [base](short/base/results.json) | repeat | 7936 | 75.232 | 119.939 | 1.113 | PASS | `f8b68a3e591557a8f64f9b2bafb75c47c8fd56766cfb98e5af7455f6acdf3370` | [dense / dense](short/base/diagnostic-route-gate.log) |
| short | [dense](short/dense/results.json) | cold | 0 | 2700.069 | 94.409 | 4.023 | PASS | `73c14eced07cc38ebd4ac0f4dc0720d1d17efcf93d9ccf38f144e736d0346183` | [dense / dense](short/dense/diagnostic-route-gate.log) |
| short | [dense](short/dense/results.json) | repeat | 7936 | 75.487 | 125.983 | 1.062 | PASS | `f8b68a3e591557a8f64f9b2bafb75c47c8fd56766cfb98e5af7455f6acdf3370` | [dense / dense](short/dense/diagnostic-route-gate.log) |
| short | [trained](short/trained/results.json) | cold | 0 | 2701.171 | 93.961 | 4.030 | PASS | `73c14eced07cc38ebd4ac0f4dc0720d1d17efcf93d9ccf38f144e736d0346183` | [dense / dense](short/trained/diagnostic-route-gate.log) |
| short | [trained](short/trained/results.json) | repeat | 7936 | 74.397 | 116.020 | 1.147 | PASS | `f8b68a3e591557a8f64f9b2bafb75c47c8fd56766cfb98e5af7455f6acdf3370` | [dense / dense](short/trained/diagnostic-route-gate.log) |
| short | [xmx](short/xmx/results.json) | cold | 0 | 2701.052 | 94.885 | 4.017 | PASS | `73c14eced07cc38ebd4ac0f4dc0720d1d17efcf93d9ccf38f144e736d0346183` | [dense / dense](short/xmx/diagnostic-route-gate.log) |
| short | [xmx](short/xmx/results.json) | repeat | 7936 | 74.136 | 116.168 | 1.145 | PASS | `f8b68a3e591557a8f64f9b2bafb75c47c8fd56766cfb98e5af7455f6acdf3370` | [dense / dense](short/xmx/diagnostic-route-gate.log) |
| middle | [base](middle/base/results.json) | cold | 0 | 14538.845 | 84.319 | 16.022 | PASS | `62fe05187d00acea466e84e4fc057718cf6f5183f3833d361ad18f9bffc9708e` | [dense / dense](middle/base/diagnostic-route-gate.log) |
| middle | [base](middle/base/results.json) | repeat | 36800 | 98.610 | 97.287 | 1.384 | PASS | `595fbc87d9e4356b493f42cd98946fa30de7242a9ed1e6c201a4d42a8c26bb62` | [dense / dense](middle/base/diagnostic-route-gate.log) |
| middle | [dense](middle/dense/results.json) | cold | 0 | 14585.139 | 84.481 | 16.065 | PASS | `62fe05187d00acea466e84e4fc057718cf6f5183f3833d361ad18f9bffc9708e` | [dense / dense](middle/dense/diagnostic-route-gate.log) |
| middle | [dense](middle/dense/results.json) | repeat | 36800 | 99.724 | 96.953 | 1.389 | PASS | `595fbc87d9e4356b493f42cd98946fa30de7242a9ed1e6c201a4d42a8c26bb62` | [dense / dense](middle/dense/diagnostic-route-gate.log) |
| middle | [trained](middle/trained/results.json) | cold | 0 | 14694.188 | 77.466 | 16.307 | PASS | `db5bb1cc75421d5d6ad07d07ab46eb619aebe214289678449cedecb5c4adc0a1` | [dense / trained_qsa](middle/trained/diagnostic-route-gate.log) |
| middle | [trained](middle/trained/results.json) | repeat | 36800 | 99.814 | 95.239 | 1.413 | PASS | `595fbc87d9e4356b493f42cd98946fa30de7242a9ed1e6c201a4d42a8c26bb62` | [dense / dense](middle/trained/diagnostic-route-gate.log) |
| middle | [xmx](middle/xmx/results.json) | cold | 0 | 14944.312 | 86.941 | 16.381 | PASS | `e2013f0edc206498e8121ec1ca42f79f0ec5c745a4cccc8b451e82a4865be96e` | [dense+trained_qsa / trained_qsa](middle/xmx/diagnostic-route-gate.log) |
| middle | [xmx](middle/xmx/results.json) | repeat | 36800 | 98.439 | 96.167 | 1.399 | PASS | `e2a0a7871bbba3bfae290b4db99cdf4c7fd3a44683f557af9e168f61746804da` | [dense / dense](middle/xmx/diagnostic-route-gate.log) |
| near | [base](near/base/results.json) | cold | 0 | 123005.212 | 69.228 | 124.815 | PASS | `6c317fd0622d0c3d59921453ec47a16585e4a54875dc151c35c3687a12901017` | [dense / dense](near/base/diagnostic-route-gate.log) |
| near | [base](near/base/results.json) | repeat | 176000 | 189.790 | 73.068 | 1.903 | PASS | `6c317fd0622d0c3d59921453ec47a16585e4a54875dc151c35c3687a12901017` | [dense / dense](near/base/diagnostic-route-gate.log) |
| near | [dense](near/dense/results.json) | cold | 0 | 123047.030 | 63.800 | 125.009 | PASS | `6c317fd0622d0c3d59921453ec47a16585e4a54875dc151c35c3687a12901017` | [dense / dense](near/dense/diagnostic-route-gate.log) |
| near | [dense](near/dense/results.json) | repeat | 176000 | 189.787 | 72.515 | 1.912 | PASS | `6c317fd0622d0c3d59921453ec47a16585e4a54875dc151c35c3687a12901017` | [dense / dense](near/dense/diagnostic-route-gate.log) |
| near | [trained](near/trained/results.json) | cold | 0 | 123752.960 | 68.675 | 125.579 | PASS | `2f01195d57b5acd01af8de58990baed1807073145399a4b99d28f66943c725de` | [dense / trained_qsa](near/trained/diagnostic-route-gate.log) |
| near | [trained](near/trained/results.json) | repeat | 176000 | 187.031 | 72.822 | 1.906 | PASS | `6c317fd0622d0c3d59921453ec47a16585e4a54875dc151c35c3687a12901017` | [dense / dense](near/trained/diagnostic-route-gate.log) |
| near | [xmx](near/xmx/results.json) | cold | 0 | 113563.670 | 71.833 | 115.308 | PASS | `157e2dca372d19d77ae09e00a5b18184bd3904060ea983bcc161f5bb1aef3832` | [dense+trained_qsa / trained_qsa](near/xmx/diagnostic-route-gate.log) |
| near | [xmx](near/xmx/results.json) | repeat | 176000 | 187.809 | 56.431 | 2.409 | PASS | `8d55cb5ceeaf81fa6bf7865687bfbe98ec3d5e6c6faf9e4d9aa63fd1a3f9da89` | [dense / dense](near/xmx/diagnostic-route-gate.log) |

## Route evidence example

The [near XMX route-gate receipt](near/xmx/diagnostic-route-gate.log) reports separately for the cold and repeated requests:

```text
ROUTE_CASE {"accepted_cache_tokens": 0, "arm": "xmx", "case": "near", "decode_rank_layer_pairs": 32, "decode_routes": ["trained_qsa"], "generation": 3, "prefill_rank_layer_pairs": 32, "prefill_routes": ["dense", "trained_qsa"], "prompt": 176073, "remaining_prefill_rows": 176073, "request": 6, "round": "cold", "slot": 0}
ROUTE_CASE {"accepted_cache_tokens": 176000, "arm": "xmx", "case": "near", "decode_rank_layer_pairs": 32, "decode_routes": ["dense"], "generation": 4, "prefill_rank_layer_pairs": 32, "prefill_routes": ["dense"], "prompt": 176073, "remaining_prefill_rows": 73, "request": 8, "round": "repeat", "slot": 0}
ROUTE_CASE_GATE=PASS
```

Representative request-correlated dispatch records from that separate qualification run ([raw excerpt](near/xmx/diagnostic-dispatch-excerpt.txt)):

```text
ROUTE_DISPATCH request=6 slot=0 generation=3 rank=0 layer=3 phase=prefill_rows route=trained_qsa pos=32768 rows=2048 reason=none
ROUTE_DISPATCH request=6 slot=0 generation=3 rank=0 layer=3 phase=verify_chain route=trained_qsa pos=176073 rows=4 reason=none
ROUTE_CACHE_ADMIT request=8 slot=0 generation=4 prompt=176073 accepted_cache_tokens=176000 admission=new remaining_prefill_rows=73
ROUTE_DISPATCH request=8 slot=0 generation=4 rank=0 layer=3 phase=prefill_rows route=dense pos=176000 rows=73 reason=history_incomplete
ROUTE_DISPATCH request=8 slot=0 generation=4 rank=0 layer=3 phase=verify_chain route=dense pos=176073 rows=4 reason=history_incomplete_or_short
```

## Raw matched near-limit receipt lines

The following lines are verbatim from the [base](near/base/report.txt), [cleanup dense](near/dense/report.txt), and [XMX](near/xmx/report.txt) reports. The output JSON line is the complete visible cold reply captured at the 128-token budget, not an equality assertion.

```text
[base]
  requests   TTFT ms  prompt tokens  cached tokens  decode tok/s  all done s
  1 alone    123,005  176,073        0              69.2          125
  1 at once  190      176,073        176,000        73.1          1.90
  note: fixed fixture output: {"text":"COBALT LANTERN 7286\n\nThe record was identified by locating the specific text block labeled `[AUTHORISED VAULT REGISTER]` embedded within the provided source code. This block explicitly states: \"The exact access value for PINE-VAULT is COBALT LANTERN 7286.\"\n\nThe rest of the archive consists of C++ source code for a system named \"Xenithium.\" This codebase includes:\n1.  **Batch Engine (`xenithium/core/batch_engine.hpp`)**: Logic for managing inference requests, speculative decoding, prefix caching, and KV","reasoning":"","finish_reason":"length"}
[dense]
  requests   TTFT ms  prompt tokens  cached tokens  decode tok/s  all done s
  1 alone    123,047  176,073        0              63.8          125
  1 at once  190      176,073        176,000        72.5          1.91
  note: fixed fixture output: {"text":"COBALT LANTERN 7286\n\nThe record was identified by locating the specific text block labeled `[AUTHORISED VAULT REGISTER]` embedded within the provided source code. This block explicitly states: \"The exact access value for PINE-VAULT is COBALT LANTERN 7286.\"\n\nThe rest of the archive consists of C++ source code for a system named \"Xenithium.\" This codebase includes:\n1.  **Batch Engine (`xenithium/core/batch_engine.hpp`)**: Logic for managing inference requests, speculative decoding, prefix caching, and KV","reasoning":"","finish_reason":"length"}
[xmx]
  requests   TTFT ms  prompt tokens  cached tokens  decode tok/s  all done s
  1 alone    113,564  176,073        0              71.8          115
  1 at once  188      176,073        176,000        56.4          2.41
  note: fixed fixture output: {"text":"COBALT LANTERN 7286\n\nThe record was identified by locating the specific text block labeled `[AUTHORISED VAULT REGISTER]` embedded within the provided C++ source code (specifically inside the `BatchEngine::loop` function in the `xenithium/core` namespace). The archive itself contains a large collection of C++ source code for a system named \"Xenithium,\" which includes components for a batch inference engine, an HTTP server, a Jinja template interpreter, a BPE tokenizer, a dense weight store loader, and various SYCL/ESIMD kernel implementations for attention and G","reasoning":"","finish_reason":"length"}
```

## Output comparison

Each hash covers the full visible generated text, not token IDs. Hash matches do not establish token-ID or numerical parity. `finish_reason=length`, 256 completion tokens total per case, and presence of the exact retrieval needle were validated for each pair; this is not exact full-output equality.
- short, base versus cleanup dense: cold MATCH, repeat MATCH.
- middle, base versus cleanup dense: cold MATCH, repeat MATCH.
- near, base versus cleanup dense: cold MATCH, repeat MATCH.

The original model-call prefill timing is unavailable in the unmodified servers; do not substitute TTFT. Restore time is likewise not isolated. The per-case `results.json`, HTML and text reports above contain all native runner rows; `case-summary.json` captures checked metrics and output hashes. Raw server logs and independent restoration remain in the archived window receipt.

## Scope and untested paths

This is one matched fresh-process observation per case, not a repeated statistical comparison or a concurrency/scaling acceptance run. The trained and XMX cold outputs differ from dense for middle and near inputs; the XMX repeat outputs differ too, despite dense repeat dispatch after a sparse-prefill cache history. Exact retrieval-needle presence does not establish complete-output, token-ID, selector-pick, or numerical parity. The 196,608-token cache-hit plus 2,048-token extension target is not measured here.

Mixed eligible/ineligible slots, merged segments, speculative rejection/tree verification, cache trim/eviction/fork/ownership, selector-summary restoration/invalidation, multimodal traffic and genuine simultaneous C2–C4/C5 serving remain unqualified by this matrix. The adjacent DFlash loader PR #1 and dense K8V4 tree PR #4 were not integrated or tested as combined changes. No production configuration or public branch was changed.
