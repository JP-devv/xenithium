# PR #4: dense K8V4 tree verification — review and planned qualification

**No result is claimed for the planned matched GPU tests or sweep.** The original standalone PR head is `f0e84a98f71c7ee43782349e62e94cf4d3fa30ca`, based on `df5f76122463eb1d22a06b6474210dc9ac3bcc43`. This review adds an isolated partial-acceptance test and this documentation; it does not incorporate trained QSA or change PR #3. The original PR diff is six files, +214/−57. The test build includes the added regression and is not the unmodified timed server.

## Change and adjacent coverage

- The Qwen35 serving option permits dense tree verification with int8-K/packed-int4-V KV instead of disabling the tree solely for that KV format. Existing tree budget and dense-attention restrictions remain.
- `gqa_decode` dispatches the K8V4 tree through its masked group kernel, including exact and int8-fast variants. The KV move uses the actual K and packed-V row widths and copies all eight BF16 V scales. This is not a tree-plus-QSA feature.
- The existing `test_tree` cases compare tree ancestry/attention and output against same-format chain references for int8, BF16 and K8V4, including exact and selected fast numerics. `test_tree_v4_vs_int8` is a **tolerance-based**, cross-format int8-fast comparison, not bitwise parity. The broad GQA test also exercises decode, prefill rows and optional two-card paths.
- The added `--tree-v4-partial` regression uses accepted path `{0,3,4,5}` at position 62 in a permuted physical page pool. It compares accepted K, packed V and scales with the chain reference and checks untouched bytes and poisoned pages, in exact and int8-fast modes. It is compiled but **has not run on a device**. Both the broad test and the focused switch contain this added case in the test build.

## Evidence status

| Check | Status | Scope and limits |
| --- | --- | --- |
| Base and candidate Release/BMG GQA builds | **Completed offline, exit 0 each** | Base binary `1f80463f3cae2e53c0bcc723e49ed9fbf67a482a2c2e6e730396b66bfd467037`; test build `9d5a029da6a329dc7c2bf7705d2cef5a13100c7e0b3916a91968fee34cb04b82`. Compilation is not execution. |
| Test fixture source comparison | **Completed offline** | The matched base and candidate `test_rows`, `make_model`, `make_card` and `rand_x` bodies are byte-identical. The overall harnesses differ in tree tests and order; runtime inputs/state have not been proven identical. |
| Fixture, binaries, toolchain pins | **Completed offline** | Three request/ID files and their hashes, both test binaries/XL libraries and oneAPI 2025.3/BMG build settings were checked. Real base/head endpoint ID parity is **not** established. |
| Host-only CTest suite | **Not run for this review** | Do not infer a host test pass from the GQA build. |
| Base and candidate unchanged broad `gqa` CTest | **Pending GPU** | Run both and preserve full exits/logs; failures are failures even if they match. A previous unmatched GQA run failed in oneDNN SDPA prefill with `could not create logical_tensor with property`; a matched base/head comparison is still required. |
| Candidate-only `--tree-v4-partial` | **Pending GPU** | Require exit 0 and six positive assertions each in exact and int8-fast modes, including unchanged and poisoned bytes. |
| Endpoint IDs, route qualification and 12-case sweep | **Pending GPU** | No measured result, output-token parity, tree route, speedup, cache ownership or concurrency claim. |

The known authorized host has two B60s, both in use by production; no spare compatible device has been identified. No controller is armed. Use only a separately authorized exclusive window, actual shared lease, disconnect-safe managed controller, graceful shutdown and independently verified restoration. The earlier draft controller's forced-stop recovery and separate-owner takeover are **not qualified**. Do not execute the commands below on production or interpret them as authorization.

## Planned device regression commands

Set machine-local paths in an isolated environment with oneAPI 2025.3 and the matching BMG runtime. Both registered CTests must be run unchanged, separately, with stdout/stderr and exits retained. The candidate has the added test; the base does not support the focused switch.

```bash
export XEN_ONEAPI=/opt/intel/oneapi/2025.3
export ONEAPI_ENV=/path/to/pinned/oneapi-env.sh
export BASE_BUILD=/path/to/base-build
export HEAD_BUILD=/path/to/candidate-test-build
source "$ONEAPI_ENV"
ctest --test-dir "$BASE_BUILD" -R '^gqa$' --output-on-failure
ctest --test-dir "$HEAD_BUILD" -R '^gqa$' --output-on-failure
"$HEAD_BUILD/xenithium-test-gqa" --tree-v4-partial
```

For broad CTest, identical failures at the same oneDNN check are an observed matched failure, not a pass or root-cause diagnosis. Different earlier failures are inconclusive. On one visible GPU the broad test skips two-card cases; when more than one is visible even the focused test constructs a `Fabric` over the first two, so device isolation must be independently verified.

## Planned C1 matched sweep — commands, not results

The preferred candidate setting for testing this feature is **head/tree7**; use it only after endpoint-ID and positive route qualification, not as an unqualified production recommendation. The separate baseline is **base/tree0**. The additional controls **base/tree7** (expected unsupported tree fallback, which must be observed) and **head/tree0** separate fallback and tree effects. All four arms use a fresh server for each of `short`, `middle`, `near`, then one cold and one repeat request at one active user / one configured slot. TP2, two B60s, context and KV pool 177216, int8-v4 KV, prefill chunk 2048, DSpine b16, prefix cache 2 GiB, `prefix-live 0`, attention/prefill page selection 0 and 128-token fixed reply are pinned in the archived argv and runner. Swift-1.5 Qwen3.8-27B compatible GPTQ W4/group-128 with BF16 bridge is the planned model. No trained QSA sidecar, profiling or replay override is used.

In **each of two clean shells**, configure machine-local paths. The immutable `tree0.argv` and `tree7.argv` are external archived files; their original SHA-256s are checked by [reproduce-server.sh](reproduce-server.sh). It replaces only the two machine-local model paths in memory. Their respective SHA-256s are `b79480f2295a343423ad6bd4b58a55fe1541a86ded991610df12448d63303dde` and `b24e2a285589461da53fa342e2f9c82a55cfcba3cef977aa0b1b31e31cd6890d`. Obtain the pinned fixtures and binaries independently; they are not distributed in this package.

```bash
cd /path/to/Xenithium/docs/benchmark/pr4-dense-k8v4-tree
export BASE_SERVER=/path/to/base-server-build/xenithium-server
export HEAD_SERVER=/path/to/exact-pr4-head-server-build/xenithium-server
export RUNNER=/path/to/pinned-6c28382/xenithium-bench
export INPUTS=/path/to/pinned-sweep-inputs
export FIXTURE_DIR="$INPUTS"
export MODEL_DIR=/path/to/compatible-swift-checkpoint
export DRAFTER_DIR=/path/to/dspine-b16-checkpoint
export ONEAPI_ENV=/path/to/pinned/oneapi-env.sh
export OUTPUT_ROOT=/path/to/new-results-directory
case_name=short  # or middle / near; restart the server for every cell
```

Use `env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH=/usr/bin:/bin /usr/bin/bash --noprofile --norc` for each shell before setting these variables. Terminal 1 starts the server; after independently verifying readiness and actual configuration, terminal 2 runs the matching benchmark. Stop and reap the test server gracefully before the next cell. The runner preserves the archived fixed-fixture methodology and its actual flags, including reserve and reply budget; `case_name` selects the recorded native input count. Establish complete post-template u32le ID count/hash for both actual endpoints **twice per fixture using separate diagnostic builds before timing**. The archived request JSON hashes are `e74a5962a9b93630b537ce11766dc20d01342e5ab783d2b5a0d7ea09058ce64f` (short), `fc2d6f26958d19d9e31b060e0eb50e51d9395038752035df34eba901bc54ef8e` (middle), `5a626fb8c27c1fcf1d9c5548348e2829b9f4eaf0df8bb6a45c61f02b8c64be8c` (near); expected complete ID counts / SHA-256s are 8007 / `bdfe6efe3f1ff2189441ac4f3c8ca720bd1ee86ca8b907fec6adbf6bedb50d6d`, 36864 / `203f11564296866f3667895153ecc28785cb4cd99c750dceee3569a1b581d661`, 176073 / `1673ac28f1aa403c9cfce733638b680ac056b2eaadb2044c608a0b6cf5e31c78`. The historical ID receipts are not endpoint qualification for PR #4. The compatible model/drafter's complete shard identity was not recorded in the old manifest; access to the original artifacts and independent verification are required for an exact match.

| Planned arm | Terminal 1: launch | Terminal 2: matching benchmark |
| --- | --- | --- |
| Baseline: base/tree0 | `bash ./reproduce-server.sh base 0` | `bash ./reproduce-bench.sh "$case_name" base-tree0` |
| Base fallback control: base/tree7 | `bash ./reproduce-server.sh base 7` | `bash ./reproduce-bench.sh "$case_name" base-tree7` |
| Head chain control: head/tree0 | `bash ./reproduce-server.sh head 0` | `bash ./reproduce-bench.sh "$case_name" head-tree0` |
| Preferred feature evaluation: head/tree7 | `bash ./reproduce-server.sh head 7` | `bash ./reproduce-bench.sh "$case_name" head-tree7` |

Route evidence must be positive, request-correlated and collected separately from unmodified timed binaries. Matching visible text or retrieval is not output token-ID/numerical parity; a reported prefix hit is not proof of selector-history or full-cache prefill skip. Report HTTP TTFT, decode and total separately; no isolated model-prefill or restore time is assumed. One observation per cell would not establish scaling or a speedup distribution.
