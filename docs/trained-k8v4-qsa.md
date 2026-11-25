# Opt-in trained K8V4 QSA for Swift

Trained QSA selects which parts of the existing pooled `int8-v4` KV cache to read during speculative decode. It does not change the KV encoding, default prefill attention, draft policy, or scheduler. The separate XMX prefill option is documented in [trained QSA XMX prefill](trained-qsa-xmx-prefill.md). An absent sidecar flag leaves stock dispatch and caching in place, with no trained allocations.

## Configuration

| Environment variable | Default | Effect |
| --- | --- | --- |
| `XEN_TRAINED_QSA_DELTA_DIR` | Unset (stock) | Directory readable by Xenithium's checkpoint loader, containing the matching trained indexer tensors for Swift. The loader checks layer geometry and tensor names, shapes and types. The export uses F32 tensors and delta RMSNorm multipliers (stored value + 1). Supply weights separately; none are included. |
| `XEN_TRAINED_QSA_TILED_SCORE` | Off | Set to `1` for the tiled chain scorer. Legal chains have at most 16 rows per segment and 32 total rows. Other legal model layouts not covered by the tile use the untiled trained scorer when otherwise sparse-eligible. | **Recommended: `1`.**
| `XEN_TRAINED_QSA_POOL_FP16` | Off (FP32 summaries) | Set to `1` to store **completed** four-token summaries in FP16 and widen before FP32 scores. This is lossy. Projections stay resident; query, partial tails, staged keys and current block summaries are not rounded to FP16 by this option. | **Recommended: `1`** (needed by the XMX scorer).
| `XEN_TRAINED_QSA_TREE` | Off | `1`: draft-tree verifier steps can take the trained sparse walk too. Each eligible segment (covered, long, at most 16 tree rows) goes sparse; the other rows of the step stay dense. Needs `XEN_TRAINED_QSA_TILED_SCORE=1` (startup refuses otherwise). Measured: 3 users at 80k, 165 → 179 tok/s. **Recommended: `1`.** |
| `XEN_QSA_XMX_SCORE` | Off | `1`: the decode selector scores on fp16 DPAS (XMX). Queries are staged in SLM as fp16 hi + lo; completed summary keys come through a transposed 2D block load. Needs `XEN_TRAINED_QSA_POOL_FP16=1`; without it the scalar scorer runs. The top-512 selection matched the reference in every live check (327,680 / 327,680 tree, 163,840 / 163,840 chain). Score stage 1.16 → 0.44 ms (1 user), 3.24 → 1.02 ms (4 users). **Recommended: `1`.** |
| `XEN_QSA_GATHER` | Off | `1`: the packer emits gather entries (16 selected four-token blocks from any pages per entry), and the walk reads only the selected tokens instead of whole 64-token pages (up to ~33k tokens per row group). The output is the same selection, read with less traffic. **Recommended: `1`.** |
| `XEN_QSA_RESIDENT_SNAPSHOTS` | Off | `1`: prefix-cache snapshots keep the trained selector's coverage, so a follow-up turn that hits the cache stays sparse instead of falling back to dense. **Recommended: `1`** with prefix caching. |
| `XEN_QSA_RESIDENT_FORK_COPY` | Off | `1` (with `XEN_QSA_RESIDENT_SNAPSHOTS=1`): a snapshot restored into another slot also copies its trained summaries, when the snapshot is page-aligned and has no rotary offset. **Recommended: `1`.** |
| `XEN_TRAINED_QSA_DECODE_MIN_POS` | `0` (the 8,192-token short threshold only) | A position below which every row decodes dense. Use it to move the dense/sparse crossover, or set it huge to force dense for an A/B. |
| `XEN_TRAINED_QSA_PREFILL_XMX` | Off | Trained sparse **prefill** on XMX (see [trained QSA XMX prefill](trained-qsa-xmx-prefill.md)); `XEN_TRAINED_QSA_PREFILL_MIN_POS`, `_4SLOT_SINGLE_SEG`, `_MERGED` and `XEN_PF_SUBTILE_SKIP` extend it. Off whenever `--prefill-select` is on (the two are exclusive). **Recommended: `0`.** `--prefill-select 256` on the int8-v4 KV is faster: 239k cold prefill 1,521 → 2,424 tok/s with 8/8 needles at 32k and 121k. |
| `XEN_PREFILL_SHARE` | Unset | `s` (0 < s < 1): while other requests decode, prompt processing gets about `s` of the loop's wall time. Not QSA-specific: every arch's batch engine. Measured with `0.33`: a decoder beside a 112k prompt being read, 47-51 → 75 tok/s; the long prompt's TTFT unchanged. **Recommended: `0.33`** for interactive multi-user use. |

Use a matching Swift 5,120-hidden, 64-layer text model with 16 compatible global-attention layers, an upstream-supported speculative drafter, a fixed-size shared K8V4 pool, and configured context above 2,048 and at most 262,144. For each exported layer, the loader looks under the target checkpoint's `layers.<layer>.self_attn._qsa_indexer.` prefix for `q_proj.weight` (`512 × 5120`), `k_proj.weight` (`128 × 5120`), and `q_norm`, `k_norm`, and `k_block_norm` (128 values each). All five tensors must be finite F32 values. Layers not present in the bank stay dense.

Do not combine trained QSA with the decode page selector (`--attn-select`). The page-selected prefill (`--prefill-select N`) works on the int8-v4 KV next to trained QSA and is the recommended prompt path. Incompatible startup settings, invalid sidecars, insufficient VRAM or an unavailable memory preflight fail startup rather than shrinking the pool or spending the runtime reserve. Vision/embedding prefill is unsupported in trained mode. Requests with nonstandard rotary position state cannot build trained history. Draft trees on the int8-v4 KV come from #4, and DSpine against a GPTQ int4 head from #6.

Stock prefix caching can be on or off. With snapshots enabled, trained mode copies snapshot state directly between the device and pinned host buffers, with dependencies on both prior writes and reads; the stock snapshot layout and matching rules are unchanged. Restoring a KV snapshot does **not** restore trained selector summaries. Both the normal K8V4 pool and cache capture buffer are charged at startup. The large five-slot configuration below fit with the two opt-ins but not with the tested resident FP32 storage configuration. Capacity is machine- and pool-dependent.

## Dispatch and correctness

| Request or step | Attention route |
| --- | --- |
| Prompt of at most 8,192 tokens | Stock dense, even if generation grows beyond 8,192. |
| Fresh long text prompt | Prefill builds trained history from token zero. A verifier row can use trained sparse once its slot has history through its full logical prefix and the step starts at or beyond 8,192 (or `XEN_TRAINED_QSA_DECODE_MIN_POS`). |
| Cache hit (live reuse, snapshot) | With `XEN_QSA_RESIDENT_SNAPSHOTS=1` (+ `_FORK_COPY=1` for another slot): the trained coverage comes back with the KV, so the row stays sparse. Without them: stock dense. Trimmed or incomplete history is always dense. |
| Tree verifier | With `XEN_TRAINED_QSA_TREE=1`: each covered segment of at most 16 tree rows goes sparse. Without it: dense. |
| A batch mixing eligible and ineligible slots | Per segment: eligible segments take the trained walk, the other rows get plan count 0, which is the dense walk bit for bit. (Earlier versions sent the whole batch dense.) |
| Layers absent from the sidecar, or rows whose selected blocks cover their whole prefix | Stock dense. |

`XEN_QSA_TRIAL_ROUTE_LOG=1` prints the route and reason per segment per step (`covered`, `covered-tree`, `short`, `incomplete`, `tree`, `mixed`). It is very verbose: one line per stream per step. `XEN_REQ_LOG=1` prints one `REQ_DONE` line per finished request.

Scores are causal and independent for each row: select up to 512 completed four-token blocks, with lower block IDs winning ties, and include the row's own block. The packer emits logical pages and block masks; the native K8V4 walk uses the slot's physical page table and clips to the row's causal end. Only accepted rows become persistent history. The tiled path preserves the evaluated stock FP32 multiply/FMA and lane reduction order; that does not imply output equivalence to dense or equivalence between FP16 and FP32 summary storage.


## Recommended configuration (fastest measured)

Two Arc Pro B60s, Swift-1.5 27B GPTQ int4 baked, DSpine b16 drafter, 6 slots. Results: 1 / 2 / 4 / 6 streams at 140 / 211 / 316 / 379 tok/s total, 4 users at ~90k context 255-261 tok/s, decode deep in a document 107 (32k) / 108 (109k) tok/s, prefill 3,315 tok/s shallow and ~2,700 deep, bench `check` 97.1%. See [the kernel map](kernel-map.md) for which kernel each of these picks.

```sh
XEN_TRAINED_QSA_DELTA_DIR=/path/to/bank XEN_TRAINED_QSA_TILED_SCORE=1 XEN_TRAINED_QSA_POOL_FP16=1 \
XEN_TRAINED_QSA_TREE=1 XEN_QSA_XMX_SCORE=1 XEN_QSA_GATHER=1 \
XEN_QSA_RESIDENT_SNAPSHOTS=1 XEN_QSA_RESIDENT_FORK_COPY=1 XEN_PREFILL_SHARE=0.33 \
xenithium-server --model ~/models/swift15-27b-int4-baked --tp 2 --slots 6 --kv int8-v4 --kv-pool-tokens 884416 --reserve 2.0 \
  --drafter 1 --drafter-model ~/models/qwen38-27b-dspine-b16 --seg-rows 16 --draft-depths 15,7,4,3 \
  --drafter-tree 7 --drafter-tree-wide 15 --drafter-wide 7 --drafter-tree-seqs 2 --drafter-cut 0.02 --drafter-mtp-from 0 \
  --adaptive-depth 1 --lookup 4 --prefill-chunk 2048 --prefill-select 256 \
  --prefill-gemm onednn-all --prefill-gemm-rows 512 --kv-lend on --prefix-cache 1 --prefix-mark-min 2048
```

Six slots are the most that fit this pool next to the trained summaries and the prefix cache on 24 GB cards. Leave the diagnostic switches (`XEN_QSA_TRIAL_ROUTE_LOG`, `XEN_PF_LOG`, `XEN_QSA_SCORE_CHECK`, `XEN_QSA_PACK_CHECK`) unset in production.

## Qualification and remaining gaps

For the change map, host regression coverage and adjacent-path test matrix, see [trained QSA validation](trained-qsa-validation.md). Repository benchmark reports and methodology remain under [benchmark](benchmark/README.md).

The numbers below come from a matched downstream qualification build and its dense control, not from an end-to-end run of the public topic. Both arms used two Intel Arc Pro B60s, DSpine, a five-slot scheduler with stock admission, an 884,416-token pool (177,216 cap per request), and prefix caching with a 2 GiB budget. The candidate enabled tiled scoring and FP16 completed summaries. This patch keeps the qualified scoring, page walk and cache-safety code but does not include the downstream drafter changes or benchmark harness.

Three distinct 8,007-token prompts, each cold then repeated with identical native IDs and a 128-token output budget, matched all six stock completion texts. Repeats restored 7,936 tokens on both arms.

| Median | Matched downstream dense control | Trained QSA candidate |
| --- | ---: | ---: |
| Cold HTTP decode | 114.35 tok/s | 114.38 tok/s |
| Repeated-prefix HTTP decode | 115.27 tok/s | 115.22 tok/s |
| Cold time to first token | 2,729.91 ms | 2,728.39 ms |
| Repeated-prefix time to first token | 75.67 ms | 74.18 ms |

The first candidate cold request waited about 1.05 s longer before model prefill started, during snapshot host pinning around startup. Rank 0 had just 34.34 MiB beyond the existing 1.5 GiB startup reserve after budgeting cache capture, KV and trained selector memory.

In a mixed stock-scheduler burst, retrieval was exact for all four requests in both arms. The 36,864-token extension reused 7,936 short-prefix tokens and stayed dense for all 251 verifier steps. A separate repeated short request reused 7,936 tokens in a different slot via a copied snapshot and also stayed dense. Cold 28,672- and 30,720-token requests ran 58/102 and 39/94 sparse verifier steps respectively, with submissions on all 16 trained layers and both ranks. HTTP decode (stock → candidate, in prompt order) was 74.82 → 78.79, 52.38 → 52.34, 66.10 → 67.50, and 131.76 → 118.78 tok/s. These are individual staggered request rates, not joint throughput or a consistent per-request improvement.

Earlier, separate synchronized long-context DSpine experiments measured 159.76 → 203.01 joint tok/s at 4 × 200,002 tokens (QSA mean/slowest 50.75/46.71), and 164.98 → 192.03 at 5 × 176,073 (38.41/36.52). Both arms retrieved all expected values; their full texts differed. These are benchmark-only workloads, not stock-scheduler or production throughput. They did not reach the near-max per-user targets, and the whole-selector five-user replay was about 15.5 ms for 16 layers per rank per verifier step vs. the 5 ms goal. A separate FP16 audit observed changed top-512 membership; retrieval checks do not establish general output parity. The downstream measurements in this section do not include trained sparse prefill, long restored-prefix sparse decode, or experimental threshold/XMX decode scoring. The separate default-off XMX prefill extension and its limits are documented in [trained QSA XMX prefill](trained-qsa-xmx-prefill.md).
