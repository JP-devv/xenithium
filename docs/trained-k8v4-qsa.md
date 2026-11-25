# Opt-in trained K8V4 QSA for Swift

Trained QSA selects which parts of the existing pooled `int8-v4` KV cache to read during speculative decode. It does not change the KV encoding, default prefill attention, draft policy, or scheduler. The separate XMX prefill option is documented in [trained QSA XMX prefill](trained-qsa-xmx-prefill.md). An absent sidecar flag leaves stock dispatch and caching in place, with no trained allocations.

## Configuration

| Environment variable | Default | Effect |
| --- | --- | --- |
| `XEN_TRAINED_QSA_DELTA_DIR` | Unset (stock) | Directory readable by Xenithium's checkpoint loader, containing the matching trained indexer tensors for Swift. The loader checks layer geometry and tensor names, shapes and types. The export uses F32 tensors and delta RMSNorm multipliers (stored value + 1). Supply weights separately; none are included. |
| `XEN_TRAINED_QSA_TILED_SCORE` | Off | Set to `1` for the tiled chain scorer. Legal chains have at most 16 rows per segment and 32 total rows. Other legal model layouts not covered by the tile use the untiled trained scorer when otherwise sparse-eligible. |
| `XEN_TRAINED_QSA_POOL_FP16` | Off (FP32 summaries) | Set to `1` to store **completed** four-token summaries in FP16 and widen before FP32 scores. This is lossy. Projections stay resident; query, partial tails, staged keys and current block summaries are not rounded to FP16 by this option. |

Use a matching Swift 5,120-hidden, 64-layer text model with 16 compatible global-attention layers, an upstream-supported speculative drafter, a fixed-size shared K8V4 pool, and configured context above 2,048 and at most 262,144. For each exported layer, the loader looks under the target checkpoint's `layers.<layer>.self_attn._qsa_indexer.` prefix for `q_proj.weight` (`512 × 5120`), `k_proj.weight` (`128 × 5120`), and `q_norm`, `k_norm`, and `k_block_norm` (128 values each). All five tensors must be finite F32 values. Layers not present in the bank stay dense.

Do not combine trained QSA with the stock prefill or decode page selector. Incompatible startup settings, invalid sidecars, insufficient VRAM or an unavailable memory preflight fail startup rather than shrinking the pool or spending the runtime reserve. Vision/embedding prefill is unsupported in trained mode. Requests with nonstandard rotary position state cannot build trained history. This patch does not add GPTQ drafter loading or K8V4 tree support; the verifier still follows upstream's supported draft routes.

Stock prefix caching can be on or off. With snapshots enabled, trained mode copies snapshot state directly between the device and pinned host buffers, with dependencies on both prior writes and reads; the stock snapshot layout and matching rules are unchanged. Restoring a KV snapshot does **not** restore trained selector summaries. Both the normal K8V4 pool and cache capture buffer are charged at startup. The large five-slot configuration below fit with the two opt-ins but not with the tested resident FP32 storage configuration. Capacity is machine- and pool-dependent.

## Dispatch and correctness

| Request or step | Attention route |
| --- | --- |
| Prompt of at most 8,192 tokens | Stock dense, even if generation grows beyond 8,192. |
| Fresh long text prompt | Dense prefill builds history from token zero. A verifier can use trained sparse only once every slot has history through its full logical prefix and the step starts at or beyond 8,192. |
| Cache hit (live reuse, in-place or copied snapshot), trimmed or incomplete history | Stock dense. Stock KV reuse is retained; existing KV pages alone are not evidence of trained history. A cache-restored request remains dense. |
| Tree verifier, or a batch containing any ineligible slot | Stock dense for the entire batch. Eligible long slots can continue accumulating history, including the accepted path through a tree. |
| Layers absent from the sidecar or rows whose selected blocks cover their whole prefix | Stock dense. |

Scores are causal and independent for each row: select up to 512 completed four-token blocks, with lower block IDs winning ties, and include the row's own block. The packer emits logical pages and block masks; the native K8V4 walk uses the slot's physical page table and clips to the row's causal end. Only accepted rows become persistent history. The tiled path preserves the evaluated stock FP32 multiply/FMA and lane reduction order; that does not imply output equivalence to dense or equivalence between FP16 and FP32 summary storage.

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
