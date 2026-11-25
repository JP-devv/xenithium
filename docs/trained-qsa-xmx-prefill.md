# Opt-in XMX scoring for trained K8V4 QSA prefill

This extends the trained decode feature in [PR #3](https://github.com/Niluthon/Xenithium/pull/3). XMX is general matrix hardware; here it computes **index scores for QSA's four-token blocks**, not dense attention. The existing native K8V4 int8-fast flash-attention body consumes the sorted group page union and each query's own mask. Independent causal selection, top-512 with the native tie rule, the physical page table and the mandatory current block are unchanged. This adds no tree decode support, loader changes or DSpine scheduling changes.

Enable with `XEN_TRAINED_QSA_PREFILL_XMX=1`. It requires Swift's 16-layer trained QSA bank, one slot, pooled `int8-v4`, native many-row int8-fast FA and no competing selector. Set `XEN_TRAINED_QSA_PREFILL_MIN_POS` to a page-aligned start in `[2112, context)`; default **32768**. The feature is otherwise off. Early chunks, requests without contiguous tracked history (including prefix restores), multimodal rows and non-native/merged chunks keep the existing dense prefill path. The pre-existing prefix-cache guards and short-prompt cutoff are unchanged. Unsupported opt-in configurations fail clearly; do not turn off the prefix cache to use this feature. Decode and its FP16-history option are independent.

Q and the completed index keys are converted to BF16 for 8×16 DPAS, with FP32 accumulators and the trained ReLU/head sum. If FP16 summaries are configured, their stored values are converted to BF16 only for this scorer. This preserves independent ranking semantics, but is **not FP32 score or selected-ID parity**. Sparse attention can change prompt hidden states and generated text even when retrieval still succeeds. `XEN_TRAINED_QSA_POOL_FP16=1` also rounds completed history to FP16; it is not a lossless option.

## Build and opt-in

On an Arc B60 with an appropriate Level Zero driver and oneAPI 2025.3 installed, use the same public checkout and build for both arms:

```sh
export XEN_ONEAPI=/opt/intel/oneapi/2025.3
source scripts/oneapi-env.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$XEN_ONEAPI/bin/icpx" -DXENITHIUM_SYCL_DEVICE=bmg
cmake --build build --target xenithium-server xenithium-test-qsa-host xenithium-test-server
ctest --test-dir build -R '^(qsa-host|server)$' --output-on-failure
```

Use loader-compatible model exports and the exact sidecar linked in PR #3. Use one freshly started server per arm and no other GPU workloads. This is a launch template for the public implementation, not a recipe for reproducing the private downstream binary. Lower the context and pool together if memory requires it, keeping room for the prompt and output budget.

```sh
export XEN_TRAINED_QSA_DELTA_DIR=/path/to/verified/sidecar-directory
export XEN_TRAINED_QSA_TILED_SCORE=1 XEN_TRAINED_QSA_POOL_FP16=1
export XEN_TRAINED_QSA_PREFILL_XMX=1  # unset this for the dense-prefill control
# Optional, default 32768: export XEN_TRAINED_QSA_PREFILL_MIN_POS=32768
build/xenithium-server --model /path/to/compatible/swift --drafter 1 \
  --drafter-model /path/to/compatible/drafter --drafter-tree 0 --drafter-mtp-from 0 \
  --host 127.0.0.1 --port 18102 --tp 2 --slots 1 --context 262144 \
  --kv-pool-tokens 262144 --kv int8-v4 --prefill-attn int8-fast \
  --prefill-chunk 2048 --prefill-w4 a8 --prefix-cache 1 --prefix-cache-gib 2 \
  --prefix-live 0 --prefill-select 0 --attn-select 0
```

Use the same frozen native prompt IDs, greedy settings and output limit in both arms. An eligible chunk emits `trained QSA prefill submitted:` per rank and trained layer with `scorer=bf16-xmx8x16` and `fa=k8v4-i8-fast-masked-union`. Measure model-call prefill separately from HTTP time to first token. Check actual cached-token counts; enabling a cache does not make a cold request a hit.

## Evidence and limits

- **Earlier downstream same-state late-chunk replay (not the public branch):** at 194,560 history, 2,048 rows, rank-0/rank-1 FP32 tiled versus BF16 XMX index scoring about **39.7/39.0 → 2.68/2.62 ms**. The XMX full sparse replay took about **87 ms**, not 2.7 ms; selection, packing and attention still ran. The selected set changed by **3,059 block replacements per rank** over 2,048 queries in that separate captured FP32-history comparison. This is not the public branch's performance or a retrieval measure.
- **Downstream DSpine cold request (not the public branch):** one fresh paired 261,744-ID RUBY request per arm, 400 greedy outputs, pooled 262,144 K8V4 tokens, 2,048-row chunks, FP16 summary storage, unchanged tree+chain QSA decode on both. Dense versus XMX sparse prefill: **1119.48 → 1221.04 model-call tok/s**, **233.988 → 214.547 s HTTP TTFT**, **237.380 → 217.581 s to last content**. Both returned the exact retrieval line, but full responses differed. This compares dense to the complete XMX-scored sparse path, not an isolated scorer toggle at equal selections. `--prefix-cache 1 --prefix-cache-gib 2 --prefix-live 0` was configured; each measured request processed all 261,744 IDs, so neither was a cache hit.
- **Earlier downstream prefix-cache controls (XMX not involved):** same prefix-cache flags, five configured slots, three distinct 8,007-ID prompts, each cold then repeated, 128 greedy outputs. Cold receipts reused **0** tokens; repeats reused **7,936** in both stock and cache-safe trained-decode arms. All six text hashes matched across builds. Cold and hit median TTFT were respectively **2729.908 / 75.667 ms stock** and **2728.391 / 74.177 ms cache-safe**. These are cache/dense-fallback controls, not measurements of XMX or prefix-cache gains from this patch.

Neither one prompt nor one sampled layer proves general output quality. See the PR description for the exact public-head build and dispatch validation scope.
