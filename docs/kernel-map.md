# Kernel map: qwen35 (Qwen3.8-27B / Swift-1.5)

This file lists which GPU kernel runs in each phase of a qwen35 step for a given set of server flags and `XEN_*` switches, with the file:line of each dispatch decision.

The model mixes full-attention layers (gated GQA, group 6, head_dim 256) with GDN linear-attention layers, and its matrices are GPTQ int4. Kernels come from two libraries:

- `libxenithium_kernels`: compiled at 128 GRF.
- `libxenithium_kernels_xl.so`: compiled at 256 GRF. It holds only the kernels that spill at 128 (`src/kernels/CMakeLists.txt`). Those are marked **[XL]** below.

The trained-QSA switches do nothing unless `XEN_TRAINED_QSA_DELTA_DIR` is set ([trained K8V4 QSA](trained-k8v4-qsa.md)). Loading the bank then requires all of these, or the server refuses to start (`src/arch/qwen35/model.cpp:197-216`):

- `--kv int8-v4`
- `--attn-select 0`
- a shared KV pool (`--kv-pool-tokens` a token count or `auto`)
- a drafter
- a context of 2,049 to 262,144 tokens

## 1. Decode attention (full-attention layers)

The entry point is `k::gqa_decode` (`src/kernels/attention/gqa.cpp`), called per layer from `Model::submit_layers`:

- q|k|v and o projections: `w4a16_gemm_rows`.
- KV write and RoPE: `prep<i8v4>`.
- Merge: `merge_d<256,gate>`, or `merge_d<256,true,true>` on the trained path.

### 1a. Numerics (`--decode-attn`)

| flag | effect |
|---|---|
| `auto` | `int8-fast` on int8 / int8-v4 KV, `exact` on bf16 (`src/arch/qwen35/serve.cpp:54`) |
| `int8-fast` | int8 q·k, P hi only |
| `int8` | int8 q·k, P hi + lo |
| `fast` | hi + lo q codes, P hi only |
| `exact` | bf16 q·k, P hi + lo |

### 1b. Dense rows

`gqa_attn_group` picks the group shape (`gqa.cpp:794-822`). It turns on the L2 prefetch for segments of more than 4 rows at 32K context and above, and for multi-row int8 q·k at 131K and above.

| segment rows | kernel |
|---|---|
| 1 | `attn_seg128` → `attn_seg<i8v4,6,1,...>` (128 GRF, `gqa_seg.hpp`). `exact` at 32K and above uses the XL instance. |
| 2 | `gqa_attn_seg_xl` **[XL]** in 2-row groups (`gqa_seg.cpp`) |
| 3-16 | `gqa_attn_seg_xl` **[XL]** in 4-row groups (8-row groups spill) |
| tree step, at most 16 rows per segment | the tree instance `attn_seg<...,TR=1>` **[XL]** |
| int8 KV (not v4) | `attn_seg<i8,...>` instances |
| `--attn-select N` | `gqa_select` + `gqa_attn_seg_sel_xl` **[XL]**. Refused with int8-v4; a tree step forces dense. |

### 1c. Trained QSA rows

**Routing.** `Model::verify_batch` (`model.cpp:2078-2143`) decides per segment. A segment goes sparse only when all of these hold:

1. **Its slot is covered.** The trained history reaches the current position, with no rotary offset. Coverage is armed only for a fresh prompt of more than 8,192 tokens (`kTrainedQsaShort`). It is also kept across a page-aligned prefix-cache restore with `XEN_QSA_RESIDENT_SNAPSHOTS=1`, and across a restore into another slot with `XEN_QSA_RESIDENT_FORK_COPY=1`. A prompt of at most 8,192 tokens is never sparse, even after it decodes past 8,192.
2. **It is past the threshold.** The position is at least max(8192, `XEN_TRAINED_QSA_DECODE_MIN_POS`).
3. **Trees are allowed for it.** In a tree step, `XEN_TRAINED_QSA_TREE=1` must be set and the segment must have at most 16 rows. Otherwise the whole tree step runs dense.

The other segments of a sparse step get plan count 0. They still run in the trained kernel, as the dense walk bit for bit.

**Per layer.** `trained_qsa::Device::stage` (`src/arch/qwen35/trained_qsa_device.cpp:739`) always runs three `dense_gemm` projections plus `norm_index` and `index_blocks` for covered rows; this keeps the history up to date. It goes no further, and the stock kernel runs, when:

- the step is not sparse;
- the bound holds 512 blocks or fewer;
- the step is a tree and `XEN_TRAINED_QSA_TREE` is not set.

**Score kernel.** In this table, "half" means `XEN_TRAINED_QSA_POOL_FP16=1`, "xmx" means `XEN_QSA_XMX_SCORE=1`, and "tiled" means `XEN_TRAINED_QSA_TILED_SCORE=1`. Tiled scoring is also required for trees (`trained_qsa_device.cpp:342`).

| step | switches | score kernel |
|---|---|---|
| tree | half + xmx | `k::qsa_trained_dec_xmx_score`, fp16 DPAS (`src/kernels/attention/qsa.cpp`) |
| tree | half | `index_scores_tile<true,true>` |
| tree | fp32 pool | `index_scores_tile<false,true>` |
| chain, at most 16 rows per segment | half + xmx | `qsa_trained_dec_xmx_score`, plus `index_scores_tile<...,Tail>` for the blocks inside the step |
| chain | tiled, half / fp32 | `index_scores_tile<true>` / `<false>` |
| chain | not tiled | `index_scores<half>`, one work-group per block |

The score is followed by these stages in every case:

| stage | kernel |
|---|---|
| select | `k::qsa_select`: top 512 four-token blocks |
| pack | `k::gqa_pack_trained_qsa` (`gqa.cpp:832-924`): a page plan, or with `XEN_QSA_GATHER=1` a gather plan (entries of 16 selected blocks from any pages) |
| walk | `gqa_attn_seg_trained_xl` → `attn_seg<i8v4,...,trained>` **[XL]**, always in 1-row groups |
| merge | `merge_d<256,true,true>` with per-row counts |

Inside the pack, a row still falls back to dense (count 0) when:

- it sits in a dense segment;
- it holds 512 blocks or fewer (about 2,048 tokens);
- its list is malformed;
- the selection covers every block anyway.

## 2. Prefill attention

The entry point is `k::gqa_rows` (`gqa.cpp:1152-1205`).

| flags | kernel |
|---|---|
| `--prefill-attn auto` | `int8-fast` on int8 / int8-v4 KV (`serve.cpp:38`). The body is flash attention. |
| default | `gqa_attn_fa` → `attn_fa<i8v4,6,40,...>` **[XL]** (`gqa_fa.cpp`), 40 rows per group; the tail merge follows `--attn-tail`. |
| `--prefill-attn onednn` | `gqa_attn_sdpa`. Refused with int8-v4. |
| `--prefill-select N`, call context over N pages (N × 64 tokens) | `gqa_select_rows` (128 GRF), then `attn_fa<...,SEL=true>` **[XL]**. The prompt's last block stays dense. |
| trained bank layer, covered slot | `Device::prefill` adds `pf_dense` ×3 + `norm_index` + `index_blocks`. This only builds the decode history; the attention stays dense or SEL. |
| `XEN_TRAINED_QSA_PREFILL_XMX=1`, `--prefill-select 0`, one segment, covered, position ≥ `XEN_TRAINED_QSA_PREFILL_MIN_POS` (default 32,768) | `prefill_plan`: `qsa_trained_xmx_pack_k/q` → `qsa_trained_xmx_score` → `qsa_select`, then `attn_fa<...,TRAINED=true>` **[XL]** |
| ... + `XEN_PF_SUBTILE_SKIP=1` | the same, skipping unselected subtiles (40-row groups only) |
| `XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG=1` | allows the XMX prefill with `--slots 4` (single-segment calls) |
| `XEN_TRAINED_QSA_PREFILL_MERGED=1` (needs the 4-slot switch) | a per-segment trained plan inside merged multi-prompt calls |

`--prefill-select` > 0 turns trained sparse prefill off (`model.cpp:2828`, `:2835`). With both set, the XMX prefill workspace is still allocated but never used.

## 3. GEMMs (GPTQ int4, symmetric, group 128)

**Decode.** `w4a16_gemm_rows` (`src/kernels/gemm/w4a16_gemm.cpp:448-480`, 128 GRF):

| rows | kernel |
|---|---|
| 1 | `gemm_m<1>` |
| 2 | `gemm_m<2>` |
| 3-8 | `w4a16_gemm_m4` |
| 9-12 | `gemm16<4>` |
| 13-16 | `gemm16<8>` |
| 17-32 | `gemm32` |

The MLP fuses gate and up as `w4a16_gemm_rows_swiglu`. `--decode-w4 a8` switches to the W4A8 rows body.

**Prefill.** `w4a16_gemm_pf` / `_swiglu` (`src/kernels/gemm/w4a16_rows.cpp:1281-1311`) **[XL]**. Despite the file name, this is the many-row GEMM.

| condition | kernel |
|---|---|
| rows ≤ `--prefill-short` (32) | the decode `w4a16_gemm_rows` |
| rows ≥ `--prefill-gemm-rows` and the matrix is wide enough (`onednn_pf_takes`, `onednn_pf.cpp:351`) | oneDNN `onednn_w4a8_pf` / `onednn_glu_q8` **[XL]**. `--prefill-gemm onednn` takes the wide matrices only (n ≥ 6,144); `onednn-all` takes every matrix. |
| otherwise | ESIMD W4A8 `a8_shape` **[XL]** |

`--prefill-dense` is qwen38-only and does nothing on qwen35 (`tools/server/options.cpp:155`).

## 4. GDN layers

**Decode.** `gdn_decode` (`src/kernels/linear_attention/gdn.cpp`). The replay log holds 4 rows without a drafter, 8 with DFlash 2, and 16 with `--seg-rows 16`. Cores:

| condition | core |
|---|---|
| segment > 8 rows, or the last accept > 8 | `core_launch16_tree` **[XL]** |
| tree step, or replay of an accepted tree path | `core_launch8_tree` **[XL]** |
| segment > 4 rows, or the last accept > 4 | `core_launch8` **[XL]** |
| otherwise | `core_launch_km<4>` (128 GRF) |

A 1-row step commits its state, so no replay is needed.

**Prefill.** `gdn_rows`: a row core handles the edges, and full 64-row chunks go to `gdn_scan` **[XL]**.

## 5. Drafter (`src/arch/common/dflash.cpp`)

**Drafter kind.** A `config.json` with `DFlash2DraftModel` is DFlash 2; `DFlashDraftModel` or `DSpineDraftModel` alone is DSpine. `--seg-rows 16` needs a block of 16 or more (DSpine b16); DFlash 2 runs `--seg-rows 8`.

**Pass layers:**

- **DFlash 2:** `df_conv` → q / kv GEMMs → `df_prep` → `df_attn` → o → MLP.
- **DSpine:** causal `df_attn`, then the bridge (`ds_bridge_in` → `ds_codes` → `ds_message` → `ds_bridge_out`).

**Head and selection.** The head is `lm_gemm` (int4 or int8), then `df_topk`. Selection runs `df_tree` / `df_select` for DFlash 2, or `ds_tree` / `ds_select` for DSpine.

**Tree or chain** (`dflash.cpp:803-809`):

- A pass drafts a tree when `--drafter-tree` > 0 and the step has at most `--drafter-tree-seqs` streams. Below 64K context the stream limit rises to at least 4. Otherwise it drafts a chain.
- A tree has `--drafter-tree` nodes (7) for 8-row blocks and `--drafter-tree-wide` nodes (15) for 16-row blocks.

**Pass graphs.** Every pass is recorded as a SYCL graph per card and replayed, whatever `--graphs` says. The cache is capped by `XEN_DRAFT_GRAPHS`.

## 6. lm_head (`src/arch/qwen35/head.cpp`)

| `--lm-head` | kernels |
|---|---|
| `auto` | `int4-rescore` when the checkpoint has both an int4 and a dense head |
| `int4-rescore` | int4 logits (`w4a16_gemm_rows`) → `head_select_tiles` (the top `--lm-head-tiles` 16-column tiles per card) → int8 rescore `dense_gemm_tiles` |
| `int8` / `int4` | `dense_gemm` / `w4a16_gemm_rows` |

Every mode finishes with `greedy_pick`.

## 7. Graphs

`--graphs 1` records step graphs only for arch flint. The qwen35 step is submitted eagerly. The drafter's pass graphs (section 5) are always on.

## Recommended setup and what it runs (2× Arc Pro B60)

The settings are listed in [recommended settings](recommended-settings.md) and [trained K8V4 QSA, "Recommended configuration"](trained-k8v4-qsa.md) and in the preset `models/presets/swift15-27b-int4-baked-qsa.env`.

| phase | kernels |
|---|---|
| decode attention, dense rows (under 8,192 tokens, uncovered, layers without a bank tensor) | int8-fast K8V4. 1 row: `attn_seg128`. 2 rows: XL 2-row groups. 3-16 rows: XL 4-row groups. Trees: the XL tree instance. |
| decode attention, sparse rows | 3× `dense_gemm` → `qsa_trained_dec_xmx_score` (+ tail tile on chains) → `qsa_select` → gather pack → `gqa_attn_seg_trained_xl` → trained merge |
| prefill attention | `attn_fa` int8-fast **[XL]** up to 256 pages (16,384 tokens of context per call); past that `gqa_select_rows` + SEL `attn_fa` **[XL]**. The bank only indexes the chunk. |
| prefill GEMMs | ≤ 32 rows: decode GEMM. 33-511 rows: ESIMD W4A8 **[XL]**. ≥ 512 rows: oneDNN for every matrix **[XL]**. |
| decode GEMMs | `w4a16_gemm_rows`: `gemm16` / `gemm32` for wide steps |
| GDN | 16-row log; `core16_tree` for 9-16-row segments and their replays, `core8_tree` for trees of up to 8 rows, `core8` / `core_km<4>` for chains. Prefill: `gdn_rows` + `gdn_scan`. |
| drafter | DSpine b16. Trees at up to 4 streams below 64K context and up to 2 past it, with 15 nodes for 16-row blocks. |

`XEN_PREFILL_SHARE` changes only how the scheduler shares time between prefill and decode, not which kernel runs.

## Open questions

These could not be settled from the code alone:

- Whether the `grf_size<256>` property on `index_scores_tile` takes effect. It is a plain SYCL kernel in `trained_qsa_device.cpp`, not part of the XL library.
- Whether the GDN out projection is int4 or int8. That depends on the checkpoint.
- Under TP2 the o and down GEMMs may run on halves of the rows. If so, the `--prefill-gemm-rows` threshold sees fewer rows per call than the chunk has.
