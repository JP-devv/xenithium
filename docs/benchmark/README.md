# Benchmark reports

The self-contained HTML reports behind the performance tables, one `xenithium-bench run` per file
([docs/bench-guide.md](../bench-guide.md) tells you how to run your own). Open any of them in a browser; they work
offline.

- [Easy mode](#easy-mode): the README's [performance table](../../README.md#performance), files ending in `-easy.html`.
- [Hard mode](#hard-mode): the [hard-mode table](../performance-hard-mode.md), the other files.

## Reproduction requirement for PR results

New PR results packages must provide copy-paste server-launch and matching benchmark commands beside each result group, including a separate baseline and when the preferred settings apply. Record the exact revision/binary, hardware and TP, model/quantization, context and KV capacity, configured slots versus actually active users, drafter/cache settings, nondefault flags and environment, fixture identity and correctness limits. Machine-local paths may be placeholders, but required artifacts and their hashes must be identified. Derive commands from the run receipts; mark historical runs and unavailable inputs explicitly rather than inventing parameters. Link one results entry from the top-level README when applicable; extend the existing entry rather than adding duplicate links.

## Easy mode

All runs are from 2026-11-21 on the release build of the engine: `xenithium-bench run --mode easy --suite standard`
against a live server on a machine with 4x Arc Pro B70 power-capped to 220 W per card, the whole model on the card
count of the file name. The concurrency sweep stops at the server's slot count (`--set concurrency.levels=...`): a level
past it would only queue requests.

To use the idle cards, some runs went side by side, each server on its own cards (`ZE_AFFINITY_MASK`) and port. Those
runs sample no power: the meter reads every card of the machine, so it would count the other server too.

| Report | Run | Cards | Max context | Slots | KV | Prefill chunk | Drafter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen38-flash-next-4-card-easy.html` | `20261121-140553-flash-next-tp4-easy` | 4 | 524,288 | 16 | `int8` | 4,096 | DFlash 2 |
| `qwen38-flash-next-2-card-easy.html` | `20261121-141224-flash-next-tp2-easy` | 2 | 524,288 | 16 | `int8-v4` | 4,096 | DFlash 2 |
| `qwen38-27b-autoround-2-card-easy.html` | `20261121-143726-27b-autoround-tp2-easy` | 2 | 262,144 | 16 | `int8` | 2,048 | DSpine b16 |
| `qwen38-27b-autoround-1-card-easy.html` | `20261121-145355-27b-autoround-tp1-easy` | 1 | 262,144 | 8 | `int8` | 2,048 | DSpine b16 |
| `qwen38-27b-swift15-2-card-easy.html` | `20261121-143946-swift15-tp2-easy` | 2 | 262,144 | 16 | `int8` | 2,048 | DSpine b16 |
| `qwen38-27b-swift15-1-card-easy.html` | `20261121-151043-swift15-tp1-easy` | 1 | 262,144 | 8 | `int8` | 2,048 (`0`: the default) | none |
| `flint-int4-2-card-easy.html` | `20261121-144927-flint-tp2-easy` | 2 | 1,048,576 | 8 | `int8` | 4,096 | DFlash 2 |
| `flint-int4-1-card-easy.html` | `20261121-152059-flint-tp1-easy` | 1 | 262,144 | 8 | `int8` | 4,096 | DFlash 2 |

Notes on single runs:

- `qwen38-27b-autoround-2-card-easy.html` started before the side-by-side runs and samples power; from 14:39 its power
  section also counts the two cards of the Swift-1.5 two-card run.
- `qwen38-27b-swift15-1-card-easy.html` ran without a drafter: its one-card settings name none, so it decodes one
  token per step.
- Qwen3.8-Flash-Next on one card has no easy-mode run.

## Hard mode

All runs are 2026-11-20 on the release build of the engine, a live server started from the preset named below, the
whole model on the card count of the file name. Each row of the table has two reports: the quick run (decode,
concurrency, prefill to 32K, decode at 32K) and a long-context run (the same server at its full max context: prefill
64K / 128K, decode at 64K / 128K).

Every run started its preset's block drafter: DFlash 2 for Qwen3.8-Flash-Next and Recursive Flint, DSpine b16 for
both Qwen3.8-27B checkpoints. The reports show the accepted tokens per speculative step the server counted.

The Slots column is what each run had. The presets have since changed two settings: 16 slots on two and four cards
(Flint 8, its most) and 4 on one card, and prefill chunks of 4,096 tokens on Qwen3.8-Flash-Next and Flint, 2,048 on
both Qwen3.8-27B checkpoints.

| Report | Run | Server preset | Max context | Slots | Drafter |
| --- | --- | --- | --- | --- | --- |
| `qwen38-flash-next-4-card.html` | `20261120-155957-flash-next-tp4` | `qwen38-w4a16-intel` + `.tp4.env` | 1,048,576 | 16 | DFlash 2 |
| `qwen38-flash-next-4-card-long-context.html` | `20261120-165023-flash-next-tp4-ctx` | as above | 1,048,576 | 16 | DFlash 2 |
| `qwen38-flash-next-2-card.html` | `20261120-160245-flash-next-tp2` | `qwen38-w4a16-intel` + `.tp2.env` | 524,288 | 8 | DFlash 2 |
| `qwen38-flash-next-2-card-long-context.html` | `20261120-165153-flash-next-tp2-ctx` | as above | 524,288 | 8 | DFlash 2 |
| `qwen38-flash-next-1-card.html` | `20261120-160943-flash-next-tp1` | `qwen38-w4a16-intel` + `.tp1.env` | 524,288 | 8 | DFlash 2 |
| `qwen38-flash-next-1-card-long-context.html` | `20261120-165613-flash-next-tp1-ctx` | as above | 524,288 | 8 | DFlash 2 |
| `qwen38-27b-swift15-2-card.html` | `20261120-162547-qwen38-27b-tp2` | `qwen38-27b` + `.tp2.env` | 524,288 | 16 | DSpine b16 |
| `qwen38-27b-swift15-2-card-long-context.html` | `20261120-164112-qwen38-27b-tp2-ctx` | as above | 524,288 | 16 | DSpine b16 |
| `qwen38-27b-swift15-1-card.html` | `20261120-162923-qwen38-27b-tp1` | `qwen38-27b` + `.tp1.env` | 131,072 | 8 | DSpine b16 |
| `qwen38-27b-swift15-1-card-long-context.html` | `20261120-164351-qwen38-27b-tp1-ctx` | as above | 131,072 | 8 | DSpine b16 |
| `qwen38-27b-autoround-2-card.html` | `20261120-155539-qwen38-27b-intel-tp2` | `qwen38-27b-intel` + `.tp2.env` | 1,048,576 | 16 | DSpine b16 |
| `qwen38-27b-autoround-2-card-long-context.html` | `20261120-164536-qwen38-27b-intel-tp2-ctx` | as above | 1,048,576 | 16 | DSpine b16 |
| `qwen38-27b-autoround-1-card.html` | `20261120-163521-qwen38-27b-intel-tp1` | `qwen38-27b-intel` + `.tp1.env` | 131,072 | 8 | DSpine b16 |
| `qwen38-27b-autoround-1-card-long-context.html` | `20261120-164825-qwen38-27b-intel-tp1-ctx` | as above | 131,072 | 8 | DSpine b16 |
| `flint-int4-2-card.html` | `20261120-174030-flint-tp2` | `flint-int4` | 1,048,576 | 8 | DFlash 2 |
| `flint-int4-2-card-long-context.html` | `20261120-174227-flint-tp2-ctx` | as above | 1,048,576 | 8 | DFlash 2 |
| `flint-int4-1-card.html` | `20261120-174413-flint-tp1` | `flint-int4.tp1` | 262,144 | 8 | DFlash 2 |
| `flint-int4-1-card-long-context.html` | `20261120-174701-flint-tp1-ctx` | as above | 262,144 | 8 | DFlash 2 |

The Swift-1.5 two-card report is the clean rerun of 16:28: the first run of the series had one concurrency-check
failure on the check harness and its numbers are not in the README. Each live server passed the seventeen
correctness checks and the 64K / 128K pass before and during its runs.
