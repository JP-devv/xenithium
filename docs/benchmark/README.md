# Benchmark reports

The self-contained HTML reports behind the README performance table, one `xenithium-bench run` per file
(`docs/bench.md` tells you how to run your own). All runs are 2026-11-20 on the release build of the engine, a live
server started from the preset named below, the whole model on the card count of the file name. Each row of the
table has two reports: the quick run (decode, concurrency, prefill to 32K, decode at 32K) and a long-context run
(the same server at its full max context: prefill 64K / 128K, decode at 64K / 128K).

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
