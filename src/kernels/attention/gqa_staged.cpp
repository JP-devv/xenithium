// Xenithium · the many-row gated GQA attention, staged K / V tiles (prefill; see gqa.hpp,
// gqa_attn_staged).
//
// Unlike the block kernel (gqa_rows.cpp), which splits the 256 dims over its 16 items and sums every q.k's partials
// through SLM, a work-group of 2 x P / 8 items handles one block of QB rows x G q heads of one kv head
// (P = QB x G pairs):
//   - per 64-token tile (one KV page of this head, contiguous [64][256]) the items stage K and V once into SLM as bf16
//     DPAS B operands (int8 codes widen exactly; 2D block loads of 16 tokens x 64 bytes), K transposed to [dims][tokens]
//     for q.k, V token-paired for P.V: 64 KB, so two work-groups share a core;
//   - item i owns 8 pairs (pair block i / 2) with their q in registers: it computes the full 256-dim q.k of its pairs
//     for every token of the tile (16 DPAS of 16 dims per 16 tokens, both halves of a pair block alike, so no exchange),
//     runs the online softmax of its 8 pairs in registers, and accumulates P.V for its half (i % 2) of the 256 dims;
//   - two barriers per tile (staging done, readers done); after the first, each item prefetches its units of the next
//     tile into the caches (prefetch_2d), so the next staging reads hit them.
// The tile's scores, max / sum and o rescaling follow the decode attention's semantics (scale 1/16, x the K scale for
// int8, P = hi + lo bf16 x the V scale, o rescaled by exp(m_old - m_new)) with a causal mask per pair; the q.k sum order
// is DPAS's over 16-dim steps (not the item order), so the bits differ from both other bodies within the softmax
// bound. Finish as: attn = bf16(bf16(o / l) bf16(sigmoid(bf16(gate)))). Tokens past the row block's last row
// in the last tile are staged as zero V and masked scores (the page's unwritten slots may hold any bits).
// Its own TU at 256 GRF (xenithium_kernels_xl).
#include <sycl/ext/intel/esimd.hpp>

#include <cmath>
#include <stdexcept>

#include "xenithium/kernels/attention/gqa.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = gqa_bf16;

constexpr int kD = kQsaDim, kTile = kQsaTile, kPage = kQsaPage;
static_assert(kTile == 64 && kPage == 64, "a staged tile is one KV page");
constexpr int kOp = 256;                     // bf16 elements of one B operand (16 k x 16 n)
constexpr int kOpBytes = kOp * 2;
constexpr int kDc = kD / 16;                 // 16-dim chunks per head
constexpr int kSubs = kTile / 16;            // 16-token sub-tiles per tile
constexpr int kKOff = 0;                     // K operands [dc][sub]
constexpr int kVOff = kDc * kSubs * kOpBytes;  // V operands [sub][dc]
constexpr int kSlm = 2 * kDc * kSubs * kOpBytes;  // 64 KB

template <QsaKv KV, int G, int QB>
class GqaAttnStagedK;

// 16 tokens x 16 dims bf16 (row-major [tok][dim]) -> the q.k B operand (k = dims, n = tokens): dword pairs transposed
inline esimd::simd<bf16, kOp> k_operand(esimd::simd<bf16, kOp> v) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint32_t, 128> kd = v.template bit_cast_view<uint32_t>();
  esimd::simd<uint32_t, 128> bd;
#pragma unroll
  for (int pp = 0; pp < 8; ++pp) bd.template select<16, 1>(pp * 16) = kd.template select<16, 8>(pp);
  return bd.template bit_cast_view<bf16>();
}
// 16 tokens x 16 dims -> the P.V B operand (k = tokens, n = dims): token pairs interleaved
inline esimd::simd<bf16, kOp> v_operand(esimd::simd<bf16, kOp> v) SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, kOp> bv;
#pragma unroll
  for (int qq = 0; qq < 8; ++qq) {
    bv.template select<16, 2>(qq * 32) = v.template select<16, 1>(2 * qq * 16);
    bv.template select<16, 2>(qq * 32 + 1) = v.template select<16, 1>((2 * qq + 1) * 16);
  }
  return bv;
}

template <QsaKv KV, int G, int QB>
sycl::event attn_staged(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy,
                        const bf16 *qb, bf16 *attn) {
  constexpr bool I8 = KV == QsaKv::i8;
  constexpr int P = G * QB;
  static_assert(P % 8 == 0, "pairs per work-group: a multiple of 8");
  constexpr int NI = 2 * (P / 8);          // items: (pair block, dim half)
  constexpr int kE = I8 ? 1 : 2;           // KV element bytes
  constexpr int kUw = 64 / kE;             // dims per staging unit (a 16-token x 64-byte 2D block)
  constexpr int kUdc = kUw / 16;           // 16-dim chunks per unit
  constexpr int kUnits = kSubs * (kD / kUw);  // units per K (and per V) tile
  const int hq = s.hq(), hkv = s.hkv(), blocks = (m + QB - 1) / QB;
  const uint8_t *kp = c.k, *vp = c.v;
  const float *ksp = c.ks, *vsp = c.vs;
  const int32_t *pages = c.pages;
  return q.parallel_for<GqaAttnStagedK<KV, G, QB>>(
      sycl::nd_range<1>(static_cast<size_t>(blocks) * hkv * NI, NI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kSlm>();
        const int g = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0));
        const int blk = g / hkv, kvh = g % hkv, r0 = blk * QB;
        const int nrows = m - r0 < QB ? m - r0 : QB;
        const int p0 = pos_word[0] + r0;  // position of the block's first row
        const int ntok = p0 + nrows, ntiles = (ntok + kTile - 1) / kTile;
        const int pb = i >> 1, half = i & 1;
        // q of the item's 8 pairs as A operands [dc][pair][16 dims]; a dead row repeats the block's first row
        esimd::simd<bf16, kDc * 128> qa;
        esimd::simd<int32_t, 8> rpos;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int pr = pb * 8 + r, j = pr / G, h = pr % G, jj = j < nrows ? j : 0;
          rpos[r] = p0 + jj;
          esimd::simd<bf16, kD> row = esimd::block_load<bf16, kD>(qb + (static_cast<size_t>(r0 + jj) * hq + kvh * G + h) * kD);
#pragma unroll
          for (int dc = 0; dc < kDc; ++dc) qa.template select<16, 1>(dc * 128 + r * 16) = row.template select<16, 1>(dc * 16);
        }
        const float kmul = 1.f / 16.f;  // 1 / sqrt(256)
        esimd::simd<float, 8 * 128> oa(0.f);  // [d of this half][pair][16 dims]
        esimd::simd<float, 8> m_run(-INFINITY), l_run(0.f);
        const esimd::simd<int32_t, kTile> ti(0, 1);
#pragma unroll 1
        for (int tile = 0; tile < ntiles; ++tile) {
          const int t0 = tile * kTile, nt = ntok - t0 < kTile ? ntok - t0 : kTile;
          const size_t slot = (static_cast<size_t>(pages[tile]) * hkv + kvh) * kPage;  // the tile's first slot
          esimd::barrier();  // the previous tile's readers are done
#pragma unroll 1
          for (int u = i; u < 2 * kUnits; u += NI) {
            const bool isv = u >= kUnits;
            const int uu = isv ? u - kUnits : u, sub = uu / (kD / kUw), gw = uu % (kD / kUw);
            const esimd::simd<int32_t, kOp> rowi = esimd::simd<int32_t, kOp>(0, 1) >> 4;
            const int live = nt - sub * 16;  // rows of this 16-token block holding tokens of the block's range
            if constexpr (I8) {
              const uint8_t *base = (isv ? vp : kp) + slot * kD;
              esimd::simd<uint8_t, 16 * kUw> raw = esimd::load_2d<uint8_t, kUw, 16>(base, kD - 1, kPage - 1, kD - 1, gw * kUw, sub * 16);
              auto r2 = raw.template bit_cast_view<int8_t, 16, kUw>();
#pragma unroll
              for (int d4 = 0; d4 < kUdc; ++d4) {
                const esimd::simd<int8_t, kOp> sb = r2.template select<16, 1, 16, 1>(0, d4 * 16);
                const esimd::simd<float, kOp> f = sb;
                esimd::simd<bf16, kOp> b = f;
                const int dc = gw * kUdc + d4;
                if (isv) {
                  b.merge(esimd::simd<bf16, kOp>(0.f), rowi >= live);
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kVOff + (sub * kDc + dc) * kOpBytes), v_operand(b));
                } else {
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kKOff + (dc * kSubs + sub) * kOpBytes), k_operand(b));
                }
              }
            } else {
              const uint16_t *base = reinterpret_cast<const uint16_t *>(isv ? vp : kp) + slot * kD;
              esimd::simd<uint16_t, 16 * kUw> raw = esimd::load_2d<uint16_t, kUw, 16>(base, kD * 2 - 1, kPage - 1, kD * 2 - 1, gw * kUw, sub * 16);
              auto r2 = raw.template bit_cast_view<uint16_t, 16, kUw>();
#pragma unroll
              for (int d2 = 0; d2 < kUdc; ++d2) {
                esimd::simd<uint16_t, kOp> sb = r2.template select<16, 1, 16, 1>(0, d2 * 16);
                const int dc = gw * kUdc + d2;
                if (isv) {
                  sb.merge(esimd::simd<uint16_t, kOp>(0), rowi >= live);
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kVOff + (sub * kDc + dc) * kOpBytes), v_operand(sb.template bit_cast_view<bf16>()));
                } else {
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kKOff + (dc * kSubs + sub) * kOpBytes), k_operand(sb.template bit_cast_view<bf16>()));
                }
              }
            }
          }
          esimd::barrier();  // the tile is staged
          if (tile + 1 < ntiles) {  // the next tile's units into the caches while this one computes
            const size_t nslot = (static_cast<size_t>(pages[tile + 1]) * hkv + kvh) * kPage;
            constexpr auto pfp = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>};
#pragma unroll 1
            for (int u = i; u < 2 * kUnits; u += NI) {
              const bool isv = u >= kUnits;
              const int uu = isv ? u - kUnits : u, sub = uu / (kD / kUw), gw = uu % (kD / kUw);
              if constexpr (I8) {
                esimd::prefetch_2d<uint8_t, kUw, 16>((isv ? vp : kp) + nslot * kD, kD - 1, kPage - 1, kD - 1, gw * kUw, sub * 16, pfp);
              } else {
                esimd::prefetch_2d<uint16_t, kUw, 16>(reinterpret_cast<const uint16_t *>(isv ? vp : kp) + nslot * kD, kD * 2 - 1, kPage - 1, kD * 2 - 1, gw * kUw, sub * 16, pfp);
              }
            }
          }
          esimd::simd<float, kTile> ks(1.f), vs(1.f);
          if constexpr (I8) {
            ks = esimd::block_load<float, kTile>(ksp + slot);
            vs = esimd::block_load<float, kTile>(vsp + slot);
          }
          // ---- q.k: S [sub][pair][16 tokens] ----
          esimd::simd<float, kSubs * 128> S(-INFINITY);
#pragma unroll
          for (int sub = 0; sub < kSubs; ++sub) {
            if (sub * 16 < nt) {
              esimd::simd<float, 128> cc(0.f);
#pragma unroll
              for (int dc = 0; dc < kDc; ++dc) {
                const esimd::simd<bf16, kOp> b = esimd::slm_block_load<bf16, kOp>(static_cast<uint32_t>(kKOff + (dc * kSubs + sub) * kOpBytes));
                const esimd::simd<bf16, 128> aa = qa.template select<128, 1>(dc * 128);
                cc = xmx::dpas<8, 8, float, float, bf16, bf16>(cc, b, aa);
              }
              S.template select<128, 1>(sub * 128) = cc;
            }
          }
          // ---- online softmax per pair, o rescaled; P (x the V scale) back into S ----
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            esimd::simd<float, kTile> sv;
#pragma unroll
            for (int sub = 0; sub < kSubs; ++sub) sv.template select<16, 1>(sub * 16) = S.template select<16, 1>(sub * 128 + r * 16);
            sv = sv * kmul;
            if constexpr (I8) sv *= ks;
            const int rp = rpos[r];
            const esimd::simd_mask<kTile> dead = (ti + t0 > rp) | (ti >= nt);
            sv.merge(esimd::simd<float, kTile>(-INFINITY), dead);
            const float mt = esimd::hmax<float>(sv);
            const float mo = m_run[r], mn = mo > mt ? mo : mt;
            const esimd::simd<float, 1> al = esimd::exp(esimd::simd<float, 1>(mo - mn));  // 0 on the first tile
            esimd::simd<float, kTile> pe = esimd::exp(sv - mn);
            pe.merge(esimd::simd<float, kTile>(0.f), dead);
            l_run[r] = l_run[r] * static_cast<float>(al[0]) + esimd::reduce<float>(pe, std::plus<>());
            m_run[r] = mn;
            if constexpr (I8) {
              pe *= vs;
              pe.merge(esimd::simd<float, kTile>(0.f), dead);  // an unwritten slot's scale may be anything
            }
#pragma unroll
            for (int sub = 0; sub < kSubs; ++sub) S.template select<16, 1>(sub * 128 + r * 16) = pe.template select<16, 1>(sub * 16);
#pragma unroll
            for (int d = 0; d < 8; ++d) oa.template select<16, 1>(d * 128 + r * 16) *= static_cast<float>(al[0]);
          }
          // ---- P.V on this half's 128 dims ----
#pragma unroll
          for (int sub = 0; sub < kSubs; ++sub) {
            if (sub * 16 >= nt) continue;
            const esimd::simd<float, 128> pw = S.template select<128, 1>(sub * 128);
            const esimd::simd<bf16, 128> ph = pw;
            const esimd::simd<bf16, 128> pl = pw - esimd::simd<float, 128>(ph);
#pragma unroll
            for (int d = 0; d < 8; ++d) {
              const int dc = half * 8 + d;
              const esimd::simd<bf16, kOp> b = esimd::slm_block_load<bf16, kOp>(static_cast<uint32_t>(kVOff + (sub * kDc + dc) * kOpBytes));
              esimd::simd<float, 128> o8 = oa.template select<128, 1>(d * 128);
              o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, b, ph);
              o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, b, pl);
              oa.template select<128, 1>(d * 128) = o8;
            }
          }
        }
        // ---- finish: o / l, bf16, gate ----
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int pr = pb * 8 + r, j = pr / G;
          if (j >= nrows) continue;
          const int row = r0 + j, h = kvh * G + pr % G;
          const float L = l_run[r];
          const float *gp = y + static_cast<size_t>(row) * ldy + h * 2 * kD + kD + half * 128;
          bf16 *op = attn + (static_cast<size_t>(row) * hq + h) * kD + half * 128;
#pragma unroll
          for (int d = 0; d < 8; ++d) {
            const esimd::simd<float, 16> ov = oa.template select<16, 1>(d * 128 + r * 16);
            const esimd::simd<float, 16> a = esimd::simd<bf16, 16>(ov / L);
            const esimd::simd<float, 16> gt = esimd::simd<bf16, 16>(esimd::block_load<float, 16>(gp + d * 16));
            const esimd::simd<float, 16> sg = esimd::simd<bf16, 16>(esimd::simd<float, 16>(1.f) / (esimd::simd<float, 16>(1.f) + esimd::exp(-gt)));
            esimd::block_store<bf16, 16>(op + d * 16, esimd::simd<bf16, 16>(a * sg));
          }
        }
      });
}

}  // namespace

sycl::event gqa_attn_staged(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy,
                            const gqa_bf16 *qb, gqa_bf16 *attn, int rows_per_group) {
  if (s.kv == QsaKv::i8v4) throw std::invalid_argument("gqa_attn_staged: int8 | bf16 KV (int8-v4 runs the fa body)");
  const bool i8 = s.kv == QsaKv::i8;
  if (c.slot_pages) throw std::invalid_argument("gqa_attn_staged: one slot (page table without slot stride)");
  if (s.grp() == 6) {
    if (rows_per_group == 8) return i8 ? attn_staged<QsaKv::i8, 6, 8>(q, s, m, c, pos_word, y, ldy, qb, attn) : attn_staged<QsaKv::bf16, 6, 8>(q, s, m, c, pos_word, y, ldy, qb, attn);
    if (rows_per_group == 0 || rows_per_group == 16)
      return i8 ? attn_staged<QsaKv::i8, 6, 16>(q, s, m, c, pos_word, y, ldy, qb, attn) : attn_staged<QsaKv::bf16, 6, 16>(q, s, m, c, pos_word, y, ldy, qb, attn);
  }
  if (s.grp() == 12) {
    if (rows_per_group == 4) return i8 ? attn_staged<QsaKv::i8, 12, 4>(q, s, m, c, pos_word, y, ldy, qb, attn) : attn_staged<QsaKv::bf16, 12, 4>(q, s, m, c, pos_word, y, ldy, qb, attn);
    if (rows_per_group == 0 || rows_per_group == 8)
      return i8 ? attn_staged<QsaKv::i8, 12, 8>(q, s, m, c, pos_word, y, ldy, qb, attn) : attn_staged<QsaKv::bf16, 12, 8>(q, s, m, c, pos_word, y, ldy, qb, attn);
  }
  throw std::invalid_argument("gqa_attn_staged: built for 6 (8 / 16 rows per group) or 12 (4 / 8) q heads per kv head");
}

}  // namespace xenithium::kernels
