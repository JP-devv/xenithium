// Xenithium · the many-row gated GQA attention (prefill,; see gqa.hpp, gqa_rows).
//
// One work-group per (block of QB consecutive rows, kv head): its P = QB x G (row, q head) pairs read every K / V
// tile once (decode reads it once per row). The decode attention's arithmetic per tile (gqa.cpp): 16 items x 16
// dims, the partial q.k of every pair summed in SLM in item order, scale 1/16 (x the token's K scale for int8 KV),
// a running max / sum per pair, o rescaled by exp(m_old - m_new), P V on DPAS with P = hi + lo bf16 (x the V
// scale). New: a causal mask per pair (token > the pair's row position -> -inf), no context splits (the rows give
// the parallelism), and the finish fused: attn = bf16(bf16(o / l) bf16(sigmoid(bf16(gate)))), merge's rounding.
// Its own TU at 256 GRF (xenithium_kernels_xl): P x 16 fp32 accumulators per item.
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

template <QsaKv KV, int G, int QB>
class GqaAttnRowsK;

template <QsaKv KV, int G, int QB>
sycl::event attn_rows(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy, const bf16 *qb,
                      bf16 *attn) {
  constexpr bool I8 = KV == QsaKv::i8;
  constexpr int kI = 16, kE = I8 ? 1 : 2, P = G * QB;
  static_assert(P % 16 == 0, "pairs per work-group: a multiple of 16");
  constexpr int kPI = P / kI;  // pairs whose softmax state an item owns
  constexpr int kTokOff = 0, kKsOff = kTokOff + kTile, kVsOff = kKsOff + kTile;
  constexpr int kSpOff = kVsOff + kTile, kSpN = kI * P * 16;
  constexpr int kScOff = kSpOff + kSpN, kPvOff = kScOff + P * kTile, kAlOff = kPvOff + P * kTile;
  constexpr int kSlm = (kAlOff + P) * 4;
  static_assert(kSlm <= 128 * 1024, "SLM");
  const int hq = s.hq(), hkv = s.hkv(), blocks = (m + QB - 1) / QB;
  const uint8_t *kp = c.k, *vp = c.v;
  const float *ksp = c.ks, *vsp = c.vs;
  const int32_t *pages = c.pages;
  return q.parallel_for<GqaAttnRowsK<KV, G, QB>>(
      sycl::nd_range<1>(static_cast<size_t>(blocks) * hkv * kI, kI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kSlm>();
        const int g = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0));
        const int blk = g / hkv, kvh = g % hkv, r0 = blk * QB;
        const int nrows = m - r0 < QB ? m - r0 : QB;
        const int p0 = pos_word[0] + r0;            // position of the block's first row
        const int ntok = p0 + nrows, ntiles = (ntok + kTile - 1) / kTile;
        auto so = [](int f) SYCL_ESIMD_FUNCTION { return static_cast<uint32_t>(f * 4); };
        esimd::simd<bf16, P * 16> qa;  // A operand: [pair = row j x G + head h][16 dims of this item]
#pragma unroll
        for (int pr = 0; pr < P; ++pr) {
          const int j = pr / G, h = pr % G, rr = j < nrows ? r0 + j : r0;  // dead rows repeat the block's first
          qa.template select<16, 1>(pr * 16) = esimd::block_load<bf16, 16>(qb + (static_cast<size_t>(rr) * hq + kvh * G + h) * kD + i * 16);
        }
        auto tile16 = [&](const uint8_t *base, esimd::simd<int32_t, 16> sl) SYCL_ESIMD_FUNCTION {
          esimd::simd<bf16, 256> out;
          if constexpr (I8) {
            esimd::simd<uint8_t, 256> raw;
#pragma unroll
            for (int tl = 0; tl < 16; ++tl)
              raw.template select<16, 1>(tl * 16) = esimd::block_load<uint8_t, 16>(base + static_cast<size_t>(sl[tl]) * kD + i * 16);
            esimd::simd<int8_t, 256> sb = raw.template bit_cast_view<int8_t>();
            const esimd::simd<float, 256> f = sb;
            out = f;
          } else {
#pragma unroll
            for (int tl = 0; tl < 16; ++tl)
              out.template select<16, 1>(tl * 16) =
                  esimd::block_load<bf16, 16>(reinterpret_cast<const bf16 *>(base + (static_cast<size_t>(sl[tl]) * kD + i * 16) * kE));
          }
          return out;
        };
        const float kmul = 1.f / 16.f;  // 1 / sqrt(256)
        esimd::simd<float, P * 16> oa(0.f);
        esimd::simd<float, kPI> m_run(-INFINITY), l_run(0.f);  // pairs i + 16 k
#pragma unroll 1
        for (int tile = 0; tile < ntiles; ++tile) {
          const int t0 = tile * kTile, nt = ntok - t0 < kTile ? ntok - t0 : kTile;
          esimd::barrier();
          {
            constexpr int kPer = kTile / kI;
            esimd::simd<int32_t, kPer> sl;
            esimd::simd<float, kPer> kss(1.f), vss(1.f);
#pragma unroll
            for (int jj = 0; jj < kPer; ++jj) {
              int t = i * kPer + jj;
              t = t < nt ? t : nt - 1;
              const int tok = t0 + t;
              const int slot = (pages[tok / kPage] * hkv + kvh) * kPage + (tok % kPage);
              sl[jj] = slot;
              if constexpr (I8) {
                kss[jj] = ksp[slot];
                vss[jj] = vsp[slot];
              }
            }
            esimd::slm_block_store<int32_t, kPer>(so(kTokOff + i * kPer), sl);
            esimd::slm_block_store<float, kPer>(so(kKsOff + i * kPer), kss);
            esimd::slm_block_store<float, kPer>(so(kVsOff + i * kPer), vss);
          }
          esimd::barrier();
          // ---- scores, 16 tokens at a time ----
#pragma unroll 1
          for (int sub = 0; sub < kTile / 16; ++sub) {
            if (sub * 16 >= nt) break;
            const esimd::simd<int32_t, 16> sl = esimd::slm_block_load<int32_t, 16>(so(kTokOff + sub * 16));
            esimd::simd<bf16, 256> kb = tile16(kp, sl);
            esimd::simd<uint32_t, 128> bd;
            {
              esimd::simd<uint32_t, 128> kd = kb.template bit_cast_view<uint32_t>();
#pragma unroll
              for (int pp = 0; pp < 8; ++pp) bd.template select<16, 1>(pp * 16) = kd.template select<16, 8>(pp);
            }
            const esimd::simd<bf16, 256> bk = bd.template bit_cast_view<bf16>();
#pragma unroll
            for (int c8 = 0; c8 < P / 8; ++c8) {
              const esimd::simd<bf16, 128> aa = qa.template select<128, 1>(c8 * 128);
              esimd::simd<float, 128> ca(0.f);
              ca = xmx::dpas<8, 8, float, float, bf16, bf16>(ca, bk, aa);
              esimd::slm_block_store<float, 128>(so(kSpOff + i * P * 16 + c8 * 128), ca);
            }
            esimd::barrier();
            // item i sums outputs [i P, i P + P) over the 16 items, in item order; output o = pair x 16 + token
            esimd::simd<float, P> acc(0.f);
#pragma unroll
            for (int it2 = 0; it2 < kI; ++it2)
#pragma unroll
              for (int k3 = 0; k3 < P / 16; ++k3)
                acc.template select<16, 1>(k3 * 16) += esimd::slm_block_load<float, 16>(so(kSpOff + it2 * P * 16 + i * P + k3 * 16));
            const esimd::simd<float, 16> ks = esimd::slm_block_load<float, 16>(so(kKsOff + sub * 16));
            const esimd::simd<int32_t, 16> tok = esimd::simd<int32_t, 16>(0, 1) + (t0 + sub * 16);
            // outputs [i P, i P + P) are pairs i P / 16 .. + P / 16 - 1, 16 tokens each: one block store per pair
#pragma unroll
            for (int k3 = 0; k3 < P / 16; ++k3) {
              const int pr = i * (P / 16) + k3, row_pos = p0 + pr / G;
              esimd::simd<float, 16> sv = acc.template select<16, 1>(k3 * 16) * kmul;
              if constexpr (I8) sv *= ks;
              sv.merge(esimd::simd<float, 16>(-INFINITY), tok > row_pos);  // causal
              esimd::slm_block_store<float, 16>(so(kScOff + pr * kTile + sub * 16), sv);
            }
            esimd::barrier();
          }
          // ---- running softmax per pair (an item owns pairs i, i + 16, ...) ----
#pragma unroll
          for (int k2 = 0; k2 < kPI; ++k2) {
            const int pr = i + 16 * k2;
            esimd::simd<float, kTile> sv = esimd::slm_block_load<float, kTile>(so(kScOff + pr * kTile));
            esimd::simd<int32_t, kTile> ti(0, 1);
            sv.merge(esimd::simd<float, kTile>(-INFINITY), ti >= nt);
            const float mt = esimd::hmax<float>(sv);
            const float mo = m_run[k2], mn = mo > mt ? mo : mt;
            const esimd::simd<float, 1> al = esimd::exp(esimd::simd<float, 1>(mo - mn));  // 0 on the first tile
            esimd::simd<float, kTile> pe = esimd::exp(sv - mn);
            pe.merge(esimd::simd<float, kTile>(0.f), ti >= nt);
            l_run[k2] = l_run[k2] * static_cast<float>(al[0]) + esimd::reduce<float>(pe, std::plus<>());
            m_run[k2] = mn;
            if constexpr (I8) pe *= esimd::slm_block_load<float, kTile>(so(kVsOff));
            esimd::slm_block_store<float, kTile>(so(kPvOff + pr * kTile), pe);
            esimd::slm_scalar_store<float>(so(kAlOff + pr), al[0]);
          }
          esimd::barrier();
          {
            esimd::simd<float, P> al;
#pragma unroll
            for (int k3 = 0; k3 < P / 16; ++k3) al.template select<16, 1>(k3 * 16) = esimd::slm_block_load<float, 16>(so(kAlOff + k3 * 16));
#pragma unroll
            for (int pr = 0; pr < P; ++pr) oa.template select<16, 1>(pr * 16) *= static_cast<float>(al[pr]);
          }
          // ---- P V on DPAS ----
#pragma unroll 1
          for (int sub = 0; sub < kTile / 16; ++sub) {
            if (sub * 16 >= nt) break;
            const esimd::simd<int32_t, 16> sl = esimd::slm_block_load<int32_t, 16>(so(kTokOff + sub * 16));
            esimd::simd<bf16, 256> vb = tile16(vp, sl);
            esimd::simd<bf16, 256> bv;
#pragma unroll
            for (int qq = 0; qq < 8; ++qq) {
              bv.template select<16, 2>(qq * 32) = vb.template select<16, 1>(2 * qq * 16);
              bv.template select<16, 2>(qq * 32 + 1) = vb.template select<16, 1>((2 * qq + 1) * 16);
            }
#pragma unroll
            for (int c8 = 0; c8 < P / 8; ++c8) {
              esimd::simd<float, 128> pw;
#pragma unroll
              for (int h = 0; h < 8; ++h) pw.template select<16, 1>(h * 16) = esimd::slm_block_load<float, 16>(so(kPvOff + (c8 * 8 + h) * kTile + sub * 16));
              esimd::simd<bf16, 128> ph = pw;
              esimd::simd<bf16, 128> pl = pw - esimd::simd<float, 128>(ph);
              esimd::simd<float, 128> o8 = oa.template select<128, 1>(c8 * 128);
              o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, bv, ph);
              o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, bv, pl);
              oa.template select<128, 1>(c8 * 128) = o8;
            }
          }
        }
        // ---- finish: o / l, bf16, gate ----
        esimd::barrier();
#pragma unroll
        for (int k2 = 0; k2 < kPI; ++k2) esimd::slm_scalar_store<float>(so(kAlOff + i + 16 * k2), l_run[k2]);
        esimd::barrier();
        esimd::simd<float, P> L;
#pragma unroll
        for (int k3 = 0; k3 < P / 16; ++k3) L.template select<16, 1>(k3 * 16) = esimd::slm_block_load<float, 16>(so(kAlOff + k3 * 16));
#pragma unroll
        for (int pr = 0; pr < P; ++pr) {
          const int j = pr / G, hh = pr % G;
          if (j >= nrows) continue;
          const int r = r0 + j, h = kvh * G + hh;
          const esimd::simd<float, 16> ov = oa.template select<16, 1>(pr * 16);
          const esimd::simd<float, 16> a = esimd::simd<bf16, 16>(ov / static_cast<float>(L[pr]));
          const esimd::simd<float, 16> gt = esimd::simd<bf16, 16>(esimd::block_load<float, 16>(y + static_cast<size_t>(r) * ldy + h * 2 * kD + kD + i * 16));
          const esimd::simd<float, 16> sg = esimd::simd<bf16, 16>(esimd::simd<float, 16>(1.f) / (esimd::simd<float, 16>(1.f) + esimd::exp(-gt)));
          esimd::block_store<bf16, 16>(attn + (static_cast<size_t>(r) * hq + h) * kD + i * 16, esimd::simd<bf16, 16>(a * sg));
        }
      });
}

}  // namespace

sycl::event gqa_attn_rows(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy,
                          const gqa_bf16 *qb, gqa_bf16 *attn) {
  if (s.kv == QsaKv::i8v4) throw std::invalid_argument("gqa_attn_rows: int8 | bf16 KV (int8-v4 runs the fa body)");
  const bool i8 = s.kv == QsaKv::i8;
  if (s.grp() == 6) {
    return i8 ? attn_rows<QsaKv::i8, 6, 8>(q, s, m, c, pos_word, y, ldy, qb, attn) : attn_rows<QsaKv::bf16, 6, 8>(q, s, m, c, pos_word, y, ldy, qb, attn);
  }
  if (s.grp() == 12) return i8 ? attn_rows<QsaKv::i8, 12, 4>(q, s, m, c, pos_word, y, ldy, qb, attn) : attn_rows<QsaKv::bf16, 12, 4>(q, s, m, c, pos_word, y, ldy, qb, attn);
  throw std::invalid_argument("gqa_attn_rows: built for 6 or 12 q heads per kv head");
}

}  // namespace xenithium::kernels
