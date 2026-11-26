// Xenithium · GDN prefill recurrence as a chunked bf16-DPAS scan.
#include "xenithium/kernels/linear_attention/gdn_scan.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <functional>

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = gdn_bf16;

class GsPre;
class GsScan;

constexpr int kD = kGdnDim;  // 128
constexpr int kC = kGdnChunk;  // 64 rows per chunk
constexpr int kItems = 16;  // items per work-group (both kernels)
constexpr int kRows = 8;  // value rows of S per scan item (kItems x kRows = 128)
// Per (chunk, value head), elements. B blocks are 256 bf16 (16 k x 16 n, VNNI: element
// ((k % 16) / 2) * 32 + (n % 16) * 2 + (k & 1)).
constexpr size_t kWkE = static_cast<size_t>(kC) * kD;  // Wk^T: k = c, n = i, blocks [ib 4][cb 8]; Qg^T the same
constexpr size_t kKtE = kWkE;  // Kt: k = i, n = c, blocks [cb 8][ib 4]
constexpr size_t kMtE = static_cast<size_t>(kC) * kC;  // M^T: k = j, n = i, blocks [ib 4][jb 4] (jb <= ib written)
constexpr size_t kUvE = static_cast<size_t>(kD) * kC;  // Uv^T fp32 [r 128][i 64]

// SLM block load / store of a multiple of 256 bytes (one message moves at most 256 bytes).
template <class T, int N>
inline esimd::simd<T, N> sld(uint32_t off) SYCL_ESIMD_FUNCTION {
  constexpr int P = 256 / sizeof(T);
  if constexpr (N <= P) {
    return esimd::slm_block_load<T, N>(off);
  } else {
    esimd::simd<T, N> v;
#pragma unroll
    for (int k = 0; k < N / P; ++k) v.template select<P, 1>(k * P) = esimd::slm_block_load<T, P>(off + k * 256);
    return v;
  }
}
template <class T, int N>
inline void sst(uint32_t off, esimd::simd<T, N> v) SYCL_ESIMD_FUNCTION {
  constexpr int P = 256 / sizeof(T);
  if constexpr (N <= P) {
    esimd::slm_block_store<T, N>(off, v);
  } else {
#pragma unroll
    for (int k = 0; k < N / P; ++k) esimd::slm_block_store<T, P>(off + k * 256, v.template select<P, 1>(k * P));
  }
}
inline float exp1(float x) SYCL_ESIMD_FUNCTION { return esimd::exp(esimd::simd<float, 1>(x))[0]; }

// rows [4][128] fp32 -> bf16 pairs as dwords, transposed to [64 c pairs][4 rows] (a VNNI fragment)
inline esimd::simd<uint32_t, 256> frag(esimd::simd<float, 4 * kD> rows) SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, 4 * kD> b(rows);
  esimd::simd<uint32_t, 4 * kD / 2> d = b.template bit_cast_view<uint32_t>();
  esimd::simd<uint32_t, 256> t;
#pragma unroll
  for (int r = 0; r < 4; ++r) t.template select<64, 4>(r) = d.template select<64, 1>(r * 64);
  return t;
}

// gdn_rows' norm helpers, verbatim (the folded norm must give its bits)
template <int N>
inline esimd::simd<float, N> round_bf16(esimd::simd<float, N> v) SYCL_ESIMD_FUNCTION {
  return esimd::simd<float, N>(esimd::simd<bf16, N>(v));
}
template <int N>
inline esimd::simd<float, N> sigmoid_v(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  return esimd::inv(esimd::simd<float, N>(1.f) + esimd::exp(-x));
}
inline float rsqrt_ieee(float v) SYCL_ESIMD_FUNCTION {
  return esimd::div_ieee(esimd::simd<float, 1>(1.f), esimd::sqrt_ieee(esimd::simd<float, 1>(v)))[0];
}

inline int clamp_accept(int a) { return a < 0 ? 0 : (a > kGdnRows ? kGdnRows : a); }

}  // namespace

GdnScanSizes gdn_scan_sizes(const GdnShape &s, int rows_max) {
  const size_t units = static_cast<size_t>(gdn_scan_max_chunks(rows_max)) * s.vh();
  return GdnScanSizes{units * kWkE, units * kMtE, units * kUvE, units};
}

sycl::event gdn_scan(sycl::queue &q, const GdnShape &sh, int m, int pos, const int32_t *accept, const float *qn, const float *kn, const float *cv,
                     const float *gate, const float *lgate, float *S, bf16 *o, const GdnScanScratch &sc, const GdnScanNorm &nm, int part, int snap_pos,
                     sycl::event *pre_ev) {
  const int vh = sh.vh(), rep = sh.rep(), KW = sh.key_w(), VW = sh.val_w(), CC = sh.conv_ch();
  const int nmax = gdn_scan_max_chunks(m);
  if (nmax == 0) return {};
  bf16 *wk = sc.wk, *qg = sc.qg, *kt = sc.kt, *mt = sc.mt, *ktl = sc.kt_lo;
  float *uv = sc.uv, *gcv = sc.gc;

  // ---- pre: one work-group of 16 items per (chunk, value head); nothing here reads S (part 2 reuses part 1's)
  sycl::event pre;
  if (part != 2) pre = q.parallel_for<GsPre>(sycl::nd_range<1>(static_cast<size_t>(nmax) * vh * kItems, kItems), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    // SLM (bytes), 64 KB (more halves the resident work-groups):
    // kKh / kKl K^T hi / lo: k = c, n = j, blocks [cb 8][jb 4] (gram B; Wk^T A source)
    // kTT T^T rows fp32 [l][i] (phases 2-3), then
    // kV V^T (exact bf16): k = c, n = l, blocks [cb 8][lb 4] (Uv^T A source, phase 4)
    // kTh / kTl Ti^T hi / lo: k = l, n = i, blocks [lb 4][ib 4] (phase 4 B)
    constexpr uint32_t kKh = 0, kKl = 16384, kTT = 32768, kV = 32768, kTh = 49152, kTl = 57344;
    esimd::slm_init<65536>();
    const int grp = static_cast<int>(it.get_group(0)), li = static_cast<int>(it.get_local_id(0));
    const int ch = grp / vh, hv = grp % vh, khh = hv / rep;
    const int A = clamp_accept(accept[0]), T = A + m;
    int lead, full;
    gdn_scan_split(pos - A, T, lead, full);
    if (ch >= full) return;  // uniform per work-group
    const int s = lead + ch * kC;
    const size_t unit = static_cast<size_t>(ch) * vh + hv;

    // g = cumsum of ln decay over the chunk's rows (in order, fp32), beta
    esimd::simd<float, kC> lg, bv;
#pragma unroll
    for (int h = 0; h < kC / 16; ++h) {
      const esimd::simd<uint32_t, 16> rowi(static_cast<uint32_t>(s + h * 16), 1);
      lg.select<16, 1>(h * 16) = esimd::gather<float, 16>(lgate, (rowi * static_cast<uint32_t>(vh) + hv) * 4u);
      bv.select<16, 1>(h * 16) = esimd::gather<float, 16>(gate, (rowi * static_cast<uint32_t>(2 * vh) + (vh + hv)) * 4u);
    }
    esimd::simd<float, kC> g;
    {
      float run = 0.f;
#pragma unroll
      for (int i = 0; i < kC; ++i) {
        run += static_cast<float>(lg[i]);
        g[i] = run;
      }
    }
    const float glast = static_cast<float>(g[kC - 1]);

    // phase 1, rows i0.. i0 + 3: K^T hi / lo -> SLM; Qg^T and Kt -> global
    const int i0 = li * 4, ib0 = i0 / 16;
    esimd::simd<uint32_t, 256> voff;  // dword offsets of this item's fragment in [cb][jb] blocks
    {
      const esimd::simd<uint32_t, 256> ln(0, 1);
      const esimd::simd<uint32_t, 256> p = ln >> 2, r = ln & 3;
      voff = ((((p >> 3) * 4 + ib0) * 128) + (p & 7) * 16 + (i0 % 16) + r) * 4u;
    }
    esimd::simd<float, 4 * kD> kr, qr;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const size_t row = static_cast<size_t>(s + i0 + r);
      kr.select<kD, 1>(r * kD) = esimd::block_load<float, kD>(kn + row * KW + khh * kD);
      qr.select<kD, 1>(r * kD) = esimd::block_load<float, kD>(qn + row * KW + khh * kD);
    }
    {
      const esimd::simd<bf16, 4 * kD> kh(kr);
      esimd::simd<uint32_t, 256> fh = frag(kr), fl = frag(kr - esimd::simd<float, 4 * kD>(kh));
#pragma unroll
      for (int h = 0; h < 8; ++h) {
        esimd::slm_scatter<uint32_t, 32>(voff.select<32, 1>(h * 32) + kKh, fh.select<32, 1>(h * 32));
        esimd::slm_scatter<uint32_t, 32>(voff.select<32, 1>(h * 32) + kKl, fl.select<32, 1>(h * 32));
      }
    }
    {
      esimd::simd<float, 4 * kD> qs, ks;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float gi = static_cast<float>(g[i0 + r]);
        qs.select<kD, 1>(r * kD) = qr.select<kD, 1>(r * kD) * exp1(gi);
        ks.select<kD, 1>(r * kD) = kr.select<kD, 1>(r * kD) * exp1(glast - gi);
      }
      // Qg^T: blocks [ib][cb], this item's [64 c pairs][4 rows] fragment by dword scatter
      {
        const esimd::simd<uint32_t, 256> ln(0, 1);
        const esimd::simd<uint32_t, 256> p = ln >> 2, r = ln & 3;
        esimd::simd<uint32_t, 256> qoff = (((ib0 * 8 + (p >> 3)) * 128) + (p & 7) * 16 + (i0 % 16) + r) * 4u;
        esimd::simd<uint32_t, 256> fq = frag(qs);
        uint32_t *q32 = reinterpret_cast<uint32_t *>(qg + unit * kWkE);
#pragma unroll
        for (int h = 0; h < 8; ++h) esimd::scatter<uint32_t, 32>(q32, qoff.select<32, 1>(h * 32), fq.select<32, 1>(h * 32));
      }
      // Kt: blocks [cb][ib], 64 bf16 per c block (this item's 2 row pairs)
      bf16 *ku = kt + unit * kKtE;
#pragma unroll
      for (int cb = 0; cb < 8; ++cb) {
        esimd::simd<float, 64> o64;
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
          o64.select<16, 2>(pp * 32) = ks.select<16, 1>((2 * pp) * kD + cb * 16);
          o64.select<16, 2>(pp * 32 + 1) = ks.select<16, 1>((2 * pp + 1) * kD + cb * 16);
        }
        const esimd::simd<bf16, 64> oh(o64);
        esimd::block_store<bf16, 64>(ku + (cb * 4 + ib0) * 256 + ((i0 % 16) / 2) * 32, oh);
        esimd::block_store<bf16, 64>(ktl + unit * kKtE + (cb * 4 + ib0) * 256 + ((i0 % 16) / 2) * 32,
                                     esimd::simd<bf16, 64>(o64 - esimd::simd<float, 64>(oh)));
      }
    }
    esimd::barrier();

    // phase 2, rows i0.. i0 + 3: [K; Q] rows x K^T on DPAS (hi / lo: hh + lh + hl) -> gram rows;
    // T^T rows to SLM (K K^T is symmetric: row l gives T[i][l] = b_i e^(g_i - g_l) k_i.k_l, i > l);
    // M rows -> M^T blocks
    {
      esimd::simd<float, 4 * 128> acc(0.f);  // [jb][8 rows: 4 k, 4 q][16 j]
#pragma unroll
      for (int cb = 0; cb < 8; ++cb) {
        esimd::simd<float, 128> af;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          af.select<16, 1>(r * 16) = kr.select<16, 1>(r * kD + cb * 16);
          af.select<16, 1>((4 + r) * 16) = qr.select<16, 1>(r * kD + cb * 16);
        }
        const esimd::simd<bf16, 128> ah(af);
        const esimd::simd<bf16, 128> al(af - esimd::simd<float, 128>(ah));
#pragma unroll
        for (int jb = 0; jb < 4; ++jb) {
          const uint32_t off = static_cast<uint32_t>((cb * 4 + jb) * 512);
          const esimd::simd<bf16, 256> bh = sld<bf16, 256>(kKh + off), bl = sld<bf16, 256>(kKl + off);
          esimd::simd<float, 128> a = acc.select<128, 1>(jb * 128);
          a = xmx::dpas<8, 8, float, float, bf16, bf16>(a, bh, ah);
          a = xmx::dpas<8, 8, float, float, bf16, bf16>(a, bl, ah);
          a = xmx::dpas<8, 8, float, float, bf16, bf16>(a, bh, al);
          acc.select<128, 1>(jb * 128) = a;
        }
      }
      esimd::simd<float, 4 * kC> qk;
      const esimd::simd<int32_t, kC> jj(0, 1);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        esimd::simd<float, kC> kkr, qkr;
#pragma unroll
        for (int jb = 0; jb < 4; ++jb) {
          kkr.select<16, 1>(jb * 16) = acc.select<16, 1>(jb * 128 + r * 16);
          qkr.select<16, 1>(jb * 16) = acc.select<16, 1>(jb * 128 + (4 + r) * 16);
        }
        const int i = i0 + r;
        const float gi = static_cast<float>(g[i]);
        esimd::simd<float, kC> at = g - gi;  // T^T row i: entries t > i
        at.merge(0.f, at > 0.f);
        esimd::simd<float, kC> tt = esimd::exp(at) * bv * kkr;
        tt.merge(0.f, jj <= i);
        sst<float, kC>(kTT + static_cast<uint32_t>(i * kC * 4), tt);
        esimd::simd<float, kC> am = gi - g;  // M row i: entries j <= i
        am.merge(0.f, am > 0.f);
        esimd::simd<float, kC> mr = esimd::exp(am) * qkr;
        mr.merge(0.f, jj > i);
        qk.select<kC, 1>(r * kC) = mr;
      }
      bf16 *mu = mt + unit * kMtE;
#pragma unroll
      for (int p = 0; p < kC / 2; ++p) {
        const int jb = p / 8;
        if (jb <= ib0) {
          esimd::simd<float, 8> v8 = qk.bit_cast_view<float, 4, kC>().select<4, 1, 2, 1>(0, 2 * p).read();
          esimd::block_store<bf16, 8>(mu + (ib0 * 4 + jb) * 256 + (p % 8) * 32 + (i0 % 16) * 2, esimd::simd<bf16, 8>(v8));
        }
      }
    }
    esimd::barrier();

    // phase 3: Ti columns j0.. j0 + 3, forward substitution in column form (fp32): once x_l is final,
    // x_i -= T[i][l] x_l for all i (T^T row l, zero for i <= l); rows l < j0 hold x_l = 0 (no-ops).
    // Then Ti^T hi / lo blocks (k = l: this item's 2 l pairs, n = i): 64 bf16 per i block.
    {
      const int j0 = li * 4;
      esimd::simd<float, 4 * kC> x(0.f);
#pragma unroll
      for (int c = 0; c < 4; ++c) x[c * kC + j0 + c] = 1.f;
      for (int lb = j0 / 8; lb < kC / 8; ++lb) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int l = lb * 8 + u;
          const esimd::simd<float, kC> tl = sld<float, kC>(kTT + static_cast<uint32_t>(l * kC * 4));
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const float xl = static_cast<float>(x[c * kC + l]);
            x.select<kC, 1>(c * kC) -= tl * xl;
          }
        }
      }
#pragma unroll
      for (int ib = 0; ib < 4; ++ib) {
        esimd::simd<float, 64> o64;
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
          o64.select<16, 2>(pp * 32) = x.select<16, 1>((2 * pp) * kC + ib * 16);
          o64.select<16, 2>(pp * 32 + 1) = x.select<16, 1>((2 * pp + 1) * kC + ib * 16);
        }
        const esimd::simd<bf16, 64> oh(o64);
        const esimd::simd<bf16, 64> ol(o64 - esimd::simd<float, 64>(oh));
        const uint32_t off = static_cast<uint32_t>((((j0 / 16) * 4 + ib) * 256 + ((j0 % 16) / 2) * 32) * 2);
        esimd::slm_block_store<bf16, 64>(kTh + off, oh);
        esimd::slm_block_store<bf16, 64>(kTl + off, ol);
      }
    }
    esimd::barrier();
    // V^T into the T^T region (phase 3 is done reading it)
    {
      esimd::simd<float, 4 * kD> vr;
#pragma unroll
      for (int r = 0; r < 4; ++r) vr.select<kD, 1>(r * kD) = esimd::block_load<float, kD>(cv + static_cast<size_t>(s + i0 + r) * CC + 2 * KW + hv * kD);
      esimd::simd<uint32_t, 256> fv = frag(vr);
#pragma unroll
      for (int h = 0; h < 8; ++h) esimd::slm_scatter<uint32_t, 32>(voff.select<32, 1>(h * 32) + kV, fv.select<32, 1>(h * 32));
    }
    esimd::barrier();

    // phase 4, rows c0.. c0 + 7 of the 128 dims: Uv^T = (V^T diag b) Ti^T and Wk^T = (K^T diag(b e^g)) Ti^T
    // on DPAS. A operands (8 c x 16 l) come from the VNNI blocks by stride-2 selects, scaled in fp32,
    // split hi / lo.
    {
      const int c0 = li * 8, cb = c0 / 16, cq = c0 % 16;
      esimd::simd<float, kC> sw = bv * esimd::exp(g);
      esimd::simd<float, 4 * 128> au(0.f), aw(0.f);  // [ib][8 c][16 i]
#pragma unroll
      for (int lb = 0; lb < 4; ++lb) {
        const uint32_t boff = static_cast<uint32_t>((cb * 4 + lb) * 512);
        esimd::simd<bf16, 256> vb = sld<bf16, 256>(kV + boff), khb = sld<bf16, 256>(kKh + boff), klb = sld<bf16, 256>(kKl + boff);
        const esimd::simd<float, 16> b16 = bv.select<16, 1>(lb * 16), w16 = sw.select<16, 1>(lb * 16);
        esimd::simd<float, 128> fu, fw;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const int cc = cq + c, sel = (cc / 2) * 32 + (cc & 1);
          fu.select<16, 1>(c * 16) = esimd::simd<float, 16>(vb.select<16, 2>(sel)) * b16;
          fw.select<16, 1>(c * 16) = (esimd::simd<float, 16>(khb.select<16, 2>(sel)) + esimd::simd<float, 16>(klb.select<16, 2>(sel))) * w16;
        }
        const esimd::simd<bf16, 128> uh(fu), wh(fw);
        const esimd::simd<bf16, 128> ul(fu - esimd::simd<float, 128>(uh)), wl(fw - esimd::simd<float, 128>(wh));
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) {
          const uint32_t toff = static_cast<uint32_t>((lb * 4 + ib) * 512);
          const esimd::simd<bf16, 256> th = sld<bf16, 256>(kTh + toff), tl = sld<bf16, 256>(kTl + toff);
          esimd::simd<float, 128> xa = au.select<128, 1>(ib * 128), ya = aw.select<128, 1>(ib * 128);
          xa = xmx::dpas<8, 8, float, float, bf16, bf16>(xa, th, uh);
          xa = xmx::dpas<8, 8, float, float, bf16, bf16>(xa, tl, uh);
          xa = xmx::dpas<8, 8, float, float, bf16, bf16>(xa, th, ul);
          ya = xmx::dpas<8, 8, float, float, bf16, bf16>(ya, th, wh);
          ya = xmx::dpas<8, 8, float, float, bf16, bf16>(ya, tl, wh);
          ya = xmx::dpas<8, 8, float, float, bf16, bf16>(ya, th, wl);
          au.select<128, 1>(ib * 128) = xa;
          aw.select<128, 1>(ib * 128) = ya;
        }
      }
      float *uu = uv + unit * kUvE;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        esimd::simd<float, kC> row;
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) row.select<16, 1>(ib * 16) = au.select<16, 1>(ib * 128 + c * 16);
        esimd::block_store<float, kC>(uu + (c0 + c) * kC, row);
      }
      // Wk^T blocks [ib][cb]: this item's 4 c pairs are 128 contiguous bf16
      bf16 *wu = wk + unit * kWkE;
#pragma unroll
      for (int ib = 0; ib < 4; ++ib) {
        esimd::simd<float, 128> ow;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          ow.select<16, 2>(p * 32) = aw.select<16, 1>(ib * 128 + (2 * p) * 16);
          ow.select<16, 2>(p * 32 + 1) = aw.select<16, 1>(ib * 128 + (2 * p + 1) * 16);
        }
        esimd::block_store<bf16, 128>(wu + (ib * 8 + cb) * 256 + (cq / 2) * 32, esimd::simd<bf16, 128>(ow));
      }
      if (li == 0) gcv[unit] = exp1(glast);
    }
  });
  if (pre_ev) *pre_ev = pre;

  // ---- scan: one work-group of 16 items per value head (item: 8 value rows of S in registers),
  // chunks in order; the work-group stages each chunk's B blocks in SLM. U and Kt enter as hi / lo bf16
  // pairs (O = Y + M Uh + M Ul; S += Kh Uh + Kl Uh + Kh Ul): plain bf16 U / Kt would lose state
  // accuracy. O goes through SLM so every output row is one
  // 256-byte store.
  const float *nz = nm.z, eps = nm.eps;
  const int ldz = nm.ldz;
  const bf16 *nw = nm.w;
  const bool nswish = nm.swish;
  bf16 *ny = nm.y;
  return q.parallel_for<GsScan>(sycl::nd_range<1>(static_cast<size_t>(vh) * kItems, kItems), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    constexpr uint32_t kSW = 0, kSQ = 16384, kSK = 32768, kSM = 49152, kSKL = 57344;  // Wk^T, Qg^T, Kt hi, M^T, Kt lo
    constexpr int kBlk = 144;  // staged 512-byte blocks
    constexpr uint32_t kSO = kBlk * 512;  // 2 x [16 rows][128] bf16 output tiles
    constexpr int kPer = kBlk / kItems;  // blocks staged per item
    esimd::slm_init<kSO + 2 * 16 * kD * 2>();
    const int hv = static_cast<int>(it.get_group(0)), li = static_cast<int>(it.get_local_id(0)), r0 = li * kRows;
    const int A = clamp_accept(accept[0]), T = A + m;
    int lead, full;
    gdn_scan_split(pos - A, T, lead, full);
    float *Sp = S + (static_cast<size_t>(hv) * kD + r0) * kD;
    esimd::simd<float, kRows * kD> Sa;  // [cb][8 rows][16]
#pragma unroll
    for (int rr = 0; rr < kRows; ++rr) {
      esimd::simd<float, kD> row = esimd::block_load<float, kD>(Sp + rr * kD);
#pragma unroll
      for (int cb = 0; cb < 8; ++cb) Sa.select<16, 1>(cb * 128 + rr * 16) = row.select<16, 1>(cb * 16);
    }
    const bool fold = ny != nullptr;  // the gated norm on the output rows
    int tile_ctr = 0;
    // part 1 / 2 = the chunks before / from the one starting at absolute snap_pos (S in
    // memory between the two launches: a snapshot of it there)
    int ch0 = 0, ch1 = full;
    if (part != 0) {
      int j = (snap_pos - (pos - A) - lead) / kC;
      j = j < 0 ? 0 : (j > full ? full : j);
      if (part == 1) ch1 = j;
      else ch0 = j;
    }
    for (int ch = ch0; ch < ch1; ++ch) {
      const int s = lead + ch * kC;
      const size_t unit = static_cast<size_t>(ch) * vh + hv;
      esimd::barrier();  // the previous chunk's SLM readers are done
      {
        esimd::simd<bf16, kPer * 256> blk;
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
          const int b = li * kPer + k;
          const bf16 *src = b < 32 ? wk + unit * kWkE + b * 256
                            : b < 64 ? qg + unit * kWkE + (b - 32) * 256
                            : b < 96 ? kt + unit * kKtE + (b - 64) * 256
                            : b < 112 ? mt + unit * kMtE + (b - 96) * 256
                                      : ktl + unit * kKtE + (b - 112) * 256;
          blk.select<256, 1>(k * 256) = esimd::block_load<bf16, 256>(src);
        }
#pragma unroll
        for (int k = 0; k < kPer; ++k) sst<bf16, 256>(static_cast<uint32_t>((li * kPer + k) * 512), blk.select<256, 1>(k * 256));
      }
      esimd::simd<float, kRows * kC> uvt;  // [rr][i]
#pragma unroll
      for (int rr = 0; rr < kRows; ++rr) uvt.select<kC, 1>(rr * kC) = esimd::block_load<float, kC>(uv + unit * kUvE + (r0 + rr) * kC);
      const float gc = gcv[unit];
      esimd::barrier();
      esimd::simd<bf16, kRows * kC> Ub, Ul;  // U hi / lo, [t][8][16]
      for (int t = 0; t < kC / 16; ++t) {
        esimd::simd<float, 128> X(0.f), Y(0.f);
#pragma unroll
        for (int cb = 0; cb < 8; ++cb) {
          const esimd::simd<bf16, 256> bw = sld<bf16, 256>(kSW + static_cast<uint32_t>((t * 8 + cb) * 512));
          const esimd::simd<bf16, 256> bq = sld<bf16, 256>(kSQ + static_cast<uint32_t>((t * 8 + cb) * 512));
          const esimd::simd<bf16, 128> a = Sa.select<128, 1>(cb * 128);
          X = xmx::dpas<8, 8, float, float, bf16, bf16>(X, bw, a);
          Y = xmx::dpas<8, 8, float, float, bf16, bf16>(Y, bq, a);
        }
        esimd::simd<float, 128> uf = uvt.bit_cast_view<float, kRows, kC>().select<kRows, 1, 16, 1>(0, t * 16).read();
        uf -= X;
        {
          const esimd::simd<bf16, 128> uh(uf);
          Ub.select<128, 1>(t * 128) = uh;
          Ul.select<128, 1>(t * 128) = esimd::simd<bf16, 128>(uf - esimd::simd<float, 128>(uh));
        }
        esimd::simd<float, 128> O = Y;
        for (int jb = 0; jb <= t; ++jb) {
          const esimd::simd<bf16, 256> bm = sld<bf16, 256>(kSM + static_cast<uint32_t>((t * 4 + jb) * 512));
          const esimd::simd<bf16, 128> a = Ub.select<128, 1>(jb * 128);
          O = xmx::dpas<8, 8, float, float, bf16, bf16>(O, bm, a);
          const esimd::simd<bf16, 128> al = Ul.select<128, 1>(jb * 128);
          O = xmx::dpas<8, 8, float, float, bf16, bf16>(O, bm, al);
        }
        // O^T tile [8 r][16 i] -> SLM rows [16 i][128 r] (double-buffered), then item li stores row li
        esimd::simd<float, 128> ot;  // [i][r]
#pragma unroll
        for (int i = 0; i < 16; ++i) ot.select<kRows, 1>(i * kRows) = O.select<kRows, 16>(i);
        esimd::simd<bf16, 128> ob(ot);
        esimd::simd<uint32_t, 64> ow = ob.bit_cast_view<uint32_t>();
        const esimd::simd<uint32_t, 64> lane(0, 1);
        const int row = s + t * 16 + li;
        const uint32_t buf = kSO + static_cast<uint32_t>(tile_ctr & 1) * (16 * kD * 2);
        esimd::simd<uint32_t, 64> off = buf + (lane >> 2) * (kD * 2) + static_cast<uint32_t>(r0 * 2) + (lane & 3) * 4u;
#pragma unroll
        for (int h = 0; h < 2; ++h) esimd::slm_scatter<uint32_t, 32>(off.select<32, 1>(h * 32), ow.select<32, 1>(h * 32));
        esimd::barrier();
        if (row >= A) {
          const esimd::simd<bf16, kD> orow = sld<bf16, kD>(buf + static_cast<uint32_t>(li * kD * 2));
          if (fold) {  // gdn_rows' norm expression, verbatim. w and z load here, per row, as there:
                       // w hoisted out of the chunk loop or z loaded before the barrier would change y's bits (fast math).
            const esimd::simd<float, kD> ov = orow;
            const float rr = rsqrt_ieee(esimd::reduce<float>(ov * ov, std::plus<>()) / static_cast<float>(kD) + eps);
            const esimd::simd<float, kD> wn = esimd::block_load<bf16, kD>(nw);
            const esimd::simd<float, kD> z = round_bf16<kD>(esimd::block_load<float, kD>(nz + static_cast<size_t>(row - A) * ldz + hv * kD));
            esimd::simd<float, kD> gate = sigmoid_v<kD>(z);
            if (nswish) gate = z * gate;  // qwen35
            const esimd::simd<float, kD> yv = ov * rr * wn * gate;
            esimd::block_store<bf16, kD>(ny + static_cast<size_t>(row - A) * VW + hv * kD, esimd::simd<bf16, kD>(yv));
          } else {
            esimd::block_store<bf16, kD>(o + static_cast<size_t>(row - A) * VW + hv * kD, orow);
          }
        }
        ++tile_ctr;
      }
#pragma unroll
      for (int cb = 0; cb < 8; ++cb) {
        esimd::simd<float, 128> acc = Sa.select<128, 1>(cb * 128) * gc;
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) {
          const esimd::simd<bf16, 256> bk = sld<bf16, 256>(kSK + static_cast<uint32_t>((cb * 4 + ib) * 512));
          const esimd::simd<bf16, 128> a = Ub.select<128, 1>(ib * 128);
          acc = xmx::dpas<8, 8, float, float, bf16, bf16>(acc, bk, a);
          const esimd::simd<bf16, 256> bkl = sld<bf16, 256>(kSKL + static_cast<uint32_t>((cb * 4 + ib) * 512));
          const esimd::simd<bf16, 128> al = Ul.select<128, 1>(ib * 128);
          acc = xmx::dpas<8, 8, float, float, bf16, bf16>(acc, bkl, a);
          acc = xmx::dpas<8, 8, float, float, bf16, bf16>(acc, bk, al);
        }
        Sa.select<128, 1>(cb * 128) = acc;
      }
    }
#pragma unroll
    for (int rr = 0; rr < kRows; ++rr) {
      esimd::simd<float, kD> row;
#pragma unroll
      for (int cb = 0; cb < 8; ++cb) row.select<16, 1>(cb * 16) = Sa.select<16, 1>(cb * 128 + rr * 16);
      esimd::block_store<float, kD>(Sp + rr * kD, row);
    }
  });
}

}  // namespace xenithium:kernels
