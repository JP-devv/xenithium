// Xenithium · the oneDNN half of the qwen35 prefill attention on oneDNN's fused GQA kernel
// (gqa.hpp gqa_attn_sdpa / gqa_sdpa_exec). The gather, q conversion and gate kernels and the per-head loop live in
// gqa.cpp (static library, 128 GRF); only the oneDNN Graph call is here, beside the dnnl link.
//
// The partition: 5D GQA (1, 1, grp, m, 256) x (1, 1, 1, S, 256) f16: MatMul q . K^T (f32 scores), x scale, bottom-right
// implicit causal mask (row r keeps columns <= r + S - m), softmax inf_as_zero -> f16 P, P . V -> f16 o; q and o read /
// written through strides (row stride hq x 256, head stride 256). oneDNN 3.11 picks `sdpa,ocl:micro:reusable` for it
// (ONEDNN_VERBOSE=all; head 256 included). The allocator is never called by this partition (no scratchpad per
// execute). One compiled partition per (m, S), cached per device (64, most recent first).
#include <sycl/sycl.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cmath>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "xenithium/kernels/attention/gqa.hpp"

namespace xenithium::kernels {
namespace {

namespace g = dnnl::graph;
using half = sycl::half;
using lt = g::logical_tensor;
using ldt = lt::data_type;
using lay = lt::layout_type;
using prop = lt::property_type;

constexpr int kD = 256;

void *dev_malloc(size_t size, size_t align, const void *dev, const void *ctx) {
  return sycl::aligned_alloc_device(align < 64 ? 64 : align, size, *static_cast<const sycl::device *>(dev),
                                    *static_cast<const sycl::context *>(ctx));
}
void dev_free(void *buf, const void *, const void *ctx, void *ev) {
  if (ev) static_cast<sycl::event *>(ev)->wait();
  sycl::free(buf, *static_cast<const sycl::context *>(ctx));
}

struct Part {
  int m = 0, S = 0, grp = 0, hq = 0;
  g::compiled_partition cp;
  lt q, k, v, scale, kvlen, qlen, ninf, out;
};

struct Dev {
  sycl::queue q;
  dnnl::engine eng;
  dnnl::stream st;
  std::list<Part> parts;  // most recent first
};
std::mutex g_mu;
std::vector<std::unique_ptr<Dev>> g_devs;

Dev &dev_of(sycl::queue &q) {
  for (auto &d : g_devs)
    if (d->q == q) return *d;
  auto d = std::make_unique<Dev>();
  d->q = q;
  static g::allocator alloc = g::sycl_interop::make_allocator(dev_malloc, dev_free);
  d->eng = g::sycl_interop::make_engine_with_allocator(q.get_device(), q.get_context(), alloc);
  d->st = dnnl::sycl_interop::make_stream(d->eng, q);
  g_devs.push_back(std::move(d));
  return *g_devs.back();
}

Part build(Dev &d, int m, int S, int grp, int hq) {
  Part p;
  p.m = m, p.S = S, p.grp = grp, p.hq = hq;
  size_t id = 0;
  const lt::dims qsz = {1, 1, grp, m, kD}, qst = {int64_t(m) * hq * kD, int64_t(m) * hq * kD, kD, int64_t(hq) * kD, 1};
  const lt::dims ksz = {1, 1, 1, S, kD}, kst = {int64_t(S) * kD, int64_t(S) * kD, int64_t(S) * kD, kD, 1};
  const lt::dims ssz = {1, 1, grp, m, S};
  g::graph gr(dnnl::engine::kind::gpu);
  p.q = lt(id++, ldt::f16, qsz, qst);
  p.k = lt(id++, ldt::f16, ksz, kst);
  lt sc(id++, ldt::f32, ssz, lay::strided);
  g::op mm1(id++, g::op::kind::MatMul, "qk");
  mm1.set_attr<bool>(g::op::attr::transpose_b, true);
  mm1.add_inputs({p.q, p.k});
  mm1.add_outputs({sc});
  p.scale = lt(id++, ldt::f32, 0, lay::strided, prop::host_scalar);
  lt scaled(id++, ldt::f32, ssz, lay::strided);
  g::op mul(id++, g::op::kind::Multiply, "scale");
  mul.add_inputs({sc, p.scale});
  mul.add_outputs({scaled});
  lt irow(id++, ldt::s32, ssz, lay::strided);
  g::op gir(id++, g::op::kind::GenIndex, "row");
  gir.set_attr<int64_t>(g::op::attr::axis, -2);
  gir.add_inputs({scaled});
  gir.add_outputs({irow});
  p.kvlen = lt(id++, ldt::s32, 0, lay::strided, prop::host_scalar);
  lt add_o(id++, ldt::s32, ssz, lay::strided);
  g::op add(id++, g::op::kind::Add, "add");
  add.add_inputs({irow, p.kvlen});
  add.add_outputs({add_o});
  p.qlen = lt(id++, ldt::s32, 0, lay::strided, prop::host_scalar);
  lt sub_o(id++, ldt::s32, ssz, lay::strided);
  g::op sub(id++, g::op::kind::Subtract, "sub");
  sub.add_inputs({add_o, p.qlen});
  sub.add_outputs({sub_o});
  lt icol(id++, ldt::s32, ssz, lay::strided);
  g::op gic(id++, g::op::kind::GenIndex, "col");
  gic.set_attr<int64_t>(g::op::attr::axis, -1);
  gic.add_inputs({scaled});
  gic.add_outputs({icol});
  lt ge_o(id++, ldt::boolean, ssz, lay::strided);
  g::op ge(id++, g::op::kind::GreaterEqual, "ge");
  ge.add_inputs({sub_o, icol});
  ge.add_outputs({ge_o});
  p.ninf = lt(id++, ldt::f32, 0, lay::strided, prop::host_scalar);
  lt masked(id++, ldt::f32, ssz, lay::strided);
  g::op sel(id++, g::op::kind::Select, "sel");
  sel.add_inputs({ge_o, scaled, p.ninf});
  sel.add_outputs({masked});
  lt probs(id++, ldt::f16, ssz, lay::strided);
  g::op sm(id++, g::op::kind::SoftMax, "softmax");
  sm.set_attr<int64_t>(g::op::attr::axis, -1);
  sm.set_attr<std::string>(g::op::attr::mode, "inf_as_zero");
  sm.add_inputs({masked});
  sm.add_outputs({probs});
  p.v = lt(id++, ldt::f16, ksz, kst);
  p.out = lt(id++, ldt::f16, qsz, qst);
  g::op mm2(id++, g::op::kind::MatMul, "pv");
  mm2.add_inputs({probs, p.v});
  mm2.add_outputs({p.out});
  for (auto *o : {&mm1, &mul, &gir, &add, &sub, &gic, &ge, &sel, &sm, &mm2}) gr.add_op(*o);
  gr.finalize();
  auto ps = gr.get_partitions();
  if (ps.size() != 1 || !ps[0].is_supported())
    throw std::runtime_error("gqa_sdpa_exec: oneDNN did not take the GQA pattern as one partition (" + std::to_string(ps.size()) + ")");
  p.cp = ps[0].compile({p.q, p.k, p.scale, p.kvlen, p.qlen, p.ninf, p.v}, {p.out}, d.eng);
  return p;
}

Part &part_of(Dev &d, int m, int S, int grp, int hq) {
  for (auto it = d.parts.begin(); it != d.parts.end(); ++it)
    if (it->m == m && it->S == S && it->grp == grp && it->hq == hq) {
      if (it != d.parts.begin()) d.parts.splice(d.parts.begin(), d.parts, it);
      return d.parts.front();
    }
  d.parts.push_front(build(d, m, S, grp, hq));
  if (d.parts.size() > 64) d.parts.pop_back();
  return d.parts.front();
}

}  // namespace

void gqa_sdpa_exec(sycl::queue &q, int m, int S, int grp, int hq, const half *q16, const half *k16, const half *v16, half *o16) {
  if (m < 1 || S < m || grp < 1 || hq < grp) throw std::invalid_argument("gqa_sdpa_exec: shape");
  std::lock_guard<std::mutex> lk(g_mu);
  Dev &d = dev_of(q);
  Part &p = part_of(d, m, S, grp, hq);
  float scale = 1.f / 16.f, ninf = -INFINITY;
  int32_t kvlen = S, qlen = m;
  std::vector<g::tensor> in = {g::tensor(p.q, d.eng, const_cast<half *>(q16)),
                               g::tensor(p.k, d.eng, const_cast<half *>(k16)),
                               g::tensor::make_scalar_tensor(p.scale, &scale),
                               g::tensor::make_scalar_tensor(p.kvlen, &kvlen),
                               g::tensor::make_scalar_tensor(p.qlen, &qlen),
                               g::tensor::make_scalar_tensor(p.ninf, &ninf),
                               g::tensor(p.v, d.eng, const_cast<half *>(v16))};
  std::vector<g::tensor> out = {g::tensor(p.out, d.eng, o16)};
  g::sycl_interop::execute(p.cp, d.st, in, out);
}

}  // namespace xenithium::kernels
