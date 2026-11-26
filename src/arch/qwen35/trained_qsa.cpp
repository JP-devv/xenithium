#include "xenithium/arch/qwen35/trained_qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "xenithium/kernels/attention/trained_qsa_pages.hpp"
#include "xenithium/kernels/pack.hpp"

namespace xenithium::arch::qwen35::trained_qsa {
namespace {
std::string dims(const std::vector<int64_t> &shape) {
  std::ostringstream s;
  s << '[';
  for (size_t i=0; i<shape.size(); ++i) s << (i ? "," : "") << shape[i];
  return s.str() + ']';
}
void rms(WideKey &x, const Key &weight, double eps) {
  double ss=0;
  for (double lane : x) ss += lane*lane;
  const double inv=1/std::sqrt(ss/dim+eps);
  for (int i=0; i<dim; ++i) x[i]=(x[i]*inv)*weight[i];
}
void rotary(WideKey &x, int64_t pos, double theta) {
  if (pos<0) throw std::invalid_argument("trained QSA: negative text position");
  // 64 NeoX-rotated lanes. On text, all MRoPE axes have the same position.
  const double base=std::pow(theta,-1.0/32.0);
  for (int j=0; j<32; ++j) {
    const double phase=static_cast<double>(pos)*std::pow(base,j);
    const double a=x[j], b=x[j+32], c=std::cos(phase), s=std::sin(phase);
    x[j]=a*c-b*s; x[j+32]=b*c+a*s;
  }
}
}  // namespace

Layer Layer::load(const model::Checkpoint &bank, const Config &target, int layer, NormStorage storage,
                  const model::BulkOptions &io) {
  if (target.hidden!=5120 || target.layers!=64 || target.full_layers()!=16 || target.full_interval!=4 ||
      target.head_dim!=256 || target.q_heads!=24 || target.kv_heads!=4 || target.rms_eps!=1e-6 ||
      target.rope_theta!=1e7 || target.partial_rotary!=.25 || !target.mrope_interleaved ||
      target.mrope_section!=std::vector<int>{11,11,10} || !target.attn_output_gate)
    throw std::runtime_error("trained QSA: target is not Swift 5120x64 with 16 GQA6/256 layers and 64-lane interleaved text MRoPE");
  if (layer<0 || layer>=target.layers || !target.full_attention.at(layer) || layer%4!=3)
    throw std::runtime_error("trained QSA: layer "+std::to_string(layer)+" is not a Swift full-attention layer");
  Layer result;
  result.hidden_=target.hidden;
  result.theta_=target.rope_theta;
  result.eps_=target.rms_eps;
  const std::string base=target.prefix+"layers."+std::to_string(layer)+".self_attn._qsa_indexer.";
  auto tensor=[&](const char *key, std::vector<int64_t> shape) {
    const auto &t=bank.at(base+key);
    if (t.dtype!=model::DType::f32 || t.shape!=shape)
      throw std::runtime_error(t.name+": expected F32 "+dims(shape)+", found "+t.dtype_str+" "+dims(t.shape));
    return &t;
  };
  const auto *q=tensor("q_proj.weight",{heads*dim,target.hidden}), *k=tensor("k_proj.weight",{dim,target.hidden});
  const auto *qn=tensor("q_norm",{dim}), *kn=tensor("k_norm",{dim}), *bn=tensor("k_block_norm",{dim});
  const auto data=model::bulk_read(bank,{q,k,qn,kn,bn},io);
  auto copy=[&](const model::TensorInfo *t, std::span<float> out, bool norm) {
    std::memcpy(out.data(),data.data(*t),out.size_bytes());
    for (float &v : out) {
      if (norm && storage==NormStorage::delta) v=1.f+v;
      if (!std::isfinite(v)) throw std::runtime_error(t->name+": nonfinite F32 weight");
    }
  };
  result.qw_.resize(q->numel()); result.kw_.resize(k->numel());
  copy(q,result.qw_,false); copy(k,result.kw_,false);
  copy(qn,result.qnorm_,true); copy(kn,result.knorm_,true); copy(bn,result.bnorm_,true);
  return result;
}

std::array<std::vector<uint16_t>,3> Layer::device_tiles() const {
  constexpr int n=(heads+1)*dim;
  std::vector<uint16_t> rows(size_t(n)*hidden_);
  std::vector<double> residual(qw_.begin(),qw_.end());
  residual.insert(residual.end(),kw_.begin(),kw_.end());
  std::array<std::vector<uint16_t>,3> out;
  for (auto &p:out) {
    for (size_t i=0;i<rows.size();++i) {
      rows[i]=kernels::bf16_bits(static_cast<float>(residual[i]));
      residual[i]-=kernels::bf16_value(rows[i]);
    }
    p.resize(kernels::dense_packed_elems(n,hidden_));
    kernels::pack_dense_bf16(rows.data(),n,hidden_,p.data());
  }
  return out;
}

Projection Layer::project(std::span<const uint16_t> hidden) const {
  if (hidden_<=0 || hidden.size()!=size_t(hidden_)) throw std::invalid_argument("trained QSA: hidden row width");
  std::vector<double> x(hidden_);
  for (int j=0; j<hidden_; ++j) x[j]=kernels::bf16_value(hidden[j]);
  auto dot=[&](const float *a) {
    double s=0;
    for (int i=0; i<hidden_; ++i) s += x[i]*a[i];
    return s;
  };
  Projection r;
  for (int d=0; d<dim; ++d) {
    for (int h=0; h<heads; ++h) r.q[h][d]=dot(qw_.data()+size_t(h*dim+d)*hidden_);
    r.k[d]=dot(kw_.data()+size_t(d)*hidden_);
  }
  return r;
}
Normalized Layer::normalize(const Projection &row, int64_t pos) const {
  Normalized result;
  result.q=row.q;
  for (auto &q : result.q) { rms(q,qnorm_,eps_); rotary(q,pos,theta_); }
  auto k=row.k;
  rms(k,knorm_,eps_);
  std::copy(k.begin(),k.end(),result.k.begin());
  return result;
}
Key Layer::pool(std::span<const Key,4> keys, int64_t pos) const {
  WideKey mean{};
  for (const auto &k : keys) for (int i=0; i<dim; ++i) mean[i]+=k[i];
  for (double &v : mean) v*=.25;
  rms(mean,bnorm_,eps_);
  rotary(mean,pos,theta_);
  Key out{};
  std::copy(mean.begin(),mean.end(),out.begin());
  return out;
}
double score(const Query &q, const Key &block) {
  double sum=0;
  for (const auto &head : q) {
    double dot=0;
    for (int j=0; j<dim; ++j) dot+=head[j]*block[j];
    sum+=std::max(0.0,dot);
  }
  return sum;
}
std::vector<int32_t> select(std::span<const double> s) {
  if (s.size()>size_t(kernels::kQsaMaxTokens / kernels::kQsaCompress))
    throw std::invalid_argument("trained QSA: too many complete blocks");
  for (double v:s) if (!std::isfinite(v)) throw std::invalid_argument("trained QSA: nonfinite score");
  std::vector<int32_t> ids(s.size());
  std::iota(ids.begin(),ids.end(),0);
  const auto count=std::min(ids.size(),size_t(kernels::kTrainedQsaTopBlocks));
  std::partial_sort(ids.begin(),ids.begin()+count,ids.end(),[&](int a,int b){return s[a]==s[b]?a<b:s[a]>s[b];});
  ids.resize(count);
  return ids;
}
}  // namespace xenithium::arch::qwen35::trained_qsa
