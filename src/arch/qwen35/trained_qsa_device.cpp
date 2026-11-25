#include "xenithium/arch/qwen35/trained_qsa_device.hpp"
#include "xenithium/kernels/attention/trained_qsa_pages.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/pack.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace xenithium::arch::qwen35::trained_qsa {
namespace k = kernels;
namespace {
constexpr int width=(heads+1)*dim, workers=32, pf_score_tile=64, pf_fa_min_rows=16;
constexpr size_t allocation_bytes(size_t n) { return std::max(size_t(4096),n); }
bool enabled(const char *name) { const char *p=std::getenv(name); return p && p[0]=='1' && !p[1]; }
struct History {
  float *tail, *pool, *staged, *closed, *query, *pf_raw;
  sycl::half *pool16;
  const float *norm, *rope;
  int cap;
};
// Position-indexed text rotary table (cos then sin), equivalent to broadcast
// text MRoPE. We cannot use stock QSA's fused BF16 prep: the trained bank
// rounds neither projected input nor normalized keys to BF16 and has a third norm.
template<bool PfQuery=false>
sycl::event norm_index(sycl::queue &q, const History &h, const std::array<float*,3> &x, float eps,
                       int rows, int slot, int pos, k::GqaStep step, bool draft,uint32_t track=~uint32_t(0)) {
  const auto *rec=step.pos;
  const int stride=step.stride, seg_rows=step.seg_rows;
  // A tree still stages the index keys, but does not score or use sparse attention.
  const bool queries=draft && !step.tree;
  const int nhead=PfQuery?heads:queries?heads+1:1;
  return q.parallel_for(sycl::nd_range<1>(size_t(rows)*nhead*workers,workers), [=](sycl::nd_item<1> it) {
    const int w=it.get_group(0), lane=it.get_local_id(0), r=PfQuery?w/heads:queries?w/(heads+1):w, head=PfQuery?w%heads:queries?w%(heads+1):heads;
    if(draft && !(track&(uint32_t(1)<<r))) return;
    const auto sr=draft?k::seg_row(rec,stride,seg_rows,r):k::SegRow{0,r,pos+r,slot};
    const size_t at=size_t(r)*width+head*dim;
    float a[dim/workers], ss=0;
    for(int i=0;i<dim/workers;++i) {
      const size_t d=at+lane+i*workers;
      a[i]=x[0][d]+(x[1][d]+x[2][d]);
      ss+=a[i]*a[i];
    }
    const float inv=sycl::rsqrt(sycl::reduce_over_group(it.get_group(),ss,sycl::plus<float>())/dim+eps);
    const float *norm=h.norm+(head==heads?dim:0);
    for(int i=0;i<dim/workers;++i) a[i]=(a[i]*inv)*norm[lane+i*workers];
    if (head==heads) {
      float *out=(draft?h.staged:h.pf_raw)+size_t(r)*dim;
      for(int i=0;i<dim/workers;++i) out[lane+i*workers]=a[i];
    } else {
      const float *rot=h.rope+size_t(sr.pos)*64;
      const float aa=a[0], bb=a[1];
      a[0]=aa*rot[lane]-bb*rot[lane+32];
      a[1]=bb*rot[lane]+aa*rot[lane+32];
      float *out=h.query+(size_t(r)*heads+head)*dim;
      for(int i=0;i<dim/workers;++i) out[lane+i*workers]=a[i];
    }
  });
}
sycl::event index_blocks(sycl::queue &q, const History &h, float eps, int rows, int slot, int pos,
                         k::GqaStep step, int32_t *eligible, bool draft,uint32_t track=~uint32_t(0)) {
  const auto *rec=step.pos;
  const int stride=step.stride, seg_rows=step.seg_rows;
  return q.parallel_for(sycl::nd_range<1>(size_t(rows)*workers,workers), [=](sycl::nd_item<1> it) {
    const int r=it.get_group(0), lane=it.get_local_id(0);
    if(draft && !(track&(uint32_t(1)<<r))) return;
    const auto sr=draft?k::seg_row(rec,stride,seg_rows,r):k::SegRow{0,r,pos+r,slot};
    if (draft && lane==0) eligible[r]=(sr.pos+1)/4;
    if (sr.pos%4!=3) return;
    const int start=draft?rec[sr.seg*stride+k::kSegPos]:pos;
    const float *tail=h.tail+size_t(sr.slot)*4*dim;
    const float *raw=draft?h.staged:h.pf_raw;
    float a[dim/workers], ss=0;
    for(int i=0;i<dim/workers;++i) {
      const int d=lane+i*workers;
      float sum=0;
      for(int t=sr.pos-3;t<=sr.pos;++t)
        sum+=t>=start?raw[size_t(r+t-sr.pos)*dim+d]:tail[size_t(t%4)*dim+d];
      a[i]=sum*.25f;
      ss+=a[i]*a[i];
    }
    const float inv=sycl::rsqrt(sycl::reduce_over_group(it.get_group(),ss,sycl::plus<float>())/dim+eps);
    for(int i=0;i<dim/workers;++i) a[i]=(a[i]*inv)*h.norm[2*dim+lane+i*workers];
    const float *rot=h.rope+size_t(sr.pos-3)*64;
    const float aa=a[0], bb=a[1];
    a[0]=aa*rot[lane]-bb*rot[lane+32];
    a[1]=bb*rot[lane]+aa*rot[lane+32];
    const size_t at=(size_t(sr.slot)*(h.cap/4)+sr.pos/4)*dim;
    for(int i=0;i<dim/workers;++i) {
      const int d=lane+i*workers;
      if(draft) h.closed[size_t(r)*dim+d]=a[i];
      else {
        if(h.pool) h.pool[at+d]=a[i];
        if(h.pool16) h.pool16[at+d]=sycl::half(a[i]);
      }
    }
  });
}
template<bool Half=false> sycl::event index_scores(sycl::queue &q, const History &h, int rows, int bound, k::GqaStep step, uint32_t *scores) {
  const auto *rec=step.pos;
  const int stride=step.stride, seg_rows=step.seg_rows, pitch=h.cap/4;
  return q.parallel_for(sycl::nd_range<1>(size_t(rows)*bound*workers,workers), [=](sycl::nd_item<1> it) {
    const int wg=it.get_group(0), r=wg/bound, b=wg%bound, lane=it.get_local_id(0);
    const auto sr=k::seg_row(rec,stride,seg_rows,r);
    const int last=4*b+3;
    if (last>sr.pos) return;
    const int start=rec[sr.seg*stride+k::kSegPos];
    const size_t at=(size_t(sr.slot)*pitch+b)*dim;
    float score=0;
    for(int j=0;j<heads;++j) {
      const float *query=h.query+(size_t(r)*heads+j)*dim;
      float dot=0;
      for(int i=lane;i<dim;i+=workers) {
        const float key=last>=start?h.closed[size_t(r+last-sr.pos)*dim+i]:
          (Half?static_cast<float>(h.pool16[at+i]):h.pool[at+i]);
        dot+=query[i]*key;
      }
      score+=sycl::fmax(0.f,sycl::reduce_over_group(it.get_group(),dot,sycl::plus<float>()));
    }
    if (!lane) scores[size_t(r)*pitch+b]=sycl::bit_cast<uint32_t>(score);
  });
}
// 00aadde stock FP32 arithmetic, with independent <=4-row tiles within
// each verify segment. DSpine can supply chains up to 16 rows per segment.
template<bool Half=false> sycl::event index_scores_tile(sycl::queue &q,const History &h,int rows,int bound,k::GqaStep step,uint32_t *scores) {
  constexpr int R=4,B=8;
  static_assert(workers==32 && dim==128);
  const auto *rec=step.pos;
  const int stride=step.stride,pitch=h.cap/4,tiles=(bound+B-1)/B,layout=step.seg_rows;
  const int segments=k::seg_count(rows,layout),groups=(k::seg_size(rows,layout)+R-1)/R;
  const auto regs=sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
  return q.parallel_for(sycl::nd_range<2>({size_t(segments*groups),size_t(tiles)*workers},{1,workers}),regs,
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(workers)]] {
    const int g=it.get_group(0),s=g/groups,j0=(g%groups)*R,b0=it.get_group(1)*B,lane=it.get_local_id(1);
    int rseg,nseg; k::seg_span(layout,rows,s,rseg,nseg);
    const int n=sycl::min(R,nseg-j0);
    if(n<1) return;
    const int start=rec[s*stride+k::kSegPos],slot=rec[s*stride+k::kSegSlot];
    const int r0=rseg+j0,nmax=(start+j0+n)/4;
    if(b0>=nmax) return;
    const auto sg=it.get_sub_group();
    float qr[R][heads][dim/workers];
#pragma unroll
    for(int r=0;r<R;++r)
#pragma unroll
      for(int head=0;head<heads;++head)
#pragma unroll
        for(int i=0;i<dim/workers;++i) qr[r][head][i]=r<n?h.query[(size_t(r0+r)*heads+head)*dim+lane+i*workers]:0.f;
    for(int b=b0;b<sycl::min(b0+B,nmax);++b) {
      const int last=4*b+3;
      const size_t at=(size_t(slot)*pitch+b)*dim;
      float kr[dim/workers],score[R]{};
#pragma unroll
      for(int i=0;i<dim/workers;++i) {
        const int d=lane+i*workers;
        kr[i]=last>=start?h.closed[size_t(rseg+last-start)*dim+d]:
          (Half?static_cast<float>(h.pool16[at+d]):h.pool[at+d]);
      }
#pragma unroll
      for(int head=0;head<heads;++head)
#pragma unroll
        for(int r=0;r<R;++r) {
          float dot=qr[r][head][0]*kr[0];
#pragma unroll
          for(int i=1;i<dim/workers;++i) dot=sycl::fma(qr[r][head][i],kr[i],dot);
          dot+=sycl::permute_group_by_xor(sg,dot,16);
          dot+=sycl::permute_group_by_xor(sg,dot,8);
          dot+=sycl::permute_group_by_xor(sg,dot,4);
          dot+=sycl::permute_group_by_xor(sg,dot,2);
          dot+=sycl::permute_group_by_xor(sg,dot,1);
          score[r]+=sycl::fmax(0.f,dot);
        }
      if(!lane)
#pragma unroll
        for(int r=0;r<R;++r) if(r<n && last<=start+j0+r) scores[size_t(r0+r)*pitch+b]=sycl::bit_cast<uint32_t>(score[r]);
    }
  });
}
}

struct Device::Impl {
  struct BankLayer {
    History h{};
    std::array<k::DenseWeights,3> w{};
    bool sparse_logged=false;
  };
  struct Card {
    std::vector<BankLayer> layer;
    std::array<float*,3> x{};
    size_t allocated=0;
    k::QsaScratch sc{};
    int32_t *entries=nullptr,*counts=nullptr,*status=nullptr;
    struct Prefill {
      k::QsaScratch sc{};
      float *query=nullptr;
      int32_t *pages=nullptr,*counts=nullptr,*entries=nullptr;
      uint16_t *masks=nullptr;
      k::gqa_bf16 *xmx_queries=nullptr,*xmx_keys=nullptr;
    } pf;
  };
  Fabric &fab;
  std::vector<Card> cards;
  std::vector<void*> held;
  const int slots, cap, prefill_rows, step_rows;
  const float eps;
  const bool tile=enabled("XEN_TRAINED_QSA_TILED_SCORE");
  const bool half=enabled("XEN_TRAINED_QSA_POOL_FP16");
  Impl(Fabric &f,int s,int c,int pf,int st,float e,size_t layers):fab(f),cards(f.size()),slots(s),cap(c),prefill_rows(pf),step_rows(st),eps(e) { held.reserve((16+8*layers)*f.size()); }
  ~Impl() { for(auto *p:held) fab.release(p); }
  template<class T> T *alloc(int card,size_t count) {
    if (!count || count>std::numeric_limits<size_t>::max()/sizeof(T))
      throw std::overflow_error("trained QSA layer3: allocation size overflow");
    const size_t bytes=allocation_bytes(count*sizeof(T));
    try {
      auto *p=static_cast<T*>(fab.alloc(card,bytes));
      held.push_back(p); cards[card].allocated+=bytes; return p;
    } catch (const std::exception &e) {
      throw std::runtime_error("trained QSA allocation failed: rank="+std::to_string(card)+" bytes="+std::to_string(bytes)+": "+e.what());
    }
  }
};
size_t Device::bytes_per_rank(size_t layers,int hidden,int slots,int cap,int pf,int st,bool xmx_prefill) {
  if (!layers || layers>64 || hidden!=5120 || slots<1 || slots>k::kSegs16 || cap<64 || cap>kMaxContext || cap%k::kQsaPage ||
      pf<1 || pf>cap || st<1 || st>k::kStepRows32)
    throw std::invalid_argument("trained QSA: invalid layer, hidden, context, slot or row capacity");
  const size_t eligible=cap/k::kQsaCompress;
  const int chunk=k::qsa_select_chunk(static_cast<int>(eligible));
  if ((eligible+chunk-1)/chunk>k::kQsaMaxChunks) throw std::invalid_argument("trained QSA: selector chunk capacity exceeded");
  const auto b=[](size_t n,size_t sz=4) { return std::max(size_t(4096),n*sz); };  // extents bounded above
  const bool half=enabled("XEN_TRAINED_QSA_POOL_FP16");
  const size_t words=size_t(slots)*eligible*dim;
  const size_t layer=b(size_t(slots)*4*dim)+b(words,half?sizeof(sycl::half):sizeof(float))+2*b(size_t(st)*dim)+b(3*dim);
  const size_t weights=3*layers*b(k::dense_packed_elems(width,hidden),2);
  const size_t cp=size_t(st)*k::kQsaMaxChunks;
  const size_t candidates=eligible>size_t(chunk)?b(cp)+2*b(cp*k::kQsaTopBlocks):0;
  const size_t scratch=3*b(size_t(std::max(pf,k::row_shape(st)))*width)+b(size_t(st)*eligible)+candidates;
  const size_t pages=cap/k::kQsaPage, groups=(pf+pf_fa_min_rows-1)/pf_fa_min_rows;
  const size_t pc=size_t(pf_score_tile)*k::kQsaMaxChunks;
  const size_t prefill=xmx_prefill ?
      b(size_t(pf_score_tile)*heads*dim)+b(groups*pages)+b(groups)+b(size_t(pf)*pages,2)+
      b(size_t(pf_score_tile)*heads*dim,2)+b(eligible*dim,2)+
      2*b(pf_score_tile)+b(size_t(pf_score_tile)*k::kTrainedQsaTopBlocks)+b(size_t(pf_score_tile)*eligible)+
      b(size_t(pf_score_tile)*k::kTrainedQsaPages)+(eligible>size_t(chunk)?b(pc)+2*b(pc*k::kQsaTopBlocks):0):0;
  return prefill+layers*layer+weights+scratch+b(size_t(cap)*64)+b(size_t(pf)*dim)+b(size_t(st)*heads*dim)+
      3*b(st)+b(size_t(st)*k::kTrainedQsaTopBlocks)+b(size_t(st)*k::kTrainedQsaPages)+b(size_t(st)*4);
}
Device::Device(Fabric &fab,std::span<const Layer> weights,int slots,int cap,int pf,int st,bool xmx_prefill) {
  const size_t planned=bytes_per_rank(weights.size(),weights.empty()?0:weights.front().hidden(),slots,cap,pf,st,xmx_prefill);
  for (const auto &w:weights) if(w.hidden()!=weights.front().hidden() || w.epsilon()!=weights.front().epsilon() || w.theta()!=weights.front().theta())
    throw std::invalid_argument("trained QSA: incompatible layer geometry");
  p_=std::make_unique<Impl>(fab,slots,cap,pf,st,weights.front().epsilon(),weights.size());
  const int eligible=cap/k::kQsaCompress;
  const int chunk=k::qsa_select_chunk(eligible);
  const int nch=(eligible-1)/chunk+1;
  const size_t candidate_pitch=size_t(st)*k::kQsaMaxChunks;
  std::vector<float> rope(size_t(cap)*64);
  const double base=std::pow(weights.front().theta(),-1.0/32);
  for(int p=0;p<cap;++p) for(int j=0;j<32;++j) {
    const double phase=p*std::pow(base,j);
    rope[p*64+j]=std::cos(phase); rope[p*64+j+32]=std::sin(phase);
  }
  std::vector<History> shared(fab.size());
  for(int c=0;c<int(fab.size());++c) {
    auto &d=p_->cards[c]; auto &q=fab.compute(c);
    auto &h=shared[c]; h.cap=cap;
    h.query=p_->alloc<float>(c,size_t(st)*heads*dim);
    h.pf_raw=p_->alloc<float>(c,size_t(pf)*dim);
    auto *rt=p_->alloc<float>(c,rope.size()); h.rope=rt;
    q.memcpy(rt,rope.data(),rope.size()*sizeof(float)).wait_and_throw();
    d.layer.resize(weights.size());
    d.sc.nblk=p_->alloc<int32_t>(c,st);
    d.sc.sel=p_->alloc<int32_t>(c,size_t(st)*k::kTrainedQsaTopBlocks);
    d.sc.sel_n=p_->alloc<int32_t>(c,st);
    for(auto &x:d.x) x=p_->alloc<float>(c,size_t(std::max(pf,k::row_shape(st)))*width);
    d.sc.score=p_->alloc<uint32_t>(c,size_t(st)*eligible);
    if(nch>1) {
      d.sc.cand_n=p_->alloc<int32_t>(c,candidate_pitch);
      d.sc.cand_id=p_->alloc<int32_t>(c,candidate_pitch*k::kQsaTopBlocks);
      d.sc.cand_key=p_->alloc<uint32_t>(c,candidate_pitch*k::kQsaTopBlocks);
    }
    d.entries=p_->alloc<int32_t>(c,size_t(st)*k::kTrainedQsaPages);
    d.counts=p_->alloc<int32_t>(c,size_t(st)*4);
    d.status=p_->alloc<int32_t>(c,st);
    if(xmx_prefill) {
      auto &w=d.pf;
      const size_t pages=cap/k::kQsaPage, groups=(pf+pf_fa_min_rows-1)/pf_fa_min_rows;
      w.query=p_->alloc<float>(c,size_t(pf_score_tile)*heads*dim);
      w.pages=p_->alloc<int32_t>(c,groups*pages); w.counts=p_->alloc<int32_t>(c,groups);
      w.masks=p_->alloc<uint16_t>(c,size_t(pf)*pages);
      w.xmx_queries=p_->alloc<k::gqa_bf16>(c,size_t(pf_score_tile)*heads*dim);
      w.xmx_keys=p_->alloc<k::gqa_bf16>(c,size_t(eligible)*dim);
      w.sc.nblk=p_->alloc<int32_t>(c,pf_score_tile); w.sc.sel_n=p_->alloc<int32_t>(c,pf_score_tile);
      w.sc.sel=p_->alloc<int32_t>(c,size_t(pf_score_tile)*k::kTrainedQsaTopBlocks);
      w.sc.score=p_->alloc<uint32_t>(c,size_t(pf_score_tile)*eligible);
      w.entries=p_->alloc<int32_t>(c,size_t(pf_score_tile)*k::kTrainedQsaPages);
      if(nch>1) {
        const size_t pitch=size_t(pf_score_tile)*k::kQsaMaxChunks;
        w.sc.cand_n=p_->alloc<int32_t>(c,pitch);
        w.sc.cand_id=p_->alloc<int32_t>(c,pitch*k::kQsaTopBlocks);
        w.sc.cand_key=p_->alloc<uint32_t>(c,pitch*k::kQsaTopBlocks);
      }
    }
  }
  for(size_t l=0;l<weights.size();++l) {
    const auto tiles=weights[l].device_tiles();
    const auto norms=weights[l].multipliers();
    for(int c=0;c<int(fab.size());++c) {
      auto &d=p_->cards[c].layer[l]; auto &q=fab.compute(c);
      d.h=shared[c];
      d.h.tail=p_->alloc<float>(c,size_t(slots)*4*dim);
      if(p_->half) d.h.pool16=p_->alloc<sycl::half>(c,size_t(slots)*eligible*dim);
      else d.h.pool=p_->alloc<float>(c,size_t(slots)*eligible*dim);
      d.h.staged=p_->alloc<float>(c,size_t(st)*dim);
      d.h.closed=p_->alloc<float>(c,size_t(st)*dim);
      auto *nr=p_->alloc<float>(c,3*dim); d.h.norm=nr;
      q.memcpy(nr,norms.data(),sizeof(norms)).wait_and_throw();
      for(int i=0;i<3;++i) {
        auto *w=p_->alloc<uint16_t>(c,tiles[i].size());
        q.memcpy(w,tiles[i].data(),tiles[i].size()*sizeof(uint16_t)).wait_and_throw();
        d.w[i]={w,nullptr,k::DenseKind::bf16,width,weights[l].hidden()};
      }
    }
  }
  for(size_t c=0;c<p_->cards.size();++c) {
    const size_t actual=p_->cards[c].allocated;
    if(actual!=planned) throw std::logic_error("trained QSA allocation ledger differs from preflight");
    std::fprintf(stderr,"trained QSA allocation: rank=%zu layers=%zu allocated_device_bytes=%zu pool_bits=%d\n",
                 c,weights.size(),actual,p_->half?16:32);
  }
}
Device::~Device()=default;
namespace {
void publish_tail(sycl::queue &q,const History &h,int slot,int pos,int rows,const float *raw) {
  const int end=pos+rows, from=std::max(pos,end-end%4), n=end-from;
  if (n>0) q.memcpy(h.tail+(size_t(slot)*4+from%4)*dim,raw+size_t(from-pos)*dim,size_t(n)*dim*sizeof(float));
}
}
k::GqaTrainedPrefill Device::prefill(int card,const k::gqa_bf16 *x,int rows,int slot,int pos,int bank_layer,int fa_rows) {
  if(rows<1 || rows>p_->prefill_rows || slot<0 || slot>=p_->slots || pos<0 || pos>p_->cap-rows)
    throw std::invalid_argument("trained QSA layer3: invalid prefill slot/positions");
  auto &q=p_->fab.compute(card); auto &d=p_->cards.at(card);
  const auto &l=d.layer.at(bank_layer);
  for(int i=0;i<3;++i) {
    const auto &w=l.w[i];
    k::pf_dense(q,rows,w,x,w.k,k::PfOut{d.x[i],nullptr,width});
  }
  norm_index(q,l.h,d.x,p_->eps,rows,slot,pos,{},false);
  index_blocks(q,l.h,p_->eps,rows,slot,pos,{},nullptr,false);
  publish_tail(q,l.h,slot,pos,rows,l.h.pf_raw);
  if(!fa_rows) return {};  // the default still builds only the decode history
  auto &w=d.pf;
  if(!q.is_in_order() || !w.masks || (fa_rows!=16 && fa_rows!=32 && fa_rows!=40) || pos/4<=k::kTrainedQsaTopBlocks)
    throw std::invalid_argument("trained XMX prefill: expected sparse workspace, supported FA geometry and long prefix");
  const int pitch=p_->cap/k::kQsaPage, live_pages=(pos+rows+k::kQsaPage-1)/k::kQsaPage;
  auto *masks=w.masks;
  auto *pages=w.pages;
  auto *counts=w.counts;
  q.fill(masks,uint16_t(0),size_t(rows)*pitch);
  const size_t origin=size_t(slot)*(p_->cap/4)*dim;
  // Keep the existing FP16 or FP32 resident summaries. Convert to the upstream
  // BF16 XMX operand layout only for this chunk; no changes to decode history.
  if(l.h.pool16) k::qsa_trained_xmx_pack_k(q,l.h.pool16+origin,(pos+rows)/4,w.xmx_keys);
  else k::qsa_trained_xmx_pack_k(q,l.h.pool+origin,(pos+rows)/4,w.xmx_keys);
  History h=l.h;
  h.query=w.query;
  for(int q0=0;q0<rows;q0+=pf_score_tile) {
    const int n=std::min(pf_score_tile,rows-q0), start=pos+q0, bound=(start+n)/4;
    std::array<float*,3> projected=d.x;
    for(auto &p:projected) p+=size_t(q0)*width;
    norm_index<true>(q,h,projected,p_->eps,n,slot,start,{},false);
    k::qsa_trained_xmx_pack_q(q,w.query,n,w.xmx_queries);
    k::qsa_trained_xmx_score(q,w.xmx_queries,w.xmx_keys,n,start,bound,w.sc.score,p_->cap/4,w.sc.nblk);
    k::qsa_select(q,n,w.sc.score,p_->cap/4,w.sc.nblk,bound,w.sc);
    const auto *ids=w.sc.sel,*ns=w.sc.sel_n;
    auto *entries=w.entries;
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id) {
      const int r=id[0], rp=start+r;
      auto *row=masks+size_t(q0+r)*pitch;
      auto *packed=entries+size_t(r)*k::kTrainedQsaPages;
      const int count=k::trained_qsa_pack_row(rp,ids+size_t(r)*k::kTrainedQsaTopBlocks,ns[r],packed);
      if (count<=0) {  // same safe fallback as trained decode; the FA causal mask still applies
        for(int pg=0;pg<=rp/k::kQsaPage;++pg) row[pg]=0xffffu;
      } else {
        for(int j=0;j<count;++j) row[k::trained_qsa_page(packed[j])]=k::trained_qsa_mask(packed[j]);
      }
    });
  }
  // Compact the union in increasing logical page order. FA retains the row's
  // mask, not the union mask; the latter decides what to stage exactly once.
  constexpr int items=128;
  const int blocks=(rows+fa_rows-1)/fa_rows;
  q.parallel_for(sycl::nd_range<1>(size_t(blocks)*items,items),[=](sycl::nd_item<1> it) {
    const int g=it.get_group(0),lane=it.get_local_id(0),r0=g*fa_rows;
    const auto group=it.get_group();
    const int rn=sycl::min(r0+fa_rows,rows), pn=(pos+rn+k::kQsaPage-1)/k::kQsaPage;
    int base=0;
    for(int p0=0;p0<pn;p0+=items) {
      const int pg=p0+lane;
      bool keep=false;
      if (pg<pn && pg<live_pages)
        for(int r=r0;r<rn && !keep;++r) keep=masks[size_t(r)*pitch+pg]!=0;
      const int before=sycl::exclusive_scan_over_group(group,int(keep),sycl::plus<int>());
      if (keep) pages[size_t(g)*pitch+base+before]=pg;
      base+=sycl::group_broadcast(group,before+int(keep),items-1);
    }
    if (!lane) counts[g]=base;
  });
  return {pages,counts,masks,pitch,fa_rows};
}
void Device::stage(int card,const k::gqa_bf16 *x,int rows,int hkv,k::GqaStep &step,int bank_layer,int model_layer,uint32_t track,bool sparse) {
  if(rows<1 || rows>p_->step_rows || step.pos_bound<rows || step.pos_bound>p_->cap)
    throw std::invalid_argument("trained QSA layer3: invalid verification chain");
  auto &q=p_->fab.compute(card); auto &d=p_->cards.at(card);
  auto &l=d.layer.at(bank_layer);
  if(!track) return;
  const uint32_t all=rows==32?~uint32_t(0):((uint32_t(1)<<rows)-1);
  if(sparse && (track&all)!=all) throw std::logic_error("trained QSA: sparse step with incomplete row mask");
  for(int i=0;i<3;++i) k::dense_gemm(q,rows,x,l.w[i],d.x[i]);
  norm_index(q,l.h,d.x,p_->eps,rows,0,0,step,true,track);
  const bool can_tile=step.stride>=k::kSegRec && k::seg_rows_ok(rows,step.seg_rows,step.stride) && k::seg_size(rows,step.seg_rows)<=16;
  const bool tiled=p_->tile && can_tile;
  if(step.tree) return;  // dense attention as dictated by the existing policy
  index_blocks(q,l.h,p_->eps,rows,0,0,step,d.sc.nblk,true,track);
  if(!sparse) return;  // retained history only; target runs the unchanged dense kernel
  const int bound=step.pos_bound/k::kQsaCompress;
  if (bound<=k::kTrainedQsaTopBlocks) return;
  if(tiled) {
    if(p_->half) index_scores_tile<true>(q,l.h,rows,bound,step,d.sc.score);
    else index_scores_tile(q,l.h,rows,bound,step,d.sc.score);
  } else {
    if(p_->half) index_scores<true>(q,l.h,rows,bound,step,d.sc.score);
    else index_scores(q,l.h,rows,bound,step,d.sc.score);
  }
  k::qsa_select(q,rows,d.sc.score,p_->cap/4,d.sc.nblk,bound,d.sc);
  k::gqa_pack_trained_qsa(q,rows,hkv,step,d.sc.sel,d.sc.sel_n,d.entries,d.counts,d.status);
  step.trained_qsa={d.entries,d.counts};
  if (model_layer>=0 && !l.sparse_logged) {
    std::fprintf(stderr,"trained QSA sparse submit: rank=%d layer=%d rows=%d max_eligible=%d\n",card,model_layer,rows,bound);
    l.sparse_logged=true;
  }
}
void Device::accept(int slot,int pos,int row0,int keep,bool tree_batch,uint64_t path) {
  if(slot<0 || slot>=p_->slots || keep<1 || pos<0 || pos>p_->cap-keep || row0<0 || row0>p_->step_rows-keep ||
     (tree_batch && keep>p_->prefill_rows) || (path && (!tree_batch || keep>k::kTreeRows || row0+k::path_row(path,keep-1)>=p_->step_rows)))
    throw std::invalid_argument("trained QSA layer3: invalid accepted prefix");
  for(int c=0;c<int(p_->cards.size());++c) {
    auto &q=p_->fab.compute(c);
    for (const auto &l:p_->cards[c].layer) {
      const auto &h=l.h;
      if(tree_batch) {
        // Pack the accepted path (identity for chain segments in this
        // batch); use the same closure/norm/rotary path as linear prefill.
        q.parallel_for(sycl::range<1>(size_t(keep)*dim),[=](sycl::id<1> id) {
          const size_t i=id[0];
          h.pf_raw[i]=h.staged[(size_t(row0)+k::path_row(path,static_cast<int>(i/dim)))*dim+i%dim];
        });
        index_blocks(q,h,p_->eps,keep,slot,pos,{},nullptr,false);
        publish_tail(q,h,slot,pos,keep,h.pf_raw);
        continue;
      }
      publish_tail(q,h,slot,pos,keep,h.staged+size_t(row0)*dim);
      const int blocks=(pos+keep)/4-pos/4;
      if(blocks && h.pool16) {
        const int i0=(pos/4)*4+3-pos;
        auto *out=h.pool16+(size_t(slot)*(p_->cap/4)+pos/4)*dim;
        const auto *src=h.closed+size_t(row0+i0)*dim;
        q.parallel_for(sycl::range<1>(size_t(blocks)*dim),[=](sycl::id<1> idx) {
          const size_t i=idx[0]; out[i]=sycl::half(src[(i/dim)*4*dim+i%dim]);
        });
      }
      if(h.pool) for(int i=0;i<keep;++i) if ((pos+i)%4==3)
        q.memcpy(h.pool+(size_t(slot)*(p_->cap/4)+(pos+i)/4)*dim,h.closed+size_t(row0+i)*dim,dim*sizeof(float));
    }
  }
}
}  // namespace xenithium::arch::qwen35::trained_qsa
