// The exclusive test window supplies a real Swift checkpoint and trained bank.
#include "xenithium/arch/qwen35/trained_qsa_device.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace qwen=xenithium::arch::qwen35;
namespace t=qwen::trained_qsa;
namespace k=xenithium::kernels;
namespace m=xenithium::model;

int main(int argc,char **argv) try {
  if(argc!=4 || std::string_view(argv[1])!="--reserved-two-b60")
    throw std::invalid_argument("usage: --reserved-two-b60 TARGET_DIR TRAINED_BANK_DIR");
  const auto target=qwen::Config::parse(m::ModelConfig::load(argv[2]));
  const auto checkpoint=m::Checkpoint::open(argv[3],false);
  std::vector<t::Layer> weights;
  for(int layer=3;layer<=63;layer+=4) weights.push_back(t::Layer::load(checkpoint,target,layer,t::NormStorage::delta));
  const auto loader=xenithium::l0::Loader::open();
  if(loader.size()!=2) throw std::runtime_error("expected two reserved B60s");
  for(int c=0;c<2;++c) if(loader.info(c).vendor_id!=0x8086 || loader.info(c).device_id!=0xe211)
    throw std::runtime_error("expected two Intel 8086:e211 devices");
  xenithium::Fabric fab(loader,{0,1},false);
  constexpr int H=5120,cap=177216,pf=2048,slots=5,step_rows=32,prefix=16382;
  for(const char *v:{"XEN_TRAINED_QSA_SHARE_SCRATCH","XEN_TRAINED_QSA_STREAM_PROJECTION","XEN_TRAINED_QSA_TILED_SCORE"}) unsetenv(v);
  t::Device stock(fab,weights,slots,cap,pf,step_rows);
  for(const char *v:{"XEN_TRAINED_QSA_SHARE_SCRATCH","XEN_TRAINED_QSA_STREAM_PROJECTION","XEN_TRAINED_QSA_TILED_SCORE"}) setenv(v,"1",1);
  t::Device trial(fab,weights,slots,cap,pf,step_rows);
  struct Alloc {
    xenithium::Fabric &fab; std::vector<void*> pointers;
    ~Alloc(){ for(int c=0;c<2;++c) fab.compute(c).wait(); for(void *p:pointers) fab.release(p); }
    void *make(int c,size_t n) { auto *p=fab.alloc(c,n); pointers.push_back(p); return p; }
  } alloc{fab,{}};
  std::array<k::gqa_bf16*,2> x{}; std::array<int32_t*,2> rec{};
  for(int c=0;c<2;++c) { x[c]=static_cast<k::gqa_bf16*>(alloc.make(c,size_t(pf)*H*2)); rec[c]=static_cast<int32_t*>(alloc.make(c,4096)); }
  std::vector<k::gqa_bf16> hidden(size_t(pf)*H);
  auto fill=[&](int r,int slot,int pos,int salt,int rank) {
    for(int i=0;i<H;++i) {
      const uint32_t u=uint32_t(i)*337u+uint32_t(pos)*19391u+uint32_t(slot+5)*3149u+uint32_t(salt)*8759u+rank*3191u;
      hidden[size_t(r)*H+i]=k::gqa_bf16(float(int((u^(u>>11)^(u*16777619u>>17))&1023)-512)/512.f);
    }
  };
  constexpr std::array<int,2> live{4,1};  // nontrivial slot ordering and distinct keys
  std::array<int,16> check_layers{};
  std::iota(check_layers.begin(),check_layers.end(),0);
  for(int slot:live) for(int p=0;p<prefix;p+=pf) {
    const int m=std::min(pf,prefix-p);
    for(int c=0;c<2;++c) {
      auto &queue=fab.compute(c);
      for(int r=0;r<m;++r) fill(r,slot,p+r,0,c);
      queue.memcpy(x[c],hidden.data(),size_t(m)*H*2).wait();
      for(int l:check_layers) { stock.prefill(c,x[c],m,slot,p,l); trial.prefill(c,x[c],m,slot,p,l); }
    }
    for(int c=0;c<2;++c) fab.compute(c).wait_and_throw();
  }
  constexpr std::array<int,4> path{0,2,4,6};
  const uint64_t path_word=k::tree_path_word(path.data(),static_cast<int>(path.size()));
  auto stage=[&](t::Device &d,int l,int p,std::array<int,2> widths,int salt,bool tree=false) {
    const int rows=widths[0]+widths[1];
    const int layout=widths[0]==widths[1]?widths[0]:k::seg_het2(widths[0],1,2,widths[1]);
    std::array<int32_t,2*k::kSegRec> host{};
    for(int s=0;s<2;++s) { host[s*k::kSegRec+k::kSegPos]=p; host[s*k::kSegRec+k::kSegSlot]=live[s]; }
    for(int c=0;c<2;++c) {
      auto &queue=fab.compute(c);
      for(int s=0,r=0;s<2;++s) for(int j=0;j<widths[s];++j,++r) {
        const int logical=tree?(j%2?j:j/2):j;
        int variant=tree && j%2?salt+20:salt;
        if(p==prefix && j==0) variant=0;
        fill(r,live[s],p+logical,variant,c);
      }
      std::fill_n(hidden.data()+size_t(rows)*H,size_t(k::row_shape(rows)-rows)*H,k::gqa_bf16(0.f));
      queue.memcpy(x[c],hidden.data(),size_t(k::row_shape(rows))*H*2).wait();
      queue.memcpy(rec[c],host.data(),sizeof(host)).wait();
      k::GqaStep step{rec[c],p+widths[0],k::kSegRec,layout}; step.tree=tree;
      d.stage(c,x[c],rows,2,step,l,3+4*l);
      if(bool(step.trained_qsa.entries)==tree || bool(step.trained_qsa.counts)==tree)
        throw std::runtime_error("unexpected tree/chain dispatch");
    }
    for(int c=0;c<2;++c) fab.compute(c).wait_and_throw();
  };
  auto commit=[&](t::Device &d,int p,int width0,int keep,bool tree=false) {
    d.accept(live[0],p,0,keep,tree,tree?path_word:0);
    d.accept(live[1],p,width0,keep,tree,tree?path_word:0);
  };
  auto compare=[&](const char *phase,int layer,int pos,int a,int b) {
    for(int c=0;c<2;++c) for(int r=0;r<a+b;++r) {
      const auto x=stock.read_selection(c,r),y=trial.read_selection(c,r);
      const int expected=(pos+(r<a?r:r-a)+1)/4;
      if(x.eligible!=expected || x.eligible!=y.eligible || x.ids.size()!=512 || x.ids!=y.ids || x.pages!=y.pages || x.pages<=0 ||
         stock.read_logical_pages(c,r,x.pages)!=trial.read_logical_pages(c,r,y.pages))
        throw std::runtime_error(std::string(phase)+": stock differs rank="+std::to_string(c)+" layer="+std::to_string(3+4*layer)+" row="+std::to_string(r));
      stock.check_selector_reference(c,r,x); trial.check_selector_reference(c,r,y);
      std::cout<<phase<<" rank="<<c<<" layer="<<3+4*layer<<" row="<<r<<" eligible="<<expected<<" ids=512 plan_pages="<<x.pages<<" exact=1\n";
    }
  };
  // Stage longer poisoned suffixes; they include the row closing the block.
  for(int l:check_layers) { stage(trial,l,prefix,{3,2},11); stage(stock,l,prefix,{1,1},0); }
  commit(trial,prefix,3,1); commit(stock,prefix,1,1);
  for(int l:check_layers) {
    stage(trial,l,prefix+1,{16,8},2); stage(stock,l,prefix+1,{16,8},2);
    compare("reject-reappend",l,prefix+1,16,8);
  }
  commit(trial,prefix+1,16,1); commit(stock,prefix+1,16,1);
  // Commit the same four logical rows through a walked non-identity tree
  // on the trial and a clean contiguous chain on stock (closes a new block).
  for(int l:check_layers) { stage(trial,l,prefix+2,{7,7},3,true); stage(stock,l,prefix+2,{4,4},3); }
  commit(trial,prefix+2,7,4,true); commit(stock,prefix+2,4,4);
  for(int l:check_layers) {
    stage(trial,l,prefix+6,{8,6},4); stage(stock,l,prefix+6,{8,6},4);
    compare("tree-reappend",l,prefix+6,8,6);
  }
  std::cout<<"PASS actual allocation at five-slot cap, both ranks, reject/reappend and walked-tree selection\n";
  return 0;
} catch(const std::exception &e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
