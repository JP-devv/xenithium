#pragma once
// Private steady-round diagnostic; no device buffer reads and no timing of text I/O.
#include <level_zero/zet_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace xenithium::util::ruby_round {
inline constexpr std::array<int64_t,6> sample_positions{261884,261909,261916,261948,261955,261972};
inline int steady_sample(int64_t pos) {
  for(size_t i=0;i<sample_positions.size();++i) if(sample_positions[i]==pos) return static_cast<int>(i);
  return -1;
}
struct Trace {
  std::ofstream events, boundaries;
  struct Interval { int64_t pos; int rank; int layer; const char *label; uint64_t begin,end; };
  std::vector<Interval> rows;
  Trace():events("ruby-verifier-events.tsv"),boundaries("ruby-verifier-boundaries.tsv") {
    events<<"pos\trank\tlayer\tlabel\tbegin_ns\tend_ns\n";
    boundaries<<"pos\tsetup_ns\tverifier_gpu_submit_wait_ns\textract_ns\toa_requested\n";
    rows.reserve(8192);
  }
  ~Trace() { for(const auto &r:rows) events<<r.pos<<'\t'<<r.rank<<'\t'<<r.layer<<'\t'<<r.label<<'\t'<<r.begin<<'\t'<<r.end<<'\n'; }
  void add(int64_t p,int c,int l,const char *name,const sycl::event &a,const sycl::event &b) {
    rows.push_back({p,c,l,name,a.get_profiling_info<sycl::info::event_profiling::command_start>(),b.get_profiling_info<sycl::info::event_profiling::command_end>()});
  }
};
struct Oa {
  static constexpr auto backend=sycl::backend::ext_oneapi_level_zero;
  ze_context_handle_t ctx=nullptr;
  ze_device_handle_t dev=nullptr;
  zet_metric_group_handle_t group=nullptr;
  zet_metric_streamer_handle_t stream=nullptr;
  std::vector<std::string> names;
  uint64_t host0=0, ticks0=0, hz=0; uint32_t bits=0;
  int64_t pos=0;
  int rank=0;
  bool active=false;
  bool check(const char *op,ze_result_t rc) const {
    if(rc!=ZE_RESULT_SUCCESS) std::fprintf(stderr,"RUBY_OA rank=%d pos=%lld op=%s rc=0x%x\n",rank,static_cast<long long>(pos),op,unsigned(rc));
    return rc==ZE_RESULT_SUCCESS;
  }
  void enter(sycl::queue &q,int c,int64_t p) {
    rank=c; pos=p;
    ctx=sycl::get_native<backend>(q.get_context());
    dev=sycl::get_native<backend>(q.get_device());
    ze_device_properties_t prop{};prop.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    if(!check("device_properties",zeDeviceGetProperties(dev,&prop)))return;
    hz=prop.timerResolution;bits=prop.timestampValidBits;
    uint32_t n=0;
    if(!check("groups",zetMetricGroupGet(dev,&n,nullptr)) || !n)return;
    std::vector<zet_metric_group_handle_t> g(n);
    if(!check("group_list",zetMetricGroupGet(dev,&n,g.data())))return;
    for(auto h:g) {
      zet_metric_group_properties_t m{};m.stype=ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
      if(!zetMetricGroupGetProperties(h,&m) && m.samplingType==ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED && !std::strcmp(m.name,"ComputeBasic")) {group=h;break;}
    }
    if(!group) {std::fprintf(stderr,"RUBY_OA rank=%d pos=%lld group_unavailable\n",rank,static_cast<long long>(pos));return;}
    uint32_t count=0; if(!check("metric_count",zetMetricGet(group,&count,nullptr)) || !count)return;
    std::vector<zet_metric_handle_t> all(count);
    if(!check("metric_list",zetMetricGet(group,&count,all.data())))return;
    for(auto h:all) {
      zet_metric_properties_t p{};p.stype=ZET_STRUCTURE_TYPE_METRIC_PROPERTIES;
      if(!check("metric_property",zetMetricGetProperties(h,&p)))return;
      names.emplace_back(p.name);
    }
    active=check("activate",zetContextActivateMetricGroups(ctx,dev,1,&group));if(!active)return;
    zet_metric_streamer_desc_t d{};d.stype=ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC;d.samplingPeriod=500000;d.notifyEveryNReports=1024;
    if(!check("stream_open",zetMetricStreamerOpen(ctx,dev,group,&d,nullptr,&stream))) {stream=nullptr;return;}
    std::fprintf(stderr,"RUBY_OA rank=%d pos=%lld opened period_ns=%u notify_reports=%u frequency_hz=%llu valid_bits=%u\n",
      rank,static_cast<long long>(pos),d.samplingPeriod,d.notifyEveryNReports,static_cast<unsigned long long>(hz),bits);
  }
  void begin() {if(stream) check("global_before",zeDeviceGetGlobalTimestamps(dev,&host0,&ticks0));}
  std::vector<uint8_t> raw;
  uint64_t host1=0,ticks1=0;
  bool got=false;
  void end() {
    if(!stream)return;
    check("global_after",zeDeviceGetGlobalTimestamps(dev,&host1,&ticks1));
    size_t size=0;
    if(check("raw_size",zetMetricStreamerReadData(stream,UINT32_MAX,&size,nullptr)) && size && size<=8*1024*1024) {
      raw.resize(size); got=check("raw_read",zetMetricStreamerReadData(stream,UINT32_MAX,&size,raw.data()));raw.resize(size);
    }
    std::fprintf(stderr,"RUBY_OA_ANCHORS rank=%d pos=%lld host0=%llu ticks0=%llu host1=%llu ticks1=%llu hz=%llu bits=%u raw_bytes=%zu good=%d\n",rank,
      static_cast<long long>(pos),static_cast<unsigned long long>(host0),static_cast<unsigned long long>(ticks0),static_cast<unsigned long long>(host1),
      static_cast<unsigned long long>(ticks1),static_cast<unsigned long long>(hz),bits,raw.size(),got);
    check("stream_close",zetMetricStreamerClose(stream));stream=nullptr;
    // Keep the metric group active until raw reports have been decoded.
  }
  void save() {
    if(raw.empty())return;
    const auto name="ruby-oa-"+std::to_string(pos)+"-rank"+std::to_string(rank);
    std::ofstream b(name+".bin",std::ios::binary);b.write(reinterpret_cast<const char *>(raw.data()),raw.size());
    uint32_t n=0;
    if(!got || !check("decode_count",zetMetricGroupCalculateMetricValues(group,ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,raw.size(),raw.data(),&n,nullptr)) || !n)return;
    std::vector<zet_typed_value_t> values(n);
    if(!check("decode",zetMetricGroupCalculateMetricValues(group,ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,raw.size(),raw.data(),&n,values.data())))return;
    if(names.empty() || n%names.size()) {std::fprintf(stderr,"RUBY_OA nonrectangular values=%u metrics=%zu\n",n,names.size());return;}
    std::ofstream f(name+".tsv");f<<std::setprecision(12);
    for(size_t j=0;j<names.size();++j)f<<names[j]<<(j+1==names.size()?'\n':'\t');
    for(uint32_t i=0;i<n;++i) {
      const auto &v=values[i];
      switch(v.type) {
        case ZET_VALUE_TYPE_UINT32: f<<v.value.ui32;break;
        case ZET_VALUE_TYPE_UINT64: f<<v.value.ui64;break;
        case ZET_VALUE_TYPE_FLOAT32: f<<v.value.fp32;break;
        case ZET_VALUE_TYPE_FLOAT64: f<<v.value.fp64;break;
        case ZET_VALUE_TYPE_BOOL8: f<<bool(v.value.b8);break;
        default:f<<"NA";break;
      }
      f<<(i%names.size()+1==names.size()?'\n':'\t');
    }
  }
  ~Oa(){if(stream)check("stream_close_fallback",zetMetricStreamerClose(stream));if(active)check("deactivate",zetContextActivateMetricGroups(ctx,dev,0,nullptr));}
};
}
