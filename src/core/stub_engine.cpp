#include "xenithium/core/stub_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace xenithium::core {

StubEngine::StubEngine(StubConfig cfg) : cfg_(std::move(cfg)) {
  if (cfg_.script.empty()) cfg_.script.push_back(0);
  cfg_.tokens_per_step = std::clamp(cfg_.tokens_per_step, 1, Event::kMaxTokens);
  stats_.slots = static_cast<uint32_t>(std::max(0, cfg_.slots));
}

StubEngine::~StubEngine() { stop(); }

void StubEngine::start(EventSink* sink) {
  sink_ = sink;
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = false;
  }
  thread_ = std::thread([this] { loop(); });
}

void StubEngine::stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void StubEngine::submit(uint64_t id, std::vector<int32_t> prompt, const SamplingParams& params) {
  {
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.waiting++;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t len = prompt.size();
    submits_.push_back({id, len, params, std::chrono::steady_clock::now(), cfg_.prefix_cache ? std::move(prompt) : std::vector<int32_t>{}});
  }
  cv_.notify_one();
}

void StubEngine::cancel(uint64_t id) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    cancels_.push_back(id);
  }
  cv_.notify_one();
}

Stats StubEngine::stats() const {
  std::lock_guard<std::mutex> lk(stats_mu_);
  return stats_;
}

bool StubEngine::is_stop_token(const Req& r, int32_t tok) const {
  if (std::find(r.params.stop_token_ids.begin(), r.params.stop_token_ids.end(), tok) != r.params.stop_token_ids.end())
    return true;
  if (r.params.ignore_eos) return false;
  return std::find(cfg_.eos_ids.begin(), cfg_.eos_ids.end(), tok) != cfg_.eos_ids.end();
}

void StubEngine::loop() {
  using clock = std::chrono::steady_clock;
  const auto step = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double, std::milli>(cfg_.step_ms));
  std::vector<Submit> submits;
  std::vector<uint64_t> cancels;
  std::vector<Event> events;
  auto next = clock::now();

  while (true) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      if (active_.empty() && waiting_.empty()) {
        cv_.wait(lk, [&] { return stop_ || !submits_.empty() || !cancels_.empty(); });
        next = clock::now();  // idle -> the next step starts now
      }
      if (stop_) return;
      submits.swap(submits_);
      cancels.swap(cancels_);
    }
    events.clear();

    using dsec = std::chrono::duration<double>;
    for (auto& s : submits) waiting_.push_back(std::move(s));
    submits.clear();

    // A request cancelled while it still waits never runs.
    for (uint64_t id : cancels) {
      const auto w = std::find_if(waiting_.begin(), waiting_.end(), [&](const Submit& s) { return s.id == id; });
      if (w != waiting_.end()) {
        waiting_.erase(w);
        {
          std::lock_guard<std::mutex> sl(stats_mu_);
          stats_.waiting--;
          stats_.finished_by[static_cast<int>(Finish::cancelled)]++;
        }
        Event e;
        e.id = id;
        e.finish = Finish::cancelled;
        events.push_back(std::move(e));
      } else if (active_.erase(id)) {
        {
          std::lock_guard<std::mutex> sl(stats_mu_);
          stats_.finished_by[static_cast<int>(Finish::cancelled)]++;
        }
        Event e;
        e.id = id;
        e.finish = Finish::cancelled;
        events.push_back(std::move(e));
      }
    }
    cancels.clear();

    // Queue deadline, then admission into free slots (all of them when slots == 0).
    {
      const auto now = clock::now();
      const auto late_at = admission_expired(waiting_, cfg_.admission, [&](const Submit& s) {
        return dsec(now - s.t_submit).count();
      });
      for (auto it = late_at.rbegin(); it != late_at.rend(); ++it) {
        Event e;
        e.id = waiting_[*it].id;
        e.finish = Finish::expired;
        e.error = "queue deadline exceeded";
        events.push_back(std::move(e));
        waiting_.erase(waiting_.begin() + static_cast<std::ptrdiff_t>(*it));
        std::lock_guard<std::mutex> sl(stats_mu_);
        stats_.waiting--;
        stats_.finished_by[static_cast<int>(Finish::expired)]++;
      }
    }
    size_t new_prompt = 0;
    while (!waiting_.empty() && (cfg_.slots <= 0 || active_.size() < static_cast<size_t>(cfg_.slots))) {
      const auto now = clock::now();
      const size_t k = admission_pick(
          waiting_, cfg_.admission, [](const Submit& s) { return s.params.max_tokens; },
          [&](const Submit& s) { return dsec(now - s.t_submit).count(); });
      Submit s = std::move(waiting_[k]);
      waiting_.erase(waiting_.begin() + static_cast<std::ptrdiff_t>(k));
      Req r;
      r.params = s.params;
      r.prompt_len = s.prompt_len;
      r.t_submit = s.t_submit;
      r.t_admit = now;
      // Deterministic: every request reads the script from the
      // start; a seed shifts the start, so seeded sampling still differs per seed.
      if (r.params.has_seed && r.params.temperature > 0.0f) r.pos = r.params.seed % cfg_.script.size();
      new_prompt += s.prompt_len;
      if (cfg_.prefix_cache && !s.prompt.empty()) {
        for (const auto& p : seen_) {
          const size_t lim = std::min(p.size(), s.prompt.size() - 1);
          size_t k = 0;
          while (k < lim && p[k] == s.prompt[k]) ++k;
          r.cached = std::max(r.cached, k);
        }
        seen_.push_back(std::move(s.prompt));
        if (seen_.size() > 16) seen_.pop_front();
        std::lock_guard<std::mutex> sl(stats_mu_);
        stats_.prefix_queries += s.prompt_len;
        stats_.prefix_hits += r.cached;
      }
      {
        std::lock_guard<std::mutex> sl(stats_mu_);
        stats_.waiting--;
        stats_.queue_time.observe(dsec(now - s.t_submit).count());
        if (r.params.max_tokens > 0) stats_.running = static_cast<uint32_t>(active_.size() + 1);
      }
      if (r.params.max_tokens <= 0) {
        Event e;
        e.id = s.id;
        e.finish = Finish::length;
        events.push_back(std::move(e));
        std::lock_guard<std::mutex> sl(stats_mu_);
        stats_.finished_by[static_cast<int>(Finish::length)]++;
        stats_.e2e.observe(dsec(clock::now() - s.t_submit).count());
        continue;
      }
      active_[s.id] = r;
    }

    uint64_t drafts = 0, accepted = 0, generated = 0, finished = 0;
    Histogram ttft_add, prefill_add, e2e_add;
    uint64_t fin_by[Stats::kFinishKinds] = {};
    for (auto it = active_.begin(); it != active_.end();) {
      Req& r = it->second;
      Event e;
      e.id = it->first;
      const int want = r.prefilled ? cfg_.tokens_per_step : 1;
      const int room = r.params.max_tokens - r.generated;
      const int n = std::min(want, room);
      for (int i = 0; i < n; ++i) {
        int32_t tok;
        if (cfg_.eos_after > 0 && !r.params.ignore_eos && r.generated >= cfg_.eos_after && !cfg_.eos_ids.empty()) {
          tok = cfg_.eos_ids[0];
        } else {
          tok = cfg_.script[r.pos++ % cfg_.script.size()];
        }
        e.tokens[e.n++] = tok;
        ++r.generated;
        if (r.params.logprobs >= 0) {  // fixed values: the server's formatting is what is under test
          TokenLogprob lp;
          lp.logprob = -1.0f;
          for (int k = 0; k < r.params.logprobs; ++k) lp.top.push_back({tok + k, -1.0f - static_cast<float>(k)});
          e.logprobs.push_back(std::move(lp));
        }
        if (is_stop_token(r, tok)) {
          e.finish = Finish::stop;
          e.stop_token = true;
          break;
        }
      }
      if (!r.prefilled) e.cached_tokens = static_cast<int32_t>(r.cached);
      if (!r.prefilled && r.params.prompt_logprobs >= 0) {
        e.prompt_logprobs.resize(r.prompt_len);
        for (size_t k = 0; k < r.prompt_len; ++k) e.prompt_logprobs[k].logprob = k == 0 ? std::nanf("") : -0.5f;
      }
      if (r.prefilled) {
        ++drafts;
        accepted += e.n > 0 ? static_cast<uint64_t>(e.n - 1) : 0;
      }
      if (!r.prefilled && e.n > 0) {
        const auto t = clock::now();
        ttft_add.observe(dsec(t - r.t_submit).count());
        prefill_add.observe(dsec(t - r.t_admit).count());
      }
      r.prefilled = true;
      generated += e.n;
      if (e.finish == Finish::none && r.generated >= r.params.max_tokens) e.finish = Finish::length;
      const bool done = e.finish != Finish::none;
      const Finish fin = e.finish;
      events.push_back(std::move(e));
      if (done) {
        ++fin_by[static_cast<int>(fin)];
        e2e_add.observe(dsec(clock::now() - r.t_submit).count());
        ++finished;
        it = active_.erase(it);
      } else {
        ++it;
      }
    }

    {
      std::lock_guard<std::mutex> lk(stats_mu_);
      ++stats_.steps;
      stats_.spec_drafts += drafts;
      stats_.spec_accepted_tokens += accepted;
      stats_.prompt_tokens += new_prompt;
      stats_.generation_tokens += generated;
      stats_.requests_finished += finished;
      stats_.running = static_cast<uint32_t>(active_.size());
      auto merge = [](Histogram& to, const Histogram& from) {
        for (int i = 0; i <= Histogram::kBounds; ++i) to.bucket[i] += from.bucket[i];
        to.sum += from.sum;
        to.count += from.count;
      };
      merge(stats_.ttft, ttft_add);
      merge(stats_.prefill_time, prefill_add);
      merge(stats_.e2e, e2e_add);
      for (int i = 0; i < Stats::kFinishKinds; ++i) stats_.finished_by[i] += fin_by[i];
    }
    if (!events.empty() && sink_) sink_->on_events(events.data(), events.size());

    next += step;
    const auto now = clock::now();
    if (next < now) next = now;
    std::this_thread::sleep_until(next);
  }
}

}  // namespace xenithium::core
