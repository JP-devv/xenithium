// Scenario "long-agents": N agents at work, each on its own long document. First every agent's document is read at
// once (N cold long prompts side by side), then every agent works through tasks of every kind of output - maths,
// code, edits, prose, JSON - turn by turn on that context, until it has produced 10K tokens. It is the load of
// several coding agents on large repositories: the endpoint holds N long contexts, every decode step attends to a
// long context per stream, and every turn re-sends its conversation and has to come from the prefix cache.
// How long "long" is depends on the endpoint: the context it can hold differs with the cards, their number and the
// KV format. So the documents are sized from what the endpoint reports - by default so that the most agents of the
// run together fill three quarters of its KV pool (about 200K tokens each for four agents on a 1M-token pool).
#include <algorithm>
#include <map>
#include <thread>

#include "scenario.hpp"
#include "stats.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"agents", "1,2,4", "agents working at once (each holds its own document: the endpoint needs room for all of them)"},
    {"depth", "auto", "each agent's document in tokens; auto = sized from pool_pct"},
    {"pool_pct", "75", "with depth auto: the share of the endpoint's KV pool that the most agents' contexts fill together, in percent"},
    {"produce_tokens", "10000", "tokens each agent generates over its tasks"},
    {"kind", "code", "corpus kind of the documents"},
};
constexpr std::string_view kLoadQuestion = "\n\nYou will be given tasks while this text stays in your context. Reply with the single word READY.";
constexpr int kLoadReplyTokens = 8;
constexpr std::string_view kNoAnswer = "(no answer)";
constexpr std::string_view kCategoryTable = "decode tok/s by kind of output, per number of agents";
// Tokens of room kept for the tasks' own text and the chat template around every turn.
constexpr int64_t kTaskTextTokens = 4000;
constexpr std::string_view kAutoDepth = "auto";
// Under this a document is not a long context and the scenario says so instead of running.
constexpr int64_t kMinDepthTokens = 8000;
// An automatic depth is rounded down to this, so two endpoints of nearly the same pool run the same depth.
constexpr int64_t kDepthStepTokens = 1000;
constexpr double kPercent = 100;

// Each agent's document length and, for the report, how it was arrived at.
struct Depth {
  int64_t tokens = 0;  // 0 = the scenario cannot run; `reason` says why
  std::string reason;
};

// The asked depth, or with "auto" the depth at which `most_agents` contexts (document + what an agent produces) fill
// pool_pct % of what the endpoint holds at once. An endpoint that reports no shared pool is taken to hold one full
// context per agent; a single request never exceeds the endpoint's context length either way.
Depth document_depth(const Session &session, const Params &params, int64_t most_agents, int64_t per_agent_extra) {
  if (params.text("depth") != kAutoDepth) return {params.integer("depth"), "as asked"};
  const double share = params.real("pool_pct") / kPercent;
  const int64_t pool = session.kv_pool_tokens(), context = session.context_tokens();
  if (pool <= 0 && context <= 0) return {0, "skipped: the endpoint reports neither a KV pool nor a context length; set --kv-pool, --context or long-agents.depth"};
  int64_t per_agent = pool > 0 ? static_cast<int64_t>(share * static_cast<double>(pool)) / most_agents : static_cast<int64_t>(share * static_cast<double>(context));
  if (context > 0) per_agent = std::min(per_agent, context);
  const int64_t depth = (per_agent - per_agent_extra) / kDepthStepTokens * kDepthStepTokens;
  const std::string basis = pool > 0 ? params.text("pool_pct") + " % of the endpoint's KV pool of " + with_thousands(pool) + " tokens over " + std::to_string(most_agents) + " agents"
                                     : params.text("pool_pct") + " % of the endpoint's context of " + with_thousands(context) + " tokens (it reports no KV pool)";
  if (depth < kMinDepthTokens) return {0, "skipped: " + basis + " leaves " + with_thousands(std::max<int64_t>(depth, 0)) + " tokens per document, too few for a long context"};
  return {depth, basis};
}

// One task of an agent: a short prompt and the request that asks it (its seed and reply length are fixed up front,
// on the calling thread; the agent's thread adds the conversation).
struct Task {
  const ShortPrompt *prompt;
  ChatRequest request;
};

// The short prompts one category after the other (maths, then code, then an edit...), from `first` on, until their
// reply lengths add up to `produce_tokens`; the last reply is cut to fit.
std::vector<Task> plan_tasks(Session &session, size_t first, int64_t produce_tokens) {
  // prompts.cpp lists three prompts per category in a row: walking it with this stride changes category every task
  constexpr size_t kPromptsPerCategory = 3;
  const auto prompts = short_prompts();
  const size_t categories = prompts.size() / kPromptsPerCategory;
  std::vector<Task> tasks;
  for (size_t step = 0; produce_tokens > 0; ++step) {
    const size_t turn = first + step, category = turn % categories, variant = (turn / categories) % kPromptsPerCategory;
    const ShortPrompt &prompt = prompts[category * kPromptsPerCategory + variant];
    const int reply_tokens = static_cast<int>(std::min<int64_t>(prompt.max_tokens, produce_tokens));
    ChatRequest request = session.sampled({}, reply_tokens);
    request.thinking = Thinking::off;  // the history holds answers, not reasoning
    tasks.push_back({&prompt, std::move(request)});
    produce_tokens -= reply_tokens;
  }
  return tasks;
}

// An agent: its document, the request that loads it, its tasks and what came back.
struct Agent {
  ChatRequest load;
  std::vector<Task> tasks;
  Sample loaded;
  std::vector<Sample> answers;  // one per task asked; shorter than `tasks` when a turn failed
};

// Runs `step` for every agent at once and waits for all.
template <typename Step>
void in_parallel(std::vector<Agent> &agents, Step step) {
  std::vector<std::jthread> threads;
  for (Agent &agent : agents) threads.emplace_back([&agent, &step] { step(agent); });
}

// The agent's turns: every request is the whole conversation so far plus the next task.
void work(const Client &client, Agent &agent) {
  std::vector<Message> conversation = agent.load.messages;
  conversation.push_back({"assistant", agent.loaded.text.empty() ? std::string(kNoAnswer) : agent.loaded.text});
  for (Task &task : agent.tasks) {
    conversation.push_back({"user", std::string(task.prompt->text)});
    task.request.messages = conversation;
    Sample answer = client.chat(task.request);
    const bool ok = answer.ok;
    conversation.push_back({"assistant", answer.text.empty() ? std::string(kNoAnswer) : answer.text});
    agent.answers.push_back(std::move(answer));
    if (!ok) return;  // the session cannot go on without this turn
  }
}

class LongAgents final : public Scenario {
 public:
  std::string_view id() const override { return "long-agents"; }
  std::string_view title() const override { return "Agents at work on long contexts (a document each, then tasks of every kind)"; }
  std::string_view about() const override {
    return "1, 2 and 4 agents, each reading its own long document at once (sized so that four fill three quarters of the endpoint's KV pool) and then "
           "producing 10K tokens over maths, code, edits and the other kinds of output on that context: the prompt rate of the simultaneous reads, decode "
           "tok/s per request and of all agents, and the TTFT of turns that re-send the conversation.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const int64_t produce_tokens = params.integer("produce_tokens");
    const std::vector<int64_t> levels = params.integers("agents");
    const std::string kind = params.text("kind");
    const Depth sized = document_depth(session, params, levels.empty() ? 1 : std::ranges::max(levels), produce_tokens + kTaskTextTokens);
    const int64_t depth = sized.tokens;
    std::string skipped = depth > 0 ? std::string() : sized.reason;
    if (depth <= 0 || depths_that_fit(session, std::span(&depth, 1), static_cast<int>(produce_tokens + kTaskTextTokens), skipped).empty()) {
      result.rows.push_back(Row::note_only(skipped));
      return result;
    }
    session.progress("long-agents: documents of " + with_thousands(depth) + " tokens (" + sized.reason + ")");
    (void)session.chars_per_token(kind);  // calibrate before anything is timed

    std::map<std::string, Row, std::less<>> by_category;  // category -> its row of the per-category table
    size_t first_file = 0;  // every agent of every level reads another part of the corpus
    for (const int64_t count : levels) {
      std::vector<Agent> agents(static_cast<size_t>(count));
      bool wrapped = false;
      for (size_t i = 0; i < agents.size(); ++i) {
        const Session::DocumentPrompt document = session.document_prompt(kind, depth, kLoadQuestion, /*unique=*/true, first_file++);
        wrapped |= document.wrapped;
        agents[i].load = session.greedy(document.text, kLoadReplyTokens);
        agents[i].load.ignore_eos = false;
        agents[i].load.thinking = Thinking::off;
        agents[i].tasks = plan_tasks(session, i, produce_tokens);  // agent i starts at another kind of task
      }
      Row row = Row::with_labels({{"agents", std::to_string(count)}});
      if (result.rows.empty()) row.notes.push_back("each agent's document: " + with_thousands(depth) + " tokens (" + sized.reason + ")");
      if (wrapped) row.notes.push_back("the " + kind + " corpus is shorter than " + with_thousands(depth) + " tokens: its files repeat inside a document");

      // every document read at once
      const Clock::time_point load_started = Clock::now();
      in_parallel(agents, [&](Agent &agent) { agent.loaded = session.client().chat(agent.load); });
      const double load_s = seconds_since(load_started);
      int64_t prompt_tokens = 0;
      std::erase_if(agents, [&](const Agent &agent) { return !tally(result, agent.loaded); });  // an agent without its document does not work
      for (const Agent &agent : agents) {
        prompt_tokens += agent.loaded.prompt_tokens;
        row.metric(std::string(metric::kPromptTokens), "tokens", Better::none).samples.push_back(agent.loaded.prompt_tokens);
        row.metric("read TTFT", "ms", Better::lower).samples.push_back(agent.loaded.ttft_ms);
      }
      if (prompt_tokens > 0) row.metric("read", "tok/s", Better::higher).samples.push_back(static_cast<double>(prompt_tokens) / load_s);
      session.progress("long-agents: " + std::to_string(count) + " document(s) of " + with_thousands(depth) + " tokens read");

      // every agent at work
      const PowerSpan power(session);
      const Clock::time_point work_started = Clock::now();
      in_parallel(agents, [&](Agent &agent) { work(session.client(), agent); });
      const double work_s = seconds_since(work_started);
      int64_t tokens = 0;
      for (const Agent &agent : agents)
        for (size_t t = 0; t < agent.answers.size(); ++t) {
          const Sample &answer = agent.answers[t];
          if (!tally(result, answer)) continue;
          tokens += answer.completion_tokens;
          row.metric("turn TTFT", "ms", Better::lower).samples.push_back(answer.ttft_ms);
          if (answer.decode_tps() <= 0) continue;
          row.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples.push_back(answer.decode_tps());
          row.metric(std::string(metric::kTokensPerUpdate), "", Better::higher).samples.push_back(answer.tokens_per_update());
          if (answer.update_gap_ms() > 0) row.metric(std::string(metric::kUpdateGap), "ms", Better::lower).samples.push_back(answer.update_gap_ms());
          Row &category = by_category[std::string(agent.tasks[t].prompt->category)];
          category.metric(std::to_string(count) + (count == 1 ? " agent" : " agents"), "tok/s", Better::higher).samples.push_back(answer.decode_tps());
        }
      if (tokens > 0) row.metric(std::string(metric::kTotal), "tok/s", Better::higher).samples.push_back(static_cast<double>(tokens) / work_s);
      power.add_to(row, tokens);
      session.progress("long-agents: " + std::to_string(count) + " agent(s) produced " + with_thousands(tokens) + " tokens");
      result.rows.push_back(std::move(row));
    }

    std::vector<std::string> level_names;
    for (const CategoryWeight &weight : category_weights()) {  // the categories in their usual order
      const auto it = by_category.find(weight.category);
      if (it == by_category.end()) continue;
      for (const Metric &m : it->second.metrics)
        if (std::ranges::find(level_names, m.name) == level_names.end()) level_names.push_back(m.name);
      it->second.labels = {{"category", std::string(weight.category)}};
      it->second.table = std::string(kCategoryTable);
      result.rows.push_back(std::move(it->second));
    }
    result.charts = {
        {ChartKind::line, "Tokens per second by agents: all together and per request", "agents", {std::string(metric::kTotal), std::string(metric::kDecode)}, true},
        {ChartKind::line, "Prompt tokens per second while every document is read at once", "agents", {"read"}, true},
        {ChartKind::range, "Time to first token of a turn on the cached context", "agents", {"turn TTFT"}, false},
        {ChartKind::bars, "Tokens per joule by agents", "agents", {std::string(metric::kEfficiency)}, false},
        {ChartKind::bars, "Decode rate by kind of output at this depth", "category", level_names, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_long_agents() { return std::make_unique<LongAgents>(); }

}  // namespace xenithium:bench
