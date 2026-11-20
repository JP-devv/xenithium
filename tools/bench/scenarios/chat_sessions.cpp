// Scenario "chat-sessions": the users of a chat service. N users talk at once, each through a conversation of several
// turns - a question, then follow-ups on the answer - with the whole conversation re-sent every turn (as chat clients
// do) and a pause between turns while the user reads and types. The users do not start together, and their pauses
// are random, so their requests interleave as a service's do. What a user feels is the time to the first token of a
// turn and the reading speed of its reply; the service's capacity is the most users it carries while those stay
// within a target (the SLO), and its goodput the tokens per second of the replies that met it.
#include <algorithm>
#include <random>
#include <thread>

#include "scenario.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"users", "4,8,16", "users chatting at once"},
    {"turns", "4", "turns of every user's conversation"},
    {"think_s", "3", "mean pause between a reply and the user's next turn, in seconds (drawn at random)"},
    {"reply_tokens", "300", "tokens per reply"},
    {"slo_ttft_ms", "2000", "the target: a turn's first token within this many ms ..."},
    {"slo_tps", "20", "... and its reply at least this fast, in tok/s"},
    {"slo_pct", "90", "a number of users is carried when at least this share of turns met the target, in percent"},
};
// What users write after the first answer: a follow-up of the kind every chat has.
constexpr std::string_view kFollowUps[] = {
    "Thanks. Can you go into more detail on the most important point?",
    "Give me a shorter version of that as a bulleted list.",
    "Can you give a concrete example?",
    "What would you do differently if you had half the time?",
    "Summarise our conversation so far in three sentences.",
};
constexpr std::string_view kNoAnswer = "(no answer)";
constexpr double kPercent = 100;

// A user: the requests of its turns (seeds and lengths fixed up front, on the calling thread), when it starts, its
// pauses, and what came back.
struct User {
  std::vector<ChatRequest> turns;
  double start_after_s = 0;
  std::vector<double> pauses_s;
  std::vector<Sample> replies;  // shorter than `turns` when a turn failed
};

// A pause of mean `mean_s` seconds, exponentially distributed (a user's reading and typing time); none for mean 0.
double pause_s(double mean_s, std::mt19937_64 &rng) { return mean_s > 0 ? exponential_gap_s(1 / mean_s, rng) : 0; }

// The first message of user `u`: the short prompts one category after the other, so that users ask different kinds of
// things (prompts.cpp lists three per category in a row).
std::string_view first_message(size_t u) {
  constexpr size_t kPromptsPerCategory = 3;
  const auto prompts = short_prompts();
  const size_t categories = prompts.size() / kPromptsPerCategory, category = u % categories, variant = (u / categories) % kPromptsPerCategory;
  return prompts[category * kPromptsPerCategory + variant].text;
}

std::vector<User> plan_users(Session &session, int64_t count, const Params &params, std::mt19937_64 &rng) {
  const double think_s = params.real("think_s");
  std::vector<User> users(static_cast<size_t>(count));
  for (size_t u = 0; u < users.size(); ++u) {
    for (int64_t t = 0; t < params.integer("turns"); ++t) {
      ChatRequest request = session.sampled({}, static_cast<int>(params.integer("reply_tokens")));
      request.thinking = Thinking::off;  // the history holds answers, not reasoning
      users[u].turns.push_back(std::move(request));
      users[u].pauses_s.push_back(pause_s(think_s, rng));
    }
    if (users[u].turns.empty()) continue;
    users[u].turns.front().messages = {{"user", std::string(first_message(u))}};
    users[u].start_after_s = pause_s(think_s, rng);
  }
  return users;
}

// The user's conversation: each turn re-sends the whole history plus the next message, then the user pauses.
void converse(const Client &client, User &user) {
  if (user.turns.empty()) return;
  std::this_thread::sleep_for(std::chrono::duration<double>(user.start_after_s));
  std::vector<Message> conversation = user.turns.front().messages;
  for (size_t t = 0; t < user.turns.size(); ++t) {
    if (t > 0) conversation.push_back({"user", std::string(kFollowUps[(t - 1) % std::size(kFollowUps)])});
    user.turns[t].messages = conversation;
    Sample reply = client.chat(user.turns[t]);
    const bool ok = reply.ok;
    conversation.push_back({"assistant", reply.text.empty() ? std::string(kNoAnswer) : reply.text});
    user.replies.push_back(std::move(reply));
    if (!ok) return;  // the conversation cannot go on without this turn
    if (t + 1 < user.turns.size()) std::this_thread::sleep_for(std::chrono::duration<double>(user.pauses_s[t]));
  }
}

class ChatSessions final : public Scenario {
 public:
  std::string_view id() const override { return "chat-sessions"; }
  std::string_view title() const override { return "Chat users with pauses, against a latency target (sampled)"; }
  std::string_view about() const override {
    return "4, 8 and 16 users each holding a conversation of four turns, re-sent every turn, with random pauses to read and type: the TTFT and reading speed "
           "of every turn, the share of turns within the target (first token in 2 s, 20 tok/s), and the tokens per second of all replies and of those "
           "within it (goodput).";
  }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const double slo_ttft_ms = params.real("slo_ttft_ms"), slo_tps = params.real("slo_tps"), slo_pct = params.real("slo_pct");
    std::mt19937_64 rng(session.seed());
    std::string carried;  // the most users whose turns met the target
    for (const int64_t count : params.integers("users")) {
      std::vector<User> users = plan_users(session, count, params, rng);
      const PowerSpan power(session);
      const Clock::time_point started = Clock::now();
      {
        std::vector<std::jthread> threads;
        for (User &user : users) threads.emplace_back([&session, &user] { converse(session.client(), user); });
      }  // joins every user
      const double wall_s = seconds_since(started);

      Row row = Row::with_labels({{"users", std::to_string(count)}});
      int64_t tokens = 0, good_tokens = 0;
      int turns = 0, good_turns = 0;
      for (const User &user : users)
        for (const Sample &reply : user.replies) {
          if (!tally(result, reply)) continue;
          ++turns, tokens += reply.completion_tokens;
          row.metric(std::string(metric::kTtft), "ms", Better::lower).samples.push_back(reply.ttft_ms);
          if (reply.decode_tps() > 0) row.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples.push_back(reply.decode_tps());
          if (reply.ttft_ms <= slo_ttft_ms && reply.decode_tps() >= slo_tps) ++good_turns, good_tokens += reply.completion_tokens;
        }
      if (turns > 0) {
        const double within_pct = kPercent * good_turns / turns;
        row.metric("within target", "%", Better::higher).samples.push_back(within_pct);
        row.metric(std::string(metric::kTotal), "tok/s", Better::higher).samples.push_back(static_cast<double>(tokens) / wall_s);
        row.metric("goodput", "tok/s", Better::higher).samples.push_back(static_cast<double>(good_tokens) / wall_s);
        if (within_pct >= slo_pct) carried = std::to_string(count);
      }
      power.add_to(row, tokens);
      session.progress("chat-sessions: " + std::to_string(count) + " users done");
      result.rows.push_back(std::move(row));
    }
    const std::string target = "first token within " + format_number(slo_ttft_ms) + " ms and at least " + format_number(slo_tps) + " tok/s for " + format_number(slo_pct) + " % of turns";
    if (!result.rows.empty())
      result.rows.back().notes.push_back(carried.empty() ? "no number of users met the target (" + target + ")"
                                                         : "the most users within the target (" + target + "): " + carried);
    result.charts = {
        {ChartKind::line, "Tokens per second of all replies, and of those within the target (goodput)", "users", {std::string(metric::kTotal), "goodput"}, true},
        {ChartKind::range, "Time to first token of a turn by users", "users", {std::string(metric::kTtft)}, false},
        {ChartKind::range, "Reading speed of a reply by users", "users", {std::string(metric::kDecode)}, false},
        {ChartKind::bars, "Turns within the target", "users", {"within target"}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_chat_sessions() { return std::make_unique<ChatSessions>(); }

}  // namespace xenithium:bench
