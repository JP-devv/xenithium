// Scenario "structured": output constrained to a JSON schema (response_format json_schema). The constraint runs on
// the server's sampling path, so it has its own speed, and a constrained reply that is not valid JSON is a bug.
#include "scenario.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"passes", "3", "requests sent"},
    {"reply_tokens", "800", "the most tokens a reply may take (the object needs about 250)"},
};
constexpr std::string_view kPrompt =
    "Produce a catalogue of three fictional products for a hardware store, following the schema.";
constexpr std::string_view kResponseFormat = R"({"type":"json_schema","json_schema":{"name":"catalogue","strict":true,"schema":{
  "type":"object","additionalProperties":false,"required":["store","products"],
  "properties":{"store":{"type":"string"},
    "products":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"object","additionalProperties":false,
      "required":["sku","name","price_eur","in_stock","tags"],
      "properties":{"sku":{"type":"string"},"name":{"type":"string"},"price_eur":{"type":"number"},"in_stock":{"type":"boolean"},
        "tags":{"type":"array","items":{"type":"string"},"maxItems":4}}}}}}}})";

constexpr size_t kReplyTailShown = 100;  // characters of a reply that did not parse, in the report

bool is_catalogue(const std::string &text) {
  xenithium::json::Value v;
  return xenithium::json::parse(text, v) && v.is_object() && v.get_array("products") != nullptr;
}

class Structured final : public Scenario {
 public:
  std::string_view id() const override { return "structured"; }
  std::string_view title() const override { return "JSON-schema constrained output"; }
  std::string_view about() const override { return "A reply constrained to a JSON schema: decode tok/s, TTFT, and whether the reply parses as the schema's object."; }
  std::span<const ParamSpec> params() const override { return kParams; }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    Row row = Row::with_labels({{"schema", "catalogue"}});
    for (int64_t pass = 0; pass < params.integer("passes"); ++pass) {
      ChatRequest request = session.sampled(std::string(kPrompt), static_cast<int>(params.integer("reply_tokens")));
      request.response_format = std::string(kResponseFormat);
      request.ignore_eos = false;  // the reply ends where the object does
      request.thinking = Thinking::off;
      const Sample sample = session.client().chat(request);
      if (!tally(result, sample)) continue;
      add_decode_metrics(row, sample);
      const bool valid = is_catalogue(sample.text);
      row.flag("valid JSON").samples.push_back(valid ? 1 : 0);
      if (!valid && row.notes.empty())
        row.notes.push_back("a reply did not parse (finish reason \"" + sample.finish_reason + "\", " + std::to_string(sample.completion_tokens) + " tokens); it ends: \"" +
                            sample.text.substr(sample.text.size() > kReplyTailShown ? sample.text.size() - kReplyTailShown : 0) + "\"");
    }
    result.rows.push_back(std::move(row));
    session.progress("structured: done");
    result.charts = {{ChartKind::dots, "Replies that parse as the schema's object", "schema", {"valid JSON"}, false}};
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_structured() { return std::make_unique<Structured>(); }

}  // namespace xenithium:bench
