// The launcher's model of server.env: the settings catalog (scripts/start-server.sh defaults +
// docs/server-env.md text + the launcher's types), env files, validation and the model scan.
// Host only; the screens are in app.cpp.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xenithium::launch {

enum class Kind { Int, Float, Choice, Bool, Path, Text };
// MoE: the routed-expert archs with a hot/cold split (qwen38); Vision: the archs whose checkpoints carry a vision tower
// (qwen38, qwen35).
enum class Scope { All, Qwen38, Qwen35, MoE, Vision };

struct Spec {
  std::string key, label, group;
  Kind kind = Kind::Text;
  Scope scope = Scope::All;
  // Choice: the closed list. Other kinds: suggestions shown in a popup (custom values allowed).
  std::vector<std::string> choices;
  // Display names of special values ("0" -> "unlimited").
  std::vector<std::pair<std::string, std::string>> names;
  std::string unit;
  double lo = -1e300, hi = 1e300;
  int64_t multiple = 0;  // Int: the value must be a multiple of this (0 = any)
  bool allow_empty = false;
  bool allow_auto = false;  // Int: the word "auto" is also valid
  bool path_file = false;  // Path: a file (else a directory)
  bool path_auto = false;  // Path: the word "auto" is also valid (the model's own file, resolved by the script)
  bool diag = false;  // read by the server from its environment (a two-column table of docs/server-env.md)
  bool known = true;  // in the launcher's schema (else a plain text setting under "Other")
  bool in_script = false, in_docs = false;
  std::string def, flag, help;  // help = the setting's row of docs/server-env.md (shown only for settings outside the schema)
  std::string brief;  // the laconic description the TUI shows: what it does, the effect, the cost (settings.cpp)
};

struct Catalog {
  std::vector<Spec> specs;
  std::vector<std::string> groups;  // display order
  std::vector<std::string> warnings;  // schema vs script vs docs mismatches
  const Spec *find(std::string_view key) const;
};

struct ScriptVar {
  std::string key, def;
};
struct DocRow {
  std::string key, def, flag, meaning;
  bool diag = false;
};

std::vector<ScriptVar> parse_script_defaults(std::string_view start_server_sh);
std::vector<DocRow> parse_docs(std::string_view server_env_md);
Catalog build_catalog(std::string_view start_server_sh, std::string_view server_env_md);
// Reads <root>/scripts/start-server.sh and <root>/docs/server-env.md.
Catalog load_catalog(const std::string &root);

using Values = std::map<std::string, std::string>;
Values defaults_of(const Catalog &cat);  // every key -> its default ($HOME expanded)

// server.env-style file: KEY=value lines (bash-sourced), comments and unknown lines kept.
class EnvFile {
 public:
  static EnvFile parse(std::string_view text);
  Values values() const;  // unquoted, $HOME / ~ expanded
  bool has(std::string_view key) const;
  void set(const std::string &key, const std::string &value);  // in place, else appended
  void erase(std::string_view key);
  std::string text() const;
  // Keeps the keys whose value differs from the default, drops the ones equal to it (keep_present: a key the file already
  // has stays, at its new value - the per-model files list their settings explicitly). Unchanged lines stay as written.
  void apply(const Values &vals, const Values &defaults, const Catalog &cat, bool keep_present = false);

 private:
  std::vector<std::string> lines_;
};

std::string home_dir();
std::string expand_home(std::string_view s);
std::string tilde(std::string_view path);  // $HOME/x -> ~/x
std::string env_quote(std::string_view v);
std::string env_unquote(std::string_view raw);

bool applies(const Spec &s, std::string_view arch);  // arch "" (unknown) -> everything applies
// "" when the server reads the setting with these values; else why not (other arch, a switch it needs
// is off, the stub engine). The screens hide settings with a reason; arch "" -> nothing is unused.
std::string unused_reason(const Spec &s, std::string_view arch, const Values &vals);
std::string scope_name(Scope s);
// One plain-language sentence on what a settings group is about (the help panel of a group row).
std::string group_brief(std::string_view group);

struct Issue {
  std::string key, msg;
  bool error = true;  // else a warning
};
// max_context: the longest XEN_CONTEXT (the max context, one request's most tokens) the model takes (0 = no limit):
// the attention kernels' 1,048,576 for qwen35 / llama / flint / qwen38.
// max_hot: the model's routed experts over all MoE layers, the cap of XEN_HOT (0 = no model / no cap).
// trained: the checkpoint's max_position_embeddings for the archs that scale past it (0 = none): a max context above it
// gets a warning naming the YaRN factor the load will apply (yarn_note).
std::vector<Issue> validate(const Catalog &cat, const Values &vals, std::string_view arch, const std::string &root, int64_t max_context = 0, int64_t max_hot = 0,
                            int64_t trained = 0);
// One value of one spec; "" = fine. arch narrows XEN_SLOTS, max_context XEN_CONTEXT.
std::string check_value(const Spec &s, const std::string &v, std::string_view arch, const std::string &root, bool *warn_only = nullptr,
                        int64_t max_context = 0);
// The longest max context a model takes: 1,048,576 (kQsaMaxTokens) for qwen35 / llama / flint / qwen38, 0 (no limit) otherwise.
int64_t max_context_of(const struct ModelInfo &m);
// The trained positions past which the load scales the rotary (qwen35 / llama / flint / qwen38: max_position_embeddings), 0 otherwise.
int64_t trained_positions_of(const struct ModelInfo &m);
// The YaRN factor the load applies for a max context (src/kernels/attention/qsa_ref.cpp qsa_yarn_for): 1 up to
// `trained, past it the smallest factor in steps of 0.01 reaching the context.
double yarn_factor_for(int64_t context, int64_t trained);
// What a max context past the trained positions does, for the warning and the help panel ("" at or below them).
std::string yarn_note(std::string_view arch, int64_t context, int64_t trained);
// XEN_HOT counts the routed experts held in VRAM over ALL MoE layers (TP2: every hot expert is split over both cards,
// half of its bytes on each). The cap = moe_layers x experts (qwen38 48 x 512); 0 for the other
// archs (no hot/cold split there).
int64_t max_hot_of(const struct ModelInfo &m);
// The card's usable VRAM in GiB, from the PCI device id in sysfs; -1 when no known card.
double card_vram_gib();
// Bytes of one expert blob on a card: 3 matrices (gate/up/down) x hidden x moe_inter at the model's
// bits, split across 2 cards (TP2); 0 when the geometry is unknown.
double bytes_per_expert_card(const struct ModelInfo &m);
// A rough hot-expert fit (total, the units of XEN_HOT) of an empty card, for when no memory plan is available.
int64_t hot_fit(const struct ModelInfo &m);

// The server's memory plan (xenithium-server --plan through scripts/start-server.sh SERVE_PLAN=1: the load's own
// arithmetic, no GPU). VRAM figures per card, GiB.
struct MemPlanView {
  bool ok = false;
  std::string arch;
  bool exact = true;  // false = an estimate (qwen38 measures its free VRAM at load)
  double card_gib = 0, vram_used_gib = 0, weights_gib = 0, kv_gib = 0, state_gib = 0, scratch_gib = 0, hot_gib = 0;
  int64_t hot_experts = 0, experts_total = 0;
  int moe_layers = 0;
  int cards = 0;  // the cards the model runs on (0 = not in the plan)
  int64_t kv_tokens = 0, context = 0;
  int slots = 0;
  double cold_gib = 0, embed_gib = 0, pinned_gib = 0, prefix_gib = 0;  // host (pinned), both cards
  double host_total_gib = 0, host_available_gib = 0;
  double host_other_gib = 0, host_keep_gib = 0;  // the process's RAM besides the pinned set; RAM left to the system
  // Host RAM the pinned set may use: free now minus the process's other RAM and the system's share.
  double host_room_gib() const { return host_available_gib - host_other_gib - host_keep_gib; }
};
bool parse_plan(const std::string &json_text, MemPlanView &p);
// What the plan says will not work: pinned host memory past the free RAM (error, unless a running server's memory is
// counted in it), a prefix cache the server will cut, a pinned hot count past the card's VRAM (warnings).
std::vector<struct Issue> plan_issues(const MemPlanView &p, const Values &vals, bool server_running);
std::string fmt_gib(double gib);  // "31.9", "120"

// ---- the KV context before a launch ----
// The settings the memory plan reads: their help panel shows the plan and the KV pool each of their values would give.
bool moves_memory(std::string_view key);
// The values that table computes for a setting: on / off, the choices, a number's suggestions with the current value
// among them (numbers ascending), a drafter path's candidates (`extra`) after "" (none). At most 8; empty = no table.
std::vector<std::string> whatif_values(const Spec &s, const std::string &current, const std::vector<std::string> &extra = {});
// A value as the table names it: on / off, (none) for an empty path, a path's folder name, else the value.
std::string whatif_name(const Spec &s, const std::string &v);
// What a pool holds, in words: "2 requests at the full 524,288-token max context, or all 16 slots at 69,500 tokens each".
std::string kv_capacity_text(int64_t kv_tokens, int64_t context, int slots);
// "+29 %", "-50 %", "same": a value's pool against the current one (0 tokens on either side = "").
std::string kv_delta_text(int64_t tokens, int64_t now);

// The dynamic XEN_HOT description: the model's expert count, how it splits, and how many fit (the plan when given).
std::string hot_description(const struct ModelInfo &m, const MemPlanView *plan = nullptr);

// Ffmpeg programs found on this host (PATH, the usual prefixes, static builds under ~/.hermes/tools/*/bin),
// each with an ffprobe beside it or on PATH; the video decoder setting offers them.
std::vector<std::string> ffmpeg_candidates();

// ---- MoE expert files (docs/server-env.md, "MoE expert files") ----
// The hot-expert ranking (XEN_HOT_RANK recorded by the server while it serves), per model in
// <experts_root>/<model folder>/; `auto` resolves there (scripts/start-server.sh does the same).
bool is_moe_arch(std::string_view arch);  // qwen38: a hot / cold expert split
std::string experts_dir_of(const std::string &experts_root, const struct ModelInfo &m);
struct ExpertFile {
  std::string path;  // resolved; "" = none (an empty setting)
  bool is_auto = false;  // the setting's value is auto
  bool exists = false;  // the file is there
  bool shape_ok = false;  // an XRC1 file of the model's layers x experts
  int layers = 0, experts = 0;
  uint64_t bytes = 0;
  int64_t mtime = 0;  // seconds since the epoch
  std::string error;  // why it cannot be used (exists but not readable / another shape)
};
// value = the setting (auto / a path / empty); file = the auto file's name ("hot-rank.xrc").
ExpertFile expert_file(const std::string &value, const std::string &file, const std::string &root, const std::string &experts_root, const struct ModelInfo &m);
// Share of a ranking's own counts that sit on the hot set with `hot` hot experts (the units of XEN_HOT: a total over all
// layers), for the ranked placement and for the default one (qwen38 the global top pairs vs the lowest ids per layer,
// hot / layers each). false when unreadable.
bool rank_coverage(const std::string &path, const struct ModelInfo &m, int64_t hot, double &ranked, double &fallback);
std::string age_text(int64_t mtime);  // "3 min ago", "2 days ago"

std::string display_value(const Spec &s, const std::string &v);
std::string with_commas(int64_t v);
std::string human_bytes(uint64_t b);

struct ModelInfo {
  std::string path, name;
  std::string arch, arch_desc, model_type;  // arch "" = not served by this engine
  bool has_config = false, drafter = false, dflash = false;  // drafter: an MTP checkpoint; dflash: config names a block drafter (DFlash 2 or DSpine)
  uint64_t bytes = 0;
  int64_t max_positions = 0;  // max_position_embeddings (text_config first), 0 = not in config.json
  int64_t experts = 0, layers = 0;  // routed experts (per layer) and layers (text_config first), 0 = not in config.json
  int64_t moe_layers = 0;  // layers with routed experts, 0 = not a MoE
  int64_t hidden = 0, moe_inter = 0;  // hidden_size and moe_intermediate_size (text_config first), 0 = not in config.json
  double bits = 0;  // expert weight precision in bits (quantization_config), 0 = not found
  std::string error;
};
ModelInfo probe_model(const std::string &dir);
std::vector<ModelInfo> scan_models(const std::string &dir);
// MTP drafter directories that fit a qwen38 target: <target>/runtime/mtp-int4-g32 and the sibling
// directories named <target>-mtp*.
std::vector<std::string> drafters_for(const std::string &target);
// Block drafter checkpoints under a models directory: the directories whose config.json names DFlash 2 or DSpine
// (arch/common/drafter_archs.hpp).
std::vector<ModelInfo> dflash_drafters(const std::string &dir);

// A trained page selection bank (XEN_TRAINED_QSA_DELTA_DIR): a folder without config.json whose safetensors hold the
// indexer tensors "...layers.<n>.self_attn._qsa_indexer.*" of one or more attention layers.
struct QsaBank {
  std::string path;
  uint64_t bytes = 0;
  std::vector<int> layers;  // ascending
};
bool probe_qsa_bank(const std::string &dir, QsaBank &out);
// The banks in these folders' subfolders (each folder once, by path), sorted by path.
std::vector<QsaBank> qsa_banks(const std::vector<std::string> &dirs);
// "16 layers (3, 7 ... 63), 200.0 MiB"
std::string qsa_bank_detail(const QsaBank &b);

bool read_text(const std::string &path, std::string &out);
bool write_text_atomic(const std::string &path, const std::string &text, std::string *err);
bool is_dir(const std::string &p);
bool is_file(const std::string &p);
std::string join_path(const std::string &a, const std::string &b);

// ---- per-model settings files (docs/server-env.md, "Per-model settings files") ----
// One checkpoint's settings live in <id>.env (its XEN_MODEL line names it). The shipped files are the presets in
// <root>/models/presets/ (tracked, never written by the tools); the user's own files are <root>/models/<id>.env
// (git-ignored, written by the launcher), and a user file replaces the preset of the same name. server.env points at
// the chosen one (XEN_MODEL_ENV=<id>); scripts/start-server.sh applies it under the settings file and the environment.
// dir/<name>.env -> dir/presets/<name>.env.
std::string preset_env_path(const std::string &path);
// Where a save of `path` goes: dir/presets/<name>.env -> dir/<name>.env; any other path as it is.
std::string user_env_path(const std::string &path);
// The file read for a user path: the user's file when it exists, else its preset when that exists, else the path.
std::string readable_env_path(const std::string &user_path);
bool same_path(const std::string &a, const std::string &b);  // the same file / directory (realpath; else string equality)
std::string model_env_model(const std::string &env_path);  // the file's XEN_MODEL ($HOME expanded); "" = none / unreadable
std::vector<std::string> model_env_files(const std::string &dir);  // dir/*.env, then dir/presets/*.env without a user file of that name; each sorted
std::string find_model_env(const std::string &dir, const std::string &model_path);  // the file for this checkpoint; "" = none
std::string model_env_id(const std::string &env_path);  // models/qwen38-flash-next.env -> qwen38-flash-next
// A settings file that starts a model with its own settings (models/<id>.env): one named server.env, or one holding
// XEN_MODEL_ENV. Any other file (a profile) is a whole settings file.
bool is_pointer_env(const std::string &path);
// The file a checkpoint without one gets: dir/<checkpoint folder>.env (a free name).
std::string model_env_new_path(const std::string &dir, const std::string &model_path);
// ---- per-setup settings files ----
// A model's file models/<id>.env holds the shared settings (the "Model & network" group and the setup key); each setup
// (card count) keeps its own fastest settings in models/<id>.tp<N>.env, over the model's file. scripts/start-server.sh
// layers them the same way: defaults < model file < setup file < settings file / environment.
// The key that picks a model's setup: XEN_Q38_TP (qwen38), XEN_TP (qwen35 / llama / flint), "" (no model).
std::string setup_key_of(std::string_view arch);
// The settings that stay in the model's file whatever the setup: the "Model & network" group and the setup keys.
bool is_shared_setting(const Spec &s);
// models/<id>.env + setup value N -> models/<id>.tp<N>.env ("" when either is empty).
std::string setup_env_path(const std::string &model_env, const std::string &setup_value);
// A setup file's name (<id>.tp<N>.env): it names no checkpoint and is never a model's file.
bool is_setup_env(const std::string &path);
// XEN_MODEL_ENV's value -> a path (an id = dir/<id>.env; a relative path is under root); "" for empty / auto.
std::string resolve_model_env(const std::string &root, const std::string &dir, const std::string &ref);
// server.env as scripts/start-server.sh reads it: the model it starts and that model's settings file. The file a pointer
// names is used only when it belongs to the model (an XEN_MODEL in server.env that differs wins: auto lookup).
struct ModelEnvView {
  std::string pointer;  // server.env's XEN_MODEL_ENV (raw; "" = not set)
  bool pointer_set = false;  // the key is in server.env (empty = no per-model file)
  std::string model;  // the checkpoint: server.env's XEN_MODEL, else the pointed file's, else the default
  std::string file;  // its per-model file, the user's path (may not exist yet: model_env_new_path); "" = none (pointer set empty)
  std::string file_source;  // the file its settings were read from: `file`, or its preset; "" = none exists
  bool file_exists = false;  // a file was read (the user's or the preset)
  Values file_values;  // the catalog keys of the per-model file
  Values env_values;  // the catalog keys of server.env itself (they win over the per-model file)
  std::string setup_file;  // the setup's file, the user's path (models/<id>.tp<N>.env; may not exist), "" = no setups
  Values setup_values;  // its catalog keys other than the shared ones (they win over the model's file)
};
ModelEnvView read_model_env(const struct Catalog &cat, const std::string &root, const std::string &dir, const std::string &env_path);
// Completions of a path being typed: the longest common extension and the candidates.
std::vector<std::string> complete_path(const std::string &typed, const std::string &root, std::string &common);

}  // namespace xenithium:launch
