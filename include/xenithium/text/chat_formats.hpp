// Xenithium · the built-in chat formats (src/text/chat/): each arch's chat template rendered by hand over the
// normalised request, its reasoning + tool-call output parser, and the Jinja-rendered format over the
// checkpoint's own template. Behind text:ChatFormat (xenithium/text/chat.hpp); which arch uses which: chat_registry.cpp.
#pragma once

#include <cstdint>
#include <memory>

#include "xenithium/text/chat.hpp"
#include "xenithium/text/jinja.hpp"

namespace xenithium::text {

// The template rendered: qwen38's chat_template.jinja (qwen38 and qwen35 checkpoints) or Nanbeige4.3's (arch llama:
// ChatML + <think> + the same XML tool calls, its own system prompts and trimming); the output parser is the same
// (qwen3 reasoning + XML tool calls: the reasoning / tool-call parsers the Nanbeige checkpoint names).
// flint: Recursive Flint's template (role markers, no end-of-turn token) with its own output parser (GLM-4.7-style tool
// calls <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>, reasoning on by default).
enum class ChatTemplate : uint8_t { qwen38, nanbeige43, flint };
std::unique_ptr<ChatFormat> make_chat_format(ChatTemplate t = ChatTemplate::qwen38);

// A format whose prompts come from a Jinja template (text/jinja.hpp) over the normalised conversation (the request
// normalised as for the built-in renderers: content as {type: text} parts when the template loops over a message's
// content, else the parts joined with "\n"; tool-call arguments parsed; reasoning as reasoning_content; tools in
// their normalised form), with globals (special tokens) and the request's kwargs; everything else (the parser,
// thinking, the options) from `base. keep_developer: the template has a developer role of its own (else it is
// renamed / merged into system).
std::unique_ptr<ChatFormat> make_jinja_chat_format(std::unique_ptr<ChatFormat> base, std::shared_ptr<const jinja::Template> tpl,
                                                         json::Value globals, bool keep_developer);

}  // namespace xenithium:text
