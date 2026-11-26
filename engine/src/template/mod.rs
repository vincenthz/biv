//! The chat templates compiled by jinja2rust: do not edit.
//!
//! [`Template::recognize`] finds a model's chat template by the hash of
//! its text, and [`Template::render`] renders it from [`Params`], the
//! parameters of all the templates: each reads those it has.

pub mod rt;

pub mod flint;
pub mod inkling2;
pub mod qwen35;
pub mod qwen36;
pub mod qwen38;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    Flint,
    Inkling2,
    Qwen35,
    Qwen36,
    Qwen38,
}

/// The parameters of the templates: each reads those it has
#[derive(Clone, Copy, Debug, Default)]
pub struct Params<'a> {
    /// the conversation (all)
    pub messages: &'a [rt::Message],
    /// the functions the model may call (all)
    pub tools: &'a [rt::Tool],
    /// `add_generation_prompt` (flint, qwen35, qwen36, qwen38)
    pub add_generation_prompt: bool,
    /// `add_vision_id` (qwen35, qwen36, qwen38)
    pub add_vision_id: bool,
    /// `enable_thinking`: `None` leaves it undefined (qwen35, qwen36, qwen38)
    pub enable_thinking: Option<bool>,
    /// `preserve_thinking`: `None` leaves it undefined (qwen36, qwen38)
    pub preserve_thinking: Option<bool>,
    /// `reasoning_effort`: `None` leaves it undefined (inkling2, qwen38)
    pub reasoning_effort: Option<&'a str>,
}

impl Template {
    pub const ALL: [Template; 5] = [
        Template::Flint,
        Template::Inkling2,
        Template::Qwen35,
        Template::Qwen36,
        Template::Qwen38,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Template::Flint => "flint",
            Template::Inkling2 => "inkling2",
            Template::Qwen35 => "qwen35",
            Template::Qwen36 => "qwen36",
            Template::Qwen38 => "qwen38",
        }
    }

    /// FNV-1a of the template's text
    pub fn hash(self) -> u64 {
        match self {
            Template::Flint => flint::TEMPLATE_HASH,
            Template::Inkling2 => inkling2::TEMPLATE_HASH,
            Template::Qwen35 => qwen35::TEMPLATE_HASH,
            Template::Qwen36 => qwen36::TEMPLATE_HASH,
            Template::Qwen38 => qwen38::TEMPLATE_HASH,
        }
    }

    /// The compiled template whose text is `source`, if any
    pub fn recognize(source: &str) -> Option<Template> {
        let hash = rt::fnv1a(source);
        Self::ALL.into_iter().find(|t| t.hash() == hash)
    }

    /// The parameters the template reads, the others of [`Params`] ignored
    pub fn params(self) -> &'static [&'static str] {
        match self {
            Template::Flint => &["messages", "tools", "add_generation_prompt"],
            Template::Inkling2 => &["messages", "tools", "reasoning_effort"],
            Template::Qwen35 => &[
                "messages",
                "tools",
                "add_generation_prompt",
                "add_vision_id",
                "enable_thinking",
            ],
            Template::Qwen36 => &[
                "messages",
                "tools",
                "add_generation_prompt",
                "add_vision_id",
                "enable_thinking",
                "preserve_thinking",
            ],
            Template::Qwen38 => &[
                "messages",
                "tools",
                "add_generation_prompt",
                "add_vision_id",
                "enable_thinking",
                "preserve_thinking",
                "reasoning_effort",
            ],
        }
    }

    pub fn render(self, p: &Params<'_>) -> rt::R<rt::Str> {
        match self {
            Template::Flint => flint::render(&flint::Params {
                messages: p.messages,
                tools: p.tools,
                add_generation_prompt: p.add_generation_prompt,
            }),
            Template::Inkling2 => inkling2::render(&inkling2::Params {
                messages: p.messages,
                tools: p.tools,
                reasoning_effort: p.reasoning_effort,
            }),
            Template::Qwen35 => qwen35::render(&qwen35::Params {
                messages: p.messages,
                tools: p.tools,
                add_generation_prompt: p.add_generation_prompt,
                add_vision_id: p.add_vision_id,
                enable_thinking: p.enable_thinking,
            }),
            Template::Qwen36 => qwen36::render(&qwen36::Params {
                messages: p.messages,
                tools: p.tools,
                add_generation_prompt: p.add_generation_prompt,
                add_vision_id: p.add_vision_id,
                enable_thinking: p.enable_thinking,
                preserve_thinking: p.preserve_thinking,
            }),
            Template::Qwen38 => qwen38::render(&qwen38::Params {
                messages: p.messages,
                tools: p.tools,
                add_generation_prompt: p.add_generation_prompt,
                add_vision_id: p.add_vision_id,
                enable_thinking: p.enable_thinking,
                preserve_thinking: p.preserve_thinking,
                reasoning_effort: p.reasoning_effort,
            }),
        }
    }
}
