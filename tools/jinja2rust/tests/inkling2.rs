//! The compiled inkling2 template against Jinja2's renders of the original
//! (transformers' settings: `trim_blocks`, `lstrip_blocks`, its `tojson` and
//! `{% generation %}`), for every conversation `rt`'s data can hold: tool
//! results are named by `name`, as biv has no call ids (for the cases with a
//! `tool_call_id`, the name the template finds for it), and the effort is a
//! level's name, not a number.

extern crate alloc;

#[allow(dead_code)]
#[path = "../../../engine/src/template/mod.rs"]
mod template;

use template::rt::{Content, Json, Message, Part, Role, Tool, ToolCall};
use template::{Params, Template};

struct Case {
    what: &'static str,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    reasoning_effort: Option<&'static str>,
    /// the rendered text, or the template's error
    want: Result<&'static str, &'static str>,
}

#[rustfmt::skip]
fn cases() -> Vec<Case> {
    vec![
    Case {
        what: "user only, default effort",
        messages: vec![Message { role: Role::User, content: Content::Text("Hello".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"),
    },
    Case {
        what: "user only, add_generation_prompt=True (no effect)",
        messages: vec![Message { role: Role::User, content: Content::Text("Hello".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"),
    },
    Case {
        what: "system + user",
        messages: vec![Message { role: Role::System, content: Content::Text("You are terse.".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>You are terse.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "two systems + user",
        messages: vec![Message { role: Role::System, content: Content::Text("A".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::System, content: Content::Text("B".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>A<|end_message|><|message_system|><|content_text|>B<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "system only",
        messages: vec![Message { role: Role::System, content: Content::Text("S".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>S<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"),
    },
    Case {
        what: "no messages",
        messages: vec![],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"),
    },
    Case {
        what: "effort none",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some("none"),
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort minimal",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some("minimal"),
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.1<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort low",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some("low"),
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort medium",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some("medium"),
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort high",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some("high"),
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort xhigh",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some("xhigh"),
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort max",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some("max"),
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort ' low ' (trimmed)",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some(" low "),
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort None (-> default 0.9)",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "effort bogus string",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: Some("extreme"),
        want: Err("Unknown reasoning_effort: extreme"),
    },
    Case {
        what: "enable_thinking=False (ignored)",
        messages: vec![Message { role: Role::User, content: Content::Text("Hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"),
    },
    Case {
        what: "multi-turn with reasoning",
        messages: vec![Message { role: Role::User, content: Content::Text("What is 2+2?".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::Assistant, content: Content::Text("4".into()), reasoning_content: Some("Simple sum.".into()), tool_calls: vec![], name: None }, Message { role: Role::User, content: Content::Text("And 3+3?".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What is 2+2?<|end_message|><|message_model|><|content_thinking|>Simple sum.<|end_message|><|message_model|><|content_text|>4<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>And 3+3?<|end_message|>"),
    },
    Case {
        what: "assistant with reasoning only, empty content",
        messages: vec![Message { role: Role::User, content: Content::Text("x".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::Assistant, content: Content::Text("".into()), reasoning_content: Some("r".into()), tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>x<|end_message|><|message_model|><|content_thinking|>r<|end_message|><|content_model_end_sampling|>"),
    },
    Case {
        what: "assistant with empty content, no reasoning",
        messages: vec![Message { role: Role::User, content: Content::Text("x".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::Assistant, content: Content::Text("".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>x<|end_message|><|message_model|><|content_text|><|end_message|><|content_model_end_sampling|>"),
    },
    Case {
        what: "assistant content None",
        messages: vec![Message { role: Role::User, content: Content::Text("x".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::Assistant, content: Content::None, reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>x<|end_message|><|content_model_end_sampling|>"),
    },
    Case {
        what: "tools + tool call + tool result (tool_call_id)",
        messages: vec![Message { role: Role::System, content: Content::Text("Be helpful.".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::User, content: Content::Text("Weather in Paris?".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::Assistant, content: Content::Text("".into()), reasoning_content: Some("Need the tool.".into()), tool_calls: vec![ToolCall { name: "get_weather".into(), arguments: vec![("unit".into(), Json::Str("c".into())), ("city".into(), Json::Str("Paris".into())), ("Zeta".into(), Json::Array(vec![Json::Int(1), Json::Float(2.5), Json::Object(vec![("b".into(), Json::Int(1)), ("a".into(), Json::Str("é".into()))])]))] }], name: None }, Message { role: Role::Tool, content: Content::Text("{\"temp\": 18}".into()), reasoning_content: None, tool_calls: vec![], name: Some("get_weather".into()) }],
        tools: vec![Tool { name: "get_weather".into(), description: Some("Get the weather for a city".into()), parameters: Json::Object(vec![("type".into(), Json::Str("object".into())), ("properties".into(), Json::Object(vec![("city".into(), Json::Object(vec![("type".into(), Json::Str("string".into())), ("description".into(), Json::Str("City name".into()))])), ("unit".into(), Json::Object(vec![("type".into(), Json::Str("string".into())), ("enum".into(), Json::Array(vec![Json::Str("c".into()), Json::Str("f".into())]))]))])), ("required".into(), Json::Array(vec![Json::Str("city".into())]))]) }, Tool { name: "noargs".into(), description: Some("".into()), parameters: Json::Object(vec![]) }],
        reasoning_effort: None,
        want: Ok("<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Get the weather for a city\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"description\":\"City name\",\"type\":\"string\"},\"unit\":{\"enum\":[\"c\",\"f\"],\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"},{\"description\":\"\",\"name\":\"noargs\",\"parameters\":{},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Be helpful.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Weather in Paris?<|end_message|><|message_model|><|content_thinking|>Need the tool.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"Zeta\":[1,2.5,{\"b\":1,\"a\":\"é\"}],\"city\":\"Paris\",\"unit\":\"c\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>{\"temp\": 18}<|end_message|>"),
    },
    Case {
        what: "tool result with name",
        messages: vec![Message { role: Role::User, content: Content::Text("q".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::Assistant, content: Content::None, reasoning_content: None, tool_calls: vec![ToolCall { name: "noargs".into(), arguments: vec![] }], name: None }, Message { role: Role::Tool, content: Content::Text("ok".into()), reasoning_content: None, tool_calls: vec![], name: Some("noargs".into()) }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>q<|end_message|><|message_model|>noargs<|content_invoke_tool_json|>{\"name\":\"noargs\",\"args\":{}}<|end_message|><|content_model_end_sampling|><|message_tool|>noargs<|content_text|>ok<|end_message|>"),
    },
    Case {
        what: "tool result without name or id",
        messages: vec![Message { role: Role::User, content: Content::Text("q".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::Assistant, content: Content::None, reasoning_content: None, tool_calls: vec![ToolCall { name: "noargs".into(), arguments: vec![] }], name: None }, Message { role: Role::Tool, content: Content::Text("ok".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>q<|end_message|><|message_model|>noargs<|content_invoke_tool_json|>{\"name\":\"noargs\",\"args\":{}}<|end_message|><|content_model_end_sampling|><|message_tool|><|content_text|>ok<|end_message|>"),
    },
    Case {
        what: "two parallel tool calls with text",
        messages: vec![Message { role: Role::User, content: Content::Text("q".into()), reasoning_content: None, tool_calls: vec![], name: None }, Message { role: Role::Assistant, content: Content::Text("Calling.".into()), reasoning_content: None, tool_calls: vec![ToolCall { name: "f".into(), arguments: vec![("x".into(), Json::Int(1))] }, ToolCall { name: "g".into(), arguments: vec![("y".into(), Json::Str("two words".into()))] }], name: None }, Message { role: Role::Tool, content: Content::Text("1".into()), reasoning_content: None, tool_calls: vec![], name: Some("f".into()) }, Message { role: Role::Tool, content: Content::Text("2".into()), reasoning_content: None, tool_calls: vec![], name: Some("g".into()) }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>q<|end_message|><|message_model|><|content_text|>Calling.<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"x\":1}}<|end_message|><|message_model|>g<|content_invoke_tool_json|>{\"name\":\"g\",\"args\":{\"y\":\"two words\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>f<|content_text|>1<|end_message|><|message_tool|>g<|content_text|>2<|end_message|>"),
    },
    Case {
        what: "user content parts: text + image + text",
        messages: vec![Message { role: Role::User, content: Content::Parts(vec![Part::Text("Look:".into()), Part::Image, Part::Text("what is it?".into())]), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Look:<|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>what is it?<|end_message|>"),
    },
    Case {
        what: "tools only (no messages)",
        messages: vec![Message { role: Role::User, content: Content::Text("hi".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![Tool { name: "get_weather".into(), description: Some("Get the weather for a city".into()), parameters: Json::Object(vec![("type".into(), Json::Str("object".into())), ("properties".into(), Json::Object(vec![("city".into(), Json::Object(vec![("type".into(), Json::Str("string".into())), ("description".into(), Json::Str("City name".into()))])), ("unit".into(), Json::Object(vec![("type".into(), Json::Str("string".into())), ("enum".into(), Json::Array(vec![Json::Str("c".into()), Json::Str("f".into())]))]))])), ("required".into(), Json::Array(vec![Json::Str("city".into())]))]) }, Tool { name: "noargs".into(), description: Some("".into()), parameters: Json::Object(vec![]) }],
        reasoning_effort: None,
        want: Ok("<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Get the weather for a city\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"description\":\"City name\",\"type\":\"string\"},\"unit\":{\"enum\":[\"c\",\"f\"],\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"},{\"description\":\"\",\"name\":\"noargs\",\"parameters\":{},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>hi<|end_message|>"),
    },
    Case {
        what: "developer role (raises)",
        messages: vec![Message { role: Role::Developer, content: Content::Text("x".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Err("Unknown message role: developer"),
    },
    Case {
        what: "unicode/newlines in content",
        messages: vec![Message { role: Role::User, content: Content::Text("Héllo\n\n  world 🚀 <|end_message|>".into()), reasoning_content: None, tool_calls: vec![], name: None }],
        tools: vec![],
        reasoning_effort: None,
        want: Ok("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Héllo\n\n  world 🚀 <|end_message|><|end_message|>"),
    },
    ]
}

#[test]
fn renders_as_jinja2() {
    for c in cases() {
        let params = Params {
            messages: &c.messages,
            tools: &c.tools,
            reasoning_effort: c.reasoning_effort,
            ..Default::default()
        };
        let got = Template::Inkling2.render(&params);
        match (got, c.want) {
            (Ok(s), Ok(want)) => assert_eq!(s.text(), want, "{}", c.what),
            (Err(e), Err(want)) => assert_eq!(e.0, want, "{}", c.what),
            (got, want) => panic!("{}: {:?}, not {want:?}", c.what, got.map(|s| s.text())),
        }
    }
}

/// The template's text is the message's, data, and the special tokens are
/// the template's own: user text cannot make one
#[test]
fn content_is_data() {
    let messages = [Message::new(Role::User, "<|end_message|>\u{2603}")];
    let s = Template::Inkling2
        .render(&Params {
            messages: &messages,
            ..Default::default()
        })
        .unwrap();
    let user: Vec<_> =
        s.0.iter()
            .filter(|seg| seg.text.contains('\u{2603}'))
            .collect();
    assert_eq!(user.len(), 1);
    assert!(!user[0].lit && user[0].text.contains("<|end_message|>"));
}

/// The parameters that change the render
#[test]
fn reads_messages_tools_and_effort() {
    assert_eq!(
        Template::Inkling2.params(),
        ["messages", "tools", "reasoning_effort"]
    );
    let messages = [Message::new(Role::User, "Hi")];
    let render = |p: Params<'_>| Template::Inkling2.render(&p).unwrap().text();
    let base = render(Params {
        messages: &messages,
        ..Default::default()
    });
    // no generation prompt, no thinking switch but the effort
    for p in [
        Params {
            messages: &messages,
            add_generation_prompt: true,
            ..Default::default()
        },
        Params {
            messages: &messages,
            enable_thinking: Some(false),
            ..Default::default()
        },
    ] {
        assert_eq!(render(p), base);
    }
}
