"""The reference outputs of the chat templates of tests/templates/corpus.

A test tool of biv only: biv runs no Python at build time or at run time.

  python3 -I tools/template_fixtures.py extract <file.gguf> <out.jinja>
      Write the tokenizer.chat_template of a GGUF file, byte for byte.

  python3 -I tools/template_fixtures.py render
      Render each template of tests/templates/corpus on each case of
      tests/templates/conversations.json, as transformers renders a chat
      template (render_jinja_template), and write
      tests/templates/expected/<template>.json.

The render uses the jinja environment of transformers: trim_blocks,
lstrip_blocks, its tojson (no ASCII escapes) and its raise_exception. Its
strftime_now reads the clock, so this tool fixes the clock to FIXED_DATE.
Each case also renders once with StrictUndefined: a case whose output
needs an undefined value is marked, because biv must decide how it treats
one.
"""

import json
import struct
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests" / "templates" / "corpus"
CONVERSATIONS = ROOT / "tests" / "templates" / "conversations.json"
EXPECTED = ROOT / "tests" / "templates" / "expected"

# The date that strftime_now gives in every fixture
FIXED_DATE = datetime(2026, 1, 5)
# The versions that the fixtures were made with
JINJA2_VERSION = "3.1.6"
TRANSFORMERS_VERSION = "5.5.2"
# The llama.cpp commit of the templates of the corpus
LLAMA_CPP_COMMIT = "ad2775726"

GGUF_MAGIC = b"GGUF"
CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
# GGUF metadata value types
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def read_bytes(f):
    return f.read(read(f, "Q"))


def skip_value(f, kind):
    if kind == GGUF_STRING:
        read_bytes(f)
    elif kind == GGUF_ARRAY:
        item = read(f, "I")
        count = read(f, "Q")
        for _ in range(count):
            skip_value(f, item)
    elif kind in GGUF_SCALARS:
        f.read(struct.calcsize("<" + GGUF_SCALARS[kind]))
    else:
        raise SystemExit(f"GgufValueTypeUnknown: {kind}")


def extract(gguf, out):
    with open(gguf, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise SystemExit(f"NotGguf: {gguf}")
        read(f, "I")
        read(f, "Q")
        for _ in range(read(f, "Q")):
            key = read_bytes(f).decode("utf-8")
            kind = read(f, "I")
            if key == CHAT_TEMPLATE_KEY:
                if kind != GGUF_STRING:
                    raise SystemExit(f"ChatTemplateNotString: {kind}")
                text = read_bytes(f)
                text.decode("utf-8")  # strict: fails on bytes that are not UTF-8
                Path(out).write_bytes(text)
                return
            skip_value(f, kind)
    raise SystemExit(f"ChatTemplateMissing: {gguf}")


# The variables that biv sets in every render. biv sets reasoning_effort
# only when enable_thinking is true: the effort none has no level.
BIV_VARIABLES = {"bos_token", "eos_token", "enable_thinking", "preserve_thinking"}
EFFORT_VARIABLE = "reasoning_effort"
THINKING_VARIABLE = "enable_thinking"


def check_case(case):
    """Refuse a case whose variables are not those that biv sets"""
    names = set(case["variables"])
    expected = set(BIV_VARIABLES)
    if case["variables"].get(THINKING_VARIABLE) is True:
        expected.add(EFFORT_VARIABLE)
    if names != expected:
        raise SystemExit(f"CaseVariablesOther: {case['name']}: {sorted(names)}, expected {sorted(expected)}")


class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_DATE


def render_all():
    import jinja2
    import transformers
    import transformers.utils.chat_template_utils as chat

    if jinja2.__version__ != JINJA2_VERSION or transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(
            f"VersionOther: jinja2 {jinja2.__version__}, transformers {transformers.__version__}"
        )
    chat.datetime = FixedClock
    cases = json.loads(CONVERSATIONS.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        check_case(case)
    EXPECTED.mkdir(exist_ok=True)
    for path in sorted(CORPUS.glob("*.jinja")):
        template = path.read_text(encoding="utf-8")
        results = []
        for case in cases:
            results.append(render_case(chat, template, case))
        fixture = {
            "header": {
                "template": path.name,
                "jinja2": JINJA2_VERSION,
                "transformers": TRANSFORMERS_VERSION,
                "llama_cpp_commit": LLAMA_CPP_COMMIT,
                "fixed_date": FIXED_DATE.strftime("%Y-%m-%d"),
                "renderer": "transformers render_jinja_template",
            },
            "cases": results,
        }
        out = EXPECTED / (path.stem + ".json")
        out.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def render_case(chat, template, case):
    import jinja2

    args = dict(case["variables"])
    args["add_generation_prompt"] = case["add_generation_prompt"]
    tools = case.get("tools")
    result = {"case": case["name"]}
    try:
        result["output"] = chat.render_jinja_template(
            conversations=[case["messages"]], tools=tools, chat_template=template, **args
        )[0][0]
    # a template can also fail with an error of Python, such as a loop over
    # a value that is none: the fixture records each failure by its type
    except Exception as e:
        result["failure"] = type(e).__name__
        result["message"] = str(e)
        return result
    result["needs_undefined"] = needs_undefined(chat, template, case, args, tools)
    return result


def needs_undefined(chat, template, case, args, tools):
    """Whether the case renders under StrictUndefined"""
    import jinja2

    compiled = chat._cached_compile_jinja_template(template)
    strict = compiled.environment.overlay(undefined=jinja2.StrictUndefined)
    try:
        strict.from_string(template).render(
            messages=case["messages"], tools=tools, documents=None, **args
        )
    except jinja2.UndefinedError:
        return True
    return False


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "render":
        render_all()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
