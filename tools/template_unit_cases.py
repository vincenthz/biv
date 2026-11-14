"""The reference outputs of the unit cases of the template subset of biv.

A test tool of biv only: biv runs no Python at build time or at run time.

  python3 -I tools/template_unit_cases.py

Each case is a small template that checks one construct or one failure.
The tool renders each with the renderer of transformers (the settings of
tools/template_fixtures.py), with VARS and the clock fixed to FIXED_DATE,
and writes tests/templates/unit_cases.json, which
engine/src/template/tests.rs reads.
"""

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "templates" / "unit_cases.json"
FIXED_DATE = datetime(2026, 1, 5)
JINJA2_VERSION = "3.1.6"
TRANSFORMERS_VERSION = "5.5.2"

VARS = {"d": {"b": 2, "a": 1, "C": 3}, "items": [{"name": "x", "type": "f", "n": 3}, {"name": "y", "type": "g", "n": 1}],
        "s": "  Hello, World  ", "n": None, "words": ["b", "A", "c"], "t": True}
CASES = [
 ("if", "{% if 1 > 2 %}a{% elif t %}b{% else %}c{% endif %}"),
 ("loop_vars", "{% for x in words %}{{ loop.index }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }}|{% endfor %}"),
 ("loop_prev_next", "{% for x in words %}[{{ loop.previtem }}-{{ loop.nextitem }}]{% endfor %}"),
 ("for_filter", "{% for x in [1,2,3,4] if x % 2 == 0 %}{{ x }}{% endfor %}"),
 ("for_items", "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}"),
 ("break_continue", "{% for x in [1,2,3,4,5] %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}{% break %}{% endif %}{{ x }}{% endfor %}"),
 ("loop_scope", "{% set x = 0 %}{% for i in [1,2] %}{% set x = i %}{{ x }}{% endfor %}{{ x }}"),
 ("namespace", "{% set ns = namespace(c=0) %}{% for i in [1,2,3] %}{% set ns.c = ns.c + i %}{% endfor %}{{ ns.c }}"),
 ("set_tuple", "{% set a, b = [1, 2] %}{{ b }}{{ a }}"),
 ("set_block", "{% set x %}in {{ 1 + 1 }}{% endset %}[{{ x }}]"),
 ("macro", "{% macro m(a, b='B') %}{{ a }}-{{ b }}-{{ t }}{% endmacro %}{{ m(1) }} {{ m(2, b='c') }}"),
 ("whitespace", "a\n  {% if true %}\n  b {{- ' c' -}}  \n  {% endif %}\nd\n"),
 ("comment", "a {# note #}\nb {#- x -#} c"),
 ("escapes", "{{ 'a\\tb\\n\\x41\\u00e9\\\\' }}"),
 ("slices", "{{ words[::-1] }}{{ s[2:7] }}{{ words[-1] }}{{ words[1:] }}"),
 ("compare_chain", "{{ 1 < 2 < 3 }}{{ 3 > 2 > 2 }}{{ 'a' in 'cat' }}{{ 4 not in [1,2] }}"),
 ("and_or", "{{ 0 or 'x' }}{{ 'y' and 'z' }}{{ n or 'none' }}"),
 ("cond_expr", "{{ 'a' if t else 'b' }}{{ 'c' if n }}."),
 ("arith", "{{ -7 % 3 }} {{ 7 / 2 }} {{ 1 + 2.5 }} {{ 'ab' * 2 }}"),
 ("concat", "{{ 1 ~ 'a' ~ n ~ t }}"),
 ("output_forms", "{{ 1.0 }} {{ 1e20 }} {{ 0.1 + 0.2 }} {{ [1, 'a', none, true] }} {{ {'k': 1} }} {{ (1,) }} {{ none }}"),
 ("default", "{{ n | default('dn') }}{{ n | default('db', true) }}{{ d.zz | default('dz') }}"),
 ("length_string", "{{ words | length }}{{ d | count }}{{ 12 | string }}{{ s | length }}"),
 ("trim", "[{{ s | trim }}][{{ 'xxaxx' | trim('x') }}]"),
 ("items_filter", "{% for k, v in d | items %}{{ k }}{{ v }}{% endfor %}"),
 ("safe_markup", "{{ 'a'|safe + '<\"&\\'>' }}{{ '<b>' + '<i>'|safe }}"),
 ("markup_concat", "{{ '<'|safe ~ '>' }}{{ ('<'|safe ~ '&') + '<' }}"),
 ("join", "{{ words | join(', ') }}|{{ items | join('-', attribute='name') }}"),
 ("list_first_last", "{{ 'abc' | list }}"),
 ("select", "{{ items | selectattr('type', 'equalto', 'f') | map(attribute='name') | list }}{{ items | rejectattr('type', 'equalto', 'f') | map(attribute='name') | join }}{{ ['a', 1, 'b'] | select('string') | list }}{{ [1, none, 2] | reject('none') | list }}"),
 ("map_filter", "{{ words | map('upper') | join }}"),
 ("dictsort", "{{ d | dictsort }}{{ d | dictsort(true) }}"),
 ("sort", "{{ words | sort }}{{ words | sort(reverse=true, case_sensitive=true) }}{{ items | sort(attribute='n') | map(attribute='name') | join }}"),
 ("case", "{{ 'aB' | lower }}{{ 'aB' | upper }}{{ 'aB'.upper() }}"),
 ("int", "{{ '42' | int }}{{ '4.7' | int }}{{ 'x' | int }}{{ 3.9 | int }}"),
 ("replace", "{{ 'aaa' | replace('a', 'b', 2) }}{{ 'a.b'.replace('.', '-') }}"),
 ("tojson", "{{ d | tojson }}|{{ items[0] | tojson(indent=2) }}|{{ 'é\"' | tojson }}|{{ 'é' | tojson(ensure_ascii=true) }}|{{ [] | tojson(indent=4) }}|{{ 1.5 | tojson }}"),
 ("tests", "{{ x is defined }}{{ x is undefined }}{{ n is none }}{{ s is string }}{{ d is mapping }}{{ words is iterable }}{{ words is sequence }}{{ t is boolean }}{{ t is true }}{{ t is false }}"),
 ("tests_args", "{{ 2 is equalto 2 }}{{ 2 is equalto 3 }}{{ none is boolean }}"),
 ("split", "{{ ' a  b c '.split() }}{{ 'a,b,,c'.split(',') }}{{ 'a b c'.split(' ', 1) }}"),
 ("strip_methods", "[{{ s.strip() }}][{{ s.lstrip() }}][{{ s.rstrip() }}][{{ 'xax'.strip('x') }}]"),
 ("starts_ends", "{{ 'abc'.startswith('ab') }}{{ 'abc'.startswith(('x', 'a')) }}{{ 'abc'.endswith('bc') }}"),
 ("dict_methods", "{{ d.get('a') }}{{ d.get('zz', 'no') }}{{ d.get('zz') }}{{ d.items() | list }}"),
 ("range", "{{ range(3) | list }}{{ range(1, 7, 2) | list }}"),
 ("strftime", "{{ strftime_now('%Y-%m-%d') }} {{ strftime_now('%d %b %Y') }} {{ strftime_now('%B %d, %Y') }}"),
 ("attribute_undefined", "[{{ d.zz }}][{{ items[5] }}][{{ d['zz'] }}][{{ d.zz | length }}]"),
 ("undefined_eq", "{{ d.zz == d.yy }}{{ d.zz == 1 }}{{ d.zz is defined }}"),
 ("in_unset", "{{ 'a' in unset_list }}{% for x in unset_list %}x{% endfor %}{% if unset_flag %}y{% endif %}{{ unset_flag is defined }}"),
 ("raise", "{{ raise_exception('stop here') }}"),
 ("type_iter_none", "{% for x in n %}{% endfor %}"),
 ("type_add", "{{ 'a' + 1 }}"),
 ("undefined_attr", "{{ d.zz.yy }}"),
]


def main():
    import jinja2
    import transformers
    import transformers.utils.chat_template_utils as chat

    if jinja2.__version__ != JINJA2_VERSION or transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(
            f"VersionOther: jinja2 {jinja2.__version__}, transformers {transformers.__version__}"
        )

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_DATE

    chat.datetime = FixedClock
    results = []
    for name, source in CASES:
        case = {"name": name, "source": source}
        try:
            case["output"] = chat._cached_compile_jinja_template(source).render(**VARS)
        except Exception as e:
            case["failure"] = type(e).__name__
            case["message"] = str(e)
        results.append(case)
    header = {
        "jinja2": JINJA2_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "fixed_date": FIXED_DATE.strftime("%Y-%m-%d"),
        "renderer": "transformers _cached_compile_jinja_template",
    }
    data = {"header": header, "variables": VARS, "cases": results}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
