//! The runtime's filters against Python's and Jinja2's results.

extern crate alloc;

#[allow(dead_code)]
#[path = "../../../engine/src/template/mod.rs"]
mod template;

use template::rt::{self, Json, Value};

fn text(v: rt::R) -> String {
    match v.unwrap() {
        Value::Str(s) => s.text(),
        other => panic!("not a string: {other:?}"),
    }
}

/// `{"b": {"z": 1, "a": [1, 2.5, "é", None, True]}, "A": "x\n"}`
fn sample() -> Value {
    rt::json(&Json::Object(vec![
        (
            "b".into(),
            Json::Object(vec![
                ("z".into(), Json::Int(1)),
                (
                    "a".into(),
                    Json::Array(vec![
                        Json::Int(1),
                        Json::Float(2.5),
                        Json::Str("é".into()),
                        Json::Null,
                        Json::Bool(true),
                    ]),
                ),
            ]),
        ),
        ("A".into(), Json::Str("x\n".into())),
    ]))
}

#[test]
fn tojson_separators() {
    let u = Value::Undefined;
    let t = Value::Bool(true);
    // json.dumps(v, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    assert_eq!(
        text(rt::f_tojson_sep(&sample(), &u, &u, &t, ",", ":")),
        r#"{"A":"x\n","b":{"a":[1,2.5,"é",null,true],"z":1}}"#
    );
    assert_eq!(
        text(rt::f_tojson_sep(&sample(), &u, &u, &u, ",", ":")),
        r#"{"b":{"z":1,"a":[1,2.5,"é",null,true]},"A":"x\n"}"#
    );
    assert_eq!(
        text(rt::f_tojson_sep(
            &sample(),
            &u,
            &Value::Int(1),
            &u,
            ";",
            "="
        )),
        "{\n \"b\"={\n  \"z\"=1;\n  \"a\"=[\n   1;\n   2.5;\n   \"é\";\n   null;\n   true\n  ]\n };\n \"A\"=\"x\\n\"\n}"
    );
    // the defaults are Python's, with and without indent
    assert_eq!(
        text(rt::f_tojson(&sample(), &u, &u, &u)),
        r#"{"b": {"z": 1, "a": [1, 2.5, "é", null, true]}, "A": "x\n"}"#
    );
    assert_eq!(
        text(rt::f_tojson(&sample(), &u, &Value::Int(2), &u)),
        "{\n  \"b\": {\n    \"z\": 1,\n    \"a\": [\n      1,\n      2.5,\n      \"é\",\n      null,\n      true\n    ]\n  },\n  \"A\": \"x\\n\"\n}"
    );
}

#[test]
fn dictsort() {
    let m = rt::json(&Json::Object(vec![
        ("b".into(), Json::Int(1)),
        ("Zeta".into(), Json::Int(2)),
        ("a".into(), Json::Int(3)),
        ("B".into(), Json::Int(4)),
    ]));
    let keys = |v: rt::R| -> Vec<String> {
        rt::iter(&v.unwrap())
            .unwrap()
            .iter()
            .map(|p| {
                let kv = rt::unpack(p, 2).unwrap();
                format!(
                    "{}={}",
                    rt::to_str(&kv[0]).text(),
                    rt::to_str(&kv[1]).text()
                )
            })
            .collect()
    };
    // Jinja2: sorted by code point, or by key.lower() with ties in order
    assert_eq!(
        keys(rt::f_dictsort(&m, &Value::Bool(true))),
        ["B=4", "Zeta=2", "a=3", "b=1"]
    );
    assert_eq!(
        keys(rt::f_dictsort(&m, &Value::Undefined)),
        ["a=3", "b=1", "B=4", "Zeta=2"]
    );
    assert!(rt::f_dictsort(&Value::Int(1), &Value::Undefined).is_err());
}

#[test]
fn float() {
    let f = |v: Value| match rt::f_float(&v, &Value::Undefined).unwrap() {
        Value::Float(f) => f,
        other => panic!("not a float: {other:?}"),
    };
    assert_eq!(f(Value::Int(3)), 3.0);
    assert_eq!(f(Value::Float(0.9)), 0.9);
    assert_eq!(f(Value::Bool(true)), 1.0);
    assert_eq!(f(Value::data(" 0.5\n")), 0.5);
    // what Python's float() refuses gives the default
    assert_eq!(f(Value::data("high")), 0.0);
    assert_eq!(f(Value::None), 0.0);
    assert_eq!(f(Value::list(vec![Value::Int(1)])), 0.0);
    assert!(matches!(
        rt::f_float(&Value::data("x"), &Value::Int(7)),
        Ok(Value::Int(7))
    ));
}

/// Python's `repr()` of a double halfway between two shortest digit strings
/// takes the one ending in an even digit
#[test]
// the literals are the doubles' exact values
#[allow(clippy::excessive_precision)]
fn float_ties() {
    let repr = |f: f64| rt::to_str(&Value::Float(f)).text();
    // exactly …765.25 and …392.25: Rust's `{}` writes …765.3 and …392.3
    assert_eq!(repr(1668570629156765.25), "1668570629156765.2");
    assert_eq!(repr(-1340368882625392.25), "-1340368882625392.2");
    assert_eq!(repr(0.223667144775390625), "0.22366714477539062");
    // halfway too, the even digit the larger
    assert_eq!(repr(1668570629156765.75), "1668570629156765.8");
    // not halfway
    assert_eq!(repr(0.1), "0.1");
    assert_eq!(repr(2.5e-7), "2.5e-07");
    assert_eq!(repr(1e16), "1e+16");
    let u = Value::Undefined;
    let v = rt::json(&Json::Array(vec![Json::Float(1668570629156765.25)]));
    assert_eq!(
        text(rt::f_tojson_sep(&v, &u, &u, &u, ",", ":")),
        "[1668570629156765.2]"
    );
}

#[test]
fn message_name() {
    let mut m = rt::Message::new(rt::Role::Tool, "ok");
    let has_name = |m: &rt::Message| {
        let v = rt::messages(core::slice::from_ref(m));
        let e = rt::iter(&v).unwrap().remove(0);
        rt::getattr(&e, "name")
    };
    assert!(has_name(&m).is_undefined());
    m.name = Some("f".into());
    assert_eq!(rt::to_str(&has_name(&m)).text(), "f");
}
