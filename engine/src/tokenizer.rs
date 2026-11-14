//! GPT-2 style byte-level BPE, as stored in GGUF (`tokenizer.ggml.model = gpt2`)
//! with the `qwen35` pre-tokenizer.

use core::cmp::Ordering;

use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

use crate::Error;
use crate::gguf::{Gguf, Metadata, need, read_metadata};
use crate::prelude::*;

const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;

pub type TokenId = u32;

/// A piece of a prompt, see [`Tokenizer::encode_prompt`]
#[derive(Debug, Clone, Copy)]
pub enum PromptPart<'a> {
    /// the chat template's own text, where special token strings are tokens
    Template(&'a str),
    /// message content, where they are plain text
    Content(&'a str),
}

pub struct Tokenizer {
    /// token text, in the GPT-2 byte-to-unicode encoding (or raw for special tokens)
    tokens: Vec<String>,
    /// decoded bytes of each token
    bytes: Vec<Vec<u8>>,
    is_special: Vec<bool>,
    by_text: ByText,
    merges: Merges,
    /// token id of each raw byte
    byte_tokens: [TokenId; 256],
    /// special tokens, longest first, for splitting text that may contain them
    specials: Vec<(String, TokenId)>,
    pub bos: Option<TokenId>,
    pub eos: TokenId,
    /// tokens that end a generation
    pub eog: Vec<TokenId>,
    /// the chat template of the model file (`tokenizer.chat_template`)
    pub chat_template: Option<String>,
}

/// One BPE merge of a left token, which [`Merges`] files it under
#[derive(Clone, Copy)]
struct Merge {
    right: TokenId,
    rank: u32,
    merged: TokenId,
}

/// The BPE merges, by left token: those of `l` are at
/// `offsets[l]..offsets[l + 1]` of the other arrays, sorted by right token.
/// The right tokens, which the lookups search, are an array of their own.
struct Merges {
    offsets: Vec<u32>,
    rights: Vec<TokenId>,
    /// rank and merged token
    merges: Vec<(u32, TokenId)>,
}

impl Merges {
    /// From `(left, merge)` in rank order; for a pair listed twice, the
    /// first (lowest rank) one counts
    fn new(n_vocab: usize, mut list: Vec<(TokenId, Merge)>) -> Self {
        list.sort_by_key(|&(l, m)| (l, m.right, m.rank));
        list.dedup_by_key(|&mut (l, m)| (l, m.right));
        let mut offsets = vec![0u32; n_vocab + 1];
        for &(l, _) in &list {
            offsets[l as usize + 1] += 1;
        }
        for i in 0..n_vocab {
            offsets[i + 1] += offsets[i];
        }
        Merges {
            offsets,
            rights: list.iter().map(|(_, m)| m.right).collect(),
            merges: list.iter().map(|(_, m)| (m.rank, m.merged)).collect(),
        }
    }

    /// The rank and the merged token of `(l, r)`
    fn get(&self, l: TokenId, r: TokenId) -> Option<(u32, TokenId)> {
        let l = l as usize;
        let start = *self.offsets.get(l)? as usize;
        let end = self.offsets[l + 1] as usize;
        let i = self.rights[start..end].binary_search(&r).ok()?;
        Some(self.merges[start + i])
    }
}

/// The buffers of [`Tokenizer::bpe`], reused from one piece to the next
#[derive(Default)]
struct Bpe {
    syms: Vec<TokenId>,
    /// the merge of each adjacent pair of `syms`: a round looks up only the
    /// pairs its merges made
    pairs: Vec<Option<(u32, TokenId)>>,
    next_syms: Vec<TokenId>,
    next_pairs: Vec<Option<(u32, TokenId)>>,
    /// index in `syms` of each symbol of `next_syms` a round left as is
    kept: Vec<Option<usize>>,
}

/// The order of `t` and the concatenation of `a` and `b`, without building it
fn cmp_concat(t: &[u8], a: &[u8], b: &[u8]) -> Ordering {
    let n = t.len().min(a.len());
    match t[..n].cmp(&a[..n]) {
        Ordering::Equal if t.len() < a.len() => Ordering::Less,
        Ordering::Equal => t[a.len()..].cmp(b),
        o => o,
    }
}

/// The first [`PREFIX`] bytes of `a` followed by `b`, zero-padded, as a
/// big-endian number: in the same order as the texts, ties apart
fn prefix(a: &[u8], b: &[u8]) -> u128 {
    let mut p = [0u8; PREFIX];
    for (d, s) in p.iter_mut().zip(a.iter().chain(b)) {
        *d = *s;
    }
    u128::from_be_bytes(p)
}

const PREFIX: usize = 16;

/// A token of [`ByText`]: its id, and the [`prefix`] and length of its text
#[derive(Clone, Copy)]
struct Entry {
    prefix: u128,
    id: TokenId,
    len: u32,
}

impl Entry {
    /// The order of this token's text and `a` followed by `b`, whose prefix
    /// is `p`
    fn cmp(&self, tokens: &[String], p: u128, a: &[u8], b: &[u8]) -> Ordering {
        self.cmp_by(p, a.len() + b.len(), || {
            cmp_concat(tokens[self.id as usize].as_bytes(), a, b)
        })
    }

    /// The order of this token's text and a text of prefix `p` and length
    /// `len`, which `texts` compares to it. With the same prefix, a text of at
    /// most [`PREFIX`] bytes is the start of the other one, so the lengths
    /// alone order them.
    fn cmp_by(&self, p: u128, len: usize, texts: impl FnOnce() -> Ordering) -> Ordering {
        self.prefix.cmp(&p).then_with(|| {
            if (self.len as usize).min(len) <= PREFIX {
                (self.len as usize).cmp(&len)
            } else {
                texts()
            }
        })
    }
}

/// The token ids sorted by text, to find a token by its text. The searches
/// compare the prefixes and lengths of the [`Entry`]s, and read a text only
/// when both are longer than their common prefix.
struct ByText(Vec<Entry>);

impl ByText {
    /// For a text listed twice, the last id counts
    fn new(tokens: &[String]) -> Self {
        let mut entries: Vec<Entry> = tokens
            .iter()
            .enumerate()
            .map(|(id, t)| Entry {
                prefix: prefix(t.as_bytes(), b""),
                id: id as TokenId,
                len: t.len() as u32,
            })
            .collect();
        let text = |e: &Entry| tokens[e.id as usize].as_bytes();
        let cmp =
            |x: &Entry, y: &Entry| x.cmp_by(y.prefix, y.len as usize, || text(x).cmp(text(y)));
        entries.sort_unstable_by(|x, y| cmp(x, y).then(y.id.cmp(&x.id)));
        entries.dedup_by(|y, x| cmp(x, y) == Ordering::Equal);
        ByText(entries)
    }

    /// The id of the token whose text is `a` followed by `b`
    fn find(&self, tokens: &[String], a: &str, b: &str) -> Option<TokenId> {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        let p = prefix(a, b);
        let i = self.0.binary_search_by(|e| e.cmp(tokens, p, a, b)).ok()?;
        Some(self.0[i].id)
    }

    /// [`find`](Self::find) for each of the `n` texts `query(i)`, at once: in
    /// the order of their prefixes, alongside the sorted ids, so each search
    /// only covers the ids with the same prefix
    fn find_all<'a>(
        &self,
        tokens: &[String],
        n: usize,
        query: impl Fn(usize) -> (&'a str, &'a str),
    ) -> Vec<Option<TokenId>> {
        let mut order: Vec<(u128, u32)> = (0..n)
            .map(|i| {
                let (a, b) = query(i);
                (prefix(a.as_bytes(), b.as_bytes()), i as u32)
            })
            .collect();
        order.sort_unstable();
        let mut found = vec![None; n];
        let mut hi = 0;
        for group in order.chunk_by(|x, y| x.0 == y.0) {
            let p = group[0].0;
            let mut lo = hi;
            while lo < self.0.len() && self.0[lo].prefix < p {
                lo += 1;
            }
            hi = lo;
            while hi < self.0.len() && self.0[hi].prefix == p {
                hi += 1;
            }
            let same = &self.0[lo..hi];
            for &(_, i) in group {
                let (a, b) = query(i as usize);
                let (a, b) = (a.as_bytes(), b.as_bytes());
                found[i as usize] = same
                    .binary_search_by(|e| e.cmp(tokens, p, a, b))
                    .ok()
                    .map(|j| same[j].id);
            }
        }
        found
    }
}

/// The characters [`byte_to_unicode`] maps bytes to are all below this
const UNICODE_BYTES: usize = 0x144;

/// GPT-2's reversible byte <-> printable unicode mapping
fn byte_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (b'!' as u32..=b'~' as u32).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

impl Tokenizer {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, Error> {
        Self::from_metadata(&gguf.metadata)
    }

    /// The tokenizer of a GGUF file, from its first bytes: only its metadata
    /// is parsed, and of it only the `tokenizer.*` keys are read. The tensor
    /// table and data, and the other shards of a split file, are not needed.
    pub fn from_gguf_bytes(buf: &[u8]) -> Result<Self, Error> {
        let m = read_metadata(buf, |key| key.starts_with("tokenizer."))?;
        Self::from_metadata(&m)
    }

    pub fn from_metadata(m: &Metadata) -> Result<Self, Error> {
        let ggml = &m.tokenizer.ggml;
        let model = need!(m, tokenizer.ggml.model)?;
        let pre = ggml.pre.as_deref().unwrap_or("qwen35");
        if model != "gpt2" || pre != "qwen35" {
            return Err(Error::UnsupportedTokenizer {
                model: model.clone(),
                pre: pre.into(),
            });
        }
        let tokens: Vec<String> = need!(m, tokenizer.ggml.tokens)?.clone();
        let types: Vec<i64> = match &ggml.token_type {
            Some(a) => a.iter().map(|&t| t.into()).collect(),
            None => vec![1; tokens.len()],
        };

        let mut unicode_to_byte = [None; UNICODE_BYTES];
        for (b, &c) in byte_to_unicode().iter().enumerate() {
            unicode_to_byte[c as usize] = Some(b as u8);
        }

        let is_special: Vec<bool> = types
            .iter()
            .map(|&t| t == TOKEN_TYPE_CONTROL || t == TOKEN_TYPE_USER_DEFINED)
            .collect();
        let bytes: Vec<Vec<u8>> = tokens
            .iter()
            .zip(&is_special)
            .map(|(t, &special)| {
                if special {
                    t.as_bytes().to_vec()
                } else {
                    t.chars()
                        .map(|c| unicode_to_byte.get(c as usize).copied().flatten())
                        .collect::<Option<Vec<u8>>>()
                        .unwrap_or_else(|| t.as_bytes().to_vec())
                }
            })
            .collect();

        let by_text = ByText::new(&tokens);
        let id = |a: &str, b: &str| by_text.find(&tokens, a, b);

        let mut byte_tokens = [0; 256];
        for (b, c) in byte_to_unicode().iter().enumerate() {
            let c = c.to_string();
            byte_tokens[b] = id(&c, "").ok_or(Error::MissingToken(c))?;
        }

        // the left, right and merged token of each merge: 3 lookups each
        let pairs: Vec<(u32, &str, &str)> = need!(m, tokenizer.ggml.merges)?
            .iter()
            .enumerate()
            .filter_map(|(rank, merge)| {
                let (l, r) = merge.split_once(' ')?;
                Some((rank as u32, l, r))
            })
            .collect();
        let ids = by_text.find_all(&tokens, 3 * pairs.len(), |i| {
            let (_, l, r) = pairs[i / 3];
            match i % 3 {
                0 => (l, ""),
                1 => (r, ""),
                _ => (l, r),
            }
        });
        let merges = pairs
            .iter()
            .zip(ids.as_chunks().0)
            .filter_map(|(&(rank, _, _), &[li, right, merged])| {
                let merge = Merge {
                    right: right?,
                    rank,
                    merged: merged?,
                };
                Some((li?, merge))
            })
            .collect();
        let merges = Merges::new(tokens.len(), merges);

        let mut specials: Vec<(String, TokenId)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| is_special[*i] && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as TokenId))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let eos = *need!(m, tokenizer.ggml.eos_token_id)?;
        let mut eog = vec![eos];
        for t in ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>"] {
            if let Some(i) = id(t, "")
                && !eog.contains(&i)
            {
                eog.push(i);
            }
        }

        Ok(Tokenizer {
            tokens,
            bytes,
            is_special,
            by_text,
            merges,
            byte_tokens,
            specials,
            bos: ggml.bos_token_id,
            eos,
            eog,
            chat_template: m.tokenizer.chat_template.clone(),
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, text: &str) -> Option<TokenId> {
        self.by_text.find(&self.tokens, text, "")
    }

    pub fn is_special(&self, id: TokenId) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn token_text(&self, id: TokenId) -> &str {
        &self.tokens[id as usize]
    }

    /// The raw bytes a token stands for (a token may hold a partial UTF-8 sequence)
    pub fn token_bytes(&self, id: TokenId) -> &[u8] {
        &self.bytes[id as usize]
    }

    pub fn decode(&self, ids: &[TokenId]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.token_bytes(i).iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Tokenize plain text; special token strings are not recognized
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    /// Tokenize text, mapping special token strings (`<|im_start|>`, …) to their ids
    pub fn encode_with_special(&self, text: &str) -> Vec<TokenId> {
        self.encode_prompt(&[PromptPart::Template(text)])
    }

    /// Tokenize a rendered prompt. Special token strings are recognized in the
    /// template's text only, and the text between special tokens is tokenized
    /// whole, across parts, as it would be in one string.
    pub fn encode_prompt(&self, parts: &[PromptPart<'_>]) -> Vec<TokenId> {
        let mut out = Vec::new();
        let mut text = String::new();
        for part in parts {
            match *part {
                PromptPart::Content(s) => text.push_str(s),
                PromptPart::Template(mut rest) => {
                    while let Some((pos, len, id)) = self.find_special(rest) {
                        text.push_str(&rest[..pos]);
                        self.encode_into(&text, &mut out);
                        text.clear();
                        out.push(id);
                        rest = &rest[pos + len..];
                    }
                    text.push_str(rest);
                }
            }
        }
        self.encode_into(&text, &mut out);
        out
    }

    /// The earliest special token string in `text`: position, length and id
    fn find_special(&self, text: &str) -> Option<(usize, usize, TokenId)> {
        // specials are sorted longest first, so the first found at a position
        // is the longest there
        let mut best: Option<(usize, usize, TokenId)> = None;
        for (s, id) in &self.specials {
            if let Some(pos) = text.find(s.as_str())
                && best.is_none_or(|(bp, _, _)| pos < bp)
            {
                best = Some((pos, s.len(), *id));
            }
        }
        best
    }

    fn encode_into(&self, text: &str, out: &mut Vec<TokenId>) {
        let mut scratch = Bpe::default();
        for piece in pre_split(text) {
            self.bpe(piece.as_bytes(), &mut scratch, out);
        }
    }

    fn bpe(&self, piece: &[u8], scratch: &mut Bpe, out: &mut Vec<TokenId>) {
        let Bpe {
            syms,
            pairs,
            next_syms,
            next_pairs,
            kept,
        } = scratch;
        syms.clear();
        syms.extend(piece.iter().map(|&b| self.byte_tokens[b as usize]));
        pairs.clear();
        pairs.extend(syms.windows(2).map(|w| self.merges.get(w[0], w[1])));
        while let Some(rank) = pairs.iter().flatten().map(|&(rank, _)| rank).min() {
            // merge every occurrence of this pair (one rank per pair), left to right
            next_syms.clear();
            kept.clear();
            let mut i = 0;
            while i < syms.len() {
                if let Some(&Some((r, m))) = pairs.get(i)
                    && r == rank
                {
                    next_syms.push(m);
                    kept.push(None);
                    i += 2;
                } else {
                    next_syms.push(syms[i]);
                    kept.push(Some(i));
                    i += 1;
                }
            }
            next_pairs.clear();
            next_pairs.extend((1..next_syms.len()).map(|j| match (kept[j - 1], kept[j]) {
                // two symbols left as is were neighbours already
                (Some(a), Some(_)) => pairs[a],
                _ => self.merges.get(next_syms[j - 1], next_syms[j]),
            }));
            core::mem::swap(syms, next_syms);
            core::mem::swap(pairs, next_pairs);
        }
        out.extend_from_slice(syms);
    }
}

/// llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN35` split, a regex:
///
/// ```text
/// (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
/// | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// matched by hand. The pieces cover the whole text: every char starts one
/// of the alternatives.
fn pre_split(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    core::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let (piece, tail) = rest.split_at(piece_len(rest));
        rest = tail;
        Some(piece)
    })
}

/// The classes of the regex: `\p{L}`, `\p{M}`, `\p{N}`, `\s`, and none of them
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Mark,
    Number,
    Space,
    Other,
}

fn class(c: char) -> Class {
    if c.is_ascii() {
        return match c {
            'a'..='z' | 'A'..='Z' => Class::Letter,
            '0'..='9' => Class::Number,
            '\t'..='\r' | ' ' => Class::Space,
            _ => Class::Other,
        };
    }
    // White_Space, as `\s`; no whitespace char is a letter, mark or number
    if c.is_whitespace() {
        return Class::Space;
    }
    match c.general_category_group() {
        GeneralCategoryGroup::Letter => Class::Letter,
        GeneralCategoryGroup::Mark => Class::Mark,
        GeneralCategoryGroup::Number => Class::Number,
        _ => Class::Other,
    }
}

/// End of the run of chars accepted by `f` in `s`, from byte `from`
fn run(s: &str, from: usize, f: impl Fn(char) -> bool) -> usize {
    from + s[from..].find(|c| !f(c)).unwrap_or(s.len() - from)
}

/// Byte length of the piece at the start of `s`: the first alternative of the
/// regex that matches there, as long as backtracking makes it
fn piece_len(s: &str) -> usize {
    use Class::*;
    let mut chars = s.chars();
    let c0 = chars.next().expect("a piece of an empty text");
    let (k0, k1) = (class(c0), chars.next().map(class));
    let letter = |c| matches!(class(c), Letter | Mark);

    // '[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]
    if c0 == '\'' {
        let lower = |i: usize| s.as_bytes().get(i).map(u8::to_ascii_lowercase);
        match (lower(1), lower(2)) {
            (Some(b's' | b't' | b'm' | b'd'), _) => return 2,
            (Some(b'r' | b'v'), Some(b'e')) | (Some(b'l'), Some(b'l')) => return 3,
            _ => {}
        }
    }
    // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if matches!(k1, Some(Letter | Mark))
        && !matches!(c0, '\r' | '\n')
        && !matches!(k0, Letter | Number)
    {
        return run(s, c0.len_utf8(), letter);
    }
    if matches!(k0, Letter | Mark) {
        return run(s, 0, letter);
    }
    // \p{N}
    if k0 == Number {
        return c0.len_utf8();
    }
    // ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
    let start = match (c0, k0, k1) {
        (_, Other, _) => Some(0),
        (' ', _, Some(Other)) => Some(1),
        _ => None,
    };
    if let Some(start) = start {
        let end = run(s, start, |c| class(c) == Other);
        return run(s, end, |c| matches!(c, '\r' | '\n'));
    }

    // whitespace is left
    debug_assert!(k0 == Space);
    let ws = run(s, 0, |c| class(c) == Space);
    // \s*[\r\n]+: up to the run's last newline
    if let Some(nl) = s[..ws].rfind(['\r', '\n']) {
        return nl + 1;
    }
    // \s+(?!\S): the run, but for its last char when a non-space follows,
    // which then starts the next piece; \s+: a single space before a non-space
    let last = s[..ws].chars().next_back().map_or(0, char::len_utf8);
    if ws == s.len() || ws == last {
        ws
    } else {
        ws - last
    }
}

/// Turns a stream of tokens into text, holding back incomplete UTF-8 sequences
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match core::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // an invalid (not just incomplete) sequence: emit it lossily
                let cut = if e.error_len().is_some() {
                    self.pending.len()
                } else {
                    valid
                };
                let s = String::from_utf8_lossy(&self.pending[..cut]).into_owned();
                self.pending.drain(..cut);
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn by_text_lookup() {
        let tokens: Vec<String> = [
            "ab",
            "abcdefghijklmnop1",
            "a",
            "abcdefghijklmnop",
            "abcdefghijklmnop2",
            "",
            "b",
            "ab",
            "abcdefghijklmnopqr",
            "a\0",
            "a\0\0",
        ]
        .map(String::from)
        .into();
        let t = ByText::new(&tokens);
        let queries = [
            // listed twice: the last id
            (("ab", ""), Some(7)),
            (("a", "b"), Some(7)),
            (("", "ab"), Some(7)),
            // the same first 16 bytes
            (("abcdefghijklmnop", ""), Some(3)),
            (("abcdefghijklmnop", "1"), Some(1)),
            (("abcdefgh", "ijklmnop2"), Some(4)),
            (("abcdefghijklmnop", "qr"), Some(8)),
            (("abcdefghijklmnop", "q"), None),
            (("abcdefghijklmnop", "3"), None),
            // the same prefix, zero-padded
            (("a", "\0"), Some(9)),
            (("a\0\0", ""), Some(10)),
            (("a\0\0\0", ""), None),
            (("", ""), Some(5)),
            (("b", ""), Some(6)),
            (("c", ""), None),
            (("abc", ""), None),
        ];
        for ((a, b), id) in queries {
            assert_eq!(t.find(&tokens, a, b), id, "{a:?} {b:?}");
        }
        let all = t.find_all(&tokens, queries.len(), |i| queries[i].0);
        assert_eq!(all, queries.map(|q| q.1));
    }

    #[test]
    fn merges_lookup() {
        let m = |right, rank, merged| Merge {
            right,
            rank,
            merged,
        };
        // in rank order, with (1, 2) listed twice
        let merges = Merges::new(
            5,
            vec![
                (1, m(2, 0, 10)),
                (3, m(4, 1, 11)),
                (1, m(0, 2, 12)),
                (1, m(2, 3, 13)),
                (4, m(4, 4, 14)),
            ],
        );
        assert_eq!(merges.get(1, 2), Some((0, 10)));
        assert_eq!(merges.get(1, 0), Some((2, 12)));
        assert_eq!(merges.get(3, 4), Some((1, 11)));
        assert_eq!(merges.get(4, 4), Some((4, 14)));
        assert_eq!(merges.get(1, 1), None);
        assert_eq!(merges.get(0, 2), None);
        assert_eq!(merges.get(2, 1), None);
        // a token past the vocabulary has no merges
        assert_eq!(merges.get(9, 1), None);
    }

    /// The regex the pre-tokenizer implements, as llama.cpp has it
    const QWEN35_PRE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn check(re: &fancy_regex::Regex, text: &str) {
        let want: Vec<&str> = re.find_iter(text).map(|m| m.unwrap().as_str()).collect();
        let got: Vec<&str> = pre_split(text).collect();
        assert_eq!(got, want, "split of {text:?}");
    }

    #[test]
    fn pre_split_matches_regex() {
        let re = fancy_regex::Regex::new(QWEN35_PRE).unwrap();
        for text in [
            "",
            "Hello world! It's 2026, isn't it?   ok\n\n  x",
            "I'LL 'Re 'sure' 'x 'd' '",
            "e\u{301}t\u{e9} \u{301}x नमस्ते ١٢٣ ½ Ⅻ 你好 🦀",
            "  \t\n \r\n  trailing   ",
            "a\u{a0}\u{a0}b\u{3000}c \u{2028}d \u{85}",
            "x  ",
            "   ",
            " .\n\r\n!",
        ] {
            check(&re, text);
        }
        // random sequences of pieces that exercise every alternative
        let pool = [
            "a",
            "Z",
            "é",
            "e\u{301}",
            "\u{301}",
            "ß",
            "中",
            "😀",
            "👍🏽",
            "\u{200d}",
            "\u{feff}",
            "0",
            "9",
            "½",
            "Ⅻ",
            "٣",
            "क",
            "ि",
            "्",
            " ",
            "  ",
            "\t",
            "\n",
            "\r",
            "\r\n",
            "\u{a0}",
            "\u{3000}",
            "\u{2028}",
            "\u{85}",
            "\u{b}",
            "\u{0}",
            "\u{1f}",
            "'",
            "'s",
            "'S",
            "'re",
            "'VE",
            "'lL",
            "'d",
            "'m",
            "'t",
            "'x",
            "'r",
            "'l",
            ".",
            ",",
            "!",
            "-",
            "_",
            "(",
            "\"",
            "/",
            "$",
            "€",
            "+",
            "<|im_start|>",
        ];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        for _ in 0..50_000 {
            let len = 1 + rand(12);
            let text: String = (0..len).map(|_| pool[rand(pool.len())]).collect();
            check(&re, &text);
        }
    }
}
