//! A conversation as token ids ([`Transcript`]), and its bytes.
//!
//! [`Transcript::to_bytes`] writes, all little-endian:
//!
//! | offset | bytes | field |
//! |---|---|---|
//! | 0 | 4 | `BIVT` |
//! | 4 | 1 | version: 1 |
//! | 5 | 1 | token encoding: 0 for [`TokenEncoding::Full`], 1 for [`TokenEncoding::Compact`] |
//! | 6 | 2 | 0 |
//! | 8 | 4 | tokens |
//! | 12 | 4 | pending tokens, the last ones of the tokens |
//! | 16 | 4 | generated tokens |
//! | 20 | | the tokens, then the generated tokens, in the token encoding |
//!
//! and nothing after them.

use crate::Error;
use crate::conversation::Conversation;
use crate::prelude::*;
use crate::tokenizer::TokenId;

const MAGIC: [u8; 4] = *b"BIVT";
const VERSION: u8 = 1;
/// Bytes before the tokens
const HEADER: usize = 20;
/// The compact encoding's escape, and the id it adds to the u32 after it
const ESCAPE: u16 = 0xffff;

/// A conversation up to some point as token ids and its messages, see
/// [`crate::chat::Chat::transcript`]: a few bytes per token, but restoring it
/// processes the tokens again. Version 2 carries the messages of the
/// conversation so that a chat can render them; version 1 (from bytes of
/// an earlier format) has no messages, and restoring it fails with
/// [`Error::TranscriptNoMessages`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// every token of the conversation, in order: the system block, the
    /// messages and the answers, as the chat template writes them
    pub tokens: Vec<TokenId>,
    /// how many of the last `tokens` the model hasn't processed yet: the end
    /// of the last answer, fed before the next message
    pub pending: usize,
    /// the tokens the model generated, for the sampler's penalties
    pub generated: Vec<TokenId>,
    /// the messages of the conversation, for the chat to render: `None` in
    /// a version 1 transcript (from bytes), which a chat refuses
    pub conversation: Option<Conversation>,
}

impl Transcript {
    /// Tokens of the conversation
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The tokens processed, then those waiting for the next message
    pub fn split(&self) -> (&[TokenId], &[TokenId]) {
        self.tokens
            .split_at(self.tokens.len().saturating_sub(self.pending))
    }

    /// The transcript as bytes (see the module), its tokens in `encoding`
    pub fn to_bytes(&self, encoding: TokenEncoding) -> Vec<u8> {
        let count = |n: usize| {
            u32::try_from(n)
                .expect("fewer than 2^32 tokens")
                .to_le_bytes()
        };
        let mut out = Vec::with_capacity(
            HEADER + encoding.encoded_len(&self.tokens) + encoding.encoded_len(&self.generated),
        );
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&[VERSION, encoding.tag(), 0, 0]);
        out.extend_from_slice(&count(self.tokens.len()));
        out.extend_from_slice(&count(self.pending));
        out.extend_from_slice(&count(self.generated.len()));
        encoding.encode(&self.tokens, &mut out);
        encoding.encode(&self.generated, &mut out);
        out
    }

    /// A transcript from the bytes of [`Transcript::to_bytes`], in either
    /// token encoding. Its token ids are checked against the vocabulary when
    /// it is restored, by [`crate::chat::Chat::restore_transcript`].
    pub fn from_bytes(mut bytes: &[u8]) -> Result<Self, Error> {
        let b = &mut bytes;
        if take(b)? != MAGIC {
            return Err(Error::Transcript("not a transcript"));
        }
        let [version, tag, r0, r1] = take(b)?;
        if version != VERSION {
            return Err(Error::Transcript("unsupported version"));
        }
        let encoding =
            TokenEncoding::from_tag(tag).ok_or(Error::Transcript("unknown token encoding"))?;
        if (r0, r1) != (0, 0) {
            return Err(Error::Transcript("reserved header bytes not 0"));
        }
        let mut count = || take(b).map(|c| u32::from_le_bytes(c) as usize);
        let (n_tokens, pending, n_generated) = (count()?, count()?, count()?);
        if pending > n_tokens {
            return Err(Error::Transcript("more pending tokens than tokens"));
        }
        let tokens = encoding.decode(b, n_tokens)?;
        let generated = encoding.decode(b, n_generated)?;
        if !b.is_empty() {
            return Err(Error::Transcript("bytes after the tokens"));
        }
        Ok(Transcript {
            tokens,
            pending,
            generated,
            conversation: None,
        })
    }
}

/// How token ids are written, all little-endian
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenEncoding {
    /// 4 bytes per token: the id as a u32
    Full,
    /// 2 bytes per token below 0xffff: the id as a u16. From 0xffff, 6
    /// bytes: 0xffff, then the id minus 0xffff as a u32, so that 0xffff is
    /// `ff ff 00 00 00 00`. The lowest ids are the BPE's earliest merges,
    /// most of the tokens of a conversation.
    Compact,
}

impl TokenEncoding {
    fn tag(self) -> u8 {
        match self {
            TokenEncoding::Full => 0,
            TokenEncoding::Compact => 1,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(TokenEncoding::Full),
            1 => Some(TokenEncoding::Compact),
            _ => None,
        }
    }

    /// Bytes of `tokens` encoded
    pub fn encoded_len(self, tokens: &[TokenId]) -> usize {
        match self {
            TokenEncoding::Full => 4 * tokens.len(),
            TokenEncoding::Compact => tokens
                .iter()
                .map(|&t| if t < ESCAPE as TokenId { 2 } else { 6 })
                .sum(),
        }
    }

    /// Append `tokens` to `out`
    pub fn encode(self, tokens: &[TokenId], out: &mut Vec<u8>) {
        out.reserve(self.encoded_len(tokens));
        for &t in tokens {
            match self {
                TokenEncoding::Full => out.extend_from_slice(&t.to_le_bytes()),
                TokenEncoding::Compact => match u16::try_from(t) {
                    Ok(v) if v != ESCAPE => out.extend_from_slice(&v.to_le_bytes()),
                    _ => {
                        out.extend_from_slice(&ESCAPE.to_le_bytes());
                        out.extend_from_slice(&(t - ESCAPE as TokenId).to_le_bytes());
                    }
                },
            }
        }
    }

    /// The `n` tokens at the start of `bytes`, which moves past them
    pub fn decode(self, bytes: &mut &[u8], n: usize) -> Result<Vec<TokenId>, Error> {
        // no more than the bytes can hold, whatever `n` says
        let mut tokens = Vec::with_capacity(n.min(bytes.len() / 2));
        for _ in 0..n {
            tokens.push(match self {
                TokenEncoding::Full => u32::from_le_bytes(take(bytes)?),
                TokenEncoding::Compact => match u16::from_le_bytes(take(bytes)?) {
                    ESCAPE => u32::from_le_bytes(take(bytes)?)
                        .checked_add(ESCAPE as TokenId)
                        .ok_or(Error::Transcript("escaped token id past 2^32"))?,
                    v => v as TokenId,
                },
            });
        }
        Ok(tokens)
    }
}

/// The first `N` bytes, moving past them
fn take<const N: usize>(bytes: &mut &[u8]) -> Result<[u8; N], Error> {
    let (head, rest) = bytes
        .split_first_chunk::<N>()
        .ok_or(Error::Transcript("truncated"))?;
    *bytes = rest;
    Ok(*head)
}

#[cfg(test)]
mod tests {
    use super::{HEADER, TokenEncoding, Transcript};
    use crate::prelude::*;
    use crate::tokenizer::TokenId;

    const IDS: [TokenId; 9] = [
        0,
        1,
        0xfffe,
        0xffff,
        0x1_0000,
        248_319,
        0xffff_fffe,
        u32::MAX,
        42,
    ];

    fn encoded(encoding: TokenEncoding, tokens: &[TokenId]) -> Vec<u8> {
        let mut out = Vec::new();
        encoding.encode(tokens, &mut out);
        out
    }

    #[test]
    fn compact_escapes_from_0xffff() {
        assert_eq!(
            encoded(TokenEncoding::Compact, &[0x1234, 0xfffe]),
            [0x34, 0x12, 0xfe, 0xff]
        );
        assert_eq!(
            encoded(TokenEncoding::Compact, &[0xffff]),
            [0xff, 0xff, 0, 0, 0, 0]
        );
        assert_eq!(
            encoded(TokenEncoding::Compact, &[0x1_0000]),
            [0xff, 0xff, 1, 0, 0, 0]
        );
        assert_eq!(
            encoded(TokenEncoding::Compact, &[u32::MAX]),
            [0xff, 0xff, 0x00, 0x00, 0xff, 0xff]
        );
    }

    #[test]
    fn full_is_a_u32_per_token() {
        assert_eq!(
            encoded(TokenEncoding::Full, &[0xffff, 0x0102_0304]),
            [0xff, 0xff, 0, 0, 4, 3, 2, 1]
        );
    }

    #[test]
    fn tokens_round_trip() {
        for encoding in [TokenEncoding::Full, TokenEncoding::Compact] {
            let bytes = encoded(encoding, &IDS);
            assert_eq!(bytes.len(), encoding.encoded_len(&IDS));
            let mut b = &bytes[..];
            assert_eq!(encoding.decode(&mut b, IDS.len()).unwrap(), IDS);
            assert!(b.is_empty());
        }
    }

    #[test]
    fn escaped_id_past_u32_is_refused() {
        let mut b: &[u8] = &[0xff, 0xff, 0x00, 0x00, 0xff, 0xff];
        assert_eq!(
            TokenEncoding::Compact.decode(&mut b, 1).unwrap(),
            [u32::MAX]
        );
        let mut b: &[u8] = &[0xff, 0xff, 0x01, 0x00, 0xff, 0xff];
        assert!(TokenEncoding::Compact.decode(&mut b, 1).is_err());
    }

    #[test]
    fn transcript_round_trip() {
        let t = Transcript {
            tokens: IDS.to_vec(),
            pending: 2,
            generated: vec![0xffff, 7, 248_000],
            conversation: None,
        };
        for encoding in [TokenEncoding::Full, TokenEncoding::Compact] {
            let bytes = t.to_bytes(encoding);
            assert_eq!(
                bytes.len(),
                HEADER + encoding.encoded_len(&t.tokens) + encoding.encoded_len(&t.generated)
            );
            assert_eq!(Transcript::from_bytes(&bytes).unwrap(), t);
            // every shorter prefix is refused, and so is a byte more
            for n in 0..bytes.len() {
                assert!(Transcript::from_bytes(&bytes[..n]).is_err(), "{n} bytes");
            }
            let mut longer = bytes.clone();
            longer.push(0);
            assert!(Transcript::from_bytes(&longer).is_err());
        }
        let empty = Transcript {
            tokens: Vec::new(),
            pending: 0,
            generated: Vec::new(),
            conversation: None,
        };
        assert_eq!(
            Transcript::from_bytes(&empty.to_bytes(TokenEncoding::Compact)).unwrap(),
            empty
        );
    }

    #[test]
    fn bad_headers_are_refused() {
        let good = Transcript {
            tokens: vec![1, 2, 3],
            pending: 1,
            generated: vec![3],
            conversation: None,
        }
        .to_bytes(TokenEncoding::Full);
        let with = |i: usize, v: u8| {
            let mut b = good.clone();
            b[i] = v;
            Transcript::from_bytes(&b)
        };
        assert!(with(0, b'X').is_err(), "magic");
        assert!(with(4, 2).is_err(), "version");
        assert!(with(5, 2).is_err(), "encoding");
        assert!(with(6, 1).is_err(), "reserved");
        assert!(with(12, 4).is_err(), "pending > tokens");
        // a huge count with no tokens after it
        let mut huge = good[..HEADER].to_vec();
        huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Transcript::from_bytes(&huge).is_err());
    }
}
