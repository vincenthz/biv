//! Token sampling: penalties, temperature, top-k, top-p, min-p.

use crate::Error;
use crate::gguf::Gguf;
use crate::math::Float;
use crate::prelude::*;
use crate::tokenizer::TokenId;

/// The sampling values a model's publisher can store in the GGUF

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0 = greedy
    pub temperature: f32,
    /// 0 = disabled
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// subtracted once from the logit of every token already generated
    pub presence_penalty: f32,
    /// divides positive / multiplies negative logits of recent tokens
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// of the random generator
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        // the model card's recommended thinking-mode settings
        SamplerConfig {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            presence_penalty: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

/// Values that replace those of [`SamplerConfig::from_gguf`]; `None` keeps
/// the GGUF's value, or the [`SamplerConfig::default`] one for a value the
/// GGUF doesn't store
#[derive(Debug, Clone)]
pub struct SamplerOverrides {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
}

impl SamplerConfig {
    /// The sampling values recommended by the model's publisher, from the
    /// GGUF, with `overrides` over them. Fails with [`Error::ZeroSeed`] when
    /// `seed` is 0: the caller names the seed, so that a run can be repeated.
    pub fn from_gguf(gguf: &Gguf, overrides: SamplerOverrides, seed: u64) -> Result<Self, Error> {
        if seed == 0 {
            return Err(Error::ZeroSeed);
        }
        let stored = &gguf.metadata.general.sampling;
        let defaults = SamplerConfig::default();
        Ok(SamplerConfig {
            temperature: overrides
                .temperature
                .or(stored.temp)
                .unwrap_or(defaults.temperature),
            top_k: overrides
                .top_k
                .or(stored.top_k.map(|v| v as usize))
                .unwrap_or(defaults.top_k),
            top_p: overrides.top_p.or(stored.top_p).unwrap_or(defaults.top_p),
            min_p: overrides.min_p.or(stored.min_p).unwrap_or(defaults.min_p),
            presence_penalty: overrides
                .presence_penalty
                .unwrap_or(defaults.presence_penalty),
            repeat_penalty: overrides.repeat_penalty.unwrap_or(defaults.repeat_penalty),
            seed,
            ..defaults
        })
    }
}

/// A rule of which tokens may come next, such as a grammar that the text
/// must follow. A caller sets it on a [`Sampler`] with
/// [`Sampler::set_mask`]; the sampler applies it before each choice, and
/// the chat tells it each chosen token through [`Sampler::accept`].
///
/// The mask sees every chosen token of the answer, the reasoning block
/// included, so a grammar of the answer text must also allow the reasoning,
/// or the caller turns thinking off. At a stop by `max_tokens`, by the end
/// of the context or by the caller, the chat closes the message with
/// `</think>`, `</tool_call>` and `<|im_end|>` as needed; those close tokens
/// pass neither the mask nor [`Sampler::accept`]. A mask that never allows
/// `<|im_end|>` or an end-of-generation token ends a turn only at
/// `max_tokens` or at the end of the context.
pub trait TokenMask {
    /// Set the logit of each token that may not come next to minus
    /// infinity. At least one token must stay allowed.
    fn mask(&self, logits: &mut [f32]);
    /// `token` was chosen, so the next [`TokenMask::mask`] follows it
    fn advance(&mut self, token: TokenId);
}

pub struct Sampler {
    pub cfg: SamplerConfig,
    rng: u64,
    mask: Option<Box<dyn TokenMask>>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let seed = if cfg.seed != 0 {
            cfg.seed
        } else {
            0x9E37_79B9_7F4A_7C15
        };
        Sampler {
            cfg,
            rng: seed | 1,
            mask: None,
        }
    }

    /// The random generator's state after the tokens sampled so far, to
    /// [`Sampler::resume`] from, e.g. with a [`crate::chat::Prefix`] of the
    /// same point
    pub fn rng_state(&self) -> u64 {
        self.rng
    }

    /// A sampler that goes on from `rng_state`, as [`Sampler::rng_state`]
    /// gave it: it samples what the sampler that gave it would have sampled.
    /// Fails with [`Error::ZeroRngState`] on 0, a state the generator never
    /// holds.
    pub fn resume(cfg: SamplerConfig, rng_state: u64) -> Result<Self, Error> {
        if rng_state == 0 {
            return Err(Error::ZeroRngState);
        }
        Ok(Sampler {
            cfg,
            rng: rng_state,
            mask: None,
        })
    }

    /// xorshift64*, uniform in [0, 1)
    pub fn uniform(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Apply `mask` before each choice from now on, until
    /// [`Sampler::clear_mask`]
    pub fn set_mask(&mut self, mask: Box<dyn TokenMask>) {
        self.mask = Some(mask);
    }

    /// Stop masking, and give the mask back
    pub fn clear_mask(&mut self) -> Option<Box<dyn TokenMask>> {
        self.mask.take()
    }

    /// Whether a mask is set. A chat drafts no tokens while it is, so that
    /// each step chooses one token through the mask.
    pub fn masked(&self) -> bool {
        self.mask.is_some()
    }

    /// `token` was chosen: the mask, if set, follows it
    pub fn accept(&mut self, token: TokenId) {
        if let Some(m) = &mut self.mask {
            m.advance(token);
        }
    }

    /// Whether the sampler picks the most likely token
    pub fn greedy(&self) -> bool {
        self.cfg.temperature <= 0.0
    }

    /// Pick the next token; `recent` is the generated history used by the
    /// penalties. Fails with [`Error::MaskAllowsNoToken`] when a mask is set
    /// and allows no token.
    pub fn sample(&mut self, logits: &mut [f32], recent: &[TokenId]) -> Result<TokenId, Error> {
        self.penalize(logits, recent);
        self.apply_mask(logits)?;
        if self.greedy() {
            return Ok(argmax(logits));
        }
        let cand = self.probs(enumerate(logits));
        Ok(self.draw(&cand))
    }

    /// The distribution [`Sampler::sample`] draws from, as (token,
    /// probability) pairs, best first, their sum 1 or a bit less. Not greedy.
    /// Fails with [`Error::MaskAllowsNoToken`] as [`Sampler::sample`] does.
    pub fn distribution(
        &self,
        logits: &mut [f32],
        recent: &[TokenId],
    ) -> Result<Vec<(TokenId, f32)>, Error> {
        self.penalize(logits, recent);
        self.apply_mask(logits)?;
        Ok(self.probs(enumerate(logits)))
    }

    /// The distribution of (token, logit) candidates, as
    /// [`Sampler::distribution`] without the penalties. Not greedy.
    pub fn distribution_of(&self, cand: Vec<(TokenId, f32)>) -> Vec<(TokenId, f32)> {
        self.probs(cand)
    }

    /// A token of (token, weight) pairs, with a chance proportional to its weight
    pub fn draw(&mut self, cand: &[(TokenId, f32)]) -> TokenId {
        let total: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = self.uniform() * total;
        for &(t, p) in cand {
            r -= p;
            if r <= 0.0 {
                return t;
            }
        }
        cand.last().unwrap().0
    }

    /// The mask, if set, over `logits`. A mask that allows no token is a
    /// fault of the mask, and no choice can follow it.
    fn apply_mask(&self, logits: &mut [f32]) -> Result<(), Error> {
        let Some(m) = &self.mask else {
            return Ok(());
        };
        m.mask(logits);
        if logits.iter().any(|l| *l > f32::NEG_INFINITY) {
            Ok(())
        } else {
            Err(Error::MaskAllowsNoToken)
        }
    }

    fn penalize(&self, logits: &mut [f32], recent: &[TokenId]) {
        let c = &self.cfg;
        if c.presence_penalty != 0.0 || c.repeat_penalty != 1.0 {
            let window = &recent[recent.len().saturating_sub(c.repeat_last_n)..];
            let mut seen = window.to_vec();
            seen.sort_unstable();
            seen.dedup();
            for &t in &seen {
                let l = &mut logits[t as usize];
                if c.repeat_penalty != 1.0 {
                    *l = if *l > 0.0 {
                        *l / c.repeat_penalty
                    } else {
                        *l * c.repeat_penalty
                    };
                }
                *l -= c.presence_penalty;
            }
        }
    }

    /// top-k, softmax with temperature, min-p, top-p of (token, logit) candidates
    fn probs(&self, mut cand: Vec<(TokenId, f32)>) -> Vec<(TokenId, f32)> {
        let c = &self.cfg;
        // candidates, best first
        let k = if c.top_k > 0 {
            c.top_k.min(cand.len())
        } else {
            cand.len()
        };
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // softmax with temperature
        let max = cand[0].1;
        let mut sum = 0.0;
        for (_, l) in cand.iter_mut() {
            *l = Float::exp((*l - max) / c.temperature);
            sum += *l;
        }
        for (_, p) in cand.iter_mut() {
            *p /= sum;
        }

        if c.min_p > 0.0 {
            let floor = cand[0].1 * c.min_p;
            cand.retain(|&(_, p)| p >= floor);
        }
        if c.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = cand.len();
            for (i, &(_, p)) in cand.iter().enumerate() {
                acc += p;
                if acc >= c.top_p {
                    keep = i + 1;
                    break;
                }
            }
            cand.truncate(keep);
        }
        cand
    }
}

fn enumerate(logits: &[f32]) -> Vec<(TokenId, f32)> {
    logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as TokenId, l))
        .collect()
}

pub fn argmax(v: &[f32]) -> TokenId {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as TokenId)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Shard;

    const SEED: u64 = 7;
    const KEY_TEMPERATURE: &str = "general.sampling.temp";
    const KEY_TOP_K: &str = "general.sampling.top_k";
    const KEY_TOP_P: &str = "general.sampling.top_p";
    /// GGUF metadata value types
    const TYPE_U32: u32 = 4;
    const TYPE_F32: u32 = 6;
    const GGUF_MAGIC: &[u8] = b"GGUF";
    const GGUF_VERSION: u32 = 3;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// A GGUF with no tensors and the metadata `kv`, as (key, type, little-endian value)
    fn gguf(kv: &[(&str, u32, [u8; 4])]) -> Gguf {
        let mut b = Vec::new();
        b.extend(GGUF_MAGIC);
        b.extend(GGUF_VERSION.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend((kv.len() as u64).to_le_bytes());
        for (key, ty, value) in kv {
            string(&mut b, key);
            b.extend(ty.to_le_bytes());
            b.extend(value);
        }
        let shard: Shard = Box::new(b);
        Gguf::from_shards(vec![shard]).unwrap()
    }

    fn no_overrides() -> SamplerOverrides {
        SamplerOverrides {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            presence_penalty: None,
            repeat_penalty: None,
        }
    }

    /// Tokens sampled before and after the state is taken
    const BEFORE: usize = 5;
    const AFTER: usize = 20;
    const VOCAB: usize = 50;

    /// Logits that leave many candidates, so that the draws decide the tokens
    fn logits() -> Vec<f32> {
        (0..VOCAB).map(|i| (i % 7) as f32 * 0.1).collect()
    }

    fn draws(s: &mut Sampler, n: usize) -> Vec<TokenId> {
        (0..n)
            .map(|_| s.sample(&mut logits(), &[]).expect("a token"))
            .collect()
    }

    #[test]
    fn resume_goes_on_from_the_state() {
        let cfg = SamplerConfig {
            top_k: 0,
            top_p: 1.0,
            seed: SEED,
            ..SamplerConfig::default()
        };
        let mut a = Sampler::new(cfg.clone());
        draws(&mut a, BEFORE);
        let mut b = Sampler::resume(cfg.clone(), a.rng_state()).unwrap();
        assert_eq!(draws(&mut a, AFTER), draws(&mut b, AFTER));
        // a new sampler with the same seed starts over instead
        let mut c = Sampler::new(cfg);
        let mut d = Sampler::new(c.cfg.clone());
        draws(&mut d, BEFORE);
        assert_ne!(draws(&mut c, AFTER), draws(&mut d, AFTER));
    }

    #[test]
    fn resume_refuses_state_zero() {
        assert!(matches!(
            Sampler::resume(SamplerConfig::default(), 0),
            Err(Error::ZeroRngState)
        ));
    }

    #[test]
    fn from_gguf_refuses_seed_zero() {
        let g = gguf(&[]);
        assert!(matches!(
            SamplerConfig::from_gguf(&g, no_overrides(), 0),
            Err(Error::ZeroSeed)
        ));
    }

    #[test]
    fn from_gguf_takes_stored_values_then_overrides() {
        let g = gguf(&[
            (KEY_TEMPERATURE, TYPE_F32, 0.6f32.to_le_bytes()),
            (KEY_TOP_K, TYPE_U32, 40u32.to_le_bytes()),
            (KEY_TOP_P, TYPE_F32, 0.8f32.to_le_bytes()),
        ]);
        let c = SamplerConfig::from_gguf(&g, no_overrides(), SEED).unwrap();
        assert_eq!((c.temperature, c.top_k, c.top_p), (0.6, 40, 0.8));
        assert_eq!(c.seed, SEED);

        let over = SamplerOverrides {
            temperature: Some(0.0),
            top_k: Some(5),
            min_p: Some(0.1),
            ..no_overrides()
        };
        let c = SamplerConfig::from_gguf(&g, over, SEED).unwrap();
        assert_eq!(
            (c.temperature, c.top_k, c.top_p, c.min_p),
            (0.0, 5, 0.8, 0.1)
        );
    }

    #[test]
    fn from_gguf_without_stored_values_keeps_the_defaults() {
        let c = SamplerConfig::from_gguf(&gguf(&[]), no_overrides(), SEED).unwrap();
        let d = SamplerConfig::default();
        assert_eq!(
            (c.temperature, c.top_k, c.top_p, c.min_p, c.repeat_last_n),
            (d.temperature, d.top_k, d.top_p, d.min_p, d.repeat_last_n)
        );
    }

    /// A mask that allows only the tokens of `allowed`, and counts the
    /// tokens it was told of
    struct Only {
        allowed: Vec<TokenId>,
        advanced: Vec<TokenId>,
    }

    impl TokenMask for Only {
        fn mask(&self, logits: &mut [f32]) {
            for (t, l) in logits.iter_mut().enumerate() {
                if !self.allowed.contains(&(t as TokenId)) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        fn advance(&mut self, token: TokenId) {
            self.advanced.push(token);
        }
    }

    /// A mask that allows only the token after the last one it was told of
    struct Next {
        last: TokenId,
    }

    impl TokenMask for Next {
        fn mask(&self, logits: &mut [f32]) {
            for (t, l) in logits.iter_mut().enumerate() {
                if t as TokenId != self.last + 1 {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        fn advance(&mut self, token: TokenId) {
            self.last = token;
        }
    }

    /// The token that the mask of the follow test starts after
    const FIRST: TokenId = 0;
    const ALLOWED: [TokenId; 2] = [2, 3];
    const DRAWS: usize = 200;
    const GREEDY: f32 = 0.0;

    fn only() -> Box<dyn TokenMask> {
        Box::new(Only {
            allowed: ALLOWED.to_vec(),
            advanced: Vec::new(),
        })
    }

    fn sampling() -> Sampler {
        Sampler::new(SamplerConfig {
            top_k: 0,
            seed: SEED,
            ..SamplerConfig::default()
        })
    }

    #[test]
    fn a_greedy_sampler_takes_the_likeliest_allowed_token() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: GREEDY,
            ..SamplerConfig::default()
        });
        s.set_mask(only());
        let mut l = logits();
        assert!(
            !ALLOWED.contains(&argmax(&l)),
            "the likeliest token is forbidden"
        );
        let t = s.sample(&mut l, &[]).expect("a token");
        assert!(ALLOWED.contains(&t), "{t} is allowed");
    }

    #[test]
    fn a_sampling_sampler_draws_only_allowed_tokens() {
        let mut s = sampling();
        s.set_mask(only());
        for t in draws(&mut s, DRAWS) {
            assert!(ALLOWED.contains(&t), "{t} is allowed");
        }
    }

    #[test]
    fn the_distribution_holds_only_allowed_tokens() {
        let mut s = sampling();
        s.set_mask(only());
        let d = s.distribution(&mut logits(), &[]).expect("a distribution");
        assert!(!d.is_empty());
        for (t, p) in d {
            assert!(ALLOWED.contains(&t) || p == 0.0, "{t} has p {p}");
        }
    }

    #[test]
    fn the_mask_follows_each_accepted_token() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: GREEDY,
            ..SamplerConfig::default()
        });
        s.set_mask(Box::new(Next { last: FIRST }));
        let mut chosen = Vec::new();
        for _ in 0..ALLOWED.len() {
            let t = s.sample(&mut logits(), &[]).expect("a token");
            s.accept(t);
            chosen.push(t);
        }
        assert_eq!(chosen, vec![1, 2]);
    }

    #[test]
    fn with_no_mask_the_draws_are_the_same() {
        let mut plain = sampling();
        let mut cleared = sampling();
        cleared.set_mask(only());
        assert!(cleared.masked());
        assert!(cleared.clear_mask().is_some());
        assert!(!cleared.masked());
        assert_eq!(draws(&mut plain, DRAWS), draws(&mut cleared, DRAWS));
    }

    #[test]
    fn a_mask_that_allows_no_token_is_refused_by_name() {
        let mut s = sampling();
        s.set_mask(Box::new(Only {
            allowed: Vec::new(),
            advanced: Vec::new(),
        }));
        assert!(matches!(
            s.sample(&mut logits(), &[]),
            Err(Error::MaskAllowsNoToken)
        ));
        assert!(matches!(
            s.distribution(&mut logits(), &[]),
            Err(Error::MaskAllowsNoToken)
        ));
    }
}
