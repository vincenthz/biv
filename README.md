# biv — Brain In a Vat

Pure Rust chat inference for **Qwen3.8** GGUF models (llama.cpp architectures
`qwen35` and `qwen35moe`), on Apple GPUs (Metal) or the CPU (aarch64 NEON, x86-64 AVX2 or AVX-VNNI). No
C/C++ code and no bindings: the GGUF loader, the quantized kernels (NEON `sdot`,
AVX2 `maddubs`, VNNI `vpdpbusd`, Metal shaders), the tokenizer, and the model graph are all in Rust
or MSL. It needs Rust 1.98 or later, for the NEON dot product intrinsics.

```sh
cargo build --release
./target/release/biv                      # hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1
./target/release/biv -m hf.co/unsloth/Qwen3.8-27B-GGUF:BF16 --no-think
./target/release/biv -m /path/to/model.gguf -p "one-shot question"
./target/release/biv --cpu                # skip the GPU
```

On x86-64 the AVX2 + FMA + F16C kernels (Haswell, Zen and later) are picked at run
time, so a default build runs on any x86-64 CPU and falls back to scalar code
without AVX2. On CPUs with AVX-VNNI (Alder Lake, Zen 5 and later) the int8
products use `vpdpbusd` instead of `maddubs` + `madd`. The two give the same
results bit for bit. `RUSTFLAGS="-C target-cpu=native"` additionally lets the compiler
use AVX2 in the non-kernel code (norms, softmax, sampling).

On macOS the Metal backend is the default. Its shaders are compiled from source
at startup, which takes about 0.5s, and the weights are used in place from the
mmap'd GGUF with no copy.

Models are looked up in the Hugging Face cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`,
`~/.cache/huggingface/hub`). That's where skelm downloads them, and it uses the
same `hf.co/<owner>/<repo>:<quant>` names. Split GGUFs (`-00001-of-0000N`) are loaded
as one model.

The `hf` command (`brew install hf`, or `pip install -U huggingface_hub`) also downloads to that cache:

```sh
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_1.gguf   # the default, 17.5 GB
hf download unsloth/Qwen3.8-27B-GGUF --include 'BF16/*'      # :BF16, two shards, 55 GB
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf   # :Q8_0, 29 GB
```

Other quants are files of the same repo (`hf download unsloth/Qwen3.8-27B-GGUF
--dry-run` lists them); `-m hf.co/unsloth/Qwen3.8-27B-GGUF:<quant>` picks one.
The kernels cover F32, F16, BF16, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K and Q6_K; files
with other types (IQ*, Q2_K, Q3_K, Q5_0, Q5_1) don't load.

Qwen3.5 dense models have the same architecture. Qwen3.5 4B runs on Metal and the
CPU, at about 45 tokens/s on an M2 Max (Q8_0):

```sh
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf     # 4.5 GB
./target/release/biv -m hf.co/unsloth/Qwen3.5-4B-GGUF:Q8_0
```

Its GGUF has no MTP head, so it decodes without drafting (`--lookup` still works).

The mixture-of-experts models (`qwen35moe`: 256 experts, 8 per token, plus a shared
expert) also run on both backends. Qwen3.6 35B-A3B (Q8_0) decodes at about 45
tokens/s on an M2 Max, and 60 to 90 with its MTP head drafting:

```sh
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-Q8_0.gguf   # 37.8 GB
./target/release/biv -m hf.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q8_0
```

Decoding is speculative by default: the model's MTP head drafts the next tokens
(4 on Metal, 1 on the CPU), and the trunk checks them in the same pass as the
current one (`--draft <n>` sets how many tokens are drafted, `0` turns it off).
`--lookup <n>` also proposes up to `n` tokens copied from earlier in the
conversation when the answer repeats it, which speeds up quoting and editing
code. The output follows the same distribution as without drafting: greedy, it
is the same text.

In the chat, `/reset` clears the conversation, `/think` and `/nothink` toggle
reasoning for the next answers, `/stats` shows context and memory use, Ctrl-C
stops an answer, and a trailing `\` continues a message on the next line. Run
`--help` for the sampling and context options. Sampling defaults come from the GGUF
(`general.sampling.*`).

## Fingerprint

`biv fingerprint [options]` prints a fingerprint of the inference stack, to
detect any change in what it generates. It takes the chat's options (`-m`,
`--cpu`, `-b`, `-d`, `--temp`, `--seed`…) and writes `key=value` lines (with
`\\`, `\n` and `\r` escaped in values), in two parts:

- the stack, what should decide the output besides the code: the model (with
  its SHA-256 when known), the OS release (its libm, and on macOS the compiler
  of the Metal shaders), the backend (CPU kernels or GPU, KV cache type, batch
  size), and the generation options. `stack_id` hashes it.
- `probe.*`, a fixed run on that stack: a prompt in passes of 64 and 16 tokens,
  12 tokens decoded greedily one per pass, the same from the restored prompt
  with drafts verified in one pass, and a sampled chat turn. Each pass records
  a hash of all its logits' bits (`.logits`) and the top 8 as `id:value`
  (`.top`). `output_id` hashes it. It takes about 5s on Metal and 30s on the CPU.

The code itself is not hashed: a change to it shows in the logits if it changes
the output, and not otherwise. `info.*` lines (CPU, threads, model path) don't
count. The hashes are 64-bit FNV-1a, to tell that something changed, not to
resist forgery.

The model's SHA-256 is given by the caller of `fingerprint::stack`, as a
`[u8; 32]`, or left out; for a split model, it is the XOR of the shards'.
`biv` takes it from the Hugging Face cache, which names each file after its
SHA-256; a change to the weights shows in the logits either way.

```sh
biv fingerprint > ref.txt
biv fingerprint --check ref.txt   # exits with 1 if anything differs
```

`--check` runs the probe again and reports the stack lines that changed, and for
the output: how many logits are identical, the largest difference of the top
logits, the first token chosen differently, and the chat text. The stack says
what changed, the probe whether the output did: a new OS can change the shaders'
code with no line changing, and a line can change with no effect on the output.

Runs repeat bit for bit: the Metal kernels use no atomics, and on the CPU each
output row is computed whole by one thread, so the thread count changes nothing
(4 and 12 threads give the same bits). How tokens are batched does change the
bits (`-b 16` splits the probe's passes of 64), and so does the backend: CPU and
Metal logits differ by up to about 0.4 on this probe.

## Benchmark

`biv bench [options] [--exact]` measures greedy decode with 0, 1 and 4 MTP
drafts per step, and 1 then 4 chats sharing passes ([batching](#batching)). It
takes the chat's `-m`, `--cpu`, `-t` and `-b`, and runs in 45s for the 27B on an
M2 Max (18s for the 35B-A3B). `--exact` takes more answers, 0 to 6 drafts, a
sampled row (temperature 1, seed 1), and 1 to 8 chats with 0 or 1 draft, in under
3 minutes. The rows give the decode rate and the speedup over the first row;
with drafts, the share of drafts kept and the tokens per step; with chats, the
rate of all of them together, prompts included.

Greedy, drafting must not change the text: `same text` counts the answers equal
to those without drafts, and any difference makes `bench` exit with 1. Batched
answers can differ from the first row's by rounding (see [batching](#batching)),
so `same answers` only counts them.

```
$ biv bench --exact
exact: 3 prompts x 128 tokens, greedy
drafts          decode  speedup  accepted  tok/step  same text
0           18.4 tok/s    1.00x         -      1.00  -
1           26.5 tok/s    1.44x       86%      1.85  3/3
2           24.9 tok/s    1.35x       75%      2.48  3/3
4           28.5 tok/s    1.54x       56%      3.20  3/3
6           28.2 tok/s    1.53x       43%      3.49  3/3
4 sampled   26.8 tok/s    1.45x       51%      3.00  -

batching: 8 prompts x 40 tokens, greedy, each chat taking the next prompt when done
chats  drafts         tok/s  speedup  same answers
    1       0    16.1 tok/s    1.00x  -
    2       0    24.6 tok/s    1.53x  8/8
    4       0    32.9 tok/s    2.04x  8/8
    8       0    53.1 tok/s    3.30x  8/8
    4       1    37.9 tok/s    2.36x  8/8
    8       1    43.4 tok/s    2.70x  8/8
```

A row is one run, so rates move by a few percent between runs (more when the
machine is warm or busy).

## Layout

Two crates: `biv` (`engine/`) is the inference engine, with no I/O, and
`biv-cli` (the root, which builds the `biv` executable) is the command line
around it: files, mmap, the Hugging Face cache, the terminal.

Without its default features the engine is `no_std` + `alloc`: the model is
parsed from the GGUF's bytes (`Gguf::from_shards`), compute runs on one
thread and float functions come from `libm`. The x86 kernels are picked at run
time either way (cpuid). Its features:

| feature | brings |
|---|---|
| `std` (default) | std's float functions |
| `rayon` (default) | multithreaded CPU compute |
| `metal` | the Metal backend (macOS) |

| file | role |
|---|---|
| `engine/src/gguf.rs` | GGUF parser over in-memory shards, split files |
| `engine/src/quant.rs` | block formats (dequant for Q2_K…Q8_0, F16, BF16), int8 activations, NEON, AVX2 and AVX-VNNI dot kernels with scalar references |
| `engine/src/ops.rs` | matmul (threaded with rayon), norms, rope, softmax |
| `engine/src/par.rs` | rayon's parallel iterators, or sequential ones of the same names |
| `engine/src/math.rs` | libm float functions without `std` |
| `engine/src/model.rs` | the `qwen35` / `qwen35moe` graph: port of llama.cpp `src/models/qwen35.cpp`, `qwen35moe.cpp` and `delta-net-base.cpp`; expert routing; MTP drafting |
| `engine/src/memory.rs` | memory of a model and its sessions: estimated from the GGUF alone (`Footprint`), or reported by engines and sessions |
| `engine/src/engine.rs` | `Engine` / `Session` / `Batch` traits shared by the backends (draft / verify / keep for speculative decoding); CPU implementation; sequential batches |
| `engine/src/batch.rs` | continuous batching: the scheduler, and the `Engine` handle whose sessions it serves |
| `engine/src/metal/mod.rs` | Metal backend: no-copy weight buffers, per-sequence KV/recurrent state, one command buffer per forward pass of one or more sequences |
| `engine/src/metal/kernels.metal` | matvec / simdgroup matmul for each quant type (from ggml-metal), fused norm, rope, split-K (decode) and flash (prefill) attention, conv and gated delta-net kernels, expert routing and the matmul / matvec over each expert's tokens |
| `engine/src/tokenizer.rs` | byte-level BPE; the `qwen35` pre-tokenizer regex, matched by hand |
| `engine/src/chat.rs` | chat template (ChatML with `<think>`), incremental multi-turn state, the speculative decoding loop |
| `engine/src/transcript.rs` | a conversation as token ids, and its bytes |
| `engine/src/sampler.rs` | temperature / top-k / top-p / min-p / penalties |
| `src/lib.rs` | GGUF files: shard discovery, mmap |
| `src/hf.rs` | Hugging Face cache lookup |
| `src/fingerprint.rs` | `biv fingerprint`: the stack, the probe, the comparison |
| `src/bench.rs` | `biv bench`: decode rates with drafts and with batched chats |
| `src/main.rs` | the chat CLI |
| `src/bin/gguf-dump.rs` | print a GGUF's metadata and tensors |

## Model notes

64 layers. Every 4th layer is gated full attention: GQA 24/4 heads of 256,
q/k RMS norm, NeoX rope on the first 64 dims (IMROPE reduces to this for
text), and a sigmoid output gate packed with Q. The other layers are Gated DeltaNet:
a causal conv (k=4), L2-normed q/k, and a delta rule over a 128×128 state per value
head (48 value heads, 16 key heads in tiled order). Each layer has a SwiGLU FFN.
Conversion has already folded `+1` into the norm weights and stores `A` as
`-exp(A_log)`.

The MTP block (`blk.64`) drafts tokens. Its row `p` is
`eh_proj · [enorm(embedding of token p+1) ‖ hnorm(normalized hidden state of token p)]`,
then a gated attention layer over its own history of rows and `shared_head_norm`;
the trunk's output head reads the next token. Every pass of the trunk also enters
its completed rows into that history. A second draft chains on the normalized
output of the first.

Speculative decoding (`Session::draft` / `verify` / `keep`): the trunk runs the
sampled token and the drafts in one pass, with logits for each. Greedy, the
drafts are the MTP head's likeliest tokens, kept while they equal the trunk's.
Sampling, each draft is drawn from the head's own distribution q (the sampler's
settings applied to its logits) and kept with a chance of p/q, p being the
trunk's distribution; otherwise the token is drawn from max(p − q, 0)
(speculative sampling). Either way the tokens follow p, but more drafts hold
than with greedy drafts: 4 drafts decode 9% faster. Rejected tokens are dropped
by moving `n_past` back for the KV caches; the delta net states have one copy
per drafted token plus one, the verify pass writes the state after each token to
the next copy, and `keep` only changes which copy is current. Those copies
(151 MB per token) cost under 1 ms of a 5-token verify: the writes overlap with
the rest of the pass.

The MTP head drafts over part of the vocabulary: the 65,536 lowest token ids
(the BPE's earliest merges, about 90% of the tokens of answers in English and
code) and the
tokens the sequence has processed, up to 16,384 more, whose rows of the output
head are copied to a buffer of their own. A draft takes 1.7 ms instead of 4.4.
When fewer than 90% of the last 128 tokens had a row (Chinese text is mostly
past the base ids), the head drafts over the whole vocabulary.

Lookup (`ChatOptions::lookup`): the continuation of the latest longest match of
the last tokens in the conversation (prompt and answers) is proposed instead of
the MTP head's drafts, if the match has 8 tokens or more, or if it has 3 or more
and the head drafts the same first 2 tokens. It is cut to 7 tokens unless the
last two continuations were all right, a copy being under way.

The chat template keeps past reasoning in the history by default, so each turn
only appends tokens. The KV cache and the recurrent states are kept between turns,
and only the new message is processed.

Tools (`ChatOptions::tools`, JSON function definitions) go in the system block
as the chat template writes them. The model's calls come as `Event::ToolCall`
(function name and parameters, as the model wrote them), and their results go
back with `Chat::send_tool_responses`.

A conversation can be saved between turns (`Chat::save`) and restored in any
chat on the same engine (`Chat::restore`), which is how a system prompt is
cached: saved before the first message, the prefix holds the processed system
block, tools included. The recurrent states can't be rewound, so a prefix is a copy of the
whole sequence state: the KV cache of its tokens plus the delta net states,
about 150 MB for this model, plus 64 KB per token on Metal (128 KB on the CPU,
whose caches are f32).

A conversation can also be kept as its token ids alone (`Chat::transcript`, a
`Transcript`: the tokens, how many of the last ones are still to be processed,
and the generated ones for the sampler's penalties), 4 bytes per token, and
restored in any chat on a model with the same vocabulary
(`Chat::restore_transcript`), which processes the tokens again. `Chat::n_tokens`
gives its length without copying it. `Transcript::to_bytes` and `from_bytes`
(`engine/src/transcript.rs`, where the layout is) write it with a 20-byte header,
its token ids in one of two encodings: `Full`, a u32 per token, or `Compact`, a
u16 per token below 0xffff and, from 0xffff, the escape 0xffff followed by a u32
of the id minus 0xffff (0xffff is `ff ff 00 00 00 00`).

## Batching

Several chats can share an engine, each on a thread of its own, with
continuous batching: a pass takes the tokens of every chat ready for one, so
the weights are read once for all of them. `batch::serve(engine, &opts, f)` runs
a scheduler on the calling thread and gives `f`, run on another thread, a
`batch::Handle`. The handle is an `Engine`: chats made on it work as on any
engine (drafts, lookup, tools, prefixes), their sessions' calls going to the
scheduler.

Between passes the scheduler takes the requests that came. Drafts, `keep` and
snapshots are served at once; tokens wait for the next pass, which takes the
decode steps first (forward and verify, in their order), then pieces of
prompts to fill the rest of the pass (`BatchOptions::tokens`, the engine's
batch size by default; `BatchOptions::logits` caps the rows of logits). Chats
join and leave between any two passes. After a pass the scheduler waits up to
`BatchOptions::wait` (5 ms) for the chats it answered, so that they make the
next pass together.

Underneath, `Engine::new_batch` makes a `Batch` of sequences whose
`Batch::run` takes the tokens of several in one pass. On Metal the
sequences share one set of activation buffers. The matmuls, norms and
feed-forwards run once over all the rows, and the attention, the delta nets and
the MTP head per sequence, on its rows. A session alone is a pass of one sequence, the
same kernels as before: the fingerprint doesn't change. The CPU backend has no
batched pass yet: its batch runs the sequences one after another.

A sequence's logits depend on the other tokens of its passes, as on how
a prompt is split: the matmul kernels are chosen by the rows of the whole
pass. On Qwen3.6 35B-A3B (Q8_0), decode steps of 2 to 8 sequences of one
token each gave the logits of each alone, bit for bit; a prompt piece in a pass
with other sequences' tokens doesn't. On the dense models the 8 answers below were the
same text as alone. On Qwen3.6 35B-A3B 5 of 8 parted after a few hundred
characters, the experts' router turning small differences into other experts.

Throughput of 8 greedy answers of 128 tokens without thinking, `j` chats at a
time, each taking the next prompt when done (`examples/batch.rs`, M2 Max), in
tokens per second for all the chats together:

| | alone | 2 chats | 4 chats | 8 chats |
|---|---|---|---|---|
| Qwen3.5 4B Q8_0 | 43.9 | 81.4 | 145.0 | 242.5 (5.5×) |
| Qwen3.8 27B Q4_1, no draft | 17.4 | 24.4 | 35.8 | 61.9 (3.6×) |
| Qwen3.8 27B Q4_1, 1 draft | 21.8 | | 49.4 | 54.4 |
| Qwen3.8 27B Q4_1, 4 drafts | 23.0 | 26.1 | 31.8 | 36.3 |
| Qwen3.6 35B-A3B Q8_0, no draft | 43.0 | | 104.3 | 143.4 (3.3×) |
| Qwen3.6 35B-A3B Q8_0, 4 drafts | 59.2 | | | 91.1 |

Drafting gains less in a batch: 8 verify passes of 5 tokens make 40 rows, in
the `mm` kernels, and each chat's drafts are passes of their own. With many
chats, `ChatOptions::draft` at 0 or 1 is faster.

## Verification

- Kernel tests: `cargo test --workspace --release` compares every SIMD kernel
  with its dequantize-then-dot reference. The x86-64 kernels can be tested on
  Apple silicon through Rosetta. It runs AVX2 code but only advertises it when
  asked: `ROSETTA_ADVERTISE_AVX=1 cargo test --workspace --release --target
  x86_64-apple-darwin` (without the variable, the same command tests the scalar
  fallback). Rosetta has no AVX-VNNI. On a CPU with it, the tests also check
  that the VNNI kernels give the AVX2 kernels' results bit for bit. The x86-64 build under Rosetta also generates the same greedy text
  as the native build. Rosetta timings are not representative of x86 hardware.
- Tokenization of the test prompts matches `llama-tokenize` exactly.
- Greedy generation matches llama.cpp (`llama-simple -ngl 0`) token for token
  on the same GGUF.
- `examples/consistency.rs`: the logits don't depend on how a prompt is split
  into forward passes, so state carry across chunks is exact.
- `examples/speculative.rs <gguf> [--cpu] [-n tokens] [-d draft]`: greedy
  answers with and without drafts are the same text, on Metal with 1 to 7
  drafts and on the CPU. Verify logits differ from those of single-token passes
  by at most 2e-4 below 2k positions, 7e-4 above (f16 queries in the split
  flash kernel).
- `examples/lookup.rs <gguf> [--cpu] [-n tokens] [-l lookup]`: greedy answers
  that copy from the prompt are the same text with and without lookup, with
  the engine's drafts and without, alone and in a batch; on a model without
  MTP head too, whose sessions keep the room to verify the proposals.
- `examples/compare.rs <gguf> [repeat]`: Metal vs CPU logits on the same prompt,
  then decode steps. The top-1 token matches at every step. Max logit differences
  are 0.2–0.4 on short prompts and up to a few units in the tail at ~700 tokens
  (f16 KV cache, half-precision matmul tiles).
- `examples/prefix.rs <gguf> [--cpu] [-n tokens]`: a chat restored from a saved
  system prompt or first turn gives the same greedy answer as the chat that
  saved it, on Metal and on the CPU, with or without drafts; so does a chat
  restored from the first turn's transcript, processed again.
- `examples/tools.rs <gguf> [--cpu]`: the first prompt with tools has the tokens
  of the template text rendered by hand; the model calls both tools of a two-part
  question, answers from their results, and answers directly when no tool
  applies, with and without thinking; a restored prefix with tools makes the
  same calls.
- `examples/agent.rs <gguf> [--cpu] [--think] [task]`: an agent loop whose tools
  do I/O in a temporary workspace: list a directory, read a file, wait for a
  file that a simulated build writes after 10 s. The model waits for the build,
  reads what it needs and answers; paths outside the workspace are refused.
- `examples/batch.rs <gguf> [--cpu] [-n tokens] [-d draft] [-j chats...]`:
  sequences run in one pass have the top-1 token they have alone; a prefix
  saved and restored through the scheduler gives the answer of the engine
  itself; then the throughput above, and how many answers are the text of the
  same chat alone. Decoded together after prompts processed alone, sequences
  have the logits they have alone, bit for bit, on Qwen3.6 35B-A3B too.
- `examples/attention.rs <gguf> [tokens]`: Metal logits after prefilling the
  same tokens in full batches (flash attention) and in passes of 8 (split-K,
  then split flash attention past 2k positions) have the same top-1 token; over
  4000 tokens they differ by at most 0.06, and by 0.1–0.3 in the next decode
  steps.
- The precise norms, softplus and decay (c2ea0d3), and the sigmoid with
  finite scores (0fd9232), measured on an Apple M3 Max on 2026-11-13 from
  20:14:44 to 20:20:02. The two binaries come from `cargo install --git` of
  this repository: A at 54d514d (`biv` sha256 c3708561a176a7c1...,
  `examples/logits` f90b35ec6fc906a1...) and B at 0fd9232 (`biv`
  4d1592f3db7704ec..., `examples/logits` 87ebb536b8e66986...). Before each
  run, the GPU rests 20 s and the A command line probes it with the
  prompt below, until a probe decodes at 50.5 tok/s or more, at most 15
  probes. All workspace tests pass on Metal at 0fd9232. The command line on
  Qwen3.8 27B Q4_1 (`--no-think -n 128 --seed 42 -p "Name three prime numbers above
  100."`) ran in the order A, B, B, A. It gave 50.54, 50.45, 50.95 and
  50.58 tok/s, so A 50.56 and B 50.70 on average, and all four outputs had
  the same bytes. The difference is inside the spread of the runs, so the
  change has no measurable cost on this model. On four small GGUF files
  with exact one-hot answers (178, 3149, 11194 and 11728 rows), every
  row of `examples/logits --engine metal` was exact with A and with B, in
  the same time within 0.6 s.
- The CPU and Metal greedy answers can differ. On an Apple M3 Max on
  2026-11-13, the plain command line on Qwen3.8 27B Q4_1 (`--no-think -n 128
  --seed 42 -p "Name three prime numbers above 100."`) gave the same first
  78 bytes on both engines, and then the first token of line 7 differed:
  `### Verification` on the CPU (16 threads, 1 draft) and
  `**Verification:**` on Metal (4 drafts). The CPU text came from biv
  0fd9232 and from the CPU math of 1744aa7, with the same bytes, so the
  CPU math of 1744aa7 is not the cause. The cause is not measured: the
  Metal logits differ from the CPU logits by the f16 KV cache and the
  half-precision matmul tiles (see `examples/compare.rs` above), and this
  token can be a near tie.

## Performance (M2 Max 38-core GPU, Q4_1)

| | biv Metal | llama.cpp Metal | biv CPU (12 threads) | llama.cpp CPU |
|---|---|---|---|---|
| decode, 4 drafts | 28 tok/s greedy, 27–28 sampled | | | |
| decode, 1 draft | 26 tok/s greedy, 26 sampled | | ~5.7 tok/s | |
| decode, no draft | 18–19 tok/s | 17.6 tok/s | ~5.3 tok/s | 4.5 tok/s |
| prompt (~20 tok) | 48 tok/s | 36.8 tok/s | ~9–10 tok/s | 5.0 tok/s |
| prompt (1541 tok) | 149 tok/s | | | |

BF16 on Metal decodes at about 7.4 tok/s without drafts, which means reading
about 400 GB/s of weights, close to the memory bandwidth, and 10.6–11.1 tok/s
with one.

The decode rates with drafts are over 8 answers of 300 tokens without thinking
(4 prompts, 2 seeds; sampled at the model's settings, temperature 1, top-k 20,
top-p 0.95). Speculative decoding (`examples/speculative.rs`, greedy, 256
tokens): the MTP head's first draft is right 86–96% of the time (about 66%
sampling at temperature 1). On Metal a step is the drafts (1.7 ms each) and a
verify pass, whose cost grows slowly with its tokens (1k positions):

| tokens verified | 1 | 2 | 4 | 5 | 8 | 16 |
|---|---|---|---|---|---|---|
| verify pass | 55 ms | 65 ms | 95 ms | 96 ms | 101 ms | 184 ms |

Up to 3 tokens, the matvec kernels run a threadgroup per token and read the
weights for the others mostly from cache. From 4 to 16, a matmul kernel on
simdgroup matrices (`mmv`) takes 16 weight rows and 8 tokens per threadgroup,
dequantizing each weight once for the 8 (before it, a 5-token pass cost 4.7×
a decode step). 4 drafts are faster than 3 or 6, and as fast as 5. On the CPU a
2-token pass costs 1.45× a decode step, which leaves a small gain for one draft.

Lookup, on an answer that copies most of a file with a function renamed (4
drafts, greedy): 46 tok/s without, 73 tok/s with `--lookup 7`, 77 with
`--lookup 15`. On ordinary answers it rarely fires and changes the rate by less
than the noise, but each token of a step needs a copy of the delta net states,
so it is off by default.

Verify passes at 2k positions or more use a split variant of the flash
attention kernel: its threadgroups take 16 (token, head) rows of a KV head and
one of up to 64 ranges of positions, so each K/V tile is read once for all the
tokens of the pass rather than once per token. A 5-token verify at 32k takes
109 ms instead of 125, an 8-token one 120 ms instead of 145.

Long contexts (`examples/longctx.rs <gguf> [depth...]`, Metal):

| context | 1k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| decode | 18.1 tok/s | 17.8 | 17.5 | 17.0 | 16.2 |
| prefill up to it | 184 tok/s | 183 | 177 | 148 | 137 |

Decode attention is split-K ("flash decoding"). Each threadgroup takes a KV head with
its 6 query heads over one range of positions, so K/V is read once per group, and
at most 64 ranges are merged by a second kernel. Each simdgroup holds two query
heads; with three or more, q and the accumulators spill out of registers and the
kernel gets 2–6× slower. At 32k the attention of a layer reads 128 MB of f16 KV
cache in about 0.45 ms (~300 GB/s).

Batches of 32 tokens or more use a flash attention kernel on simdgroup matrices
instead: a threadgroup takes 16 query rows (token, head pairs of a KV head) and
walks the positions 64 at a time, Q·Kᵀ, online softmax, then P·V, with each
simdgroup holding a quarter of the head dimension of the output in registers.
It runs at about 6.2 TFLOPS from 2k to 32k positions, 2.4× the split-K kernel
over a 256-token batch, which took about 37% of prefill time at 16–32k. Its
loops over register arrays must be fully unrolled: left to the compiler, the
arrays went to memory and the kernel was slower than split-K.

## Next steps

- Ctrl-C during a long prefill.
- i8mm (`smmla`) CPU prompt kernel; f16 KV cache on CPU.
- AVX-512 (VNNI) kernels for x86-64, 64 bytes per instruction. They are not
  testable under Rosetta, so they need real hardware. AVX-VNNI alone gains
  little: it saves one instruction per 32 products, while the per-block float
  scaling dominates.
- Draft on the GPU without a round trip per token.
- A stronger drafter: DFlash2 (a 5-layer model reading 5 of the trunk's layers,
  drafting a 7-token block in one pass) keeps 3.1–3.3 tokens per step against
  2.8–2.9 for the MTP head in HyperQwen's measurements, at 1.2 GB read per step
  once in int4. Or verify trees of drafts (several candidates per position, as
  in EAGLE), which needs a copy of the delta net states per branch.
- A 2-row CPU dot kernel, for speculative decoding on the CPU.
- Batched passes on the CPU: `Model` carrying several `State`s through one pass.
- Batched drafting: the MTP head's steps of all the sequences of a batch in one
  pass, and fewer drafts as more chats share the passes.
- One attention dispatch for all the sequences of a pass (per-row positions
  and caches) rather than one per sequence.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
