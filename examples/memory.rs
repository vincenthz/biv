//! The memory a model and its sessions take: estimated from the GGUF alone,
//! then checked against sessions made on each backend.
//! `memory <gguf> [context...]`
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use biv::engine::Engine;
use biv::memory::{Backend, Footprint, SessionMemory};
use biv::model::Model;
use biv::sampler::argmax;

/// The system allocator, counting the bytes live and their peak: what the
/// CPU's passes allocate, against [`SessionMemory::transient`]
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        grow(l.size());
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        grow(l.size());
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        match new >= l.size() {
            true => grow(new - l.size()),
            false => _ = LIVE.fetch_sub(l.size() - new, Relaxed),
        }
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The peak of the bytes allocated by `f` beyond those live before
fn peak<R>(f: impl FnOnce() -> R) -> (usize, R) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let r = f();
    (PEAK.load(Relaxed) - base, r)
}

fn mb(bytes: usize) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

fn line(m: &SessionMemory) -> String {
    format!(
        "allocated {:>10}: kv {:>10}, recurrent {:>9}, draft head {:>9}, scratch {:>9}; transient {:>9}",
        mb(m.allocated()),
        mb(m.kv),
        mb(m.recurrent),
        mb(m.draft_head),
        mb(m.scratch),
        mb(m.transient)
    )
}

/// Make a session, fill it with `n` tokens, and compare what it reports
/// with the estimate
fn check(name: &str, engine: &dyn Engine, estimate: SessionMemory, context: usize, draft: usize) {
    let before = engine.device_memory();
    let mut s = engine.new_session(context, draft).unwrap();
    let measured = s.memory();
    let after = engine.device_memory();
    let n = 300.min(context - 8);
    s.prefill(&vec![9707; n]);
    // a pass of the engine's batch size, and verify passes: on the CPU, what
    // they allocate is the transient
    let room = context.saturating_sub(n + 8 + 4 * (draft + 1));
    let batch = engine.batch_size().min(room).max(1);
    let (mut transient, logits) = peak(|| s.forward(&vec![9707; batch]));
    let mut next = argmax(&logits) as u32;
    for _ in 0..4 {
        let drafts = s.draft(next, draft, &mut |l| l.argmax());
        let mut tokens = vec![next];
        tokens.extend(&drafts);
        let (bytes, logits) = peak(|| {
            let logits = s.verify(&tokens);
            s.keep(1);
            logits
        });
        transient = transient.max(bytes);
        next = argmax(&logits[0]) as u32;
    }
    let same = if measured == estimate {
        "same"
    } else {
        "DIFFERENT"
    };
    println!("  {name} ctx {context} draft {draft}: {same}");
    println!("    estimated {}", line(&estimate));
    if measured != estimate {
        println!("    measured  {}", line(&measured));
    }
    println!(
        "    {} tokens: kv in use {}",
        s.n_past(),
        mb(measured.kv_used(s.n_past()))
    );
    if name == "cpu" {
        println!(
            "    passes allocated at most {} of the estimated transient {}{}",
            mb(transient),
            mb(estimate.transient),
            if transient > estimate.transient {
                ": MORE"
            } else {
                ""
            }
        );
    }
    if let (Some(b), Some(a)) = (before, after) {
        let up = a.allocated - b.allocated;
        println!(
            "    the device's own count went up by {} ({}), now {} of {}",
            mb(up),
            match up as i64 - measured.allocated() as i64 {
                0 => "as estimated".to_string(),
                d => format!("{d:+} bytes from the estimate"),
            },
            mb(a.allocated),
            mb(a.working_set)
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: memory <gguf> [context...]");
    let mut contexts: Vec<usize> = args[1..].iter().map(|s| s.parse().unwrap()).collect();
    if contexts.is_empty() {
        contexts = vec![4096, 65536];
    }

    let g = biv_cli::open_gguf(path).unwrap();
    let f = Footprint::from_gguf(&g).unwrap();
    let cpu = Backend::Cpu { batch: 64 };
    println!(
        "model: weights {}, mtp {}, params {}",
        mb(f.model(cpu).weights),
        mb(f.model(cpu).mtp_weights),
        mb(f.model(cpu).params)
    );
    for budget in [16e9, 32e9, 64e9] {
        match f.max_context(cpu, 1, budget as usize) {
            Some(n) => println!(
                "  fits in {:.0} GB with a context of at most {n} tokens (cpu, drafting)",
                budget / 1e9
            ),
            None => println!("  doesn't fit in {:.0} GB", budget / 1e9),
        }
    }

    let model = Model::load(&g).unwrap();
    assert_eq!(model.model_memory(), f.model(cpu));
    println!("cpu: the loaded model has the estimated memory");
    for &c in &contexts {
        for draft in [0, 1] {
            check("cpu", &model, f.session(cpu, c, draft), c, draft);
        }
    }

    #[cfg(target_os = "macos")]
    {
        let metal = Backend::Metal { max_batch: 256 };
        let before = biv::metal::MetalEngine::new(&model, 256)
            .unwrap()
            .device_memory()
            .unwrap();
        println!(
            "metal: an engine alone, {} allocated on the device",
            mb(before.allocated)
        );
        let e = biv::metal::MetalEngine::new(&model, 256).unwrap();
        assert_eq!(e.model_memory(), f.model(metal));
        println!("metal: the engine has the estimated model memory");
        for &c in &contexts {
            for draft in [0, 4] {
                check("metal", &e, f.session(metal, c, draft), c, draft);
            }
        }
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        let cuda = Backend::Cuda { max_batch: 256 };
        let e = biv::cuda::CudaEngine::new(&model, 256).unwrap();
        assert_eq!(e.model_memory(), f.model(cuda));
        println!("cuda: the engine has the estimated model memory");
        for &c in &contexts {
            for draft in [0, 4] {
                check("cuda", &e, f.session(cuda, c, draft), c, draft);
            }
        }
    }
}
