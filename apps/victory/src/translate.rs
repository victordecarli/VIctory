//! The offline batch: every shader found goes through the translator, and
//! what it cannot handle yet is tallied by reason, so the most common gaps
//! show what to implement next.

mod tally;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::emit::host::{self, Host};
use victory_shader::emit::{self, validator::Validator};
use victory_shader::structure::{self, Atom, LaneExit};
use victory_shader::{
    Container, FourCC, HwStage, NativeProgram, RootType, ShaderError, ir, native_programs,
};
use victory_shader::{io, resource};

use crate::shader::{Source, sources};
use crate::walk::{self, MAX_FILE};
use tally::{Reasons, Tally};

/// Emitted shaders by label, shared between threads.
type Kept = std::sync::Arc<Mutex<Vec<(String, emit::HostShader)>>>;

/// The optional checks after emission, one set per thread.
struct Checks {
    /// Shaders named in `VICTORY_PAIRS` (`vs+ps,...`, label substrings),
    /// kept to build those pipelines after the walk.
    keep: Option<Kept>,
    validator: Option<Validator>,
    host: Option<(Host, emit::HostShader)>,
    /// Validator failures seen so far by first error line, for the dumps.
    dumped: HashMap<String, usize>,
}

impl Checks {
    /// Built before the walk, not in the workers: a validator or device
    /// that does not come up is the environment's failure and must stop the
    /// run, where a worker going without would silently thin out the counts.
    fn new(
        dll: Option<&Path>,
        validate: bool,
        pipelines: bool,
        keep: Option<Kept>,
    ) -> Result<Self> {
        let validator = validate
            .then(|| Validator::load(dll))
            .transpose()
            .map_err(anyhow::Error::msg)?;
        let host = pipelines
            .then(|| Host::new(false).and_then(|h| host::stub_ps().map(|ps| (h, ps))))
            .transpose()
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            keep,
            validator,
            host,
            dumped: HashMap::new(),
        })
    }
}

pub fn run(
    paths: Vec<PathBuf>,
    validate: bool,
    pipelines: bool,
    list: bool,
    jobs: Option<usize>,
) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let dll = validate.then(Validator::default_path).flatten();
    let pairs: Vec<(String, String)> = std::env::var("VICTORY_PAIRS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|p| p.split_once('+'))
        .map(|(v, p)| (v.trim().to_owned(), p.trim().to_owned()))
        .collect();
    let kept = std::sync::Arc::new(Mutex::new(Vec::new()));
    // At least one: with none, every file would be read and nothing counted.
    let threads = jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
        .max(1);
    let mut checks = (0..threads)
        .map(|_| {
            let keep = (!pairs.is_empty()).then(|| kept.clone());
            Checks::new(dll.as_deref(), validate, pipelines, keep)
        })
        .collect::<Result<Vec<_>>>()?;
    if validate {
        println!(
            "validator    {}",
            dll.as_ref()
                .map_or("dxil.dll".into(), |p| p.display().to_string())
        );
    }
    let seen = Mutex::new(HashSet::new());
    let total = Mutex::new(Tally::new(list));
    for (path, data) in walk::files(&paths, MAX_FILE) {
        let data = match data {
            Ok(data) => data,
            Err(skip) => {
                lock(&total).unread(&path, skip);
                continue;
            }
        };
        lock(&total).files += 1;
        let jobs = sources(&data);
        let next = AtomicUsize::new(0);
        let (jobs, next, seen, total, path) = (&jobs, &next, &seen, &total, &path);
        std::thread::scope(|s| {
            for checks in checks.iter_mut().take(jobs.len()) {
                s.spawn(move || {
                    let mut t = Tally::new(list);
                    while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        translate(job, path, seen, &mut t, checks);
                    }
                    lock(total).merge(t);
                });
            }
        });
    }
    let t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    let verdict = t.verdict();
    println!("files        {} read, {} above the size limit skipped", t.files, t.oversized);
    println!(
        "containers   {} ({} duplicates skipped)",
        t.containers, t.duplicates
    );
    println!(
        "programs     {} translated of {} ({} IR instructions)",
        t.translated, t.programs, t.instructions
    );
    println!(
        "structured   {} of {} ({} vector instructions, {} guarded, {} on opaque masks; \
         branches {} per lane, {} uniform)",
        t.structured,
        t.translated,
        t.vector,
        t.guarded,
        t.opaque,
        t.lane_branches,
        t.uniform_branches
    );
    println!(
        "resources    {} of {} tracked (descriptors {} loaded, {} built, {} constant; \
         {} scalar accesses; {} with a computed term)",
        t.tracked, t.translated, t.loaded, t.built, t.constant, t.scalar, t.dynamic
    );
    println!(
        "  used-resource entries {}: {} not traced {:?}; {} traced parameters not listed",
        t.used,
        t.untraced_used.values().sum::<usize>(),
        t.untraced_used,
        t.unlisted
    );
    println!(
        "I/O          {} of {} vertex and pixel programs raised ({} interpolations, \
         {} vertex values; exports {:?}, {} still f16 pairs)",
        t.io_raised, t.io_programs, t.interpolated, t.vertex_values, t.exports, t.compressed
    );
    println!(
        "DXIL         {} emitted of {} vertex and pixel programs; {} validated, {} pipelines",
        t.emitted, t.emit_programs, t.validated, t.pipelines
    );
    println!(
        "             {} emitted of {} compute programs; {} validated, {} pipelines",
        t.cs_emitted, t.cs_programs, t.cs_validated, t.cs_pipelines
    );
    println!(
        "             {} emitted of {} mesh programs; {} validated, {} pipelines",
        t.ms_emitted, t.ms_programs, t.ms_validated, t.ms_pipelines
    );
    if !pairs.is_empty() {
        let host = Host::new(false).map_err(anyhow::Error::msg)?;
        let kept = lock(&kept);
        let find = |name: &str| kept.iter().find(|(l, _)| l.contains(name));
        for (v, p) in &pairs {
            let made = match (find(v), find(p)) {
                (Some((_, vs)), Some((_, ps))) => if vs.stage == HwStage::Ms {
                    host.mesh_pipeline(vs, ps)
                } else {
                    host.pipeline(vs, ps)
                }
                .map(|_| "pipeline created".to_owned())
                .unwrap_or_else(|e| e),
                _ => "not emitted".to_owned(),
            };
            println!("pair {v} + {p}: {made}");
        }
    }
    let mut lines = t.lines;
    lines.sort();
    for l in lines {
        println!("{l}");
    }
    print_reasons(t.failed);
    if !t.infra.is_empty() {
        println!("infrastructure failures (every count above is a lower bound):");
        print_reasons(t.infra);
    }
    verdict
}

fn print_reasons(reasons: Reasons) {
    let mut reasons: Vec<_> = reasons.into_iter().collect();
    reasons.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (reason, (n, example)) in reasons {
        println!("  {n:>7}  {reason}  (e.g. {example})");
    }
}

fn translate(
    source: &Source,
    path: &Path,
    seen: &Mutex<HashSet<blake3::Hash>>,
    t: &mut Tally,
    checks: &mut Checks,
) {
    let example = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        return t.fail("corrupt record".into(), example);
    };
    t.containers += 1;
    let Ok(container) = Container::parse(&bytes) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    let Some(dxil) = container.part(FourCC::DXIL).map(|p| p.data) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    if !is_new(seen, &container) {
        t.duplicates += 1;
        return;
    }
    let programs = match native_programs(dxil) {
        Ok(p) => p,
        Err(e) => return t.fail(e.to_string(), example),
    };
    for p in programs {
        t.programs += 1;
        t.isolate(
            || format!("{} {:?}", example(), p.stage),
            |t| program(&container, &p, &example, t, checks),
        );
    }
}

/// One program through every stage, counted into `t`.
fn program(
    container: &Container<'_>,
    p: &NativeProgram,
    example: &dyn Fn() -> String,
    t: &mut Tally,
    checks: &mut Checks,
) {
    match ir::build(&p.patched_code(), p.wave64, p.vgprs) {
        Ok(program) => {
            t.translated += 1;
            t.instructions += program.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
            // Per-lane descriptors are traced on the per-lane form; its
            // failure is listed after the resources' and I/O's.
            let structured = structure::structure(&program);
            let lanes = structured.as_ref().ok();
            let res = match resource::track(&program, &p.layout, p.stage, lanes) {
                Ok(r) => {
                    tally_resources(p, &r, t);
                    Some(r)
                }
                Err(ShaderError::Unsupported { offset, what }) => {
                    t.fail(format!("resources: {what}"), || {
                        format!("{} {:?} at {offset:#x}", example(), p.stage)
                    });
                    None
                }
                Err(e) => {
                    t.fail(e.to_string(), || format!("{} {:?}", example(), p.stage));
                    None
                }
            };
            let graphics = matches!(p.stage, HwStage::Vs | HwStage::Ps);
            let compute = p.stage == HwStage::Cs;
            let mut raised = None;
            if graphics || compute || p.stage == HwStage::Ms {
                t.io_programs += usize::from(graphics);
                match io::raise(&program, p) {
                    Ok(r) => {
                        if graphics {
                            tally_io(&r, t);
                        }
                        raised = Some(r);
                    }
                    Err(ShaderError::Unsupported { offset, what }) => t
                        .fail(format!("I/O: {what}"), || {
                            format!("{} {:?} at {offset:#x}", example(), p.stage)
                        }),
                    Err(e) => t.fail(e.to_string(), || format!("{} {:?}", example(), p.stage)),
                }
            }
            let structured = match structured {
                Ok(s) => {
                    tally_structured(&program, &s, t);
                    Some(s)
                }
                Err(e) => {
                    let reason = match e {
                        ShaderError::Unsupported { what, .. } => format!("structure: {what}"),
                        ShaderError::Isa { reason, .. } => format!("structure: {reason}"),
                        e => e.to_string(),
                    };
                    t.fail(reason, || format!("{} {:?}", example(), p.stage));
                    None
                }
            };
            if let (Some(r), Some(i), Some(s)) = (res, raised, structured) {
                let label = || format!("{} {:?}", example(), p.stage);
                *counters(t, p.stage)[0] += 1;
                match emit::emit(container, p, &program, &s, &i, &r) {
                    Ok(shader) => {
                        *counters(t, p.stage)[1] += 1;
                        check(shader, t, checks, label);
                    }
                    Err(ShaderError::Unsupported { offset, what }) => t
                        .fail(format!("emit: {what}"), || {
                            format!("{} at {offset:#x}", label())
                        }),
                    Err(e) => t.fail(format!("emit: {e}"), label),
                }
            }
        }
        Err(e) => {
            let reason = match e {
                ShaderError::Unsupported { what, .. } => format!("unsupported: {what}"),
                e => e.to_string(),
            };
            t.fail(reason, || format!("{} {:?}", example(), p.stage));
        }
    }
}

/// Validates an emitted shader if asked to.
fn check(
    mut shader: emit::HostShader,
    t: &mut Tally,
    checks: &mut Checks,
    label: impl Fn() -> String,
) {
    if let Some(keep) = &checks.keep {
        lock(keep).push((label(), shader.clone()));
    }
    // `VICTORY_EMIT_DUMP` with `--list`: every emitted shader as a file.
    if t.list
        && let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP")
    {
        let name: String = label()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        write_dump(
            format!("{dir}/{}.dxil", &name[name.len().saturating_sub(100)..]),
            &shader.dxil,
        );
    }
    if let Some((h, stub_ps)) = &checks.host {
        let made = match shader.stage {
            HwStage::Vs => h.pipeline(&shader, stub_ps).map(drop),
            HwStage::Cs => h.compute_pipeline(&shader).map(drop),
            HwStage::Ms => h.mesh_pipeline(&shader, stub_ps).map(drop),
            HwStage::Ls | HwStage::Hs | HwStage::Es | HwStage::Gs | HwStage::Ps => {
                host::stub_vs(&shader).and_then(|vs| h.pipeline(&vs, &shader).map(drop))
            }
        };
        match made {
            Ok(()) => {
                *counters(t, shader.stage)[3] += 1;
                if t.list {
                    t.lines.push(format!("{}  pipeline created", label()));
                }
            }
            Err(e) => t.fail(format!("pipeline: {e}"), &label),
        }
    }
    let Some(v) = checks.validator.as_mut() else {
        return;
    };
    match v.validate(&mut shader.dxil) {
        Ok(()) => {
            *counters(t, shader.stage)[2] += 1;
            if t.list {
                t.lines.push(format!("{}  validated", label()));
            }
        }
        Err(text) => {
            let first = text
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or(text.lines().next().unwrap_or(""))
                .to_owned();
            if let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP") {
                let seen = checks.dumped.entry(first.clone()).or_default();
                let n = *seen;
                *seen += 1;
                if n < 3 {
                    let name = format!(
                        "{dir}/{}_{n}",
                        first
                            .chars()
                            .filter(char::is_ascii_alphanumeric)
                            .take(40)
                            .collect::<String>()
                    );
                    write_dump(format!("{name}.dxil"), &shader.dxil);
                    write_dump(format!("{name}.txt"), format!("{}\n{text}", label()));
                }
            }
            t.fail(format!("validator: {first}"), label);
        }
    }
}

/// Whether no container that translates the same way was seen before. The
/// key covers every part translation reads ([`emit::INPUT_PARTS`]), not the
/// DXIL part alone: containers sharing their programs but not their
/// signatures emit different shaders. Parts nothing reads (the digest, a
/// debug name) are left out, so they do not defeat the dedup.
fn is_new(seen: &Mutex<HashSet<blake3::Hash>>, container: &Container<'_>) -> bool {
    let mut key = blake3::Hasher::new();
    for fourcc in emit::INPUT_PARTS {
        key.update(&fourcc.0);
        // Length-prefixed, and a missing part apart from an empty one.
        match container.part(fourcc) {
            Some(part) => {
                key.update(&[1]);
                key.update(&(part.data.len() as u64).to_le_bytes());
                key.update(part.data);
            }
            None => {
                key.update(&[0]);
            }
        }
    }
    lock(seen).insert(key.finalize())
}

/// Writes a `VICTORY_EMIT_DUMP` file. A dump is a debugging aid, so a
/// failed write does not stop the run, but it is logged with its path;
/// returns whether the file was written.
fn write_dump(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> bool {
    let path = path.as_ref();
    match std::fs::write(path, contents) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "emit dump not written");
            false
        }
    }
}

/// Emission counts of `stage`'s kind: 0 programs, 1 emitted, 2 validated,
/// 3 pipelines. Callers index with constants, which rustc bounds-checks.
fn counters(t: &mut Tally, stage: HwStage) -> [&mut usize; 4] {
    match stage {
        HwStage::Cs => [
            &mut t.cs_programs,
            &mut t.cs_emitted,
            &mut t.cs_validated,
            &mut t.cs_pipelines,
        ],
        HwStage::Ms => [
            &mut t.ms_programs,
            &mut t.ms_emitted,
            &mut t.ms_validated,
            &mut t.ms_pipelines,
        ],
        HwStage::Ls | HwStage::Hs | HwStage::Es | HwStage::Gs | HwStage::Vs | HwStage::Ps => [
            &mut t.emit_programs,
            &mut t.emitted,
            &mut t.validated,
            &mut t.pipelines,
        ],
    }
}

fn tally_io(r: &io::Io, t: &mut Tally) {
    t.io_raised += 1;
    for (_, i) in &r.interps {
        match i {
            io::Interp::Interpolated { .. } => t.interpolated += 1,
            io::Interp::Vertex { .. } => t.vertex_values += 1,
        }
    }
    for e in &r.exports {
        let kind = match e.target {
            io::Target::Color(_) => "color",
            io::Target::Depth => "depth",
            io::Target::Null => "null",
            io::Target::Position(_) => "position",
            io::Target::Param { .. } => "param",
            io::Target::Unread => "unread param",
            io::Target::Primitive => "primitive",
            io::Target::PrimitiveParam { .. } => "primitive param",
        };
        *t.exports.entry(kind).or_default() += 1;
        t.compressed += usize::from(e.compressed);
    }
}

fn tally_resources(p: &NativeProgram, r: &resource::Resources, t: &mut Tally) {
    t.tracked += 1;
    t.scalar += r.scalar.len();
    t.dynamic += r.scalar.iter().filter(|s| s.1.dynamic.is_some()).count();
    let address = |load| r.scalar.iter().find(|s| s.0 == load).map(|s| s.1);
    let mut traced = HashSet::new();
    for leaf in r.descriptors.iter().flat_map(|d| d.source.leaves()) {
        let a = match *leaf {
            resource::Source::Loaded { load, .. } | resource::Source::Fetched { load, .. } => {
                t.loaded += 1;
                address(load)
            }
            resource::Source::Built { address, .. } => {
                t.built += 1;
                t.dynamic += usize::from(address.dynamic.is_some());
                Some(address)
            }
            resource::Source::Constant(_) => {
                t.constant += 1;
                None
            }
            resource::Source::Phi { .. }
            | resource::Source::Select { .. }
            | resource::Source::Lanes { .. }
            | resource::Source::LanePhi { .. } => {
                unreachable!("leaves are no phis, selects or per-lane choices")
            }
        };
        // Heaps hold dynamic resources, which no used entry lists; code
        // tables are no root parameter.
        if let Some(a) = a
            && a.param != resource::CODE
            && !matches!(
                p.layout[a.param].root,
                RootType::ResourceHeap | RootType::SamplerHeap
            )
        {
            traced.insert(p.layout[a.param].user_data);
        }
    }
    // Root constants are listed as a CBV, read straight from the SGPRs.
    traced.extend(
        r.params
            .iter()
            .map(|&(_, param, _)| &p.layout[param])
            .filter(|l| l.root == RootType::Constants)
            .map(|l| l.user_data),
    );
    let listed: HashSet<u8> = p.used.iter().map(|u| u.user_data).collect();
    t.used += p.used.len();
    // Immediate samplers are literals in the code: the distinct sampler
    // values (decoded) are matched against the listed entries, which do not
    // say which value is whose.
    let mut values: Vec<emit::StaticSampler> = Vec::new();
    for d in r
        .descriptors
        .iter()
        .filter(|d| d.role == resource::Role::Sampler)
    {
        for leaf in d.source.leaves() {
            if let resource::Source::Constant(words) = leaf
                && let Ok(w) = <[u32; 4]>::try_from(&words[..words.len().min(4)])
            {
                let s = emit::StaticSampler::from_words(w);
                if !values.contains(&s) {
                    values.push(s);
                }
            }
        }
    }
    // Engines often declare equal samplers at several registers, so equal
    // values may stand for several entries, but never more values than
    // entries.
    let immediate = p
        .used
        .iter()
        .filter(|u| u.root == RootType::ImmediateSampler)
        .count();
    let samplers_match = !values.is_empty() && values.len() <= immediate;
    let matched = |u: &&victory_shader::UsedResource| {
        if u.root == RootType::ImmediateSampler {
            samplers_match
        } else {
            traced.contains(&u.user_data)
        }
    };
    for u in p.used.iter().filter(|u| !matched(u)) {
        *t.untraced_used
            .entry(format!("{:?} via {:?}", u.kind, u.root))
            .or_default() += 1;
    }
    t.unlisted += traced.difference(&listed).count();
}

fn tally_structured(program: &ir::Program, s: &structure::Structured, t: &mut Tally) {
    t.structured += 1;
    for (block, guards) in program.blocks.iter().zip(&s.guards) {
        t.vector += block
            .insts
            .iter()
            .filter(|i| i.read(ir::Reg::S(ir::EXEC)).is_some())
            .count();
        for &g in guards.iter().flatten() {
            t.guarded += 1;
            let support = s.bools.support(g);
            t.opaque += usize::from(support.iter().any(|a| matches!(a, Atom::Opaque(_))));
        }
    }
    for exit in &s.exits {
        if let LaneExit::Branch { when, .. } = *exit {
            if s.bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_)))
            {
                t.uniform_branches += 1;
            } else {
                t.lane_branches += 1;
            }
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::Part;

    #[test]
    fn dump_is_written() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("a.dxil");
        assert!(write_dump(&path, b"DXBC"));
        assert_eq!(std::fs::read(&path).expect("dump read back"), b"DXBC");
    }

    #[test]
    fn failed_dump_is_reported_not_swallowed() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("missing").join("a.dxil");
        assert!(!write_dump(&path, b"DXBC"));
        assert!(!path.exists());
    }

    /// A synthetic container of `parts`; only the parts are read by the key.
    fn container<'a>(digest: u8, parts: &[(FourCC, &'a [u8])]) -> Container<'a> {
        Container {
            digest: [digest; 16],
            version: (1, 0),
            parts: parts
                .iter()
                .map(|&(fourcc, data)| Part {
                    fourcc,
                    offset: 0,
                    data,
                })
                .collect(),
        }
    }

    #[test]
    fn same_dxil_with_other_signatures_is_translated_too() {
        let seen = Mutex::new(HashSet::new());
        let a = container(0, &[(FourCC::DXIL, b"program"), (FourCC::ISG1, b"a")]);
        let b = container(0, &[(FourCC::DXIL, b"program"), (FourCC::ISG1, b"b")]);
        assert!(is_new(&seen, &a));
        assert!(is_new(&seen, &b));
        assert!(!is_new(&seen, &a));
    }

    #[test]
    fn parts_emit_does_not_read_do_not_defeat_dedup() {
        let seen = Mutex::new(HashSet::new());
        let a = container(1, &[(FourCC::DXIL, b"program"), (FourCC::ILDN, b"a.pdb")]);
        let b = container(2, &[(FourCC::ILDN, b"b.pdb"), (FourCC::DXIL, b"program")]);
        assert!(is_new(&seen, &a));
        assert!(!is_new(&seen, &b));
    }

    #[test]
    fn a_missing_part_differs_from_an_empty_one() {
        let seen = Mutex::new(HashSet::new());
        let a = container(0, &[(FourCC::DXIL, b"program")]);
        let b = container(0, &[(FourCC::DXIL, b"program"), (FourCC::PSV0, b"")]);
        assert!(is_new(&seen, &a));
        assert!(is_new(&seen, &b));
    }
}
