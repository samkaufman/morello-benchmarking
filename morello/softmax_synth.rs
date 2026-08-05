use morello::codegen::CodeGen;
use morello::common::Dtype;
use morello::cost::Cost;
use morello::db::{FilesDatabase, TileScale};
use morello::grid::canon::CanonicalBimap;
use morello::grid::general::BiMap;
use morello::imp::functions::FunctionApp;
use morello::imp::ImplNode;
use morello::layout::row_major;
use morello::scheduling::{Action, ActionT};
use morello::scheduling_sugar::SchedulingSugar;
use morello::search::top_down_many_impls;
use morello::shape;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{Avx2Target, Avx512Target, CpuMemory, CpuTarget, Target};
use morello::tensorspec::TensorSpecAux;

use std::cell::Cell;
use std::env;
use std::num::NonZeroU32;
use std::panic;
use std::path::PathBuf;
use std::process;

#[cfg(all(
    feature = "softmax-disable-offline-rewrites",
    feature = "softmax-disable-online-rewrites"
))]
compile_error!("softmax_synth cannot disable both online and offline softmax rewrites");

struct Args {
    db: Option<PathBuf>,
    batch_size: NonZeroU32,
    seq_len: NonZeroU32,
    parallel: bool,
    avx512: bool,
    emit_benchmark_c: bool,
}

/// Extends [ImplNode] with the `try_synthesize_all` method.
trait TrySynthesizeAll<Tgt: Target> {
    fn try_synthesize_all(self, db: &FilesDatabase) -> Option<(ImplNode<Tgt>, Cost)>
    where
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>;
}

impl<Tgt: Target> TrySynthesizeAll<Tgt> for ImplNode<Tgt> {
    fn try_synthesize_all(self, db: &FilesDatabase) -> Option<(ImplNode<Tgt>, Cost)>
    where
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let succeeded = Cell::new(true);
        let imp = self.map_spec_leaves(&|spec_app| {
            let spec = spec_app.0.clone();
            let Some(body) = top_down_many_impls(db, &[spec.clone()])
                .into_iter()
                .next()
                .and_then(|implementations| implementations.into_iter().next())
            else {
                succeeded.set(false);
                return spec_app.into();
            };

            FunctionApp {
                body: Box::new(body),
                parameters: spec_app.1,
                spec: Some(spec),
            }
            .into()
        });

        succeeded.get().then(|| {
            let cost = Cost::from_impl(&imp);
            (imp, cost)
        })
    }
}

fn usage(program_name: &str) -> String {
    format!(
        "Usage: {program_name} [--db <path>] [--parallel] [--avx512] [--emit-benchmark-c] <batch_size> <seq_len>"
    )
}

fn parse_args() -> Args {
    let mut args_iter = env::args();
    let program_name = args_iter
        .next()
        .unwrap_or_else(|| String::from("softmax_synth"));

    let mut parallel = false;
    let mut avx512 = false;
    let mut emit_benchmark_c = false;
    let mut integer_args = vec![];
    let mut db = None;

    while let Some(arg) = args_iter.next() {
        if arg == "--parallel" {
            parallel = true;
            continue;
        }
        if arg == "--avx512" {
            avx512 = true;
            continue;
        }
        if arg == "--emit-benchmark-c" {
            emit_benchmark_c = true;
            continue;
        }
        if arg == "--db" {
            let Some(path) = args_iter.next() else {
                eprintln!("--db flag requires a path argument");
                eprintln!("{}", usage(&program_name));
                process::exit(2);
            };
            db = Some(PathBuf::from(path));
            continue;
        }
        if let Ok(value) = arg.parse::<u32>() {
            integer_args.push(value);
            continue;
        }

        eprintln!("Unrecognized argument: {arg}");
        eprintln!("{}", usage(&program_name));
        process::exit(2);
    }

    let [batch_size, seq_len] = integer_args[..] else {
        eprintln!("incorrect arguments");
        eprintln!("{}", usage(&program_name));
        process::exit(2);
    };

    let Some(batch_size) = NonZeroU32::new(batch_size) else {
        eprintln!("batch_size must be non-zero");
        process::exit(2);
    };
    let Some(seq_len) = NonZeroU32::new(seq_len) else {
        eprintln!("seq_len must be non-zero");
        process::exit(2);
    };

    Args {
        db,
        batch_size,
        seq_len,
        parallel,
        avx512,
        emit_benchmark_c,
    }
}

/// Synthesize a single-row softmax [Spec] using the lowest-cost enabled top-level rewrite.
fn schedule_softmax_leaf<Tgt>(leaf: &Spec<Tgt>, db: &FilesDatabase) -> ImplNode<Tgt>
where
    Tgt: CpuTarget,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    assert!(
        matches!(&leaf.0, LogicalSpec::Primitive(basics, _, _)
            if basics.typ == (PrimitiveSpecType::Softmax { scan_dim: 1 })
                && basics.spec_shape[0].get() == 1
                && basics.dtypes.as_slice() == &[Dtype::Float32, Dtype::Float32]),
        "Expected a single-row f32 softmax leaf over scan dimension 1, got {}",
        leaf.0
    );

    eprintln!("Synthesizing: {}", leaf.0);
    Tgt::actions(&leaf.0)
        .filter_map(|action| {
            let action_enabled = match &action {
                Action::ToSoftmaxParts(a) => a.denominator_layout.is_row_major(),
                Action::ToSoftmaxPartsRecompute(a) => {
                    a.max_layout.is_row_major() && a.denominator_layout.is_row_major()
                }
                _ => false,
            };
            if !action_enabled {
                return None;
            }

            action.apply(leaf).ok()?.try_synthesize_all(db)
        })
        .min_by(|(_, a_cost), (_, b_cost)| a_cost.cmp(b_cost))
        .map(|(implementation, _)| implementation)
        .unwrap_or_else(|| {
            panic!(
                "No softmax top-level candidate was applicable for {}",
                leaf.0
            )
        })
}

fn main() {
    let args = parse_args();

    if args.avx512 {
        main_per_target::<Avx512Target>(args);
    } else {
        main_per_target::<Avx2Target>(args);
    }
}

fn main_per_target<Tgt>(args: Args)
where
    Tgt: CpuTarget,
    Tgt::Memory: CanonicalBimap + From<CpuMemory>,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let db = FilesDatabase::new::<Tgt>(args.db.as_deref(), TileScale::PowerOfTwo, 1, 10_000, 1);

    let shape = shape![args.batch_size, args.seq_len];
    let logical_spec = LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Softmax { scan_dim: 1 },
            spec_shape: shape.clone(),
            dtypes: vec![Dtype::Float32; shape.len()],
        },
        vec![
            TensorSpecAux {
                memory: CpuMemory::GL.into(),
                layout: row_major(&shape),
                vector_size: None,
            };
            2
        ],
        !args.parallel,
    );
    let spec = Spec::<Tgt>(logical_spec, Tgt::max_mem());

    // Schedule either the whole single-row spec or each row-sized leaf after batch tiling.
    //
    // Specifically, we restrict the space of synthesized implementations to those which:
    // - Are tiled, possibly with a parallel loop, to size-one batches (i.e., per-row).
    // - After tiling, the `Softmax` Spec is scheduled with either `ToSoftmaxParts`
    //   or `ToSoftmaxPartsRecompute`.
    //   - Intermediate memories (denominators, exps, or maxes) are restricted to
    //     row-major.
    let implementation = if args.batch_size.get() > 1 {
        let tile_shape = [1, args.seq_len.get()];
        if args.parallel {
            let serial_tiled = spec
                .tile_out(&tile_shape)
                .apply_to_default_leaf(|leaf| schedule_softmax_leaf(leaf, &db));
            let parallel_tiled = spec
                .tile_out_parallel(&tile_shape)
                .apply_to_default_leaf(|leaf| schedule_softmax_leaf(leaf, &db));
            if Cost::from_impl(&parallel_tiled) <= Cost::from_impl(&serial_tiled) {
                parallel_tiled
            } else {
                serial_tiled
            }
        } else {
            spec.tile_out(&tile_shape)
                .apply_to_default_leaf(|leaf| schedule_softmax_leaf(leaf, &db))
        }
    } else {
        schedule_softmax_leaf(&spec, &db)
    };

    // If the verification flag is set, let's additionally double-check that the lowered
    // code builds and produces the correct results.
    #[cfg(feature = "verification")]
    {
        match implementation.build(false) {
            Ok(artifact) => {
                if !artifact.check_correctness(&spec) {
                    panic!("Generated code returned incorrect output");
                }
            }
            Err(e) => {
                panic!("Failed to build generated code: {e}");
            }
        }
    }

    if args.emit_benchmark_c {
        let mut source = String::new();
        implementation
            .emit(true, None, &mut source)
            .unwrap_or_else(|e| panic!("Failed to generate benchmark code: {e}"));
        print!("{source}");
        return;
    }

    // Benchmark.
    let build_result = implementation
        .build(true)
        .unwrap_or_else(|e| panic!("Failed to build generated code for benchmarking: {}", e));
    println!("{}", build_result.binary_path().display());
}
