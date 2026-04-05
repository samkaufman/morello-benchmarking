use morello::codegen::CodeGen;
use morello::common::Dtype;
use morello::db::FilesDatabase;
use morello::layout::row_major;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::shape;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::CpuKernel;
use morello::target::{
    Avx2Target,
    CpuMemory::{self, L1, RF},
    Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use std::env;
use std::io;
use std::num::NonZeroU32;
use std::panic;
use std::path::PathBuf;
use std::process;

const BATCH_SIZE_PER_THREAD: u32 = 32;

struct Args {
    db: Option<PathBuf>,
    batch_size: NonZeroU32,
    seq_len: NonZeroU32,
    parallel: bool,
}

fn usage(program_name: &str) -> String {
    format!("Usage: {program_name} [--db <path>] [--parallel] <batch_size> <seq_len>")
}

fn parse_args() -> Args {
    let mut args_iter = env::args();
    let program_name = args_iter
        .next()
        .unwrap_or_else(|| String::from("softmax_3pass_synth"));

    let mut parallel = false;
    let mut integer_args = vec![];
    let mut db = None;

    while let Some(arg) = args_iter.next() {
        if arg == "--parallel" {
            parallel = true;
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
    }
}

fn main() {
    let args = parse_args();

    let db = FilesDatabase::new::<Avx2Target>(args.db.as_deref(), true, 1, 10_000, 1);

    let shape = shape![args.batch_size, args.seq_len];
    let batch_size = shape[0].get();
    let seq_len = shape[1].get();
    let logical_spec = LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Softmax { scan_dim: 1 },
            spec_shape: shape.clone(),
            dtypes: vec![Dtype::Float32; shape.len()],
        },
        vec![
            TensorSpecAux {
                memory: CpuMemory::GL,
                layout: row_major(&shape),
                vector_size: None,
            };
            2
        ],
        !args.parallel,
    );
    let spec = Spec::<Avx2Target>(logical_spec, Avx2Target::max_mem());
    println!("Logical Spec: {}", spec.0);

    // Tile across the batch dimension. (We cannot tile across the scan dimension.)
    let implementation = if batch_size == 1 {
        spec.to_softmax_parts(RF, row_major, None, L1, row_major, None)
    } else {
        let tiled = match (args.parallel, batch_size) {
            (true, 2..=BATCH_SIZE_PER_THREAD) => spec.tile_out_parallel(&[1, seq_len]),
            (true, _) => spec
                .tile_out_parallel(&[BATCH_SIZE_PER_THREAD, seq_len])
                .tile_out(&[1, seq_len]),
            _ => spec.tile_out(&[1, seq_len]),
        };
        tiled.to_softmax_parts(RF, row_major, None, L1, row_major, None)
    }
        .subschedule(&[1], |dvs| dvs.select(CpuKernel::DivideVecScalarReciprocal))
        .synthesize_all(&db);

    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, ImplPrintStyle::Compact);

    println!("\nThe above Impl lowered to C:");
    implementation
        .emit(false, None, &mut ToWriteFmt(io::stdout()))
        .unwrap_or_else(|e| panic!("Failed to generate code: {e}"));

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

    // Benchmark.
    let build_result = implementation
        .build(true)
        .unwrap_or_else(|e| panic!("Failed to build generated code for benchmarking: {}", e));
    println!("{}", build_result.binary_path().display());
}
