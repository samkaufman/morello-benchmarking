use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype, Shape};
use morello::db::{FilesDatabase, TileScale};
use morello::grid::canon::CanonicalBimap;
use morello::grid::general::BiMap;
use morello::imp::{subspecs::SpecApp, Impl, ImplNode};
use morello::layout;
use morello::layout::row_major;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::SchedulingSugar;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::Memory;
use morello::target::{
    Avx2Target, Avx512Kernel, Avx512Target, CpuKernel,
    CpuMemory::{GL, L1, RF, VRF},
    CpuTarget, Target, TargetId,
};
use morello::utils::{prev_power_of_two_u32, ToWriteFmt};
use morello::views::ViewE;
use morello::{shape, spec};
use nonzero::nonzero as nz;
use smallvec::SmallVec;
use std::{env, fmt::Debug, io, path::Path, process};

const MC: u32 = 528;
const KC: u32 = 528;
const NC: u32 = 1056;
const AVX512_BF16_MC: u32 = 2400;
const AVX512_BF16_KC: u32 = 2400;
const AVX512_BF16_NC: u32 = 2400;
const AVX512_BF16_MR: u32 = 4;
const AVX512_BF16_NR: u32 = 16;
const AVX512_BF16_L1_KC: u32 = 336;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatmulKind {
    F32,
    I32,
    BF16F32,
}

impl MatmulKind {
    fn parse(arg: &str) -> Option<Self> {
        match arg {
            "f32" => Some(Self::F32),
            "i32" => Some(Self::I32),
            "bf16f32" => Some(Self::BF16F32),
            _ => None,
        }
    }

    fn packed_n_widths(self) -> (DimSize, DimSize) {
        match self {
            Self::F32 => (nz!(16u32), nz!(48u32)),
            Self::BF16F32 => (nz!(16u32), nz!(32u32)),
            // Int32-output modes currently need a conservative packed-N width for correctness.
            Self::I32 => (nz!(4u32), nz!(4u32)),
        }
    }

    fn avx2_mr(self) -> DimSize {
        if self == Self::BF16F32 {
            nz!(6u32)
        } else {
            nz!(4u32)
        }
    }
}

fn main() {
    env_logger::init();
    let mut use_avx512 = false;
    let mut integer_args = vec![];
    let mut db_path: Option<String> = None;
    let mut kind: Option<MatmulKind> = None;

    let mut args_iter = env::args().skip(1);
    while let Some(arg) = args_iter.next() {
        match arg.as_str() {
            "--avx512" => use_avx512 = true,
            "--db" => {
                db_path = Some(
                    args_iter
                        .next()
                        .unwrap_or_else(|| panic!("--db flag requires a path argument")),
                );
            }
            _ => {
                if let Some(arg_kind) = MatmulKind::parse(&arg) {
                    if kind.replace(arg_kind).is_some() {
                        panic!("dtype specified multiple times");
                    }
                } else if let Ok(v) = arg.parse::<u32>() {
                    integer_args.push(v);
                } else {
                    panic!("Unrecognized argument: {}", arg);
                }
            }
        }
    }

    let Some(kind) = kind else {
        eprint_usage_message();
        process::exit(2);
    };
    let [batch_size, m, k, n] = integer_args[..] else {
        eprint_usage_message();
        process::exit(2);
    };

    let (avx2_v_n_size, avx512_v_n_size) = kind.packed_n_widths();

    if use_avx512 {
        let mr = if kind == MatmulKind::BF16F32 {
            DimSize::try_from(AVX512_BF16_MR).unwrap()
        } else {
            nz!(8u32)
        };
        main_per_target::<Avx512Target>(
            batch_size,
            m,
            k,
            n,
            avx512_v_n_size,
            mr,
            db_path,
            kind,
        );
    } else {
        main_per_target::<Avx2Target>(
            batch_size,
            m,
            k,
            n,
            avx2_v_n_size,
            kind.avx2_mr(),
            db_path,
            kind,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn main_per_target<Tgt>(
    batch_size: u32,
    m: u32,
    k: u32,
    n: u32,
    v_n_size: DimSize,
    mr: DimSize,
    db_path: Option<String>,
    kind: MatmulKind,
) where
    Tgt: Bf16InnerSchedule,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let db_path_ref = db_path.as_deref().map(Path::new);
    let db = FilesDatabase::new::<Tgt>(db_path_ref, TileScale::Linear, 1, 30_000, 1);

    let mut spec: Spec<Tgt> = match kind {
        MatmulKind::F32 => spec!(MatmulAccum(
            [batch_size, m, k, n],
            (f32, GL, row_major),
            (f32, GL, row_major),
            (f32, GL, row_major)
        )),
        MatmulKind::I32 => spec!(MatmulAccum(
            [batch_size, m, k, n],
            (i32, GL, row_major),
            (i32, GL, row_major),
            (i32, GL, row_major)
        )),
        MatmulKind::BF16F32 => spec!(MatmulAccum(
            [batch_size, m, k, n],
            (bf16, GL, row_major),
            (bf16, GL, row_major),
            (f32, GL, row_major)
        )),
    };
    spec.0.set_serial_only(batch_size == 1);
    spec.canonicalize().unwrap();

    let implementation = tile_out_parallel_ensure_continue(&spec, &[1, m, n], |s| {
        schedule_matmul_serial(s, m, n, v_n_size, mr)
    });
    let implementation = apply_rewrites(implementation);
    let implementation = implementation.synthesize_all(&db);
    drop(db); // spill to disk here
    implementation
        .emit(
            false,
            Some(ImplPrintStyle::Compact),
            &mut ToWriteFmt(io::stdout()),
        )
        .unwrap_or_else(|e| panic!("Failed to generate code: {}", e));

    let skip_build = env::var("MATMUL_SKIP_BUILD").is_ok();
    if !skip_build {
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
                    panic!("Failed to build generated code: {}", e);
                }
            }
        }

        // Benchmark.
        let build_result = implementation
            .build(true)
            .unwrap_or_else(|e| panic!("Failed to build generated code for benchmarking: {}", e));
        println!("{}", build_result.binary_path().display());
    }
}

fn eprint_usage_message() {
    eprintln!(
        "Usage: matmul_x86_parameterized [--avx512] [--db <path>] <f32|i32|bf16f32> <batch_size> <m> <k> <n>"
    );
}

fn spec_of<Tgt: Target>(node: &ImplNode<Tgt>) -> &Spec<Tgt> {
    let ImplNode::SpecApp(SpecApp(spec, ..)) = node else {
        unreachable!();
    };
    spec
}

fn matmul_accum<Tgt: Target>(spec: &Spec<Tgt>) -> Option<bool> {
    let LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum },
            ..
        },
        ..,
    ) = &spec.0
    else {
        return None;
    };
    Some(*accum)
}

fn is_move<Tgt: Target>(spec: &Spec<Tgt>) -> bool {
    matches!(
        &spec.0,
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Move,
                ..
            },
            ..
        )
    )
}

fn schedule_matmul_serial<Tgt: Bf16InnerSchedule>(
    spec_app: &ImplNode<Tgt>,
    m: u32,
    n: u32,
    v_n_size: DimSize,
    mr: DimSize,
) -> ImplNode<Tgt> {
    // vec_size is largest register width divided by the 4-byte output element size.
    let vec_size =
        DimSize::try_from(*Tgt::Memory::from(VRF).vector_bytes().iter().max().unwrap() / 4)
            .unwrap();

    spec_app.tile_out_ensure_continue(&[1, (m / mr.get()) * mr.get(), n], |a| {
        // layout_a packs the M dimension up to size `mr`. (If smaller than `mr`, layout_a will
        // just be canonicalized to batched column-major.)
        let spec_a = spec_of(a);
        let m_inner = spec_a.0.parameter_shape(0)[1].min(mr);
        let bf16f32 = is_bf16f32_matmul(spec_a);
        let avx2_bf16 = bf16f32 && Tgt::target_id() == TargetId::Avx2;
        let avx512_bf16 = bf16f32 && Tgt::target_id() == TargetId::Avx512;
        let (mc, kc, nc) = if avx512_bf16 {
            (AVX512_BF16_MC, AVX512_BF16_KC, AVX512_BF16_NC)
        } else {
            (MC, KC, NC)
        };
        let layout_a =
            if bf16f32 && !avx2_bf16 && spec_a.0.parameter_shape(0)[2].get().is_multiple_of(2) {
                layout![0, 1, 2, 1 p(m_inner), 2 p(2)]
            } else {
                layout![0, 1, 2, 1 p(m_inner)]
            };

        a.tile_out_ensure_continue(&[1, mc, n], |b| {
            b.split_saturating_ensure_continue(kc, |c| {
                // TODO: move_relayout(0,..) does some redundant work here
                let stripped_n = (n / v_n_size) * v_n_size.get(); // peels some off for Packed dim.
                let c = if avx2_bf16 {
                    c.clone()
                } else {
                    c.move_relayout(0, GL, layout_a.clone(), None)
                };
                c.tile_out_ensure_continue(&[1, mc, stripped_n], |d| {
                    let spec_d = spec_of(d);
                    let n_inner = spec_d.0.parameter_shape(1)[2].min(v_n_size);
                    let bf16f32 = is_bf16f32_matmul(spec_d);
                    let avx2_bf16 = bf16f32 && Tgt::target_id() == TargetId::Avx2;
                    let layout_b = if bf16f32
                        && !avx2_bf16
                        && spec_d.0.parameter_shape(1)[1].get().is_multiple_of(2)
                    {
                        layout![0, 2, 1, 2 p(n_inner), 1 p(2)]
                    } else {
                        layout![0, 2, 1, 2 p(n_inner)]
                    };

                    d.tile_out_ensure_continue(&[1, mc, nc], |e| {
                        let spec_e = spec_of(&e);
                        let e = if avx2_bf16 {
                            e.clone()
                        } else {
                            e.move_relayout(1, GL, layout_b.clone(), None)
                        };
                        let out_shape = spec_e.0.parameter_shape(2);
                        if avx2_bf16 {
                            let n_tile =
                                [nz!(16u32), nz!(8u32), nz!(4u32)]
                                    .into_iter()
                                    .find(|&width| {
                                        out_shape[2] >= width
                                            && out_shape[2].get().is_multiple_of(width.get())
                                    });
                            let m_tile = if out_shape[1] >= nz!(6u32) {
                                nz!(6u32)
                            } else {
                                nz!(1u32)
                            };
                            if let Some(n_tile) = n_tile {
                                if out_shape[1].get().is_multiple_of(m_tile.get()) {
                                    let f32_e = e
                                        .cast(0, Dtype::Float32, GL, row_major, None)
                                        .cast(1, Dtype::Float32, GL, row_major, None);
                                    let vector_width = n_tile.min(nz!(8u32));
                                    return f32_e.tile_out_ensure_continue(
                                        &[1, m_tile.get(), n_tile.get()],
                                        |tile| {
                                            schedule_broadcast_vec_mult_add_tile(tile, vector_width)
                                        },
                                    );
                                }
                            }
                        }
                        let mc_tile_size = spec_e.0.parameter_shape(0)[1].get();
                        chain_tile(
                            &e,
                            &[
                                shape![1, mc_tile_size, v_n_size.get()],
                                shape![1, mc_tile_size, 16],
                                shape![1, mc_tile_size, 8],
                                shape![1, mc_tile_size, 4],
                            ],
                            &|f| {
                                let spec_f = spec_of(f);
                                if is_bf16f32_matmul(spec_f)
                                    && Tgt::target_id() == TargetId::Avx512
                                    && spec_f.0.parameter_shape(2)[2] >= nz!(16u32)
                                {
                                    return Tgt::schedule_bf16_inner_tile(f, vec_size);
                                }

                                f.tile_out_ensure_continue(&[1, mr.get(), v_n_size.get()], |i| {
                                    let spec_i = spec_of(i);
                                    let bf16_scalar_n_tail = is_bf16f32_matmul(spec_i)
                                        && Tgt::target_id() != TargetId::Avx512
                                        && spec_i.0.parameter_shape(1)[2] < nz!(8u32);
                                    if bf16_scalar_n_tail {
                                        schedule_bf16_scalar_tail_tile(i)
                                    } else if spec_i.0.parameter_shape(1)[2] < nz!(4u32) {
                                        i.split(1)
                                    } else {
                                        let width = [v_n_size, nz!(16u32), nz!(8u32), nz!(4u32)]
                                            .into_iter()
                                            .find(|&s| spec_i.0.parameter_shape(1)[2] >= s)
                                            .unwrap();
                                        let v = vec_size.min(width);
                                        if is_bf16f32_matmul(spec_i) {
                                            Tgt::schedule_bf16_inner_tile(i, v)
                                        } else {
                                            i.move_param(2, L1).move_vrf(2, VRF, v.get())
                                        }
                                    }
                                })
                            },
                        )
                    })
                })
            })
        })
    })
}

fn is_bf16f32_matmul<Tgt: CpuTarget>(spec: &Spec<Tgt>) -> bool {
    matmul_accum(spec) == Some(true)
        && spec.0.parameter(0).dtype() == Dtype::Bfloat16
        && spec.0.parameter(1).dtype() == Dtype::Bfloat16
        && spec.0.parameter(2).dtype() == Dtype::Float32
}

trait Bf16InnerSchedule: CpuTarget {
    /// Applies the target-specific innermost BF16 matmul tiling before selecting a kernel.
    ///
    /// AVX2 keeps the scalar-K schedule for remainder tiles. AVX512 splits K to a
    /// cache-sized leaf, keeps the packed B panel in L1 across the M microtiles,
    /// and tiles the output to an M block by up to two 16-lane N vectors so the
    /// selected `VDPBF16PS` kernel has enough independent accumulator chains for
    /// Zen 5 while still fitting in zmm registers.
    fn schedule_bf16_inner_tile(
        matmul: &ImplNode<Self>,
        vector_width: DimSize,
    ) -> ImplNode<Self>;
}

impl Bf16InnerSchedule for Avx2Target {
    fn schedule_bf16_inner_tile(matmul: &ImplNode<Self>, vector_width: DimSize) -> ImplNode<Self> {
        schedule_bf16_scalar_k_tile(matmul, vector_width, CpuKernel::BroadcastVecMultAddBf16F32)
    }
}

impl Bf16InnerSchedule for Avx512Target {
    fn schedule_bf16_inner_tile(matmul: &ImplNode<Self>, vector_width: DimSize) -> ImplNode<Self> {
        if vector_width < nz!(16u32) {
            return schedule_avx512_bf16_n_tail_tile(matmul, vector_width);
        }

        let spec = spec_of(matmul);
        let m_block = spec.0.parameter_shape(2)[1]
            .min(DimSize::try_from(AVX512_BF16_MR).unwrap());
        let full_m = spec.0.parameter_shape(2)[1].get();
        let n_block = spec.0.parameter_shape(2)[2].min(DimSize::try_from(AVX512_BF16_NR).unwrap());
        matmul.tile_out_ensure_continue(&[1, full_m, n_block.get()], |n_tile| {
            n_tile.split_saturating_ensure_continue(AVX512_BF16_L1_KC, |k_tile| {
                let spec = spec_of(k_tile);
                let mut k_panel = k_tile.clone();
                if spec.0.parameter(1).memory() != L1 {
                    k_panel = k_panel.move_param(1, L1);
                }

                k_panel.tile_out_ensure_continue(&[1, m_block.get(), n_block.get()], |micro_tile| {
                    let spec = spec_of(micro_tile);
                    let mut scheduled = micro_tile.clone();
                    if spec.0.parameter(2).memory() != VRF {
                        scheduled = scheduled.move_vrf(2, VRF, vector_width.get());
                    }
                    if spec.0.parameter(0).memory() != L1 {
                        scheduled = scheduled.move_param(0, L1);
                    }

                    if spec.0.parameter_shape(0)[2].get().is_multiple_of(2) {
                        scheduled.select(Avx512Kernel::MatmulLoopVdpbf16ps)
                    } else {
                        schedule_bf16_scalar_k_tile(
                            &scheduled,
                            vector_width,
                            Avx512Kernel::Cpu(CpuKernel::BroadcastVecMultAddBf16F32),
                        )
                    }
                })
            })
        })
    }
}

fn schedule_avx512_bf16_n_tail_tile(
    matmul: &ImplNode<Avx512Target>,
    vector_width: DimSize,
) -> ImplNode<Avx512Target> {
    let k = spec_of(matmul).0.parameter_shape(0)[2].get();
    let dot_product_k = (k.min(512) / 32) * 32;
    if dot_product_k != 0 && dot_product_k < k {
        matmul.split_saturating_ensure_continue(dot_product_k, |k_tile| {
            schedule_avx512_bf16_n_tail_tile(k_tile, vector_width)
        })
    } else {
        schedule_avx512_bf16_n_tail_k_tile(matmul, vector_width)
    }
}

fn schedule_avx512_bf16_n_tail_k_tile(
    matmul: &ImplNode<Avx512Target>,
    vector_width: DimSize,
) -> ImplNode<Avx512Target> {
    let k = spec_of(matmul).0.parameter_shape(0)[2].get();
    if k.is_multiple_of(32) {
        let m = spec_of(matmul).0.parameter_shape(2)[1].get();
        matmul.tile_out_ensure_continue(&[1, m, 1], |n_col_tile| {
            n_col_tile
                .move_relayout(1, L1, layout![0, 2, 1], None)
                .tile_out_ensure_continue(&[1, 1, 1], |scalar_tile| {
                    scalar_tile
                        .move_relayout(0, L1, row_major, None)
                        .move_param(2, RF)
                        .select(Avx512Kernel::DotProductLoopVdpbf16ps)
                })
        })
    } else if vector_width < nz!(8u32) {
        schedule_bf16_scalar_tail_tile(matmul)
    } else {
        schedule_bf16_scalar_k_tile(
            matmul,
            vector_width,
            Avx512Kernel::Cpu(CpuKernel::BroadcastVecMultAddBf16F32),
        )
    }
}

fn schedule_bf16_scalar_k_tile<Tgt: CpuTarget>(
    matmul: &ImplNode<Tgt>,
    vector_width: DimSize,
    kernel: impl Into<Tgt::Kernel> + Copy,
) -> ImplNode<Tgt> {
    matmul.tile_out_ensure_continue(&[1, 1, vector_width.get()], |m_tile| {
        m_tile.split_saturating_ensure_continue(1, |k_tile| {
            k_tile
                .move_param(0, RF)
                .move_vrf(1, VRF, vector_width.get())
                .move_vrf(2, VRF, vector_width.get())
                .select(kernel)
        })
    })
}

fn schedule_broadcast_vec_mult_add_tile<Tgt: CpuTarget>(
    matmul: &ImplNode<Tgt>,
    vector_width: DimSize,
) -> ImplNode<Tgt> {
    matmul
        .move_vrf(2, VRF, vector_width.get())
        .split_saturating_ensure_continue(1, |k_tile| {
            let spec = spec_of(k_tile);
            let n_tile = spec.0.parameter_shape(2)[2].get();
            k_tile
                .move_vrf(1, VRF, vector_width.get())
                .tile_out_ensure_continue(&[1, 1, n_tile], |row_tile| {
                    row_tile
                        .move_param(0, L1)
                        .select(CpuKernel::BroadcastVecMultAdd)
                })
        })
}

fn schedule_bf16_scalar_tail_tile<Tgt: CpuTarget>(matmul: &ImplNode<Tgt>) -> ImplNode<Tgt> {
    matmul.tile_out_ensure_continue(&[1, 1, 1], |m_tile| {
        m_tile.split_saturating_ensure_continue(1, |k_tile| {
            k_tile
                .cast(0, Dtype::Float32, RF, row_major, None)
                .cast(1, Dtype::Float32, RF, row_major, None)
                .move_param(2, RF)
                .select(CpuKernel::MultAdd)
        })
    })
}

fn is_bf16_to_f32_move<Tgt: CpuTarget>(spec: &Spec<Tgt>) -> bool {
    is_move(spec) && {
        let params = spec.0.parameters();
        params.len() == 2
            && params[0].dtype() == Dtype::Bfloat16
            && params[1].dtype() == Dtype::Float32
    }
}

fn schedule_scalar_bf16_to_f32_move<Tgt: CpuTarget>(mov: &ImplNode<Tgt>) -> ImplNode<Tgt> {
    let spec = spec_of(mov);

    let mut scheduled = mov.clone();
    if spec.0.parameter(0).memory() != RF {
        scheduled = scheduled.move_param(0, RF);
    }
    if spec.0.parameter(1).memory() != RF {
        scheduled = scheduled.move_param(1, RF);
    }
    scheduled.select(CpuKernel::CastBf16F32)
}

fn schedule_bf16_to_f32_move<Tgt: CpuTarget>(
    mov: &ImplNode<Tgt>,
    spec: &Spec<Tgt>,
) -> ImplNode<Tgt> {
    let output = spec.0.parameter(1);
    let output_shape = output.shape();
    if output.volume().get() < 16 {
        let tile_shape: Vec<u32> = output_shape.iter().map(|_| 1).collect();
        let scheduled = mov.tile_out_ensure_continue(&tile_shape, schedule_scalar_bf16_to_f32_move);
        return apply_rewrites(scheduled);
    }

    let mut tile_shape: Vec<u32> = output_shape.iter().map(|_| 1).collect();
    let vector_dim = output_shape
        .iter()
        .rposition(|d| d.get() >= 16)
        .expect("volume >= 16 should have a dimension large enough for vector cast");
    tile_shape[vector_dim] = 16;
    let scheduled = mov.tile_out_ensure_continue(&tile_shape, |tile| {
        let tile_spec = spec_of(tile);
        if tile_spec.0.parameter(1).volume().get() < 16 {
            let scalar_shape: Vec<u32> = tile_spec.0.parameter_shape(1).iter().map(|_| 1).collect();
            tile.tile_out_ensure_continue(&scalar_shape, schedule_scalar_bf16_to_f32_move)
        } else {
            tile.move_relayout(0, VRF, row_major, Some(16))
                .move_relayout(1, VRF, row_major, Some(8))
                .select(CpuKernel::VectorCastBf16F32)
        }
    });
    apply_rewrites(scheduled)
}

fn apply_rewrites<Tgt: CpuTarget>(implementation: ImplNode<Tgt>) -> ImplNode<Tgt> {
    implementation.map_spec_leaves(&|spec_app| {
        if !is_move(&spec_app.0) {
            return ImplNode::SpecApp(spec_app);
        }

        // TODO: Remove this wrap n' clone
        let implementation = ImplNode::SpecApp(spec_app.clone());

        let spec: &Spec<Tgt> = &spec_app.0;
        if is_bf16_to_f32_move(spec) {
            return schedule_bf16_to_f32_move(&implementation, spec);
        }

        let params = spec.0.parameters();
        if params.len() == 2
            && params[0].dtype() == params[1].dtype()
            && CpuKernel::Assign.applies_to_logical_spec(&spec.0)
        {
            return implementation.select(CpuKernel::Assign);
        }

        let output_idx = spec.0.unique_output_index().unwrap();
        let new_tile_shape = valid_small_move_tile_shape(spec, output_idx);
        let mut new_impl = implementation.tile_out_ensure(&new_tile_shape);
        let mut changed = false;
        if new_impl.spec().unwrap().0.parameter_shape(output_idx)
            != spec.0.parameter_shape(output_idx)
        {
            changed = true;
        }
        for idx in [0u8, 1] {
            if spec.0.parameter_memory(idx.into()) == GL {
                new_impl = new_impl.move_param(idx, L1);
                changed = true;
            }
        }
        if changed {
            new_impl = apply_rewrites(new_impl);
        }
        new_impl
    })
}

fn valid_small_move_tile_shape<Tgt: CpuTarget>(spec: &Spec<Tgt>, output_idx: usize) -> Vec<u32> {
    let output = spec.0.parameter(output_idx);
    let shape = output.shape();
    let candidate_sizes: Vec<Vec<u32>> = shape
        .iter()
        .map(|d| {
            let max_size = d.get().min(16);
            let preferred = prev_power_of_two_u32(max_size);
            let mut candidates = vec![preferred];
            candidates.extend((1..=max_size).rev().filter(|&s| s != preferred));
            candidates
        })
        .collect();

    let mut chosen = Vec::with_capacity(shape.len());
    find_valid_tile_shape(&output, &candidate_sizes, &mut chosen)
        .expect("at least scalar tiling should apply to every move layout")
}

fn find_valid_tile_shape<Tgt: CpuTarget>(
    output: &morello::tensorspec::TensorSpec<Tgt>,
    candidate_sizes: &[Vec<u32>],
    chosen: &mut Vec<u32>,
) -> Option<Vec<u32>> {
    if chosen.len() == candidate_sizes.len() {
        let shape: Shape = chosen
            .iter()
            .map(|&d| DimSize::try_from(d).unwrap())
            .collect();
        return output
            .is_valid_tile_shape(&shape, false)
            .then(|| chosen.clone());
    }

    for &candidate in &candidate_sizes[chosen.len()] {
        chosen.push(candidate);
        if let Some(shape) = find_valid_tile_shape(output, candidate_sizes, chosen) {
            return Some(shape);
        }
        chosen.pop();
    }

    None
}

fn chain_tile<Tgt, F>(imp: &ImplNode<Tgt>, shapes: &[Shape], inner_fn: &F) -> ImplNode<Tgt>
where
    Tgt: CpuTarget,
    F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
{
    if let Some(next_shape) = shapes.first() {
        let next_shape_u32: SmallVec<[u32; 5]> = next_shape.iter().map(|d| d.get()).collect();
        imp.tile_out_ensure_continue(&next_shape_u32, |child| {
            if spec_of(child).0.unique_output().unwrap().shape() == &next_shape[..] {
                inner_fn(child)
            } else {
                chain_tile(child, &shapes[1..], inner_fn)
            }
        })
    } else {
        inner_fn(imp)
    }
}

trait TileOutContinue<Tgt: CpuTarget>: SchedulingSugar<Tgt> {
    fn tile_out_ensure_continue<F>(&self, output_shape: &[u32], continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>;

    fn split_saturating_ensure_continue<F>(&self, k: u32, continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>;
}

fn tile_out_parallel_ensure_continue<Tgt, F>(
    spec: &Spec<Tgt>,
    output_shape: &[u32],
    continuation: F,
) -> ImplNode<Tgt>
where
    Tgt: CpuTarget,
    F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
{
    apply_fn_to_leaves(&spec.tile_out_parallel_ensure(output_shape), &continuation)
}

impl<Tgt: CpuTarget> TileOutContinue<Tgt> for ImplNode<Tgt> {
    fn tile_out_ensure_continue<F>(&self, output_shape: &[u32], continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        apply_continue_impl(
            self,
            continuation,
            |child, cont| child.tile_out_ensure_continue(output_shape, cont),
            |node| node.tile_out_ensure(output_shape),
        )
    }

    fn split_saturating_ensure_continue<F>(&self, k: u32, continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        apply_continue_impl(
            self,
            continuation,
            |child, cont| child.split_saturating_ensure_continue(k, cont),
            |node| node.split_saturating_ensure(k),
        )
    }
}

fn apply_continue_impl<Tgt, F, RecurseFn, EnsureFn>(
    node: &ImplNode<Tgt>,
    continuation: F,
    recurse_fn: RecurseFn,
    ensure_fn: EnsureFn,
) -> ImplNode<Tgt>
where
    Tgt: CpuTarget,
    F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    RecurseFn: FnOnce(&ImplNode<Tgt>, F) -> ImplNode<Tgt>,
    EnsureFn: FnOnce(&ImplNode<Tgt>) -> ImplNode<Tgt>,
{
    if let Some(default_child_idx) = node.default_child() {
        let mut children = node.children().to_vec();
        children[default_child_idx] = recurse_fn(&children[default_child_idx], continuation);
        node.replace_children(children.into_iter())
    } else {
        apply_fn_to_leaves(&ensure_fn(node), &continuation)
    }
}

fn apply_fn_to_leaves<Tgt, F>(node: &ImplNode<Tgt>, f: &F) -> ImplNode<Tgt>
where
    Tgt: CpuTarget,
    F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
{
    match node {
        ImplNode::SpecApp(_) => f(node),
        _ if node.children().is_empty() => node.clone(),
        _ => map_children(node, |child| apply_fn_to_leaves(child, f)),
    }
}

fn map_children<Tgt, F>(node: &ImplNode<Tgt>, f: F) -> ImplNode<Tgt>
where
    Tgt: Target,
    F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
{
    node.replace_children(node.children().iter().map(f))
}

trait SchedulingSugarExt<Tgt: Target> {
    fn tile_out_saturating(&self, output_shape: &[u32]) -> ImplNode<Tgt>;
    fn tile_out_parallel_saturating(&self, output_shape: &[u32]) -> ImplNode<Tgt>;
    fn tile_out_ensure(&self, output_shape: &[u32]) -> ImplNode<Tgt>;
    fn tile_out_parallel_ensure(&self, output_shape: &[u32]) -> ImplNode<Tgt>;
    fn split_saturating(&self, k: u32) -> ImplNode<Tgt>;
    fn split_saturating_ensure(&self, k: u32) -> ImplNode<Tgt>;
}

trait SpecProvider<Tgt: Target> {
    fn get_spec(&self) -> Option<&Spec<Tgt>>;
    fn into_implnode(self) -> ImplNode<Tgt>;
    fn into_specapp(self) -> SpecApp<ViewE<Tgt>>;
    fn child_count(&self) -> usize;
}

impl<Tgt: Target> SpecProvider<Tgt> for Spec<Tgt> {
    fn get_spec(&self) -> Option<&Spec<Tgt>> {
        Some(self)
    }

    fn into_implnode(self) -> ImplNode<Tgt> {
        self.into_specapp().into()
    }

    fn into_specapp(self) -> SpecApp<ViewE<Tgt>> {
        SpecApp::new_with_default_params(self)
    }

    fn child_count(&self) -> usize {
        0
    }
}

impl<Tgt: Target> SpecProvider<Tgt> for ImplNode<Tgt> {
    fn get_spec(&self) -> Option<&Spec<Tgt>> {
        match self {
            ImplNode::SpecApp(app) => Some(&app.0),
            _ => None,
        }
    }

    fn into_implnode(self) -> ImplNode<Tgt> {
        self
    }

    fn into_specapp(self) -> SpecApp<ViewE<Tgt>> {
        match self {
            ImplNode::SpecApp(app) => app,
            _ => unimplemented!(),
        }
    }

    fn child_count(&self) -> usize {
        self.children().len()
    }
}

fn tile_out_saturating_impl<T, Tgt>(node: &T, output_shape: &[u32], parallel: bool) -> ImplNode<Tgt>
where
    T: SchedulingSugar<Tgt> + SpecProvider<Tgt> + Clone + Debug,
    Tgt: Target,
{
    if node.child_count() != 0 {
        return node
            .clone()
            .into_implnode()
            .apply_to_default_leaf(|spec| tile_out_saturating_impl(spec, output_shape, parallel));
    }

    // Get the current output shape from the spec
    let Some(spec) = node.get_spec() else {
        panic!("Spec not found for node: {node:?}");
    };
    let Some(output_idx) = spec.0.unique_output_index() else {
        return if parallel {
            node.tile_out_parallel(output_shape)
        } else {
            node.tile_out(output_shape)
        };
    };
    let current_shape = spec.0.parameter_shape(output_idx);

    // If the tiling shape is the same as current output, do nothing
    if current_shape.len() == output_shape.len()
        && current_shape
            .iter()
            .zip(output_shape.iter())
            .all(|(c, o)| c.get() <= *o)
    {
        return node.clone().into_specapp().into();
    }

    // Saturate dimensions that are larger than the target
    let saturated_shape: Vec<u32> = current_shape
        .iter()
        .zip(output_shape)
        .map(|(c, &o)| c.get().min(o))
        .collect();

    if parallel {
        node.tile_out_parallel(&saturated_shape)
    } else {
        node.tile_out(&saturated_shape)
    }
}

impl<T, Tgt> SchedulingSugarExt<Tgt> for T
where
    T: SchedulingSugar<Tgt> + SpecProvider<Tgt> + Clone + Debug,
    Tgt: Target,
{
    fn tile_out_saturating(&self, output_shape: &[u32]) -> ImplNode<Tgt> {
        tile_out_saturating_impl(self, output_shape, false)
    }

    fn tile_out_parallel_saturating(&self, output_shape: &[u32]) -> ImplNode<Tgt> {
        tile_out_saturating_impl(self, output_shape, true)
    }

    fn tile_out_ensure(&self, output_shape: &[u32]) -> ImplNode<Tgt> {
        if self.child_count() != 0 {
            return self
                .clone()
                .into_implnode()
                .apply_to_default_leaf(|spec| spec.tile_out_ensure(output_shape));
        }

        let initial_result = self.tile_out_saturating(output_shape);

        // Recursively process the entire tree to ensure all leaves have appropriate output shapes
        tile_out_until_fit(&initial_result, output_shape)
    }

    fn tile_out_parallel_ensure(&self, output_shape: &[u32]) -> ImplNode<Tgt> {
        if self.child_count() != 0 {
            return self
                .clone()
                .into_implnode()
                .apply_to_default_leaf(|spec| spec.tile_out_parallel_ensure(output_shape));
        }

        let initial_result = self.tile_out_parallel_saturating(output_shape);

        // Recursively process the entire tree to ensure all leaves have appropriate output shapes
        tile_out_until_fit(&initial_result, output_shape)
    }

    fn split_saturating(&self, k: u32) -> ImplNode<Tgt> {
        if self.child_count() != 0 {
            return self
                .clone()
                .into_implnode()
                .apply_to_default_leaf(|spec| spec.split_saturating(k));
        }

        let spec = self.get_spec().unwrap();
        if matmul_accum(spec).is_none() {
            unimplemented!();
        }

        let current_k = spec.0.parameter_shape(0)[2].get();
        if current_k <= k {
            return self.clone().into_specapp().into();
        }
        self.split(k)
    }

    fn split_saturating_ensure(&self, k: u32) -> ImplNode<Tgt> {
        if self.child_count() != 0 {
            return self
                .clone()
                .into_implnode()
                .apply_to_default_leaf(|spec| spec.split_saturating_ensure(k));
        }

        let initial_result = self.split_saturating(k);
        hardcore_process_all_splits(&initial_result, k)
    }
}

fn tile_out_until_fit<Tgt: Target>(node: &ImplNode<Tgt>, output_shape: &[u32]) -> ImplNode<Tgt> {
    match node {
        ImplNode::SpecApp(spec_app) => {
            let spec = &spec_app.0;

            if let Some(output_idx) = spec.0.unique_output_index() {
                let current_shape = spec.0.parameter_shape(output_idx);
                if current_shape
                    .iter()
                    .zip(output_shape.iter())
                    .any(|(current, &target)| current.get() > target)
                {
                    let tiled_result = node.tile_out_ensure(output_shape);
                    return tile_out_until_fit(&tiled_result, output_shape);
                }
            }
            node.clone()
        }
        _ => map_children(node, |child| tile_out_until_fit(child, output_shape)),
    }
}

fn hardcore_process_all_splits<Tgt: Target>(node: &ImplNode<Tgt>, k: u32) -> ImplNode<Tgt> {
    match node {
        ImplNode::SpecApp(spec_app) => {
            let spec = &spec_app.0;
            if matmul_accum(spec).is_some() {
                let current_k = spec.0.parameter_shape(0)[2].get();

                if current_k > k {
                    let split_result = node.split_saturating_ensure(k);
                    return hardcore_process_all_splits(&split_result, k);
                }
            }

            node.clone()
        }
        _ => map_children(node, |child| hardcore_process_all_splits(child, k)),
    }
}
