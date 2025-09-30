use morello::codegen::CodeGen;
use morello::common::{DimSize, Shape};
use morello::db::FilesDatabase;
use morello::grid::canon::CanonicalBimap;
use morello::grid::general::BiMap;
use morello::imp::functions::FunctionApp;
use morello::imp::subspecs::SpecApp;
use morello::imp::{Impl, ImplNode};
use morello::layout;
use morello::layout::row_major;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::SchedulingSugar;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{Avx2Target, CpuMemoryLevel::GL};
use morello::target::{Avx512Target, CpuKernel, CpuMemoryLevel, CpuTarget, Target};
use morello::utils::ToWriteFmt;
use morello::{shape, spec};
use nonzero::nonzero as nz;
use smallvec::SmallVec;
use std::fmt::Debug;
use std::path::Path;
use std::{env, io, iter, process};

const MC: u32 = 528;
const KC: u32 = 528;
const NC: u32 = 1056;
const K_SPLIT: u32 = 48; // TODO: Get rid of this.

fn main() {
    let mut use_avx512 = false;
    let mut integer_args = vec![];
    for arg in env::args().skip(1) {
        if arg == "--avx512" {
            use_avx512 = true;
            continue;
        } else if let Ok(v) = arg.parse::<u32>() {
            integer_args.push(v);
        } else {
            panic!("Unrecognized argument: {}", arg);
        }
    }
    let [batch_size, m, k, n] = integer_args[..] else {
        eprintln!("incorrect arguments");
        process::exit(2);
    };

    if use_avx512 {
        main_per_target::<Avx512Target>(batch_size, m, k, n, nz!(16u32), nz!(48u32), nz!(8u32));
    } else {
        main_per_target::<Avx2Target>(batch_size, m, k, n, nz!(8u32), nz!(16u32), nz!(4u32));
    }
}

fn main_per_target<Tgt>(
    batch_size: u32,
    m: u32,
    k: u32,
    n: u32,
    vec_size: DimSize,
    v_n_size: DimSize,
    mr: DimSize,
) where
    Tgt: CpuTarget,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let db = FilesDatabase::new(
        // Some(Path::new("/tmp/working_matmul_db_2025sep29")),
        None, true, 1, 10_000, 1,
    );

    let mut spec: Spec<Tgt> = spec!(MatmulAccum(
        [batch_size, m, k, n],
        (f32, GL, row_major),
        (f32, GL, row_major),
        (f32, GL, row_major)
    ));
    spec.canonicalize().unwrap();

    let implementation = spec.tile_out_parallel_ensure_continue(&[1, m, n], |s| {
        schedule_serial_matmul(s, m, n, vec_size, v_n_size, mr)
    });
    let implementation = apply_rewrites(&implementation, vec_size, v_n_size);
    let implementation = implementation.synthesize_all(&db);
    implementation
        .emit(
            true,
            Some(ImplPrintStyle::Compact),
            &mut ToWriteFmt(io::stdout()),
        )
        .unwrap_or_else(|e| panic!("Failed to generate code: {}", e));

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

fn schedule_serial_matmul<Tgt: CpuTarget>(
    spec: &ImplNode<Tgt>,
    m: u32,
    n: u32,
    vec_size: DimSize,
    v_n_size: DimSize,
    mr: DimSize,
) -> ImplNode<Tgt> {
    // spec.tile_out_ensure_continue(&[1, (m / 8) * 8, n], |a| {
    //     let ImplNode::SpecApp(SpecApp(spec_a, ..)) = a else {
    //         unreachable!();
    //     };
    //     if spec_a.0.parameter_shape(0)[1].get() >= 8 {
    //         schedule_single_matmul_m_main(a, m, n, vec_size, v_n_size, mr)
    //     } else {
    //         schedule_single_matmul_boundary(a, mr)
    //     }
    // })
    schedule_single_matmul_m_main(spec, m, n, vec_size, v_n_size, mr)
}

fn schedule_single_matmul_m_main<Tgt: CpuTarget>(
    spec_app: &ImplNode<Tgt>,
    m: u32,
    n: u32,
    vec_size: DimSize,
    v_n_size: DimSize,
    mr: DimSize,
) -> ImplNode<Tgt> {
    spec_app.tile_out_ensure_continue(&[1, (m / mr.get()) * mr.get(), n], |a| {
        // layout_a packs the M dimension up to size `mr`. (If smaller than `mr`, layout_a will
        // just be canonicalized to batched column-major.)
        let ImplNode::SpecApp(SpecApp(spec_a, ..)) = a else {
            unreachable!();
        };
        let m_inner = spec_a.0.parameter_shape(0)[1].min(mr);
        let layout_a = layout![0, 1, 2, 1 p(m_inner)];

        a.tile_out_ensure_continue(&[1, MC, n], |b| {
            b.split_saturating_ensure_continue(KC, |c| {
                // TODO: move_relayout(0,..) does some redundant work here
                c.move_relayout(0, CpuMemoryLevel::GL, layout_a.clone(), None)
                    .tile_out_ensure_continue(&[1, MC, (n / v_n_size) * v_n_size.get()], |d| {
                        let ImplNode::SpecApp(SpecApp(spec_d, ..)) = d else {
                            unreachable!();
                        };
                        let n_inner = spec_d.0.parameter_shape(1)[2].min(v_n_size);
                        let layout_b = layout![0, 2, 1, 2 p(n_inner)];

                        d.tile_out_ensure_continue(&[1, MC, NC], |e| {
                            let lb0 = layout_b.clone();
                            let e = e.move_relayout(1, CpuMemoryLevel::GL, lb0, None);
                            chain_tile(
                                &e,
                                &[
                                    shape![1, MC, v_n_size.get()],
                                    shape![1, MC, 8],
                                    shape![1, MC, 4],
                                ],
                                &|f| {
                                    f.tile_out_ensure(&[1, mr.get(), v_n_size.get()])
                                        .split_saturating_ensure_continue(K_SPLIT, |i| {
                                            let ImplNode::SpecApp(SpecApp(spec_i, ..)) = i else {
                                                unreachable!();
                                            };
                                            if spec_i.0.parameter_shape(1)[2] >= v_n_size {
                                                i.move_param(2, CpuMemoryLevel::L1)
                                                    .move_vrf(2, CpuMemoryLevel::VRF, vec_size)
                                                    .split(1)
                                                    .move_param(1, CpuMemoryLevel::L1)
                                                    .move_param(0, CpuMemoryLevel::L1)
                                                    .move_param(0, CpuMemoryLevel::RF)
                                                    .move_vrf(1, CpuMemoryLevel::VRF, vec_size)
                                                    .tile_out(&[1, 1, v_n_size.get()])
                                                    .select(CpuKernel::BroadcastVecMultAdd)
                                            } else if spec_i.0.parameter_shape(1)[2] >= nz!(8u32) {
                                                i.move_param(2, CpuMemoryLevel::L1)
                                                    .move_vrf(2, CpuMemoryLevel::VRF, nz!(8u32))
                                                    .split(1)
                                                    .move_param(1, CpuMemoryLevel::L1)
                                                    .move_param(0, CpuMemoryLevel::L1)
                                                    .move_param(0, CpuMemoryLevel::RF)
                                                    .move_vrf(1, CpuMemoryLevel::VRF, nz!(8u32))
                                                    .tile_out(&[1, 1, 8])
                                                    .select(CpuKernel::BroadcastVecMultAdd)
                                            } else if spec_i.0.parameter_shape(1)[2] >= nz!(4u32) {
                                                i.move_param(2, CpuMemoryLevel::L1)
                                                    .move_vrf(2, CpuMemoryLevel::VRF, nz!(4u32))
                                                    .split(1)
                                                    .move_param(1, CpuMemoryLevel::L1)
                                                    .move_param(0, CpuMemoryLevel::L1)
                                                    .move_param(0, CpuMemoryLevel::RF)
                                                    .move_vrf(1, CpuMemoryLevel::VRF, nz!(4u32))
                                                    .tile_out(&[1, 1, 4])
                                                    .select(CpuKernel::BroadcastVecMultAdd)
                                            } else {
                                                i.move_param(2, CpuMemoryLevel::L1)
                                                    .split(1)
                                                    .move_param(1, CpuMemoryLevel::L1)
                                                    .move_param(0, CpuMemoryLevel::L1)
                                                    .tile_out(&[1, 1, 1])
                                                    .move_param(2, CpuMemoryLevel::RF)
                                                    .move_param(0, CpuMemoryLevel::RF)
                                                    .move_param(1, CpuMemoryLevel::RF)
                                                    .select(CpuKernel::MultAdd)
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

fn apply_rewrites<Tgt: CpuTarget>(
    implementation: &ImplNode<Tgt>,
    vec_size: DimSize,
    v_n_size: DimSize,
) -> ImplNode<Tgt> {
    match implementation {
        ImplNode::SpecApp(spec_app) => {
            let logical_spec = &spec_app.0 .0;
            match &logical_spec {
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::Move,
                        ..
                    },
                    ..,
                ) => schedule_move(implementation, &spec_app.0, vec_size, v_n_size),
                _ => implementation.clone(),
            }
        }
        _ => implementation.replace_children(
            implementation
                .children()
                .iter()
                .map(|xc| apply_rewrites(xc, vec_size, v_n_size)),
        ),
    }
}

fn schedule_move<Tgt: CpuTarget>(
    implementation: &ImplNode<Tgt>,
    spec: &Spec<Tgt>,
    vec_size: DimSize,
    v_n_size: DimSize,
) -> ImplNode<Tgt> {
    if format!("{}", spec.0) == "Move((1×16×16, f32, L1, RM:c1), (1×16×16, f32, L1, <[1,2,1], [Dynamic, Dynamic, Packed(8)]>:c2), serial)"
        || format!("{}", spec.0) == "Move((1×16×16, f32, L1, RM:c1), (1×16×16, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×8×16, f32, L1, RM:c1), (1×8×16, f32, VRF, RM:c1, 16), serial)"
        || format!("{}", spec.0) == "Move((1×16×8, f32, L1, RM:c1), (1×16×8, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×16×4, f32, L1, RM:c1), (1×16×4, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×8×8, f32, L1, RM:c1), (1×8×8, f32, VRF, 8), serial)"
        || format!("{}", spec.0) == "Move((1×16×8, f32, L1, RM:c1), (1×16×8, f32, L1, <[1,2,1], [Dynamic, Dynamic, Packed(8)]>:c2), serial)"
        || format!("{}", spec.0) == "Move((1×8×8, f32, VRF, 8), (1×8×8, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×8×16, f32, L1, RM:c1), (1×8×16, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×8×4, f32, L1, RM:c1), (1×8×4, f32, VRF, 4), serial)"
        || format!("{}", spec.0) == "Move((1×4×16, f32, L1, RM:c1), (1×4×16, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×8×8, f32, L1, RM:c1), (1×8×8, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×4×8, f32, L1, RM:c1), (1×4×8, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×8×4, f32, VRF, 4), (1×8×4, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×8×4, f32, L1, RM:c1), (1×8×4, f32, L1, RM:c1), serial)"
        || format!("{}", spec.0) == "Move((1×8×8, f32, L1, RM:c1), (1×8×8, f32, L1, [2,1]), serial)"
        || format!("{}", spec.0) == "Move((1×4×16, f32, L1, RM:c1), (1×4×16, f32, L1, [2,1]), serial)"
        || format!("{}", spec.0) == "Move((1×4×8, f32, L1, RM:c1), (1×4×8, f32, L1, [2,1]), serial)"
    {
        return apply_rewrites(&implementation.tile_out_saturating(&[1, 1, 16]), vec_size, v_n_size);
    }
    if format!("{}", spec.0) == "Move((1×16×4, f32, L1, RM:c1), (1×16×4, f32, L1, <[1,2,1], [Dynamic, Dynamic, Packed(8)]>:c2), serial)"
        || format!("{}", spec.0) == "Move((1×8×16, f32, L1, RM:c1), (1×8×16, f32, L1, [2,1]), serial)"
        || format!("{}", spec.0) == "Move((1×8×4, f32, L1, RM:c1), (1×8×4, f32, L1, [2,1]), serial)"
    {
        return apply_rewrites(&implementation.tile_out_saturating(&[1, 16, 1]), vec_size, v_n_size);
    }

    let output_idx = spec.0.unique_output_index().unwrap();
    let new_tile_shape: Vec<u32> = spec
        .0
        .parameter_shape(output_idx)
        .iter()
        .map(|o| morello::utils::prev_power_of_two_u32(o.get().min(16)))
        .collect();
    let mut new_impl = implementation.tile_out_ensure(&new_tile_shape);
    let mut changed = false;
    if new_impl.spec().unwrap().0.parameter_shape(output_idx) != spec.0.parameter_shape(output_idx)
    {
        changed = true;
    }
    if spec.0.parameter_level(0) == CpuMemoryLevel::GL {
        new_impl = new_impl.move_param(0, CpuMemoryLevel::L1);
        changed = true;
    }
    if spec.0.parameter_level(1) == CpuMemoryLevel::GL {
        new_impl = new_impl.move_param(1, CpuMemoryLevel::L1);
        changed = true;
    }
    if changed {
        new_impl = apply_rewrites(&new_impl, vec_size, v_n_size);
    }
    new_impl
}

fn chain_tile<Tgt, F>(imp: &ImplNode<Tgt>, shapes: &[Shape], inner_fn: &F) -> ImplNode<Tgt>
where
    Tgt: CpuTarget,
    F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
{
    if let Some(next_shape) = shapes.first() {
        let next_shape_u32: SmallVec<[u32; 5]> = next_shape.iter().map(|d| d.get()).collect();
        imp.tile_out_ensure_continue(&next_shape_u32, |child| {
            let ImplNode::SpecApp(child_app) = child else {
                unreachable!();
            };
            if child_app.0 .0.unique_output().unwrap().shape() == &next_shape[..] {
                inner_fn(child)
            } else {
                chain_tile(child, &shapes[1..], inner_fn)
            }
        })
    } else {
        inner_fn(imp)
    }
}

fn faster_level<Tgt: CpuTarget>(source: Tgt::Level) -> Vec<CpuMemoryLevel> {
    match source.into() {
        CpuMemoryLevel::GL => vec![CpuMemoryLevel::L1],
        CpuMemoryLevel::L1 => vec![CpuMemoryLevel::VRF],
        CpuMemoryLevel::RF => vec![],
        CpuMemoryLevel::VRF => vec![],
    }
}

trait TileOutContinue<Tgt: CpuTarget>: SchedulingSugar<Tgt> {
    fn tile_out_ensure_continue<F>(&self, output_shape: &[u32], continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>;

    fn tile_out_parallel_ensure_continue<F>(
        &self,
        output_shape: &[u32],
        continuation: F,
    ) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>;

    /// Apply `split_saturating_hardcore` with the given k, then apply the continuation function to all SpecApp bodies
    /// in the resulting Loop.
    fn split_saturating_ensure_continue<F>(&self, k: u32, continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>;
}

impl<Tgt: CpuTarget> TileOutContinue<Tgt> for Spec<Tgt> {
    fn tile_out_ensure_continue<F>(&self, _output_shape: &[u32], _continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        todo!()
    }

    fn tile_out_parallel_ensure_continue<F>(
        &self,
        output_shape: &[u32],
        continuation: F,
    ) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        let new_loop = self.tile_out_parallel_ensure(output_shape);
        apply_fn_to_leaves(&new_loop, &continuation)
    }

    fn split_saturating_ensure_continue<F>(&self, _k: u32, _continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        todo!()
    }
}

impl<Tgt: CpuTarget> TileOutContinue<Tgt> for ImplNode<Tgt> {
    fn tile_out_ensure_continue<F>(&self, output_shape: &[u32], continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        if let Some(default_child_idx) = self.default_child() {
            let mut children = self.children().to_vec();
            children[default_child_idx] =
                children[default_child_idx].tile_out_ensure_continue(output_shape, continuation);
            self.replace_children(children.into_iter())
        } else {
            let new_loop = self.tile_out_ensure(output_shape);
            apply_fn_to_leaves(&new_loop, &continuation)
        }
    }

    fn tile_out_parallel_ensure_continue<F>(
        &self,
        output_shape: &[u32],
        continuation: F,
    ) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        if let Some(default_child_idx) = self.default_child() {
            let mut children = self.children().to_vec();
            children[default_child_idx] = children[default_child_idx]
                .tile_out_parallel_ensure_continue(output_shape, continuation);
            self.replace_children(children.into_iter())
        } else {
            let new_loop = self.tile_out_parallel_ensure(output_shape);
            apply_fn_to_leaves(&new_loop, &continuation)
        }
    }

    fn split_saturating_ensure_continue<F>(&self, k: u32, continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        if let Some(default_child_idx) = self.default_child() {
            let mut children = self.children().to_vec();
            children[default_child_idx] =
                children[default_child_idx].split_saturating_ensure_continue(k, continuation);
            self.replace_children(children.into_iter())
        } else {
            let new_loop = self.split_saturating_ensure(k);
            apply_fn_to_leaves(&new_loop, &continuation)
        }
    }
}

fn apply_fn_to_leaves<Tgt, F>(node: &ImplNode<Tgt>, f: &F) -> ImplNode<Tgt>
where
    Tgt: CpuTarget,
    F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
{
    if node.children().is_empty() {
        match node {
            ImplNode::SpecApp(_) => f(node),
            _ => node.clone(),
        }
    } else {
        node.replace_children(node.children().iter().map(|c| match c {
            ImplNode::SpecApp(_) => f(c),
            _ => apply_fn_to_leaves(c, f),
        }))
    }
}

trait SchedulingSugarExt<Tgt: Target> {
    fn tile_out_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt>;
    fn tile_out_parallel_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt>;
    fn tile_out_ensure(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt>;
    fn tile_out_parallel_ensure(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt>;
    fn split_saturating(&self, k: u32) -> morello::imp::ImplNode<Tgt>;
    fn split_saturating_ensure(&self, k: u32) -> morello::imp::ImplNode<Tgt>;
    fn move_param_saturating(&self, source_idx: u8, destination_level: Tgt::Level)
        -> ImplNode<Tgt>;
    fn move_vrf_saturating(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_vector_size: DimSize,
    ) -> ImplNode<Tgt>;
}

trait SpecProvider<Tgt: morello::target::Target> {
    fn get_spec(&self) -> Option<&Spec<Tgt>>;
    fn into_implnode(self) -> morello::imp::ImplNode<Tgt>;
    fn into_specapp(self) -> morello::imp::subspecs::SpecApp<morello::views::ViewE<Tgt>>;
    fn child_count(&self) -> usize;
}

impl<Tgt: morello::target::Target> SpecProvider<Tgt> for Spec<Tgt> {
    fn get_spec(&self) -> Option<&Spec<Tgt>> {
        Some(self)
    }

    fn into_implnode(self) -> morello::imp::ImplNode<Tgt> {
        self.into_specapp().into()
    }

    fn into_specapp(self) -> morello::imp::subspecs::SpecApp<morello::views::ViewE<Tgt>> {
        morello::imp::subspecs::SpecApp::new_with_default_params(self)
    }

    fn child_count(&self) -> usize {
        0
    }
}

impl<Tgt: morello::target::Target> SpecProvider<Tgt> for morello::imp::ImplNode<Tgt> {
    fn get_spec(&self) -> Option<&Spec<Tgt>> {
        match self {
            morello::imp::ImplNode::SpecApp(app) => Some(&app.0),
            _ => None,
        }
    }

    fn into_implnode(self) -> morello::imp::ImplNode<Tgt> {
        self
    }

    fn into_specapp(self) -> morello::imp::subspecs::SpecApp<morello::views::ViewE<Tgt>> {
        match self {
            morello::imp::ImplNode::SpecApp(app) => app,
            _ => unimplemented!(),
        }
    }

    fn child_count(&self) -> usize {
        use morello::imp::Impl;

        self.children().len()
    }
}

impl<T, Tgt> SchedulingSugarExt<Tgt> for T
where
    T: SchedulingSugar<Tgt> + SpecProvider<Tgt> + Clone + Debug,
    Tgt: morello::target::Target,
{
    fn tile_out_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
        if self.child_count() != 0 {
            return apply_to_leaf_spec(&self.clone().into_implnode(), |spec| {
                spec.tile_out_saturating(output_shape)
            });
        }

        // Get the current output shape from the spec
        let Some(spec) = self.get_spec() else {
            panic!("Spec not found for node: {self:?}");
        };
        let Some(output_idx) = spec.0.unique_output_index() else {
            return self.tile_out(output_shape);
        };
        let current_shape = spec.0.parameter_shape(output_idx);

        // If the tiling shape is the same as current output, do nothing
        if current_shape.len() == output_shape.len()
            && current_shape
                .iter()
                .zip(output_shape.iter())
                .all(|(c, o)| c.get() <= *o)
        {
            return self.clone().into_specapp().into();
        }

        // Saturate dimensions that are larger than the target
        let saturated_shape: Vec<u32> = current_shape
            .iter()
            .zip(output_shape)
            .map(|(c, &o)| c.get().min(o))
            .collect();

        self.tile_out(&saturated_shape)
    }

    fn tile_out_parallel_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
        if self.child_count() != 0 {
            return apply_to_leaf_spec(&self.clone().into_implnode(), |spec| {
                spec.tile_out_parallel_saturating(output_shape)
            });
        }

        // Get the current output shape from the spec
        let Some(spec) = self.get_spec() else {
            panic!("Spec not found for node: {self:?}");
        };
        let Some(output_idx) = spec.0.unique_output_index() else {
            return self.tile_out_parallel(output_shape);
        };
        let current_shape = spec.0.parameter_shape(output_idx);

        // If the tiling shape is the same as current output, do nothing
        if current_shape.len() == output_shape.len()
            && current_shape
                .iter()
                .zip(output_shape.iter())
                .all(|(c, o)| c.get() <= *o)
        {
            return self.clone().into_specapp().into();
        }

        // Saturate dimensions that are larger than the target
        let saturated_shape: Vec<u32> = current_shape
            .iter()
            .zip(output_shape)
            .map(|(c, &o)| c.get().min(o))
            .collect();

        self.tile_out_parallel(&saturated_shape)
    }

    fn tile_out_ensure(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
        if self.child_count() != 0 {
            return apply_to_leaf_spec(&self.clone().into_implnode(), |spec| {
                spec.tile_out_ensure(output_shape)
            });
        }

        let initial_result = self.tile_out_saturating(output_shape);

        // Recursively process the entire tree to ensure all leaves have appropriate output shapes
        hardcore_process_all_leaves(&initial_result, output_shape)
    }

    fn tile_out_parallel_ensure(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
        if self.child_count() != 0 {
            return apply_to_leaf_spec(&self.clone().into_implnode(), |spec| {
                spec.tile_out_parallel_ensure(output_shape)
            });
        }

        let initial_result = self.tile_out_parallel_saturating(output_shape);

        // Recursively process the entire tree to ensure all leaves have appropriate output shapes
        hardcore_process_all_leaves(&initial_result, output_shape)
    }

    fn split_saturating(&self, k: u32) -> morello::imp::ImplNode<Tgt> {
        use morello::spec::LogicalSpec;

        if self.child_count() != 0 {
            return apply_to_leaf_spec(&self.clone().into_implnode(), |spec| {
                spec.split_saturating(k)
            });
        }

        let spec = self.get_spec().unwrap();
        let LogicalSpec::Primitive(
            spec::PrimitiveBasics {
                typ: spec::PrimitiveSpecType::Matmul { .. },
                ..
            },
            ..,
        ) = &spec.0
        else {
            unimplemented!();
        };

        let current_k = spec.0.parameter_shape(0)[2].get();
        if current_k <= k {
            return self.clone().into_specapp().into();
        }
        self.split(k)
    }

    fn split_saturating_ensure(&self, k: u32) -> morello::imp::ImplNode<Tgt> {
        if self.child_count() != 0 {
            return apply_to_leaf_spec(&self.clone().into_implnode(), |spec| {
                spec.split_saturating_ensure(k)
            });
        }

        let initial_result = self.split_saturating(k);
        hardcore_process_all_splits(&initial_result, k)
    }

    fn move_param_saturating(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
    ) -> ImplNode<Tgt> {
        if self.child_count() != 0 {
            return apply_to_leaf_spec(&self.clone().into_implnode(), |spec| {
                spec.move_param_saturating(source_idx, destination_level)
            });
        }

        if self
            .get_spec()
            .unwrap()
            .0
            .parameter_level(source_idx.into())
            == destination_level
        {
            return self.clone().into_specapp().into();
        }

        self.move_param(source_idx, destination_level)
    }

    fn move_vrf_saturating(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_vector_size: DimSize,
    ) -> ImplNode<Tgt> {
        if self.child_count() != 0 {
            apply_to_leaf_spec(&self.clone().into_implnode(), |spec| {
                spec.move_vrf_saturating(source_idx, destination_level, destination_vector_size)
            })
        } else if self
            .get_spec()
            .unwrap()
            .0
            .parameter_level(source_idx.into())
            == destination_level
        {
            self.clone().into_specapp().into()
        } else {
            self.move_vrf(source_idx, destination_level, destination_vector_size)
        }
    }
}

/// Recursively processes all SpecApp leaves in the tree to ensure they have output shapes within target
fn hardcore_process_all_leaves<Tgt: morello::target::Target>(
    node: &morello::imp::ImplNode<Tgt>,
    output_shape: &[u32],
) -> morello::imp::ImplNode<Tgt> {
    match node {
        morello::imp::ImplNode::SpecApp(spec_app) => {
            let spec = &spec_app.0;

            // Check if this SpecApp has an output shape that exceeds the target
            if let Some(output_idx) = spec.0.unique_output_index() {
                let current_shape = spec.0.parameter_shape(output_idx);

                // Check if any dimension exceeds the target output shape
                let needs_further_tiling = current_shape
                    .iter()
                    .zip(output_shape.iter())
                    .any(|(current, &target)| current.get() > target);

                if needs_further_tiling {
                    // Use tile_out_hardcore to ensure complete tiling
                    let tiled_result = node.tile_out_ensure(output_shape);
                    return hardcore_process_all_leaves(&tiled_result, output_shape);
                }
            }

            // Return the SpecApp as-is if it doesn't need further tiling
            node.clone()
        }
        _ => {
            // For non-SpecApp nodes, recursively process their children
            let processed_children: Vec<_> = node
                .children()
                .iter()
                .map(|child| hardcore_process_all_leaves(child, output_shape))
                .collect();

            node.replace_children(processed_children.into_iter())
        }
    }
}

/// Recursively processes all SpecApp leaves in the tree to ensure they are properly split by k
fn hardcore_process_all_splits<Tgt: morello::target::Target>(
    node: &morello::imp::ImplNode<Tgt>,
    k: u32,
) -> morello::imp::ImplNode<Tgt> {
    use morello::spec::LogicalSpec;

    match node {
        morello::imp::ImplNode::SpecApp(spec_app) => {
            let spec = &spec_app.0;

            // Check if this SpecApp is a MatmulAccum that needs further splitting
            if let LogicalSpec::Primitive(
                morello::spec::PrimitiveBasics {
                    typ: morello::spec::PrimitiveSpecType::Matmul { .. },
                    ..
                },
                ..,
            ) = &spec.0
            {
                let current_k = spec.0.parameter_shape(0)[2].get();

                if current_k > k {
                    // Apply split_saturating_hardcore recursively to this SpecApp
                    let split_result = node.split_saturating_ensure(k);
                    return hardcore_process_all_splits(&split_result, k);
                }
            }

            // Return the SpecApp as-is if it doesn't need further splitting
            node.clone()
        }
        _ => {
            // For non-SpecApp nodes, recursively process their children
            let processed_children: Vec<_> = node
                .children()
                .iter()
                .map(|child| hardcore_process_all_splits(child, k))
                .collect();

            node.replace_children(processed_children.into_iter())
        }
    }
}

// TODO: Don't copy this function in. Instead, extend the API!
fn apply_to_leaf_spec<Tgt, F>(node: &ImplNode<Tgt>, f: F) -> ImplNode<Tgt>
where
    Tgt: Target,
    F: FnOnce(&Spec<Tgt>) -> ImplNode<Tgt>,
{
    match node {
        ImplNode::SpecApp(app) => ImplNode::from(FunctionApp {
            body: Box::new(f(&app.0)),
            parameters: app.1.clone(),
            spec: Some(app.0.clone()),
        }),
        _ => match &node.children() {
            [] => panic!("Not a Spec application and no children."),
            [child] => node.replace_children(iter::once(apply_to_leaf_spec(child, f))),
            children => {
                if let Some(default_child_idx) = node.default_child() {
                    let replaced_child = apply_to_leaf_spec(&children[default_child_idx], f);
                    node.replace_children(
                        children[..default_child_idx]
                            .iter()
                            .chain(iter::once(&replaced_child))
                            .chain(&children[(default_child_idx + 1)..])
                            .cloned(),
                    )
                } else {
                    panic!("Ambiguous choice of child. Use `subschedule`.")
                }
            }
        },
    }
}
