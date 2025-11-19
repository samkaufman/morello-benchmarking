use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype, Shape};
use morello::db::FilesDatabase;
use morello::grid::canon::CanonicalBimap;
use morello::grid::general::BiMap;
use morello::imp::{subspecs::SpecApp, Impl, ImplNode};
use morello::layout;
use morello::layout::row_major;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::SchedulingSugar;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{
    Avx2Target, Avx512Target,
    CpuMemoryLevel::{GL, L1, VRF},
    CpuTarget, Target,
};
use morello::target::{CpuMemoryLevel, MemoryLevel};
use morello::utils::ToWriteFmt;
use morello::{shape, spec};
use nonzero::nonzero as nz;
use smallvec::SmallVec;
use std::{env, fmt::Debug, io, process};

const MC: u32 = 528;
const KC: u32 = 528;
const NC: u32 = 1056;

fn main() {
    let mut use_avx512 = false;
    let mut integer_args = vec![];
    let mut db_path: Option<String> = None;

    let mut args_iter = env::args().skip(1);
    while let Some(arg) = args_iter.next() {
        if arg == "--avx512" {
            use_avx512 = true;
            continue;
        } else if arg == "--db" {
            db_path = args_iter.next();
            if db_path.is_none() {
                panic!("--db flag requires a path argument");
            }
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
        main_per_target::<Avx512Target>(batch_size, m, k, n, nz!(48u32), nz!(8u32), db_path);
    } else {
        main_per_target::<Avx2Target>(batch_size, m, k, n, nz!(16u32), nz!(4u32), db_path);
    }
}

fn main_per_target<Tgt>(
    batch_size: u32,
    m: u32,
    k: u32,
    n: u32,
    v_n_size: DimSize,
    mr: DimSize,
    db_path: Option<String>,
) where
    Tgt: CpuTarget,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let db_path_ref = db_path.as_deref().map(std::path::Path::new);
    let db = FilesDatabase::new::<Tgt>(db_path_ref, true, 1, 10_000, 1);

    let mut spec: Spec<Tgt> = spec!(MatmulAccum(
        [batch_size, m, k, n],
        (f32, GL, row_major),
        (f32, GL, row_major),
        (f32, GL, row_major)
    ));
    spec.0.set_serial_only(batch_size == 1);
    spec.canonicalize().unwrap();

    let implementation = spec.tile_out_parallel_ensure_continue(&[1, m, n], |s| {
        schedule_matmul_serial(s, m, n, v_n_size, mr)
    });
    let implementation = apply_rewrites(implementation);
    let implementation = implementation.synthesize_all(&db);
    implementation
        .emit(
            false,
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

fn schedule_matmul_serial<Tgt: CpuTarget>(
    spec_app: &ImplNode<Tgt>,
    m: u32,
    n: u32,
    v_n_size: DimSize,
    mr: DimSize,
) -> ImplNode<Tgt> {
    // vec_size is largest register name divided by size of f32
    let vec_size = DimSize::try_from(
        *Tgt::Level::from(CpuMemoryLevel::VRF)
            .vector_bytes()
            .iter()
            .max()
            .unwrap()
            / Dtype::Float32.size() as u32,
    )
    .unwrap();

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
                let stripped_n = (n / v_n_size) * v_n_size.get(); // peels some off for Packed dim.
                c.move_relayout(0, GL, layout_a.clone(), None)
                    .tile_out_ensure_continue(&[1, MC, stripped_n], |d| {
                        let ImplNode::SpecApp(SpecApp(spec_d, ..)) = d else {
                            unreachable!();
                        };
                        let n_inner = spec_d.0.parameter_shape(1)[2].min(v_n_size);
                        let layout_b = layout![0, 2, 1, 2 p(n_inner)];

                        d.tile_out_ensure_continue(&[1, MC, NC], |e| {
                            let ImplNode::SpecApp(SpecApp(spec_e, ..)) = &e else {
                                unreachable!();
                            };

                            let lb0 = layout_b.clone();
                            let e = e.move_relayout(1, GL, lb0, None);
                            let mc_tile_size = spec_e.0.parameter_shape(0)[1].get();
                            chain_tile(
                                &e,
                                &[
                                    shape![1, mc_tile_size, v_n_size.get()],
                                    shape![1, mc_tile_size, 8],
                                    shape![1, mc_tile_size, 4],
                                ],
                                &|f| {
                                    f.tile_out_ensure_continue(
                                        &[1, mr.get(), v_n_size.get()],
                                        |i| {
                                            let ImplNode::SpecApp(SpecApp(spec_i, ..)) = &i else {
                                                unreachable!();
                                            };
                                            let j = i.move_param(2, L1);
                                            if spec_i.0.parameter_shape(1)[2] < nz!(4u32) {
                                                j.split(1)
                                            } else {
                                                let width = [v_n_size, nz!(8u32), nz!(4u32)]
                                                    .into_iter()
                                                    .find(|&s| spec_i.0.parameter_shape(1)[2] >= s)
                                                    .unwrap();
                                                let v = vec_size.min(width);
                                                j.move_vrf(2, VRF, v).split(1)
                                            }
                                        },
                                    )
                                },
                            )
                        })
                    })
            })
        })
    })
}

fn apply_rewrites<Tgt: CpuTarget>(implementation: ImplNode<Tgt>) -> ImplNode<Tgt> {
    implementation.map_spec_leaves(&|spec_app| {
        if !matches!(
            &spec_app.0 .0,
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Move,
                    ..
                },
                ..
            )
        ) {
            return ImplNode::SpecApp(spec_app);
        }

        // TODO: Remove this wrap n' clone
        let implementation = ImplNode::SpecApp(spec_app.clone());

        let spec: &Spec<Tgt> = &spec_app.0;
        let output_idx = spec.0.unique_output_index().unwrap();
        let new_tile_shape: Vec<u32> = spec
            .0
            .parameter_shape(output_idx)
            .iter()
            .map(|o| morello::utils::prev_power_of_two_u32(o.get().min(16)))
            .collect();
        let mut new_impl = implementation.tile_out_ensure(&new_tile_shape);
        let mut changed = false;
        if new_impl.spec().unwrap().0.parameter_shape(output_idx)
            != spec.0.parameter_shape(output_idx)
        {
            changed = true;
        }
        for idx in [0u8, 1] {
            if spec.0.parameter_level(idx.into()) == GL {
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
        apply_continue_impl(
            self,
            continuation,
            |child, cont| child.tile_out_ensure_continue(output_shape, cont),
            |node| node.tile_out_ensure(output_shape),
        )
    }

    fn tile_out_parallel_ensure_continue<F>(
        &self,
        output_shape: &[u32],
        continuation: F,
    ) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>,
    {
        apply_continue_impl(
            self,
            continuation,
            |child, cont| child.tile_out_parallel_ensure_continue(output_shape, cont),
            |node| node.tile_out_parallel_ensure(output_shape),
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

fn tile_out_saturating_impl<T, Tgt>(
    node: &T,
    output_shape: &[u32],
    parallel: bool,
) -> morello::imp::ImplNode<Tgt>
where
    T: SchedulingSugar<Tgt> + SpecProvider<Tgt> + Clone + Debug,
    Tgt: morello::target::Target,
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
    Tgt: morello::target::Target,
{
    fn tile_out_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
        tile_out_saturating_impl(self, output_shape, false)
    }

    fn tile_out_parallel_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
        tile_out_saturating_impl(self, output_shape, true)
    }

    fn tile_out_ensure(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
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

    fn tile_out_parallel_ensure(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
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

    fn split_saturating(&self, k: u32) -> morello::imp::ImplNode<Tgt> {
        use morello::spec::LogicalSpec;

        if self.child_count() != 0 {
            return self
                .clone()
                .into_implnode()
                .apply_to_default_leaf(|spec| spec.split_saturating(k));
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
            return self
                .clone()
                .into_implnode()
                .apply_to_default_leaf(|spec| spec.split_saturating_ensure(k));
        }

        let initial_result = self.split_saturating(k);
        hardcore_process_all_splits(&initial_result, k)
    }
}

/// Recursively processes all SpecApp leaves in the tree to ensure they have output shapes within target
fn tile_out_until_fit<Tgt: morello::target::Target>(
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
                    return tile_out_until_fit(&tiled_result, output_shape);
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
                .map(|child| tile_out_until_fit(child, output_shape))
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
