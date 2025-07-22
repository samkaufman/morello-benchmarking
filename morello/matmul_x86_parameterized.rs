use morello::codegen::CodeGen;
use morello::common::DimSize;
use morello::cost::Cost;
use morello::imp::{Impl, ImplNode};
use morello::layout::{row_major, Layout, PhysDim};
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{CpuKernel, CpuMemoryLevel, Target};
use morello::target::{
    CpuMemoryLevel::{GL, L1, RF, VRF},
    X86Target,
};
use morello::utils::ToWriteFmt;
use morello::{shape, spec};
use nonzero::nonzero as nz;
use std::env;
use std::fmt::Debug;
use std::io;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n = args[3].parse::<u32>().unwrap();
    let mut spec: Spec<X86Target> = spec!(MatmulAccum(
        [
            1,
            args[1].parse::<u32>().unwrap(),
            args[2].parse::<u32>().unwrap(),
            n
        ],
        (f32, GL, row_major),
        (f32, GL, row_major),
        (f32, GL, row_major),
        serial
    ));
    spec.canonicalize().unwrap();

    let mat1_pack_size = nz!(16u32);
    let layout_b = Layout::new(vec![
        (0, PhysDim::Dynamic),
        (2, PhysDim::Dynamic),
        (1, PhysDim::Dynamic),
        (2, PhysDim::Packed(mat1_pack_size)),
    ]);

    let implementation = spec.split_saturating_ensure_continue(128, |s| {
        s.move_relayout(1, GL, layout_b.clone(), None)
            .subschedule(&[0], |pack_b| {
                // TODO: This stinks. Use vectors at least.
                pack_b
                    .tile_out(&[1, 1, 1])
                    .move_param(0, L1)
                    .move_param(1, L1)
                    .move_param(0, RF)
                    .subschedule(&[0], |m0| m0.select(CpuKernel::ValueAssign))
                    .subschedule(&[1], |m0| m0.select(CpuKernel::ValueAssign))
            })
            .tile_out_ensure_continue(&[1, 128, 1024.min(n)], |a| {
                a.tile_out_ensure_continue(&[1, 6, 16], |b| {
                    b.move_param_saturating(0, L1)
                        .move_param_saturating(1, L1)
                        .move_param_saturating(2, L1)
                        .move_vrf_saturating(2, VRF, nz!(8u32))
                        .split_saturating(1)
                        .tile_out_saturating(&[1, 1, 16])
                        .move_vrf_saturating(1, VRF, nz!(8u32))
                        .move_vrf_saturating(2, VRF, nz!(8u32))
                        .select(CpuKernel::BroadcastVecMultAdd)
                })
            })
    });
    let implementation = apply_rewrites(&implementation);
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
    let iterations = std::env::var("CHERRYBENCH_LOOP_STEPS")
        .unwrap()
        .parse::<u32>()
        .unwrap();
    let result = implementation
        .bench(iterations, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {}", e));
    for duration in &result.inner_loop_runtimes {
        println!("run: {:.4}s", duration.as_secs_f64());
    }
}

/// Traverses an ImplNode tree and replaces any SpecApp containing a Move with VectorAssign and
/// any MatmulAccum with a tiled BroadcastVecMultAdd.
fn apply_rewrites(implementation: &ImplNode<X86Target>) -> ImplNode<X86Target> {
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
                ) => schedule_move(implementation, logical_spec),
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::Matmul { accum: true },
                        ..
                    },
                    ..,
                ) => schedule_matmul_macrokernel(implementation, logical_spec),
                _ => implementation.clone(),
            }
        }
        _ => implementation.replace_children(implementation.children().iter().map(apply_rewrites)),
    }
}

fn schedule_move(
    implementation: &ImplNode<X86Target>,
    spec: &LogicalSpec<X86Target>,
) -> ImplNode<X86Target> {
    let has_rf_param = spec.parameter_level(0) == CpuMemoryLevel::RF
        || spec.parameter_level(1) == CpuMemoryLevel::RF;
    let is_scalar_move = spec.parameter_shape(0).iter().all(|d| d.get() == 1);

    if has_rf_param && is_scalar_move {
        apply_rewrites(&implementation.select(CpuKernel::ValueAssign))
    } else {
        apply_rewrites(
            &implementation
                .tile_out(&[1, 1, 8])
                .select(CpuKernel::VectorAssign),
        )
    }
}

fn schedule_matmul_macrokernel(
    implementation: &ImplNode<X86Target>,
    spec: &LogicalSpec<X86Target>,
) -> ImplNode<X86Target> {
    let param_levels = (
        spec.parameter_level(0),
        spec.parameter_level(1),
        spec.parameter_level(2),
    );
    if param_levels == (CpuMemoryLevel::L1, CpuMemoryLevel::VRF, CpuMemoryLevel::VRF) {
        return apply_rewrites(&implementation.move_param(0, CpuMemoryLevel::RF));
    }
    if param_levels == (CpuMemoryLevel::RF, CpuMemoryLevel::VRF, CpuMemoryLevel::VRF) {
        return schedule_matmul_microkernel(implementation, spec);
    }
    implementation.clone()
}

/// Handles MatmulAccum when parameters are in optimal memory locations
fn schedule_matmul_microkernel(
    implementation: &ImplNode<X86Target>,
    spec: &LogicalSpec<X86Target>,
) -> ImplNode<X86Target> {
    let expected_shapes = [shape![1, 1, 1], shape![1, 1, 16], shape![1, 1, 16]];
    let param_shapes = spec.parameter_shapes();
    let output_shape = spec.parameter_shape(2);

    if param_shapes == expected_shapes {
        implementation.select(CpuKernel::BroadcastVecMultAdd)
    } else if output_shape != shape![1, 1, 16] {
        apply_rewrites(&implementation.tile_out(&[1, 1, 16]))
    } else {
        apply_rewrites(&implementation.split(1))
    }
}

trait TileOutContinue<Tgt: Target>: SchedulingSugar<Tgt> {
    fn tile_out_ensure_continue<F>(&self, output_shape: &[u32], continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>;

    /// Apply `split_saturating_hardcore` with the given k, then apply the continuation function to all SpecApp bodies
    /// in the resulting Loop.
    fn split_saturating_ensure_continue<F>(&self, k: u32, continuation: F) -> ImplNode<Tgt>
    where
        F: Fn(&ImplNode<Tgt>) -> ImplNode<Tgt>;
}

impl<T> TileOutContinue<X86Target> for T
where
    T: SchedulingSugar<X86Target> + SpecProvider<X86Target> + Clone + Debug,
{
    fn tile_out_ensure_continue<F>(
        &self,
        output_shape: &[u32],
        continuation: F,
    ) -> ImplNode<X86Target>
    where
        F: Fn(&ImplNode<X86Target>) -> ImplNode<X86Target>,
    {
        apply_fn_to_leaves(&self.tile_out_ensure(output_shape), &continuation)
    }

    fn split_saturating_ensure_continue<F>(&self, k: u32, continuation: F) -> ImplNode<X86Target>
    where
        F: Fn(&ImplNode<X86Target>) -> ImplNode<X86Target>,
    {
        apply_fn_to_leaves(&self.split_saturating_ensure(k), &continuation)
    }
}

fn apply_fn_to_leaves<F>(node: &ImplNode<X86Target>, f: &F) -> ImplNode<X86Target>
where
    F: Fn(&ImplNode<X86Target>) -> ImplNode<X86Target>,
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

trait SchedulingSugarExt<Tgt: morello::target::Target> {
    fn tile_out_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt>;
    fn tile_out_ensure(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt>;
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
            return morello::scheduling_sugar::apply_to_leaf_spec(
                &self.clone().into_implnode(),
                |spec| spec.tile_out_saturating(output_shape),
            );
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

    fn tile_out_ensure(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
        if self.child_count() != 0 {
            return morello::scheduling_sugar::apply_to_leaf_spec(
                &self.clone().into_implnode(),
                |spec| spec.tile_out_ensure(output_shape),
            );
        }

        let initial_result = self.tile_out_saturating(output_shape);

        // Recursively process the entire tree to ensure all leaves have appropriate output shapes
        hardcore_process_all_leaves(&initial_result, output_shape)
    }

    fn split_saturating(&self, k: u32) -> morello::imp::ImplNode<Tgt> {
        use morello::spec::LogicalSpec;

        if self.child_count() != 0 {
            return morello::scheduling_sugar::apply_to_leaf_spec(
                &self.clone().into_implnode(),
                |spec| spec.split_saturating(k),
            );
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
            return morello::scheduling_sugar::apply_to_leaf_spec(
                &self.clone().into_implnode(),
                |spec| spec.split_saturating_ensure(k),
            );
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
            return morello::scheduling_sugar::apply_to_leaf_spec(
                &self.clone().into_implnode(),
                |spec| spec.move_param_saturating(source_idx, destination_level),
            );
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
            return morello::scheduling_sugar::apply_to_leaf_spec(
                &self.clone().into_implnode(),
                |spec| {
                    spec.move_vrf_saturating(source_idx, destination_level, destination_vector_size)
                },
            );
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

        self.move_vrf(source_idx, destination_level, destination_vector_size)
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
