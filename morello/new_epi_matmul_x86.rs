trait SchedulingSugarExt<Tgt: morello::target::Target> {
    fn tile_out_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt>;
    fn split_saturating(&self, k: u32) -> morello::imp::ImplNode<Tgt>;
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
    T: SchedulingSugar<Tgt> + SpecProvider<Tgt> + Clone,
    Tgt: morello::target::Target,
{
    fn tile_out_saturating(&self, output_shape: &[u32]) -> morello::imp::ImplNode<Tgt> {
        if self.child_count() != 0 {
            // TODO: Will this lose Spec arguments?
            return morello::scheduling_sugar::apply_to_leaf_spec(
                &self.clone().into_implnode(),
                |spec| spec.tile_out_saturating(output_shape),
            );
        }

        // Get the current output shape from the spec
        let spec = self.get_spec().unwrap();
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

    fn split_saturating(&self, k: u32) -> morello::imp::ImplNode<Tgt> {
        use morello::spec::LogicalSpec;

        if self.child_count() != 0 {
            // TODO: Will this lose Spec arguments?
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
}
