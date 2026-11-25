//! The root signature the host gives translated shaders, decided in one
//! place: [`HostRootLayout`].
//!
//! The layout lists every root parameter and static sampler with all the
//! choices already made (type, register, space, visibility, volatility,
//! flags), and for each stage where its bindings landed. Whoever creates the
//! host root signature (`emit::host` on Windows) only copies it into D3D12
//! structures, so the game's pipelines and the tests that check the shaders
//! on a GPU build the same table. This module is pure: it builds on every
//! host.
//!
//! Order: per stage, in the order the stages are given, the stage's user
//! data (only when the shader reads any), its root CBVs in the order of
//! [`HostShader::root_cbvs`], then its root SRVs. A view read by two stages
//! is one parameter per stage, each visible to its own stage. Static
//! samplers are `s<n>` per stage, `n` from 0.

use super::shader::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_REGISTER, USER_DATA_SPACE};
use super::{HostShader, StaticSampler, UserData};

/// The graphics stages a host pipeline has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsStage {
    Vertex,
    Pixel,
    Mesh,
}

/// The stages that see a root parameter or sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Every stage (the only one a compute layout uses).
    All,
    Vertex,
    Pixel,
    Mesh,
}

impl From<GraphicsStage> for Visibility {
    fn from(stage: GraphicsStage) -> Self {
        match stage {
            GraphicsStage::Vertex => Visibility::Vertex,
            GraphicsStage::Pixel => Visibility::Pixel,
            GraphicsStage::Mesh => Visibility::Mesh,
        }
    }
}

/// What a root view is bound as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootView {
    Cbv,
    Srv,
}

/// What a root parameter holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// 32-bit root constants.
    Constants { dwords: u32 },
    /// A root descriptor; `volatile` is `DATA_VOLATILE`.
    View { view: RootView, volatile: bool },
}

/// One root parameter: what it holds and where the shader sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostRootParam {
    pub kind: ParamKind,
    pub register: u32,
    pub space: u32,
    pub visibility: Visibility,
}

/// One static sampler: `s<register>` in `space`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostSampler {
    pub sampler: StaticSampler,
    pub register: u32,
    pub space: u32,
    pub visibility: Visibility,
}

/// The root signature flags the host sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootFlags {
    /// `ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT`: every layout but compute's.
    pub input_assembler: bool,
    /// `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`.
    pub resource_heap_indexed: bool,
    /// `SAMPLER_HEAP_DIRECTLY_INDEXED`.
    pub sampler_heap_indexed: bool,
}

/// A root view of a stage: the title's root parameter `slot` is ours `param`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundView {
    pub slot: u8,
    pub view: RootView,
    pub param: u32,
}

/// The values of a stage's root constants, ready to set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootConstants {
    /// The root parameter.
    pub param: u32,
    /// Its size in dwords; `values.len()`.
    pub dwords: u32,
    pub values: Vec<u32>,
}

/// Where one stage's bindings are in the layout.
#[derive(Debug, Clone, Default)]
pub struct HostRootStage {
    user_data: Vec<UserData>,
    /// The root parameter of the user data and its size in dwords.
    constants: Option<(u32, u32)>,
    views: Vec<BoundView>,
}

impl HostRootStage {
    /// The stage's user data with every dword it names set to what `resolve`
    /// gives for its entry (called once per entry, in order; when two entries
    /// name the same dword the last wins) and the rest zero; `None` when the
    /// stage has no root constants.
    pub fn user_data(&self, mut resolve: impl FnMut(&UserData) -> u32) -> Option<RootConstants> {
        let (param, dwords) = self.constants?;
        let resolved: Vec<(u32, u32)> =
            self.user_data.iter().map(|u| (u.index, resolve(u))).collect();
        let values = (0..dwords)
            .map(|i| {
                resolved.iter().rev().find(|(index, _)| *index == i).map_or(0, |&(_, value)| value)
            })
            .collect();
        Some(RootConstants { param, dwords, values })
    }

    /// The root views of the stage: CBVs, then SRVs.
    pub fn views(&self) -> &[BoundView] {
        &self.views
    }
}

/// The root signature for the stages of one host pipeline.
#[derive(Debug, Clone)]
pub struct HostRootLayout {
    params: Vec<HostRootParam>,
    samplers: Vec<HostSampler>,
    flags: RootFlags,
    stages: Vec<HostRootStage>,
}

impl HostRootLayout {
    /// The layout of a compute pipeline.
    pub fn compute(cs: &HostShader) -> Self {
        Self::build(&[(cs, Visibility::All)], false)
    }

    /// The layout of a graphics pipeline: the stages in the order given
    /// (vertex alone, vertex and pixel, or mesh and pixel).
    pub fn graphics(stages: &[(&HostShader, GraphicsStage)]) -> Self {
        let stages: Vec<(&HostShader, Visibility)> =
            stages.iter().map(|&(shader, stage)| (shader, stage.into())).collect();
        Self::build(&stages, true)
    }

    fn build(stages: &[(&HostShader, Visibility)], input_assembler: bool) -> Self {
        let mut params = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::with_capacity(stages.len());
        for &(shader, visibility) in stages {
            let mut stage =
                HostRootStage { user_data: shader.user_data.clone(), ..HostRootStage::default() };
            let dwords = shader.user_data_dwords();
            if dwords > 0 {
                stage.constants = Some((next_index(&params), dwords));
                params.push(HostRootParam {
                    kind: ParamKind::Constants { dwords },
                    register: USER_DATA_REGISTER,
                    space: USER_DATA_SPACE,
                    visibility,
                });
            }
            for (slots, view) in
                [(&shader.root_cbvs, RootView::Cbv), (&shader.root_srvs, RootView::Srv)]
            {
                for &slot in slots {
                    stage.views.push(BoundView { slot, view, param: next_index(&params) });
                    params.push(HostRootParam {
                        kind: ParamKind::View { view, volatile: true },
                        register: u32::from(slot),
                        space: ROOT_CBV_SPACE,
                        visibility,
                    });
                }
            }
            samplers.extend(shader.static_samplers.iter().zip(0..).map(|(&sampler, register)| {
                HostSampler { sampler, register, space: SAMPLER_SPACE, visibility }
            }));
            bound.push(stage);
        }
        HostRootLayout {
            params,
            samplers,
            flags: RootFlags {
                input_assembler,
                resource_heap_indexed: true,
                sampler_heap_indexed: true,
            },
            stages: bound,
        }
    }

    /// The root parameters, in root signature order.
    pub fn params(&self) -> &[HostRootParam] {
        &self.params
    }

    pub fn static_samplers(&self) -> &[HostSampler] {
        &self.samplers
    }

    pub fn flags(&self) -> RootFlags {
        self.flags
    }

    /// One entry per stage, in the order the stages were given.
    pub fn stages(&self) -> &[HostRootStage] {
        &self.stages
    }
}

/// The index the next parameter pushed onto `params` will have.
fn next_index(params: &[HostRootParam]) -> u32 {
    // A root signature holds at most 64 DWORDs, so far fewer parameters
    // than `u32::MAX`; saturating keeps the function total.
    u32::try_from(params.len()).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{HwStage, RootType};

    fn shader(stage: HwStage) -> HostShader {
        HostShader {
            stage,
            dxil: Vec::new(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            wave_size: None,
        }
    }

    fn data(index: u32, slot: u8, root: RootType, dword: u32) -> UserData {
        UserData { index, slot, root, dword }
    }

    fn constants(dwords: u32, visibility: Visibility) -> HostRootParam {
        HostRootParam { kind: ParamKind::Constants { dwords }, register: 0, space: 0, visibility }
    }

    fn view(view: RootView, register: u32, visibility: Visibility) -> HostRootParam {
        HostRootParam {
            kind: ParamKind::View { view, volatile: true },
            register,
            space: ROOT_CBV_SPACE,
            visibility,
        }
    }

    #[test]
    fn a_compute_layout_is_user_data_then_views_then_samplers() {
        let mut cs = shader(HwStage::Cs);
        cs.user_data = vec![data(2, 0, RootType::Constants, 0)];
        cs.root_cbvs = vec![3];
        cs.root_srvs = vec![5];
        cs.static_samplers = vec![StaticSampler::from_words([0; 4])];
        let layout = HostRootLayout::compute(&cs);
        assert_eq!(
            layout.params(),
            [
                constants(4, Visibility::All),
                view(RootView::Cbv, 3, Visibility::All),
                view(RootView::Srv, 5, Visibility::All),
            ]
        );
        let [sampler] = layout.static_samplers() else {
            panic!("one sampler expected");
        };
        assert_eq!(sampler.register, 0);
        assert_eq!(sampler.space, SAMPLER_SPACE);
        assert_eq!(sampler.visibility, Visibility::All);
        assert_eq!(
            layout.stages()[0].views(),
            [
                BoundView { slot: 3, view: RootView::Cbv, param: 1 },
                BoundView { slot: 5, view: RootView::Srv, param: 2 },
            ]
        );
    }

    #[test]
    fn a_view_two_stages_read_is_a_parameter_per_stage() {
        let mut vs = shader(HwStage::Vs);
        vs.root_cbvs = vec![3];
        let mut ps = shader(HwStage::Ps);
        ps.root_cbvs = vec![3];
        let layout =
            HostRootLayout::graphics(&[(&vs, GraphicsStage::Vertex), (&ps, GraphicsStage::Pixel)]);
        assert_eq!(
            layout.params(),
            [view(RootView::Cbv, 3, Visibility::Vertex), view(RootView::Cbv, 3, Visibility::Pixel),]
        );
        assert_eq!(layout.stages()[0].views()[0].param, 0);
        assert_eq!(layout.stages()[1].views()[0].param, 1);
    }

    #[test]
    fn a_stage_without_user_data_has_no_constants() {
        let layout = HostRootLayout::compute(&shader(HwStage::Cs));
        assert!(layout.params().is_empty());
        assert_eq!(layout.stages()[0].user_data(|_| panic!("nothing to resolve")), None);
    }

    #[test]
    fn only_compute_layouts_leave_out_the_input_assembler() {
        let (vs, ps, ms, cs) =
            (shader(HwStage::Vs), shader(HwStage::Ps), shader(HwStage::Ms), shader(HwStage::Cs));
        let compute = HostRootLayout::compute(&cs).flags();
        let vertex =
            HostRootLayout::graphics(&[(&vs, GraphicsStage::Vertex), (&ps, GraphicsStage::Pixel)])
                .flags();
        let mesh =
            HostRootLayout::graphics(&[(&ms, GraphicsStage::Mesh), (&ps, GraphicsStage::Pixel)])
                .flags();
        assert!(!compute.input_assembler);
        assert!(vertex.input_assembler);
        assert!(mesh.input_assembler);
        for flags in [compute, vertex, mesh] {
            assert!(flags.resource_heap_indexed && flags.sampler_heap_indexed);
        }
    }

    #[test]
    fn parameters_follow_stage_then_kind_order() {
        let sampler = StaticSampler::from_words([0; 4]);
        let mut vs = shader(HwStage::Vs);
        vs.user_data = vec![data(0, 0, RootType::Constants, 0)];
        vs.root_cbvs = vec![7, 2];
        vs.root_srvs = vec![4];
        vs.static_samplers = vec![sampler, sampler];
        let mut ps = shader(HwStage::Ps);
        ps.user_data = vec![data(0, 0, RootType::Constants, 0)];
        ps.root_srvs = vec![1];
        ps.static_samplers = vec![sampler];
        let layout =
            HostRootLayout::graphics(&[(&vs, GraphicsStage::Vertex), (&ps, GraphicsStage::Pixel)]);
        assert_eq!(
            layout.params(),
            [
                constants(4, Visibility::Vertex),
                view(RootView::Cbv, 7, Visibility::Vertex),
                view(RootView::Cbv, 2, Visibility::Vertex),
                view(RootView::Srv, 4, Visibility::Vertex),
                constants(4, Visibility::Pixel),
                view(RootView::Srv, 1, Visibility::Pixel),
            ]
        );
        let samplers: Vec<(u32, Visibility)> =
            layout.static_samplers().iter().map(|s| (s.register, s.visibility)).collect();
        assert_eq!(
            samplers,
            [(0, Visibility::Vertex), (1, Visibility::Vertex), (0, Visibility::Pixel),]
        );
    }

    #[test]
    fn user_data_fills_each_dword_it_names() {
        let mut cs = shader(HwStage::Cs);
        cs.user_data = vec![data(5, 1, RootType::Table, 0), data(0, 2, RootType::Constants, 2)];
        cs.root_cbvs = vec![9];
        let layout = HostRootLayout::compute(&cs);
        let filled = layout.stages()[0]
            .user_data(|u| match u.root {
                RootType::Table => 50,
                _ => 20 + u.dword,
            })
            .expect("the shader reads user data");
        assert_eq!(filled.param, 0);
        assert_eq!(filled.dwords, 8);
        assert_eq!(filled.values, [22, 0, 0, 0, 0, 50, 0, 0]);
    }

    #[test]
    fn every_stage_gets_an_entry() {
        let mut vs = shader(HwStage::Vs);
        vs.root_cbvs = vec![1];
        let alone = HostRootLayout::graphics(&[(&vs, GraphicsStage::Vertex)]);
        assert_eq!(alone.stages().len(), 1);
        let ps = shader(HwStage::Ps);
        let both =
            HostRootLayout::graphics(&[(&vs, GraphicsStage::Vertex), (&ps, GraphicsStage::Pixel)]);
        assert_eq!(both.stages().len(), 2);
        assert!(both.stages()[1].views().is_empty());
    }
}
