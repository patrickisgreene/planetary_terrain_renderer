use crate::shaders::DEPTH_COPY_SHADER;
use bevy::{
    core_pipeline::{FullscreenShader, core_3d::CORE_3D_DEPTH_FORMAT},
    ecs::entity::EntityHash,
    prelude::*,
    render::{
        Extract,
        camera::ExtractedCamera,
        render_phase::{
            CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem, PhaseItemExtraIndex,
            SortedPhaseItem, ViewSortedRenderPhases,
        },
        render_resource::{binding_types::texture_depth_2d_multisampled, *},
        renderer::{RenderContext, RenderDevice, ViewQuery},
        sync_world::MainEntity,
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, RetainedViewEntity, ViewDepthTexture, ViewTarget},
    },
};
use indexmap::IndexMap;
use std::ops::Range;

pub(crate) const TERRAIN_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32FloatStencil8;

pub struct TerrainItem {
    pub representative_entity: (Entity, MainEntity),
    pub draw_function: DrawFunctionId,
    pub pipeline: CachedRenderPipelineId,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
    pub order: u32,
}

impl PhaseItem for TerrainItem {
    const AUTOMATIC_BATCHING: bool = false;

    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    #[inline]
    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for TerrainItem {
    type SortKey = u32;

    fn sort_key(&self) -> Self::SortKey {
        u32::MAX - self.order
    }

    fn recalculate_sort_keys(
        _items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _view: &ExtractedView,
    ) {
        // The sort key is derived from `order`, which is fixed at insertion time and
        // does not depend on the view, so there is nothing to recalculate here.
    }

    fn indexed(&self) -> bool {
        false
    }
}

impl CachedRenderPipelinePhaseItem for TerrainItem {
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

pub fn extract_terrain_phases(
    mut terrain_phases: ResMut<ViewSortedRenderPhases<TerrainItem>>,
    cameras: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    terrain_phases.clear();

    for (entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }

        terrain_phases.insert(
            RetainedViewEntity {
                main_entity: entity.into(),
                auxiliary_entity: Entity::PLACEHOLDER.into(),
                subview_index: 0,
            },
            default(),
        );
    }
}

#[derive(Component)]
pub struct TerrainViewDepthTexture {
    texture: Texture,
    pub view: TextureView,
    pub depth_view: TextureView,
    pub stencil_view: TextureView,
}

impl TerrainViewDepthTexture {
    pub fn new(texture: CachedTexture) -> Self {
        let depth_view = texture.texture.create_view(&TextureViewDescriptor {
            aspect: TextureAspect::DepthOnly,
            ..default()
        });
        let stencil_view = texture.texture.create_view(&TextureViewDescriptor {
            aspect: TextureAspect::StencilOnly,
            ..default()
        });

        Self {
            texture: texture.texture,
            view: texture.default_view,
            depth_view,
            stencil_view,
        }
    }

    pub fn get_attachment(&self) -> RenderPassDepthStencilAttachment<'_> {
        RenderPassDepthStencilAttachment {
            view: &self.view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(0.0), // Clear depth
                store: StoreOp::Store,
            }),
            stencil_ops: Some(Operations {
                load: LoadOp::Clear(0), // Initialize stencil to 0 (lowest priority)
                store: StoreOp::Store,
            }),
        }
    }
}

pub fn prepare_terrain_depth_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views_3d: Query<(Entity, &ExtractedCamera, &Msaa)>,
) {
    for (view, camera, msaa) in &views_3d {
        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };

        let descriptor = TextureDescriptor {
            label: Some("view_depth_texture"),
            size: Extent3d {
                depth_or_array_layers: 1,
                width: physical_target_size.x,
                height: physical_target_size.y,
            },
            mip_level_count: 1,
            sample_count: msaa.samples(),
            dimension: TextureDimension::D2,
            format: TERRAIN_DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let cached_texture = texture_cache.get(&device, descriptor);

        commands
            .entity(view)
            .insert(TerrainViewDepthTexture::new(cached_texture));
    }
}

#[derive(Resource)]
pub struct DepthCopyPipeline {
    layout: BindGroupLayout,
    id: CachedRenderPipelineId,
}

impl FromWorld for DepthCopyPipeline {
    fn from_world(world: &mut World) -> Self {
        let fullscreen = FullscreenShader::from_world(world);
        let device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_depth_2d_multisampled(),),
        );

        let layout = device.create_bind_group_layout(None, &entries);

        let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: None,
            layout: vec![BindGroupLayoutDescriptor::new(
                "depth_copy_bind_group_layout",
                &entries,
            )],
            immediate_size: Default::default(),
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: world.load_asset(DEPTH_COPY_SHADER),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![],
            }),
            primitive: Default::default(),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true.into(),
                depth_compare: CompareFunction::Always.into(),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState {
                count: 4, // Todo: specialize per camera ...
                ..Default::default()
            },
            zero_initialize_workgroup_memory: false,
        });

        Self { layout, id }
    }
}

pub fn terrain_pass(
    world: &World,
    view: ViewQuery<(
        MainEntity,
        &ExtractedCamera,
        &ViewTarget,
        &ViewDepthTexture,
        &TerrainViewDepthTexture,
    )>,
    mut ctx: RenderContext,
) {
    let render_view = view.entity();
    let (main_view, camera, target, depth, terrain_depth) = view.into_inner();

    let pipeline_cache = world.resource::<PipelineCache>();
    let depth_copy_pipeline = world.resource::<DepthCopyPipeline>();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(depth_copy_pipeline.id) else {
        return;
    };

    let Some(terrain_phase) = world
        .get_resource::<ViewSortedRenderPhases<TerrainItem>>()
        .and_then(|phase| {
            phase.get(&RetainedViewEntity {
                main_entity: main_view.into(),
                auxiliary_entity: Entity::PLACEHOLDER.into(),
                subview_index: 0,
            })
        })
    else {
        return;
    };

    if terrain_phase.items.is_empty() {
        return;
    }

    // Todo: prepare this in a separate system
    let terrain_depth_view = terrain_depth.texture.create_view(&TextureViewDescriptor {
        aspect: TextureAspect::DepthOnly,
        ..default()
    });
    let depth_copy_bind_group = ctx.render_device().create_bind_group(
        None,
        &depth_copy_pipeline.layout,
        &BindGroupEntries::single(&terrain_depth_view),
    );

    let color_attachments = [Some(target.get_color_attachment())];
    let terrain_depth_stencil_attachment = Some(terrain_depth.get_attachment());
    let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

    {
        let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("terrain_pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: terrain_depth_stencil_attachment,
            ..default()
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        terrain_phase
            .render(&mut render_pass, world, render_view)
            .unwrap();
    }

    {
        let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            depth_stencil_attachment,
            ..default()
        });
        render_pass.set_bind_group(0, &depth_copy_bind_group, &[]);
        render_pass.set_render_pipeline(pipeline);
        render_pass.draw(0..3, 0..1);
    }
}
