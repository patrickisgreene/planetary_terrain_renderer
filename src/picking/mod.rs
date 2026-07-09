use crate::{
    render::{TerrainViewDepthTexture, terrain_pass},
    shaders::PICKING_SHADER,
};
use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{Core3dSystems, core_3d::main_opaque_pass_3d, schedule::Core3d},
    ecs::{lifecycle::HookContext, query::QueryItem, world::DeferredWorld},
    prelude::*,
    render::{
        RenderApp,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{
                storage_buffer, texture_2d_multisampled, texture_depth_2d_multisampled,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        storage::{GpuShaderBuffer, ShaderBuffer},
        sync_component::SyncComponent,
    },
    window::PrimaryWindow,
};
use big_space::prelude::*;

pub fn picking_system(
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform, &CellCoord, &PickingData)>,
) {
    let Ok(window) = window.single() else {
        return;
    };
    let Some(position) = window.cursor_position() else {
        return;
    };
    let cursor_coords = Vec2::new(position.x, window.size().y - position.y) / window.size();

    for (camera, global_transform, &cell, picking_data) in &camera {
        let mut buffer = buffers.get_mut(&picking_data.buffer).unwrap();
        let data = GpuPickingData {
            cursor_coords,
            depth: 0.0,
            stencil: 255,
            world_from_clip: global_transform.to_matrix() * camera.clip_from_view().inverse(),
            cell: IVec3::new(cell.x, cell.y, cell.z),
        };
        buffer.set_data(data);
    }
}

pub fn picking_readback(trigger: On<ReadbackComplete>, mut picking_data: Query<&mut PickingData>) {
    let GpuPickingData {
        cursor_coords,
        depth,
        stencil: _stencil,
        world_from_clip,
        cell,
    } = trigger.event().to_shader_type();

    let ndc_coords = (2.0 * cursor_coords - 1.0).extend(depth);

    let mut picking_data = picking_data.get_mut(trigger.entity).unwrap();
    picking_data.cursor_coords = cursor_coords;
    picking_data.cell = CellCoord::new(cell.x, cell.y, cell.z);
    picking_data.translation = (depth > 0.0).then(|| world_from_clip.project_point3(ndc_coords));
    picking_data.world_from_clip = world_from_clip;

    // dbg!(cursor_coords);
    // dbg!(1.0 / depth);
    // dbg!(stencil);
}

pub fn picking_hook(mut world: DeferredWorld, context: HookContext) {
    let mut buffers = world.resource_mut::<Assets<ShaderBuffer>>();
    let mut buffer = ShaderBuffer::with_size(
        GpuPickingData::min_size().get() as usize,
        RenderAssetUsages::default(),
    );
    buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
    let buffer = buffers.add(buffer);

    world
        .commands()
        .entity(context.entity)
        .insert(Readback::buffer(buffer.clone()))
        .observe(picking_readback);

    let mut picking_data = world.get_mut::<PickingData>(context.entity).unwrap();
    picking_data.buffer = buffer;
}

#[derive(Default, Clone, Component)]
#[component(on_add = picking_hook)]
pub struct PickingData {
    pub cursor_coords: Vec2,
    pub cell: CellCoord,           // cell of floating origin (camera)
    pub translation: Option<Vec3>, // relative to floating origin cell
    pub world_from_clip: Mat4,
    buffer: Handle<ShaderBuffer>,
}

impl SyncComponent for PickingData {
    type Target = GpuPickingBuffer;
}

impl ExtractComponent for PickingData {
    type QueryData = &'static PickingData;
    type QueryFilter = ();
    type Out = GpuPickingBuffer;

    fn extract_component(data: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(GpuPickingBuffer(data.buffer.id()))
    }
}

#[derive(Component)]
pub struct GpuPickingBuffer(AssetId<ShaderBuffer>);

#[derive(Default, Debug, Clone, ShaderType)]
pub struct GpuPickingData {
    pub cursor_coords: Vec2,
    pub depth: f32,
    pub stencil: u32,
    pub world_from_clip: Mat4,
    pub cell: IVec3,
}

#[derive(Resource)]
pub struct PickingPipeline {
    id: CachedComputePipelineId,
    layout: BindGroupLayout,
}

impl FromWorld for PickingPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<GpuPickingData>(false),
                texture_depth_2d_multisampled(),
                texture_2d_multisampled(TextureSampleType::Uint),
            ),
        );

        let layout = device.create_bind_group_layout(None, &entries);

        let id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![BindGroupLayoutDescriptor::new(
                "picking_bind_group_layout",
                &entries,
            )],
            immediate_size: Default::default(),
            shader: world.load_asset(PICKING_SHADER),
            shader_defs: vec![],
            entry_point: Some("pick".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self { id, layout }
    }
}

pub fn picking_pass(
    world: &World,
    view: ViewQuery<(&GpuPickingBuffer, &TerrainViewDepthTexture)>,
    mut ctx: RenderContext,
) {
    let pipeline_cache = world.resource::<PipelineCache>();
    let picking_pipeline = world.resource::<PickingPipeline>();
    let buffer = world.resource::<RenderAssets<GpuShaderBuffer>>();

    let Some(pipeline) = pipeline_cache.get_compute_pipeline(picking_pipeline.id) else {
        return;
    };

    let (picking_buffer, depth) = view.into_inner();

    let Some(buffer) = buffer.get(picking_buffer.0) else {
        return;
    };

    let bind_group = ctx.render_device().create_bind_group(
        None,
        &picking_pipeline.layout,
        &BindGroupEntries::sequential((
            buffer.buffer.as_entire_binding(),
            &depth.depth_view,
            &depth.stencil_view,
        )),
    );

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());
    pass.set_bind_group(0, &bind_group, &[]);
    pass.set_pipeline(pipeline);
    pass.dispatch_workgroups(1, 1, 1);
}

pub struct TerrainPickingPlugin;

impl Plugin for TerrainPickingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            picking_system.after(TransformSystems::Propagate),
        )
        .add_plugins(ExtractComponentPlugin::<PickingData>::default());

        app.sub_app_mut(RenderApp).add_systems(
            Core3d,
            picking_pass
                .after(terrain_pass)
                .before(main_opaque_pass_3d)
                .in_set(Core3dSystems::MainPass),
        );
    }
    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<PickingPipeline>();
    }
}
