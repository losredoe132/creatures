use bevy::prelude::*;

use crate::brain::{compute_features, think_with_vision};
use crate::config::SimulationConfig;
use crate::creature::{Animal, Carcass, Diet, Plant};
use crate::mlp::{MLP_INPUTS, MLP_OUTPUTS};
use crate::sense::{AnimalSnapshot, CarcassSnapshot, PerceptionWorld, PlantSnapshot};
use crate::simulation::{GlobalFrameCounter, ManualZooSpawnEvent, PopulationSizeTracker};
pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HoveredAnimal>()
            .init_resource::<SelectedAnimalId>()
            .init_resource::<CurrentMlpState>()
            .init_resource::<MlpLayout>()
            .init_resource::<MlpHovered>()
            .add_systems(
                Startup,
                (setup_visualization, setup_hover_panel, setup_mlp_tooltip),
            )
            .add_systems(PostUpdate, update_time_display)
            .add_systems(Update, draw_world_boundary)
            .add_systems(Update, draw_animal_perceptive_field)
            .add_systems(Update, draw_animal_movement_arrows)
            .add_systems(Update, attach_animal_visuals)
            .add_systems(Update, attach_plant_visuals)
            //.add_systems(Update, attach_carcass_visuals)
            .add_systems(Update, update_animal_visual_sizes)
            .add_systems(Update, update_plant_visual_sizes)
            .add_systems(Update, handle_animal_click)
            .add_systems(Update, refresh_selected_animal.after(handle_animal_click))
            .add_systems(Update, refresh_mlp_state.after(refresh_selected_animal))
            .add_systems(Update, update_hover_panel.after(refresh_selected_animal))
            .add_systems(
                Update,
                draw_selection_indicator.after(refresh_selected_animal),
            )
            .add_systems(Update, compute_mlp_layout.after(refresh_mlp_state))
            .add_systems(Update, detect_mlp_hover.after(compute_mlp_layout))
            .add_systems(Update, update_mlp_tooltip.after(detect_mlp_hover))
            .add_systems(
                Update,
                draw_mlp_visualization
                    .after(compute_mlp_layout)
                    .after(detect_mlp_hover),
            )
            .add_systems(Update, handle_pause_button)
            .add_systems(Update, handle_pause_keyboard)
            .add_systems(Update, handle_zoo_spawn_button)
            .add_systems(
                Update,
                update_pause_button_text
                    .after(handle_pause_button)
                    .after(handle_pause_keyboard),
            );
    }
}

struct HoveredAnimalData {
    id: u64,
    parent_id: Option<u64>,
    diet: Diet,
    position: Vec2,
    velocity: Vec2,
    energy: f32,
    initial_energy: f32,
    size: f32,
    vision_range: f32,
    family: u32,
    generation: u32,
    age: u64,
    genes: Vec<f32>,
}

#[derive(Resource, Default)]
struct HoveredAnimal(Option<HoveredAnimalData>);

#[derive(Resource, Default)]
struct SelectedAnimalId(Option<u64>);

const INPUT_LABELS: [&str; MLP_INPUTS] = ["plant dx", "plant dy"];
const OUTPUT_LABELS: [&str; MLP_OUTPUTS] = ["move x", "move y"];

#[derive(Resource)]
struct MlpLayout {
    x_in: f32,
    x_out: f32,
    y_top: f32,
    y_bot: f32,
}

impl Default for MlpLayout {
    fn default() -> Self {
        Self {
            x_in: 0.0,
            x_out: 0.0,
            y_top: 20.0,
            y_bot: 580.0,
        }
    }
}

#[derive(Resource, Default)]
struct CurrentMlpState {
    features: [f32; MLP_INPUTS],
    outputs: [f32; MLP_OUTPUTS],
    valid: bool,
}

#[derive(Default, PartialEq, Clone, Copy)]
enum MlpHoverTarget {
    #[default]
    None,
    InputNode(usize),
    OutputNode(usize),
    Edge { input: usize, output: usize },
}

#[derive(Resource, Default)]
struct MlpHovered {
    target: MlpHoverTarget,
    value: f32,
    screen_pos: Vec2,
}

#[derive(Component)]
struct MlpTooltip;

#[derive(Component)]
struct MlpTooltipText;

#[derive(Component)]
struct HoverPanel;

#[derive(Component)]
struct HoverPanelText;

#[derive(Component)]
struct PauseButton;

#[derive(Component)]
struct PauseButtonText;

#[derive(Component)]
struct ZooSpawnButton;

#[derive(Component)]
struct TimeDisplay;

fn setup_hover_panel(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                right: Val::Px(8.0),
                padding: UiRect::all(Val::Px(12.0)),
                min_width: Val::Px(180.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.04, 0.04, 0.12, 0.88)),
            Visibility::Hidden,
            HoverPanel,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                HoverPanelText,
            ));
        });
}

fn setup_mlp_tooltip(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.05, 0.1, 0.9)),
            Visibility::Hidden,
            MlpTooltip,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                MlpTooltipText,
            ));
        });
}

fn handle_animal_click(
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    animals: Query<&Animal>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut selected: ResMut<SelectedAnimalId>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok(world_pos) = camera.viewport_to_world_2d(cam_transform, cursor_pos) else {
        return;
    };

    let mut closest: Option<(u64, f32)> = None;
    for animal in &animals {
        let dist = animal.position.distance(world_pos);
        if dist <= animal.size {
            if closest.is_none_or(|(_, d)| dist < d) {
                closest = Some((animal.id, dist));
            }
        }
    }
    selected.0 = closest.map(|(id, _)| id);
}

fn refresh_selected_animal(
    animals: Query<&Animal>,
    mut selected: ResMut<SelectedAnimalId>,
    frame_count: Res<GlobalFrameCounter>,
    mut hovered: ResMut<HoveredAnimal>,
) {
    let Some(id) = selected.0 else {
        hovered.0 = None;
        return;
    };

    match animals.iter().find(|a| a.id == id) {
        None => {
            selected.0 = None;
            hovered.0 = None;
        }
        Some(animal) => {
            hovered.0 = Some(HoveredAnimalData {
                id: animal.id,
                parent_id: animal.parent_id,
                diet: animal.diet,
                position: animal.position,
                velocity: animal.velocity,
                energy: animal.energy,
                initial_energy: animal.initial_energy,
                size: animal.size,
                vision_range: animal.vision.range,
                family: animal.family,
                generation: animal.generation,
                age: frame_count.0.saturating_sub(animal.spawn_at),
                genes: animal.genome.genes.clone(),
            });
        }
    }
}

fn draw_selection_indicator(hovered: Res<HoveredAnimal>, mut gizmos: Gizmos) {
    let Some(data) = &hovered.0 else { return };
    gizmos.circle_2d(
        data.position,
        data.size * 2.0,
        Color::srgba(1.0, 1.0, 1.0, 0.75),
    );
}

fn update_hover_panel(
    hovered: Res<HoveredAnimal>,
    mut panel_q: Query<&mut Visibility, With<HoverPanel>>,
    mut text_q: Query<&mut Text, With<HoverPanelText>>,
) {
    let Ok(mut visibility) = panel_q.single_mut() else {
        return;
    };
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };

    match &hovered.0 {
        None => {
            *visibility = Visibility::Hidden;
        }
        Some(d) => {
            *visibility = Visibility::Visible;
            **text = format!(
                "ID: {}\nParent: {}\nDiet: {:?}\nEnergy: {:.1}\nSpeed: {:.2}\nSize: {:.2}\nVision: {:.1}\nFamily: {}\nGeneration: {}\nAge: {}",
                d.id,
                d.parent_id.map_or("none".to_string(), |p| p.to_string()),
                d.diet,
                d.energy,
                d.velocity.length(),
                d.size,
                d.vision_range,
                d.family,
                d.generation,
                d.age,
            );
        }
    }
}

fn setup_visualization(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn((
        Text::new("Frame: 0\nPlants: 0\nAnimals: 0\nFamilies: 0"),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(8.0),
            left: Val::Px(8.0),
            ..default()
        },
        TimeDisplay,
    ));

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(8.0),
                right: Val::Px(8.0),
                padding: UiRect::axes(Val::Px(16.0), Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.15, 0.15, 0.25, 0.9)),
            PauseButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Pause"),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                PauseButtonText,
            ));
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(8.0),
                right: Val::Px(100.0),
                padding: UiRect::axes(Val::Px(16.0), Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.1, 0.2, 0.1, 0.9)),
            ZooSpawnButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Spawn Random"),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
            ));
        });
}

fn handle_pause_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<PauseButton>),
    >,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    for (interaction, mut bg) in &mut interaction_q {
        match interaction {
            Interaction::Pressed => {
                if virtual_time.is_paused() {
                    virtual_time.unpause();
                } else {
                    virtual_time.pause();
                }
                *bg = BackgroundColor(Color::srgba(0.35, 0.35, 0.55, 0.9));
            }
            Interaction::Hovered => {
                *bg = BackgroundColor(Color::srgba(0.25, 0.25, 0.4, 0.9));
            }
            Interaction::None => {
                *bg = BackgroundColor(Color::srgba(0.15, 0.15, 0.25, 0.9));
            }
        }
    }
}

fn handle_pause_keyboard(
    input: Res<ButtonInput<KeyCode>>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if input.just_pressed(KeyCode::Space) {
        if virtual_time.is_paused() {
            virtual_time.unpause();
        } else {
            virtual_time.pause();
        }
    }
}

fn handle_zoo_spawn_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<ZooSpawnButton>),
    >,
    mut events: MessageWriter<ManualZooSpawnEvent>,
) {
    for (interaction, mut bg) in &mut interaction_q {
        match interaction {
            Interaction::Pressed => {
                events.write(ManualZooSpawnEvent);
                *bg = BackgroundColor(Color::srgba(0.3, 0.5, 0.3, 0.9));
            }
            Interaction::Hovered => {
                *bg = BackgroundColor(Color::srgba(0.2, 0.35, 0.2, 0.9));
            }
            Interaction::None => {
                *bg = BackgroundColor(Color::srgba(0.1, 0.2, 0.1, 0.9));
            }
        }
    }
}

fn update_pause_button_text(
    virtual_time: Res<Time<Virtual>>,
    mut text_q: Query<&mut Text, With<PauseButtonText>>,
) {
    if !virtual_time.is_changed() {
        return;
    }
    for mut text in &mut text_q {
        **text = if virtual_time.is_paused() {
            "Resume".to_string()
        } else {
            "Pause".to_string()
        };
    }
}

fn update_time_display(
    mut query: Query<&mut Text, With<TimeDisplay>>,
    frame_count: Res<GlobalFrameCounter>,
    tracker: Res<PopulationSizeTracker>,
) {
    let mut families: Vec<(u32, usize)> =
        tracker.families.iter().map(|(&id, &n)| (id, n)).collect();
    families.sort_by(|a, b| b.1.cmp(&a.1));
    let families_str = families
        .iter()
        .map(|(id, n)| format!("  #{}: {}", id, n))
        .collect::<Vec<_>>()
        .join("\n");

    let total_animals = tracker.animals.herbivores
        + tracker.animals.omnivores
        + tracker.animals.carnivores
        + tracker.animals.scavengers;

    for mut text in &mut query {
        **text = format!(
            "Frame: {}\nPlants: {}\nAnimals: {}\n  Herbivore: {}\n  Omnivore: {}\n  Carnivore: {}\n  Scavenger: {}\nFamilies: {}\n{}",
            frame_count.0,
            tracker.plants,
            total_animals,
            tracker.animals.herbivores,
            tracker.animals.omnivores,
            tracker.animals.carnivores,
            tracker.animals.scavengers,
            families.len(),
            families_str,
        );
    }
}

fn draw_world_boundary(mut gizmos: Gizmos, config: Res<SimulationConfig>) {
    let bounds = &config.world_bounds;
    let half_w = bounds.half_width;
    let half_h = bounds.half_height;
    let boundary_color = Color::srgba(0.5, 0.5, 0.5, 0.6);

    let top_left = Vec2::new(-half_w, half_h);
    let top_right = Vec2::new(half_w, half_h);
    let bottom_right = Vec2::new(half_w, -half_h);
    let bottom_left = Vec2::new(-half_w, -half_h);

    gizmos.line_2d(top_left, top_right, boundary_color);
    gizmos.line_2d(top_right, bottom_right, boundary_color);
    gizmos.line_2d(bottom_right, bottom_left, boundary_color);
    gizmos.line_2d(bottom_left, top_left, boundary_color);
}

fn draw_animal_perceptive_field(hovered: Res<HoveredAnimal>, mut gizmos: Gizmos) {
    let Some(data) = &hovered.0 else { return };
    gizmos.circle_2d(data.position, data.vision_range, Color::srgba(0.2, 0.9, 1.0, 0.18));
}

fn draw_animal_movement_arrows(
    mut gizmos: Gizmos,
    animals: Query<&Animal>,
    plants: Query<&Plant>,
    carcasses: Query<&Carcass>,
) {
    let plants_snapshot: Vec<PlantSnapshot> = plants
        .iter()
        .map(|plant| PlantSnapshot {
            position: plant.position,
            energy: plant.energy,
        })
        .collect();

    let animals_snapshot: Vec<AnimalSnapshot> = animals
        .iter()
        .map(|animal| AnimalSnapshot {
            diet: animal.diet,
            position: animal.position,
            velocity: animal.velocity,
            energy: animal.energy,
        })
        .collect();

    let carcasses_snapshot: Vec<CarcassSnapshot> = carcasses
        .iter()
        .map(|carcass| CarcassSnapshot {
            position: carcass.position,
            energy: carcass.energy,
        })
        .collect();

    let world = PerceptionWorld {
        plants: &plants_snapshot,
        animals: &animals_snapshot,
        carcasses: &carcasses_snapshot,
    };

    let arrow_color = Color::srgba(1.0, 0.95, 0.2, 0.9);
    for animal in &animals {
        let movement = think_with_vision(
            &animal.vision,
            &animal.genome,
            animal.position,
            animal.velocity,
            animal.energy,
            &world,
        );
        let magnitude = movement.length();
        if magnitude <= f32::EPSILON {
            continue;
        }

        let direction = movement / magnitude;
        let arrow_len = (magnitude * 16.0).clamp(6.0, 20.0);
        let start = animal.position;
        let end = start + direction * arrow_len;
        draw_arrow_2d(&mut gizmos, start, end, arrow_color);
    }
}

fn draw_arrow_2d(gizmos: &mut Gizmos, start: Vec2, end: Vec2, color: Color) {
    gizmos.line_2d(start, end, color);

    let shaft = end - start;
    let len = shaft.length();
    if len <= f32::EPSILON {
        return;
    }

    let dir = shaft / len;
    let head_len = (len * 0.35).clamp(2.0, 6.0);
    let left = Vec2::from_angle(dir.to_angle() + 2.6) * head_len;
    let right = Vec2::from_angle(dir.to_angle() - 2.6) * head_len;
    gizmos.line_2d(end, end + left, color);
    gizmos.line_2d(end, end + right, color);
}

fn attach_animal_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    query: Query<(Entity, &Animal), Added<Animal>>,
) {
    for (entity, animal) in &query {
        commands.entity(entity).insert((
            Mesh2d(meshes.add(Circle::new(1.0))),
            MeshMaterial2d(materials.add(animal.color)),
            Transform::from_translation(animal.position.extend(0.0))
                .with_scale(Vec3::splat(animal.size)),
        ));
    }
}

fn update_animal_visual_sizes(mut query: Query<(&Animal, &mut Transform), Changed<Animal>>) {
    for (animal, mut transform) in &mut query {
        transform.scale = Vec3::splat(animal.size);
    }
}

fn attach_plant_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    query: Query<(Entity, &Plant), Added<Plant>>,
) {
    for (entity, plant) in &query {
        commands.entity(entity).insert((
            Mesh2d(meshes.add(Circle::new(1.0))),
            MeshMaterial2d(materials.add(Color::srgba(0.3, 0.5, 0.3, 0.9))),
            Transform::from_translation(plant.position.extend(0.0))
                .with_scale(Vec3::splat(plant.size)),
        ));
    }
}

fn update_plant_visual_sizes(mut query: Query<(&Plant, &mut Transform), Changed<Plant>>) {
    for (plant, mut transform) in &mut query {
        transform.scale = Vec3::splat(plant.size);
    }
}

fn refresh_mlp_state(
    hovered: Res<HoveredAnimal>,
    animals: Query<&Animal>,
    plants: Query<&Plant>,
    carcasses: Query<&Carcass>,
    mut state: ResMut<CurrentMlpState>,
) {
    let Some(data) = &hovered.0 else {
        state.valid = false;
        return;
    };

    let plants_snap: Vec<PlantSnapshot> = plants
        .iter()
        .map(|p| PlantSnapshot {
            position: p.position,
            energy: p.energy,
        })
        .collect();
    let animals_snap: Vec<AnimalSnapshot> = animals
        .iter()
        .map(|a| AnimalSnapshot {
            diet: a.diet,
            position: a.position,
            velocity: a.velocity,
            energy: a.energy,
        })
        .collect();
    let carcasses_snap: Vec<CarcassSnapshot> = carcasses
        .iter()
        .map(|c| CarcassSnapshot {
            position: c.position,
            energy: c.energy,
        })
        .collect();
    let world = PerceptionWorld {
        plants: &plants_snap,
        animals: &animals_snap,
        carcasses: &carcasses_snap,
    };

    state.features = compute_features(
        data.vision_range,
        data.position,
        data.velocity,
        data.energy,
        &world,
    );

    let genes = &data.genes;
    for o in 0..MLP_OUTPUTS {
        state.outputs[o] = genes
            .get(MLP_INPUTS * MLP_OUTPUTS + o)
            .copied()
            .unwrap_or(0.0);
        for i in 0..MLP_INPUTS {
            state.outputs[o] += state.features[i] * genes[i * MLP_OUTPUTS + o];
        }
    }
    state.valid = true;
}

fn compute_mlp_layout(
    windows: Query<&Window>,
    panel_q: Query<(&ComputedNode, &Visibility), With<HoverPanel>>,
    mut layout: ResMut<MlpLayout>,
) {
    let Ok(window) = windows.single() else { return };
    let w = window.width();
    let h = window.height();
    let scale = window.scale_factor() as f32;

    layout.x_in = w - 190.0;
    layout.x_out = w - 50.0;
    layout.y_bot = h - 20.0;
    layout.y_top = match panel_q.single() {
        Ok((computed, vis)) if *vis != Visibility::Hidden => 8.0 + computed.size.y / scale + 12.0,
        _ => 20.0,
    };
}

fn seg_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = if ab.dot(ab) > 0.0 { ((p - a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0) } else { 0.0 };
    (p - (a + ab * t)).length()
}

fn detect_mlp_hover(
    windows: Query<&Window>,
    layout: Res<MlpLayout>,
    mlp_state: Res<CurrentMlpState>,
    animal: Res<HoveredAnimal>,
    mut hovered: ResMut<MlpHovered>,
) {
    hovered.target = MlpHoverTarget::None;
    if !mlp_state.valid { return; }
    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else { return };

    let x_in  = layout.x_in;
    let x_out = layout.x_out;
    let y_top = layout.y_top;
    let y_bot = layout.y_bot;
    let node_hit = 8.0f32;
    let edge_hit = 5.0f32;

    let input_sy  = |i: usize| y_top + (i as f32 + 0.5) * (y_bot - y_top) / MLP_INPUTS  as f32;
    let output_sy = |o: usize| y_top + (o as f32 + 0.5) * (y_bot - y_top) / MLP_OUTPUTS as f32;

    // Nodes take priority over edges
    for i in 0..MLP_INPUTS {
        let sy = input_sy(i);
        if cursor.distance(Vec2::new(x_in, sy)) <= node_hit {
            hovered.target    = MlpHoverTarget::InputNode(i);
            hovered.value     = mlp_state.features[i];
            hovered.screen_pos = Vec2::new(x_in, sy);
            return;
        }
    }
    for o in 0..MLP_OUTPUTS {
        let sy = output_sy(o);
        if cursor.distance(Vec2::new(x_out, sy)) <= node_hit {
            hovered.target    = MlpHoverTarget::OutputNode(o);
            hovered.value     = mlp_state.outputs[o];
            hovered.screen_pos = Vec2::new(x_out, sy);
            return;
        }
    }

    // Edge hit-test — weight is read directly from the selected animal's genome
    if let Some(data) = &animal.0 {
        for i in 0..MLP_INPUTS {
            let a = Vec2::new(x_in, input_sy(i));
            for o in 0..MLP_OUTPUTS {
                let b = Vec2::new(x_out, output_sy(o));
                if seg_dist(cursor, a, b) <= edge_hit {
                    hovered.target     = MlpHoverTarget::Edge { input: i, output: o };
                    hovered.value      = data.genes.get(i * MLP_OUTPUTS + o).copied().unwrap_or(0.0);
                    hovered.screen_pos = (a + b) * 0.5;
                    return;
                }
            }
        }
    }
}

fn update_mlp_tooltip(
    hovered: Res<MlpHovered>,
    mut tooltip_q: Query<(&mut Node, &mut Visibility), With<MlpTooltip>>,
    mut text_q: Query<&mut Text, With<MlpTooltipText>>,
) {
    let Ok((mut node, mut vis)) = tooltip_q.single_mut() else { return };
    let Ok(mut text) = text_q.single_mut() else { return };

    let label = match hovered.target {
        MlpHoverTarget::None => { *vis = Visibility::Hidden; return; }
        MlpHoverTarget::InputNode(i)  => format!("{}: {:.3}", INPUT_LABELS[i],  hovered.value),
        MlpHoverTarget::OutputNode(o) => format!("{}: {:.3}", OUTPUT_LABELS[o], hovered.value),
        MlpHoverTarget::Edge { input: i, output: o } =>
            format!("{} → {}: {:.3}", INPUT_LABELS[i], OUTPUT_LABELS[o], hovered.value),
    };

    *vis = Visibility::Visible;
    **text = label;
    node.left = Val::Px(hovered.screen_pos.x - 150.0);
    node.top  = Val::Px(hovered.screen_pos.y - 12.0);
}

fn activation_color(v: f32) -> Color {
    // dark gray at zero, orange at +1, blue at -1
    if v >= 0.0 {
        Color::srgba(0.15 + v * 0.8, 0.15 + v * 0.55, 0.15 - v * 0.05, 0.95)
    } else {
        let t = v + 1.0;
        Color::srgba(0.15 + t * 0.05, 0.15 + t * 0.25, 0.95 - t * 0.8, 0.95)
    }
}

fn draw_mlp_visualization(
    hovered: Res<HoveredAnimal>,
    mlp_state: Res<CurrentMlpState>,
    node_hover: Res<MlpHovered>,
    layout: Res<MlpLayout>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut gizmos: Gizmos,
) {
    if hovered.0.is_none() || !mlp_state.valid {
        return;
    }
    let data = hovered.0.as_ref().unwrap();
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };

    let Ok(p0) = camera.viewport_to_world_2d(cam_transform, Vec2::ZERO) else {
        return;
    };
    let Ok(p1) = camera.viewport_to_world_2d(cam_transform, Vec2::new(1.0, 0.0)) else {
        return;
    };
    let px = (p1 - p0).length();

    let x_in = layout.x_in;
    let x_out = layout.x_out;
    let y_top = layout.y_top;
    let y_bot = layout.y_bot;
    let node_r_in = px * 4.5;
    let node_r_out = px * 6.5;

    let to_world = |sx: f32, sy: f32| -> Option<Vec2> {
        camera
            .viewport_to_world_2d(cam_transform, Vec2::new(sx, sy))
            .ok()
    };

    let genes = &data.genes;
    let max_w = genes[..MLP_INPUTS * MLP_OUTPUTS]
        .iter()
        .map(|w| w.abs())
        .fold(0.0f32, f32::max)
        .max(1.0);

    // Connections
    for i in 0..MLP_INPUTS {
        let iy = y_top + (i as f32 + 0.5) * (y_bot - y_top) / MLP_INPUTS as f32;
        let Some(iw) = to_world(x_in, iy) else { continue };
        for o in 0..MLP_OUTPUTS {
            let oy = y_top + (o as f32 + 0.5) * (y_bot - y_top) / MLP_OUTPUTS as f32;
            let Some(ow) = to_world(x_out, oy) else { continue };
            let weight = genes[i * MLP_OUTPUTS + o];
            let edge_hovered = node_hover.target == (MlpHoverTarget::Edge { input: i, output: o });
            let alpha = if edge_hovered { 1.0 } else { (weight.abs() / max_w * 0.65).max(0.04) };
            let color = if weight >= 0.0 {
                Color::srgba(0.25, 0.5, 1.0, alpha)
            } else {
                Color::srgba(1.0, 0.3, 0.2, alpha)
            };
            gizmos.line_2d(iw, ow, color);
            if edge_hovered {
                // second pass at slight world-space offset for a thicker appearance
                let perp = (ow - iw).perp().normalize_or_zero() * px * 1.5;
                gizmos.line_2d(iw + perp, ow + perp, color);
                gizmos.line_2d(iw - perp, ow - perp, color);
            }
        }
    }

    // Input nodes
    for i in 0..MLP_INPUTS {
        let iy = y_top + (i as f32 + 0.5) * (y_bot - y_top) / MLP_INPUTS as f32;
        let Some(iw) = to_world(x_in, iy) else {
            continue;
        };
        let color = activation_color(mlp_state.features[i].clamp(-1.0, 1.0));
        let hovered_this = node_hover.target == MlpHoverTarget::InputNode(i);
        let r = if hovered_this {
            node_r_in * 1.6
        } else {
            node_r_in
        };
        gizmos.circle_2d(iw, r, color);
    }

    // Output nodes
    for o in 0..MLP_OUTPUTS {
        let oy = y_top + (o as f32 + 0.5) * (y_bot - y_top) / MLP_OUTPUTS as f32;
        let Some(ow) = to_world(x_out, oy) else {
            continue;
        };
        let t = (mlp_state.outputs[o].tanh() + 1.0) * 0.5;
        let color = Color::srgba(0.9, 0.6 + t * 0.4, 0.1, 0.95);
        let hovered_this = node_hover.target == MlpHoverTarget::OutputNode(o);
        let r = if hovered_this {
            node_r_out * 1.6
        } else {
            node_r_out
        };
        gizmos.circle_2d(ow, r, color);
    }
}

fn attach_carcass_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    query: Query<(Entity, &Carcass), Added<Carcass>>,
) {
    for (entity, carcass) in &query {
        commands.entity(entity).insert((
            Mesh2d(meshes.add(Circle::new(1.0))),
            MeshMaterial2d(materials.add(Color::srgba(0.45, 0.3, 0.15, 0.8))),
            Transform::from_translation(carcass.position.extend(-0.1))
                .with_scale(Vec3::splat(carcass.size)),
        ));
    }
}
