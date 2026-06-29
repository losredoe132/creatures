use bevy::prelude::*;

use crate::brain::think_with_vision;
use crate::config::SimulationConfig;
use crate::creature::{Animal, Carcass, Diet, Plant};
use crate::sense::{AnimalSnapshot, CarcassSnapshot, PerceptionWorld, PlantSnapshot};
use crate::simulation::{GlobalFrameCounter, PopulationSizeTracker};
pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HoveredAnimal>()
            .add_systems(Startup, (setup_visualization, setup_hover_panel))
            .add_systems(PostUpdate, update_time_display)
            .add_systems(Update, draw_world_boundary)
            .add_systems(Update, draw_animal_perceptive_field)
            .add_systems(Update, draw_animal_movement_arrows)
            .add_systems(Update, attach_animal_visuals)
            .add_systems(Update, attach_plant_visuals)
            .add_systems(Update, attach_carcass_visuals)
            .add_systems(Update, update_animal_visual_sizes)
            .add_systems(Update, update_plant_visual_sizes)
            .add_systems(Update, detect_animal_hover)
            .add_systems(Update, update_hover_panel.after(detect_animal_hover))
            .add_systems(Update, handle_pause_button)
            .add_systems(Update, handle_pause_keyboard)
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
    energy: f32,
    initial_energy: f32,
    speed: f32,
    size: f32,
    vision_range: f32,
    family: u32,
    generation: u32,
    age: u64,
    genes: Vec<f32>,
}

#[derive(Resource, Default)]
struct HoveredAnimal(Option<HoveredAnimalData>);

#[derive(Component)]
struct HoverPanel;

#[derive(Component)]
struct HoverPanelText;

#[derive(Component)]
struct PauseButton;

#[derive(Component)]
struct PauseButtonText;

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

fn detect_animal_hover(
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    animals: Query<&Animal>,
    mut hovered: ResMut<HoveredAnimal>,
    frame_count: Res<GlobalFrameCounter>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };

    let Some(cursor_pos) = window.cursor_position() else {
        hovered.0 = None;
        return;
    };

    let Ok(world_pos) = camera.viewport_to_world_2d(cam_transform, cursor_pos) else {
        hovered.0 = None;
        return;
    };

    let mut closest: Option<(&Animal, f32)> = None;
    for animal in &animals {
        let dist = animal.position.distance(world_pos);
        if dist <= animal.size {
            if closest.is_none_or(|(_, d)| dist < d) {
                closest = Some((animal, dist));
            }
        }
    }

    hovered.0 = closest.map(|(animal, _)| HoveredAnimalData {
        id: animal.id,
        parent_id: animal.parent_id,
        diet: animal.diet,
        energy: animal.energy,
        initial_energy: animal.initial_energy,
        speed: animal.velocity.length(),
        size: animal.size,
        vision_range: animal.vision.range,
        family: animal.family,
        generation: animal.generation,
        age: frame_count.0.saturating_sub(animal.spawn_at),
        genes: animal.genome.genes.clone(),
    });
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
            let max_abs = d
                .genes
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1.0);
            let genome_str = d
                .genes
                .chunks(10)
                .map(|row| {
                    row.iter()
                        .map(|v| {
                            let t = (v.abs() / max_abs).clamp(0.0, 1.0);
                            let block = if t < 0.2 {
                                '░'
                            } else if t < 0.5 {
                                '▒'
                            } else if t < 0.8 {
                                '▓'
                            } else {
                                '█'
                            };
                            format!("{}{:+.2}", block, v)
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect::<Vec<_>>()
                .join("\n");
            **text = format!(
                "ID: {}\nParent: {}\nDiet: {:?}\nEnergy: {:.1}\nSpeed: {:.2}\nSize: {:.2}\nVision: {:.1}\nFamily: {}\nGeneration: {}\nAge: {}\nGenome ({}):\n{}",
                d.id,
                d.parent_id.map_or("none".to_string(), |p| p.to_string()),
                d.diet,
                d.energy,
                d.speed,
                d.size,
                d.vision_range,
                d.family,
                d.generation,
                d.age,
                d.genes.len(),
                genome_str,
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

fn draw_animal_perceptive_field(mut gizmos: Gizmos, query: Query<&Animal>) {
    let cone_color = Color::srgba(0.2, 0.9, 1.0, 0.12);
    for animal in &query {
        let origin = animal.position;

        let range = animal.vision.range;

        gizmos.circle_2d(origin, range, cone_color);
    }
}

fn draw_animal_movement_arrows(mut gizmos: Gizmos, animals: Query<&Animal>, plants: Query<&Plant>, carcasses: Query<&Carcass>) {
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
