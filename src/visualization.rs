use bevy::prelude::*;

use crate::brain::think_with_vision;
use crate::creature::{Animal, Plant};
use crate::sense::{AnimalSnapshot, PerceptionWorld, PlantSnapshot};

pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_visualization)
            .add_systems(Update, update_time_display)
            .add_systems(Update, draw_animal_perceptive_field)
            .add_systems(Update, draw_animal_movement_arrows)
            .add_systems(Update, attach_animal_visuals)
            .add_systems(Update, attach_plant_visuals)
            .add_systems(Update, update_animal_visual_sizes)
            .add_systems(Update, update_plant_visual_sizes);
    }
}

#[derive(Component)]
struct TimeDisplay;

fn setup_visualization(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn((
        Text::new("Time: 0.0s"),
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
}

fn update_time_display(mut query: Query<&mut Text, With<TimeDisplay>>, time: Res<Time>) {
    for mut text in &mut query {
        let elapsed = time.elapsed_secs();
        **text = format!("Time: {:.1}s", elapsed);
    }
}

fn draw_animal_perceptive_field(mut gizmos: Gizmos, query: Query<&Animal>) {
    let cone_color = Color::srgba(0.2, 0.9, 1.0, 0.22);
    let edge_color = Color::srgba(0.2, 0.9, 1.0, 0.45);

    for animal in &query {
        let origin = animal.position;
        let forward = animal.velocity.normalize_or_zero();
        let forward = if forward == Vec2::ZERO {
            Vec2::X
        } else {
            forward
        };
        let range = animal.vision.range;

        gizmos.circle_2d(origin, range, cone_color);
    }
}

fn draw_animal_movement_arrows(mut gizmos: Gizmos, animals: Query<&Animal>, plants: Query<&Plant>) {
    let plants_snapshot: Vec<PlantSnapshot> = plants
        .iter()
        .map(|plant| PlantSnapshot {
            position: plant.position,
            energy: plant.energy,
            radius: plant.size,
        })
        .collect();

    let animals_snapshot: Vec<AnimalSnapshot> = animals
        .iter()
        .map(|animal| AnimalSnapshot {
            position: animal.position,
            energy: animal.energy,
            radius: animal.size,
        })
        .collect();

    let world = PerceptionWorld {
        plants: &plants_snapshot,
        animals: &animals_snapshot,
    };

    let arrow_color = Color::srgba(1.0, 0.95, 0.2, 0.9);
    for animal in &animals {
        let movement = think_with_vision(
            &animal.vision,
            &animal.genome,
            animal.position,
            animal.velocity,
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
            MeshMaterial2d(materials.add(plant.color)),
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
