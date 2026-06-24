use bevy::prelude::*;

use crate::creature::{Animal, Plant};

pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_visualization)
            .add_systems(Update, update_time_display)
            .add_systems(Update, attach_animal_visuals)
            .add_systems(Update, attach_plant_visuals);
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

fn attach_animal_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    query: Query<(Entity, &Animal), Added<Animal>>,
) {
    for (entity, animal) in &query {
        commands.entity(entity).insert((
            Mesh2d(meshes.add(Circle::new(animal.radius))),
            MeshMaterial2d(materials.add(animal.color)),
            Transform::from_translation(animal.position.extend(0.0)),
        ));
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
            Mesh2d(meshes.add(Circle::new(plant.radius))),
            MeshMaterial2d(materials.add(plant.color)),
            Transform::from_translation(plant.position.extend(0.0)),
        ));
    }
}