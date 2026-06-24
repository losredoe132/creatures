use bevy::prelude::*;

use crate::config::SimulationConfig;
use crate::creature::{Animal, Plant};
use crate::simulation::SharedGridCells;

pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_visualization)
            .add_systems(Update, update_time_display)
            .add_systems(Update, draw_grid_overlay)
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

fn draw_grid_overlay(
    mut gizmos: Gizmos,
    config: Res<SimulationConfig>,
    shared_cells: Res<SharedGridCells>,
) {
    let world_bounds = &config.world_bounds;
    let grid = &config.grid_config;
    let dims = grid.dimensions(world_bounds);
    let cell_size = grid.cell_size(world_bounds);

    let left = -world_bounds.half_width;
    let right = world_bounds.half_width;
    let bottom = -world_bounds.half_height;
    let top = world_bounds.half_height;
    let line_color = Color::srgba(0.7, 0.7, 0.75, 0.35);

    for col in 0..=dims.x {
        let x = (left + col as f32 * cell_size.x).min(right);
        gizmos.line_2d(Vec2::new(x, bottom), Vec2::new(x, top), line_color);
    }

    for row in 0..=dims.y {
        let y = (bottom + row as f32 * cell_size.y).min(top);
        gizmos.line_2d(Vec2::new(left, y), Vec2::new(right, y), line_color);
    }

    let occupied_color = Color::srgba(1.0, 0.4, 0.2, 0.85);
    for occupancy in &shared_cells.cells {
        let center = grid.cell_center(occupancy.cell, world_bounds);
        draw_cell_outline(
            &mut gizmos,
            center,
            cell_size,
            occupied_color,
        );
    }
}

fn draw_cell_outline(gizmos: &mut Gizmos, center: Vec2, size: Vec2, color: Color) {
    let half = size * 0.5;
    let bl = Vec2::new(center.x - half.x, center.y - half.y);
    let br = Vec2::new(center.x + half.x, center.y - half.y);
    let tr = Vec2::new(center.x + half.x, center.y + half.y);
    let tl = Vec2::new(center.x - half.x, center.y + half.y);

    gizmos.line_2d(bl, br, color);
    gizmos.line_2d(br, tr, color);
    gizmos.line_2d(tr, tl, color);
    gizmos.line_2d(tl, bl, color);
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