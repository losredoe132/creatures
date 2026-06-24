use bevy::prelude::*;

use crate::config::WorldBounds;
use crate::creature::{Animal, EnergyPosition, Movable, Plant};

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_world)
            .add_systems(Update, move_animals);
    }
}

fn setup_world(mut commands: Commands) {
    let animal = Animal {
        position: Vec2::new(0.0, 0.0),
        velocity: Vec2::new(100.0, 1.0),
        energy: 100.0,
        radius: 20.0,
        color: Color::srgb(0.8, 0.2, 0.4),
    };

    let plant = Plant {
        position: Vec2::new(-200.0, 100.0),
        energy: 60.0,
        radius: 14.0,
        color: Color::srgb(0.3, 0.6, 0.2),
    };

    commands.spawn(animal);
    commands.spawn(plant);
}

fn move_animals(
    mut query: Query<(&mut Animal, &mut Transform)>,
    time: Res<Time>,
    world_bounds: Res<WorldBounds>,
) {
    for (mut animal, mut transform) in &mut query {
        transform.translation += animal.velocity().extend(0.0) * time.delta_secs();
        ensure_torodial_world(&mut transform.translation, &world_bounds);
        animal.set_position(transform.translation.xy());
    }
}

fn ensure_torodial_world(translation: &mut Vec3, world_bounds: &WorldBounds) {
    if translation.x < -world_bounds.half_width {
        translation.x = world_bounds.half_width;
    } else if translation.x > world_bounds.half_width {
        translation.x = -world_bounds.half_width;
    }

    if translation.y < -world_bounds.half_height {
        translation.y = world_bounds.half_height;
    } else if translation.y > world_bounds.half_height {
        translation.y = -world_bounds.half_height;
    }
}
