use bevy::prelude::*;

use crate::creature::{Animal, EnergyPosition, Movable, Plant};

pub struct SimulationPlugin;

const WORLD_HALF_WIDTH: f32 = 400.0;
const WORLD_HALF_HEIGHT: f32 = 250.0;

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

fn move_animals(mut query: Query<(&mut Animal, &mut Transform)>, time: Res<Time>) {
    for (mut animal, mut transform) in &mut query {
        transform.translation += animal.velocity().extend(0.0) * time.delta_secs();
        ensure_torodial_world(&mut transform.translation);
        animal.set_position(transform.translation.xy());
    }
}

fn ensure_torodial_world(translation: &mut Vec3) {
    if translation.x < -WORLD_HALF_WIDTH {
        translation.x = WORLD_HALF_WIDTH;
    } else if translation.x > WORLD_HALF_WIDTH {
        translation.x = -WORLD_HALF_WIDTH;
    }

    if translation.y < -WORLD_HALF_HEIGHT {
        translation.y = WORLD_HALF_HEIGHT;
    } else if translation.y > WORLD_HALF_HEIGHT {
        translation.y = -WORLD_HALF_HEIGHT;
    }
}