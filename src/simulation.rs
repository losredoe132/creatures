use bevy::prelude::*;
use rand::Rng;

use crate::config::{SimulationConfig, WorldBounds};
use crate::creature::{Animal, EnergyPosition, Movable, Plant};

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_world)
            .add_systems(Update, move_animals);
    }
}

fn setup_world(mut commands: Commands, config: Res<SimulationConfig>) {
    let animal = Animal {
        position: Vec2::new(0.0, 0.0),
        velocity: Vec2::new(100.0, 1.0),
        energy: 100.0,
        radius: 20.0,
        color: Color::srgb(0.8, 0.2, 0.4),
    };


    let mut rng = rand::thread_rng();
    for _ in 0..config.spawn_config.n_plants {
        let plant = Plant {
            position: Vec2::new(
                rng.gen_range(-config.world_bounds.half_width..config.world_bounds.half_width),
                rng.gen_range(-config.world_bounds.half_height..config.world_bounds.half_height),
            ),
            energy: 60.0,
            radius: 14.0,
            color: Color::srgb(0.3, 0.6, 0.2),
        };
        commands.spawn(plant);
    }

    commands.spawn(animal);
}

fn move_animals(
    mut query: Query<(&mut Animal, &mut Transform)>,
    time: Res<Time>,
    config: Res<SimulationConfig>,
) {
    for (mut animal, mut transform) in &mut query {
        transform.translation += animal.velocity().extend(0.0) * time.delta_secs();
        ensure_torodial_world(&mut transform.translation, &config.world_bounds);
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
