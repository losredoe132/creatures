mod creature;
mod config;
mod simulation;
mod visualization;

use bevy::prelude::*;
use config::WorldBounds;
use simulation::SimulationPlugin;
use visualization::VisualizationPlugin;

fn main() {
    dotenvy::dotenv().ok();

    App::new()
        .insert_resource(WorldBounds::from_env())
        .add_plugins(DefaultPlugins)
        .add_plugins(SimulationPlugin)
        .add_plugins(VisualizationPlugin)
        .run();
}