mod creature;
mod brain;
mod config;
mod mlp;
mod simulation;
mod sense;
mod utils;
mod visualization;

use bevy::prelude::*;
use config::SimulationConfig;
use simulation::SimulationPlugin;
use visualization::VisualizationPlugin;

fn main() {
    dotenvy::dotenv()
        .or_else(|_| dotenvy::from_filename(".env.example"))
        .ok();

    App::new()
        .insert_resource(SimulationConfig::from_env())
        .add_plugins(DefaultPlugins)
        .add_plugins(SimulationPlugin)
        .add_plugins(VisualizationPlugin)
        .run();
}