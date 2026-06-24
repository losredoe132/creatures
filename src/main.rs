mod creature;
mod simulation;
mod visualization;

use bevy::prelude::*;
use simulation::SimulationPlugin;
use visualization::VisualizationPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SimulationPlugin)
        .add_plugins(VisualizationPlugin)
        .run();
}