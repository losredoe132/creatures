use bevy::prelude::Vec2;
use nalgebra as na;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const MLP_INPUTS: usize = 6;
pub const MLP_HIDDEN_1: usize = 12;
pub const MLP_OUTPUTS: usize = 2;

pub const W1_SIZE: usize = MLP_INPUTS * MLP_HIDDEN_1;
pub const W2_SIZE: usize = MLP_HIDDEN_1 * MLP_OUTPUTS;
pub const GENOME_LEN: usize = W1_SIZE + MLP_HIDDEN_1 + W2_SIZE + MLP_OUTPUTS;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub genes: Vec<f32>,
}

#[allow(dead_code)]
impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        let genes = (0..GENOME_LEN).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { genes }
    }

    pub fn mutated(&self, rng: &mut impl Rng, strength: f32) -> Self {
        if strength <= 0.0 {
            return self.clone();
        }

        let genes = self
            .genes
            .iter()
            .map(|gene| *gene + rng.gen_range(-strength..strength))
            .collect();
        Self { genes }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MovementOutput {
    pub vector: Vec2,
}

pub struct MlpActivations {
    pub hidden: [f32; MLP_HIDDEN_1],
    pub output: [f32; MLP_OUTPUTS],
}

pub fn mlp_forward(features: [f32; MLP_INPUTS], genome: &Genome) -> MlpActivations {
    assert_eq!(genome.genes.len(), GENOME_LEN, "Genome length mismatch");

    let x = na::RowSVector::<f32, MLP_INPUTS>::from_row_slice(&features);

    let w1 = na::SMatrix::<f32, MLP_INPUTS, MLP_HIDDEN_1>::from_row_slice(
        &genome.genes[..W1_SIZE],
    );
    let b1 = na::RowSVector::<f32, MLP_HIDDEN_1>::from_row_slice(
        &genome.genes[W1_SIZE..W1_SIZE + MLP_HIDDEN_1],
    );
    let hidden_act = (x * w1 + b1).map(|v| v.tanh());

    let w2_start = W1_SIZE + MLP_HIDDEN_1;
    let w2 = na::SMatrix::<f32, MLP_HIDDEN_1, MLP_OUTPUTS>::from_row_slice(
        &genome.genes[w2_start..w2_start + W2_SIZE],
    );
    let b2_start = w2_start + W2_SIZE;
    let b2 = na::RowSVector::<f32, MLP_OUTPUTS>::from_row_slice(
        &genome.genes[b2_start..b2_start + MLP_OUTPUTS],
    );
    let y = hidden_act * w2 + b2;

    MlpActivations {
        hidden: std::array::from_fn(|i| hidden_act[(0, i)]),
        output: [y[(0, 0)], y[(0, 1)]],
    }
}

pub fn mlp_movement(features: [f32; MLP_INPUTS], genome: &Genome) -> MovementOutput {
    let act = mlp_forward(features, genome);
    MovementOutput {
        vector: Vec2::new(act.output[0], act.output[1]),
    }
}
