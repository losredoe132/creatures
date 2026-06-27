use bevy::prelude::Vec2;
use nalgebra as na;
use rand::Rng;

pub const MLP_INPUTS: usize = 8;
pub const MLP_OUTPUTS: usize = 2;

pub const GENOME_LEN: usize = MLP_INPUTS * MLP_OUTPUTS + MLP_OUTPUTS;

#[derive(Debug, Clone)]
pub struct Genome {
    pub genes: Vec<f32>,
}

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

pub fn mlp_movement(features: [f32; MLP_INPUTS], genome: &Genome) -> MovementOutput {
    assert_eq!(genome.genes.len(), GENOME_LEN, "Genome length mismatch");

    // x: 1x3
    let x: na::RowSVector<f32, MLP_INPUTS> = na::RowSVector::from_row_slice(&features);

    // W: 3x2 (row-major slice)
    let w: na::SMatrix<f32, MLP_INPUTS, MLP_OUTPUTS> =
        na::SMatrix::from_row_slice(&genome.genes[0..(MLP_INPUTS * MLP_OUTPUTS)]);

    // b: 1x2
    let b_start = MLP_INPUTS * MLP_OUTPUTS;
    let b: na::RowSVector<f32, MLP_OUTPUTS> =
        na::RowSVector::from_row_slice(&genome.genes[b_start..b_start + MLP_OUTPUTS]);

    // y: 1x2
    let y = x * w + b * 0.1;

    MovementOutput {
        vector: Vec2::new(y[0], y[1]),
    }
}
