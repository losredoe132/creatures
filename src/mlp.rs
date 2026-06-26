use bevy::prelude::Vec2;
use rand::Rng;

pub const MLP_INPUTS: usize = 5;
pub const MLP_HIDDEN: usize = 1;
pub const MLP_OUTPUTS: usize = 2;
pub const GENOME_LEN: usize = MLP_INPUTS * MLP_HIDDEN
    + MLP_HIDDEN
    + MLP_HIDDEN * MLP_OUTPUTS
    + MLP_OUTPUTS;

#[derive(Debug, Clone)]
pub struct Genome {
    pub genes: Vec<f32>,
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        let genes = (0..GENOME_LEN)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
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
    if genome.genes.len() != GENOME_LEN {
        return MovementOutput { vector: Vec2::ZERO };
    }

    let mut index = 0usize;

    let w1 = &genome.genes[index..index + MLP_INPUTS * MLP_HIDDEN];
    index += MLP_INPUTS * MLP_HIDDEN;

    let b1 = &genome.genes[index..index + MLP_HIDDEN];
    index += MLP_HIDDEN;

    let w2 = &genome.genes[index..index + MLP_HIDDEN * MLP_OUTPUTS];
    index += MLP_HIDDEN * MLP_OUTPUTS;

    let b2 = &genome.genes[index..index + MLP_OUTPUTS];

    let mut hidden = [0.0f32; MLP_HIDDEN];
    for h in 0..MLP_HIDDEN {
        let mut sum = b1[h];
        for i in 0..MLP_INPUTS {
            sum += features[i] * w1[h * MLP_INPUTS + i];
        }
        hidden[h] = sum.tanh();
    }

    let mut output = [0.0f32; MLP_OUTPUTS];
    for o in 0..MLP_OUTPUTS {
        let mut sum = b2[o];
        for h in 0..MLP_HIDDEN {
            sum += hidden[h] * w2[o * MLP_HIDDEN + h];
        }
        output[o] = sum.tanh();
    }

    MovementOutput {
        vector: Vec2::new(output[0], output[1]),
    }
}
