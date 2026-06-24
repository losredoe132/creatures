use bevy::prelude::Vec2;

pub fn sigmoid(x: f32) -> f32 {
	1.0 / (1.0 + (-x).exp())
}

pub fn limit_speed_sigmoid(velocity: Vec2, max_speed: f32, steepness: f32) -> Vec2 {
	if max_speed <= 0.0 {
		return Vec2::ZERO;
	}

	let speed = velocity.length();
	if speed <= f32::EPSILON {
		return velocity;
	}

	let normalized_speed = speed / max_speed;
	let limited_speed = max_speed * (2.0 * sigmoid(steepness * normalized_speed) - 1.0);
	velocity * (limited_speed / speed)
}
