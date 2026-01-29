/**
 * MLSMPMStages.ts - TSL Compute Functions for MLS-MPM Fluid Simulation
 * ==================================================================
 *
 * Implements the MLS-MPM (Moving Least Squares Material Point Method) algorithm
 * stages using TSL compute functions. These functions can be dispatched via
 * Three.js WebGPU renderer's compute() API.
 *
 * Based on:
 * - Story 4.7 implementation requirements for WebGPU fluid activation
 * - https://github.com/matsuoka-601/WaterBall
 * - Three.js TSL compute shader patterns
 */

import {
  Fn,
  instanceIndex,
  float,
  vec3,
  uniform,
  instancedBufferAttribute,
  add,
  sub,
  mul,
  div,
  dot,
  cross,
  length,
  normalize,
  distance,
  sin,
  cos,
  pow,
  abs,
  min,
  max,
  floor,
  sqrt,
  clamp,
  smoothstep,
  If,
  Else,
  assign,
  cond,
  vec2,
  int,
  uint,
} from 'three/tsl';
import * as THREE from 'three';
import { FLUID_CONFIG } from '../types';

const logger = {
  log: (msg: string, ...args: unknown[]) => console.log(`[MLSMPMStages] ${msg}`, ...args),
  warn: (msg: string, ...args: unknown[]) => console.warn(`[MLSMPMStages] ${msg}`, ...args),
};

// ============================================================================
// TSL COMPUTE STAGE DEFINITIONS
// ============================================================================

/**
 * Uniform values for MLS-MPM simulation
 * These are shared across all compute stages
 */
export interface MLSMPMUniforms {
  deltaTime: THREE.UniformNode;
  gravity: THREE.UniformNode;
  restDensity: THREE.UniformNode;
  stiffness: THREE.UniformNode;
  viscosity: THREE.UniformNode;
  sphereRadius: THREE.UniformNode;
  sphereCenter: THREE.UniformNode;
  gridResolution: THREE.UniformNode;
  boxSize: THREE.UniformNode;
  dt: THREE.UniformNode;
}

/**
 * Default uniform values
 */
export const DEFAULT_MLSPM_UNIFORMS = {
  deltaTime: 0.016,      // 60 FPS
  gravity: 9.8,
  restDensity: FLUID_CONFIG.restDensity,
  stiffness: FLUID_CONFIG.stiffness,
  viscosity: FLUID_CONFIG.dynamicViscosity,
  sphereRadius: 15.0,
  sphereCenter: new THREE.Vector3(0, 0, 0),
  gridResolution: FLUID_CONFIG.gridResolution,
  boxSize: new THREE.Vector3(52, 52, 52),
  dt: FLUID_CONFIG.dt,
};

/**
 * Create TSL uniform nodes for MLS-MPM simulation
 */
export function createMLSMPMUniforms(values: Partial<typeof DEFAULT_MLSPM_UNIFORMS> = {}): MLSMPMUniforms {
  const merged = { ...DEFAULT_MLSPM_UNIFORMS, ...values };

  return {
    deltaTime: uniform(merged.deltaTime),
    gravity: uniform(merged.gravity),
    restDensity: uniform(merged.restDensity),
    stiffness: uniform(merged.stiffness),
    viscosity: uniform(merged.viscosity),
    sphereRadius: uniform(merged.sphereRadius),
    sphereCenter: uniform(merged.sphereCenter),
    gridResolution: uniform(merged.gridResolution),
    boxSize: uniform(merged.boxSize),
    dt: uniform(merged.dt),
  };
}

// ============================================================================
// STAGE 1: PARTICLE UPDATE (Simplified Physics)
// ============================================================================

/**
 * Stage 1: Update particle positions and apply basic physics
 * This is a simplified version that applies:
 * - Gravity
 * - Velocity integration
 * - Boundary constraints
 * - Sphere constraint (for tunnel animation)
 *
 * @param positionBuffer - InstancedBufferAttribute for positions
 * @param velocityBuffer - InstancedBufferAttribute for velocities
 * @param forceBuffer - InstancedBufferAttribute for forces
 * @param uniforms - Simulation uniform values
 * @returns TSL compute function
 */
export function createParticleUpdateStage(
  positionBuffer: THREE.InstancedBufferAttribute,
  velocityBuffer: THREE.InstancedBufferAttribute,
  forceBuffer: THREE.InstancedBufferAttribute,
  uniforms: MLSMPMUniforms
  // @ts-ignore - TSL compute function types
) {
  // Create TSL buffer attributes
  const positions = instancedBufferAttribute(positionBuffer, 'vec3');
  const velocities = instancedBufferAttribute(velocityBuffer, 'vec3');
  const forces = instancedBufferAttribute(forceBuffer, 'vec3');

  // Particle index
  const idx = instanceIndex;

  // Get current particle data
  const position = positions.element(idx);
  const velocity = velocities.element(idx);
  const force = forces.element(idx);

  // Gravity vector
  const gravityVec = vec3(0, float(0).sub(uniforms.gravity), 0);

  // Apply gravity to force
  const totalForce = force.add(gravityVec);

  // Update velocity: v = v + (F/m * dt)
  // Assuming unit mass for particles
  const newVelocity = velocity.add(totalForce.mul(uniforms.dt));

  // Update position: p = p + v * dt
  const newPosition = position.add(newVelocity.mul(uniforms.dt));

  // ============================================================================
  // BOUNDARY CONSTRAINTS with If()
  // ============================================================================

  const boxHalf = uniforms.boxSize.mul(0.5);
  const damping = float(0.5); // Velocity damping on collision

  // X-axis boundary
  const constrainedX = Fn(() => {
    If(newPosition.x.abs().greaterThan(boxHalf.x), () => {
      // Reflect velocity
      newVelocity.x.assign(newVelocity.x.negate().mul(damping));
      // Clamp position to boundary
      newPosition.x.assign(newPosition.x.sign().mul(boxHalf.x));
    });
  })();

  // Z-axis boundary
  const constrainedZ = Fn(() => {
    If(newPosition.z.abs().greaterThan(boxHalf.z), () => {
      newVelocity.z.assign(newVelocity.z.negate().mul(damping));
      newPosition.z.assign(newPosition.z.sign().mul(boxHalf.z));
    });
  })();

  // Y-axis boundary (floor)
  const constrainedY = Fn(() => {
    If(newPosition.y.lessThan(0), () => {
      newVelocity.y.assign(newVelocity.y.negate().mul(damping));
      newPosition.y.assign(float(0));
    });
  })();

  // ============================================================================
  // SPHERE CONSTRAINT (for tunnel animation)
  // ============================================================================

  // Distance from sphere center (in XZ plane)
  const distXZ = distance(
    vec2(newPosition.x, newPosition.z),
    vec2(uniforms.sphereCenter.x, uniforms.sphereCenter.z)
  );

  // Sphere constraint: particles outside radius are pulled inward
  const sphereConstrained = Fn(() => {
    If(distXZ.greaterThan(uniforms.sphereRadius), () => {
      // Direction from center to particle
      const dirX = normalize(vec2(newPosition.x, newPosition.z).sub(
        vec2(uniforms.sphereCenter.x, uniforms.sphereCenter.z)
      ));

      // Pull position to sphere surface
      newPosition.x.assign(uniforms.sphereCenter.x.add(dirX.mul(uniforms.sphereRadius).x));
      newPosition.z.assign(uniforms.sphereCenter.z.add(dirX.mul(uniforms.sphereRadius).y));

      // Reflect velocity inward
      const dotVel = newVelocity.x.mul(dirX.x).add(newVelocity.z.mul(dirX.y));
      const inwardVelX = dirX.x.mul(float(1).sub(dotVel.mul(float(0.5))));
      const inwardVelZ = dirX.y.mul(float(1).sub(dotVel.mul(float(0.5))));

      newVelocity.x.assign(inwardVelX.mul(damping));
      newVelocity.z.assign(inwardVelZ.mul(damping));
    });
  })();

  // ============================================================================
  // WRITE BACK TO BUFFERS
  // ============================================================================

  // Update buffers with new values
  positions.element(idx).assign(newPosition);
  velocities.element(idx).assign(newVelocity);

  // Reset force for next frame (gravity will be re-applied)
  forces.element(idx).assign(vec3(0, 0, 0));

  return Fn(() => {
    constrainedX;
    constrainedY;
    constrainedZ;
    sphereConstrained;
  }, 'particle_update');
}

// ============================================================================
// STAGE 2: PARTICLE-TO-GRID (P2G) - Simplified
// ============================================================================

/**
 * Stage 2: Transfer particle masses and velocities to grid
 * This is a simplified P2G stage for basic fluid behavior
 *
 * @param positionBuffer - Particle positions
 * @param velocityBuffer - Particle velocities
 * @param gridBuffer - Grid cell data
 * @param uniforms - Simulation parameters
 * @returns TSL compute function
 */
export function createP2GStage(
  positionBuffer: THREE.InstancedBufferAttribute,
  velocityBuffer: THREE.InstancedBufferAttribute,
  gridBuffer: THREE.InstancedBufferAttribute,
  uniforms: MLSMPMUniforms
  // @ts-ignore
) {
  const positions = instancedBufferAttribute(positionBuffer, 'vec3');
  const velocities = instancedBufferAttribute(velocityBuffer, 'vec3');
  const grid = instancedBufferAttribute(gridBuffer, 'vec4'); // mass, vx, vy, vz

  const idx = instanceIndex;
  const position = positions.element(idx);
  const velocity = velocities.element(idx);

  // Compute grid cell coordinates
  const gridX = floor(position.x.mul(uniforms.gridResolution));
  const gridY = floor(position.y.mul(uniforms.gridResolution));
  const gridZ = floor(position.z.mul(uniforms.gridResolution));

  // Grid cell index (3D to 1D)
  const gridRes = uniforms.gridResolution;
  const gridIdx = uint(gridX.add(gridY.mul(gridRes)).add(gridZ.mul(gridRes).mul(gridRes)));

  // Transfer particle velocity to grid (simplified - no splat weights)
  const gridCell = grid.element(gridIdx);
  gridCell.x.addAssign(float(1)); // Accumulate mass
  gridCell.y.addAssign(velocity.x); // Accumulate velocity
  gridCell.z.addAssign(velocity.y);
  gridCell.w.addAssign(velocity.z);
  grid.element(gridIdx).assign(gridCell);

  return Fn(() => {}, 'p2g');
}

// ============================================================================
// STAGE 3: GRID UPDATE
// ============================================================================

/**
 * Stage 3: Update grid velocities and apply constraints
 *
 * @param gridBuffer - Grid cell data
 * @param uniforms - Simulation parameters
 * @returns TSL compute function
 */
export function createGridUpdateStage(
  gridBuffer: THREE.InstancedBufferAttribute,
  uniforms: MLSMPMUniforms
  // @ts-ignore
) {
  const grid = instancedBufferAttribute(gridBuffer, 'vec4');
  const idx = instanceIndex;
  const cell = grid.element(idx);

  // Normalize velocity by accumulated mass
  const mass = cell.x;
  const normalizedVelocity = If(
    mass.greaterThan(float(0)),
    vec3(cell.y.div(mass), cell.z.div(mass), cell.w.div(mass)),
    vec3(0, 0, 0)
  );

  // Apply damping
  const dampedVelocity = normalizedVelocity.mul(float(0.99));

  // Write back
  cell.assign(vec4(mass, dampedVelocity.x, dampedVelocity.y, dampedVelocity.z));
  grid.element(idx).assign(cell);

  return Fn(() => {}, 'grid_update');
}

// ============================================================================
// STAGE 4: GRID-TO-PARTICLE (G2P)
// ============================================================================

/**
 * Stage 4: Transfer grid velocities back to particles
 *
 * @param positionBuffer - Particle positions
 * @param velocityBuffer - Particle velocities
 * @param gridBuffer - Grid cell data
 * @param uniforms - Simulation parameters
 * @returns TSL compute function
 */
export function createG2PStage(
  positionBuffer: THREE.InstancedBufferAttribute,
  velocityBuffer: THREE.InstancedBufferAttribute,
  gridBuffer: THREE.InstancedBufferAttribute,
  uniforms: MLSMPMUniforms
  // @ts-ignore
) {
  const positions = instancedBufferAttribute(positionBuffer, 'vec3');
  const velocities = instancedBufferAttribute(velocityBuffer, 'vec3');
  const grid = instancedBufferAttribute(gridBuffer, 'vec4');

  const idx = instanceIndex;
  const position = positions.element(idx);

  // Compute grid cell coordinates
  const gridX = floor(position.x.mul(uniforms.gridResolution));
  const gridY = floor(position.y.mul(uniforms.gridResolution));
  const gridZ = floor(position.z.mul(uniforms.gridResolution));

  const gridRes = uniforms.gridResolution;
  const gridIdx = uint(gridX.add(gridY.mul(gridRes)).add(gridZ.mul(gridRes).mul(gridRes)));

  // Get grid velocity
  const cell = grid.element(gridIdx);
  const gridVelocity = vec3(cell.y, cell.z, cell.w);

  // Update particle velocity from grid (PIC: Particle In Cell)
  // Could blend with FLIP (Fluid Implicit Particle) for better accuracy
  const picWeight = float(0.9); // PIC/FLIP blend
  const currentVelocity = velocities.element(idx);
  const newVelocity = currentVelocity.mul(float(1).sub(picWeight)).add(gridVelocity.mul(picWeight));

  velocities.element(idx).assign(newVelocity);

  return Fn(() => {}, 'g2p');
}

// ============================================================================
// SIMPLE FLUID SIMULATION (All-in-one for easier integration)
// ============================================================================

/**
 * Create a simplified fluid simulation compute function
 * Combines all stages into a single compute pass for easier integration
 *
 * This implements:
 * - Velocity integration with gravity
 * - Position update
 * - Boundary collision
 * - Sphere constraint for tunnel animation
 *
 * @param positionBuffer - Particle positions (InstancedBufferAttribute)
 * @param velocityBuffer - Particle velocities (InstancedBufferAttribute)
 * @param forceBuffer - Particle forces (InstancedBufferAttribute)
 * @param uniforms - Simulation parameters
 * @returns TSL compute function ready for dispatch
 */
export function createSimpleFluidCompute(
  positionBuffer: THREE.InstancedBufferAttribute,
  velocityBuffer: THREE.InstancedBufferAttribute,
  forceBuffer: THREE.InstancedBufferAttribute,
  uniforms: MLSMPMUniforms
) {
  // @ts-ignore - TSL instancedBufferAttribute
  const positions = instancedBufferAttribute(positionBuffer, 'vec3');
  // @ts-ignore
  const velocities = instancedBufferAttribute(velocityBuffer, 'vec3');
  // @ts-ignore
  const forces = instancedBufferAttribute(forceBuffer, 'vec3');

  const idx = instanceIndex;
  const dt = uniforms.dt;
  const damping = float(0.5);
  const restitution = float(0.3); // Bounciness

  // Get current particle data
  const position = positions.element(idx);
  const velocity = velocities.element(idx);
  const force = forces.element(idx);

  // Gravity force
  const gravityForce = vec3(0, float(0).sub(uniforms.gravity), 0);

  // Total force with gravity
  const totalForce = force.add(gravityForce);

  // Symplectic Euler integration:
  // v_new = v_old + F * dt
  // p_new = p_old + v_new * dt
  const newVelocity = velocity.add(totalForce.mul(dt));
  const newPosition = position.add(newVelocity.mul(dt));

  // ============================================================================
  // COLLISION DETECTION with If()
  // ============================================================================

  const boxHalf = uniforms.boxSize.mul(0.5);

  // Floor collision (y = 0)
  If(newPosition.y.lessThan(0), () => {
    newPosition.y.assign(float(0));
    newVelocity.y.assign(newVelocity.y.negate().mul(damping));
  });

  // Ceiling collision
  If(newPosition.y.greaterThan(boxHalf.y.mul(2)), () => {
    newPosition.y.assign(boxHalf.y.mul(2));
    newVelocity.y.assign(newVelocity.y.negate().mul(damping));
  });

  // X-axis walls
  If(newPosition.x.abs().greaterThan(boxHalf.x), () => {
    const signX = newPosition.x.sign();
    newPosition.x.assign(signX.mul(boxHalf.x));
    newVelocity.x.assign(newVelocity.x.negate().mul(damping));
  });

  // Z-axis walls
  If(newPosition.z.abs().greaterThan(boxHalf.z), () => {
    const signZ = newPosition.z.sign();
    newPosition.z.assign(signZ.mul(boxHalf.z));
    newVelocity.z.assign(newVelocity.z.negate().mul(damping));
  });

  // ============================================================================
  // SPHERE CONSTRAINT (Tunnel Animation)
  // ============================================================================

  // Calculate distance from sphere center in XZ plane
  const distFromCenterX = newPosition.x.sub(uniforms.sphereCenter.x);
  const distFromCenterZ = newPosition.z.sub(uniforms.sphereCenter.z);
  const distXZ = sqrt(distFromCenterX.mul(distFromCenterX).add(distFromCenterZ.mul(distFromCenterZ)));

  // Apply sphere constraint
  If(distXZ.greaterThan(uniforms.sphereRadius), () => {
    // Normalize direction in XZ plane
    const dirX = distFromCenterX.div(distXZ);
    const dirZ = distFromCenterZ.div(distXZ);

    // Constrain position to sphere surface
    newPosition.x.assign(uniforms.sphereCenter.x.add(dirX.mul(uniforms.sphereRadius)));
    newPosition.z.assign(uniforms.sphereCenter.z.add(dirZ.mul(uniforms.sphereRadius)));

    // Reflect velocity toward center
    const velDotDir = newVelocity.x.mul(dirX).add(newVelocity.z.mul(dirZ));
    If(velDotDir.greaterThan(0), () => {
      // Velocity is outward, reflect it
      const reflectedX = newVelocity.x.sub(dirX.mul(velDotDir).mul(float(1).add(restitution)));
      const reflectedZ = newVelocity.z.sub(dirZ.mul(velDotDir).mul(float(1).add(restitution)));

      newVelocity.x.assign(reflectedX.mul(damping));
      newVelocity.z.assign(reflectedZ.mul(damping));
    });
  });

  // ============================================================================
  // WRITE RESULTS
  // ============================================================================

  positions.element(idx).assign(newPosition);
  velocities.element(idx).assign(newVelocity);

  // Reset force for next frame
  forces.element(idx).assign(vec3(0, 0, 0));

  // Return the compute function
  // @ts-ignore
  return Fn(() => {
    // All operations are applied via assign() calls in the If() blocks above
  }, 'simple_fluid_update');
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Update uniform values
 */
export function updateUniforms(uniforms: MLSMPMUniforms, values: Partial<typeof DEFAULT_MLSPM_UNIFORMS>): void {
  if (values.deltaTime !== undefined) uniforms.deltaTime.value = values.deltaTime;
  if (values.gravity !== undefined) uniforms.gravity.value = values.gravity;
  if (values.restDensity !== undefined) uniforms.restDensity.value = values.restDensity;
  if (values.stiffness !== undefined) uniforms.stiffness.value = values.stiffness;
  if (values.viscosity !== undefined) uniforms.viscosity.value = values.viscosity;
  if (values.sphereRadius !== undefined) uniforms.sphereRadius.value = values.sphereRadius;
  if (values.sphereCenter !== undefined) (uniforms.sphereCenter.value as THREE.Vector3).copy(values.sphereCenter);
  if (values.gridResolution !== undefined) uniforms.gridResolution.value = values.gridResolution;
  if (values.boxSize !== undefined) (uniforms.boxSize.value as THREE.Vector3).copy(values.boxSize);
  if (values.dt !== undefined) uniforms.dt.value = values.dt;
}

/**
 * Get current uniform values
 */
export function getUniformValues(uniforms: MLSMPMUniforms): typeof DEFAULT_MLSPM_UNIFORMS {
  return {
    deltaTime: uniforms.deltaTime.value as number,
    gravity: uniforms.gravity.value as number,
    restDensity: uniforms.restDensity.value as number,
    stiffness: uniforms.stiffness.value as number,
    viscosity: uniforms.viscosity.value as number,
    sphereRadius: uniforms.sphereRadius.value as number,
    sphereCenter: uniforms.sphereCenter.value as THREE.Vector3,
    gridResolution: uniforms.gridResolution.value as number,
    boxSize: uniforms.boxSize.value as THREE.Vector3,
    dt: uniforms.dt.value as number,
  };
}

export default {
  createMLSMPMUniforms,
  createParticleUpdateStage,
  createP2GStage,
  createGridUpdateStage,
  createG2PStage,
  createSimpleFluidCompute,
  updateUniforms,
  getUniformValues,
  DEFAULT_MLSPM_UNIFORMS,
};
