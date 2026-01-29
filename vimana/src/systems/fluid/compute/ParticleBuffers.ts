/**
 * ParticleBuffers.ts - Storage Buffer Management for TSL Compute
 * ===========================================================
 *
 * Manages GPU storage buffers for fluid particle simulation using TSL.
 * Provides Three.js StorageBufferAttribute instances for particle data.
 *
 * Based on Story 4.7 implementation requirements for WebGPU fluid activation.
 */

import * as THREE from 'three';
import { FLUID_CONFIG, vec3 } from '../types';

const logger = {
  log: (msg: string, ...args: unknown[]) => console.log(`[ParticleBuffers] ${msg}`, ...args),
  warn: (msg: string, ...args: unknown[]) => console.warn(`[ParticleBuffers] ${msg}`, ...args),
  error: (msg: string, ...args: unknown[]) => console.error(`[ParticleBuffers] ${msg}`, ...args),
};

/**
 * Particle data structure for storage buffers
 * Matches TSL instancedArray layout: vec3 for each property
 */
export interface ParticleBufferData {
  positions: Float32Array;
  velocities: Float32Array;
  forces: Float32Array;
  densities: Float32Array;
}

/**
 * Result from buffer initialization
 */
export interface BufferResult {
  positionBuffer: THREE.InstancedBufferAttribute;
  velocityBuffer: THREE.InstancedBufferAttribute;
  forceBuffer: THREE.InstancedBufferAttribute;
  densityBuffer: THREE.InstancedBufferAttribute;
  count: number;
}

/**
 * Storage buffer management for fluid particles
 * Uses InstancedBufferAttribute for TSL compute shader compatibility
 */
export class ParticleBuffers {
  public readonly particleCount: number;
  public readonly maxParticles: number;

  // TSL-compatible storage buffers (InstancedBufferAttribute)
  public positionBuffer: THREE.InstancedBufferAttribute;
  public velocityBuffer: THREE.InstancedBufferAttribute;
  public forceBuffer: THREE.InstancedBufferAttribute;
  public densityBuffer: THREE.InstancedBufferAttribute;

  // Raw data arrays for CPU access
  private positionArray: Float32Array;
  private velocityArray: Float32Array;
  private forceArray: Float32Array;
  private densityArray: Float32Array;

  constructor(particleCount: number = FLUID_CONFIG.maxParticles) {
    this.particleCount = particleCount;
    this.maxParticles = FLUID_CONFIG.maxParticles;

    const stride = 3; // vec3

    // Initialize data arrays
    this.positionArray = new Float32Array(particleCount * stride);
    this.velocityArray = new Float32Array(particleCount * stride);
    this.forceArray = new Float32Array(particleCount * stride);
    this.densityArray = new Float32Array(particleCount);

    // Create InstancedBufferAttribute for TSL compute compatibility
    // These can be used with instancedArray() in TSL
    this.positionBuffer = new THREE.InstancedBufferAttribute(this.positionArray, stride);
    this.velocityBuffer = new THREE.InstancedBufferAttribute(this.velocityArray, stride);
    this.forceBuffer = new THREE.InstancedBufferAttribute(this.forceArray, stride);
    this.densityBuffer = new THREE.InstancedBufferAttribute(this.densityArray, 1);

    // Mark for update
    this.positionBuffer.setUsage(THREE.DynamicDrawUsage);
    this.velocityBuffer.setUsage(THREE.DynamicDrawUsage);
    this.forceBuffer.setUsage(THREE.DynamicDrawUsage);
    this.densityBuffer.setUsage(THREE.DynamicDrawUsage);

    logger.log(`Created particle buffers for ${particleCount} particles`);
  }

  /**
   * Initialize particles in a grid formation (dam-break pattern)
   * Particles spawn in a corner and will flow under gravity
   */
  public initializeGrid(
    boxSize: vec3 = [26, 26, 52],
    sphereRadius: number = 15.0
  ): void {
    const spacing = FLUID_CONFIG.particleSpacing;
    let index = 0;

    // Grid layout in corner (dam-break setup)
    for (let x = 0; x < boxSize[0] / spacing && index < this.particleCount; x++) {
      for (let y = 0; y < boxSize[1] / spacing && index < this.particleCount; y++) {
        for (let z = 0; z < boxSize[2] / spacing && index < this.particleCount; z++) {
          const i = index * 3;

          // Position: centered in box, offset from bottom
          this.positionArray[i] = (x * spacing) - boxSize[0] / 2;
          this.positionArray[i + 1] = (y * spacing) + 0.5; // Offset from floor
          this.positionArray[i + 2] = (z * spacing) - boxSize[2] / 2;

          // Velocity: initially at rest
          this.velocityArray[i] = 0;
          this.velocityArray[i + 1] = 0;
          this.velocityArray[i + 2] = 0;

          // Force: gravity applied in compute shader
          this.forceArray[i] = 0;
          this.forceArray[i + 1] = -9.8; // Initial gravity
          this.forceArray[i + 2] = 0;

          // Density: initial rest density
          this.densityArray[index] = FLUID_CONFIG.restDensity;

          index++;
        }
      }
    }

    this.markNeedsUpdate();
    logger.log(`Initialized ${index} particles in grid formation`);
  }

  /**
   * Initialize particles randomly within a sphere
   * Useful for testing sphere constraint behavior
   */
  public initializeSphere(
    center: vec3 = [0, 5, 0],
    radius: number = 8.0
  ): void {
    for (let i = 0; i < this.particleCount; i++) {
      const idx = i * 3;

      // Random point in sphere
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = Math.cbrt(Math.random()) * radius;

      this.positionArray[idx] = center[0] + r * Math.sin(phi) * Math.cos(theta);
      this.positionArray[idx + 1] = center[1] + r * Math.sin(phi) * Math.sin(theta);
      this.positionArray[idx + 2] = center[2] + r * Math.cos(phi);

      // Zero initial velocity
      this.velocityArray[idx] = 0;
      this.velocityArray[idx + 1] = 0;
      this.velocityArray[idx + 2] = 0;

      // Gravity force
      this.forceArray[idx] = 0;
      this.forceArray[idx + 1] = -9.8;
      this.forceArray[idx + 2] = 0;

      this.densityArray[i] = FLUID_CONFIG.restDensity;
    }

    this.markNeedsUpdate();
    logger.log(`Initialized ${this.particleCount} particles in sphere formation`);
  }

  /**
   * Get particle at index
   */
  public getParticle(index: number): { position: vec3; velocity: vec3; force: vec3; density: number } {
    const i = index * 3;
    return {
      position: [
        this.positionArray[i],
        this.positionArray[i + 1],
        this.positionArray[i + 2],
      ],
      velocity: [
        this.velocityArray[i],
        this.velocityArray[i + 1],
        this.velocityArray[i + 2],
      ],
      force: [
        this.forceArray[i],
        this.forceArray[i + 1],
        this.forceArray[i + 2],
      ],
      density: this.densityArray[index],
    };
  }

  /**
   * Set particle at index
   */
  public setParticle(
    index: number,
    position?: vec3,
    velocity?: vec3,
    force?: vec3,
    density?: number
  ): void {
    const i = index * 3;

    if (position) {
      this.positionArray[i] = position[0];
      this.positionArray[i + 1] = position[1];
      this.positionArray[i + 2] = position[2];
    }

    if (velocity) {
      this.velocityArray[i] = velocity[0];
      this.velocityArray[i + 1] = velocity[1];
      this.velocityArray[i + 2] = velocity[2];
    }

    if (force) {
      this.forceArray[i] = force[0];
      this.forceArray[i + 1] = force[1];
      this.forceArray[i + 2] = force[2];
    }

    if (density !== undefined) {
      this.densityArray[index] = density;
    }

    this.markNeedsUpdate();
  }

  /**
   * Mark all buffers for GPU update
   */
  public markNeedsUpdate(): void {
    this.positionBuffer.needsUpdate = true;
    this.velocityBuffer.needsUpdate = true;
    this.forceBuffer.needsUpdate = true;
    this.densityBuffer.needsUpdate = true;
  }

  /**
   * Get raw data for debugging/inspection
   */
  public getData(): ParticleBufferData {
    return {
      positions: this.positionArray,
      velocities: this.velocityArray,
      forces: this.forceArray,
      densities: this.densityArray,
    };
  }

  /**
   * Dispose of all buffers
   */
  public dispose(): void {
    this.positionBuffer.dispose();
    this.velocityBuffer.dispose();
    this.forceBuffer.dispose();
    this.densityBuffer.dispose();
    logger.log('Particle buffers disposed');
  }

  /**
   * Create buffer result for passing to other systems
   */
  public toBufferResult(): BufferResult {
    return {
      positionBuffer: this.positionBuffer,
      velocityBuffer: this.velocityBuffer,
      forceBuffer: this.forceBuffer,
      densityBuffer: this.densityBuffer,
      count: this.particleCount,
    };
  }
}

/**
 * Create particle buffers with default initialization
 */
export function createParticleBuffers(
  particleCount: number = FLUID_CONFIG.maxParticles,
  initPattern: 'grid' | 'sphere' = 'grid',
  initParams?: { boxSize?: vec3; sphereRadius?: number; sphereCenter?: vec3 }
): ParticleBuffers {
  const buffers = new ParticleBuffers(particleCount);

  if (initPattern === 'grid') {
    buffers.initializeGrid(initParams?.boxSize);
  } else if (initPattern === 'sphere') {
    buffers.initializeSphere(initParams?.sphereCenter, initParams?.sphereRadius);
  }

  return buffers;
}

export default ParticleBuffers;
