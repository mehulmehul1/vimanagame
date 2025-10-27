/**
 * Dissolve Particle System
 *
 * Manages particle emission from dissolving edges with physics simulation.
 * Particles spawn at the dissolve boundary and animate with wave offsets.
 */

import * as THREE from "three";

export class DissolveParticleSystem {
  constructor(geometry, dispersion = 8.0, velocitySpread = 0.15) {
    this.geometry = geometry;
    this.count = geometry.getAttribute("position").array.length / 3;

    // Store parameters
    this.dispersion = dispersion; // Max travel distance
    this.velocitySpread = velocitySpread; // Velocity randomness

    // Scalar quantities
    this.maxOffsetArr = new Float32Array(this.count);
    this.scaleArr = new Float32Array(this.count);
    this.distArr = new Float32Array(this.count);
    this.rotationArr = new Float32Array(this.count);

    // Vector quantities
    this.currentPositionArr = new Float32Array(
      this.geometry.getAttribute("position").array
    );
    this.initPositionArr = new Float32Array(
      this.geometry.getAttribute("position").array
    );
    this.velocityArr = new Float32Array(this.count * 3);

    this.setAttributesValues();
  }

  /**
   * Initialize particle attributes with random values
   */
  setAttributesValues() {
    const minDispersion = this.dispersion * 0.25; // At least 25% of max

    for (let i = 0; i < this.count; i++) {
      let x = i * 3 + 0;
      let y = i * 3 + 1;
      let z = i * 3 + 2;

      // Max travel distance based on dispersion parameter
      this.maxOffsetArr[i] = Math.random() * this.dispersion + minDispersion;
      this.scaleArr[i] = Math.random();
      this.rotationArr[i] = Math.random() * 2 * Math.PI;

      // Velocities based on velocity spread parameter
      this.velocityArr[x] = (Math.random() - 0.5) * this.velocitySpread; // Can go left or right
      this.velocityArr[y] = Math.random() * this.velocitySpread + 0.05; // Mostly up
      this.velocityArr[z] = (Math.random() - 0.5) * this.velocitySpread; // Can go forward or back

      this.distArr[i] = 0.01;
    }

    this.setAttributes();
  }

  /**
   * Update particle positions and attributes each frame
   */
  updateAttributesValues() {
    for (let i = 0; i < this.count; i++) {
      this.rotationArr[i] += 0.1;

      let x = i * 3 + 0;
      let y = i * 3 + 1;
      let z = i * 3 + 2;

      const speed = 0.3;

      // Reduced wave influence to prevent convergence
      let waveOffset1 = Math.sin(this.currentPositionArr[y] * 2.0) * 0.08;
      let waveOffset2 = Math.sin(this.currentPositionArr[x] * 2.0) * 0.08;

      // Don't use Math.abs - let particles go in their natural direction
      this.currentPositionArr[x] += (this.velocityArr[x] + waveOffset1) * speed;
      this.currentPositionArr[y] += (this.velocityArr[y] + waveOffset2) * speed;
      this.currentPositionArr[z] += this.velocityArr[z] * speed;

      const initpos = new THREE.Vector3(
        this.initPositionArr[x],
        this.initPositionArr[y],
        this.initPositionArr[z]
      );
      const newpos = new THREE.Vector3(
        this.currentPositionArr[x],
        this.currentPositionArr[y],
        this.currentPositionArr[z]
      );
      const dist = initpos.distanceTo(newpos);
      this.distArr[i] = dist;

      if (dist > this.maxOffsetArr[i]) {
        this.currentPositionArr[x] = this.initPositionArr[x];
        this.currentPositionArr[y] = this.initPositionArr[y];
        this.currentPositionArr[z] = this.initPositionArr[z];
        this.distArr[i] = 0.01;
      }
    }
    this.setAttributes();
  }

  /**
   * Apply custom attributes to geometry
   * @private
   */
  setAttributes() {
    this.geometry.setAttribute(
      "aOffset",
      new THREE.BufferAttribute(this.maxOffsetArr, 1)
    );
    this.geometry.setAttribute(
      "aDist",
      new THREE.BufferAttribute(this.distArr, 1)
    );
    this.geometry.setAttribute(
      "aRotation",
      new THREE.BufferAttribute(this.rotationArr, 1)
    );
    this.geometry.setAttribute(
      "aScale",
      new THREE.BufferAttribute(this.scaleArr, 1)
    );
    this.geometry.setAttribute(
      "aPosition",
      new THREE.BufferAttribute(this.currentPositionArr, 3)
    );
    this.geometry.setAttribute(
      "aVelocity",
      new THREE.BufferAttribute(this.velocityArr, 3)
    );
  }

  /**
   * Dispose of resources
   */
  dispose() {
    // Attributes will be disposed with geometry
  }
}

export default DissolveParticleSystem;
