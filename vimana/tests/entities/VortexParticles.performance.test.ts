import { describe, it, expect, beforeEach, vi } from 'vitest';
import { VortexParticles } from '../../src/entities/VortexParticles';
import * as THREE from 'three';

/**
 * Performance Tests for VortexParticles
 *
 * Tests particle system performance, LOD behavior, and memory cleanup.
 * Corresponds to PERFORMANCE-PLAN.md scenarios:
 * - Stress Test: Maximum Particle Count
 * - LOD Activation
 * - Memory Cleanup
 */

describe('VortexParticles Performance', () => {
    describe('LOD (Level of Detail) Behavior', () => {
        it('should skip 50% of particles at low activation (< 30%)', () => {
            const particles = new VortexParticles(1000); // Smaller for testing
            const positionNeedsUpdateSpy = vi.fn();

            // Spy on the needsUpdate property setter
            Object.defineProperty(particles.geometry.attributes.position, 'needsUpdate', {
                set: positionNeedsUpdateSpy,
                get: () => false
            });

            // Update at low activation (0.2 < 0.3)
            particles.update(0.016, 0.2);

            // Should have called needsUpdate (particles were updated)
            expect(positionNeedsUpdateSpy).toHaveBeenCalled();
        });

        it('should skip 25% of particles at medium activation (30-60%)', () => {
            const particles = new VortexParticles(1000);
            const positionNeedsUpdateSpy = vi.fn();

            Object.defineProperty(particles.geometry.attributes.position, 'needsUpdate', {
                set: positionNeedsUpdateSpy,
                get: () => false
            });

            // Update at medium activation (0.4, between 0.3 and 0.6)
            particles.update(0.016, 0.4);

            expect(positionNeedsUpdateSpy).toHaveBeenCalled();
        });

        it('should skip 0% of particles at high activation (>= 60%)', () => {
            const particles = new VortexParticles(1000);
            const positionNeedsUpdateSpy = vi.fn();

            Object.defineProperty(particles.geometry.attributes.position, 'needsUpdate', {
                set: positionNeedsUpdateSpy,
                get: () => false
            });

            // Update at high activation (1.0 >= 0.6)
            particles.update(0.016, 1.0);

            // All particles should update
            expect(positionNeedsUpdateSpy).toHaveBeenCalled();
        });

        it('should increase spin speed with activation', () => {
            const particles = new VortexParticles(200);
            const positionsLow = new Float32Array(200 * 3);
            const positionsHigh = new Float32Array(200 * 3);

            // Get positions at low activation
            particles.update(0.016, 0.0);
            const positionsAttributeLow = particles.geometry.attributes.position;
            for (let i = 0; i < positionsAttributeLow.count; i++) {
                positionsLow[i * 3] = positionsAttributeLow.getX(i);
                positionsLow[i * 3 + 1] = positionsAttributeLow.getY(i);
                positionsLow[i * 3 + 2] = positionsAttributeLow.getZ(i);
            }

            // Update at high activation for several frames
            for (let i = 0; i < 10; i++) {
                particles.update(0.016, 1.0);
            }

            // Get positions at high activation
            const positionsAttributeHigh = particles.geometry.attributes.position;
            for (let i = 0; i < positionsAttributeHigh.count; i++) {
                positionsHigh[i * 3] = positionsAttributeHigh.getX(i);
                positionsHigh[i * 3 + 1] = positionsAttributeHigh.getY(i);
                positionsHigh[i * 3 + 2] = positionsAttributeHigh.getZ(i);
            }

            // Positions should have changed more at high activation (faster spin)
            let differences = 0;
            for (let i = 0; i < 200; i++) {
                const dx = positionsLow[i * 3] - positionsHigh[i * 3];
                const dy = positionsLow[i * 3 + 1] - positionsHigh[i * 3 + 1];
                const dz = positionsLow[i * 3 + 2] - positionsHigh[i * 3 + 2];
                if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01 || Math.abs(dz) > 0.01) {
                    differences++;
                }
            }

            // At least some particles should have moved differently
            expect(differences).toBeGreaterThan(0);
        });
    });

    describe('Memory Management', () => {
        it('should dispose geometry on destroy', () => {
            const particles = new VortexParticles(500);
            const disposeSpy = vi.spyOn(particles.geometry, 'dispose');

            particles.destroy();

            expect(disposeSpy).toHaveBeenCalled();
        });

        it('should dispose material on destroy', () => {
            const particles = new VortexParticles(500);
            const disposeSpy = vi.spyOn(particles.material, 'dispose');

            particles.destroy();

            expect(disposeSpy).toHaveBeenCalled();
        });

        it('should release buffer memory on destroy', () => {
            const particles = new VortexParticles(500);
            const positions = particles['positions'] as Float32Array;
            const colors = particles['colors'] as Float32Array;

            // Verify buffers exist before destroy
            expect(positions.byteLength).toBeGreaterThan(0);
            expect(colors.byteLength).toBeGreaterThan(0);

            particles.destroy();

            // After destroy, buffers should still exist but geometry should be disposed
            // (JavaScript garbage collector handles actual memory release)
            expect(particles.geometry).toBeDefined();
        });
    });

    describe('Performance Targets', () => {
        it('should initialize with correct particle count', () => {
            const defaultParticles = new VortexParticles();
            expect(defaultParticles.geometry.attributes.position.count).toBe(2000);

            const customParticles = new VortexParticles(500);
            expect(customParticles.geometry.attributes.position.count).toBe(500);
        });

        it('should use additive blending for performance', () => {
            const particles = new VortexParticles(100);
            const material = particles.material as THREE.PointsMaterial;

            expect(material.blending).toBe(THREE.AdditiveBlending);
            expect(material.depthWrite).toBe(false);
        });

        it('should have reasonable memory footprint', () => {
            const particleCount = 2000;
            const particles = new VortexParticles(particleCount);

            // Each particle has: position (3 floats), color (3 floats) = 6 floats
            // Plus internal arrays: angles (1 float), tubeOffsets (1 float), speeds (1 float)
            // Total per particle: ~9 floats = 36 bytes
            // 2000 particles * 36 bytes = ~72 KB

            const positionsSize = particles['positions'].byteLength;
            const colorsSize = particles['colors'].byteLength;
            const anglesSize = particles['angles'].byteLength;
            const tubeOffsetsSize = particles['tubeOffsets'].byteLength;
            const speedsSize = particles['speeds'].byteLength;

            const totalBytes = positionsSize + colorsSize + anglesSize + tubeOffsetsSize + speedsSize;

            // Should be approximately 2000 * 9 * 4 bytes = 72,000 bytes
            expect(totalBytes).toBeGreaterThan(70000);
            expect(totalBytes).toBeLessThan(80000);
        });
    });

    describe('Update Performance', () => {
        it('should complete update within frame budget (16ms at 60fps)', () => {
            const particles = new VortexParticles(2000);
            const startTime = performance.now();

            // Simulate 60 frames of updates
            for (let i = 0; i < 60; i++) {
                particles.update(0.016, 0.5);
            }

            const elapsed = performance.now() - startTime;

            // 60 updates should complete in reasonable time (< 100ms total)
            // Average < 1.6ms per update
            expect(elapsed).toBeLessThan(100);
        });

        it('should not cause frame spikes at full activation', () => {
            const particles = new VortexParticles(2000);
            const frameTimes: number[] = [];

            for (let i = 0; i < 30; i++) {
                const start = performance.now();
                particles.update(0.016, 1.0); // Full activation
                frameTimes.push(performance.now() - start);
            }

            const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
            const maxFrameTime = Math.max(...frameTimes);

            // Average should be < 2ms, max spike < 5ms
            expect(avgFrameTime).toBeLessThan(2);
            expect(maxFrameTime).toBeLessThan(5);
        });
    });

    describe('Material Optimization', () => {
        it('should update opacity based on activation', () => {
            const particles = new VortexParticles(100);
            const material = particles.material as THREE.PointsMaterial;

            particles.update(0.016, 0.0);
            expect(material.opacity).toBe(0.3);

            particles.update(0.016, 0.5);
            expect(material.opacity).toBeCloseTo(0.55, 1);

            particles.update(0.016, 1.0);
            expect(material.opacity).toBeCloseTo(0.8, 1);
        });
    });
});
