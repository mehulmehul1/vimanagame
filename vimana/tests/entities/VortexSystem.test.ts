import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import { VortexSystem } from '../../src/entities/VortexSystem';
import { WaterMaterial } from '../../src/entities/WaterMaterial';

describe('VortexSystem', () => {
    let vortexSystem: VortexSystem;

    beforeEach(() => {
        vortexSystem = new VortexSystem(100); // Reduced particle count for tests
    });

    afterEach(() => {
        if (vortexSystem) {
            vortexSystem.destroy();
        }
        vi.clearAllMocks();
    });

    it('should initialize with correct geometry and position', () => {
        expect(vortexSystem).toBeDefined();
        expect(vortexSystem).toBeInstanceOf(THREE.Group);

        // Check local position of mesh
        const mesh = vortexSystem.children.find(c => c.type === 'Mesh') as THREE.Mesh;
        expect(mesh).toBeDefined();
        expect(mesh.position.x).toBe(0);
        expect(mesh.position.y).toBe(0);
        expect(mesh.position.z).toBe(0);

        // Check world positioning defaults
        expect(vortexSystem.position.y).toBeGreaterThan(0);
    });

    it('should update activation levels', () => {
        vortexSystem.updateDuetProgress(0.5);
        expect(vortexSystem.getActivation()).toBe(0.5);

        vortexSystem.updateDuetProgress(1.5); // Should clamp
        expect(vortexSystem.getActivation()).toBe(1.0);

        vortexSystem.updateDuetProgress(-0.5); // Should clamp
        expect(vortexSystem.getActivation()).toBe(0);
    });

    it('should detect full activation', () => {
        vortexSystem.updateDuetProgress(0.98);
        expect(vortexSystem.isFullyActivated()).toBe(false);

        vortexSystem.updateDuetProgress(0.995);
        expect(vortexSystem.isFullyActivated()).toBe(true);
    });

    it('should link with water material', () => {
        const waterMaterial = new WaterMaterial();
        const setSpy = vi.spyOn(waterMaterial, 'setHarmonicResonance');

        vortexSystem.setWaterMaterial(waterMaterial);

        // Update should propagate to water
        vortexSystem.update(0.016, new THREE.Vector3(0, 0, 0));

        expect(setSpy).toHaveBeenCalled();
    });
});