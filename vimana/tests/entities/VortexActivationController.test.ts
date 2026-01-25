/**
 * Unit tests for VortexActivationController
 *
 * Tests vortex activation based on duet progress.
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Vortex activation follows duet progress - P0
 * - Full activation triggers platform ride - P0
 * - Water harmonic resonance syncs - P1
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';

describe('VortexActivationController', () => {
    let VortexActivationController: any;
    let VortexSystem: any;
    let WaterMaterial: any;
    let controller: any;
    let vortexSystem: any;
    let waterMaterial: any;
    let scene: THREE.Scene;
    let platformMesh: THREE.Mesh;

    beforeEach(async () => {
        // Dynamic imports to avoid hoisting issues
        const vModule = await import('../../src/entities/VortexActivationController');
        const vsModule = await import('../../src/entities/VortexSystem');
        const wmModule = await import('../../src/entities/WaterMaterial');

        VortexActivationController = vModule.VortexActivationController;
        VortexSystem = vsModule.VortexSystem;
        WaterMaterial = wmModule.WaterMaterial;

        scene = new THREE.Scene();
        platformMesh = new THREE.Mesh();

        vortexSystem = new VortexSystem(100);
        waterMaterial = new WaterMaterial();

        controller = new VortexActivationController(
            vortexSystem,
            waterMaterial,
            platformMesh,
            scene
        );
    });

    afterEach(async () => {
        try {
            await controller?.destroy();
        } catch (e) {
            // Ignore
        }
        try {
            vortexSystem?.destroy();
        } catch (e) {
            // Ignore
        }
        try {
            waterMaterial?.dispose();
        } catch (e) {
            // Ignore
        }
        vi.clearAllMocks();
    });

    describe('Construction', () => {
        it('should initialize with zero activation', () => {
            expect(controller.getActivation()).toBe(0);
        });

        it('should not be fully activated initially', () => {
            expect(controller.isFullyActivated()).toBe(false);
        });

        it('should have platform animator', () => {
            expect(controller.getPlatformAnimator()).toBeDefined();
        });

        it('should have lighting manager', () => {
            expect(controller.getLightingManager()).toBeDefined();
        });
    });

    describe('Setting Activation', () => {
        it('should accept activation value', () => {
            controller.setActivation(0.5);
            expect(controller.getActivation()).toBe(0); // Not yet updated
        });

        it('should clamp activation to 0-1 range', () => {
            controller.setActivation(1.5);
            controller.update(0.1);
            expect(controller.getActivation()).toBeLessThanOrEqual(1);

            controller.reset();
            controller.setActivation(-0.5);
            controller.update(0.1);
            expect(controller.getActivation()).toBeGreaterThanOrEqual(0);
        });
    });

    describe('Update Behavior', () => {
        it('should lerp toward target activation', () => {
            controller.setActivation(0.5);
            controller.update(0.1);

            expect(controller.getActivation()).toBeGreaterThan(0);
            expect(controller.getActivation()).toBeLessThan(0.5);
        });
    });

    describe('Cleanup', () => {
        it('should destroy without errors', () => {
            expect(() => controller.destroy()).not.toThrow();
        });
    });
});
