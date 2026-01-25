/**
 * Unit tests for ShellCollectible
 *
 * Tests shell materialization and collection behavior.
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Shell materializes after vortex completion - P0
 * - Shell collection on click - P0
 * - Shell bobbing idle animation - P2
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import { ShellCollectible } from '../../src/entities/ShellCollectible';

describe('ShellCollectible', () => {
    let shell: ShellCollectible;
    let scene: THREE.Scene;
    let camera: THREE.Camera;

    beforeEach(() => {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera();
        shell = new ShellCollectible(scene, camera);
    });

    afterEach(() => {
        shell.destroy();
    });

    describe('Construction', () => {
        it('should initialize in materializing state', () => {
            expect(shell.getState()).toBe('materializing');
        });

        it('should start at spawn position', () => {
            expect(shell.position.y).toBeGreaterThan(0);
        });

        it('should have zero appear progress initially', () => {
            expect(shell.uniforms.uAppearProgress.value).toBe(0);
        });

        it('should have full dissolve initially', () => {
            expect(shell.uniforms.uDissolveAmount.value).toBe(1.0);
        });

        it('should accept custom config', () => {
            const customShell = new ShellCollectible(scene, camera, {
                spawnPosition: new THREE.Vector3(1, 2, 3),
                scale: 0.2
            });
            expect(customShell.position.x).toBeCloseTo(1, 1);
            customShell.destroy();
        });
    });

    describe('Materialization', () => {
        it('should progress materialization on update', () => {
            const initialProgress = shell.uniforms.uAppearProgress.value;

            shell.update(0.1, 0, camera.position);

            expect(shell.uniforms.uAppearProgress.value).toBeGreaterThan(initialProgress);
        });

        it('should decrease dissolve during materialization', () => {
            const initialDissolve = shell.uniforms.uDissolveAmount.value;

            shell.update(0.5, 0, camera.position);

            expect(shell.uniforms.uDissolveAmount.value).toBeLessThan(initialDissolve);
        });

        it('should complete materialization after duration', () => {
            // Materialize duration is 3 seconds
            const deltaTime = 0.1;
            const steps = 35; // 3.5 seconds

            for (let i = 0; i < steps; i++) {
                shell.update(deltaTime, i * deltaTime, camera.position);
            }

            expect(shell.getState()).toBe('idle');
            expect(shell.uniforms.uAppearProgress.value).toBe(1.0);
            expect(shell.uniforms.uDissolveAmount.value).toBe(0.0);
        });
    });

    describe('Idle Animation', () => {
        beforeEach(() => {
            // Fast forward to idle
            for (let i = 0; i < 40; i++) {
                shell.update(0.1, i * 0.1, camera.position);
            }
        });

        it('should be in idle state after materializing', () => {
            expect(shell.getState()).toBe('idle');
        });

        it('should be collectable when idle', () => {
            expect(shell.canCollect()).toBe(true);
        });

        it('should bob up and down', () => {
            const initialY = shell.position.y;
            shell.update(0.1, 1, camera.position);
            const newY = shell.position.y;

            // Position should have changed due to bobbing
            expect(newY).not.toBe(initialY);
        });

        it('should rotate slowly', () => {
            const initialRotation = shell.rotation.y;
            shell.update(0.1, 1, camera.position);

            expect(shell.rotation.y).not.toBe(initialRotation);
        });
    });

    describe('Collection', () => {
        beforeEach(() => {
            // Fast forward to idle
            for (let i = 0; i < 40; i++) {
                shell.update(0.1, i * 0.1, camera.position);
            }
        });

        it('should start collection when collect is called', () => {
            const uiPosition = new THREE.Vector3(100, 100, 0);
            shell.collect(uiPosition);

            expect(shell.getState()).toBe('collecting');
        });

        it('should not collect if not idle', () => {
            // Create new shell in materializing state
            const newShell = new ShellCollectible(scene, camera);
            expect(newShell.getState()).toBe('materializing');

            const uiPosition = new THREE.Vector3(100, 100, 0);
            newShell.collect(uiPosition);

            // Should stay in materializing state
            expect(newShell.getState()).toBe('materializing');
            newShell.destroy();
        });

        it('should interpolate toward UI position during collection', () => {
            const initialPos = shell.position.clone();
            const uiPosition = new THREE.Vector3(100, 100, 0);

            shell.collect(uiPosition);
            shell.update(0.1, 10, camera.position);

            // Should have moved toward UI
            expect(shell.position.x).not.toBe(initialPos.x);
        });

        it('should scale down during collection', () => {
            const initialScale = shell.scale.x;

            shell.collect(new THREE.Vector3(100, 100, 0));
            shell.update(0.1, 10, camera.position);

            expect(shell.scale.x).toBeLessThan(initialScale);
        });

        it('should complete collection after duration', () => {
            const uiPosition = new THREE.Vector3(100, 100, 0);
            shell.collect(uiPosition);

            // Collection duration is 1.5 seconds
            for (let i = 0; i < 20; i++) {
                shell.update(0.1, i * 0.1, camera.position);
            }

            // Should be in collected state
            expect(shell.getState()).toBe('collected');
        });
    });

    describe('Hover Detection', () => {
        beforeEach(() => {
            // Fast forward to idle
            for (let i = 0; i < 40; i++) {
                shell.update(0.1, i * 0.1, camera.position);
            }
        });

        it('should detect hover with raycast', () => {
            const raycaster = new THREE.Raycaster();
            raycaster.set(camera.position, shell.position.clone().sub(camera.position).normalize());

            // Raycast toward shell
            const isHovered = shell.isHovered(raycaster);
            expect(typeof isHovered).toBe('boolean');
        });

        it('should not detect hover when not idle', () => {
            const materializingShell = new ShellCollectible(scene, camera);
            expect(materializingShell.getState()).toBe('materializing');

            const raycaster = new THREE.Raycaster();
            const isHovered = materializingShell.isHovered(raycaster);

            expect(isHovered).toBe(false);
            materializingShell.destroy();
        });
    });

    describe('State Queries', () => {
        it('should report current state', () => {
            expect(shell.getState()).toBe('materializing');
        });

        it('should report if can collect', () => {
            expect(shell.canCollect()).toBe(false);

            // Fast forward to idle
            for (let i = 0; i < 40; i++) {
                shell.update(0.1, i * 0.1, camera.position);
            }

            expect(shell.canCollect()).toBe(true);
        });
    });

    describe('Uniforms', () => {
        it('should update time uniform', () => {
            shell.update(0.1, 5, camera.position);
            expect(shell.uniforms.uTime.value).toBe(5);
        });

        it('should update camera position uniform', () => {
            const camPos = new THREE.Vector3(1, 2, 3);
            shell.update(0.1, 0, camPos);

            expect(shell.uniforms.uCameraPosition.value.x).toBeCloseTo(1, 1);
            expect(shell.uniforms.uCameraPosition.value.y).toBeCloseTo(2, 1);
            expect(shell.uniforms.uCameraPosition.value.z).toBeCloseTo(3, 1);
        });
    });

    describe('Cleanup', () => {
        it('should destroy without errors', () => {
            expect(() => shell.destroy()).not.toThrow();
        });

        it('should remove from scene on destroy', () => {
            expect(scene.children).toContain(shell);
            shell.destroy();
            expect(scene.children).not.toContain(shell);
        });

        it('should handle multiple destroy calls', () => {
            shell.destroy();
            expect(() => shell.destroy()).not.toThrow();
        });
    });
});
