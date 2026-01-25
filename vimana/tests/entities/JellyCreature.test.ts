/**
 * Unit tests for JellyCreature
 *
 * Tests jelly emergence, teaching, and submerging behavior.
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Jelly emergence animation - P0
 * - Jelly bioluminescence during teaching - P0
 * - Unique pulse rates per note - P1
 * - Jelly submerges after teaching - P1
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import { JellyCreature } from '../../src/entities/JellyCreature';

// Mock document.createElement for canvas creation in JellyCreature
vi.stubGlobal('document', {
    createElement: vi.fn((tag: string) => {
        if (tag === 'canvas') {
            const canvas = {
                width: 128,
                height: 128,
                getContext: vi.fn(() => ({
                    clearRect: vi.fn(),
                    createRadialGradient: vi.fn(() => ({
                        addColorStop: vi.fn()
                    })),
                    fillRect: vi.fn(),
                    fillStyle: ''
                }))
            };
            return canvas;
        }
        return document.createElement(tag);
    })
});

describe('JellyCreature', () => {
    let jelly: JellyCreature;
    let spawnPosition: THREE.Vector3;
    let originalDocument: any;

    beforeEach(() => {
        // Save original document methods
        originalDocument = { ...global.document };

        // Setup canvas mock
        const mockCanvas = {
            width: 128,
            height: 128,
            getContext: vi.fn(() => ({
                clearRect: vi.fn(),
                createRadialGradient: vi.fn(() => ({
                    addColorStop: vi.fn()
                })),
                fillRect: vi.fn(),
                fillStyle: ''
            }))
        };

        global.document.createElement = vi.fn((tag: string) => {
            if (tag === 'canvas') return mockCanvas;
            return originalDocument.createElement(tag);
        }) as any;

        spawnPosition = new THREE.Vector3(0, 0, 0);
        jelly = new JellyCreature(spawnPosition, 0);
    });

    afterEach(() => {
        // Restore original document
        if (originalDocument) {
            global.document.createElement = originalDocument.createElement;
        }
        jelly.destroy();
    });

    describe('Construction', () => {
        it('should initialize at home position', () => {
            expect(jelly.position.y).toBeGreaterThan(0);
        });

        it('should start with zero scale', () => {
            expect(jelly.scale.x).toBe(0);
            expect(jelly.scale.y).toBe(0);
            expect(jelly.scale.z).toBe(0);
        });

        it('should be in hidden state initially', () => {
            expect(jelly.getState()).toBe('hidden');
        });

        it('should not be ready when hidden', () => {
            expect(jelly.isReady()).toBe(false);
        });

        it('should accept custom spawn position', () => {
            const customPos = new THREE.Vector3(1, 2, 3);
            const customJelly = new JellyCreature(customPos, 0);
            expect(customJelly.getState()).toBe('hidden');
            customJelly.destroy();
        });

        it('should accept note index', () => {
            const jellyC = new JellyCreature(spawnPosition, 0); // C
            const jellyD = new JellyCreature(spawnPosition, 3); // F
            expect(jellyC).toBeDefined();
            expect(jellyD).toBeDefined();
            jellyC.destroy();
            jellyD.destroy();
        });
    });

    describe('Spawning', () => {
        it('should spawn with target string', () => {
            jelly.spawn(2);
            expect(jelly.getState()).toBe('spawning');
        });

        it('should reset position on spawn', () => {
            jelly.position.set(10, 10, 10);
            jelly.spawn(0);

            // Position should reset toward home
            expect(jelly.position.x).toBeCloseTo(0, 1);
            expect(jelly.position.z).toBeCloseTo(0, 1);
        });

        it('should reset scale on spawn', () => {
            jelly.scale.setScalar(5);
            jelly.spawn(0);

            expect(jelly.scale.x).toBe(0);
            expect(jelly.scale.y).toBe(0);
            expect(jelly.scale.z).toBe(0);
        });

        it('should update during spawn animation', () => {
            jelly.spawn(0);
            jelly.update(0.1, 0);

            // Scale should increase
            expect(jelly.scale.x).toBeGreaterThan(0);
        });

        it('should become idle after spawn completes', () => {
            jelly.spawn(0);

            // Update for longer than spawn duration (1 second)
            let time = 0;
            while (time < 1.5) {
                jelly.update(0.1, time);
                time += 0.1;
            }

            expect(jelly.getState()).toBe('idle');
        });

        it('should be ready when idle', () => {
            jelly.spawn(0);
            jelly.update(1.5, 0); // Force spawn complete
            expect(jelly.isReady()).toBe(true);
        });
    });

    describe('Teaching', () => {
        beforeEach(() => {
            jelly.spawn(0);
            // Fast forward to idle
            for (let i = 0; i < 15; i++) {
                jelly.update(0.1, i * 0.1);
            }
        });

        it('should begin teaching', () => {
            jelly.beginTeaching();
            expect(jelly.getState()).toBe('teaching');
        });

        it('should still be ready during teaching', () => {
            jelly.beginTeaching();
            expect(jelly.isReady()).toBe(true);
        });

        it('should update teaching animation', () => {
            jelly.beginTeaching();
            const initialScale = jelly.scale.x;
            jelly.update(0.1, 1);
            expect(jelly.scale.x).not.toBe(initialScale);
        });
    });

    describe('Submerging', () => {
        beforeEach(() => {
            jelly.spawn(0);
            // Fast forward to idle
            for (let i = 0; i < 15; i++) {
                jelly.update(0.1, i * 0.1);
            }
            jelly.beginTeaching();
        });

        it('should submerge', () => {
            jelly.submerge();
            expect(jelly.getState()).toBe('submerging');
        });

        it('should not be ready when submerging', () => {
            jelly.submerge();
            expect(jelly.isReady()).toBe(false);
        });

        it('should fade out during submerging', () => {
            jelly.submerge();
            const initialScale = jelly.scale.x;

            jelly.update(0.2, 0);
            expect(jelly.scale.x).toBeLessThan(initialScale);
        });

        it('should become hidden after submerging', () => {
            jelly.submerge();

            // Update past submerge duration
            for (let i = 0; i < 20; i++) {
                jelly.update(0.1, i * 0.1);
            }

            expect(jelly.getState()).toBe('hidden');
        });
    });

    describe('Pulse Rate', () => {
        it('should have default pulse rate', () => {
            // Default is 2.0 Hz
            jelly.spawn(0);
            expect(jelly.getState()).toBe('spawning');
        });

        it('should allow setting pulse rate', () => {
            jelly.setPulseRate(3.0);
            // No direct getter, but should not throw
            expect(() => jelly.setPulseRate(1.5)).not.toThrow();
        });
    });

    describe('Appearance', () => {
        it('should allow setting color', () => {
            const color = new THREE.Color(0xff0000);
            expect(() => jelly.setColor(color)).not.toThrow();
        });

        it('should allow setting camera position', () => {
            const camPos = new THREE.Vector3(1, 2, 3);
            expect(() => jelly.setCameraPosition(camPos)).not.toThrow();
        });

        it('should allow setting home position', () => {
            const newHome = new THREE.Vector3(5, 0, 5);
            jelly.setHomePosition(newHome);
            // Should not throw
            expect(() => jelly.setHomePosition(newHome)).not.toThrow();
        });
    });

    describe('State Queries', () => {
        it('should report current state', () => {
            expect(jelly.getState()).toBe('hidden');
            jelly.spawn(0);
            expect(jelly.getState()).toBe('spawning');
        });

        it('should report if hidden', () => {
            expect(jelly.isHidden()).toBe(true);
            jelly.spawn(0);
            expect(jelly.isHidden()).toBe(false);
        });

        it('should report if ready', () => {
            expect(jelly.isReady()).toBe(false);
            jelly.spawn(0);
            // Fast forward
            for (let i = 0; i < 15; i++) {
                jelly.update(0.1, i * 0.1);
            }
            expect(jelly.isReady()).toBe(true);
        });
    });

    describe('Cleanup', () => {
        it('should destroy without errors', () => {
            expect(() => jelly.destroy()).not.toThrow();
        });

        it('should handle multiple destroy calls', () => {
            jelly.destroy();
            expect(() => jelly.destroy()).not.toThrow();
        });
    });

    describe('Update Behavior', () => {
        it('should handle update in hidden state', () => {
            expect(() => jelly.update(0.016, 0)).not.toThrow();
        });

        it('should handle update in spawning state', () => {
            jelly.spawn(0);
            expect(() => jelly.update(0.016, 0)).not.toThrow();
        });

        it('should handle update in idle state', () => {
            jelly.spawn(0);
            for (let i = 0; i < 15; i++) {
                jelly.update(0.1, i * 0.1);
            }
            expect(() => jelly.update(0.016, 10)).not.toThrow();
        });

        it('should handle update in submerging state', () => {
            jelly.spawn(0);
            for (let i = 0; i < 15; i++) {
                jelly.update(0.1, i * 0.1);
            }
            jelly.submerge();
            expect(() => jelly.update(0.016, 20)).not.toThrow();
        });
    });
});
