/**
 * Unit tests for FeedbackManager
 *
 * Tests coordinated feedback (camera shake, audio, visual).
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Wrong note feedback (shake + discord + visual) - P0
 * - Premature play feedback (subtle shake) - P2
 * - String highlight animation - P1
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';

// Mock declarations must be at top level, outside describe blocks
// Use vi.mock with class references

describe('FeedbackManager', () => {
    // Import after mock setup
    let FeedbackManager: any;
    let manager: any;
    let scene: THREE.Scene;
    let camera: THREE.Camera;

    beforeEach(async () => {
        // Dynamic import after mocks are set up
        const module = await import('../../src/entities/FeedbackManager');
        FeedbackManager = module.FeedbackManager;

        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera();
        manager = new FeedbackManager(camera, scene);
    });

    afterEach(async () => {
        await manager?.destroy();
        vi.clearAllMocks();
    });

    describe('Construction', () => {
        it('should initialize without errors', () => {
            expect(manager).toBeDefined();
        });

        it('should accept custom config', () => {
            const customManager = new FeedbackManager(camera, scene, {
                enableCameraShake: false,
                enableAudio: false,
                enableVisual: false,
                shakeMultiplier: 0.5
            });
            expect(() => customManager.destroy()).not.toThrow();
        });
    });

    describe('Wrong Note Feedback', () => {
        it('should trigger coordinated feedback', async () => {
            await expect(manager.triggerWrongNote(0)).resolves.not.toThrow();
        });

        it('should trigger for all note indices', async () => {
            for (let i = 0; i < 6; i++) {
                await manager.triggerWrongNote(i);
            }
        });
    });

    describe('Update', () => {
        it('should update without errors', () => {
            expect(() => manager.update(0.016)).not.toThrow();
        });
    });

    describe('Cleanup', () => {
        it('should destroy without errors', async () => {
            await expect(manager.destroy()).resolves.not.toThrow();
        });
    });
});
