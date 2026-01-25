/**
 * Unit tests for GentleFeedback
 *
 * Tests camera shake for non-punitive feedback.
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Camera shake on wrong note - P0
 * - Subtle shake on premature play - P2
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import { GentleFeedback } from '../../src/entities/GentleFeedback';

describe('GentleFeedback', () => {
    let camera: THREE.Camera;
    let feedback: GentleFeedback;
    let originalPerformanceNow: () => number;

    beforeEach(() => {
        camera = new THREE.PerspectiveCamera();
        feedback = new GentleFeedback(camera);

        // Mock performance.now for deterministic tests
        originalPerformanceNow = performance.now;
        let mockTime = 0;
        performance.now = vi.fn(() => mockTime) as any;
    });

    afterEach(() => {
        performance.now = originalPerformanceNow;
        feedback.destroy();
        vi.clearAllMocks();
    });

    describe('Construction', () => {
        it('should initialize with default config', () => {
            const config = feedback.getConfig();
            expect(config.decayRate).toBe(8.0);
            expect(config.maxOffset).toBe(0.5);
            expect(config.frequencies.x).toBe(17);
            expect(config.frequencies.y).toBe(23);
            expect(config.frequencies.z).toBe(29);
        });

        it('should not be shaking initially', () => {
            expect(feedback.isActive()).toBe(false);
            expect(feedback.getIntensity()).toBe(0);
        });

        it('should accept custom config', () => {
            const customFeedback = new GentleFeedback(camera, {
                maxOffset: 1.0,
                decayRate: 5.0
            });
            const config = customFeedback.getConfig();
            expect(config.maxOffset).toBe(1.0);
            expect(config.decayRate).toBe(5.0);
            customFeedback.destroy();
        });
    });

    describe('Intensity Presets', () => {
        it('should have SUBTLE preset at 0.1', () => {
            expect(GentleFeedback.INTENSITY.SUBTLE).toBe(0.1);
        });

        it('should have PREMATURE preset at 0.3', () => {
            expect(GentleFeedback.INTENSITY.PREMATURE).toBe(0.3);
        });

        it('should have WRONG_NOTE preset at 0.5', () => {
            expect(GentleFeedback.INTENSITY.WRONG_NOTE).toBe(0.5);
        });
    });

    describe('Wrong Note Shake', () => {
        it('should start shake on wrong note', () => {
            feedback.shakeWrongNote();
            expect(feedback.isActive()).toBe(true);
            expect(feedback.getIntensity()).toBe(0.5);
        });

        it('should use WRONG_NOTE intensity preset', () => {
            feedback.shakeWrongNote();
            expect(feedback.getIntensity()).toBe(GentleFeedback.INTENSITY.WRONG_NOTE);
        });

        it('should apply shake offset to camera', () => {
            const initialPos = camera.position.clone();
            feedback.shakeWrongNote();

            // Mock time to be non-zero for deterministic shake
            (performance.now as any).mockReturnValue(100);
            feedback.update(0.016);

            // Camera position should have changed
            expect(camera.position.x).not.toBe(initialPos.x);
        });

        it('should decay intensity over time', () => {
            (performance.now as any).mockReturnValue(0);
            feedback.shakeWrongNote();
            expect(feedback.getIntensity()).toBe(0.5);

            // Advance time and update
            (performance.now as any).mockReturnValue(100);
            feedback.update(0.1);

            // Intensity should have decreased
            expect(feedback.getIntensity()).toBeLessThan(0.5);
        });

        it('should stop after duration expires', () => {
            (performance.now as any).mockReturnValue(0);
            feedback.shake(0.5, 0.5); // 500ms duration
            expect(feedback.isActive()).toBe(true);

            // Advance past duration
            (performance.now as any).mockReturnValue(600);
            feedback.update(0.1);

            expect(feedback.isActive()).toBe(false);
        });
    });

    describe('Premature Play Shake', () => {
        it('should start subtle shake for premature play', () => {
            feedback.shakePremature();
            expect(feedback.isActive()).toBe(true);
            expect(feedback.getIntensity()).toBe(0.3);
        });

        it('should use PREMATURE intensity preset', () => {
            feedback.shakePremature();
            expect(feedback.getIntensity()).toBe(GentleFeedback.INTENSITY.PREMATURE);
        });

        it('should be weaker than wrong note shake', () => {
            feedback.shakePremature();
            const prematureIntensity = feedback.getIntensity();

            feedback.stop();
            feedback.shakeWrongNote();
            const wrongNoteIntensity = feedback.getIntensity();

            expect(prematureIntensity).toBeLessThan(wrongNoteIntensity);
        });
    });

    describe('Subtle Shake', () => {
        it('should start subtle shake', () => {
            feedback.shakeSubtle();
            expect(feedback.isActive()).toBe(true);
            expect(feedback.getIntensity()).toBe(0.1);
        });

        it('should use SUBTLE intensity preset', () => {
            feedback.shakeSubtle();
            expect(feedback.getIntensity()).toBe(GentleFeedback.INTENSITY.SUBTLE);
        });
    });

    describe('Shake Offset Calculation', () => {
        it('should use sinusoidal pattern for organic motion', () => {
            feedback.shake(1.0, 1.0);

            // Mock time to get non-zero shake
            (performance.now as any).mockReturnValue(100);
            feedback.update(0.016);

            const offset = feedback.getOffset();
            // Offset should be non-zero (or very close to it)
            // At worst case, sin(0) = 0, so we check multiple components
            const hasMovement = Math.abs(offset.x) > 0 || Math.abs(offset.y) > 0 || Math.abs(offset.z) > 0;
            expect(hasMovement).toBe(true);
        });

        it('should use different frequencies per axis', () => {
            const config = feedback.getConfig();
            expect(config.frequencies.x).not.toBe(config.frequencies.y);
            expect(config.frequencies.y).not.toBe(config.frequencies.z);
        });

        it('should clamp offset to maximum', () => {
            feedback.shake(1.0, 1.0);
            feedback.update(0.016);

            const offset = feedback.getOffset();
            const config = feedback.getConfig();
            expect(offset.length()).toBeLessThanOrEqual(config.maxOffset);
        });
    });

    describe('Enable/Disable', () => {
        it('should be enabled by default', () => {
            // Can't directly check enabled, but shake should work
            feedback.shakeWrongNote();
            expect(feedback.isActive()).toBe(true);
        });

        it('should not shake when disabled', () => {
            feedback.setEnabled(false);
            feedback.shakeWrongNote();
            expect(feedback.isActive()).toBe(false);
        });

        it('should stop current shake when disabled', () => {
            feedback.shakeWrongNote();
            expect(feedback.isActive()).toBe(true);

            feedback.setEnabled(false);
            expect(feedback.isActive()).toBe(false);
        });

        it('should resume shaking when re-enabled', () => {
            feedback.setEnabled(false);
            feedback.setEnabled(true);
            feedback.shakeWrongNote();
            expect(feedback.isActive()).toBe(true);
        });
    });

    describe('Manual Stop', () => {
        it('should stop shaking immediately', () => {
            feedback.shakeWrongNote();
            expect(feedback.isActive()).toBe(true);

            feedback.stop();
            expect(feedback.isActive()).toBe(false);
            expect(feedback.getIntensity()).toBe(0);
        });

        it('should reset offset on stop', () => {
            feedback.shakeWrongNote();
            feedback.update(0.016);

            feedback.stop();
            const offset = feedback.getOffset();
            expect(offset.x).toBe(0);
            expect(offset.y).toBe(0);
            expect(offset.z).toBe(0);
        });
    });

    describe('Configuration', () => {
        it('should update config', () => {
            feedback.updateConfig({
                maxOffset: 1.0,
                decayRate: 10.0
            });

            const config = feedback.getConfig();
            expect(config.maxOffset).toBe(1.0);
            expect(config.decayRate).toBe(10.0);
        });

        it('should preserve partial config updates', () => {
            const originalDecay = feedback.getConfig().decayRate;
            feedback.updateConfig({ maxOffset: 1.0 });

            expect(feedback.getConfig().maxOffset).toBe(1.0);
            expect(feedback.getConfig().decayRate).toBe(originalDecay);
        });
    });

    describe('Clamping', () => {
        it('should clamp intensity to 0-1 range', () => {
            feedback.shake(1.5); // Above max
            expect(feedback.getIntensity()).toBe(1);

            feedback.stop();
            feedback.shake(-0.5); // Below min
            expect(feedback.getIntensity()).toBe(0);
        });
    });

    describe('Cleanup', () => {
        it('should cleanup without errors', () => {
            expect(() => feedback.destroy()).not.toThrow();
        });

        it('should be callable multiple times', () => {
            feedback.destroy();
            expect(() => feedback.destroy()).not.toThrow();
        });
    });
});
