import { describe, it, expect } from 'vitest';
import { QualityPresets, QUALITY_PRESETS } from '../../src/utils/QualityPresets';
import { DeviceTier } from '../../src/utils/DeviceCapabilities';

/**
 * Performance Tests for QualityPresets
 *
 * Tests quality preset configuration and application to renderer.
 * Corresponds to PERFORMANCE-PLAN.md scenarios:
 * - Quality Preset Application
 * - LOD Settings per Tier
 */

describe('QualityPresets Performance', () => {
    describe('Preset Definitions', () => {
        it('should have ultra preset with maximum quality', () => {
            const ultra = QUALITY_PRESETS.ultra;

            expect(ultra.shaderQuality).toBe(1.0);
            expect(ultra.particleMultiplier).toBe(2.0);
            expect(ultra.textureMultiplier).toBe(2.0);
            expect(ultra.enableShadows).toBe(true);
            expect(ultra.enablePostProcessing).toBe(true);
            expect(ultra.enableReflections).toBe(true);
            expect(ultra.enableVolumetric).toBe(true);
        });

        it('should have high preset with good quality', () => {
            const high = QUALITY_PRESETS.high;

            expect(high.shaderQuality).toBe(0.85);
            expect(high.particleMultiplier).toBe(1.5);
            expect(high.textureMultiplier).toBe(1.0);
            expect(high.enableShadows).toBe(true);
            expect(high.enablePostProcessing).toBe(true);
            expect(high.enableReflections).toBe(true);
        });

        it('should have medium preset with balanced quality', () => {
            const medium = QUALITY_PRESETS.medium;

            expect(medium.shaderQuality).toBe(0.7);
            expect(medium.particleMultiplier).toBe(1.0);
            expect(medium.textureMultiplier).toBe(1.0);
            expect(medium.enableShadows).toBe(true);
            expect(medium.enablePostProcessing).toBe(false);
            expect(medium.enableReflections).toBe(false);
            expect(medium.enableVolumetric).toBe(false);
        });

        it('should have low preset with minimum quality', () => {
            const low = QUALITY_PRESETS.low;

            expect(low.shaderQuality).toBe(0.5);
            expect(low.particleMultiplier).toBe(0.5);
            expect(low.textureMultiplier).toBe(0.5);
            expect(low.enableShadows).toBe(false);
            expect(low.enablePostProcessing).toBe(false);
            expect(low.enableReflections).toBe(false);
            expect(low.enableVolumetric).toBe(false);
            expect(low.antialiasSamples).toBe(0);
        });
    });

    describe('Particle Count Calculation', () => {
        it('should calculate correct particle count for each tier', () => {
            const baseCount = 2000; // VortexParticles.MAX_PARTICLES

            const ultra = QualityPresets.getSettings('ultra');
            const high = QualityPresets.getSettings('high');
            const medium = QualityPresets.getSettings('medium');
            const low = QualityPresets.getSettings('low');

            expect(ultra.particleMultiplier * baseCount).toBe(4000);
            expect(high.particleMultiplier * baseCount).toBe(3000);
            expect(medium.particleMultiplier * baseCount).toBe(2000);
            expect(low.particleMultiplier * baseCount).toBe(1000);
        });

        it('should get particle count from instance', () => {
            const presets = new QualityPresets('low');
            expect(presets.getParticleCount(2000)).toBe(1000);

            presets.setTier('high');
            expect(presets.getParticleCount(2000)).toBe(3000);
        });
    });

    describe('Texture Size Calculation', () => {
        it('should calculate correct texture size for each tier', () => {
            const baseSize = 1024;

            const ultra = QualityPresets.getSettings('ultra');
            const high = QualityPresets.getSettings('high');
            const medium = QualityPresets.getSettings('medium');
            const low = QualityPresets.getSettings('low');

            expect(ultra.textureMultiplier * baseSize).toBe(2048);
            expect(high.textureMultiplier * baseSize).toBe(1024);
            expect(medium.textureMultiplier * baseSize).toBe(1024);
            expect(low.textureMultiplier * baseSize).toBe(512);
        });

        it('should get texture size from instance', () => {
            const presets = new QualityPresets('medium');
            expect(presets.getTextureSize(1024)).toBe(1024);

            presets.setTier('low');
            expect(presets.getTextureSize(1024)).toBe(512);
        });
    });

    describe('LOD Distance Multiplier', () => {
        it('should have correct LOD distance for each tier', () => {
            const ultra = QUALITY_PRESETS.ultra;
            const high = QUALITY_PRESETS.high;
            const medium = QUALITY_PRESETS.medium;
            const low = QUALITY_PRESETS.low;

            expect(ultra.lodDistanceMultiplier).toBe(2.0);
            expect(high.lodDistanceMultiplier).toBe(1.5);
            expect(medium.lodDistanceMultiplier).toBe(1.0);
            expect(low.lodDistanceMultiplier).toBe(0.7);
        });
    });

    describe('Feature Flags', () => {
        it('should disable expensive features on low tier', () => {
            const low = QUALITY_PRESETS.low;

            expect(low.enableShadows).toBe(false);
            expect(low.enablePostProcessing).toBe(false);
            expect(low.enableReflections).toBe(false);
            expect(low.enableVolumetric).toBe(false);
            expect(low.antialiasSamples).toBe(0);
        });

        it('should enable all features on ultra tier', () => {
            const ultra = QUALITY_PRESETS.ultra;

            expect(ultra.enableShadows).toBe(true);
            expect(ultra.enablePostProcessing).toBe(true);
            expect(ultra.enableReflections).toBe(true);
            expect(ultra.enableVolumetric).toBe(true);
            expect(ultra.antialiasSamples).toBe(8);
        });
    });

    describe('QualityPresets Class', () => {
        it('should create instance with default tier', () => {
            const presets = new QualityPresets();
            expect(presets.getTier()).toBe('medium');
        });

        it('should create instance with specified tier', () => {
            const presets = new QualityPresets('high');
            expect(presets.getTier()).toBe('high');
        });

        it('should change tier dynamically', () => {
            const presets = new QualityPresets('low');
            expect(presets.getTier()).toBe('low');
            expect(presets.shadowsEnabled()).toBe(false);

            presets.setTier('high');
            expect(presets.getTier()).toBe('high');
            expect(presets.shadowsEnabled()).toBe(true);
        });

        it('should update individual settings', () => {
            const presets = new QualityPresets('medium');

            expect(presets.getSettings().particleMultiplier).toBe(1.0);

            presets.updateSetting('particleMultiplier', 0.75);

            expect(presets.getSettings().particleMultiplier).toBe(0.75);
        });

        it('should check if shadows are enabled', () => {
            const low = new QualityPresets('low');
            const high = new QualityPresets('high');

            expect(low.shadowsEnabled()).toBe(false);
            expect(high.shadowsEnabled()).toBe(true);
        });

        it('should check if post-processing is enabled', () => {
            const medium = new QualityPresets('medium');
            const ultra = new QualityPresets('ultra');

            expect(medium.postProcessingEnabled()).toBe(false);
            expect(ultra.postProcessingEnabled()).toBe(true);
        });

        it('should get shader quality value', () => {
            const low = new QualityPresets('low');
            const ultra = new QualityPresets('ultra');

            expect(low.getShaderQuality()).toBe(0.5);
            expect(ultra.getShaderQuality()).toBe(1.0);
        });
    });

    describe('Custom Presets', () => {
        it('should create custom preset from base', () => {
            const custom = QualityPresets.createCustom('medium', {
                particleMultiplier: 0.5,
                enableShadows: false
            });

            // Should inherit medium defaults
            expect(custom.shaderQuality).toBe(0.7);
            expect(custom.textureMultiplier).toBe(1.0);

            // Should apply overrides
            expect(custom.particleMultiplier).toBe(0.5);
            expect(custom.enableShadows).toBe(false);
        });
    });

    describe('Performance Impact', () => {
        it('should reduce render load on low tier', () => {
            const low = QUALITY_PRESETS.low;
            const ultra = QUALITY_PRESETS.ultra;

            // Low tier should have significantly fewer particles
            const particleRatio = low.particleMultiplier / ultra.particleMultiplier;
            expect(particleRatio).toBeLessThan(0.3); // < 30% of ultra

            // Low tier should have smaller textures
            const textureRatio = low.textureMultiplier / ultra.textureMultiplier;
            expect(textureRatio).toBeLessThan(0.3); // < 30% of ultra

            // Low tier should disable expensive features
            expect(low.enableShadows).toBe(false);
            expect(ultra.enableShadows).toBe(true);
        });

        it('should scale max lights appropriately', () => {
            const low = QUALITY_PRESETS.low;
            const medium = QUALITY_PRESETS.medium;
            const high = QUALITY_PRESETS.high;
            const ultra = QUALITY_PRESETS.ultra;

            expect(low.maxLights).toBe(2);
            expect(medium.maxLights).toBe(4);
            expect(high.maxLights).toBe(8);
            expect(ultra.maxLights).toBe(16);
        });
    });
});
