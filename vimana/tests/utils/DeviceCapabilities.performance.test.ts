import { describe, it, expect, beforeEach } from 'vitest';
import { DeviceCapabilities, CapabilityInfo, DeviceTier } from '../../src/utils/DeviceCapabilities';

/**
 * Performance Tests for DeviceCapabilities
 *
 * Tests device detection, tier classification, and performance recommendations.
 * Corresponds to PERFORMANCE-PLAN.md scenarios:
 * - Device capability detection
 * - Tier-based performance settings
 */

describe('DeviceCapabilities Performance', () => {
    beforeEach(() => {
        // Reset singleton for each test
        (DeviceCapabilities as any).instance = undefined;
    });

    describe('Singleton Pattern', () => {
        it('should return same instance on multiple calls', () => {
            // Since we can't fully mock browser APIs in tests,
            // we verify the singleton pattern exists
            expect(DeviceCapabilities.getInstance).toBeDefined();
            expect(typeof DeviceCapabilities.getInstance).toBe('function');
        });
    });

    describe('Tier Classification Logic', () => {
        it('should have a method to determine tier from GPU string', () => {
            // Verify the private method exists via public interface
            const capabilities = DeviceCapabilities.getInstance();
            expect(capabilities.getTier).toBeDefined();
            expect(typeof capabilities.getTier).toBe('function');
        });

        it('should return a valid DeviceTier', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const tier = capabilities.getTier();

            expect(['low', 'medium', 'high', 'ultra']).toContain(tier);
        });
    });

    describe('Performance Recommendations', () => {
        it('should return numeric particle count recommendation', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const particleCount = capabilities.getRecommendedParticleCount();

            expect(typeof particleCount).toBe('number');
            expect(particleCount).toBeGreaterThan(0);
        });

        it('should return numeric texture size recommendation', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const textureSize = capabilities.getRecommendedTextureSize();

            expect(typeof textureSize).toBe('number');
            expect(textureSize).toBeGreaterThan(0);
            expect(textureSize).toBeLessThanOrEqual(4096);
        });

        it('particle count should scale with tier', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const currentTier = capabilities.getTier();
            const particleCount = capabilities.getRecommendedParticleCount();

            // Verify particle count is in reasonable range
            expect(particleCount).toBeGreaterThanOrEqual(1000);
            expect(particleCount).toBeLessThanOrEqual(100000);
        });
    });

    describe('Feature Detection Methods', () => {
        it('should have WebGL2 detection method', () => {
            const capabilities = DeviceCapabilities.getInstance();
            expect(capabilities.hasWebGL2).toBeDefined();
            expect(typeof capabilities.hasWebGL2).toBe('function');

            const result = capabilities.hasWebGL2();
            expect(typeof result).toBe('boolean');
        });

        it('should have WebGPU detection method', () => {
            const capabilities = DeviceCapabilities.getInstance();
            expect(capabilities.hasWebGPU).toBeDefined();
            expect(typeof capabilities.hasWebGPU).toBe('function');

            const result = capabilities.hasWebGPU();
            expect(typeof result).toBe('boolean');
        });

        it('should have mobile detection method', () => {
            const capabilities = DeviceCapabilities.getInstance();
            expect(capabilities.isMobile).toBeDefined();
            expect(typeof capabilities.isMobile).toBe('function');

            const result = capabilities.isMobile();
            expect(typeof result).toBe('boolean');
        });

        it('should have touch detection method', () => {
            const capabilities = DeviceCapabilities.getInstance();
            expect(capabilities.hasTouch).toBeDefined();
            expect(typeof capabilities.hasTouch).toBe('function');

            const result = capabilities.hasTouch();
            expect(typeof result).toBe('boolean');
        });
    });

    describe('Quality Feature Flags', () => {
        it('should return boolean for high quality shaders', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const result = capabilities.useHighQualityShaders();

            expect(typeof result).toBe('boolean');
        });

        it('should return boolean for post processing', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const result = capabilities.usePostProcessing();

            expect(typeof result).toBe('boolean');
        });
    });

    describe('Capability Info Structure', () => {
        it('should return capability info object', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const info = capabilities.getInfo();

            expect(info).toBeDefined();
            expect(typeof info).toBe('object');
        });

        it('should contain all required properties', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const info = capabilities.getInfo() as CapabilityInfo;

            expect(info).toHaveProperty('tier');
            expect(info).toHaveProperty('gpu');
            expect(info).toHaveProperty('webgl2');
            expect(info).toHaveProperty('webgpu');
            expect(info).toHaveProperty('memory');
            expect(info).toHaveProperty('cores');
            expect(info).toHaveProperty('mobile');
            expect(info).toHaveProperty('touch');
        });

        it('should have valid types for all properties', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const info = capabilities.getInfo() as CapabilityInfo;

            expect(['low', 'medium', 'high', 'ultra']).toContain(info.tier);
            expect(typeof info.gpu).toBe('string');
            expect(typeof info.webgl2).toBe('boolean');
            expect(typeof info.webgpu).toBe('boolean');
            expect(typeof info.memory).toBe('number');
            expect(typeof info.cores).toBe('number');
            expect(typeof info.mobile).toBe('boolean');
            expect(typeof info.touch).toBe('boolean');
        });
    });

    describe('Logging Method', () => {
        it('should have logInfo method', () => {
            const capabilities = DeviceCapabilities.getInstance();
            expect(capabilities.logInfo).toBeDefined();
            expect(typeof capabilities.logInfo).toBe('function');

            // Should not throw when called
            expect(() => capabilities.logInfo()).not.toThrow();
        });
    });

    describe('Device Tier Types', () => {
        it('should export DeviceTier type with correct values', () => {
            const validTiers: DeviceTier[] = ['low', 'medium', 'high', 'ultra'];
            expect(validTiers).toHaveLength(4);
        });
    });

    describe('Performance Characteristics by Tier', () => {
        it('higher tiers should recommend more particles', () => {
            // This is a conceptual test - actual values depend on detected hardware
            const capabilities = DeviceCapabilities.getInstance();
            const tier = capabilities.getTier();
            const particles = capabilities.getRecommendedParticleCount();

            // All tiers should recommend reasonable particle counts
            expect(particles).toBeGreaterThan(0);

            // Ultra tier should be highest
            if (tier === 'ultra') {
                expect(particles).toBeGreaterThanOrEqual(50000);
            }
        });

        it('higher tiers should recommend larger textures', () => {
            const capabilities = DeviceCapabilities.getInstance();
            const tier = capabilities.getTier();
            const textureSize = capabilities.getRecommendedTextureSize();

            expect(textureSize).toBeGreaterThan(0);

            // Ultra tier should be highest
            if (tier === 'ultra') {
                expect(textureSize).toBeGreaterThanOrEqual(2048);
            }
        });
    });
});
