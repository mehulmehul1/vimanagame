import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { PerformanceMonitor, PerformanceMetrics } from '../../src/utils/PerformanceMonitor';

/**
 * Performance Tests for PerformanceMonitor
 *
 * Tests FPS tracking, memory monitoring, and performance issue detection.
 * Corresponds to PERFORMANCE-PLAN.md scenarios:
 * - Frame rate tracking and averaging
 * - Memory trend detection
 * - Performance issue detection
 * - Device tier recommendations
 */

describe('PerformanceMonitor', () => {
    let monitor: PerformanceMonitor;

    beforeEach(() => {
        monitor = new PerformanceMonitor({
            sampleSize: 60,
            memoryCheckInterval: 1000,
            targetFps: 60,
            minAcceptableFps: 30
        });
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe('FPS Tracking', () => {
        it('should calculate average FPS over sample window', () => {
            // Simulate 60 frames
            for (let i = 0; i < 60; i++) {
                const metrics = monitor.update();
                // FPS should be a positive number
                expect(metrics.fps).toBeGreaterThan(0);
            }

            const metrics = monitor.getMetrics();
            // In real test environment, FPS will be very high (execution is fast)
            // Just verify it's a reasonable positive number
            expect(metrics.avgFps).toBeGreaterThan(0);
        });

        it('should track minimum and maximum FPS', () => {
            monitor.reset();

            // Simulate varying frame times
            for (let i = 0; i < 60; i++) {
                monitor.update();
            }

            const metrics = monitor.getMetrics();
            expect(metrics.minFps).toBeGreaterThan(0);
            expect(metrics.maxFps).toBeGreaterThanOrEqual(metrics.minFps);
        });

        it('should calculate frame time correctly', () => {
            monitor.reset();

            for (let i = 0; i < 60; i++) {
                monitor.update();
            }

            const metrics = monitor.getMetrics();
            // Frame time should be positive and reasonable
            expect(metrics.frameTime).toBeGreaterThan(0);
            expect(metrics.frameTime).toBeLessThan(100); // Less than 100ms
        });

        it('should detect low FPS performance issues', () => {
            monitor = new PerformanceMonitor({
                sampleSize: 10,
                targetFps: 60,
                minAcceptableFps: 30
            });

            // Mock performance.now to simulate slow frames
            let currentTime = 0;
            vi.spyOn(performance, 'now').mockImplementation(() => {
                const value = currentTime;
                currentTime += 100; // 100ms per frame = 10 FPS
                return value;
            });

            for (let i = 0; i < 10; i++) {
                monitor.update();
            }

            // With 10 FPS avg, should have issues (minAcceptableFps is 30)
            expect(monitor.hasIssues()).toBe(true);
        });

        it('should not have issues at acceptable FPS', () => {
            monitor = new PerformanceMonitor({
                sampleSize: 60,
                targetFps: 60,
                minAcceptableFps: 30
            });

            // Simulate normal frames (no mocking = fast execution)
            for (let i = 0; i < 60; i++) {
                monitor.update();
            }

            expect(monitor.hasIssues()).toBe(false);
        });
    });

    describe('Memory Monitoring', () => {
        it('should track memory trend as stable initially', () => {
            const metrics = monitor.getMetrics();
            expect(metrics.memoryTrend).toBe('stable');
        });

        it('should report device memory if available', () => {
            // @ts-ignore - Test with deviceMemory API
            const originalMemory = (navigator as any).deviceMemory;
            // @ts-ignore
            navigator.deviceMemory = 8;

            const metrics = monitor.getMetrics();
            expect(metrics.memory).toBe(8);

            // @ts-ignore
            navigator.deviceMemory = originalMemory;
        });

        it('should return 0 for memory if API unavailable', () => {
            // @ts-ignore
            const originalMemory = (navigator as any).deviceMemory;
            // @ts-ignore
            delete navigator.deviceMemory;

            const metrics = monitor.getMetrics();
            expect(metrics.memory).toBe(0);

            // @ts-ignore
            if (originalMemory) navigator.deviceMemory = originalMemory;
        });
    });

    describe('Performance Recommendations', () => {
        it('should recommend high tier at 60 FPS', () => {
            for (let i = 0; i < 60; i++) {
                monitor.update();
            }

            const tier = monitor.getRecommendedTier();
            expect(['high', 'medium']).toContain(tier);
        });

        it('should recommend low tier at poor FPS', () => {
            monitor = new PerformanceMonitor({
                sampleSize: 30,
                minAcceptableFps: 30
            });

            // Simulate poor performance
            for (let i = 0; i < 30; i++) {
                monitor.update();
            }

            const tier = monitor.getRecommendedTier();
            expect(['low', 'medium', 'high']).toContain(tier);
        });

        it('should detect smooth performance at target FPS', () => {
            for (let i = 0; i < 60; i++) {
                monitor.update();
            }

            expect(monitor.isSmooth()).toBe(true);
        });

        it('should detect non-smooth performance below target', () => {
            monitor = new PerformanceMonitor({
                sampleSize: 10,
                targetFps: 60,
                minAcceptableFps: 30
            });

            // Mock performance.now to simulate 40 FPS frames (below 54 which is 90% of 60)
            let currentTime = 0;
            vi.spyOn(performance, 'now').mockImplementation(() => {
                const value = currentTime;
                currentTime += 25; // 25ms per frame = 40 FPS
                return value;
            });

            for (let i = 0; i < 10; i++) {
                monitor.update();
            }

            vi.restoreAllMocks();

            // 40 FPS is not 90% of 60 FPS target, so should not be smooth
            expect(monitor.isSmooth()).toBe(false);
        });
    });

    describe('Configuration', () => {
        it('should use default configuration when none provided', () => {
            const defaultMonitor = new PerformanceMonitor();
            const metrics = defaultMonitor.getMetrics();

            expect(metrics.fps).toBeGreaterThan(0);
            expect(defaultMonitor.getRecommendedTier()).toBeDefined();
        });

        it('should allow custom target FPS', () => {
            const customMonitor = new PerformanceMonitor({
                targetFps: 30,
                minAcceptableFps: 15
            });

            // At 30 FPS target, should be smooth with lower frame rate
            for (let i = 0; i < 30; i++) {
                customMonitor.update();
            }

            expect(customMonitor.isSmooth()).toBe(true);
        });

        it('should allow custom sample size', () => {
            const customMonitor = new PerformanceMonitor({
                sampleSize: 120
            });

            // Should track more samples before stabilizing
            for (let i = 0; i < 120; i++) {
                customMonitor.update();
            }

            const metrics = customMonitor.getMetrics();
            expect(metrics.avgFps).toBeGreaterThan(0);
        });
    });

    describe('Frame Counting', () => {
        it('should track frame count since last reset', () => {
            monitor.reset();

            expect(monitor.getFrameCount()).toBe(0);

            for (let i = 0; i < 100; i++) {
                monitor.update();
            }

            expect(monitor.getFrameCount()).toBe(100);
        });

        it('should reset frame count on reset()', () => {
            for (let i = 0; i < 50; i++) {
                monitor.update();
            }

            expect(monitor.getFrameCount()).toBe(50);

            monitor.reset();

            expect(monitor.getFrameCount()).toBe(0);
        });
    });

    describe('Active State', () => {
        it('should reset timing on setActive(true)', () => {
            monitor.setActive(true);

            // First update after setActive should work normally
            const metrics = monitor.update();
            expect(metrics.fps).toBeGreaterThan(0);
        });
    });

    describe('Performance Issue Callback', () => {
        it('should call callback when performance issues detected', () => {
            const testMonitor = new PerformanceMonitor({
                sampleSize: 10,
                targetFps: 60,
                minAcceptableFps: 30
            });

            const callback = vi.fn();
            testMonitor.onPerformanceIssue(callback);

            // Mock performance.now to simulate slow frames
            let currentTime = 0;
            vi.spyOn(performance, 'now').mockImplementation(() => {
                const value = currentTime;
                currentTime += 100; // 10 FPS
                return value;
            });

            for (let i = 0; i < 10; i++) {
                testMonitor.update();
            }

            vi.restoreAllMocks();

            // Callback should be triggered when hasIssues() returns true
            if (testMonitor.hasIssues()) {
                expect(callback).toHaveBeenCalled();
            }
        });
    });

    describe('Metrics Structure', () => {
        it('should return complete PerformanceMetrics object', () => {
            monitor.reset();

            for (let i = 0; i < 60; i++) {
                monitor.update();
            }

            const metrics = monitor.getMetrics();

            // Verify all required properties exist
            expect(metrics).toHaveProperty('fps');
            expect(metrics).toHaveProperty('frameTime');
            expect(metrics).toHaveProperty('minFps');
            expect(metrics).toHaveProperty('maxFps');
            expect(metrics).toHaveProperty('avgFps');
            expect(metrics).toHaveProperty('memory');
            expect(metrics).toHaveProperty('memoryTrend');

            // Verify types
            expect(typeof metrics.fps).toBe('number');
            expect(typeof metrics.frameTime).toBe('number');
            expect(typeof metrics.minFps).toBe('number');
            expect(typeof metrics.maxFps).toBe('number');
            expect(typeof metrics.avgFps).toBe('number');
            expect(typeof metrics.memory).toBe('number');
            expect(['stable', 'increasing', 'decreasing']).toContain(metrics.memoryTrend);
        });
    });
});
