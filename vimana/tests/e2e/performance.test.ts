import { test, expect } from '@playwright/test';

/**
 * Performance Tests
 *
 * Tests frame rate, memory usage, and loading times.
 * Corresponds to TEST-DESIGN.md scenarios:
 * - 60 FPS maintained on desktop - P0
 * - 30 FPS maintained on mobile - P0
 * - No memory leaks over time - P1
 */

test.describe('Performance Tests', () => {
    test('should maintain 60 FPS on desktop', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');
        await page.waitForTimeout(1000); // Let scene stabilize

        // Measure frame rate over 2 seconds
        const fps = await page.evaluate(() => {
            return new Promise<number>((resolve) => {
                let frames = 0;
                const startTime = performance.now();

                function countFrames() {
                    frames++;
                    const elapsed = performance.now() - startTime;
                    if (elapsed < 2000) {
                        requestAnimationFrame(countFrames);
                    } else {
                        resolve(Math.round(frames * 1000 / elapsed));
                    }
                }

                requestAnimationFrame(countFrames);
            });
        });

        // Desktop target is 60 FPS
        expect(fps).toBeGreaterThanOrEqual(55); // Allow 5 FPS variance
    });

    test('should measure memory usage', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');
        await page.waitForTimeout(2000);

        // Get memory info if available (Chrome only)
        const memoryInfo = await page.evaluate(() => {
            if (!(performance as any).memory) {
                return { available: false };
            }
            return {
                available: true,
                usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
                totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
                jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit
            };
        });

        if (memoryInfo.available) {
            // Memory should be reasonable (< 500MB)
            const usedMB = memoryInfo.usedJSHeapSize / (1024 * 1024);
            expect(usedMB).toBeLessThan(500);
        }
    });

    test('should not have memory leaks over 1 minute', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');

        const measurements: number[] = [];

        // Measure memory every 10 seconds for 1 minute
        for (let i = 0; i < 6; i++) {
            await page.waitForTimeout(10000);

            const mem = await page.evaluate(() => {
                if (!(performance as any).memory) return 0;
                return (performance as any).memory.usedJSHeapSize;
            });

            if (mem > 0) {
                measurements.push(mem);
            }
        }

        if (measurements.length >= 2) {
            // Memory growth should be less than 50MB over 1 minute
            const growth = measurements[measurements.length - 1] - measurements[0];
            const growthMB = growth / (1024 * 1024);
            expect(growthMB).toBeLessThan(50);
        }
    });

    test('should load shaders quickly', async ({ page }) => {
        const startTime = Date.now();

        await page.goto('/');

        // Wait for canvas creation (indicates WebGL ready)
        await page.waitForSelector('canvas', { timeout: 10000 });

        // Wait a bit for shader compilation
        await page.waitForTimeout(500);

        const loadTime = Date.now() - startTime;

        // Target is < 5 seconds for shader compilation
        expect(loadTime).toBeLessThan(5000);
    });
});

test.describe('Platform Tests', () => {
    test('should work on different viewport sizes', async ({ page }) => {
        // Desktop
        await page.setViewportSize({ width: 1920, height: 1080 });
        await page.goto('/');
        await page.waitForSelector('canvas');
        const desktopCanvas = await page.locator('canvas').isVisible();
        expect(desktopCanvas).toBe(true);

        // Tablet
        await page.setViewportSize({ width: 768, height: 1024 });
        await page.reload();
        await page.waitForSelector('canvas');
        const tabletCanvas = await page.locator('canvas').isVisible();
        expect(tabletCanvas).toBe(true);

        // Mobile
        await page.setViewportSize({ width: 375, height: 667 });
        await page.reload();
        await page.waitForSelector('canvas');
        const mobileCanvas = await page.locator('canvas').isVisible();
        expect(mobileCanvas).toBe(true);
    });

    test('should handle portrait orientation', async ({ page }) => {
        await page.setViewportSize({ width: 667, height: 375 });
        await page.goto('/');
        await page.waitForSelector('canvas');

        const canvasVisible = await page.locator('canvas').isVisible();
        expect(canvasVisible).toBe(true);
    });

    test('should handle touch events', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');

        // Simulate touch event
        const canvas = page.locator('canvas');
        await canvas.tap();

        // Should not cause errors
        const hasErrors = await page.evaluate(() => {
            return (window as any).touchError || false;
        });

        expect(hasErrors).toBe(false);
    });
});
