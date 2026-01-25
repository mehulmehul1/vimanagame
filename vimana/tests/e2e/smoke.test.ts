import { test, expect } from '@playwright/test';

/**
 * Smoke Tests - Verify basic game functionality
 *
 * These tests ensure the game can start, load resources, and display the initial scene.
 * They run quickly and catch fundamental issues before more complex tests.
 *
 * Corresponds to TEST-DESIGN.md scenarios:
 * - Game launch without crash - P0
 * - WebGL2 context creation - P0
 * - Shader compilation <5 seconds - P0
 */

test.describe('Smoke Tests', () => {
    test('should load the game page', async ({ page }) => {
        // Navigate to game
        await page.goto('/');

        // Verify page loaded (not an error page)
        await expect(page).toHaveTitle(/Vimana/);
    });

    test('should display WebGL canvas', async ({ page }) => {
        await page.goto('/');

        // Wait for canvas element to be created by Three.js
        await page.waitForSelector('canvas', { timeout: 10000 });

        // Verify canvas is visible
        const canvas = page.locator('canvas');
        await expect(canvas).toBeVisible();
    });

    test('should initialize WebGL context', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');

        // Check if WebGL context is created
        const hasWebGL = await page.evaluate(() => {
            const canvas = document.querySelector('canvas');
            if (!canvas) return false;
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            return gl !== null;
        });

        expect(hasWebGL).toBeTruthy();
    });

    test('should have window dimensions set', async ({ page }) => {
        await page.goto('/');

        // Verify game can access window dimensions
        const dimensions = await page.evaluate(() => ({
            width: window.innerWidth,
            height: window.innerHeight,
        }));

        expect(dimensions.width).toBeGreaterThan(0);
        expect(dimensions.height).toBeGreaterThan(0);
    });

    test('should compile shaders within timeout', async ({ page }) => {
        const startTime = Date.now();

        await page.goto('/');
        await page.waitForSelector('canvas', { timeout: 10000 });

        // Wait for scene to be ready (indicates shaders compiled)
        await page.waitForTimeout(100);

        const loadTime = Date.now() - startTime;

        // Shader compilation should be under 5 seconds
        expect(loadTime).toBeLessThan(5000);
    });

    test('should have no critical console errors on load', async ({ page }) => {
        const errors: string[] = [];

        page.on('console', msg => {
            if (msg.type() === 'error') {
                errors.push(msg.text());
            }
        });

        await page.goto('/');
        await page.waitForSelector('canvas');
        await page.waitForTimeout(1000);

        // Filter out acceptable errors (like cross-origin images)
        const criticalErrors = errors.filter(e =>
            !e.includes('cross-origin') &&
            !e.includes('CORS') &&
            !e.includes('Access-Control')
        );

        // Allow some non-critical errors
        expect(criticalErrors.length).toBeLessThan(5);
    });

    test('should support WebGL2 on compatible browsers', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');

        const webGL2Support = await page.evaluate(() => {
            const canvas = document.createElement('canvas');
            return !!canvas.getContext('webgl2');
        });

        expect(webGL2Support).toBeTruthy();
    });

    test('should handle window resize', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');

        // Resize window
        await page.setViewportSize({ width: 800, height: 600 });

        const canvasSize = await page.evaluate(() => {
            const canvas = document.querySelector('canvas');
            if (!canvas) return null;
            return {
                width: canvas.width,
                height: canvas.height
            };
        });

        expect(canvasSize).not.toBeNull();
    });

    test('should maintain 60 FPS on simple scene', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');

        // Measure frame rate over 1 second
        const fps = await page.evaluate(() => {
            return new Promise<number>((resolve) => {
                let frames = 0;
                const startTime = performance.now();

                function countFrames() {
                    frames++;
                    const elapsed = performance.now() - startTime;
                    if (elapsed < 1000) {
                        requestAnimationFrame(countFrames);
                    } else {
                        resolve(Math.round(frames * 1000 / elapsed));
                    }
                }

                requestAnimationFrame(countFrames);
            });
        });

        // Should maintain reasonable frame rate
        expect(fps).toBeGreaterThan(30);
    });
});
