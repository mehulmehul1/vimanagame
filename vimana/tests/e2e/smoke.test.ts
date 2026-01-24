import { test, expect } from '@playwright/test';

/**
 * Smoke Tests - Verify basic game functionality
 *
 * These tests ensure the game can start, load resources, and display the initial scene.
 * They run quickly and catch fundamental issues before more complex tests.
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
});
