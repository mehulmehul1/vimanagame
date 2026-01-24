import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E Test Configuration for Vimana
 *
 * This configuration supports:
 * - Desktop browsers (Chrome, Firefox, Safari)
 * - Mobile browsers (emulated)
 * - Headless CI mode
 * - Video recording on failure
 * - Screenshot capture on failure
 */
export default defineConfig({
    // Test directory
    testDir: './tests/e2e',

    // Run all tests in parallel
    fullyParallel: true,

    // Fail the build on CI if you accidentally left test.only in the source code
    forbidOnly: !!process.env.CI,

    // Retry on CI only
    retries: process.env.CI ? 2 : 0,

    // Limit workers in CI for resource management
    workers: process.env.CI ? 1 : undefined,

    // Reporter configuration
    reporter: [
        ['html', { outputFolder: 'playwright-report' }],
        ['list'],
        ['junit', { outputFile: 'test-results/junit.xml' }],
    ],

    // Shared settings for all tests
    use: {
        // Base URL for tests - uses Vite dev server
        baseURL: 'http://localhost:5173',

        // Collect trace when retrying the failed test
        trace: 'on-first-retry',

        // Screenshot on failure
        screenshot: 'only-on-failure',

        // Video on failure
        video: 'retain-on-failure',

        // Action timeout for slower WebGL initialization
        actionTimeout: 10000,

        // Navigation timeout
        navigationTimeout: 30000,
    },

    // Configure projects for different browsers
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },

        {
            name: 'firefox',
            use: { ...devices['Desktop Firefox'] },
        },

        {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
        },

        /* Mobile browsers */
        {
            name: 'Mobile Chrome',
            use: { ...devices['Pixel 5'] },
        },
        {
            name: 'Mobile Safari',
            use: { ...devices['iPhone 12'] },
        },
    ],

    // Run your local dev server before starting the tests
    webServer: {
        command: 'npm run dev',
        url: 'http://localhost:5173',
        reuseExistingServer: !process.env.CI,
        timeout: 120000,
    },
});
