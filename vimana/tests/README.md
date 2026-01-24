# Vimana Test Framework Documentation

## Overview

Vimana uses a dual testing framework optimized for web-based 3D games built with Three.js:

- **Vitest** - Fast unit testing with jsdom environment for component logic
- **Playwright** - End-to-end testing for real browser scenarios, WebGL rendering, and user interactions

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Unit Testing** | Vitest 4.x | Fast TypeScript unit tests with mocking |
| **E2E Testing** | Playwright 1.58+ | Cross-browser end-to-end tests |
| **Test Environment** | jsdom | Browser API simulation for unit tests |
| **Coverage** | c8/v8 | Built-in Vitest coverage |

## Project Structure

vimana/
├── src/                          # Source code
│   ├── entities/                 # Game entities
│   ├── audio/                    # Audio systems
│   ├── scenes/                   # Scene controllers
│   ├── ui/                       # UI components
│   ├── utils/                    # Utilities
│   └── shaders/                  # GLSL shaders
│
├── tests/                        # All test files
│   ├── README.md                 # This file
│   ├── setup.ts                  # Vitest global setup (Three.js mocks)
│   ├── entities/                 # Entity unit tests
│   ├── audio/                    # Audio system unit tests
│   ├── scenes/                   # Scene controller unit tests
│   ├── utils/                    # Utility unit tests
│   ├── integration/              # Integration tests (Vitest)
│   └── e2e/                      # End-to-end tests (Playwright)
│
├── vitest.config.ts              # Vitest configuration
├── playwright.config.ts          # Playwright configuration
└── package.json                  # Test scripts

## Running Tests Locally

### Unit Tests (Vitest)

```bash
# Run all unit tests
npm test

# Run in watch mode
npm test -- --watch

# Run with UI dashboard
npm run test:ui

# Run coverage report
npm run test:coverage

# Run specific test file
npm test -- entities/PatientJellyManager.test.ts
```

### E2E Tests (Playwright)

```bash
# Install Playwright browsers (first time only)
npx playwright install

# Run all E2E tests
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e:ui

# Run E2E tests in debug mode
npm run test:e2e:debug

# Run E2E tests in headed mode (visible browser)
npm run test:e2e:headed
```

### Run All Tests

```bash
npm run test:all
```

## Test Configuration

### Vitest Configuration (vitest.config.ts)
- Environment: jsdom (Browser API simulation)
- Globals: true (describe, it, expect available globally)
- Setup files: ./tests/setup.ts
- Path alias: @engine -> ../src

### Playwright Configuration (playwright.config.ts)
- Base URL: http://localhost:5173 (Vite dev server)
- Browsers: Chromium, Firefox, WebKit, Mobile (Pixel 5, iPhone 12)
- Auto-runs dev server before tests
- Captures screenshots/video on failure
- Generates HTML report + JUnit XML

## Test Setup (tests/setup.ts)

The global setup file provides:

- Three.js mocks (WebGLRenderer, AudioListener, etc.)
- Canvas mocks (avoids actual WebGL context creation)
- Pointer Lock API mocks
- Window properties (1920x1080 viewport)

## Writing Tests

### Unit Test Pattern (Vitest)

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { PatientJellyManager } from '../../src/entities/PatientJellyManager';

describe('PatientJellyManager', () => {
    let manager: PatientJellyManager;

    beforeEach(() => {
        manager = new PatientJellyManager(mockJelly, mockHarmony, mockFeedback);
    });

    it('should initialize with IDLE state', () => {
        expect(manager.getState()).toBe('IDLE');
    });
});
```

### E2E Test Pattern (Playwright)

```typescript
import { test, expect } from '@playwright/test';

test.describe('Harp Interaction', () => {
    test('should load the game scene', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');
        const canvas = page.locator('canvas');
        await expect(canvas).toBeVisible();
    });
});
```

## Test Categories

### Unit Tests
- Test individual classes in isolation
- Mock all external dependencies (Three.js, Howler, Rapier3d)
- Fast execution (milliseconds per test)
- Cover business logic, state machines, calculations

### Integration Tests (Vitest)
- Test multiple components working together
- Verify manager-entity interactions
- Test audio-visual feedback coordination

### E2E Tests (Playwright)
- Real browser environment with WebGL
- Actual user interactions (click, keyboard, touch)
- Cross-browser compatibility
- Performance validation (FPS, load times)

## Coverage Goals

| Category | Target | Notes |
|----------|--------|-------|
| Entities | 90%+ | Core game logic |
| Audio | 85%+ | Limited by Web Audio API |
| Utils | 95%+ | Pure functions |
| Scenes | 80%+ | Integration heavy |
| Overall | 85%+ | Business critical |

## Best Practices

### 1. Test Isolation
Use beforeEach to reset state between tests.

### 2. Descriptive Names
Test names should describe what and why:
```typescript
it('should advance to next note on correct input', () => { ... });
```

### 3. Arrange-Act-Assert Pattern
Structure tests clearly:
```typescript
it('should play harmony chord on sequence completion', () => {
    // Arrange
    jellyManager.startSequence(0);
    // Act
    jellyManager.handlePlayerInput(2);
    // Assert
    expect(mockHarmony.playCompletionChord).toHaveBeenCalled();
});
```

### 4. Mock External Dependencies
Don't test Three.js, Howler, or browser APIs.

### 5. Test Edge Cases
Cover both success and failure scenarios.

## Troubleshooting

### Vitest Issues

**Problem**: Tests fail with "WebGL not supported"
**Solution**: Tests use mocked Three.js. Check tests/setup.ts mocks.

**Problem**: Import errors for @engine alias
**Solution**: Verify vitest.config.ts has correct path alias.

### Playwright Issues

**Problem**: "Browsers not installed"
**Solution**: Run npx playwright install

**Problem**: Tests timeout waiting for page load
**Solution**: Increase navigationTimeout in playwright.config.ts

## Music Room Epic Coverage

The test suite covers all 9 stories from the Music Room Epic:

| Story | Tests |
|-------|-------|
| 1.1 Visual Foundation | VortexSystem, WaterMaterial |
| 1.2 Jelly Creatures | JellyCreature, JellyManager |
| 1.3 Gentle Feedback | GentleFeedback, GentleAudioFeedback |
| 1.4 Duet Mechanics | PatientJellyManager, DuetProgressTracker |
| 1.5 Vortex Activation | VortexActivationController, VortexLightingManager |
| 1.6 Shell Collection | ShellCollectible, ShellManager |
| 1.7 UI Overlay | ShellUIOverlay |
| 1.8 White Flash Ending | WhiteFlashEnding, WhiteFlashAudio |
| 1.9 Performance & Polish | PerformanceMonitor, QualityPresets |
