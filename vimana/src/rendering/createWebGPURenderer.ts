/**
 * createWebGPURenderer.ts - WebGPU Renderer Initialization with Visionary Support
 * =============================================================================
 *
 * Creates WebGPU renderer with feature detection, async compilation, and
 * graceful WebGL2 fallback. Includes Visionary platform detection for
 * Gaussian splat support.
 *
 * Platform Support for Visionary:
 * - Windows 10/11: ✅ RECOMMENDED (discrete GPU)
 * - Ubuntu/Linux: ❌ NOT SUPPORTED (fp16 WebGPU bug)
 * - macOS: ⚠️ LIMITED (M4 Max+ recommended)
 * - Mobile: ⚠️ LIMITED (performance varies)
 *
 * ==============================================================================
 */

import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';

const logger = {
  log: (msg: string, ...args: unknown[]) => console.log(`[WebGPURenderer] ${msg}`, ...args),
  warn: (msg: string, ...args: unknown[]) => console.warn(`[WebGPURenderer] ${msg}`, ...args),
  error: (msg: string, ...args: unknown[]) => console.error(`[WebGPURenderer] ${msg}`, ...args),
};

/**
 * Platform detection result
 */
export interface PlatformInfo {
  isMobile: boolean;
  isUbuntu: boolean;
  isMacOS: boolean;
  isWindows: boolean;
  platform: string;
  userAgent: string;
  visionarySupported: boolean;
  visionaryWarning?: string;
}

/**
 * Result from createWebGPURenderer
 */
export interface RendererResult {
  renderer: THREE.WebGLRenderer | THREE.WebGPURenderer;
  type: 'WebGPU' | 'WebGL2';
  visionarySupported: boolean;
  platform: PlatformInfo;
  compileAsync?: (scene: THREE.Scene) => Promise<void>;
}

/**
 * Options for createWebGPURenderer
 */
export interface WebGPURendererOptions {
  alpha?: boolean;
  antialias?: boolean;
  pixelRatioLimitMobile?: number;
  pixelRatioLimitDesktop?: number;
  canvas?: HTMLCanvasElement | null;
}

/**
 * Default options
 */
const DEFAULT_OPTIONS: WebGPURendererOptions = {
  alpha: true,
  antialias: false,
  pixelRatioLimitMobile: 1.5,
  pixelRatioLimitDesktop: 2.0,
  canvas: null,
};

/**
 * Detect platform and Visionary support
 */
export function detectPlatform(): PlatformInfo {
  const userAgent = navigator.userAgent;
  const platform = navigator.platform;

  // Mobile detection
  const isMobile = /iPhone|iPad|iPod|Android/i.test(userAgent);

  // Ubuntu/Linux detection
  const isUbuntu = /Ubuntu/i.test(userAgent) || /Linux/i.test(platform);

  // macOS detection
  const isMacOS = /Mac|iPod|iPhone|iPad/i.test(platform);

  // Windows detection
  const isWindows = /Win/i.test(platform);

  let visionarySupported = true;
  let visionaryWarning: string | undefined;

  // Visionary platform limitations
  if (isUbuntu) {
    visionarySupported = false;
    visionaryWarning = 'Visionary is NOT supported on Ubuntu/Linux due to fp16 WebGPU bug. Gaussian splats will be disabled.';
  } else if (isMacOS) {
    // macOS is supported but performance is limited
    visionaryWarning = 'Visionary performance on macOS may be limited. M4 Max+ chip recommended for optimal performance.';
  } else if (isMobile) {
    visionaryWarning = 'Visionary performance on mobile may vary. Not all devices are supported.';
  }

  return {
    isMobile,
    isUbuntu,
    isMacOS,
    isWindows,
    platform,
    userAgent,
    visionarySupported,
    visionaryWarning,
  };
}

/**
 * Log platform information
 */
export function logPlatformInfo(platform: PlatformInfo): void {
  logger.log('🖥️ Platform Detection:');
  logger.log(`  Platform: ${platform.platform}`);
  logger.log(`  Mobile: ${platform.isMobile}`);
  logger.log(`  Ubuntu: ${platform.isUbuntu}`);
  logger.log(`  macOS: ${platform.isMacOS}`);
  logger.log(`  Windows: ${platform.isWindows}`);
  logger.log(`  Visionary Support: ${platform.visionarySupported ? '✅' : '❌'}`);

  if (platform.visionaryWarning) {
    logger.warn(`⚠️ ${platform.visionaryWarning}`);
  }
}

/**
 * Calculate pixel ratio with platform-specific limits
 */
export function calculatePixelRatio(
  platform: PlatformInfo,
  mobileLimit: number,
  desktopLimit: number
): number {
  const cap = platform.isMobile ? mobileLimit : desktopLimit;
  return Math.min(window.devicePixelRatio, cap);
}

/**
 * Check if WebGPU is available
 */
export async function isWebGPUSupported(): Promise<boolean> {
  if (!navigator.gpu) {
    logger.warn('WebGPU not supported: navigator.gpu is undefined');
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      logger.warn('WebGPU adapter request failed');
      return false;
    }
    return true;
  } catch (e) {
    if (e instanceof Error) {
      logger.warn(`WebGPU check failed: ${e.message}`);
    }
    return false;
  }
}

/**
 * Configure renderer properties (color space, tone mapping)
 */
function configureRendererProperties(
  renderer: THREE.WebGLRenderer | THREE.WebGPURenderer
): void {
  // Color management - required for correct color reproduction
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // ACES filmic tone mapping for cinematic look
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.5;

  // Shadow settings (WebGL2 only)
  if (renderer instanceof THREE.WebGLRenderer) {
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  }
}

/**
 * Show loading indicator during async operations
 */
export function showLoadingIndicator(message: string = 'Initializing...'): void {
  const existing = document.getElementById('loading-indicator');
  if (existing) {
    const textElement = existing.querySelector('.loading-text');
    if (textElement) {
      textElement.textContent = message;
    }
    return;
  }

  const indicator = document.createElement('div');
  indicator.id = 'loading-indicator';
  indicator.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0, 0, 0, 0.85);
    color: white;
    padding: 24px 48px;
    border-radius: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 16px;
    z-index: 10000;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  `;

  const textElement = document.createElement('div');
  textElement.className = 'loading-text';
  textElement.textContent = message;
  indicator.appendChild(textElement);

  const spinner = document.createElement('div');
  spinner.style.cssText = `
    margin: 16px auto 0;
    width: 32px;
    height: 32px;
    border: 3px solid rgba(255, 255, 255, 0.2);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  `;
  indicator.appendChild(spinner);

  // Add spinner keyframes if not already present
  if (!document.getElementById('spinner-keyframes')) {
    const style = document.createElement('style');
    style.id = 'spinner-keyframes';
    style.textContent = `
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
  }

  document.body.appendChild(indicator);
}

/**
 * Update loading indicator message
 */
export function updateLoadingIndicator(message: string): void {
  const indicator = document.getElementById('loading-indicator');
  if (indicator) {
    const textElement = indicator.querySelector('.loading-text');
    if (textElement) {
      textElement.textContent = message;
    }
  }
}

/**
 * Hide loading indicator
 */
export function hideLoadingIndicator(): void {
  const indicator = document.getElementById('loading-indicator');
  if (indicator) {
    indicator.remove();
  }
}

/**
 * Create WebGPU renderer with WebGL2 fallback
 *
 * @param options - Renderer configuration options
 * @returns Promise<RendererResult> - Result with renderer and capabilities
 */
export async function createWebGPURenderer(
  options: WebGPURendererOptions = {}
): Promise<RendererResult> {
  const mergedOptions = { ...DEFAULT_OPTIONS, ...options };
  const platform = detectPlatform();

  // Log platform information
  logPlatformInfo(platform);

  // Calculate pixel ratio
  const pixelRatio = calculatePixelRatio(
    platform,
    mergedOptions.pixelRatioLimitMobile!,
    mergedOptions.pixelRatioLimitDesktop!
  );
  logger.log(`📱 Pixel Ratio: ${window.devicePixelRatio} → capped at ${pixelRatio}`);

  // Check WebGPU support
  const supportsWebGPU = await isWebGPUSupported();

  if (supportsWebGPU) {
    try {
      logger.log('✨ Creating WebGPU renderer...');

      const renderer = new WebGPURenderer({
        alpha: mergedOptions.alpha,
        antialias: mergedOptions.antialias,
      });

      // WebGPURenderer requires async initialization
      await renderer.init();

      // Configure renderer properties
      configureRendererProperties(renderer);

      // Set pixel ratio
      renderer.setPixelRatio(pixelRatio);

      // Set initial size
      const getViewportSize = () => {
        if (window.visualViewport) {
          return { width: window.visualViewport.width, height: window.visualViewport.height };
        }
        return { width: window.innerWidth, height: window.innerHeight };
      };

      const initialSize = getViewportSize();
      renderer.setSize(initialSize.width, initialSize.height);

      logger.log('✅ WebGPU renderer created successfully');
      window.rendererType = 'WebGPU';

      // Return result with async compile function
      return {
        renderer,
        type: 'WebGPU',
        visionarySupported: platform.visionarySupported,
        platform,
        compileAsync: async (scene: THREE.Scene) => {
          showLoadingIndicator('Compiling shaders...');
          try {
            await renderer.compileAsync(scene);
            logger.log('✅ Shaders compiled successfully');
          } catch (e) {
            logger.error(`Shader compilation failed: ${e instanceof Error ? e.message : String(e)}`);
            throw e;
          } finally {
            hideLoadingIndicator();
          }
        },
      };
    } catch (e) {
      logger.error(`WebGPU renderer creation failed: ${e instanceof Error ? e.message : String(e)}`);
      logger.warn('Falling back to WebGL2...');
    }
  }

  // Fallback to WebGL2
  logger.log('📱 Creating WebGL2 renderer (fallback mode)...');

  const renderer = new THREE.WebGLRenderer({
    alpha: mergedOptions.alpha,
    antialias: mergedOptions.antialias,
  });

  // Configure renderer properties
  configureRendererProperties(renderer);

  // Set pixel ratio
  renderer.setPixelRatio(pixelRatio);

  // Set initial size
  const getViewportSize = () => {
    if (window.visualViewport) {
      return { width: window.visualViewport.width, height: window.visualViewport.height };
    }
    return { width: window.innerWidth, height: window.innerHeight };
  };

  const initialSize = getViewportSize();
  renderer.setSize(initialSize.width, initialSize.height);

  logger.log('✅ WebGL2 renderer created');
  window.rendererType = 'WebGL2';

  // Note: Visionary not supported in WebGL2 fallback
  // Visionary requires WebGPU for depth occlusion
  if (platform.visionarySupported) {
    logger.warn('⚠️ Visionary Gaussian splats are disabled in WebGL2 fallback mode');
  }

  return {
    renderer,
    type: 'WebGL2',
    visionarySupported: false, // Visionary requires WebGPU
    platform,
    // WebGL2 doesn't have compileAsync - provide no-op
    compileAsync: async (_scene: THREE.Scene) => {
      // No-op for WebGL2
    },
  };
}

/**
 * Get info about the current renderer for debugging
 */
export function getRendererInfo(renderer: THREE.WebGLRenderer | THREE.WebGPURenderer): {
  type: string;
  isWebGPU: boolean;
  hasCompute: boolean;
  hasTSL: boolean;
} {
  const isWebGPU = renderer instanceof WebGPURenderer;

  return {
    type: isWebGPU ? 'WebGPU' : 'WebGL2',
    isWebGPU,
    hasCompute: isWebGPU,
    hasTSL: isWebGPU,
  };
}

/**
 * Display renderer info in console
 */
export function logRendererInfo(renderer: THREE.WebGLRenderer | THREE.WebGPURenderer): void {
  const info = getRendererInfo(renderer);

  logger.log('📊 Renderer Info:', {
    Type: info.type,
    'Compute Shaders': info.hasCompute ? '✅' : '❌',
    'TSL Support': info.hasTSL ? '✅' : '❌',
  });
}

/**
 * Set up animation loop with proper time handling
 *
 * Uses Three.js setAnimationLoop which handles:
 * - XR sessions
 * - Visibility changes
 * - Frame timing
 *
 * @param renderer - The renderer to use
 * @param callback - Frame update callback
 */
export function setupAnimationLoop(
  renderer: THREE.WebGLRenderer | THREE.WebGPURenderer,
  callback: (time: number, delta: number) => void
): void {
  let lastTime = 0;

  renderer.setAnimationLoop((time: number) => {
    const t = time * 0.001; // Convert to seconds
    const delta = Math.min(0.1, t - lastTime); // Cap delta to prevent huge jumps
    lastTime = t;

    callback(t, delta);
  });
}

// Export all utilities
export {
  showLoadingIndicator as showLoading,
  hideLoadingIndicator as hideLoading,
  updateLoadingIndicator as updateLoading,
};
