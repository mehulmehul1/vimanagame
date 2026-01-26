/**
 * renderer.js - WebGPU Detection and Renderer Selection
 * =============================================================================
 *
 * Handles WebGPU feature detection and provides appropriate renderer:
 * - WebGPURenderer (preferred) for modern browsers
 * - WebGLRenderer fallback for unsupported browsers
 *
 * Phase 1 of WebGPU migration: Switch renderer while keeping all materials
 * working with existing GLSL shaders.
 *
 * ==============================================================================
 */

import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';

const logger = {
    log: (msg, ...args) => console.log(`[Renderer] ${msg}`, ...args),
    warn: (msg, ...args) => console.warn(`[Renderer] ${msg}`, ...args),
    error: (msg, ...args) => console.error(`[Renderer] ${msg}`, ...args),
};

/**
 * Check if WebGPU is available in this browser
 */
export async function isWebGPUSupported() {
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
        // Adapter doesn't have close() - just check existence
        // We'll request a new adapter+device when creating renderer
        return true;
    } catch (e) {
        logger.warn(`WebGPU check failed: ${e.message}`);
        return false;
    }
}

/**
 * Get the best available renderer for this browser
 *
 * @param {Object} options - Renderer options
 * @param {HTMLElement} canvas - Optional existing canvas element
 * @param {Object} constraints - Optional constraints that force specific renderer
 * @param {boolean} constraints.requiresWebGL - Force WebGL2 (e.g., for SparkRenderer)
 * @returns {Promise<THREE.WebGLRenderer | WebGPURenderer>}
 */
export async function createOptimalRenderer(options = {}, canvas = null, constraints = {}) {
    const { requiresWebGL = false } = constraints;
    const supportsWebGPU = !requiresWebGL && await isWebGPUSupported();

    const rendererOptions = {
        alpha: true,
        antialias: false,
        ...options,
    };

    if (supportsWebGPU) {
        try {
            logger.log('✨ Creating WebGPU renderer...');

            // WebGPURenderer requires async initialization
            const renderer = new WebGPURenderer(rendererOptions);

            // WebGPURenderer needs to be initialized before use
            await renderer.init();

            // Set standard properties (WebGPURenderer supports these)
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 0.5;
            renderer.outputColorSpace = THREE.SRGBColorSpace;

            logger.log('✅ WebGPU renderer created successfully');
            window.rendererType = 'WebGPU';

            return renderer;
        } catch (e) {
            logger.error(`WebGPU renderer creation failed: ${e.message}`);
            logger.warn('Falling back to WebGL2...');
        }
    }

    // Fallback to WebGL2 (or forced WebGL2 for SparkRenderer compatibility)
    if (requiresWebGL) {
        logger.log('📱 Creating WebGL2 renderer (required by SparkRenderer)...');
    } else {
        logger.log('📱 Creating WebGL2 renderer (fallback mode)...');
    }

    const renderer = new THREE.WebGLRenderer(rendererOptions);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.5;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    logger.log('✅ WebGL2 renderer created');
    window.rendererType = 'WebGL2';

    return renderer;
}

/**
 * Get info about the current renderer for debugging
 */
export function getRendererInfo(renderer) {
    if (!renderer) return null;

    const isWebGPU = renderer instanceof WebGPURenderer;

    return {
        type: isWebGPU ? 'WebGPU' : 'WebGL2',
        supportsWebGPU: isWebGPU,
        supportsCompute: isWebGPU,
        supportsTSL: isWebGPU,
        // Get WebGPU device if available
        device: isWebGPU ? renderer.device : null,
        adapter: isWebGPU ? renderer.adapter : null,
    };
}

/**
 * Display renderer info in console
 */
export function logRendererInfo(renderer) {
    const info = getRendererInfo(renderer);
    if (!info) {
        logger.warn('No renderer to inspect');
        return;
    }

    logger.log('📊 Renderer Info:', {
        Type: info.type,
        'Compute Shaders': info.supportsCompute ? '✅' : '❌',
        'TSL Support': info.supportsTSL ? '✅' : '❌',
        Device: info.device?.label || 'N/A',
    });
}

/**
 * Check if the current renderer supports advanced features
 */
export function getRendererCapabilities(renderer) {
    const info = getRendererInfo(renderer);
    if (!info) return { capabilities: [] };

    const capabilities = [];

    if (info.supportsWebGPU) {
        capabilities.push('webgpu', 'compute', 'tsl', 'storage-buffers', 'bind-groups');
    } else {
        capabilities.push('webgl2', 'glsl-shaders');
    }

    return {
        type: info.type,
        capabilities,
        hasCapability: (cap) => capabilities.includes(cap),
    };
}

// Export singleton for global access
export const RendererCapabilities = {
    instance: null,
    async init(renderer) {
        this.instance = getRendererCapabilities(renderer);
        return this.instance;
    },
    isWebGPU() {
        return this.instance?.type === 'WebGPU';
    },
    hasCompute() {
        return this.instance?.hasCapability('compute');
    },
    hasTSL() {
        return this.instance?.hasCapability('tsl');
    },
};
