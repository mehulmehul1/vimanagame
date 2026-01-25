/**
 * QualityPresets - Quality level presets for different device tiers
 *
 * Provides preset configurations for visual quality settings
 * based on device capabilities and user preferences.
 */

import { DeviceTier } from './DeviceCapabilities';

export interface QualitySettings {
    /** Shader quality (0-1) */
    shaderQuality: number;
    /** Particle count multiplier */
    particleMultiplier: number;
    /** Texture resolution multiplier */
    textureMultiplier: number;
    /** Shadow map size */
    shadowMapSize: number;
    /** Enable shadows */
    enableShadows: boolean;
    /** Enable post-processing */
    enablePostProcessing: boolean;
    /** Enable reflections */
    enableReflections: boolean;
    /** Antialiasing samples */
    antialiasSamples: number;
    /** Maximum lights */
    maxLights: number;
    /** LOD distance multiplier */
    lodDistanceMultiplier: number;
    /** Enable volumetric effects */
    enableVolumetric: boolean;
}

export const QUALITY_PRESETS: Record<DeviceTier, QualitySettings> = {
    ultra: {
        shaderQuality: 1.0,
        particleMultiplier: 2.0,
        textureMultiplier: 2.0,
        shadowMapSize: 2048,
        enableShadows: true,
        enablePostProcessing: true,
        enableReflections: true,
        antialiasSamples: 8,
        maxLights: 16,
        lodDistanceMultiplier: 2.0,
        enableVolumetric: true
    },
    high: {
        shaderQuality: 0.85,
        particleMultiplier: 1.5,
        textureMultiplier: 1.0,
        shadowMapSize: 1024,
        enableShadows: true,
        enablePostProcessing: true,
        enableReflections: true,
        antialiasSamples: 4,
        maxLights: 8,
        lodDistanceMultiplier: 1.5,
        enableVolumetric: true
    },
    medium: {
        shaderQuality: 0.7,
        particleMultiplier: 1.0,
        textureMultiplier: 1.0,
        shadowMapSize: 512,
        enableShadows: true,
        enablePostProcessing: false,
        enableReflections: false,
        antialiasSamples: 2,
        maxLights: 4,
        lodDistanceMultiplier: 1.0,
        enableVolumetric: false
    },
    low: {
        shaderQuality: 0.5,
        particleMultiplier: 0.5,
        textureMultiplier: 0.5,
        shadowMapSize: 256,
        enableShadows: false,
        enablePostProcessing: false,
        enableReflections: false,
        antialiasSamples: 0,
        maxLights: 2,
        lodDistanceMultiplier: 0.7,
        enableVolumetric: false
    }
};

export class QualityPresets {
    private currentTier: DeviceTier;
    private currentSettings: QualitySettings;

    constructor(tier: DeviceTier = 'medium') {
        this.currentTier = tier;
        this.currentSettings = { ...QUALITY_PRESETS[tier] };
    }

    /**
     * Get settings for a specific tier
     */
    public static getSettings(tier: DeviceTier): QualitySettings {
        return { ...QUALITY_PRESETS[tier] };
    }

    /**
     * Get current settings
     */
    public getSettings(): QualitySettings {
        return { ...this.currentSettings };
    }

    /**
     * Set quality tier
     */
    public setTier(tier: DeviceTier): void {
        this.currentTier = tier;
        this.currentSettings = { ...QUALITY_PRESETS[tier] };
    }

    /**
     * Get current tier
     */
    public getTier(): DeviceTier {
        return this.currentTier;
    }

    /**
     * Update specific setting
     */
    public updateSetting<K extends keyof QualitySettings>(
        key: K,
        value: QualitySettings[K]
    ): void {
        this.currentSettings[key] = value;
    }

    /**
     * Apply settings to Three.js renderer
     */
    public applyToRenderer(renderer: THREE.WebGLRenderer): void {
        const settings = this.currentSettings;

        // Antialiasing
        // Note: This requires renderer recreation for proper effect
        // stored here for reference when creating renderer

        // Shadow map size
        if (renderer.shadowMap) {
            renderer.shadowMap.type = settings.shaderQuality > 0.7
                ? THREE.PCFSoftShadowMap
                : THREE.BasicShadowMap;
        }

        // Pixel ratio
        const pixelRatio = Math.min(
            window.devicePixelRatio * settings.textureMultiplier,
            2
        );
        renderer.setPixelRatio(pixelRatio);

        // Tone mapping
        renderer.toneMapping = settings.shaderQuality > 0.7
            ? THREE.ACESFilmicToneMapping
            : THREE.LinearToneMapping;

        // Tone mapping exposure
        renderer.toneMappingExposure = 1.0;

        // Output encoding
        renderer.outputColorSpace = THREE.SRGBColorSpace;
    }

    /**
     * Get particle count for base count
     */
    public getParticleCount(baseCount: number): number {
        return Math.floor(baseCount * this.currentSettings.particleMultiplier);
    }

    /**
     * Get texture size for base size
     */
    public getTextureSize(baseSize: number): number {
        return Math.floor(baseSize * this.currentSettings.textureMultiplier);
    }

    /**
     * Check if shadows are enabled
     */
    public shadowsEnabled(): boolean {
        return this.currentSettings.enableShadows;
    }

    /**
     * Check if post-processing is enabled
     */
    public postProcessingEnabled(): boolean {
        return this.currentSettings.enablePostProcessing;
    }

    /**
     * Get shader quality (0-1)
     */
    public getShaderQuality(): number {
        return this.currentSettings.shaderQuality;
    }

    /**
     * Create a custom preset
     */
    public static createCustom(
        baseTier: DeviceTier,
        overrides: Partial<QualitySettings>
    ): QualitySettings {
        const base = QUALITY_PRESETS[baseTier];
        return { ...base, ...overrides };
    }
}
