import * as THREE from 'three';
import EditorManager from './EditorManager.js';

/**
 * Quality preset definitions
 */
export interface QualityPreset {
    name: string;
    pixelRatio: number;
    shadowsEnabled: boolean;
    antialiasing: boolean;
    renderScale: number;
}

/**
 * Quality presets
 */
export const QUALITY_PRESETS: Record<string, QualityPreset> = {
    low: {
        name: 'Low',
        pixelRatio: 0.5,
        shadowsEnabled: false,
        antialiasing: false,
        renderScale: 0.5
    },
    medium: {
        name: 'Medium',
        pixelRatio: 1.0,
        shadowsEnabled: true,
        antialiasing: true,
        renderScale: 1.0
    },
    high: {
        name: 'High',
        pixelRatio: 2.0,
        shadowsEnabled: true,
        antialiasing: true,
        renderScale: 1.5
    },
    ultra: {
        name: 'Ultra',
        pixelRatio: window.devicePixelRatio || 2,
        shadowsEnabled: true,
        antialiasing: true,
        renderScale: 2.0
    }
};

/**
 * QualitySettings - Manages editor quality settings
 *
 * Features:
 * - Pixel ratio selector: Low (0.5), Medium (1), High (2)
 * - Editor renders at pixelRatio: 1 by default
 * - High DPI mode for screenshots only
 * - UI in menubar to select quality
 *
 * Usage:
 * QualitySettings.getInstance().setQualityLevel('medium');
 */
class QualitySettings {
    private static instance: QualitySettings;

    private editorManager: EditorManager;

    // Current quality settings
    private currentQuality: keyof typeof QUALITY_PRESETS = 'medium';
    private customPixelRatio: number | null = null;
    private isHighDPIMode: boolean = false;

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        this.editorManager = EditorManager.getInstance();
        console.log('QualitySettings: Constructor complete');
    }

    public static getInstance(): QualitySettings {
        if (!QualitySettings.instance) {
            QualitySettings.instance = new QualitySettings();
        }
        return QualitySettings.instance;
    }

    /**
     * Initialize quality settings
     */
    public async initialize(): Promise<void> {
        console.log('QualitySettings: Initializing...');

        // Load saved quality preference from localStorage
        const savedQuality = localStorage.getItem('editor-quality') as keyof typeof QUALITY_PRESETS;
        if (savedQuality && QUALITY_PRESETS[savedQuality]) {
            this.currentQuality = savedQuality;
        }

        // Apply current quality
        this.applyQuality(this.currentQuality);

        this.emit('initialized');
    }

    /**
     * Set quality level by preset name
     */
    public setQualityLevel(level: keyof typeof QUALITY_PRESETS): void {
        if (!QUALITY_PRESETS[level]) {
            console.warn(`QualitySettings: Unknown quality level "${level}"`);
            return;
        }

        this.currentQuality = level;
        this.applyQuality(level);

        // Save to localStorage
        localStorage.setItem('editor-quality', level);

        console.log(`QualitySettings: Quality set to ${level}`);
        this.emit('qualityChanged', { level, preset: QUALITY_PRESETS[level] });
    }

    /**
     * Apply quality preset to renderer
     */
    private applyQuality(level: keyof typeof QUALITY_PRESETS): void {
        const preset = QUALITY_PRESETS[level];
        const renderer = this.editorManager.renderer;

        if (!renderer) {
            console.warn('QualitySettings: Renderer not available');
            return;
        }

        // Apply pixel ratio (unless custom is set)
        if (this.customPixelRatio === null) {
            const pixelRatio = this.isHighDPIMode ? window.devicePixelRatio : preset.pixelRatio;
            renderer.setPixelRatio(pixelRatio);
        }

        // Apply shadow map
        renderer.shadowMap.enabled = preset.shadowsEnabled;

        // Store settings for future reference
        renderer.userData.qualityPreset = level;
        renderer.userData.qualitySettings = preset;
    }

    /**
     * Set custom pixel ratio
     */
    public setPixelRatio(ratio: number): void {
        this.customPixelRatio = ratio;

        const renderer = this.editorManager.renderer;
        if (renderer) {
            renderer.setPixelRatio(ratio);
        }

        console.log(`QualitySettings: Custom pixel ratio set to ${ratio}`);
        this.emit('pixelRatioChanged', { ratio });
    }

    /**
     * Reset to preset pixel ratio
     */
    public resetPixelRatio(): void {
        this.customPixelRatio = null;
        this.applyQuality(this.currentQuality);

        console.log('QualitySettings: Pixel ratio reset to preset');
        this.emit('pixelRatioReset');
    }

    /**
     * Enable high DPI mode (for screenshots)
     */
    public setHighDPIMode(enabled: boolean): void {
        this.isHighDPIMode = enabled;

        const preset = QUALITY_PRESETS[this.currentQuality];
        const renderer = this.editorManager.renderer;

        if (renderer) {
            const pixelRatio = enabled ? window.devicePixelRatio : preset.pixelRatio;
            renderer.setPixelRatio(pixelRatio);
        }

        console.log(`QualitySettings: High DPI mode ${enabled ? 'enabled' : 'disabled'}`);
        this.emit('highDPIToggled', { enabled });
    }

    /**
     * Get current quality level
     */
    public getQualityLevel(): keyof typeof QUALITY_PRESETS {
        return this.currentQuality;
    }

    /**
     * Get current quality preset
     */
    public getQualityPreset(): QualityPreset {
        return QUALITY_PRESETS[this.currentQuality];
    }

    /**
     * Get all available quality presets
     */
    public static getQualityPresets(): typeof QUALITY_PRESETS {
        return QUALITY_PRESETS;
    }

    /**
     * Get current pixel ratio
     */
    public getCurrentPixelRatio(): number {
        const renderer = this.editorManager.renderer;
        return renderer?.getPixelRatio() ?? 1;
    }

    /**
     * Set render scale (affects resolution)
     */
    public setRenderScale(scale: number): void {
        const renderer = this.editorManager.renderer;
        if (!renderer) return;

        const currentSize = renderer.getSize(new THREE.Vector2());
        renderer.setSize(currentSize.x * scale, currentSize.y * scale);

        // Update renderer size style to maintain display size
        if (renderer.domElement) {
            renderer.domElement.style.width = '100%';
            renderer.domElement.style.height = '100%';
        }

        console.log(`QualitySettings: Render scale set to ${scale}x`);
        this.emit('renderScaleChanged', { scale });
    }

    /**
     * Toggle shadows
     */
    public setShadowsEnabled(enabled: boolean): void {
        const renderer = this.editorManager.renderer;
        if (!renderer) return;

        renderer.shadowMap.enabled = enabled;

        console.log(`QualitySettings: Shadows ${enabled ? 'enabled' : 'disabled'}`);
        this.emit('shadowsToggled', { enabled });
    }

    /**
     * Take a high quality screenshot
     */
    public takeScreenshot(): string {
        // Enable high DPI temporarily
        const wasHighDPI = this.isHighDPIMode;
        const previousPixelRatio = this.getCurrentPixelRatio();

        this.setHighDPIMode(true);

        // Force a render
        this.editorManager.renderer.render(
            this.editorManager.scene,
            this.editorManager.camera
        );

        // Capture screenshot
        const dataURL = this.editorManager.renderer.domElement.toDataURL('image/png');

        // Restore previous settings
        this.setHighDPIMode(wasHighDPI);

        console.log('QualitySettings: Screenshot captured');
        this.emit('screenshotTaken', { dataURL });

        return dataURL;
    }

    /**
     * Export current settings
     */
    public exportSettings(): Record<string, any> {
        return {
            quality: this.currentQuality,
            pixelRatio: this.getCurrentPixelRatio(),
            highDPIMode: this.isHighDPIMode,
            preset: QUALITY_PRESETS[this.currentQuality]
        };
    }

    /**
     * Import settings
     */
    public importSettings(settings: Record<string, any>): void {
        if (settings.quality && QUALITY_PRESETS[settings.quality]) {
            this.setQualityLevel(settings.quality);
        }

        if (typeof settings.pixelRatio === 'number') {
            this.setPixelRatio(settings.pixelRatio);
        }

        if (typeof settings.highDPIMode === 'boolean') {
            this.setHighDPIMode(settings.highDPIMode);
        }

        console.log('QualitySettings: Settings imported');
        this.emit('settingsImported', { settings });
    }

    /**
     * Register event listener
     */
    public on(eventName: string, callback: Function): void {
        if (!this.eventListeners.has(eventName)) {
            this.eventListeners.set(eventName, new Set());
        }
        this.eventListeners.get(eventName)!.add(callback);
    }

    /**
     * Unregister event listener
     */
    public off(eventName: string, callback: Function): void {
        const listeners = this.eventListeners.get(eventName);
        if (listeners) {
            listeners.delete(callback);
        }
    }

    /**
     * Emit event
     */
    private emit(eventName: string, data?: any): void {
        const listeners = this.eventListeners.get(eventName);
        if (listeners) {
            listeners.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`QualitySettings: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }
}

export default QualitySettings;
