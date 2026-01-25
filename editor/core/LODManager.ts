import * as THREE from 'three';
import EditorManager from './EditorManager.js';

/**
 * LOD level definitions for Gaussian splats
 */
export interface LODLevel {
    name: string;
    distance: number; // Maximum distance for this LOD level
    quality: number; // Splat quality setting
    splatCount: number; // Target splat count
}

/**
 * LOD configuration for an object
 */
export interface LODConfig {
    enabled: boolean;
    levels: LODLevel[];
    currentLevel: number;
    transitionDistance: number; // Smooth transition zone
}

/**
 * Default LOD levels for Gaussian splats
 */
const DEFAULT_LOD_LEVELS: LODLevel[] = [
    { name: 'Near', distance: 20, quality: 1.0, splatCount: 16000 },
    { name: 'Medium', distance: 50, quality: 0.6, splatCount: 35000 },
    { name: 'Far', distance: 100, quality: 0.3, splatCount: 8000 },
    { name: 'Preview', distance: Infinity, quality: 0.1, splatCount: 2000 }
];

/**
 * LODManager - Manages Level of Detail for Gaussian splats
 *
 * Features:
 * - Distance-based quality adjustment for Gaussian splats
 * - Smooth transitions between LOD levels
 * - Per-object LOD configuration via userData
 * - Automatic LOD updates based on camera distance
 *
 * Usage:
 * 1. Enable LOD on objects: object.userData.lodEnabled = true
 * 2. Set custom levels: object.userData.lodLevels = [...]
 * 3. LODManager automatically updates quality based on distance
 */
class LODManager {
    private static instance: LODManager;

    private editorManager: EditorManager;

    // LOD settings
    private enabled: boolean = true;
    private updateInterval: number = 100; // ms between LOD updates
    private lodObjects: Map<THREE.Object3D, LODConfig> = new Map();

    // Update loop
    private updateTimer: number | null = null;
    private isRunning: boolean = false;

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        this.editorManager = EditorManager.getInstance();
        console.log('LODManager: Constructor complete');
    }

    public static getInstance(): LODManager {
        if (!LODManager.instance) {
            LODManager.instance = new LODManager();
        }
        return LODManager.instance;
    }

    /**
     * Initialize the LOD manager
     */
    public async initialize(): Promise<void> {
        console.log('LODManager: Initializing...');
        this.emit('initialized');
    }

    /**
     * Start LOD updates
     */
    public start(): void {
        if (this.isRunning) return;

        this.isRunning = true;
        this.updateLOD();

        console.log('LODManager: Started LOD updates');
        this.emit('started');
    }

    /**
     * Stop LOD updates
     */
    public stop(): void {
        if (!this.isRunning) return;

        this.isRunning = false;
        if (this.updateTimer !== null) {
            clearTimeout(this.updateTimer);
            this.updateTimer = null;
        }

        console.log('LODManager: Stopped LOD updates');
        this.emit('stopped');
    }

    /**
     * Enable/disable LOD system
     */
    public setEnabled(enabled: boolean): void {
        this.enabled = enabled;

        if (!enabled) {
            // Reset all objects to highest quality
            this.lodObjects.forEach((config, object) => {
                this.applyLODLevel(object, 0);
            });
        }

        console.log(`LODManager: ${enabled ? 'Enabled' : 'Disabled'}`);
        this.emit('enabledChanged', { enabled });
    }

    /**
     * Register an object for LOD management
     */
    public registerObject(object: THREE.Object3D, config?: Partial<LODConfig>): void {
        const lodConfig: LODConfig = {
            enabled: config?.enabled ?? true,
            levels: config?.levels ?? DEFAULT_LOD_LEVELS,
            currentLevel: 0,
            transitionDistance: config?.transitionDistance ?? 5
        };

        // Store config in userData for persistence
        object.userData.lodConfig = lodConfig;
        this.lodObjects.set(object, lodConfig);

        console.log(`LODManager: Registered object ${object.name || object.uuid} for LOD`);
        this.emit('objectRegistered', { object, config: lodConfig });
    }

    /**
     * Unregister an object from LOD management
     */
    public unregisterObject(object: THREE.Object3D): void {
        const config = this.lodObjects.get(object);
        if (config) {
            // Reset to highest quality before unregistering
            this.applyLODLevel(object, 0);
        }

        this.lodObjects.delete(object);
        delete object.userData.lodConfig;

        console.log(`LODManager: Unregistered object ${object.name || object.uuid}`);
        this.emit('objectUnregistered', { object });
    }

    /**
     * Update LOD for all registered objects
     */
    private updateLOD(): void {
        if (!this.isRunning || !this.enabled) {
            this.updateTimer = window.setTimeout(() => this.updateLOD(), this.updateInterval);
            return;
        }

        const camera = this.editorManager.camera;

        this.lodObjects.forEach((config, object) => {
            if (!config.enabled || !object.visible) return;

            // Calculate distance from camera
            const distance = camera.position.distanceTo(object.position);

            // Find appropriate LOD level
            const newLevel = this.calculateLODLevel(distance, config.levels);

            // Apply LOD if level changed
            if (newLevel !== config.currentLevel) {
                const oldLevel = config.currentLevel;
                config.currentLevel = newLevel;
                this.applyLODLevel(object, newLevel, config.levels[newLevel]);

                this.emit('lodChanged', {
                    object,
                    oldLevel,
                    newLevel,
                    distance
                });
            }
        });

        // Schedule next update
        this.updateTimer = window.setTimeout(() => this.updateLOD(), this.updateInterval);
    }

    /**
     * Calculate the appropriate LOD level based on distance
     */
    private calculateLODLevel(distance: number, levels: LODLevel[]): number {
        for (let i = 0; i < levels.length; i++) {
            if (distance <= levels[i].distance) {
                return i;
            }
        }
        return levels.length - 1;
    }

    /**
     * Apply LOD level to object
     */
    private applyLODLevel(object: THREE.Object3D, levelIndex: number, level?: LODLevel): void {
        // Check if this is a SplatMesh with quality control
        if ('splatData' in object && 'setQuality' in object) {
            // @ts-ignore - SplatMesh custom properties
            const quality = level?.quality ?? (levelIndex === 0 ? 1.0 : 1.0 - (levelIndex * 0.3));
            // @ts-ignore
            if (typeof object.setQuality === 'function') {
                // @ts-ignore
                object.setQuality(quality);
            }
        }

        // For Three.js objects, adjust render quality
        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                // Adjust material quality based on LOD
                if (child.material instanceof THREE.Material) {
                    child.material.needsUpdate = true;
                }
            }
        });

        // Store current LOD level
        object.userData.currentLOD = levelIndex;
    }

    /**
     * Force a specific LOD level for an object
     */
    public forceLODLevel(object: THREE.Object3D, level: number): void {
        const config = this.lodObjects.get(object);
        if (config) {
            config.currentLevel = Math.max(0, Math.min(level, config.levels.length - 1));
            this.applyLODLevel(object, config.currentLevel, config.levels[config.currentLevel]);
        }
    }

    /**
     * Get the current LOD level for an object
     */
    public getCurrentLODLevel(object: THREE.Object3D): number | null {
        const config = this.lodObjects.get(object);
        return config ? config.currentLevel : null;
    }

    /**
     * Set LOD update interval
     */
    public setUpdateInterval(interval: number): void {
        this.updateInterval = interval;
        console.log(`LODManager: Update interval set to ${interval}ms`);
    }

    /**
     * Get default LOD levels
     */
    public static getDefaultLODLevels(): LODLevel[] {
        return [...DEFAULT_LOD_LEVELS];
    }

    /**
     * Create custom LOD levels
     */
    public static createCustomLODLevels(config: {
        near?: { distance: number; quality: number };
        medium?: { distance: number; quality: number };
        far?: { distance: number; quality: number };
    }): LODLevel[] {
        return [
            {
                name: 'Near',
                distance: config.near?.distance ?? 20,
                quality: config.near?.quality ?? 1.0,
                splatCount: 16000
            },
            {
                name: 'Medium',
                distance: config.medium?.distance ?? 50,
                quality: config.medium?.quality ?? 0.6,
                splatCount: 35000
            },
            {
                name: 'Far',
                distance: config.far?.distance ?? 100,
                quality: config.far?.quality ?? 0.3,
                splatCount: 8000
            },
            {
                name: 'Preview',
                distance: Infinity,
                quality: 0.1,
                splatCount: 2000
            }
        ];
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
                    console.error(`LODManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Clean up
     */
    public destroy(): void {
        this.stop();
        this.lodObjects.clear();
        this.eventListeners.clear();
        console.log('LODManager: Destroyed');
    }
}

export default LODManager;
