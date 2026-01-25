import * as THREE from 'three';
import EditorManager from './EditorManager.js';

/**
 * Culling configuration for an object
 */
export interface CullingConfig {
    enabled: boolean;
    cullingRadius: number; // Radius for frustum testing
    excludeFromCulling: boolean; // Flag to always render
    isCulled: boolean; // Current culling state
}

/**
 * FrustumCullingManager - Manages frustum culling for scene objects
 *
 * Features:
 * - Culls objects outside camera frustum
 * - Per-object culling radius via userData.cullingRadius
 * - Debug overlay showing culled objects (toggle with F8)
 * - excludeFromCulling flag to always render certain objects
 *
 * Usage:
 * 1. Enable culling on objects: object.userData.cullingRadius = 5
 * 2. Or exclude from culling: object.userData.excludeFromCulling = true
 * 3. FrustumCullingManager automatically updates visibility
 */
class FrustumCullingManager {
    private static instance: FrustumCullingManager;

    private editorManager: EditorManager;

    // Culling state
    private enabled: boolean = true;
    private showDebugOverlay: boolean = false;
    private cullingObjects: Map<THREE.Object3D, CullingConfig> = new Map();

    // Frustum cache
    private frustum: THREE.Frustum = new THREE.Frustum();
    private projScreenMatrix: THREE.Matrix4 = new THREE.Matrix4();

    // Update loop
    private updateTimer: number | null = null;
    private updateInterval: number = 50; // ms between updates

    // Debug visualization
    private debugOverlay: HTMLElement | null = null;

    // Stats
    private stats = {
        total: 0,
        culled: 0,
        visible: 0
    };

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        this.editorManager = EditorManager.getInstance();
        console.log('FrustumCullingManager: Constructor complete');
    }

    public static getInstance(): FrustumCullingManager {
        if (!FrustumCullingManager.instance) {
            FrustumCullingManager.instance = new FrustumCullingManager();
        }
        return FrustumCullingManager.instance;
    }

    /**
     * Initialize the frustum culling manager
     */
    public async initialize(): Promise<void> {
        console.log('FrustumCullingManager: Initializing...');

        // Setup keyboard shortcut for debug toggle (F8)
        window.addEventListener('keydown', this.handleKeyDown);

        // Create debug overlay
        this.createDebugOverlay();

        this.emit('initialized');
    }

    /**
     * Start frustum culling updates
     */
    public start(): void {
        if (this.updateTimer !== null) return;

        this.updateFrustumCulling();

        console.log('FrustumCullingManager: Started culling updates');
        this.emit('started');
    }

    /**
     * Stop frustum culling updates
     */
    public stop(): void {
        if (this.updateTimer !== null) {
            clearTimeout(this.updateTimer);
            this.updateTimer = null;
        }

        // Reset all objects to visible
        this.cullingObjects.forEach((config, object) => {
            if (config.isCulled) {
                object.visible = true;
                config.isCulled = false;
            }
        });

        console.log('FrustumCullingManager: Stopped culling updates');
        this.emit('stopped');
    }

    /**
     * Enable/disable frustum culling
     */
    public setEnabled(enabled: boolean): void {
        this.enabled = enabled;

        if (!enabled) {
            // Reset all objects to visible
            this.cullingObjects.forEach((config, object) => {
                object.visible = true;
                config.isCulled = false;
            });
        }

        console.log(`FrustumCullingManager: ${enabled ? 'Enabled' : 'Disabled'}`);
        this.emit('enabledChanged', { enabled });
    }

    /**
     * Register an object for frustum culling
     */
    public registerObject(
        object: THREE.Object3D,
        config?: Partial<CullingConfig>
    ): void {
        const cullingConfig: CullingConfig = {
            enabled: config?.enabled ?? true,
            cullingRadius: config?.cullingRadius ?? this.calculateBoundingRadius(object),
            excludeFromCulling: config?.excludeFromCulling ?? false,
            isCulled: false
        };

        // Store config in userData
        object.userData.cullingConfig = cullingConfig;
        this.cullingObjects.set(object, cullingConfig);

        console.log(`FrustumCullingManager: Registered ${object.name || object.uuid} (radius: ${cullingConfig.cullingRadius.toFixed(2)})`);
        this.emit('objectRegistered', { object, config: cullingConfig });
    }

    /**
     * Unregister an object from frustum culling
     */
    public unregisterObject(object: THREE.Object3D): void {
        const config = this.cullingObjects.get(object);
        if (config && config.isCulled) {
            object.visible = true;
        }

        this.cullingObjects.delete(object);
        delete object.userData.cullingConfig;

        console.log(`FrustumCullingManager: Unregistered ${object.name || object.uuid}`);
        this.emit('objectUnregistered', { object });
    }

    /**
     * Update frustum culling for all registered objects
     */
    private updateFrustumCulling(): void {
        if (!this.enabled) {
            this.updateTimer = window.setTimeout(() => this.updateFrustumCulling(), this.updateInterval);
            return;
        }

        const camera = this.editorManager.camera;

        // Update frustum
        this.projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        this.frustum.setFromProjectionMatrix(this.projScreenMatrix);

        // Reset stats
        this.stats.total = this.cullingObjects.size;
        this.stats.culled = 0;
        this.stats.visible = 0;

        // Test each object
        this.cullingObjects.forEach((config, object) => {
            if (!config.enabled || config.excludeFromCulling) {
                this.stats.visible++;
                return;
            }

            const isVisible = this.testFrustum(object, config);

            if (isVisible && config.isCulled) {
                // Object became visible
                object.visible = true;
                config.isCulled = false;
                this.emit('objectVisible', { object });
            } else if (!isVisible && !config.isCulled) {
                // Object became culled
                object.visible = false;
                config.isCulled = true;
                this.stats.culled++;
                this.emit('objectCulled', { object });
            } else if (!isVisible) {
                this.stats.culled++;
            } else {
                this.stats.visible++;
            }
        });

        // Update debug overlay
        if (this.showDebugOverlay) {
            this.updateDebugOverlay();
        }

        // Schedule next update
        this.updateTimer = window.setTimeout(() => this.updateFrustumCulling(), this.updateInterval);
    }

    /**
     * Test if an object intersects the frustum
     */
    private testFrustum(object: THREE.Object3D, config: CullingConfig): boolean {
        // Create bounding sphere at object position
        const sphere = new THREE.Sphere(
            object.position.clone(),
            config.cullingRadius
        );

        return this.frustum.intersectsSphere(sphere);
    }

    /**
     * Calculate bounding radius for an object
     */
    private calculateBoundingRadius(object: THREE.Object3D): number {
        // Check for geometry-based bounding sphere
        if (object instanceof THREE.Mesh && object.geometry) {
            object.geometry.computeBoundingSphere();
            if (object.geometry.boundingSphere) {
                return object.geometry.boundingSphere.radius * Math.max(
                    object.scale.x,
                    object.scale.y,
                    object.scale.z
                );
            }
        }

        // Check children
        let maxRadius = 1;
        object.children.forEach(child => {
            const childRadius = this.calculateBoundingRadius(child);
            const childDistance = child.position.length() + childRadius;
            if (childDistance > maxRadius) {
                maxRadius = childDistance;
            }
        });

        return maxRadius;
    }

    /**
     * Create debug overlay element
     */
    private createDebugOverlay(): void {
        this.debugOverlay = document.createElement('div');
        this.debugOverlay.className = 'frustum-debug-overlay';
        this.debugOverlay.style.cssText = `
            position: fixed;
            top: 60px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: #0f0;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            z-index: 10000;
            display: none;
            pointer-events: none;
        `;
        document.body.appendChild(this.debugOverlay);
    }

    /**
     * Update debug overlay content
     */
    private updateDebugOverlay(): void {
        if (!this.debugOverlay) return;

        this.debugOverlay.innerHTML = `
            <div>FRUSTUM CULLING DEBUG</div>
            <div>Total: ${this.stats.total}</div>
            <div>Visible: ${this.stats.visible}</div>
            <div>Culled: ${this.stats.culled}</div>
            <div style="color: ${this.stats.culled > this.stats.visible ? '#ff0' : '#0f0'}">
                Saved: ${Math.round((this.stats.culled / this.stats.total) * 100)}%
            </div>
        `;
    }

    /**
     * Toggle debug overlay
     */
    public toggleDebugOverlay(): void {
        this.showDebugOverlay = !this.showDebugOverlay;

        if (this.debugOverlay) {
            this.debugOverlay.style.display = this.showDebugOverlay ? 'block' : 'none';
        }

        console.log(`FrustumCullingManager: Debug overlay ${this.showDebugOverlay ? 'enabled' : 'disabled'}`);
        this.emit('debugToggled', { enabled: this.showDebugOverlay });
    }

    /**
     * Handle keyboard shortcuts
     */
    private handleKeyDown = (event: KeyboardEvent): void => {
        if (event.key === 'F8') {
            event.preventDefault();
            this.toggleDebugOverlay();
        }
    };

    /**
     * Manually test if an object is in frustum
     */
    public isInFrustum(object: THREE.Object3D, radius?: number): boolean {
        const camera = this.editorManager.camera;
        const testFrustum = new THREE.Frustum();
        const testMatrix = new THREE.Matrix4();

        testMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        testFrustum.setFromProjectionMatrix(testMatrix);

        const sphere = new THREE.Sphere(
            object.position.clone(),
            radius ?? this.calculateBoundingRadius(object)
        );

        return testFrustum.intersectsSphere(sphere);
    }

    /**
     * Get culling stats
     */
    public getStats(): { total: number; culled: number; visible: number } {
        return { ...this.stats };
    }

    /**
     * Set update interval
     */
    public setUpdateInterval(interval: number): void {
        this.updateInterval = interval;
        console.log(`FrustumCullingManager: Update interval set to ${interval}ms`);
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
                    console.error(`FrustumCullingManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Clean up
     */
    public destroy(): void {
        this.stop();

        window.removeEventListener('keydown', this.handleKeyDown);

        if (this.debugOverlay && this.debugOverlay.parentElement) {
            this.debugOverlay.parentElement.removeChild(this.debugOverlay);
        }

        this.cullingObjects.clear();
        this.eventListeners.clear();

        console.log('FrustumCullingManager: Destroyed');
    }
}

export default FrustumCullingManager;
