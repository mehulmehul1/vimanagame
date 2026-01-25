import * as THREE from 'three';
import EditorManager from './EditorManager.js';
import SceneLoader, { SceneObjectData, SceneDataFormat } from './SceneLoader.js';

/**
 * PriorityLoadManager - Manages priority-based scene object loading
 *
 * Features:
 * - Sorts objects by priority property before loading
 * - Loads visible-in-frustum objects first (using frustum culling check)
 * - Shows loading progress indicator
 * - Cancels loading when switching scenes
 *
 * Usage:
 * 1. Set priority on objects: objectData.priority (0 = highest, 100 = lowest)
 * 2. Call loadSceneWithPriority() to load a scene with smart ordering
 */
class PriorityLoadManager {
    private static instance: PriorityLoadManager;

    private editorManager: EditorManager;
    private sceneLoader: SceneLoader;

    // Loading state
    private isLoading: boolean = false;
    private currentLoadAbortController: AbortController | null = null;
    private loadProgress: { loaded: number; total: number; current: string } = {
        loaded: 0,
        total: 0,
        current: ''
    };

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        this.editorManager = EditorManager.getInstance();
        this.sceneLoader = SceneLoader.getInstance();
        console.log('PriorityLoadManager: Constructor complete');
    }

    public static getInstance(): PriorityLoadManager {
        if (!PriorityLoadManager.instance) {
            PriorityLoadManager.instance = new PriorityLoadManager();
        }
        return PriorityLoadManager.instance;
    }

    /**
     * Initialize the priority load manager
     */
    public async initialize(): Promise<void> {
        console.log('PriorityLoadManager: Initializing...');
        this.emit('initialized');
    }

    /**
     * Load a scene with priority-based ordering
     */
    public async loadSceneWithPriority(
        sceneData: SceneDataFormat,
        options?: {
            frustumFirst?: boolean; // Load frustum-visible objects first
            progressCallback?: (progress: { loaded: number; total: number; current: string }) => void;
        }
    ): Promise<void> {
        if (this.isLoading) {
            this.cancelLoading();
        }

        this.isLoading = true;
        this.currentLoadAbortController = new AbortController();

        console.log('PriorityLoadManager: Starting priority load...');

        try {
            // Convert scene data to array
            const objectsArray = Object.entries(sceneData).map(([id, data]) => ({
                id,
                ...data
            }));

            this.loadProgress.total = objectsArray.length;
            this.loadProgress.loaded = 0;
            this.emit('loadStarted', { total: objectsArray.length });

            // Sort by priority (lower number = higher priority)
            const sortedByPriority = objectsArray.sort((a, b) => {
                const priorityA = a.priority ?? 50;
                const priorityB = b.priority ?? 50;
                return priorityA - priorityB;
            });

            let loadOrder = sortedByPriority;

            // Optionally sort by frustum visibility
            if (options?.frustumFirst) {
                loadOrder = this.sortByFrustumVisibility(loadOrder);
            }

            // Load objects in order
            for (const objectData of loadOrder) {
                // Check if loading was cancelled
                if (this.currentLoadAbortController?.signal.aborted) {
                    console.log('PriorityLoadManager: Loading cancelled');
                    break;
                }

                this.loadProgress.current = objectData.id;
                this.emit('loadProgress', { ...this.loadProgress });

                if (options?.progressCallback) {
                    options.progressCallback({ ...this.loadProgress });
                }

                await this.sceneLoader.loadSceneObject(objectData as SceneObjectData);

                this.loadProgress.loaded++;
            }

            this.isLoading = false;
            this.emit('loadComplete', { loaded: this.loadProgress.loaded, total: this.loadProgress.total });

            console.log(`PriorityLoadManager: Loaded ${this.loadProgress.loaded}/${this.loadProgress.total} objects`);
        } catch (error) {
            console.error('PriorityLoadManager: Load failed:', error);
            this.isLoading = false;
            this.emit('loadError', { error });
            throw error;
        }
    }

    /**
     * Sort objects by frustum visibility (visible objects first)
     */
    private sortByFrustumVisibility(objects: Array<SceneObjectData & { id: string }>): Array<SceneObjectData & { id: string }> {
        const camera = this.editorManager.camera;
        const frustum = new THREE.Frustum();
        const projScreenMatrix = new THREE.Matrix4();

        projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        frustum.setFromProjectionMatrix(projScreenMatrix);

        // Create a temporary sphere for each object to test frustum visibility
        const frustumVisible: Array<SceneObjectData & { id: string }> = [];
        const frustumCulled: Array<SceneObjectData & { id: string }> = [];

        for (const obj of objects) {
            const position = new THREE.Vector3(obj.position.x, obj.position.y, obj.position.z);
            const radius = obj.options?.gizmo ? 2 : 5; // Default culling radius

            const sphere = new THREE.Sphere(position, radius);

            if (frustum.intersectsSphere(sphere)) {
                frustumVisible.push(obj);
            } else {
                frustumCulled.push(obj);
            }
        }

        console.log(`PriorityLoadManager: ${frustumVisible.length} visible, ${frustumCulled.length} culled objects`);

        // Return visible first, then culled
        return [...frustumVisible, ...frustumCulled];
    }

    /**
     * Check if an object is within the camera frustum
     */
    public isInFrustum(object: THREE.Object3D): boolean {
        const camera = this.editorManager.camera;
        const frustum = new THREE.Frustum();
        const projScreenMatrix = new THREE.Matrix4();

        projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        frustum.setFromProjectionMatrix(projScreenMatrix);

        // Get bounding sphere
        const sphere = new THREE.Sphere();
        object.updateWorldMatrix(true, false);

        if (object.geometry) {
            sphere.copy(object.geometry.boundingSphere || new THREE.Sphere());
            sphere.applyMatrix4(object.matrixWorld);
        } else {
            sphere.set(object.position, 1);
        }

        return frustum.intersectsSphere(sphere);
    }

    /**
     * Cancel the current loading operation
     */
    public cancelLoading(): void {
        if (this.currentLoadAbortController) {
            this.currentLoadAbortController.abort();
            this.currentLoadAbortController = null;
            this.isLoading = false;
            console.log('PriorityLoadManager: Loading cancelled');
            this.emit('loadCancelled');
        }
    }

    /**
     * Get current loading progress
     */
    public getLoadProgress(): { loaded: number; total: number; current: string } {
        return { ...this.loadProgress };
    }

    /**
     * Check if currently loading
     */
    public getIsLoading(): boolean {
        return this.isLoading;
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
                    console.error(`PriorityLoadManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }
}

export default PriorityLoadManager;
