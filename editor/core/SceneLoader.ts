import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { SplatMesh } from '@sparkjsdev/spark';
import EditorManager from './EditorManager.js';

/**
 * Scene Object Data Interface - matches sceneData.js format
 */
export interface SceneObjectData {
    id: string;
    type: 'splat' | 'gltf' | 'primitive';
    path?: string;
    description?: string;
    position: { x: number; y: number; z: number };
    rotation: { x: number; y: number; z: number };
    scale: { x: number; y: number; z: number } | number;
    priority?: number;
    preload?: boolean;
    criteria?: Record<string, any>;
    options?: {
        useContainer?: boolean;
        visible?: boolean;
        physicsCollider?: boolean;
        triggerColliders?: boolean;
        shadowBlocker?: boolean;
        debugMaterial?: boolean;
        envMap?: {
            metalness?: number;
            roughness?: number;
            envMapIntensity?: number;
            materials?: string[];
            excludeMaterials?: string[];
            materialOverrides?: Record<string, { metalness?: number; roughness?: number; envMapIntensity?: number }>;
        };
        materialRenderOrder?: Record<string, { renderOrder: number; criteria?: any }>;
        contactShadow?: {
            size?: { x: number; y: number };
            offset?: { x: number; y: number; z: number };
            blur?: number;
            darkness?: number;
            opacity?: number;
            cameraHeight?: number;
            renderTargetSize?: number;
            trackMesh?: string;
            updateFrequency?: number;
            isStatic?: boolean;
            debug?: boolean;
            criteria?: any;
        };
        gizmo?: boolean;
    };
    envMapWorldCenter?: { x: number; y: number; z: number };
    animations?: Array<{
        id: string;
        clipName?: string;
        loop: boolean;
        criteria?: any;
        autoPlay?: boolean;
        timeScale?: number;
        removeObjectOnFinish?: boolean;
    }>;
    zone?: string;
    gizmo?: boolean;
    loadByDefault?: boolean;
}

export interface SceneDataFormat {
    [objectId: string]: SceneObjectData;
}

/**
 * SceneLoader - Loads Shadow Czar scene objects into the editor
 *
 * Handles:
 * - Gaussian Splat (.sog) files using SplatMesh
 * - GLTF/GLB models using three.js GLTFLoader
 * - Primitive object creation
 * - Transform application
 */
class SceneLoader {
    private static instance: SceneLoader;

    private editorManager: EditorManager;
    private gltfLoader: GLTFLoader;

    // Loaded objects tracking
    private loadedObjects: Map<string, THREE.Object3D> = new Map();

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        this.editorManager = EditorManager.getInstance();
        this.gltfLoader = new GLTFLoader();
        console.log('SceneLoader: Constructor complete');
    }

    public static getInstance(): SceneLoader {
        if (!SceneLoader.instance) {
            SceneLoader.instance = new SceneLoader();
        }
        return SceneLoader.instance;
    }

    /**
     * Initialize the scene loader
     */
    public async initialize(): Promise<void> {
        console.log('SceneLoader: Initializing...');
        this.emit('initialized');
        console.log('SceneLoader: Initialization complete');
    }

    /**
    * Load a single scene object from sceneData
    */
    public async loadSceneObject(objectData: SceneObjectData, options: { showDebugColliders?: boolean } = {}): Promise<THREE.Object3D | null> {
        const scene = this.editorManager.scene;

        console.log(`SceneLoader: Loading object ${objectData.id} of type ${objectData.type}`);

        try {
            let object3D: THREE.Object3D | null = null;

            switch (objectData.type) {
                case 'splat':
                    object3D = await this.loadSplat(objectData);
                    break;

                case 'gltf':
                    object3D = await this.loadGLTF(objectData);
                    break;

                case 'primitive':
                    object3D = this.createPrimitive(objectData);
                    break;

                default:
                    console.warn(`SceneLoader: Unknown object type: ${objectData.type}`);
                    return null;
            }

            if (object3D) {
                // Apply transforms
                object3D.position.set(
                    objectData.position.x,
                    objectData.position.y,
                    objectData.position.z
                );

                object3D.rotation.set(
                    objectData.rotation.x,
                    objectData.rotation.y,
                    objectData.rotation.z
                );

                // Apply scale
                if (typeof objectData.scale === 'number') {
                    object3D.scale.set(objectData.scale, objectData.scale, objectData.scale);
                } else {
                    object3D.scale.set(objectData.scale.x, objectData.scale.y, objectData.scale.z);
                }

                // Set name and metadata
                object3D.name = objectData.id;
                object3D.userData.objectData = objectData;

                // Apply options (visibility, debug material, etc.)
                this.applyOptions(object3D, objectData, options);

                // Add to scene
                scene.add(object3D);

                // Track loaded object
                this.loadedObjects.set(objectData.id, object3D);

                console.log(`SceneLoader: Successfully loaded ${objectData.id}`);
                this.emit('objectLoaded', { id: objectData.id, object: object3D });
            }

            return object3D;
        } catch (error) {
            console.error(`SceneLoader: Failed to load object ${objectData.id}:`, error);
            this.emit('objectLoadError', { id: objectData.id, error });
            return null;
        }
    }

    /**
     * Load a Gaussian Splat (.sog) file using SplatMesh
     */
    private async loadSplat(objectData: SceneObjectData): Promise<THREE.Object3D> {
        if (!objectData.path) {
            throw new Error(`Splat object ${objectData.id} has no path`);
        }

        console.log(`SceneLoader: Loading splat from ${objectData.path}`);

        try {
            // Create SplatMesh - this matches your game runtime's approach
            const splatMesh = new SplatMesh({
                url: objectData.path,
                editable: false, // Don't apply SplatEdit operations to scene splats
            });

            // Wait for initialization - ensures asset is fully loaded
            await splatMesh.initialized;

            console.log(`SceneLoader: Splat ${objectData.id} loaded successfully`);
            return splatMesh;
        } catch (error) {
            console.error(`SceneLoader: Failed to load splat ${objectData.path}:`, error);
            // Create a placeholder on error
            return this.createPlaceholder(objectData, 0xff00ff);
        }
    }

    /**
     * Load a GLTF/GLB model
     */
    private async loadGLTF(objectData: SceneObjectData): Promise<THREE.Object3D> {
        if (!objectData.path) {
            throw new Error(`GLTF object ${objectData.id} has no path`);
        }

        console.log(`SceneLoader: Loading GLTF from ${objectData.path}`);

        return new Promise((resolve) => {
            this.gltfLoader.load(
                objectData.path!,
                (gltf) => {
                    let model: THREE.Object3D;

                    // Check if we should use a container
                    if (objectData.options?.useContainer) {
                        const container = new THREE.Group();
                        container.add(gltf.scene);
                        model = container;
                    } else {
                        model = gltf.scene;
                    }

                    // Store animations
                    if (gltf.animations && gltf.animations.length > 0) {
                        model.userData.animations = gltf.animations;
                        model.userData.animationClips = gltf.animations;
                    }

                    console.log(`SceneLoader: GLTF loaded successfully`);
                    resolve(model);
                },
                (progress) => {
                    const percent = (progress.loaded / (progress.total || 1)) * 100;
                    console.log(`SceneLoader: GLTF loading ${percent.toFixed(1)}%`);
                },
                (error) => {
                    console.error(`SceneLoader: GLTF loading failed:`, error);
                    resolve(this.createPlaceholder(objectData, 0xffff00));
                }
            );
        });
    }

    /**
     * Create a primitive object
     */
    private createPrimitive(objectData: SceneObjectData): THREE.Object3D {
        let geometry: THREE.BufferGeometry;
        let material = new THREE.MeshStandardMaterial({ color: 0x5c9aff });

        // Determine primitive type from description or default to box
        const type = objectData.description?.toLowerCase() || '';

        if (type.includes('sphere')) {
            geometry = new THREE.SphereGeometry(0.5, 32, 32);
        } else if (type.includes('plane')) {
            geometry = new THREE.PlaneGeometry(2, 2);
            material.side = THREE.DoubleSide;
        } else if (type.includes('cone')) {
            geometry = new THREE.ConeGeometry(0.5, 1, 32);
        } else if (type.includes('cylinder')) {
            geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
        } else {
            geometry = new THREE.BoxGeometry(1, 1, 1);
        }

        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = objectData.id;
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        return mesh;
    }

    /**
     * Create a placeholder object for failed loads
     */
    private createPlaceholder(objectData: SceneObjectData, color: number = 0x00ff00): THREE.Object3D {
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshStandardMaterial({
            color: color,
            wireframe: true
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = objectData.id + '_placeholder';

        // Add label
        mesh.userData.description = objectData.description || objectData.id;

        return mesh;
    }

    /**
    * Apply object options (visibility, debug material, etc.)
    */
    private applyOptions(object: THREE.Object3D, objectData: SceneObjectData, debugOptions: { showDebugColliders?: boolean } = {}): void {
        if (!objectData.options) return;

        // Visibility
        if (objectData.options.visible === false) {
            // Force visibility if it's a collider and we want to see them
            if (debugOptions.showDebugColliders && (objectData.options.physicsCollider || objectData.options.triggerColliders)) {
                object.visible = true;
            } else {
                object.visible = false;
            }
        }

        // Debug material (wireframe)
        // If showDebugColliders is on, force wireframe for anything that's a collider or shadow blocker
        const forceDebug = debugOptions.showDebugColliders &&
            (objectData.options.physicsCollider || objectData.options.triggerColliders || objectData.options.shadowBlocker);

        if (objectData.options.debugMaterial || forceDebug) {
            object.traverse((child) => {
                if (child instanceof THREE.Mesh) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(mat => {
                            mat.wireframe = true;
                            mat.transparent = true;
                            mat.opacity = 0.5;
                        });
                    } else {
                        child.material.wireframe = true;
                        child.material.transparent = true;
                        child.material.opacity = 0.5;
                    }
                }
            });
        }

        // EnvMap settings (for GLTF materials)
        if (objectData.options.envMap) {
            object.traverse((child) => {
                if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshStandardMaterial) {
                    const envMap = objectData.options!.envMap!;
                    if (envMap.metalness !== undefined) child.material.metalness = envMap.metalness;
                    if (envMap.roughness !== undefined) child.material.roughness = envMap.roughness;
                    // Note: envMapIntensity requires actual envMap texture to be set
                }
            });
        }

        // Shadow blocker (special material)
        if (objectData.options.shadowBlocker) {
            this._applyShadowBlockerMaterial(objectData.id, object, debugOptions.showDebugColliders);
        }
    }

    /**
    * Apply shadow blocker material (depth-only, no color) to block contact shadows
    */
    private _applyShadowBlockerMaterial(_id: string, object: THREE.Object3D, showDebug: boolean = false): void {
        const material = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            wireframe: true,
            transparent: true,
            opacity: showDebug ? 0.5 : 0.0,
            depthWrite: true,
            side: THREE.DoubleSide
        });

        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                child.material = material;
                child.renderOrder = 9998.1;
            }
        });

        // Always visible in Editor context if we want to see it, 
        // otherwise it just does its depth magic silently
        object.visible = showDebug || !object.userData.objectData?.options?.visible === false;
    }

    /**
     * Unload a scene object
     */
    public unloadSceneObject(objectId: string): void {
        const object = this.loadedObjects.get(objectId);
        if (!object) {
            console.warn(`SceneLoader: Object ${objectId} not found`);
            return;
        }

        const scene = this.editorManager.scene;
        scene.remove(object);

        // Clean up Three.js resources
        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                child.geometry.dispose();
                if (Array.isArray(child.material)) {
                    child.material.forEach(mat => mat.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });

        this.loadedObjects.delete(objectId);
        console.log(`SceneLoader: Unloaded ${objectId}`);
        this.emit('objectUnloaded', { id: objectId });
    }

    /**
     * Unload all scene objects
     */
    public unloadAll(): void {
        const objectIds = Array.from(this.loadedObjects.keys());
        objectIds.forEach(id => this.unloadSceneObject(id));
        console.log('SceneLoader: All objects unloaded');
    }

    /**
     * Get a loaded object by ID
     */
    public getObject(objectId: string): THREE.Object3D | undefined {
        return this.loadedObjects.get(objectId);
    }

    /**
     * Get all loaded object IDs
     */
    public getLoadedObjectIds(): string[] {
        return Array.from(this.loadedObjects.keys());
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
                    console.error(`SceneLoader: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }
}

export default SceneLoader;
