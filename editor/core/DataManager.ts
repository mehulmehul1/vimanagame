import * as THREE from 'three';
import EditorManager from './EditorManager.js';
import FileWatcher from '../utils/FileWatcher.js';

/**
 * Scene Data Interface - matches sceneData.js format
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
 * DataManager - Handles sceneData.js read/write operations
 *
 * This class:
 * - Loads existing sceneData.js files
 * - Parses scene objects, transforms, and criteria
 * - Builds Three.js scene from data
 * - Writes editor state to sceneData.js format
 * - Preserves existing format exactly (game runtime depends on it)
 *
 * CRITICAL: Do not modify the packages/engine/src/ directory - this is editor-only code
 */
class DataManager {
    // Singleton instance
    private static instance: DataManager;

    // Scene data cache
    private sceneData: SceneDataFormat = {};
    private sceneDataFilePath: string = '../packages/engine/src/sceneData.js';

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    // Editor manager reference
    private editorManager: EditorManager;

    // File watcher
    private fileWatcher: FileWatcher;

    // Hot reload enabled flag
    // @ts-ignore - Used in enableHotReload method
    private hotReloadEnabled: boolean = false;

    /**
     * Private constructor for singleton pattern
     */
    private constructor() {
        this.editorManager = EditorManager.getInstance();
        this.fileWatcher = FileWatcher.getInstance();
        console.log('DataManager: Constructor complete');
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): DataManager {
        if (!DataManager.instance) {
            DataManager.instance = new DataManager();
        }
        return DataManager.instance;
    }

    /**
     * Load sceneData.js file
     */
    public async loadSceneData(filePath?: string): Promise<SceneDataFormat> {
        const targetPath = filePath || this.sceneDataFilePath;
        console.log('DataManager: Loading scene data from', targetPath);

        try {
            // Fetch the sceneData.js file
            const response = await fetch(targetPath);

            if (!response.ok) {
                throw new Error(`Failed to load sceneData.js: ${response.status} ${response.statusText}`);
            }

            const jsCode = await response.text();

            // Parse the JavaScript code to extract sceneObjects
            // We need to evaluate it safely - we'll use dynamic import
            const moduleUrl = this.createModuleUrl(jsCode);
            const sceneModule = await import(/* @vite-ignore */ moduleUrl);

            if (sceneModule.sceneObjects) {
                this.sceneData = sceneModule.sceneObjects;
                console.log('DataManager: Loaded', Object.keys(this.sceneData).length, 'scene objects');
                this.emit('dataLoaded', { sceneData: this.sceneData });
                return this.sceneData;
            } else {
                throw new Error('sceneData.js does not export sceneObjects');
            }
        } catch (error) {
            console.error('DataManager: Failed to load scene data:', error);
            throw error;
        }
    }

    /**
     * Create a module URL from JavaScript code
     * This allows us to dynamically import the sceneData.js
     */
    private createModuleUrl(jsCode: string): string {
        // Create a blob with the JavaScript code
        const blob = new Blob([jsCode], { type: 'application/javascript' });
        return URL.createObjectURL(blob);
    }

    /**
     * Build Three.js scene from sceneData
     */
    public async buildSceneFromData(sceneData: SceneDataFormat): Promise<void> {
        console.log('DataManager: Building scene from data');

        const scene = this.editorManager.scene;

        // Clear existing scene (except grid, lights, etc.)
        const objectsToRemove: THREE.Object3D[] = [];
        scene.traverse((object) => {
            if (object instanceof THREE.Mesh && object.name !== 'Grid') {
                objectsToRemove.push(object);
            }
        });

        objectsToRemove.forEach(obj => {
            scene.remove(obj);
            if (obj instanceof THREE.Mesh) {
                obj.geometry.dispose();
                if (Array.isArray(obj.material)) {
                    obj.material.forEach(mat => mat.dispose());
                } else {
                    obj.material.dispose();
                }
            }
        });

        // Load each scene object
        for (const [objectId, objectData] of Object.entries(sceneData)) {
            try {
                await this.loadSceneObject(objectId, objectData);
            } catch (error) {
                console.error(`DataManager: Failed to load object ${objectId}:`, error);
            }
        }

        console.log('DataManager: Scene built successfully');
        this.emit('sceneBuilt', { sceneData });
    }

    /**
     * Load a single scene object
     */
    private async loadSceneObject(_objectId: string, objectData: SceneObjectData): Promise<void> {

        switch (objectData.type) {
            case 'splat':
                // TODO: Load Gaussian Splat using Spark.js
                console.log(`DataManager: Splat loading not yet implemented: ${objectData.path}`);
                // For now, create a placeholder
                this.createPlaceholder(objectData);
                break;

            case 'gltf':
                // TODO: Load GLTF model
                console.log(`DataManager: GLTF loading not yet implemented: ${objectData.path}`);
                this.createPlaceholder(objectData);
                break;

            case 'primitive':
                // Create primitive object
                this.createPrimitive(objectData);
                break;

            default:
                console.warn(`DataManager: Unknown object type: ${objectData.type}`);
                this.createPlaceholder(objectData);
        }
    }

    /**
     * Create a placeholder object for assets not yet loaded
     */
    private createPlaceholder(objectData: SceneObjectData): void {
        const scene = this.editorManager.scene;

        // Create a box as placeholder
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshStandardMaterial({
            color: 0x00ff00,
            wireframe: true
        });
        const mesh = new THREE.Mesh(geometry, material);

        // Set transform
        mesh.position.set(objectData.position.x, objectData.position.y, objectData.position.z);
        mesh.rotation.set(objectData.rotation.x, objectData.rotation.y, objectData.rotation.z);

        // Handle scale
        if (typeof objectData.scale === 'number') {
            mesh.scale.set(objectData.scale, objectData.scale, objectData.scale);
        } else {
            mesh.scale.set(objectData.scale.x, objectData.scale.y, objectData.scale.z);
        }

        mesh.name = objectData.id;
        mesh.userData.objectData = objectData;

        scene.add(mesh);
    }

    /**
     * Create a primitive object
     */
    private createPrimitive(objectData: SceneObjectData): void {
        const scene = this.editorManager.scene;

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

        // Set transform
        mesh.position.set(objectData.position.x, objectData.position.y, objectData.position.z);
        mesh.rotation.set(objectData.rotation.x, objectData.rotation.y, objectData.rotation.z);

        // Handle scale
        if (typeof objectData.scale === 'number') {
            mesh.scale.set(objectData.scale, objectData.scale, objectData.scale);
        } else {
            mesh.scale.set(objectData.scale.x, objectData.scale.y, objectData.scale.z);
        }

        mesh.name = objectData.id;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.objectData = objectData;

        scene.add(mesh);
    }

    /**
     * Convert Three.js scene to sceneData format
     */
    public exportSceneToData(): SceneDataFormat {
        console.log('DataManager: Exporting scene to data format');

        const scene = this.editorManager.scene;
        const sceneData: SceneDataFormat = {};

        scene.traverse((object) => {
            // Skip helper objects and lights
            if (object.name === 'Grid' ||
                object.name === 'Axes' ||
                object.name === 'AmbientLight' ||
                object.name === 'DirectionalLight' ||
                object instanceof THREE.GridHelper ||
                object instanceof THREE.AxesHelper ||
                object instanceof THREE.Light) {
                return;
            }

            // Only process meshes
            if (object instanceof THREE.Mesh) {
                const objectData: SceneObjectData = {
                    id: object.name || `object_${object.uuid}`,
                    type: 'primitive', // Default to primitive for now
                    description: object.userData.description || object.name,
                    position: {
                        x: parseFloat(object.position.x.toFixed(6)),
                        y: parseFloat(object.position.y.toFixed(6)),
                        z: parseFloat(object.position.z.toFixed(6))
                    },
                    rotation: {
                        x: parseFloat(object.rotation.x.toFixed(6)),
                        y: parseFloat(object.rotation.y.toFixed(6)),
                        z: parseFloat(object.rotation.z.toFixed(6))
                    },
                    scale: {
                        x: parseFloat(object.scale.x.toFixed(6)),
                        y: parseFloat(object.scale.y.toFixed(6)),
                        z: parseFloat(object.scale.z.toFixed(6))
                    }
                };

                // Preserve existing object data if available
                if (object.userData.objectData) {
                    Object.assign(objectData, object.userData.objectData);
                }

                // Preserve material properties
                if (object.material instanceof THREE.MeshStandardMaterial) {
                    objectData.options = objectData.options || {};
                    objectData.options.envMap = objectData.options.envMap || {};
                    objectData.options.envMap.metalness = object.material.metalness;
                    objectData.options.envMap.roughness = object.material.roughness;
                }

                sceneData[objectData.id] = objectData;
            }
        });

        console.log('DataManager: Exported', Object.keys(sceneData).length, 'objects');
        return sceneData;
    }

    /**
     * Write sceneData to JavaScript file format
     * This generates the exact format expected by sceneData.js
     */
    public async writeSceneData(sceneData: SceneDataFormat, _filePath?: string): Promise<string> {
        console.log('DataManager: Writing scene data to file');

        try {
            // Generate JavaScript code in sceneData.js format
            let jsCode = this.generateSceneDataJS(sceneData);

            // In a real implementation, this would write to the file system
            // For now, we'll emit an event with the code
            this.emit('dataWritten', { sceneData, jsCode });

            console.log('DataManager: Scene data written (code emitted)');
            return jsCode;
        } catch (error) {
            console.error('DataManager: Failed to write scene data:', error);
            throw error;
        }
    }

    /**
     * Generate JavaScript code in sceneData.js format
     */
    private generateSceneDataJS(sceneData: SceneDataFormat): string {
        let code = `/**
 * Scene Data Structure
 *
 * Defines scene objects like splat meshes and GLTF models.
 *
 * Auto-generated by Shadow Web Editor
 * Generated: ${new Date().toISOString()}
 */

import { GAME_STATES } from "./gameData.js";

export const sceneObjects = {
`;

        for (const [objectId, objectData] of Object.entries(sceneData)) {
            code += `  ${objectId}: {\n`;
            code += `    id: "${objectData.id}",\n`;
            code += `    type: "${objectData.type}",\n`;

            if (objectData.path) {
                code += `    path: "${objectData.path}",\n`;
            }

            if (objectData.description) {
                code += `    description: "${objectData.description}",\n`;
            }

            // Position
            code += `    position: { x: ${objectData.position.x}, y: ${objectData.position.y}, z: ${objectData.position.z} },\n`;

            // Rotation
            code += `    rotation: { x: ${objectData.rotation.x}, y: ${objectData.rotation.y}, z: ${objectData.rotation.z} },\n`;

            // Scale
            if (typeof objectData.scale === 'number') {
                code += `    scale: ${objectData.scale},\n`;
            } else {
                code += `    scale: { x: ${objectData.scale.x}, y: ${objectData.scale.y}, z: ${objectData.scale.z} },\n`;
            }

            // Optional properties
            if (objectData.priority !== undefined) {
                code += `    priority: ${objectData.priority},\n`;
            }

            if (objectData.preload !== undefined) {
                code += `    preload: ${objectData.preload},\n`;
            }

            if (objectData.criteria) {
                code += `    criteria: ${JSON.stringify(objectData.criteria, null, 6)},\n`;
            }

            if (objectData.options) {
                code += `    options: ${JSON.stringify(objectData.options, null, 6)},\n`;
            }

            if (objectData.envMapWorldCenter) {
                code += `    envMapWorldCenter: { x: ${objectData.envMapWorldCenter.x}, y: ${objectData.envMapWorldCenter.y}, z: ${objectData.envMapWorldCenter.z} },\n`;
            }

            if (objectData.animations) {
                code += `    animations: ${JSON.stringify(objectData.animations, null, 6)},\n`;
            }

            if (objectData.zone) {
                code += `    zone: "${objectData.zone}",\n`;
            }

            code += `  },\n`;
        }

        code += `};

export default sceneObjects;
`;

        return code;
    }

    /**
     * Get scene data
     */
    public getSceneData(): SceneDataFormat {
        return this.sceneData;
    }

    /**
     * Update scene data from Three.js scene
     */
    public async updateSceneData(): Promise<void> {
        console.log('DataManager: Updating scene data from Three.js scene');
        this.sceneData = this.exportSceneToData();
        this.emit('dataUpdated', { sceneData: this.sceneData });
    }

    /**
     * Enable hot reload for sceneData.js
     *
     * @param enabled - Enable or disable hot reload
     * @param autoReload - Automatically reload scene when file changes
     */
    public async enableHotReload(enabled: boolean = true, autoReload: boolean = true): Promise<void> {
        this.hotReloadEnabled = enabled;

        if (enabled) {
            console.log('DataManager: Enabling hot reload for', this.sceneDataFilePath);

            // Setup file watcher
            if (autoReload) {
                await this.fileWatcher.watch(this.sceneDataFilePath, async (file) => {
                    console.log('DataManager: Scene file modified, reloading...');

                    // Show reload prompt
                    this.emit('externalFileChange', { file });

                    // Auto-reload after debounce
                    try {
                        await this.loadSceneData(file);
                        await this.buildSceneFromData(this.sceneData);

                        this.emit('hotReloadComplete', { sceneData: this.sceneData });
                        console.log('DataManager: Hot reload complete');
                    } catch (error) {
                        console.error('DataManager: Hot reload failed:', error);
                        this.emit('hotReloadError', { error });
                    }
                });
            }

            this.emit('hotReloadEnabled', { filePath: this.sceneDataFilePath });
        } else {
            console.log('DataManager: Disabling hot reload');
            this.fileWatcher.unwatch(this.sceneDataFilePath);
            this.emit('hotReloadDisabled', { filePath: this.sceneDataFilePath });
        }
    }

    /**
     * Reload scene from file
     */
    public async reloadScene(): Promise<void> {
        console.log('DataManager: Reloading scene from file');

        try {
            await this.loadSceneData(this.sceneDataFilePath);
            await this.buildSceneFromData(this.sceneData);

            this.emit('sceneReloaded', { sceneData: this.sceneData });
            console.log('DataManager: Scene reloaded successfully');
        } catch (error) {
            console.error('DataManager: Failed to reload scene:', error);
            throw error;
        }
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
                    console.error(`DataManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }
}

export default DataManager;
