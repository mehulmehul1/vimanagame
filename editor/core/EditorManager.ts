import * as THREE from 'three';
// @ts-ignore - OrbitControls import path may vary by Three.js version
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer } from '@sparkjsdev/spark';

/**
 * EditorManager - Core singleton class for Shadow Web Editor
 *
 * This class manages:
 * - Three.js scene, camera, and renderer
 * - Spark.js Gaussian Splatting renderer
 * - Object selection and manipulation
 * - Play/Edit mode switching
 * - Event emission
 *
 * CRITICAL: Do not modify the src/ directory - this is editor-only code
 */
class EditorManager {
    // Singleton instance
    private static instance: EditorManager;

    // Three.js core objects
    public scene: THREE.Scene;
    public camera: THREE.PerspectiveCamera;
    public renderer: THREE.WebGLRenderer;
    public canvas: HTMLCanvasElement;

    // Spark.js renderer
    public sparkRenderer: SparkRenderer | null = null;

    // Camera controls
    public orbitControls: OrbitControls | null = null;

    // Editor state
    public isInitialized: boolean = false;
    public isPlaying: boolean = false;
    public selectedObject: THREE.Object3D | null = null;

    // Render loop
    private animationFrameId: number | null = null;
    private isRenderLoopRunning: boolean = false;

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    // Canvas container
    private canvasContainer: HTMLElement | null = null;

    /**
     * Private constructor for singleton pattern
     */
    private constructor() {
        // Initialize Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);

        // Initialize camera
        this.camera = new THREE.PerspectiveCamera(
            75, // FOV
            window.innerWidth / window.innerHeight, // Aspect ratio
            0.1, // Near
            1000 // Far
        );
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);

        // Create canvas
        this.canvas = document.createElement('canvas');

        // Initialize WebGL renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;

        console.log('EditorManager: Constructor complete');
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): EditorManager {
        if (!EditorManager.instance) {
            EditorManager.instance = new EditorManager();
        }
        return EditorManager.instance;
    }

    /**
     * Initialize the editor manager
     * Call this after creating the editor UI
     */
    public async initialize(container: HTMLElement): Promise<void> {
        if (this.isInitialized) {
            console.warn('EditorManager already initialized');
            return;
        }

        console.log('EditorManager: Initializing...');

        try {
            // Store container reference
            this.canvasContainer = container;

            // Add canvas to container
            if (this.canvasContainer) {
                this.canvasContainer.appendChild(this.canvas);
            }

            // Add basic scene elements
            this.setupBasicScene();

            // Setup OrbitControls for camera manipulation
            this.orbitControls = new OrbitControls(this.camera, this.canvas);
            this.orbitControls.enableDamping = true;
            this.orbitControls.dampingFactor = 0.05;
            this.orbitControls.screenSpacePanning = true;
            this.orbitControls.minDistance = 0.1;
            this.orbitControls.maxDistance = 500;
            this.orbitControls.maxPolarAngle = Math.PI; // Allow full vertical rotation
            console.log('EditorManager: OrbitControls initialized');

            // Initialize Spark.js (optional, may fail if not supported)
            try {
                // Note: SparkRenderer API may vary - adjust constructor as needed
                // this.sparkRenderer = new SparkRenderer(this.renderer);
                console.log('EditorManager: Spark.js initialization skipped (to be integrated when needed)');
            } catch (sparkError) {
                console.warn('EditorManager: Spark.js initialization failed:', sparkError);
                this.sparkRenderer = null;
            }

            // Setup resize handler
            window.addEventListener('resize', this.handleResize);

            // Start render loop
            this.startRenderLoop();

            this.isInitialized = true;
            console.log('EditorManager: Initialization complete');

            this.emit('initialized');
        } catch (error) {
            console.error('EditorManager: Initialization failed:', error);
            throw error;
        }
    }

    /**
     * Setup basic scene elements (grid, lights, etc.)
     */
    private setupBasicScene(): void {
        // Add grid helper
        const gridHelper = new THREE.GridHelper(10, 10, 0x4d4d4d, 0x2d2d2d);
        gridHelper.name = 'Grid';
        this.scene.add(gridHelper);

        // Add axes helper
        const axesHelper = new THREE.AxesHelper(5);
        axesHelper.name = 'Axes';
        this.scene.add(axesHelper);

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        ambientLight.name = 'AmbientLight';
        this.scene.add(ambientLight);

        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7.5);
        directionalLight.castShadow = true;
        directionalLight.name = 'DirectionalLight';
        this.scene.add(directionalLight);

        console.log('EditorManager: Basic scene setup complete');
    }

    /**
     * Start the render loop
     */
    private startRenderLoop(): void {
        if (this.isRenderLoopRunning) {
            return;
        }

        this.isRenderLoopRunning = true;
        const render = () => {
            if (!this.isRenderLoopRunning) {
                return;
            }

            // Update OrbitControls
            if (this.orbitControls) {
                this.orbitControls.update();
            }

            // Update Spark.js if available
            if (this.sparkRenderer && this.isPlaying) {
                // Spark.js handles its own rendering during play mode
            } else {
                // Standard Three.js rendering
                this.renderer.render(this.scene, this.camera);
            }

            this.animationFrameId = requestAnimationFrame(render);
        };

        render();
        console.log('EditorManager: Render loop started');
    }

    /**
     * Stop the render loop
     */
    private stopRenderLoop(): void {
        this.isRenderLoopRunning = false;
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        console.log('EditorManager: Render loop stopped');
    }

    /**
     * Handle window resize
     */
    private handleResize = (): void => {
        if (!this.canvasContainer) return;

        const width = this.canvasContainer.clientWidth;
        const height = this.canvasContainer.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);

        this.emit('resize', { width, height });
    };

    /**
     * Enter play mode
     */
    public enterPlayMode(): void {
        if (this.isPlaying) {
            console.warn('EditorManager: Already in play mode');
            return;
        }

        console.log('EditorManager: Entering play mode');
        this.isPlaying = true;

        // Disable OrbitControls in play mode
        if (this.orbitControls) {
            this.orbitControls.enabled = false;
        }

        // TODO: Pause editor systems
        // TODO: Resume game systems from src/

        this.emit('playModeEntered');
    }

    /**
     * Exit play mode
     */
    public exitPlayMode(): void {
        if (!this.isPlaying) {
            console.warn('EditorManager: Not in play mode');
            return;
        }

        console.log('EditorManager: Exiting play mode');
        this.isPlaying = false;

        // Enable OrbitControls in edit mode
        if (this.orbitControls) {
            this.orbitControls.enabled = true;
        }

        // TODO: Resume editor systems
        // TODO: Pause game systems from src/

        this.emit('playModeExited');
    }

    /**
     * Select an object
     */
    public selectObject(object: THREE.Object3D | null): void {
        const previous = this.selectedObject;
        this.selectedObject = object;

        console.log('EditorManager: Object selected:', object?.name || 'null');

        this.emit('selectionChanged', {
            previous,
            current: object
        });
    }

    /**
     * Create a primitive object
     */
    public createPrimitive(type: 'box' | 'sphere' | 'plane' | 'cone' | 'cylinder'): THREE.Object3D {
        let geometry: THREE.BufferGeometry;
        let material = new THREE.MeshStandardMaterial({ color: 0x5c9aff });

        switch (type) {
            case 'box':
                geometry = new THREE.BoxGeometry(1, 1, 1);
                break;
            case 'sphere':
                geometry = new THREE.SphereGeometry(0.5, 32, 32);
                break;
            case 'plane':
                geometry = new THREE.PlaneGeometry(2, 2);
                material.side = THREE.DoubleSide;
                break;
            case 'cone':
                geometry = new THREE.ConeGeometry(0.5, 1, 32);
                break;
            case 'cylinder':
                geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
                break;
            default:
                geometry = new THREE.BoxGeometry(1, 1, 1);
        }

        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `${type.charAt(0).toUpperCase() + type.slice(1)}`;
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        this.scene.add(mesh);
        this.selectObject(mesh);

        console.log('EditorManager: Created primitive', type, mesh.name);
        this.emit('objectCreated', { object: mesh, type });

        return mesh;
    }

    /**
     * Delete an object
     */
    public deleteObject(object: THREE.Object3D): void {
        if (!object) return;

        this.scene.remove(object);

        // Clean up geometry and material
        if (object instanceof THREE.Mesh) {
            object.geometry.dispose();
            if (Array.isArray(object.material)) {
                object.material.forEach(mat => mat.dispose());
            } else {
                object.material.dispose();
            }
        }

        if (this.selectedObject === object) {
            this.selectObject(null);
        }

        console.log('EditorManager: Deleted object', object.name);
        this.emit('objectDeleted', { object });
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
     * Emit event (public for use by other managers)
     */
    public emit(eventName: string, data?: any): void {
        const listeners = this.eventListeners.get(eventName);
        if (listeners) {
            listeners.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`EditorManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Destroy the editor manager and clean up
     */
    public destroy(): void {
        console.log('EditorManager: Destroying...');

        this.stopRenderLoop();

        window.removeEventListener('resize', this.handleResize);

        // Clean up Spark.js
        if (this.sparkRenderer) {
            // Note: SparkRenderer may not have dispose method
            // this.sparkRenderer.dispose();
            this.sparkRenderer = null;
        }

        // Clean up Three.js
        this.scene.traverse((object) => {
            if (object instanceof THREE.Mesh) {
                object.geometry.dispose();
                if (Array.isArray(object.material)) {
                    object.material.forEach(mat => mat.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });

        this.renderer.dispose();

        // Remove canvas from DOM
        if (this.canvasContainer && this.canvas.parentElement === this.canvasContainer) {
            this.canvasContainer.removeChild(this.canvas);
        }

        this.eventListeners.clear();
        this.isInitialized = false;

        console.log('EditorManager: Destroyed');
    }
}

export default EditorManager;
