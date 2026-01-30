import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VortexSystem } from '../entities/VortexSystem';
import { WaterMaterial } from '../entities/WaterMaterial';
import { FeedbackManager } from '../entities/FeedbackManager';
import { JellyManager } from '../entities/JellyManager';
import { PatientJellyManager } from '../entities/PatientJellyManager';
import { HarmonyChord } from '../audio/HarmonyChord';
import { VortexActivationController } from '../entities/VortexActivationController';
import { ShellManager } from '../entities/ShellManager';
import { ShellUIOverlay } from '../ui/ShellUIOverlay';
import { WhiteFlashEnding } from '../entities/WhiteFlashEnding';
import { SummonRing } from '../entities/SummonRing';
import { TeachingBeam } from '../entities/TeachingBeam';
import { StringHighlight } from '../entities/StringHighlight';
import { JellyLabelManager } from '../entities/JellyLabelManager';
import { NoteVisualizer } from '../entities/NoteVisualizer';
import { HarpCameraController, HarpInteractionState } from '../entities/HarpCameraController';
import { SynchronizedSplashEffect } from '../entities/SynchronizedSplashEffect';

// WaterBall fluid simulation system
import { MLSMPMSimulator } from '../systems/fluid';
import { DepthThicknessRenderer } from '../systems/fluid/render/DepthThicknessRenderer';
import { FluidSurfaceRenderer } from '../systems/fluid/render/FluidSurfaceRenderer';
import { SphereConstraintAnimator } from '../systems/fluid/animation/SphereConstraintAnimator';
import { HarpWaterInteraction, createHarpInteractionSystem, PlayerWaterInteraction, PlayerWakeEffect } from '../systems/fluid/interaction';
import { setupDebugViews } from '../systems/fluid';

/**
 * HarpRoom - Main scene controller for the Archive of Voices chamber
 *
 * Manages the scene setup, water surface, vortex system, feedback systems,
 * and coordinates all visual effects for the music room puzzle.
 */
export class HarpRoom {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: THREE.WebGLRenderer;
    private clock: THREE.Clock;

    // Core systems
    private vortexSystem: VortexSystem;
    private vortexPosition: THREE.Vector3 = new THREE.Vector3();  // Store vortex world position
    private waterMaterial?: WaterMaterial;
    private arenaFloor?: THREE.Mesh;
    private waterSurfaceY: number = 0.0;  // Actual water surface Y from ArenaFloor bounding box
    private platformMesh?: THREE.Mesh;  // The platform where harp sits - moves to vortex

    // WaterBall fluid simulation system
    private fluidSimulator?: InstanceType<typeof MLSMPMSimulator>;
    private fluidDepthRenderer?: DepthThicknessRenderer;
    private fluidSurfaceRenderer?: FluidSurfaceRenderer;
    private sphereAnimator?: SphereConstraintAnimator;
    private harpWaterInteraction?: HarpWaterInteraction;
    private playerWaterInteraction?: PlayerWaterInteraction;
    private wakeEffect?: PlayerWakeEffect;
    private renderUniformBuffer?: GPUBuffer;
    private fluidEnabled: boolean = false;  // DISABLED: WebGPU incompatible with existing GLSL shaders

    private feedbackManager?: FeedbackManager;
    private jellyManager?: JellyManager;
    private patientJellyManager?: PatientJellyManager;
    private harmonyChord?: HarmonyChord;
    private activationController?: VortexActivationController;
    private shellUI?: ShellUIOverlay;
    private whiteFlash?: WhiteFlashEnding;

    // NEW: Visual enhancement systems
    private summonRings: Map<number, SummonRing> = new Map();
    private teachingBeam?: TeachingBeam;
    private stringHighlight?: StringHighlight;
    private jellyLabels?: JellyLabelManager;
    private noteVisualizer?: NoteVisualizer;
    private cameraController?: HarpCameraController;
    private splashEffect?: SynchronizedSplashEffect;

    // String positions cache (for visual effects)
    private stringPositions: THREE.Vector3[] = [];

    // Harp position for camera lock-on
    private harpPosition: THREE.Vector3 = new THREE.Vector3(0, 0, 0);

    // Scene state
    private initialized: boolean = false;
    private duetProgress: number = 0;

    // Game state management
    private gameManager?: any; // GameManager from shadowczar engine
    private splatRenderer?: any; // VisionarySplatRenderer

    // Visual click feedback
    private clickMarker?: THREE.Mesh;

    // Debug hitboxes (stored for toggle)
    private hitboxMeshes: THREE.Mesh[] = [];
    private debugHitboxes: boolean = false;

    // Callbacks
    private onCompleteCallback?: () => void;

    constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera, renderer: any, gameManager?: any, splatRenderer?: any) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        this.gameManager = gameManager;
        this.splatRenderer = splatRenderer;
        this.clock = new THREE.Clock();

        // Create vortex system
        this.vortexSystem = new VortexSystem(2000);
        this.scene.add(this.vortexSystem);

        // Create visual click marker (red sphere)
        const markerGeo = new THREE.SphereGeometry(0.2, 16, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true });
        this.clickMarker = new THREE.Mesh(markerGeo, markerMat);
        this.clickMarker.visible = false;
        this.scene.add(this.clickMarker);

        // Event listeners
        this.setupEventListeners();

        // Listen to game state changes if gameManager is available
        if (this.gameManager) {
            this.gameManager.on('state:changed', this.handleGameStateChanged);
        }
    }

    /**
     * Create perspective camera
     */
    private createCamera(): THREE.PerspectiveCamera {
        const camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        camera.position.set(0, 2, 5);
        camera.lookAt(0, 0, 0);
        return camera;
    }

    /**
     * Create WebGL renderer
     */
    private createRenderer(canvas: HTMLCanvasElement): THREE.WebGLRenderer {
        const renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true,
            alpha: true
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;
        return renderer;
    }

    /**
     * Setup scene lighting
     */
    private setupLighting(): void {
        // Ambient light
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambient);

        // Main directional light
        const mainLight = new THREE.DirectionalLight(0xffffff, 1.0);
        mainLight.position.set(5, 10, 5);
        this.scene.add(mainLight);

        // Fill light with cyan tint
        const fillLight = new THREE.DirectionalLight(0x00ffff, 0.3);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);

        // Point light near vortex position
        const vortexLight = new THREE.PointLight(0x00ffff, 0.5, 10);
        vortexLight.position.set(0, 1, 2);
        this.scene.add(vortexLight);
    }

    /**
     * Setup window event listeners
     */
    private setupEventListeners(): void {
        window.addEventListener('resize', this.onResize.bind(this));
    }

    /**
     * Handle window resize
     */
    private onResize(): void {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    /**
     * Helper to find an object anywhere in the scene hierarchy by name
     * THREE.Object3D.getObjectByName only searches direct children
     */
    private findObjectByName(scene: THREE.Object3D, name: string): THREE.Object3D | null {
        let result: THREE.Object3D | null = null;
        console.log(`[HarpRoom] Searching for "${name}"...`);
        scene.traverse((child) => {
            if (!result && child.name === name) {
                result = child;
                console.log(`[HarpRoom] ✅ Found "${name}" at position:`, child.position);
            }
        });
        if (!result) {
            console.log(`[HarpRoom] ❌ "${name}" not found. Dumping scene structure:`);
            let count = 0;
            scene.traverse((child) => {
                if (child instanceof THREE.Mesh || child instanceof THREE.Group || child.type === 'Object3D') {
                    count++;
                    console.log(`  [${count}] ${child.name || '(unnamed)'} (${child.type}) parent: ${child.parent?.name || '(root)'}`);
                }
            });
        }
        return result;
    }

    /**
     * Initialize the room components using the already loaded scene
     * @param loadedScene - The GLTF scene object or root Object3D
     */
    public async initialize(loadedScene?: THREE.Object3D): Promise<void> {
        if (this.initialized) return;

        try {
            const root = loadedScene || this.scene;
            console.log('[HarpRoom] ========================================');
            console.log('[HarpRoom] Initializing with scene root:', root.name || 'main', 'type:', root.type);
            console.log('[HarpRoom] Scene children count:', root.children.length);
            console.log('[HarpRoom] Scene children:', root.children.map(c => `"${c.name}" (${c.type})`).join(', '));

            // The GLB might be loaded as a child of this.scene with name like "Scene" or "Scene Collection"
            // Try to find the actual GLB root by looking for common names
            let glbRoot = root;
            const possibleNames = ['Scene', 'Scene Collection', 'Root', 'music_room', 'musicroom'];

            for (const name of possibleNames) {
                const found = this.findObjectByName(root, name);
                if (found) {
                    console.log(`[HarpRoom] ✅ Found GLB root: "${name}"`);
                    glbRoot = found;
                    break;
                }
            }

            // If no named root found, use the first Group child (likely the GLB content)
            if (glbRoot === root && root.children.length > 0) {
                const firstGroup = root.children.find(c => c instanceof THREE.Group || c.type === 'Group' || c.type === 'Scene');
                if (firstGroup) {
                    console.log(`[HarpRoom] Using first group as GLB root: "${firstGroup.name}" (${firstGroup.type})`);
                    glbRoot = firstGroup;
                }
            }

            // Dump ALL objects in the glbRoot to help debug
            console.log('[HarpRoom] Dumping all objects in glbRoot:');
            let objCount = 0;
            glbRoot.traverse((child) => {
                objCount++;
                const meshInfo = (child instanceof THREE.Mesh) ? ' [MESH]' : '';
                const nameInfo = child.name ? `"${child.name}"` : '(unnamed)';
                console.log(`  [${objCount}] ${nameInfo} (${child.type})${meshInfo}`);
            });
            console.log(`[HarpRoom] Total objects in glbRoot: ${objCount}`);

            // Find and apply water material to ArenaFloor (search recursively)
            this.arenaFloor = this.findObjectByName(glbRoot, 'ArenaFloor') as THREE.Mesh;
            if (this.arenaFloor) {
                // Detect actual water surface level using Box3 (Three.js bounding box)
                const bbox = new THREE.Box3().setFromObject(this.arenaFloor);
                this.waterSurfaceY = bbox.max.y; // Top of the ArenaFloor = water surface
                console.log(`[HarpRoom] 💧 Water surface Y detected: ${this.waterSurfaceY.toFixed(3)}`);
                console.log(`[HarpRoom] 📦 ArenaFloor bounds: min=${bbox.min.y.toFixed(2)}, max=${bbox.max.y.toFixed(2)}`);

                if (this.fluidEnabled) {
                    // Initialize WaterBall fluid simulation system
                    this.initializeFluidSystem();
                } else {
                    // Fall back to old water material
                    this.waterMaterial = new WaterMaterial();
                    this.arenaFloor.material = this.waterMaterial;
                    this.vortexSystem.setWaterMaterial(this.waterMaterial);
                    console.log('[HarpRoom] ✅ Water material applied to ArenaFloor');
                }
            } else {
                console.warn('[HarpRoom] ⚠️ ArenaFloor not found in scene');
            }

            // Position vortex at the gate (search recursively in glbRoot)
            const vortexGate = this.findObjectByName(glbRoot, 'Vortex_gate');
            if (vortexGate) {
                const worldPos = new THREE.Vector3();
                vortexGate.getWorldPosition(worldPos);
                this.vortexSystem.position.copy(worldPos);
                // Store vortex position for platform ride target
                this.vortexPosition.copy(worldPos);
                // Ensure particles are at local zero
                (this.vortexSystem as any).particles.position.set(0, 0, 0);
                console.log(`[HarpRoom] ✅ Vortex positioned at: ${worldPos.x}, ${worldPos.y}, ${worldPos.z}`);
            } else {
                console.warn('[HarpRoom] ⚠️ Vortex_gate not found in scene');
            }

            // Find the platform mesh (where harp sits - this will move to vortex)
            // From GLB analysis: Mesh #3 is "Platform" (Cube, 232 vertices)
            this.platformMesh = this.findObjectByName(glbRoot, 'Platform') as THREE.Mesh;
            if (this.platformMesh) {
                console.log(`[HarpRoom] ✅ Found platform mesh "Platform" at position:`, this.platformMesh.position);
            } else {
                console.warn('[HarpRoom] ⚠️ Platform mesh not found!');
            }

            // Expose platform finder to window for debugging
            (window as any).findPlatform = () => {
                const results: { name: string; pos: THREE.Vector3 }[] = [];
                glbRoot.traverse((child) => {
                    if (child instanceof THREE.Mesh) {
                        results.push({
                            name: child.name || '(unnamed)',
                            pos: child.position.clone()
                        });
                    }
                });
                console.table(results.map(r => ({
                    name: r.name,
                    x: r.pos.x.toFixed(2),
                    y: r.pos.y.toFixed(2),
                    z: r.pos.z.toFixed(2)
                })));
                return results;
            };

            // Find harp strings for interaction setup (use glbRoot)
            this.setupHarpStrings(glbRoot);

            // Create all managers and core game logic
            this.setupManagers();

            // Update jelly positions to be near actual harp strings
            this.updateJellyPositions();

            this.initialized = true;
            console.log('[HarpRoom] ========================================');
        } catch (error) {
            console.error('[HarpRoom] Failed to initialize:', error);
            // Create fallback geometry if initialization fails
            this.createFallbackScene();
        }
    }

    /**
     * Load GLB file
     */
    private loadGLB(path: string): Promise<any> {
        return new Promise((resolve, reject) => {
            const loader = new GLTFLoader();
            loader.load(
                path,
                (gltf) => resolve(gltf),
                undefined,
                reject
            );
        });
    }

    /**
     * Setup harp string interaction
     */
    private setupHarpStrings(scene: THREE.Object3D): void {
        console.log('[HarpRoom] setupHarpStrings called with scene:', scene.name, 'type:', scene.type);
        console.log('[HarpRoom] Scene children count:', scene.children.length);
        console.log('[HarpRoom] Scene is:', scene);

        // First, let's see what objects contain "String" or "harp" in their name
        console.log('[HarpRoom] Searching for all objects with "String" or "harp" in name...');
        const stringCandidates: { name: string; obj: THREE.Object3D }[] = [];
        scene.traverse((child) => {
            const nameLower = (child.name || '').toLowerCase();
            if (nameLower.includes('string') || nameLower.includes('harp')) {
                stringCandidates.push({ name: child.name, obj: child });
                console.log(`  Found: "${child.name}" (type: ${child.type})`);
            }
        });

        // Find all string objects
        // The GLB meshes can be named in various ways:
        // - "String 1" through "String 6" (with space)
        // - "String.1" through "String.6" (with dot)
        // - "String_1" through "String_6" (with underscore) <- Actual GLB naming
        const stringNameVariants = [
            ['String 1', 'String 2', 'String 3', 'String 4', 'String 5', 'String 6'],
            ['String.1', 'String.2', 'String.3', 'String.4', 'String.5', 'String.6'],
            ['String_1', 'String_2', 'String_3', 'String_4', 'String_5', 'String_6'],
        ];

        console.log('[HarpRoom] Attempting to find harp strings using variants:', stringNameVariants);

        for (let i = 0; i < 6; i++) {
            let foundString = false;
            for (let variant = 0; variant < stringNameVariants.length; variant++) {
                const name = stringNameVariants[variant][i];
                console.log(`[HarpRoom] Looking for string ${i} with name: "${name}"`);
                const stringObj = this.findObjectByName(scene, name);

                if (stringObj) {
                    console.log(`[HarpRoom] ✅ FOUND string object for index ${i}: "${name}"`);
                    // Store reference for raycasting
                    stringObj.userData.isHarpString = true;
                    stringObj.userData.stringIndex = i;
                    console.log(`[HarpRoom] ✅ Setup string: "${name}" at index ${i}, local position:`, stringObj.position);

                    // Get WORLD position of the string (for proper hitbox placement)
                    const worldPos = new THREE.Vector3();
                    stringObj.getWorldPosition(worldPos);
                    console.log(`[HarpRoom] ✅ String ${i} WORLD position:`, worldPos);

                    // Store world position for jelly positioning
                    stringObj.userData.worldPosition = worldPos.clone();

                    // Add invisible hitbox for easier clicking
                    // Harp strings are thin and hard to click, so we add a larger invisible box
                    // Make hitbox narrower (0.25) to avoid overlap between closely spaced strings
                    const hitbox = new THREE.Mesh(
                        new THREE.BoxGeometry(0.25, 2, 0.25), // Narrower to prevent overlap
                        new THREE.MeshBasicMaterial({
                            color: 0xff00ff,
                            wireframe: false, // Solid mesh for reliable raycasting
                            transparent: true,
                            opacity: 0.05, // Nearly invisible but still raycastable
                            visible: true,
                            depthWrite: false // Don't affect depth buffer
                        })
                    );

                    // Position hitbox at the WORLD position of the string
                    // Add hitbox directly to scene (not as child) to avoid transform issues
                    hitbox.position.copy(worldPos);
                    hitbox.userData.isHarpString = true;
                    hitbox.userData.stringIndex = i;
                    hitbox.userData.isHitbox = true; // Mark as hitbox for debugging

                    console.log(`[HarpRoom] Adding hitbox to scene...`);
                    console.log(`[HarpRoom] Hitbox position:`, hitbox.position);
                    console.log(`[HarpRoom] Scene before add:`, this.scene);
                    this.scene.add(hitbox);
                    console.log(`[HarpRoom] Scene after add:`, this.scene);
                    console.log(`[HarpRoom] Hitbox parent:`, hitbox.parent);
                    console.log(`[HarpRoom] Hitbox visible:`, hitbox.visible);
                    console.log(`[HarpRoom] Hitbox in scene children:`, this.scene.children.includes(hitbox));

                    // Store reference for debug toggle
                    this.hitboxMeshes.push(hitbox);

                    console.log(`[HarpRoom] ✅ Added invisible hitbox for string ${i} at world position:`, worldPos);
                    console.log(`[HarpRoom] Total hitboxes created so far: ${this.hitboxMeshes.length}`);
                    foundString = true;
                    break;
                } else {
                    console.log(`[HarpRoom] ❌ String "${name}" NOT found`);
                }
            }

            if (!foundString) {
                console.warn(`[HarpRoom] ❌ String NOT found for index ${i} (tried all naming variants)`);
            }
        }

        // Final count
        let setupCount = 0;
        this.scene.traverse((obj) => {
            if (obj.userData && obj.userData.isHarpString) {
                setupCount++;
            }
        });
        console.log(`[HarpRoom] Total harp strings setup: ${setupCount}/6`);
        console.log(`[HarpRoom] Total hitboxes in array: ${this.hitboxMeshes.length}`);
    }

    /**
     * Update jelly positions to be near the actual harp strings
     * This ensures jellies emerge from water near their target strings
     */
    private updateJellyPositions(): void {
        if (!this.jellyManager) {
            console.warn('[HarpRoom] No jellyManager to position');
            return;
        }

        // Find all harp string world positions
        const stringPositions: THREE.Vector3[] = [];
        this.scene.traverse((obj) => {
            if (obj.userData && obj.userData.isHarpString === true && obj.userData.worldPosition) {
                const index = obj.userData.stringIndex;
                if (index !== undefined) {
                    stringPositions[index] = obj.userData.worldPosition;
                }
            }
        });

        if (stringPositions.length !== 6) {
            console.warn(`[HarpRoom] Expected 6 string positions, found ${stringPositions.length}`);
            return;
        }

        // Cache string positions for visual effects
        this.stringPositions = stringPositions;

        // Calculate harp center position for camera
        let centerSum = new THREE.Vector3();
        stringPositions.forEach(pos => centerSum.add(pos));
        this.harpPosition.copy(centerSum).divideScalar(6);

        // Update camera controller with harp position
        if (this.cameraController) {
            this.cameraController.setHarpPosition(this.harpPosition);
        }

        // Update note visualizer with string positions
        if (this.noteVisualizer) {
            this.noteVisualizer.setStringPositions(stringPositions);
        }

        console.log('[HarpRoom] Updating jelly positions to match harp strings:');
        console.log(`[HarpRoom] Harp center at: (${this.harpPosition.x.toFixed(2)}, ${this.harpPosition.y.toFixed(2)}, ${this.harpPosition.z.toFixed(2)})`);
        console.log(`[HarpRoom] Water surface Y: ${this.waterSurfaceY.toFixed(3)}`);

        for (let i = 0; i < 6; i++) {
            const stringPos = stringPositions[i];
            const jelly = this.jellyManager.getJelly(i);
            if (jelly && stringPos) {
                // Position jelly near the string, in front of it (toward camera)
                // Z offset should be negative to be in front of harp (toward player at Z=5)
                const jellyHomePos = new THREE.Vector3(
                    stringPos.x,
                    this.waterSurfaceY,  // Use detected water surface Y
                    stringPos.z - 1.5  // Negative Z = toward camera/player
                );
                // Use setHomePosition to properly set spawn/submerge location
                jelly.setHomePosition(jellyHomePos);
                console.log(`  Jelly ${i}: (${jellyHomePos.x.toFixed(1)}, ${jellyHomePos.y.toFixed(1)}, ${jellyHomePos.z.toFixed(1)})`);
            }
        }

        // Also update JellyManager's water surface reference
        if (this.jellyManager) {
            (this.jellyManager as any).updateWaterSurface(this.waterSurfaceY);
        }
    }

    /**
     * Setup all game managers
     */
    private setupManagers(): void {
        this.feedbackManager = new FeedbackManager(this.camera, this.scene);
        this.jellyManager = new JellyManager();
        this.scene.add(this.jellyManager);

        this.harmonyChord = new HarmonyChord();

        // NEW: Create visual enhancement systems
        this.teachingBeam = new TeachingBeam();
        this.scene.add(this.teachingBeam);

        this.stringHighlight = new StringHighlight(this.scene);

        this.jellyLabels = new JellyLabelManager();
        this.scene.add(this.jellyLabels);

        this.noteVisualizer = new NoteVisualizer();
        this.scene.add(this.noteVisualizer);

        // Create summon rings for each string
        for (let i = 0; i < 6; i++) {
            const ring = new SummonRing();
            this.summonRings.set(i, ring);
            this.scene.add(ring);
        }

        // Create synchronized splash effect
        this.splashEffect = new SynchronizedSplashEffect();
        this.scene.add(this.splashEffect);

        // Create camera controller for harp interaction
        this.cameraController = new HarpCameraController(this.camera);
        this.cameraController.setHarpPosition(this.harpPosition);

        // Set up camera controller state change handlers
        this.cameraController.setOnStateChange((state) => {
            console.log('[HarpRoom] Camera state changed to:', state);
        });

        this.patientJellyManager = new PatientJellyManager(
            this.jellyManager,
            this.harmonyChord,
            this.feedbackManager,
            {
                onNoteComplete: (seq, note) => {
                    console.log(`Note complete: ${seq}-${note}`);
                    if (this.waterMaterial) {
                        this.triggerStringRipple(this.patientJellyManager!.getTargetNote());
                    }
                    // Visualize the note being played correctly
                    this.noteVisualizer?.showNote(note, 0.8);
                },
                onSequenceComplete: (seq) => {
                    console.log(`Sequence complete: ${seq}`);
                    // Play completion chord visualization
                    this.noteVisualizer?.showChord([0, 2, 4], 2.0); // C major chord visual
                    // Update game state when sequence is completed
                    this.updateGameStateOnSequenceComplete();
                },
                onDuetComplete: () => {
                    console.log('Duet complete!');
                    // All required sequences done - trigger full completion
                    if (this.gameManager) {
                        const state = this.gameManager.getState();
                        this.gameManager.setState({
                            currentState: 11, // HARP_COMPLETE
                            harpState: 5, // COMPLETE
                            vortexActivation: 1.0,
                        });
                    }
                },
                onWrongNote: (target, played) => {
                    console.log(`Wrong note: played ${played}, target ${target}`);
                    // Visual feedback for wrong note
                    this.triggerWrongNote(target);
                },
                onDemonstrationStart: (noteIndex) => {
                    console.log(`[HarpRoom] Demonstration start for note ${noteIndex}`);
                    this.onDemonstrationStart(noteIndex);
                },
                onDemonstrationEnd: (noteIndex) => {
                    console.log(`[HarpRoom] Demonstration end for note ${noteIndex}`);
                    this.onDemonstrationEnd(noteIndex);
                },
                onNoteDemonstrated: (noteIndex, seqIdx) => {
                    const jelly = this.jellyManager?.getJelly(noteIndex);
                    if (jelly && this.jellyLabels) {
                        this.jellyLabels.showSequenceLabel(noteIndex, seqIdx, jelly.position);
                    }
                },
                onTurnSignalComplete: () => {
                    console.log('[HarpRoom] Turn signal complete - player can respond now');
                    // Hide all sequence indicators when player's turn starts
                    this.jellyLabels?.hideAll();
                },
                onSynchronizedSplash: (indices) => {
                    console.log('[HarpRoom] Synchronized splash for indices:', indices);
                    if (this.splashEffect) {
                        const positions = indices.map(idx => this.stringPositions[idx]);
                        // Move ripples forward slightly like jellies (Z offset matches updateJellyPositions)
                        const splashPositions = positions.map(pos => {
                            const p = pos.clone();
                            p.z -= 1.5;
                            p.y = this.waterSurfaceY;
                            return p;
                        });
                        this.splashEffect.trigger(splashPositions);
                    }
                }
            }
        );


        // STORY-HARP-101: Set teaching mode
        this.patientJellyManager.setTeachingMode('phrase-first');


        // Start the teaching sequence!
        console.log('[HarpRoom] Starting duet teaching sequence...');
        this.patientJellyManager.start();

        // DEBUG: Spawn a test jelly immediately to verify visibility
        setTimeout(() => {
            if (this.jellyManager) {
                console.log('[HarpRoom] DEBUG: Spawning test jelly at string 0');
                this.spawnJellyWithEffects(0);
            }
        }, 1000);

        this.activationController = new VortexActivationController(
            this.vortexSystem,
            this.waterMaterial!,
            this.platformMesh || null,  // Use the actual platform mesh, not arenaFloor!
            this.scene,
            this.camera,  // Pass camera so player moves with platform
            this.vortexPosition  // Pass vortex position as target for platform ride
        );

        this.shellUI = new ShellUIOverlay();
        this.whiteFlash = new WhiteFlashEnding();
        // Don't add to scene yet - only add when triggered
        // this.scene.add(this.whiteFlash);

        // Connect duet progress to vortex
        window.addEventListener('duet-progress', (e: any) => {
            if (this.activationController) {
                this.activationController.setActivation(e.detail.progress);
            }
        });

        // DEBUG: Expose test function to trigger platform ride
        (window as any).testPlatformRide = () => {
            console.log('[HarpRoom] 🎢 Testing platform ride...');
            if (this.platformMesh) {
                console.log(`   Platform found: "${this.platformMesh.name}"`);
                console.log(`   Start position:`, this.platformMesh.position);
            } else {
                console.error('   ❌ No platform mesh found!');
            }
            if (this.activationController) {
                // Trigger full activation
                this.activationController.setActivation(1.0);
                console.log('   ✅ Activation set to 1.0 - platform should move');
            } else {
                console.error('   ❌ No activationController found!');
            }
        };
    }

    /**
     * Initialize WaterBall MLS-MPM fluid simulation system
     * Replaces the static water plane with real-time particle-based fluid
     */
    private async initializeFluidSystem(): Promise<void> {
        console.log('[HarpRoom] Initializing WaterBall fluid simulation...');

        try {
            // Check if WebGPU is available
            const renderer = this.renderer as any;
            if (!renderer.device) {
                console.warn('[HarpRoom] WebGPU not available, falling back to water material');
                this.fluidEnabled = false;
                this.waterMaterial = new WaterMaterial();
                if (this.arenaFloor) {
                    this.arenaFloor.material = this.waterMaterial.getMaterial();
                }
                return;
            }

            const device = renderer.device;

            // 0. Create render uniform buffer
            this.renderUniformBuffer = device.createBuffer({
                label: 'Fluid Render Uniforms',
                size: 256, // Sufficient for view/proj matrices
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });

            // 1. Create particle physics simulator
            this.fluidSimulator = new MLSMPMSimulator({
                device,
                renderUniformBuffer: this.renderUniformBuffer
            });
            console.log('[HarpRoom] ✅ MLS-MPM Simulator created');

            // 2. Create depth/thickness renderer
            this.fluidDepthRenderer = new DepthThicknessRenderer({
                device,
                canvas: this.renderer.domElement,
                posvelBuffer: this.fluidSimulator.posvelBuffer,
                renderUniformBuffer: this.renderUniformBuffer
            });
            console.log('[HarpRoom] ✅ Depth/Thickness Renderer created');

            // 3. Create fluid surface renderer (final water visual)
            this.fluidSurfaceRenderer = new FluidSurfaceRenderer({
                device,
                canvas: this.renderer.domElement,
                renderUniformBuffer: this.renderUniformBuffer,
                depthTextureView: this.fluidDepthRenderer.getDepthTextureView(),
                thicknessTextureView: this.fluidDepthRenderer.getThicknessTextureView()
            });
            console.log('[HarpRoom] ✅ Fluid Surface Renderer created');

            // 4. Create sphere constraint animator (for tunnel transformation)
            this.sphereAnimator = new SphereConstraintAnimator({
                simulator: this.fluidSimulator,
                initBoxSize: [52, 52, 52] as any // Use default from simulator
            });
            console.log('[HarpRoom] ✅ Sphere Animator created');

            // 5. Create harp-water interaction system
            this.harpWaterInteraction = createHarpInteractionSystem(device);
            console.log('[HarpRoom] ✅ Harp-Water Interaction created');

            // 6. Create player-water interaction
            this.playerWaterInteraction = new PlayerWaterInteraction(this.fluidSimulator, {
                onWaterEntry: () => console.log('[Fluid] Player entered water'),
                onWaterExit: () => console.log('[Fluid] Player exited water'),
            });
            console.log('[HarpRoom] ✅ Player-Water Interaction created');

            // 7. Create wake effect for player movement
            this.wakeEffect = new PlayerWakeEffect({});
            console.log('[HarpRoom] ✅ Wake Effect created');

            // 8. Set up debug views
            // Note: setupDebugViews might need updates, but leaving as is for now if it accepts these types
            setupDebugViews(
                this.fluidSimulator,
                this.fluidDepthRenderer,
                this.fluidSurfaceRenderer,
                this.sphereAnimator,
                this.harpWaterInteraction,
                this.playerWaterInteraction,
                this.wakeEffect
            );
            console.log('[HarpRoom] ✅ Debug views available at window.debugVimana.fluid');

            // 9. Apply fluid surface material to arena floor
            // Note: FluidSurfaceRenderer renders to screen/texture, not directly as material.
            // Disabling direct material assignment for now to prevent errors.
            if (this.arenaFloor && this.fluidSurfaceRenderer) {
                // this.arenaFloor.material = ...; // Not supported by current renderer
                this.arenaFloor.visible = false; // Hide floor, fluid will render on top
                console.log('[HarpRoom] ✅ ArenaFloor hidden (Fluid Surface Renderer handles visualization)');
            }

            console.log('[HarpRoom] 🎉 WaterBall fluid system fully initialized!');
        } catch (error) {
            console.error('[HarpRoom] Failed to initialize fluid system:', error);
            console.log('[HarpRoom] Falling back to water material...');
            this.fluidEnabled = false;
            this.waterMaterial = new WaterMaterial();
            if (this.arenaFloor) {
                this.arenaFloor.material = this.waterMaterial.getMaterial();
            }
        }
    }

    /**
     * Spawn a jelly with full visual effects (summon ring, beam, labels, etc.)
     */
    private spawnJellyWithEffects(stringIndex: number): void {
        if (!this.jellyManager) return;

        // Get string position
        const stringPos = this.stringPositions[stringIndex];
        if (!stringPos) return;

        // Show summon ring first
        const ring = this.summonRings.get(stringIndex);
        if (ring) {
            ring.position.set(stringPos.x, 0.05, stringPos.z);
            ring.play(() => {
                // After ring completes, spawn the jelly
                this.jellyManager?.spawnJelly(stringIndex);
                setTimeout(() => {
                    this.jellyManager?.beginTeaching();

                    // Show additional effects when jelly spawns
                    this.showTeachingEffects(stringIndex);
                }, 300);
            });
        }
    }

    /**
     * Show all teaching effects for a note
     */
    private showTeachingEffects(stringIndex: number): void {
        const stringPos = this.stringPositions[stringIndex];
        if (!stringPos) return;

        const jelly = this.jellyManager?.getJelly(stringIndex);
        if (!jelly) return;

        // Activate teaching beam from jelly to string
        if (this.teachingBeam) {
            this.teachingBeam.activate(jelly.position, stringPos);
        }

        // Highlight the target string
        if (this.stringHighlight) {
            this.stringHighlight.highlightString(stringIndex, stringPos);
        }

        // Show label above jelly
        if (this.jellyLabels) {
            this.jellyLabels.showLabel(stringIndex, jelly.position);
        }

        // Visualize the note
        if (this.noteVisualizer) {
            this.noteVisualizer.showNote(stringIndex, 1.5);
        }
    }

    /**
     * Hide all teaching effects
     */
    private hideTeachingEffects(): void {
        this.teachingBeam?.deactivate();
        this.stringHighlight?.clearAll();
        this.jellyLabels?.hideAll();
    }

    /**
     * Handle demonstration start - trigger all visual effects
     */
    private onDemonstrationStart(noteIndex: number): void {
        const stringPos = this.stringPositions[noteIndex];
        if (!stringPos) {
            console.warn(`[HarpRoom] No string position for note ${noteIndex}`);
            return;
        }

        // Show summon ring at string position
        const ring = this.summonRings.get(noteIndex);
        if (ring) {
            ring.position.set(stringPos.x, 0.05, stringPos.z);
            ring.play();
        }

        // Wait for jelly to spawn, then show additional effects
        setTimeout(() => {
            this.showTeachingEffects(noteIndex);
        }, 700); // After summon ring + spawn animation
    }

    /**
     * Handle demonstration end - hide all teaching effects
     */
    private onDemonstrationEnd(noteIndex: number): void {
        this.hideTeachingEffects();
    }

    /**
     * Handle click interaction
     */
    public onClick(raycaster: THREE.Raycaster): void {
        if (!this.initialized) return;

        // DIAGNOSTIC LOG: Log the raycaster received from main.js
        console.log('[HarpRoom DIAGNOSTIC] onClick called');
        console.log('[HarpRoom DIAGNOSTIC] Raycaster received:', {
            rayOrigin: raycaster.ray.origin,
            rayDirection: raycaster.ray.direction,
            near: raycaster.near,
            far: raycaster.far
        });

        // DIAGNOSTIC LOG: Count all objects in scene
        let sceneObjectCount = 0;
        let harpStringCount = 0;
        let hitboxCount = 0;
        this.scene.traverse((obj) => {
            sceneObjectCount++;
            if (obj.userData && obj.userData.isHarpString) harpStringCount++;
            if (obj.userData && obj.userData.isHitbox) hitboxCount++;
        });
        console.log('[HarpRoom DIAGNOSTIC] Scene objects:', {
            total: sceneObjectCount,
            harpStrings: harpStringCount,
            hitboxes: hitboxCount,
            hitboxMeshesArray: this.hitboxMeshes.length
        });

        // DIAGNOSTIC LOG: Filter out camera from raycast
        // Only intersect with visible objects that are not the camera
        const raycastTargets = this.scene.children.filter(obj => obj !== this.camera);
        console.log('[HarpRoom DIAGNOSTIC] Raycasting against', raycastTargets.length, 'objects (excluding camera)');

        // Use recursive raycast on the entire scene
        const intersects = raycaster.intersectObjects(this.scene.children, true);

        // DIAGNOSTIC LOG: Log intersection results
        console.log('[HarpRoom DIAGNOSTIC] Intersections:', {
            count: intersects.length,
            objects: intersects.map(i => ({
                name: i.object.name,
                visible: i.object.visible,
                isHarpString: i.object.userData.isHarpString,
                isHitbox: i.object.userData.isHitbox,
                stringIndex: i.object.userData.stringIndex,
                distance: i.distance.toFixed(3)
            }))
        });

        // DIAGNOSTIC LOG: Check raycaster near/far configuration
        console.log('[HarpRoom DIAGNOSTIC] Raycaster configuration:', {
            near: raycaster.near,
            far: raycaster.far,
            cameraPosition: this.camera.position,
            rayOrigin: raycaster.ray.origin,
            rayDirection: raycaster.ray.direction
        });

        // Show click marker at hit position (visual feedback)
        if (this.clickMarker && intersects.length > 0) {
            this.clickMarker.visible = true;
            this.clickMarker.position.copy(intersects[0].point);
            // Hide marker after 500ms
            setTimeout(() => {
                if (this.clickMarker) this.clickMarker.visible = false;
            }, 500);
        }

        if (!this.patientJellyManager) {
            console.warn('[HarpRoom] No patientJellyManager available');
            return;
        }

        // Show current duet state for debugging
        const state = this.patientJellyManager.getState();
        console.log('[HarpRoom] Click! Duet state:', state, '| Hit', intersects.length, 'objects');

        if (intersects.length > 0) {
            const hit = intersects[0];
            console.log('[HarpRoom] First hit:', hit.object.name || '(unnamed)', 'type:', hit.object.type, 'userData:', hit.object.userData);
        }

        const stringIntersect = intersects.find(i => i.object.userData.isHarpString);

        if (stringIntersect) {
            const index = stringIntersect.object.userData.stringIndex;
            console.log('[HarpRoom] ✅ String clicked! Index:', index, 'State was:', state);

            // Always visualize the note being clicked
            this.noteVisualizer?.showNote(index, 0.5);

            // Update game state when string is played
            this.updateGameStateOnStringPlay(index);

            // Handle the input with the jelly manager
            this.patientJellyManager.handlePlayerInput(index);
        } else {
            console.log('[HarpRoom] ❌ No harp string hit.');
        }
    }

    /**
     * Toggle hitbox debug visibility
     * Shows/hides the magenta wireframe boxes around harp strings
     */
    public toggleHitboxDebug(): void {
        this.debugHitboxes = !this.debugHitboxes;

        for (const hitbox of this.hitboxMeshes) {
            if (hitbox.material) {
                const mat = hitbox.material as THREE.MeshBasicMaterial;
                if (this.debugHitboxes) {
                    // Debug mode: show wireframe for visibility
                    mat.wireframe = true;
                    mat.opacity = 0.3;
                } else {
                    // Normal mode: solid for reliable raycasting
                    mat.wireframe = false;
                    mat.opacity = 0.05;
                }
            }
        }

        console.log(`[HarpRoom] Hitbox debug: ${this.debugHitboxes ? 'ON' : 'OFF'}`);
    }

    /**
     * Debug method to check hitbox status
     * Call from browser console: app.harpRoom.debugHitboxStatus()
     */
    public debugHitboxStatus(): void {
        console.log('[HarpRoom] === HITBOX DEBUG ===');
        console.log('[HarpRoom] Hitboxes in array:', this.hitboxMeshes.length);

        this.hitboxMeshes.forEach((hitbox, i) => {
            console.log(`[HarpRoom] Hitbox ${i}:`, {
                position: hitbox.position,
                visible: hitbox.visible,
                inScene: this.scene.children.includes(hitbox),
                parent: hitbox.parent?.name || hitbox.parent?.type || 'none',
                isHarpString: hitbox.userData.isHarpString,
                stringIndex: hitbox.userData.stringIndex,
                isHitbox: hitbox.userData.isHitbox
            });
        });

        // Check scene children
        let hitboxInScene = 0;
        this.scene.traverse((obj) => {
            if (obj.userData?.isHitbox) hitboxInScene++;
        });
        console.log('[HarpRoom] Hitboxes found via scene.traverse:', hitboxInScene);
        console.log('[HarpRoom] =====================');
    }

    /**
     * Create fallback scene if GLB loading fails
     */
    private createFallbackScene(): void {
        // Create simple arena floor
        const floorGeo = new THREE.PlaneGeometry(10, 10, 32, 32);
        this.waterMaterial = new WaterMaterial();
        this.arenaFloor = new THREE.Mesh(floorGeo, this.waterMaterial);
        this.arenaFloor.rotation.x = -Math.PI / 2;
        this.arenaFloor.name = 'ArenaFloor';
        this.scene.add(this.arenaFloor);
        this.vortexSystem.setWaterMaterial(this.waterMaterial);

        // Create feedback manager for fallback scene
        this.feedbackManager = new FeedbackManager(this.camera, this.scene);

        this.initialized = true;
    }

    /**
     * Main render loop - call each frame
     */
    public render(): void {
        if (!this.initialized) return;

        const deltaTime = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        // Update camera controller
        if (this.cameraController) {
            this.cameraController.update(deltaTime);
        }

        // Update vortex system
        this.vortexSystem.update(deltaTime, this.camera.position);

        // Update water material time
        if (this.waterMaterial) {
            this.waterMaterial.setTime(time);
            this.waterMaterial.setCameraPosition(this.camera.position);

            // Update jelly-to-water coupling for ripples
            if (this.jellyManager) {
                this.waterMaterial.updateJellyPositions(this.jellyManager);
            }
        }

        // Update WaterBall fluid simulation system
        if (this.fluidEnabled && this.fluidSimulator) {
            // Update player interaction (needs camera position)
            if (this.playerWaterInteraction) {
                this.playerWaterInteraction.update(this.camera.position, deltaTime);
            }

            // Update harp-water interaction (ripple decay)
            if (this.harpWaterInteraction) {
                this.harpWaterInteraction.update(deltaTime);
            }

            // Run particle physics simulation
            this.fluidSimulator.simulate(deltaTime);

            // Render fluid phases
            // Note: This requires a command encoder. We'll attempt to use the device queue if possible
            // or we need to hook into the renderer's command encoder.
            // For now, we'll try to create a one-off encoder which is not ideal but valid.
            const commandEncoder = (this.renderer as any).device.createCommandEncoder();

            // Render depth/thickness textures
            if (this.fluidDepthRenderer) {
                this.fluidDepthRenderer.execute(commandEncoder, this.fluidSimulator.numParticles);
            }

            // Render final fluid surface to texture
            // Warning: FluidSurfaceRenderer.execute requires a view to render TO.
            // We usually render to the screen via context.getCurrentTexture().createView()
            // But doing this here might conflict with Three.js rendering.
            // We need a proper hook. For now, skipping execution to avoid crash/conflict
            // until we integrate with Visionary/Three.js render loop properly.
            /*
            if (this.fluidSurfaceRenderer) {
                 const context = (this.renderer.domElement as any).getContext('webgpu');
                 if (context) {
                    this.fluidSurfaceRenderer.execute(commandEncoder, context.getCurrentTexture().createView());
                 }
            }
            */

            // Submit commands
            (this.renderer as any).device.queue.submit([commandEncoder.finish()]);

            // Update wake effect particles
            if (this.wakeEffect && this.playerWaterInteraction) {
                this.wakeEffect.update(
                    deltaTime,
                    this.camera.position as any, // vec3 type mismatch shim
                    this.playerWaterInteraction.getPlayerMovementDirection() as any, // vec3 shim
                    this.playerWaterInteraction.isInWater()
                );
            }
        }

        // Update feedback manager (camera shake, highlights)
        if (this.feedbackManager) {
            this.feedbackManager.update(deltaTime);
        }

        // Update jelly manager
        if (this.jellyManager) {
            this.jellyManager.update(deltaTime, time, this.camera.position);
        }

        // Update splash effects
        if (this.splashEffect) {
            this.splashEffect.update(deltaTime);
        }

        // NEW: Update visual enhancement systems
        if (this.teachingBeam) {
            this.teachingBeam.update(deltaTime, time, this.camera.position);
        }

        if (this.stringHighlight) {
            this.stringHighlight.update(deltaTime, time);
            // Update camera position for string highlights
            this.scene.traverse((obj) => {
                if ((obj as any).setCameraPosition) {
                    (obj as any).setCameraPosition(this.camera.position);
                }
            });
        }

        if (this.jellyLabels) {
            this.jellyLabels.update(deltaTime, this.camera.position);
        }

        if (this.noteVisualizer) {
            this.noteVisualizer.update(deltaTime, time);
            // Update camera position for note visualizer
            this.scene.traverse((obj) => {
                if (obj instanceof THREE.Mesh && (obj.material as any).uniforms?.uCameraPosition) {
                    (obj.material as any).uniforms.uCameraPosition.value.copy(this.camera.position);
                }
            });
        }

        // Update summon rings
        this.summonRings.forEach(ring => ring.update(deltaTime));

        if (this.activationController) this.activationController.update(deltaTime);
        if (this.whiteFlash && this.whiteFlash.visible) {
            this.whiteFlash.positionInFrontOfCamera(this.camera);
            this.whiteFlash.update(deltaTime, time);
        }

        // Render
        this.renderer.render(this.scene, this.camera);

        // Visionary specific splat rendering (must happen after main scene render for depth)
        if (this.splatRenderer) {
            this.splatRenderer.render(this.camera);
        }

        // Check for completion
        if (this.vortexSystem.isFullyActivated() && this.onCompleteCallback) {
            this.onCompleteCallback();
        }
    }

    /**
     * Update duet progress (called by game logic)
     */
    public updateDuetProgress(progress: number): void {
        this.duetProgress = Math.max(0, Math.min(1, progress));
        this.vortexSystem.updateDuetProgress(this.duetProgress);

        // Update WaterBall sphere constraint animator (transforms plane to tunnel)
        if (this.fluidEnabled && this.sphereAnimator) {
            this.sphereAnimator.update(this.duetProgress, 0.016); // ~60fps frame time
        }
    }

    /**
     * Trigger string ripple effect
     */
    public triggerStringRipple(stringIndex: number, intensity: number = 1.0): void {
        // Trigger vortex ripple (old system)
        this.vortexSystem.triggerStringRipple(stringIndex, intensity);

        // Trigger WaterBall fluid harp interaction
        if (this.fluidEnabled && this.harpWaterInteraction) {
            this.harpWaterInteraction.onStringPlucked(stringIndex, intensity);
        }
    }

    /**
     * Trigger feedback for wrong note
     */
    public triggerWrongNote(targetNoteIndex: number): void {
        if (this.feedbackManager) {
            this.feedbackManager.triggerWrongNote(targetNoteIndex);
        }
    }

    /**
     * Trigger feedback for premature play
     */
    public triggerPrematurePlay(): void {
        if (this.feedbackManager) {
            this.feedbackManager.triggerPrematurePlay();
        }
    }

    /**
     * Trigger feedback for correct note
     */
    public async triggerCorrectNote(noteIndex: number): Promise<void> {
        if (this.feedbackManager) {
            await this.feedbackManager.triggerCorrectNote(noteIndex);
        }
    }

    /**
     * Set completion callback
     */
    public onComplete(callback: () => void): void {
        this.onCompleteCallback = callback;
    }

    /**
     * Get scene for external access
     */
    public getScene(): THREE.Scene {
        return this.scene;
    }

    /**
     * Get camera for external access
     */
    public getCamera(): THREE.Camera {
        return this.camera;
    }

    /**
     * Get feedback manager for external access
     */
    public getFeedbackManager(): FeedbackManager | undefined {
        return this.feedbackManager;
    }

    /**
     * Handle game state changes from GameManager
     * Called when gameManager emits 'state:changed' event
     */
    private handleGameStateChanged = (newState: any, oldState: any): void => {
        console.log('[HarpRoom] Game state changed:', {
            oldState: oldState.currentState,
            newState: newState.currentState,
            archiveOfVoices: newState.archiveOfVoices,
            harpSequencesCompleted: newState.harpSequencesCompleted,
        });

        // Sync harp puzzle progress from state
        if (newState.harpSequencesCompleted !== undefined) {
            const activation = newState.harpSequencesCompleted / (newState.harpRequiredSequences || 3);
            this.vortexSystem.updateDuetProgress(Math.min(1, activation));
        }

        // Sync vortex activation from state
        if (newState.vortexActivation !== undefined) {
            this.vortexSystem.updateDuetProgress(newState.vortexActivation);
        }

        // Handle shell collection state
        const oldCollected = oldState.archiveOfVoices || false;
        const newCollected = newState.archiveOfVoices || false;
        if (newCollected && !oldCollected) {
            console.log('[HarpRoom] Archive of Voices shell collected!');
            // Trigger white flash ending
            this.triggerWhiteFlash();
        }
    };

    /**
     * Trigger the white flash ending sequence
     */
    private triggerWhiteFlash(): void {
        if (!this.whiteFlash) return;

        console.log('[HarpRoom] Triggering white flash ending...');

        // Add white flash to scene and trigger
        if (!this.whiteFlash.parent) {
            this.scene.add(this.whiteFlash);
        }

        this.whiteFlash.positionInFrontOfCamera(this.camera);
        this.whiteFlash.trigger(() => {
            console.log('[HarpRoom] White flash complete, transitioning to transcendent state');
            if (this.gameManager) {
                this.gameManager.setState({ currentState: 13 }); // TRANSSCENDENT
            }
        });
    }

    /**
     * Update game state when harp string is played
     */
    private updateGameStateOnStringPlay(stringIndex: number): void {
        if (!this.gameManager) return;

        const state = this.gameManager.getState();
        const playedStrings = [...(state.harpPlayedStrings || []), stringIndex];

        this.gameManager.setState({
            harpPlayedStrings: playedStrings,
            vibratingString: stringIndex,
            waterRippleIntensity: 1.0,
        });

        // Reset vibration after a delay
        setTimeout(() => {
            this.gameManager?.setState({ vibratingString: -1, waterRippleIntensity: 0 });
        }, 500);
    }

    /**
     * Update game state when sequence is completed
     */
    private updateGameStateOnSequenceComplete(): void {
        if (!this.gameManager) return;

        const state = this.gameManager.getState();
        const newCompleted = (state.harpSequencesCompleted || 0) + 1;
        const required = state.harpRequiredSequences || 3;

        console.log(`[HarpRoom] Sequence complete! ${newCompleted}/${required}`);

        const newState = {
            harpSequencesCompleted: newCompleted,
            vortexActivation: newCompleted / required,
        };

        // Check if puzzle is complete
        if (newCompleted >= required) {
            console.log('[HarpRoom] HARP PUZZLE COMPLETE! Shell available.');
            (newState as any).currentState = 11; // HARP_COMPLETE
            (newState as any).harpState = 5; // COMPLETE
        }

        this.gameManager.setState(newState);
    }

    /**
     * Cleanup and destroy all resources
     * CRITICAL: Must be called before scene transitions
     */
    public async destroy(): Promise<void> {
        // Remove game manager event listener
        if (this.gameManager) {
            this.gameManager.off('state:changed', this.handleGameStateChanged);
        }

        // Destroy feedback manager
        if (this.feedbackManager) {
            await this.feedbackManager.destroy();
        }

        // NEW: Destroy visual enhancement systems
        if (this.teachingBeam) {
            this.teachingBeam.destroy();
            this.scene.remove(this.teachingBeam);
        }

        if (this.stringHighlight) {
            this.stringHighlight.destroy();
        }

        if (this.jellyLabels) {
            this.jellyLabels.destroy();
            this.scene.remove(this.jellyLabels);
        }

        if (this.noteVisualizer) {
            this.noteVisualizer.destroy();
            this.scene.remove(this.noteVisualizer);
        }

        // Destroy summon rings
        this.summonRings.forEach(ring => {
            ring.destroy();
            this.scene.remove(ring);
        });
        this.summonRings.clear();

        // Destroy vortex system
        this.vortexSystem.destroy();

        // Destroy water material
        if (this.waterMaterial) {
            this.waterMaterial.destroy();
        }

        // Dispose renderer
        this.renderer.dispose();

        // Remove event listeners
        window.removeEventListener('resize', this.onResize.bind(this));

        // Clear references
        this.waterMaterial = undefined;
        this.arenaFloor = undefined;
        this.feedbackManager = undefined;
        this.gameManager = undefined;
        this.teachingBeam = undefined;
        this.stringHighlight = undefined;
        this.jellyLabels = undefined;
        this.noteVisualizer = undefined;
        this.cameraController = undefined;
    }

    /**
     * Check if player can interact with harp (for camera lock-on)
     */
    public canInteractWithHarp(playerPosition: THREE.Vector3): boolean {
        if (!this.cameraController) return false;
        return this.cameraController.canInteract(playerPosition);
    }

    /**
     * Engage camera lock-on for harp interaction
     */
    public engageHarpLockOn(): void {
        if (!this.cameraController) return;
        this.cameraController.engageLockOn();
    }

    /**
     * Disengage camera lock-on
     */
    public disengageHarpLockOn(): void {
        if (!this.cameraController) return;
        this.cameraController.disengageLockOn();
    }

    /**
     * Get current camera interaction state
     */
    public getCameraState(): HarpInteractionState | undefined {
        return this.cameraController?.getState();
    }

    /**
     * Check if camera is locked onto harp
     */
    public isCameraLocked(): boolean {
        return this.cameraController?.isLocked() ?? false;
    }

    /**
     * Check if player controls are enabled
     */
    public arePlayerControlsEnabled(): boolean {
        return this.cameraController?.areControlsEnabled() ?? true;
    }

    /**
     * Get audio systems for external resumption
     * Used to resume AudioContext on first user interaction
     */
    public getAudioSystems(): { harmonyChord?: HarmonyChord; gentleAudioFeedback?: any } {
        return {
            harmonyChord: this.harmonyChord,
            gentleAudioFeedback: this.feedbackManager?.getAudioFeedback()
        };
    }
}
