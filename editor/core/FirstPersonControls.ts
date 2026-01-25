/**
 * FirstPersonControls.ts - First-Person Camera Controls for Editor
 *
 * Provides WASD + mouse look controls for walking through the scene
 * in editor mode. Similar to game CharacterController but simplified
 * for editor use (no physics, just camera movement).
 *
 * Features:
 * - WASD movement (forward, left, backward, right)
 * - Mouse look (pointer lock)
 * - Shift to sprint
 * - Space/Shift for up/down (fly mode)
 * - Adjustable camera height and speed
 * - Collision with floor (raycast down)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

interface FirstPersonControlsOptions {
    moveSpeed?: number;
    sprintMultiplier?: number;
    lookSensitivity?: number;
    cameraHeight?: number;
    enableFlyMode?: boolean;
    orbitControls?: OrbitControls | null;
    scene?: THREE.Scene | null;
}

interface FirstPersonState {
    position: THREE.Vector3;
    rotation: { yaw: number; pitch: number };
    velocity: THREE.Vector3;
    isPointerLocked: boolean;
    moveForward: boolean;
    moveBackward: boolean;
    moveLeft: boolean;
    moveRight: boolean;
    moveUp: boolean;
    moveDown: boolean;
    isSprinting: boolean;
}

/**
 * FirstPersonControls Manager
 *
 * Usage:
 * ```typescript
 * const controls = new FirstPersonControls(camera, renderer.domElement);
 * controls.initialize();
 * controls.enable(); // Start first-person mode
 * controls.disable(); // Return to orbit controls
 * ```
 */
class FirstPersonControls {
    private camera: THREE.Camera;
    private domElement: HTMLElement;
    private options: Required<FirstPersonControlsOptions>;
    private orbitControls: OrbitControls | null;
    private scene: THREE.Scene | null;
    private raycaster: THREE.Raycaster;
    private downVector: THREE.Vector3;

    // Camera state
    private state: FirstPersonState = {
        position: new THREE.Vector3(0, 1.7, 0), // Eye level height
        rotation: { yaw: 0, pitch: 0 },
        velocity: new THREE.Vector3(),
        isPointerLocked: false,
        moveForward: false,
        moveBackward: false,
        moveLeft: false,
        moveRight: false,
        moveUp: false,
        moveDown: false,
        isSprinting: false,
    };

    // Previous camera state (for restoring)
    private previousCameraState: {
        position: THREE.Vector3;
        rotation: THREE.Euler;
    } | null = null;

    // Input state
    private keys: Record<string, boolean> = {};
    private onMouseMoveBound: ((event: MouseEvent) => void) | null = null;
    private onPointerLockChangeBound: ((event: Event) => void) | null = null;
    private onKeyDownBound: ((event: KeyboardEvent) => void) | null = null;
    private onKeyUpBound: ((event: KeyboardEvent) => void) | null = null;

    // Update loop
    private animationFrameId: number | null = null;
    private lastTime: number = 0;

    // Events
    private eventListeners: Map<string, Set<Function>> = new Map();


    constructor(camera: THREE.Camera, domElement: HTMLElement, options: FirstPersonControlsOptions = {}) {
        this.camera = camera;
        this.domElement = domElement;
        this.orbitControls = options.orbitControls ?? null;
        this.scene = options.scene ?? null;
        this.raycaster = new THREE.Raycaster();
        this.downVector = new THREE.Vector3(0, -1, 0);

        this.options = {
            moveSpeed: options.moveSpeed ?? 5.0,
            sprintMultiplier: options.sprintMultiplier ?? 2.0,
            lookSensitivity: options.lookSensitivity ?? 0.002,
            cameraHeight: options.cameraHeight ?? 1.7,
            enableFlyMode: options.enableFlyMode ?? true,
            orbitControls: options.orbitControls ?? null,
            scene: options.scene ?? null,
        };

        console.log('FirstPersonControls: Created', this.options);
    }

    /**
     * Get ground height at a position using raycasting
     * @param x - X position to check
     * @param z - Z position to check
     * @param maxHeight - Maximum height to raycast from (default 100)
     * @returns Ground height at position, or 0 if no ground found
     */
    private getGroundHeight(x: number, z: number, maxHeight: number = 100): number {
        if (!this.scene) return 0;

        // Cast ray from above down to find ground
        this.raycaster.set(new THREE.Vector3(x, maxHeight, z), this.downVector);

        // Get all objects in scene that could be ground
        const objects = this.scene.children.filter(obj => {
            // Filter out helpers, gizmos, and invisible objects
            return !obj.name.includes('Grid') &&
                   !obj.name.includes('Axes') &&
                   !obj.name.includes('Helper') &&
                   obj.visible !== false;
        });

        if (objects.length === 0) return 0;

        const intersects = this.raycaster.intersectObjects(objects, true);

        if (intersects.length > 0 && intersects[0]) {
            return intersects[0].point.y;
        }

        return 0;
    }

    /**
     * Initialize the controls (call before first use)
     */
    public initialize(): void {
        // Save initial camera position
        this.state.position.copy(this.camera.position);
        this.state.position.y = this.options.cameraHeight; // Ensure eye level

        // Get initial rotation from camera
        const euler = new THREE.Euler(0, 0, 0, 'YXZ');
        euler.copy(this.camera.rotation);
        this.state.rotation.pitch = euler.x;
        this.state.rotation.yaw = euler.y;

        console.log('FirstPersonControls: Initialized', {
            position: this.state.position,
            rotation: this.state.rotation,
        });
    }

    /**
     * Enable first-person controls
     */
    public enable(): void {
        console.log('FirstPersonControls: Enabling...');

        // Disable OrbitControls to prevent conflict
        if (this.orbitControls) {
            this.orbitControls.enabled = false;
            console.log('FirstPersonControls: OrbitControls disabled');
        }

        // Save current camera state for restoration
        this.previousCameraState = {
            position: this.camera.position.clone(),
            rotation: this.camera.rotation.clone(),
        };

        // Find ground height at current camera X,Z position
        const groundHeight = this.getGroundHeight(this.camera.position.x, this.camera.position.z);

        // Set camera to ground level at the same X,Z position, with eye-level height
        this.state.position.set(
            this.camera.position.x,
            groundHeight + this.options.cameraHeight,
            this.camera.position.z
        );

        // Set camera to first-person position
        this.camera.position.copy(this.state.position);
        this.updateCameraRotation();

        console.log('FirstPersonControls: Placed on ground at height', groundHeight);

        // Request pointer lock
        this.domElement.requestPointerLock();

        // Setup event listeners
        this.setupEventListeners();

        // Start update loop
        this.lastTime = performance.now();
        this.startUpdateLoop();

        this.emit('enabled');

        console.log('FirstPersonControls: Enabled');
    }

    /**
     * Disable first-person controls and restore camera
     */
    public disable(): void {
        console.log('FirstPersonControls: Disabling...');

        // Stop update loop
        this.stopUpdateLoop();

        // Remove event listeners
        this.removeEventListeners();

        // Exit pointer lock
        if (document.pointerLockElement === this.domElement) {
            document.exitPointerLock();
        }

        // Restore previous camera state if available
        if (this.previousCameraState) {
            this.camera.position.copy(this.previousCameraState.position);
            this.camera.rotation.copy(this.previousCameraState.rotation);
        }

        // Re-enable OrbitControls
        if (this.orbitControls) {
            this.orbitControls.enabled = true;
            console.log('FirstPersonControls: OrbitControls re-enabled');
        }

        this.emit('disabled');

        console.log('FirstPersonControls: Disabled');
    }

    /**
     * Toggle first-person mode
     */
    public toggle(): void {
        if (this.state.isPointerLocked) {
            this.disable();
        } else {
            this.enable();
        }
    }

    /**
     * Set camera position (for teleporting)
     */
    public setPosition(position: THREE.Vector3 | { x: number; y: number; z: number }): void {
        this.state.position.set(position.x, position.y, position.z);
        this.camera.position.copy(this.state.position);
    }

    /**
     * Set camera height
     */
    public setCameraHeight(height: number): void {
        this.options.cameraHeight = height;
        this.state.position.y = height;
        this.camera.position.y = height;
    }

    /**
     * Get current state
     */
    public getState(): Readonly<FirstPersonState> {
        return { ...this.state };
    }

    /**
     * Setup event listeners
     */
    private setupEventListeners(): void {
        // Mouse move for look
        this.onMouseMoveBound = this.onMouseMove.bind(this);
        document.addEventListener('mousemove', this.onMouseMoveBound);

        // Pointer lock change
        this.onPointerLockChangeBound = this.onPointerLockChange.bind(this);
        document.addEventListener('pointerlockchange', this.onPointerLockChangeBound);

        // Keyboard input
        this.onKeyDownBound = this.onKeyDown.bind(this);
        this.onKeyUpBound = this.onKeyUp.bind(this);
        document.addEventListener('keydown', this.onKeyDownBound);
        document.addEventListener('keyup', this.onKeyUpBound);

        // Track initial key states
        for (const key in this.keys) {
            delete this.keys[key];
        }
    }

    /**
     * Remove event listeners
     */
    private removeEventListeners(): void {
        if (this.onMouseMoveBound) {
            document.removeEventListener('mousemove', this.onMouseMoveBound);
            this.onMouseMoveBound = null;
        }
        if (this.onPointerLockChangeBound) {
            document.removeEventListener('pointerlockchange', this.onPointerLockChangeBound);
            this.onPointerLockChangeBound = null;
        }
        if (this.onKeyDownBound) {
            document.removeEventListener('keydown', this.onKeyDownBound);
            this.onKeyDownBound = null;
        }
        if (this.onKeyUpBound) {
            document.removeEventListener('keyup', this.onKeyUpBound);
            this.onKeyUpBound = null;
        }
    }

    /**
     * Handle mouse movement for look
     */
    private onMouseMove(event: MouseEvent): void {
        if (!this.state.isPointerLocked) return;

        const movementX = event.movementX || 0;
        const movementY = event.movementY || 0;

        // Update yaw (horizontal rotation)
        this.state.rotation.yaw -= movementX * this.options.lookSensitivity;

        // Update pitch (vertical rotation) with clamping
        this.state.rotation.pitch -= movementY * this.options.lookSensitivity;
        this.state.rotation.pitch = Math.max(
            -Math.PI / 2 + 0.01, // Look down limit (almost straight down)
            Math.min(Math.PI / 2 - 0.01, this.state.rotation.pitch) // Look up limit (almost straight up)
        );

        this.updateCameraRotation();
    }

    /**
     * Handle pointer lock change
     */
    private onPointerLockChange(): void {
        this.state.isPointerLocked = document.pointerLockElement === this.domElement;

        if (this.state.isPointerLocked) {
            this.emit('pointerLockAcquired');
        } else {
            this.emit('pointerLockReleased');
            // Optionally disable controls when pointer lock is lost
            // this.disable();
        }
    }

    /**
     * Handle key down
     */
    private onKeyDown(event: KeyboardEvent): void {
        // Ignore if typing in input
        if (event.target instanceof HTMLInputElement ||
            event.target instanceof HTMLTextAreaElement) {
            return;
        }

        this.keys[event.code] = true;

        // Update movement flags
        this.updateMovementFlags();
    }

    /**
     * Handle key up
     */
    private onKeyUp(event: KeyboardEvent): void {
        this.keys[event.code] = false;

        // Update movement flags
        this.updateMovementFlags();
    }

    /**
     * Update movement flags from key state
     */
    private updateMovementFlags(): void {
        this.state.moveForward = this.keys['KeyW'] || false;
        this.state.moveBackward = this.keys['KeyS'] || false;
        this.state.moveLeft = this.keys['KeyA'] || false;
        this.state.moveRight = this.keys['KeyD'] || false;
        this.state.isSprinting = this.keys['ShiftLeft'] || this.keys['ShiftRight'] || false;

        if (this.options.enableFlyMode) {
            this.state.moveUp = this.keys['Space'] || false;
            this.state.moveDown = this.keys['ControlLeft'] || this.keys['ControlRight'] || false;
        }
    }

    /**
     * Update camera rotation from yaw/pitch
     */
    private updateCameraRotation(): void {
        const euler = new THREE.Euler(
            this.state.rotation.pitch,
            this.state.rotation.yaw,
            0,
            'YXZ'
        );
        this.camera.quaternion.setFromEuler(euler);
    }

    /**
     * Start update loop
     */
    private startUpdateLoop(): void {
        const animate = (time: number) => {
            this.update(time);
            this.animationFrameId = requestAnimationFrame(animate);
        };
        this.animationFrameId = requestAnimationFrame(animate);
    }

    /**
     * Stop update loop
     */
    private stopUpdateLoop(): void {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Update loop
     */
    private update(time: number): void {
        if (!this.state.isPointerLocked) return;

        const deltaTime = Math.min((time - this.lastTime) / 1000, 0.1); // Cap delta time
        this.lastTime = time;

        // Calculate movement direction
        const moveDirection = new THREE.Vector3();

        // Get forward/right vectors from camera rotation (no pitch for forward/right movement)
        const forward = new THREE.Vector3(0, 0, -1);
        forward.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.state.rotation.yaw);

        const right = new THREE.Vector3(1, 0, 0);
        right.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.state.rotation.yaw);

        // Apply movement input
        if (this.state.moveForward) moveDirection.add(forward);
        if (this.state.moveBackward) moveDirection.sub(forward);
        if (this.state.moveRight) moveDirection.add(right);
        if (this.state.moveLeft) moveDirection.sub(right);

        // Normalize diagonal movement
        if (moveDirection.length() > 0) {
            moveDirection.normalize();
        }

        // Apply speed
        let speed = this.options.moveSpeed;
        if (this.state.isSprinting) {
            speed *= this.options.sprintMultiplier;
        }

        // Calculate new position
        const deltaPosition = moveDirection.multiplyScalar(speed * deltaTime);

        // Apply vertical movement (fly mode)
        if (this.options.enableFlyMode) {
            if (this.state.moveUp) {
                deltaPosition.y += speed * deltaTime;
            }
            if (this.state.moveDown) {
                deltaPosition.y -= speed * deltaTime;
            }
        }

        // Update position
        this.state.position.add(deltaPosition);

        // Ground collision - use raycasting to find actual ground height
        const groundHeight = this.getGroundHeight(this.state.position.x, this.state.position.z);
        const targetY = groundHeight + this.options.cameraHeight;

        // If fly mode is disabled, always keep camera on ground
        // If fly mode is enabled, allow manual vertical movement but clamp to ground minimum
        if (!this.options.enableFlyMode) {
            // Walking mode - always stay on ground
            this.state.position.y = targetY;
        } else {
            // Fly mode - allow going up/down, but don't go below ground
            if (this.state.position.y < targetY) {
                this.state.position.y = targetY;
            }
        }

        // Apply to camera
        this.camera.position.copy(this.state.position);

        // Emit position changed event
        this.emit('positionChanged', {
            position: this.state.position.clone(),
            rotation: { ...this.state.rotation },
        });
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
                    console.error(`FirstPersonControls: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Clean up
     */
    public destroy(): void {
        this.disable();
        this.eventListeners.clear();
        console.log('FirstPersonControls: Destroyed');
    }
}

export default FirstPersonControls;
