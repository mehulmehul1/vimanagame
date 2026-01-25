/**
 * SimpleFirstPersonControls.ts - EXACT Game-Like First-Person Controls
 *
 * Replicates the game's CharacterController behavior for editor use:
 * - Same camera smoothing (0.15)
 * - Same speed (2.5 base, 1.75x sprint)
 * - Same pitch limits (-60° to +89°)
 * - Same mouse sensitivity (0.0025)
 * - Headbob while moving
 * - Body yaw follows camera during movement
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

interface SimpleFirstPersonControlsOptions {
    moveSpeed?: number;
    sprintMultiplier?: number;
    mouseSensitivity?: number;
    cameraHeight?: number;
    cameraSmoothingFactor?: number;
    orbitControls?: OrbitControls | null;
    scene?: THREE.Scene | null;
}

/**
 * EXACT game-like first-person controls
 */
class SimpleFirstPersonControls {
    private camera: THREE.Camera;
    private domElement: HTMLElement;
    private orbitControls: OrbitControls | null;
    private scene: THREE.Scene | null;

    // Options - matching game defaults
    private moveSpeed: number;
    private sprintMultiplier: number;
    private mouseSensitivity: number;
    private cameraHeight: number;
    private cameraSmoothingFactor: number;

    // State - EXACTLY matching CharacterController
    private enabled: boolean = false;
    private isPointerLocked: boolean = false;

    // Camera rotation with targets (like the game)
    private yaw: number = 0;
    private pitch: number = 0;
    private targetYaw: number = 0;
    private targetPitch: number = 0;

    // Body rotation (separate from camera, follows during movement)
    private bodyYaw: number = 0;

    // Input state
    private keys: Set<string> = new Set();

    // Previous state for restoration
    private previousCameraState: {
        position: THREE.Vector3;
        rotation: THREE.Euler;
    } | null = null;

    // Headbob state
    private headbobTime: number = 0;
    private headbobIntensity: number = 0;
    private headbobEnabled: boolean = true;

    // Reusable vectors
    private readonly _tempForward = new THREE.Vector3();
    private readonly _tempRight = new THREE.Vector3();
    private readonly _yAxis = new THREE.Vector3(0, 1, 0);

    // Track original frustumCulled states for restoration
    private originalFrustumCulled: Map<THREE.Object3D, boolean> = new Map();

    // Event handlers (bound)
    private onKeyDown: (e: KeyboardEvent) => void;
    private onKeyUp: (e: KeyboardEvent) => void;
    private onMouseMove: (e: MouseEvent) => void;
    private onPointerLockChange: () => void;
    private onClick: (e: MouseEvent) => void;

    // Update loop
    private animationFrameId: number | null = null;
    private lastTime: number = 0;

    constructor(camera: THREE.Camera, domElement: HTMLElement, options: SimpleFirstPersonControlsOptions = {}) {
        this.camera = camera;
        this.domElement = domElement;
        this.orbitControls = options.orbitControls ?? null;
        this.scene = options.scene ?? null;

        // Options with EXACT game defaults
        this.moveSpeed = options.moveSpeed ?? 2.5;           // Game's baseSpeed
        this.sprintMultiplier = options.sprintMultiplier ?? 1.75; // Game's sprintMultiplier
        this.mouseSensitivity = options.mouseSensitivity ?? 0.0025; // Game's mouseSensitivity
        this.cameraHeight = options.cameraHeight ?? 0.9;     // Game's cameraHeight (from capsule center)
        this.cameraSmoothingFactor = options.cameraSmoothingFactor ?? 0.15; // Game's smoothing

        // Bind event handlers
        this.onKeyDown = this.handleKeyDown.bind(this);
        this.onKeyUp = this.handleKeyUp.bind(this);
        this.onMouseMove = this.handleMouseMove.bind(this);
        this.onPointerLockChange = this.handlePointerLockChange.bind(this);
        this.onClick = this.handleClick.bind(this);

        console.log('SimpleFirstPersonControls: Created with game-exact defaults');
    }

    /**
     * Calculate headbob offset (matching game's calculateHeadbob)
     */
    private calculateHeadbob(isSprinting: boolean): { vertical: number; horizontal: number } {
        if (this.headbobIntensity <= 0.01) {
            return { vertical: 0, horizontal: 0 };
        }

        // Headbob parameters matching the game
        const bobFrequency = isSprinting ? 10 : 8;      // Cycles per second
        const verticalAmplitude = isSprinting ? 0.08 : 0.05;  // Height bob amount
        const horizontalAmplitude = isSprinting ? 0.04 : 0.025; // Side-to-side amount

        const verticalBob = Math.sin(this.headbobTime * bobFrequency * Math.PI * 2) * verticalAmplitude * this.headbobIntensity;
        const horizontalBob = Math.cos(this.headbobTime * bobFrequency * Math.PI * 2) * horizontalAmplitude * this.headbobIntensity;

        return { vertical: verticalBob, horizontal: horizontalBob };
    }

    /**
     * Get forward/right vectors based on body yaw (like the game's getForwardRightVectors)
     */
    private getForwardRightVectors(): { forward: THREE.Vector3; right: THREE.Vector3 } {
        // Forward vector based on body yaw (not camera yaw - for movement)
        this._tempForward.set(0, 0, -1).applyAxisAngle(this._yAxis, this.bodyYaw);
        // Right vector (perpendicular to forward)
        this._tempRight.set(1, 0, 0).applyAxisAngle(this._yAxis, this.bodyYaw);

        return { forward: this._tempForward, right: this._tempRight };
    }

    /**
     * Enable first-person controls
     */
    public enable(): void {
        if (this.enabled) return;

        console.log('SimpleFirstPersonControls: Enabling...');

        // Save current camera state
        this.previousCameraState = {
            position: this.camera.position.clone(),
            rotation: this.camera.rotation.clone()
        };

        // Disable OrbitControls completely
        if (this.orbitControls) {
            this.orbitControls.enabled = false;
            this.orbitControls.enableDamping = false;
        }

        // Get current camera rotation as starting point
        const euler = new THREE.Euler(0, 0, 0, 'YXZ');
        euler.copy(this.camera.rotation);
        this.pitch = euler.x;
        this.yaw = euler.y;
        this.targetYaw = this.yaw;
        this.targetPitch = this.pitch;
        this.bodyYaw = this.yaw; // Body starts facing same direction

        // Place camera at a reasonable starting height
        // If camera is very high (orbit mode), bring it down to eye level
        if (this.camera.position.y > 5) {
            // Camera is high up - set to reasonable eye level
            this.camera.position.y = 1.7; // Average person's eye level
        }

        // Setup event listeners
        document.addEventListener('keydown', this.onKeyDown);
        document.addEventListener('keyup', this.onKeyUp);
        document.addEventListener('mousemove', this.onMouseMove);
        document.addEventListener('pointerlockchange', this.onPointerLockChange);
        this.domElement.addEventListener('click', this.onClick);

        // Request pointer lock (will be fulfilled on user click)
        this.domElement.requestPointerLock();

        // Disable frustum culling so models don't disappear when inside them
        this.disableFrustumCulling();

        // Start update loop
        this.lastTime = performance.now();
        this.startUpdate();

        this.enabled = true;

        console.log('SimpleFirstPersonControls: Enabled, camera at y=', this.camera.position.y);
    }

    /**
     * Disable first-person controls
     */
    public disable(): void {
        if (!this.enabled) return;

        console.log('SimpleFirstPersonControls: Disabling...');

        this.enabled = false;
        this.isPointerLocked = false;

        // Stop update loop
        this.stopUpdate();

        // Restore frustum culling to original states
        this.restoreFrustumCulling();

        // Remove event listeners
        document.removeEventListener('keydown', this.onKeyDown);
        document.removeEventListener('keyup', this.onKeyUp);
        document.removeEventListener('mousemove', this.onMouseMove);
        document.removeEventListener('pointerlockchange', this.onPointerLockChange);
        this.domElement.removeEventListener('click', this.onClick);

        // Exit pointer lock
        if (document.pointerLockElement === this.domElement) {
            document.exitPointerLock();
        }

        // Restore OrbitControls
        if (this.orbitControls) {
            this.orbitControls.enabled = true;
            this.orbitControls.enableDamping = true;
        }

        // Restore camera state
        if (this.previousCameraState) {
            this.camera.position.copy(this.previousCameraState.position);
            this.camera.rotation.copy(this.previousCameraState.rotation);
        }

        console.log('SimpleFirstPersonControls: Disabled');
    }

    /**
     * Handle keyboard down
     */
    private handleKeyDown(e: KeyboardEvent): void {
        // Ignore if typing in input
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
            return;
        }

        this.keys.add(e.code);

        // Handle escape to exit
        if (e.code === 'Escape') {
            this.disable();
        }
    }

    /**
     * Handle keyboard up
     */
    private handleKeyUp(e: KeyboardEvent): void {
        this.keys.delete(e.code);
    }

    /**
     * Handle mouse movement for look (EXACT game behavior)
     */
    private handleMouseMove(e: MouseEvent): void {
        if (!this.isPointerLocked) return;

        const movementX = e.movementX || 0;
        const movementY = e.movementY || 0;

        // Apply to targets (with smoothing applied later in update)
        // Game uses: targetYaw -= inputX; targetPitch -= inputY;
        this.targetYaw -= movementX * this.mouseSensitivity;
        this.targetPitch -= movementY * this.mouseSensitivity;

        // EXACT game pitch limits: -Math.PI / 3 to Math.PI / 2 - 0.01
        // (-60 degrees to ~89 degrees)
        this.targetPitch = Math.max(
            -Math.PI / 3,
            Math.min(Math.PI / 2 - 0.01, this.targetPitch)
        );
    }

    /**
     * Handle pointer lock change
     */
    private handlePointerLockChange(): void {
        const wasLocked = this.isPointerLocked;
        this.isPointerLocked = document.pointerLockElement === this.domElement;

        // Debug logging
        if (!wasLocked && this.isPointerLocked) {
            console.log('SimpleFirstPersonControls: Pointer lock ACQUIRED');
        } else if (wasLocked && !this.isPointerLocked) {
            console.log('SimpleFirstPersonControls: Pointer lock LOST');
        }
    }

    /**
     * Handle click to request pointer lock
     * Browser requires user gesture to acquire pointer lock
     */
    private handleClick(e: MouseEvent): void {
        if (!this.isPointerLocked && this.enabled) {
            console.log('SimpleFirstPersonControls: Click detected, requesting pointer lock...');
            this.domElement.requestPointerLock();
        }
    }

    /**
     * Start update loop
     */
    private startUpdate(): void {
        const update = (time: number) => {
            this.update(time);
            if (this.enabled) {
                this.animationFrameId = requestAnimationFrame(update);
            }
        };
        this.animationFrameId = requestAnimationFrame(update);
    }

    /**
     * Stop update loop
     */
    private stopUpdate(): void {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Update loop (EXACT game behavior from CharacterController.update())
     */
    private update(time: number): void {
        if (!this.enabled) return;

        const deltaTime = Math.min((time - this.lastTime) / 1000, 0.1);
        this.lastTime = time;

        // Smooth camera rotation (EXACT game behavior)
        // Game: yaw += (targetYaw - yaw) * cameraSmoothingFactor;
        this.yaw += (this.targetYaw - this.yaw) * this.cameraSmoothingFactor;
        this.pitch += (this.targetPitch - this.pitch) * this.cameraSmoothingFactor;

        // Apply rotation to camera
        const euler = new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ');
        this.camera.quaternion.setFromEuler(euler);

        // Get movement input
        const { forward, right } = this.getForwardRightVectors();

        let moveX = 0;
        let moveY = 0;

        // Game's movement: y is forward, x is right (getMovementInput returns {x, y})
        if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) moveY += 1;
        if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) moveY -= 1;
        if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) moveX += 1;
        if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) moveX -= 1;

        // Check if sprinting
        const isSprinting = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
        const currentSpeed = isSprinting
            ? this.moveSpeed * this.sprintMultiplier
            : this.moveSpeed;

        const isMoving = moveX !== 0 || moveY !== 0;

        // Calculate movement direction
        const movementDir = new THREE.Vector3();
        if (moveY !== 0) {
            movementDir.add(forward.clone().multiplyScalar(moveY));
        }
        if (moveX !== 0) {
            movementDir.add(right.clone().multiplyScalar(moveX));
        }

        if (movementDir.lengthSq() > 0) {
            movementDir.normalize().multiplyScalar(currentSpeed * deltaTime);
            this.camera.position.add(movementDir);

            // When moving, body gradually aligns with camera (EXACT game behavior)
            // Game: bodyYaw += (yaw - bodyYaw) * 0.15;
            const angleDiff = this.yaw - this.bodyYaw;
            const normalizedDiff = Math.atan2(Math.sin(angleDiff), Math.cos(angleDiff));
            this.bodyYaw += normalizedDiff * 0.15;

            // Update headbob
            this.headbobTime += deltaTime;
        }

        // Smooth headbob intensity transition
        const targetIntensity = isMoving ? 1.0 : 0.0;
        this.headbobIntensity += (targetIntensity - this.headbobIntensity) * 0.15;

        // Apply headbob offset
        const headbob = this.calculateHeadbob(isSprinting);

        // Get ground height at current position
        const groundHeight = this.getGroundHeight(this.camera.position.x, this.camera.position.z);

        // Set Y position: ground height + camera height + headbob
        // If no ground found (returns 0), keep current Y to avoid snapping to origin in splat-only scenes
        if (groundHeight > 0 || this.camera.position.y <= 2) {
            this.camera.position.y = groundHeight + this.cameraHeight + headbob.vertical;
        } else {
            // Keep current height when no ground is found below
            this.camera.position.y += headbob.vertical;
        }
    }

    /**
     * Get ground height at position using raycasting
     *
     * Casts from slightly above current camera position downward to find
     * the floor beneath the player. This ensures we find the floor inside
     * a building, not the roof above.
     */
    private getGroundHeight(x: number, z: number): number {
        if (!this.scene) return 0;

        // Cast ray from slightly above current camera position, going down
        // This finds the floor beneath us, not the roof above
        const raycaster = new THREE.Raycaster();
        const downVector = new THREE.Vector3(0, -1, 0);
        const rayOrigin = new THREE.Vector3(x, this.camera.position.y + 0.5, z);
        raycaster.set(rayOrigin, downVector);
        raycaster.far = 10; // Only look 10 units down (should find floor within this distance)

        // Filter objects - only check visible meshes, EXCLUDE splat meshes (Spark.js)
        // Splat meshes don't support standard raycasting and cause crashes
        const objects: THREE.Object3D[] = [];
        this.scene.traverse((obj) => {
            if (!obj.visible) return;

            const name = obj.name || '';
            const type = obj.type || '';

            // Skip helpers, gizmos, grids
            if (name.includes('Grid') ||
                name.includes('Axes') ||
                name.includes('Helper') ||
                name.includes('Gizmo') ||
                name.includes('TransformControls')) {
                return;
            }

            // Skip Spark.js SplatMesh - they don't support raycasting
            if (type.includes('SplatMesh') ||
                obj.constructor.name?.includes('Splat') ||
                name.toLowerCase().includes('splat')) {
                return;
            }

            // Only include Mesh objects (not splats)
            if (obj instanceof THREE.Mesh) {
                objects.push(obj);
            }
        });

        if (objects.length === 0) return 0;

        const intersects = raycaster.intersectObjects(objects, true);

        if (intersects.length > 0 && intersects[0]) {
            return intersects[0].point.y;
        }

        // No floor found below - return 0 as fallback
        return 0;
    }

    /**
     * Disable frustum culling on all scene objects
     * This prevents models from disappearing when camera is inside them
     */
    private disableFrustumCulling(): void {
        if (!this.scene) return;

        this.scene.traverse((obj) => {
            if (obj instanceof THREE.Mesh) {
                // Store original state and disable frustum culling
                this.originalFrustumCulled.set(obj, obj.frustumCulled);
                obj.frustumCulled = false;
            }
        });

        console.log('SimpleFirstPersonControls: Frustum culling disabled on scene objects');
    }

    /**
     * Restore original frustum culling states
     */
    private restoreFrustumCulling(): void {
        this.originalFrustumCulled.forEach((originalState, obj) => {
            if (obj instanceof THREE.Mesh) {
                obj.frustumCulled = originalState;
            }
        });

        this.originalFrustumCulled.clear();
        console.log('SimpleFirstPersonControls: Frustum culling restored');
    }

    /**
     * Toggle first-person mode
     */
    public toggle(): void {
        if (this.enabled) {
            this.disable();
        } else {
            this.enable();
        }
    }

    /**
     * Clean up
     */
    public dispose(): void {
        this.disable();
        this.keys.clear();
        console.log('SimpleFirstPersonControls: Disposed');
    }
}

export default SimpleFirstPersonControls;
