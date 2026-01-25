import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import EditorManager from './EditorManager';

/**
 * TransformGizmoManager - Transform gizmos for object manipulation
 *
 * Features:
 * - Translate gizmo (G key or toolbar)
 * - Rotate gizmo (R key)
 * - Scale gizmo (S key)
 * - Gizmo sync with Inspector inputs
 * - Keyboard shortcuts
 * - Visual mode indicator
 *
 * CRITICAL: Do not modify the src/ directory - this is editor-only code
 */
class TransformGizmoManager {
    private static instance: TransformGizmoManager;

    private transformControls: TransformControls | null = null;
    private editorManager: EditorManager | null = null;
    private currentMode: 'translate' | 'rotate' | 'scale' = 'translate';
    private eventListeners: Map<string, Set<Function>> = new Map();

    // Keyboard shortcuts
    private keyboardShortcuts = {
        translate: ['g', 't'],  // Removed 'w' to avoid conflict with first-person movement
        rotate: ['r'],
        scale: ['s']
    };

    // Flag to enable/disable keyboard shortcuts (disabled during first-person mode)
    private keyboardShortcutsEnabled = true;

    private constructor() {
        console.log('TransformGizmoManager: Initialized');
    }

    public static getInstance(): TransformGizmoManager {
        if (!TransformGizmoManager.instance) {
            TransformGizmoManager.instance = new TransformGizmoManager();
        }
        return TransformGizmoManager.instance;
    }

    public initialize(editorManager: EditorManager): void {
        this.editorManager = editorManager;

        if (!editorManager.camera || !editorManager.scene) {
            console.error('TransformGizmoManager: EditorManager not properly initialized');
            return;
        }

        // Skip if already initialized
        if (this.transformControls) {
            console.warn('TransformGizmoManager: Already initialized, skipping');
            return;
        }

        // Create TransformControls
        this.transformControls = new TransformControls(
            editorManager.camera,
            editorManager.renderer.domElement
        );

        // CRITICAL: TransformControls MUST be added to scene for rendering
        // It's a helper object that needs to be in the scene graph
        // @ts-ignore - TransformControls is not a standard Object3D but can be added to scene
        editorManager.scene.add(this.transformControls);

        // Setup event listeners
        this.setupEventListeners();

        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();

        console.log('TransformGizmoManager: Initialized with TransformControls');
    }

    private setupEventListeners(): void {
        if (!this.transformControls) return;

        // When dragging starts, disable OrbitControls to prevent camera movement
        this.transformControls.addEventListener('dragging-changed', (event) => {
            // Disable OrbitControls when dragging gizmo, enable when not
            if (this.editorManager?.orbitControls) {
                this.editorManager.orbitControls.enabled = !event.value;
            }

            this.emit('draggingChanged', event.value);
        });

        // When object is being transformed (during drag)
        this.transformControls.addEventListener('change', () => {
            if (this.transformControls?.object) {
                this.emit('transforming', {
                    object: this.transformControls.object,
                    mode: this.currentMode
                });
                // Also emit through EditorManager for Inspector sync
                if (this.editorManager) {
                    this.editorManager.emit('transformChanged', {
                        object: this.transformControls.object,
                        mode: this.currentMode
                    });
                }
            }
        });

        // Listen for selection changes from EditorManager
        if (this.editorManager) {
            this.editorManager.on('selectionChanged', (data: any) => {
                this.attachTo(data.current);
            });
        }
    }

    private setupKeyboardShortcuts(): void {
        const handleKeyDown = (event: KeyboardEvent) => {
            // Check if keyboard shortcuts are enabled (disabled in first-person mode)
            if (!this.keyboardShortcutsEnabled) {
                return;
            }

            // Ignore if typing in input
            if (event.target instanceof HTMLInputElement ||
                event.target instanceof HTMLTextAreaElement) {
                return;
            }

            const key = event.key.toLowerCase();

            // Check for translate shortcuts
            if (this.keyboardShortcuts.translate.includes(key)) {
                event.preventDefault();
                this.setMode('translate');
                return;
            }

            // Check for rotate shortcuts
            if (this.keyboardShortcuts.rotate.includes(key)) {
                event.preventDefault();
                this.setMode('rotate');
                return;
            }

            // Check for scale shortcuts
            if (this.keyboardShortcuts.scale.includes(key)) {
                event.preventDefault();
                this.setMode('scale');
                return;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
    }

    /**
     * Enable or disable keyboard shortcuts
     * Useful for disabling during first-person mode to avoid WASD conflicts
     */
    public setKeyboardShortcutsEnabled(enabled: boolean): void {
        this.keyboardShortcutsEnabled = enabled;
        console.log('TransformGizmoManager: Keyboard shortcuts', enabled ? 'enabled' : 'disabled');
    }

    public attachTo(object: THREE.Object3D | null): void {
        if (!this.transformControls) return;

        if (object) {
            this.transformControls.attach(object);
            console.log('TransformGizmoManager: Attached to', object.name);
        } else {
            this.transformControls.detach();
            console.log('TransformGizmoManager: Detached');
        }
    }

    public detach(): void {
        if (!this.transformControls) return;
        this.transformControls.detach();
    }

    public setMode(mode: 'translate' | 'rotate' | 'scale'): void {
        if (!this.transformControls) return;

        this.currentMode = mode;
        this.transformControls.setMode(mode);

        console.log('TransformGizmoManager: Mode set to', mode);
        this.emit('modeChanged', mode);
    }

    public getMode(): 'translate' | 'rotate' | 'scale' {
        return this.currentMode;
    }

    public setSpace(space: 'world' | 'local'): void {
        if (!this.transformControls) return;
        this.transformControls.setSpace(space);
        console.log('TransformGizmoManager: Space set to', space);
    }

    public setSize(size: number): void {
        if (!this.transformControls) return;
        this.transformControls.setSize(size);
    }

    public toggleMode(): void {
        switch (this.currentMode) {
            case 'translate':
                this.setMode('rotate');
                break;
            case 'rotate':
                this.setMode('scale');
                break;
            case 'scale':
                this.setMode('translate');
                break;
        }
    }

    public getAttachedObject(): THREE.Object3D | null {
        return this.transformControls?.object || null;
    }

    public isDragging(): boolean {
        return this.transformControls?.dragging || false;
    }

    /**
     * Set gizmo visibility
     * Useful for hiding gizmos in play mode
     */
    public setVisible(visible: boolean): void {
        if (!this.transformControls) return;

        if (visible) {
            // Re-enable gizmo
            this.transformControls.enabled = true;
        } else {
            // Disable and detach gizmo
            this.transformControls.enabled = false;
            this.transformControls.detach();
        }

        console.log('TransformGizmoManager: Visibility set to', visible);
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
                    console.error(`TransformGizmoManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Get current transform state of attached object
     */
    public getTransformState(): {
        position: THREE.Vector3;
        rotation: THREE.Euler;
        scale: THREE.Vector3;
    } | null {
        const object = this.getAttachedObject();
        if (!object) return null;

        return {
            position: object.position.clone(),
            rotation: object.rotation.clone(),
            scale: object.scale.clone()
        };
    }

    /**
     * Destroy and clean up
     */
    public destroy(): void {
        if (this.transformControls) {
            this.transformControls.detach();
            // Remove from scene before disposing
            if (this.editorManager?.scene) {
                // @ts-ignore - TransformControls is not a standard Object3D but can be removed from scene
                this.editorManager.scene.remove(this.transformControls);
            }
            this.transformControls.dispose();
            this.transformControls = null;
        }

        this.eventListeners.clear();
        this.editorManager = null;

        console.log('TransformGizmoManager: Destroyed');
    }
}

export default TransformGizmoManager;
