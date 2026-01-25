import * as THREE from 'three';

/**
 * SelectionManager - Multi-object selection system
 *
 * Manages object selection with support for:
 * - Single selection (click)
 * - Multi-selection (Ctrl+Click)
 * - Selection bounding box calculation
 * - Selection events
 * - Selection synchronization across panels
 *
 * CRITICAL: Do not modify the src/ directory - this is editor-only code
 */
class SelectionManager {
    // Singleton instance
    private static instance: SelectionManager;

    // Selected objects
    private selectedObjects: Set<THREE.Object3D> = new Set();

    // Primary selection (for single-selection workflows)
    private primarySelection: THREE.Object3D | null = null;

    // Selection bounding box
    private boundingBox: THREE.Box3 | null = null;
    private boundingBoxHelper: THREE.Box3Helper | null = null;

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    // Scene reference for adding/removing helpers
    private scene: THREE.Scene | null = null;

    /**
     * Private constructor for singleton pattern
     */
    private constructor() {
        console.log('SelectionManager: Initialized');
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): SelectionManager {
        if (!SelectionManager.instance) {
            SelectionManager.instance = new SelectionManager();
        }
        return SelectionManager.instance;
    }

    /**
     * Initialize the selection manager
     */
    public initialize(scene: THREE.Scene): void {
        this.scene = scene;
        console.log('SelectionManager: Scene reference set');
    }

    /**
     * Select an object (single selection, clears previous)
     */
    public select(object: THREE.Object3D | null | undefined): void {
        this.clearSelection();

        if (object) {
            this.selectedObjects.add(object);
            this.primarySelection = object;
        }

        this.updateBoundingBox();
        this.emit('selectionChanged', {
            objects: Array.from(this.selectedObjects),
            primary: this.primarySelection
        });

        console.log('SelectionManager: Single selection:', object?.name || 'null');
    }

    /**
     * Add object to selection (for multi-selection with Ctrl)
     */
    public addToSelection(object: THREE.Object3D): void {
        if (this.selectedObjects.has(object)) {
            // Remove if already selected (toggle behavior)
            this.selectedObjects.delete(object);

            // Update primary selection
            if (this.primarySelection === object) {
                const first = Array.from(this.selectedObjects)[0];
                this.primarySelection = first || null;
            }
        } else {
            // Add to selection
            this.selectedObjects.add(object);

            // Set as primary if first selection
            if (this.selectedObjects.size === 1) {
                this.primarySelection = object;
            }
        }

        this.updateBoundingBox();
        this.emit('selectionChanged', {
            objects: Array.from(this.selectedObjects),
            primary: this.primarySelection
        });

        console.log('SelectionManager: Multi-selection, count:', this.selectedObjects.size);
    }

    /**
     * Remove object from selection
     */
    public removeFromSelection(object: THREE.Object3D): void {
        this.selectedObjects.delete(object);

        if (this.primarySelection === object) {
            const first = Array.from(this.selectedObjects)[0];
            this.primarySelection = first || null;
        }

        this.updateBoundingBox();
        this.emit('selectionChanged', {
            objects: Array.from(this.selectedObjects),
            primary: this.primarySelection
        });
    }

    /**
     * Clear all selections
     */
    public clearSelection(): void {
        this.selectedObjects.clear();
        this.primarySelection = null;
        this.boundingBox = null;

        this.removeBoundingBoxHelper();

        this.emit('selectionChanged', {
            objects: [],
            primary: null
        });

        console.log('SelectionManager: Selection cleared');
    }

    /**
     * Check if object is selected
     */
    public isSelected(object: THREE.Object3D): boolean {
        return this.selectedObjects.has(object);
    }

    /**
     * Get all selected objects
     */
    public getSelectedObjects(): THREE.Object3D[] {
        return Array.from(this.selectedObjects);
    }

    /**
     * Get primary selection (for single-selection workflows)
     */
    public getPrimarySelection(): THREE.Object3D | null {
        return this.primarySelection;
    }

    /**
     * Get selection count
     */
    public getSelectionCount(): number {
        return this.selectedObjects.size;
    }

    /**
     * Update bounding box for selection
     */
    private updateBoundingBox(): void {
        this.removeBoundingBoxHelper();

        if (this.selectedObjects.size === 0) {
            this.boundingBox = null;
            return;
        }

        this.boundingBox = new THREE.Box3();

        // Calculate bounding box encompassing all selected objects
        for (const object of this.selectedObjects) {
            const box = new THREE.Box3().setFromObject(object);
            this.boundingBox.union(box);
        }

        // Create visual helper
        if (this.scene && this.boundingBox) {
            this.boundingBoxHelper = new THREE.Box3Helper(this.boundingBox, 0xffff00);
            this.scene.add(this.boundingBoxHelper);
        }
    }

    /**
     * Remove bounding box helper from scene
     */
    private removeBoundingBoxHelper(): void {
        if (this.boundingBoxHelper && this.scene) {
            this.scene.remove(this.boundingBoxHelper);
            this.boundingBoxHelper = null;
        }
    }

    /**
     * Get selection bounding box
     */
    public getBoundingBox(): THREE.Box3 | null {
        return this.boundingBox;
    }

    /**
     * Get center point of selection
     */
    public getSelectionCenter(): THREE.Vector3 | null {
        if (!this.boundingBox) return null;

        const center = new THREE.Vector3();
        this.boundingBox.getCenter(center);
        return center;
    }

    /**
     * Select all objects in scene
     */
    public selectAll(scene: THREE.Scene): void {
        this.clearSelection();

        scene.traverse((object) => {
            // Filter out helpers
            if (!object.name.includes('Grid') &&
                !object.name.includes('Axes') &&
                object instanceof THREE.Mesh) {
                this.selectedObjects.add(object);
            }
        });

        // Set primary selection
        if (this.selectedObjects.size > 0) {
            const first = Array.from(this.selectedObjects)[0];
            this.primarySelection = first || null;
        } else {
            this.primarySelection = null;
        }

        this.updateBoundingBox();
        this.emit('selectionChanged', {
            objects: Array.from(this.selectedObjects),
            primary: this.primarySelection
        });

        console.log('SelectionManager: Selected all objects, count:', this.selectedObjects.size);
    }

    /**
     * Invert selection in scene
     */
    public invertSelection(scene: THREE.Scene): void {
        const newSelection = new Set<THREE.Object3D>();

        scene.traverse((object) => {
            // Filter out helpers
            if (!object.name.includes('Grid') &&
                !object.name.includes('Axes') &&
                object instanceof THREE.Mesh) {
                if (!this.selectedObjects.has(object)) {
                    newSelection.add(object);
                }
            }
        });

        this.selectedObjects = newSelection;

        // Set primary selection
        const first = Array.from(this.selectedObjects)[0];
        this.primarySelection = first || null;

        this.updateBoundingBox();
        this.emit('selectionChanged', {
            objects: Array.from(this.selectedObjects),
            primary: this.primarySelection
        });

        console.log('SelectionManager: Selection inverted, count:', this.selectedObjects.size);
    }

    /**
     * Delete all selected objects
     */
    public deleteSelected(scene: THREE.Scene): void {
        const objectsToDelete = Array.from(this.selectedObjects);

        for (const object of objectsToDelete) {
            scene.remove(object);

            // Clean up geometry and material
            if (object instanceof THREE.Mesh) {
                object.geometry.dispose();
                if (Array.isArray(object.material)) {
                    object.material.forEach(mat => mat.dispose());
                } else {
                    object.material.dispose();
                }
            }
        }

        this.clearSelection();

        this.emit('objectsDeleted', { objects: objectsToDelete });

        console.log('SelectionManager: Deleted objects, count:', objectsToDelete.length);
    }

    /**
     * Duplicate selected objects
     */
    public duplicateSelected(scene: THREE.Scene): THREE.Object3D[] {
        const duplicates: THREE.Object3D[] = [];

        for (const object of this.selectedObjects) {
            if (object instanceof THREE.Mesh) {
                const clone = object.clone();
                clone.name = `${object.name}_copy`;
                clone.position.x += 1; // Offset slightly
                scene.add(clone);
                duplicates.push(clone);
            }
        }

        // Select the duplicates
        this.clearSelection();
        for (const duplicate of duplicates) {
            this.selectedObjects.add(duplicate);
        }
        this.primarySelection = duplicates[0] || null;

        this.updateBoundingBox();
        this.emit('objectsDuplicated', { originals: Array.from(this.selectedObjects), duplicates });
        this.emit('selectionChanged', {
            objects: Array.from(this.selectedObjects),
            primary: this.primarySelection
        });

        console.log('SelectionManager: Duplicated objects, count:', duplicates.length);

        return duplicates;
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
                    console.error(`SelectionManager: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Destroy and clean up
     */
    public destroy(): void {
        this.removeBoundingBoxHelper();
        this.clearSelection();
        this.eventListeners.clear();
        this.scene = null;
        console.log('SelectionManager: Destroyed');
    }
}

export default SelectionManager;
